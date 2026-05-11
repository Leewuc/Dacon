#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import train_cv_convnext as convnext_mod  # noqa: E402


class LegacyDualStreamModel(nn.Module):
    def __init__(self, backbone_name: str):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        feat_dim = int(self.backbone.num_features)
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, 1),
        )

    def forward(self, front: torch.Tensor, top: torch.Tensor) -> torch.Tensor:
        f = self.backbone(front)
        t = self.backbone(top)
        return self.head(torch.cat([f, t], dim=1)).squeeze(-1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OOF-based meta ensemble for physics submissions.")
    p.add_argument("--data-root", type=Path, default=Path("/data/AskFake/Image/physics"))
    p.add_argument("--convnext-run", type=Path, required=True)
    p.add_argument("--geometry-run", type=Path, required=True)
    p.add_argument("--convnext-test-sub", type=Path, required=True)
    p.add_argument("--geometry-test-sub", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def pooled_labels(data_root: Path) -> pd.DataFrame:
    train_df = pd.read_csv(data_root / "train.csv", encoding="utf-8-sig")
    dev_df = pd.read_csv(data_root / "dev.csv", encoding="utf-8-sig")
    train_df["target"] = (train_df["label"] == "unstable").astype(int)
    dev_df["target"] = (dev_df["label"] == "unstable").astype(int)
    return pd.concat([train_df[["id", "target"]], dev_df[["id", "target"]]], ignore_index=True)


def rebuild_convnext_oof(run_dir: Path, data_root: Path, device: torch.device) -> pd.DataFrame:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    proxy_manifest = Path(config["proxy_manifest"]) if config.get("proxy_manifest") else None
    pooled_df, _ = convnext_mod.load_tables(data_root, proxy_manifest)
    if bool(config.get("use_geometry", False)):
        pooled_df = convnext_mod.enrich_geometry(pooled_df, use_top_normalize=bool(config.get("use_top_normalize", False)))
    if bool(config.get("grouped_cv", False)):
        pooled_df = convnext_mod.assign_geometry_clusters(
            pooled_df,
            n_clusters=int(config.get("n_clusters", 16)),
            random_state=int(config.get("seed", 42)),
        )

    if bool(config.get("grouped_cv", False)):
        splitter = StratifiedGroupKFold(
            n_splits=int(config.get("n_folds", 5)),
            shuffle=True,
            random_state=int(config.get("seed", 42)),
        )
        split_iter = splitter.split(pooled_df, pooled_df["target"], groups=pooled_df["geometry_cluster"])
    else:
        splitter = StratifiedKFold(
            n_splits=int(config.get("n_folds", 5)),
            shuffle=True,
            random_state=int(config.get("seed", 42)),
        )
        split_iter = splitter.split(pooled_df, pooled_df["target"])

    oof_rows = []
    for fold, (_tr_idx, va_idx) in enumerate(split_iter):
        fold_dir = run_dir / f"fold{fold}"
        if not fold_dir.exists():
            continue
        val_df = pooled_df.iloc[va_idx].reset_index(drop=True)
        ds = convnext_mod.StructureDataset(
            val_df,
            image_size=int(config["image_size"]),
            is_train=False,
            is_test=True,
            use_top_normalize=bool(config.get("use_top_normalize", False)),
            eval_background_gray=bool(config.get("eval_background_gray", False)),
        )
        loader = DataLoader(
            ds,
            batch_size=16,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        state = torch.load(fold_dir / "best.pt", map_location="cpu", weights_only=False)
        temp_info = json.loads((fold_dir / "temperature.json").read_text(encoding="utf-8"))
        if any(k.startswith("top_gate.") for k in state.keys()):
            model = convnext_mod.DualStreamModel(
                config["model_name"],
                pretrained=False,
                use_geometry=bool(config.get("use_geometry", False)),
                separate_encoders=bool(config.get("separate_encoders", False)),
                top_feature_scale=float(config.get("top_feature_scale", 1.35)),
                top_geometry_scale=float(config.get("top_geometry_scale", 1.5)),
                use_foreground_branch=bool(config.get("use_foreground_branch", False)),
            ).to(device)
            model.load_state_dict(state)
            ids, preds = convnext_mod.predict_fold(model, loader, device, tta="none", temperature=float(temp_info["temperature"]))
        else:
            model = LegacyDualStreamModel(config["model_name"]).to(device)
            model.load_state_dict(state)
            model.eval()
            ids = []
            pred_chunks = []
            with torch.no_grad():
                for batch in loader:
                    front = batch["front"].to(device, non_blocking=True)
                    top = batch["top"].to(device, non_blocking=True)
                    with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                        logits = model(front, top)
                    logits = logits / max(float(temp_info["temperature"]), 1e-6)
                    probs = torch.sigmoid(logits.float()).cpu().numpy()
                    ids.extend(batch["id"])
                    pred_chunks.append(probs)
            preds = np.concatenate(pred_chunks)
        oof_rows.extend([{"id": sid, "convnext_oof": float(p)} for sid, p in zip(ids, preds)])
    return pd.DataFrame(oof_rows)


def rebuild_geometry_oof(run_dir: Path) -> pd.DataFrame:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    pooled = pd.read_csv(run_dir / "pooled_features.csv", encoding="utf-8-sig")
    feat_cols = [c for c in pooled.columns if c not in {"id", "target", "label", "front_path", "top_path", "source_split", "geometry_cluster"}]
    X = pooled[feat_cols].to_numpy(dtype=np.float32)
    y = pooled["target"].to_numpy(dtype=np.int64)
    groups = pooled["geometry_cluster"].to_numpy(dtype=np.int64)
    splitter = StratifiedGroupKFold(
        n_splits=int(config.get("n_folds", 5)),
        shuffle=True,
        random_state=int(config.get("seed", 42)),
    )
    oof_rows = []
    for fold, (_tr_idx, va_idx) in enumerate(splitter.split(X, y, groups=groups)):
        fold_dir = run_dir / f"fold{fold}"
        if not fold_dir.exists():
            continue
        with (fold_dir / "artifacts.pkl").open("rb") as f:
            art = pickle.load(f)
        X_va = pooled.iloc[va_idx][art["features"]].to_numpy(dtype=np.float32)
        X_scaled = art["scaler"].transform(X_va)
        p_lr = art["logreg"].predict_proba(X_scaled)[:, 1]
        p_mlp = art["mlp"].predict_proba(X_scaled)[:, 1]
        p_hgb = art["hgb"].predict_proba(X_va)[:, 1]
        p = 0.20 * p_lr + 0.25 * p_mlp + 0.55 * p_hgb
        p = convnext_mod.np.clip(p, 1e-7, 1 - 1e-7)
        logit = np.log(p / (1.0 - p))
        temp = float(art["temperature"])
        p = 1.0 / (1.0 + np.exp(-np.clip(logit / max(temp, 1e-6), -60.0, 60.0)))
        ids = pooled.iloc[va_idx]["id"].tolist()
        oof_rows.extend([{"id": sid, "geometry_oof": float(prob)} for sid, prob in zip(ids, p)])
    return pd.DataFrame(oof_rows)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    labels = pooled_labels(args.data_root)
    conv_oof = rebuild_convnext_oof(args.convnext_run, args.data_root, device)
    geo_oof = rebuild_geometry_oof(args.geometry_run)
    train_df = labels.merge(conv_oof, on="id", how="inner").merge(geo_oof, on="id", how="inner")
    if len(train_df) != len(labels):
        print(f"warning: OOF rows {len(train_df)} / labels {len(labels)}")

    x1 = np.clip(train_df["convnext_oof"].to_numpy(dtype=np.float64), 1e-7, 1 - 1e-7)
    x2 = np.clip(train_df["geometry_oof"].to_numpy(dtype=np.float64), 1e-7, 1 - 1e-7)
    X = np.stack(
        [
            x1,
            x2,
            np.log(x1 / (1.0 - x1)),
            np.log(x2 / (1.0 - x2)),
            x1 - x2,
            x1 * x2,
        ],
        axis=1,
    )
    y = train_df["target"].to_numpy(dtype=np.int64)

    meta = LogisticRegression(C=1.0, max_iter=4000)
    meta.fit(X, y)
    oof_pred = meta.predict_proba(X)[:, 1]
    print(json.dumps({"stack_oof_logloss": float(log_loss(y, oof_pred)), "stack_oof_auc": float(roc_auc_score(y, oof_pred))}))

    conv_sub = pd.read_csv(args.convnext_test_sub, encoding="utf-8-sig")
    geo_sub = pd.read_csv(args.geometry_test_sub, encoding="utf-8-sig")
    if not np.array_equal(conv_sub["id"].values, geo_sub["id"].values):
        raise RuntimeError("submission id order mismatch")
    t1 = np.clip(conv_sub["unstable_prob"].to_numpy(dtype=np.float64), 1e-7, 1 - 1e-7)
    t2 = np.clip(geo_sub["unstable_prob"].to_numpy(dtype=np.float64), 1e-7, 1 - 1e-7)
    XT = np.stack(
        [
            t1,
            t2,
            np.log(t1 / (1.0 - t1)),
            np.log(t2 / (1.0 - t2)),
            t1 - t2,
            t1 * t2,
        ],
        axis=1,
    )
    pred = np.clip(meta.predict_proba(XT)[:, 1], 1e-7, 1 - 1e-7)
    out = conv_sub[["id"]].copy()
    out["unstable_prob"] = pred
    out["stable_prob"] = 1.0 - pred
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"rows={len(out)} -> {args.output}")


if __name__ == "__main__":
    main()
