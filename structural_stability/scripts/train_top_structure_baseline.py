#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from train_block_graph_baseline import (
    apply_top_roi_crop,
    build_tables,
    component_summary,
    extract_top_components,
    global_mask_stats,
    maybe_top_normalizer,
    probability_temperature,
    safe_auc,
    safe_log_loss,
    fit_temperature,
)
from train_support_graph_baseline import merge_top_components


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Top-only structural baseline.")
    sub = p.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train")
    train.add_argument("--data-root", type=Path, default=Path("/data/AskFake/Image/physics"))
    train.add_argument("--out-dir", type=Path, required=True)
    train.add_argument("--n-folds", type=int, default=5)
    train.add_argument("--n-clusters", type=int, default=16)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--use-top-normalize", action="store_true")
    train.add_argument("--top-roi-margin-frac", type=float, default=0.15)
    train.add_argument("--top-color-bin", type=int, default=24)
    train.add_argument("--top-min-comp-frac", type=float, default=0.0012)

    predict = sub.add_parser("predict")
    predict.add_argument("--data-root", type=Path, default=Path("/data/AskFake/Image/physics"))
    predict.add_argument("--run-dir", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def largest_block_features(components: list[dict[str, float]], k: int = 3) -> dict[str, float]:
    feats: dict[str, float] = {}
    comps = sorted(components, key=lambda x: x["area_frac"], reverse=True)
    for i in range(k):
        if i < len(comps):
            c = comps[i]
            feats[f"top_l{i+1}_area"] = float(c["area_frac"])
            feats[f"top_l{i+1}_cx"] = float(c["cx"])
            feats[f"top_l{i+1}_cy"] = float(c["cy"])
            feats[f"top_l{i+1}_w"] = float(c["bbox_w_frac"])
            feats[f"top_l{i+1}_h"] = float(c["bbox_h_frac"])
            feats[f"top_l{i+1}_extent"] = float(abs(c["cx"]) + 0.5 * c["bbox_w_frac"])
        else:
            feats[f"top_l{i+1}_area"] = 0.0
            feats[f"top_l{i+1}_cx"] = 0.0
            feats[f"top_l{i+1}_cy"] = 0.0
            feats[f"top_l{i+1}_w"] = 0.0
            feats[f"top_l{i+1}_h"] = 0.0
            feats[f"top_l{i+1}_extent"] = 0.0
    return feats


def top_structure_features(components: list[dict[str, float]]) -> dict[str, float]:
    feats: dict[str, float] = {}
    n = len(components)
    if n == 0:
        return {
            "top_mass_eccentricity": 0.0,
            "top_span_proxy": 0.0,
            "top_left_extent": 0.0,
            "top_right_extent": 0.0,
            "top_overhang_asymmetry": 0.0,
            "top_largest_pair_gap": 0.0,
            "top_largest_pair_overlap": 0.0,
            "top_mass_on_main": 0.0,
            "top_num_large_blocks": 0.0,
            "top_explicit_stability": 0.0,
        }
    comps = sorted(components, key=lambda x: x["area_frac"], reverse=True)
    areas = np.asarray([c["area_frac"] for c in comps], dtype=np.float32)
    cxs = np.asarray([c["cx"] for c in comps], dtype=np.float32)
    widths = np.asarray([c["bbox_w_frac"] for c in comps], dtype=np.float32)
    left_edges = cxs - 0.5 * widths
    right_edges = cxs + 0.5 * widths
    mass_center = float(np.average(cxs, weights=np.maximum(areas, 1e-6)))
    span = float(right_edges.max() - left_edges.min())
    left_extent = float(max(0.0, -left_edges.min()))
    right_extent = float(max(0.0, right_edges.max()))
    asym = float(abs(right_extent - left_extent))
    large_mask = areas >= max(0.5 * float(areas.max()), 0.01)
    mass_on_main = float(areas[large_mask].sum() / max(areas.sum(), 1e-6))
    num_large = float(large_mask.sum())

    if len(comps) >= 2:
        a, b = comps[0], comps[1]
        gap = abs(a["cx"] - b["cx"]) - 0.5 * (a["bbox_w_frac"] + b["bbox_w_frac"])
        overlap = max(0.0, 0.5 * (a["bbox_w_frac"] + b["bbox_w_frac"]) - abs(a["cx"] - b["cx"]))
    else:
        gap = 0.0
        overlap = 0.0

    stability = (
        1.10 * (1.0 - abs(mass_center))
        + 0.80 * mass_on_main
        + 0.45 * overlap
        - 0.90 * asym
        - 0.60 * max(gap, 0.0)
    )
    feats["top_mass_eccentricity"] = abs(mass_center)
    feats["top_span_proxy"] = span
    feats["top_left_extent"] = left_extent
    feats["top_right_extent"] = right_extent
    feats["top_overhang_asymmetry"] = asym
    feats["top_largest_pair_gap"] = float(gap)
    feats["top_largest_pair_overlap"] = float(overlap)
    feats["top_mass_on_main"] = mass_on_main
    feats["top_num_large_blocks"] = num_large
    feats["top_explicit_stability"] = float(stability)
    return feats


def extract_feature_row(top_img: Image.Image, color_bin: int, min_comp_frac: float, roi_margin_frac: float) -> dict[str, float]:
    top_roi = apply_top_roi_crop(top_img, roi_margin_frac)
    top_rgb = np.asarray(top_roi.convert("RGB"), dtype=np.uint8)
    top_mask, top_components = extract_top_components(top_rgb, color_bin=color_bin, min_comp_frac=min_comp_frac)
    top_components = merge_top_components(top_components)
    feats: dict[str, float] = {}
    feats.update({f"top_global_{k}": v for k, v in global_mask_stats(top_mask).items()})
    feats.update(component_summary(top_components, "top_blocks"))
    feats.update(largest_block_features(top_components, k=3))
    feats.update(top_structure_features(top_components))
    feats["top_balance_proxy"] = (
        1.0
        - feats["top_mass_eccentricity"]
        - 0.8 * feats["top_overhang_asymmetry"]
        - 0.5 * max(feats["top_largest_pair_gap"], 0.0)
    )
    return feats


def build_feature_table(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    top_norm = maybe_top_normalizer(args.use_top_normalize)
    rows = []
    total = len(df)
    for idx, row in enumerate(df.itertuples(index=False), start=1):
        top_img = Image.open(row.top_path).convert("RGB")
        if top_norm is not None:
            top_img = top_norm.normalize(row.top_path, top_img)
        feats = extract_feature_row(
            top_img=top_img,
            color_bin=args.top_color_bin,
            min_comp_frac=args.top_min_comp_frac,
            roi_margin_frac=args.top_roi_margin_frac,
        )
        feats["id"] = row.id
        rows.append(feats)
        if idx == 1 or idx % 100 == 0 or idx == total:
            print(json.dumps({"stage": "feature_table", "done": idx, "total": total, "last_id": str(row.id)}), flush=True)
    return pd.DataFrame(rows)


def add_clusters(df: pd.DataFrame, path_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    top_norm = maybe_top_normalizer(args.use_top_normalize)
    vecs = []
    total = len(path_df)
    for idx, row in enumerate(path_df.itertuples(index=False), start=1):
        top_img = Image.open(row.top_path).convert("RGB")
        if top_norm is not None:
            top_img = top_norm.normalize(row.top_path, top_img)
        top_roi = apply_top_roi_crop(top_img, args.top_roi_margin_frac)
        top_small = top_roi.convert("L").resize((24, 24))
        vecs.append(np.asarray(top_small, dtype=np.float32).reshape(-1) / 255.0)
        if idx == 1 or idx % 100 == 0 or idx == total:
            print(json.dumps({"stage": "cluster_vectors", "done": idx, "total": total, "last_id": str(row.id)}), flush=True)
    X = np.stack(vecs, axis=0)
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=args.n_clusters, random_state=args.seed, n_init=20)
    out = df.copy()
    out["geometry_cluster"] = km.fit_predict(Xs)
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {"id", "target", "front_path", "top_path", "geometry_cluster", "label"}
    return [c for c in df.columns if c not in exclude]


def sanitize_feature_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    out[cols] = out[cols].replace([np.inf, -np.inf], np.nan)
    out[cols] = out[cols].fillna(0.0)
    return out


def train_cv(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pooled_paths, test_paths = build_tables(args.data_root)

    pooled_feat = build_feature_table(pooled_paths, args)
    pooled = pooled_paths[["id", "target", "front_path", "top_path"]].merge(pooled_feat, on="id", how="left")
    pooled = add_clusters(pooled, pooled_paths[["id", "top_path"]], args)

    test_feat = build_feature_table(test_paths, args)
    test_df = test_paths[["id", "front_path", "top_path"]].merge(test_feat, on="id", how="left")

    pooled.to_csv(args.out_dir / "pooled_features.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(args.out_dir / "test_features.csv", index=False, encoding="utf-8-sig")

    feats = feature_columns(pooled)
    pooled = sanitize_feature_frame(pooled, feats)
    test_df = sanitize_feature_frame(test_df, feats)
    X = pooled[feats].to_numpy(dtype=np.float32)
    y = pooled["target"].to_numpy(dtype=np.int64)
    groups = pooled["geometry_cluster"].to_numpy(dtype=np.int64)

    splitter = StratifiedGroupKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    oof = np.zeros(len(pooled), dtype=np.float64)
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(X, y, groups=groups)):
        fold_dir = args.out_dir / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        scaler = StandardScaler().fit(X_tr)
        X_tr_scaled = scaler.transform(X_tr)
        X_va_scaled = scaler.transform(X_va)

        logreg = LogisticRegression(C=1.5, max_iter=4000, random_state=args.seed + fold)
        rf = RandomForestClassifier(
            n_estimators=600,
            max_depth=8,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=args.seed + fold,
            n_jobs=-1,
        )
        hgb = HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_depth=4,
            max_iter=600,
            l2_regularization=5e-3,
            min_samples_leaf=16,
            random_state=args.seed + fold,
        )

        logreg.fit(X_tr_scaled, y_tr)
        rf.fit(X_tr, y_tr)
        hgb.fit(X_tr, y_tr)

        p_lr = logreg.predict_proba(X_va_scaled)[:, 1]
        p_rf = rf.predict_proba(X_va)[:, 1]
        p_hgb = hgb.predict_proba(X_va)[:, 1]
        p_blend = 0.18 * p_lr + 0.32 * p_rf + 0.50 * p_hgb
        temp = fit_temperature(p_blend, y_va)
        p_cal = probability_temperature(p_blend, temp)
        oof[va_idx] = p_cal

        ll = safe_log_loss(y_va, p_cal)
        auc = safe_auc(y_va, p_cal)
        fold_scores.append({"fold": fold, "val_logloss": float(ll), "val_auc": float(auc), "temperature": float(temp)})
        print(json.dumps(fold_scores[-1]))

        with (fold_dir / "artifacts.pkl").open("wb") as f:
            pickle.dump(
                {
                    "features": feats,
                    "scaler": scaler,
                    "logreg": logreg,
                    "rf": rf,
                    "hgb": hgb,
                    "temperature": temp,
                },
                f,
            )

    overall = {"oof_logloss": safe_log_loss(y, oof), "oof_auc": safe_auc(y, oof)}
    print(json.dumps(overall))
    with (args.out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2, default=str)
    with (args.out_dir / "scores.json").open("w", encoding="utf-8") as f:
        json.dump({"folds": fold_scores, "overall": overall}, f, ensure_ascii=False, indent=2)


def predict(args: argparse.Namespace) -> None:
    _config = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    test_df = pd.read_csv(args.run_dir / "test_features.csv", encoding="utf-8-sig")
    fold_dirs = sorted([p for p in args.run_dir.iterdir() if p.is_dir() and p.name.startswith("fold")])
    preds = []
    for fold_dir in fold_dirs:
        with (fold_dir / "artifacts.pkl").open("rb") as f:
            art = pickle.load(f)
        test_df = sanitize_feature_frame(test_df, art["features"])
        X = test_df[art["features"]].to_numpy(dtype=np.float32)
        X_scaled = art["scaler"].transform(X)
        p_lr = art["logreg"].predict_proba(X_scaled)[:, 1]
        p_rf = art["rf"].predict_proba(X)[:, 1]
        p_hgb = art["hgb"].predict_proba(X)[:, 1]
        p_blend = 0.18 * p_lr + 0.32 * p_rf + 0.50 * p_hgb
        preds.append(probability_temperature(p_blend, float(art["temperature"])))

    final = np.mean(np.stack(preds, axis=0), axis=0)
    final = np.clip(final, 1e-7, 1 - 1e-7)
    out = test_df[["id"]].copy()
    out["unstable_prob"] = final
    out["stable_prob"] = 1.0 - final
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"rows={len(out)} -> {args.output}")


def main() -> None:
    args = parse_args()
    if args.command == "train":
        train_cv(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
