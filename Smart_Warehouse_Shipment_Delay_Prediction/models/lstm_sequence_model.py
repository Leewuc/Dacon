"""
LSTM Sequence Model for Warehouse Delay Prediction.

핵심 아이디어:
  현재 트리 모델: 각 time step을 독립 샘플로 취급
  LSTM: 시나리오 내 24 time step을 시퀀스로 처리

  "step 5에서 혼잡도가 올라가기 시작하면 step 18에서 지연 폭발"
  → LSTM이 이런 패턴을 자연스럽게 학습 가능

  트리 모델과 앙상블하면 다양성 확보 + 일반화 개선

아키텍처:
  - 입력: [batch, seq_len, n_features]  (seq_len ≤ 24)
  - BiLSTM 2층  (양방향으로 과거/미래 컨텍스트)
  - Attention 레이어 (중요한 time step에 집중)
  - 출력: [batch, seq_len]  (각 time step의 delay 예측)

훈련:
  - 시나리오 단위로 배치
  - MAE loss
  - AdamW optimizer + CosineAnnealingLR
  - Gradient clipping (안정성)

앙상블:
  - LSTM 예측 + LightGBM 예측을 blend
  - CV에서 최적 blend 비율 탐색
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from compare_models import (
    TARGET,
    build_split_masks_v2,
    default_feature_cache_dir,
    load_bundle,
    load_feature_cache,
    to_builtin,
    utc_now_string,
    write_yaml_summary,
)


# ── 핵심 피처: LSTM에 넣을 피처만 선택 ──────────────────────
# 너무 많은 피처는 LSTM에서 오히려 노이즈 → 중요 피처만 선택
LSTM_FEATURE_COLS = [
    # 시계열 핵심 동적 피처
    "congestion_score", "pack_utilization", "loading_dock_util",
    "max_zone_density", "battery_mean", "low_battery_ratio",
    "charge_queue_length", "robot_utilization", "order_inflow_15m",
    "staging_area_util", "outbound_truck_wait_min",
    "wms_response_time_ms", "label_print_queue",
    # 모멘텀/트렌드 피처 (이미 피처 캐시에 있음)
    "congestion_score_momentum3", "pack_utilization_momentum3",
    "congestion_score_trend3", "pack_utilization_trend3",
    "low_battery_ratio_momentum3", "order_inflow_15m_momentum3",
    # 시간 위치
    "time_idx", "time_frac",
    # 레이아웃 정적 피처 (시나리오 내 일정)
    "robot_total", "pack_station_count", "charger_count",
    "floor_area_sqm", "layout_compactness",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Dataset ──────────────────────────────────────────────────
class ScenarioDataset(Dataset):
    """시나리오 단위 데이터셋. 각 샘플 = 1개 시나리오의 전체 시퀀스."""

    def __init__(
        self,
        features: pd.DataFrame,
        feature_cols: list[str],
        has_target: bool = True,
        max_seq_len: int = 25,
    ):
        self.has_target = has_target
        self.max_seq_len = max_seq_len
        self.scenarios: list[tuple[np.ndarray, np.ndarray | None]] = []

        # Vectorized grouping — much faster than iterating GroupBy for 8k+ scenarios
        sid_arr = features["scenario_id"].to_numpy()
        feat_arr = features[feature_cols].to_numpy(dtype=np.float32)
        y_arr = features[TARGET].to_numpy(dtype=np.float32) if has_target else None
        tidx_arr = features["time_idx"].to_numpy() if "time_idx" in features.columns else None

        # sort by (scenario_id, time_idx) so groups are contiguous
        if tidx_arr is not None:
            order = np.lexsort((tidx_arr, sid_arr))
        else:
            order = np.argsort(sid_arr, kind="stable")
        sid_sorted = sid_arr[order]
        feat_sorted = feat_arr[order]
        y_sorted = y_arr[order] if y_arr is not None else None

        # find group boundaries
        change = np.concatenate([[True], sid_sorted[1:] != sid_sorted[:-1], [True]])
        boundaries = np.where(change)[0]
        for i in range(len(boundaries) - 1):
            s, e = boundaries[i], boundaries[i + 1]
            X = feat_sorted[s:e]
            y = y_sorted[s:e] if y_sorted is not None else None
            self.scenarios.append((X, y))

    def __len__(self) -> int:
        return len(self.scenarios)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        X, y = self.scenarios[idx]
        seq_len = min(X.shape[0], self.max_seq_len)  # truncate if longer than max
        X = X[:seq_len]
        if y is not None:
            y = y[:seq_len]
        mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
        mask[:seq_len] = True
        # 패딩 (max_seq_len보다 짧은 시나리오 처리)
        if seq_len < self.max_seq_len:
            pad = np.zeros((self.max_seq_len - seq_len, X.shape[1]), dtype=np.float32)
            X = np.vstack([X, pad])
        X_t = torch.from_numpy(X)
        y_t = torch.from_numpy(y) if y is not None else None
        if y_t is not None and seq_len < self.max_seq_len:
            pad_y = torch.zeros(self.max_seq_len - seq_len)
            y_t = torch.cat([y_t, pad_y])
        return X_t, y_t, mask


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    Xs, ys, masks = zip(*batch)
    X_batch = torch.stack(Xs)
    mask_batch = torch.stack(masks)
    y_batch = torch.stack(ys) if ys[0] is not None else None
    return X_batch, y_batch, mask_batch


# ── Model ────────────────────────────────────────────────────
class WarehouseLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        lstm_out_size = hidden_size * 2  # bidirectional
        # Attention
        self.attn = nn.Linear(lstm_out_size, 1)
        # Output head
        self.head = nn.Sequential(
            nn.Linear(lstm_out_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        x    : [B, T, F]
        mask : [B, T]  True = 실제 데이터
        return: [B, T]  각 time step 예측값
        """
        x = self.input_norm(x)
        out, _ = self.lstm(x)  # [B, T, 2H]

        # Attention weight (masked)
        attn_logit = self.attn(out).squeeze(-1)  # [B, T]
        attn_logit = attn_logit.masked_fill(~mask, float("-inf"))
        attn_w = torch.softmax(attn_logit, dim=-1).unsqueeze(-1)  # [B, T, 1]

        # Context-weighted representation for each step
        context = (out * attn_w).sum(dim=1, keepdim=True)  # [B, 1, 2H]
        context = context.expand_as(out)  # [B, T, 2H]
        combined = out + context  # residual

        pred = self.head(combined).squeeze(-1)  # [B, T]
        pred = torch.relu(pred)  # 지연 >= 0
        return pred


# ── Training ─────────────────────────────────────────────────
def train_lstm(
    train_ds: ScenarioDataset,
    val_ds: ScenarioDataset,
    n_features: int,
    cfg: dict[str, Any],
) -> tuple[WarehouseLSTM, float]:
    model = WarehouseLSTM(
        input_size=n_features,
        hidden_size=cfg.get("hidden_size", 128),
        num_layers=cfg.get("num_layers", 2),
        dropout=cfg.get("dropout", 0.3),
    ).to(DEVICE)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.get("batch_size", 64),
        shuffle=True, collate_fn=collate_fn, num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.get("lr", 1e-3),
        weight_decay=cfg.get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.get("epochs", 30)
    )

    best_val_mae = float("inf")
    best_state = None

    for epoch in range(cfg.get("epochs", 30)):
        # Train
        model.train()
        for X_b, y_b, mask_b in train_loader:
            X_b = X_b.to(DEVICE)
            y_b = y_b.to(DEVICE)
            mask_b = mask_b.to(DEVICE)
            optimizer.zero_grad()
            pred = model(X_b, mask_b)
            # MAE loss on valid positions only
            loss = (pred[mask_b] - y_b[mask_b]).abs().mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Validate
        if (epoch + 1) % 5 == 0 or epoch == cfg.get("epochs", 30) - 1:
            val_mae = evaluate_lstm(model, val_ds)
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  epoch {epoch+1:3d}  val_mae={val_mae:.5f}  best={best_val_mae:.5f}",
                  flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_mae


def evaluate_lstm(model: WarehouseLSTM, ds: ScenarioDataset) -> float:
    model.eval()
    loader = DataLoader(ds, batch_size=128, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)
    all_pred: list[np.ndarray] = []
    all_true: list[np.ndarray] = []
    with torch.no_grad():
        for X_b, y_b, mask_b in loader:
            X_b = X_b.to(DEVICE)
            mask_b = mask_b.to(DEVICE)
            pred = model(X_b, mask_b)
            for i in range(len(X_b)):
                m = mask_b[i].cpu().numpy()
                all_pred.append(pred[i][mask_b[i]].cpu().numpy())
                if y_b is not None:
                    all_true.append(y_b[i][m].numpy())
    y_pred = np.concatenate(all_pred)
    y_true = np.concatenate(all_true)
    return float(mean_absolute_error(y_true, y_pred))


def predict_lstm(model: WarehouseLSTM, ds: ScenarioDataset) -> np.ndarray:
    """Returns flat predictions aligned with original row order."""
    model.eval()
    loader = DataLoader(ds, batch_size=128, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)
    all_pred: list[np.ndarray] = []
    with torch.no_grad():
        for X_b, _, mask_b in loader:
            X_b = X_b.to(DEVICE)
            mask_b = mask_b.to(DEVICE)
            pred = model(X_b, mask_b)
            for i in range(len(X_b)):
                all_pred.append(pred[i][mask_b[i]].cpu().numpy())
    return np.concatenate(all_pred)


# ── Main ─────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=base_dir)
    parser.add_argument("--experiment-dir", type=Path, default=base_dir / "experiments")
    parser.add_argument("--output-dir", type=Path, default=base_dir / "submissions")
    parser.add_argument("--cache-name", type=str, default="features_v9_full")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--mode", choices=["cv", "predict", "cv_predict"], default="cv_predict")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--blend-weights", type=float, nargs="+",
                        default=[0.1, 0.2, 0.3, 0.4, 0.5],
                        help="LSTM blend weights to search (1-w = lgbm weight)")
    parser.add_argument(
        "--lgbm-submission",
        type=Path,
        default=base_dir / "submissions"
        / "conditional_ensemble_20260408T054845Z_triple_specialist_v3_late16_tail85"
        / "submission_triple_specialist_v3_late16_tail85.csv",
        help="Best existing lgbm submission to blend with lstm",
    )
    parser.add_argument("--notes", type=str,
        default="BiLSTM with attention on scenario sequences + blend with lgbm")
    return parser.parse_args()


LOG_COLUMNS = [
    "run_id", "timestamp_utc", "variant", "split_name",
    "mae", "rmse", "n_train", "n_valid", "avg_mae", "notes",
]


def append_log(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=LOG_COLUMNS)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in LOG_COLUMNS})


def main() -> None:
    args = parse_args()
    run_id = utc_now_string()
    print(f"Device: {DEVICE}", flush=True)

    # Load cache
    cache_dir = args.cache_dir or default_feature_cache_dir(args.data_dir, args.cache_name)
    cache = load_feature_cache(cache_dir)

    # 사용할 피처 필터링 (캐시에 있는 것만)
    feat_cols = [c for c in LSTM_FEATURE_COLS if c in cache.train_features.columns]
    print(f"LSTM features: {len(feat_cols)}", flush=True)

    # 스케일링 (LSTM은 정규화 필수)
    # Only copy the columns needed — avoid slow full 527-col DataFrame copy
    scaler = StandardScaler()
    train_arr = scaler.fit_transform(cache.train_features[feat_cols].fillna(0))
    test_arr = scaler.transform(cache.test_features[feat_cols].fillna(0))

    train_slim_cols = list(dict.fromkeys(["scenario_id", "time_idx", TARGET] + feat_cols))
    train_slim_cols = [c for c in train_slim_cols if c in cache.train_features.columns]
    test_slim_cols = list(dict.fromkeys(["scenario_id", "time_idx"] + feat_cols))
    test_slim_cols = [c for c in test_slim_cols if c in cache.test_features.columns]
    train_scaled = cache.train_features[train_slim_cols].copy()
    test_scaled = cache.test_features[test_slim_cols].copy()
    train_scaled[feat_cols] = train_arr
    test_scaled[feat_cols] = test_arr
    print(f"Data ready: train={len(train_scaled):,} test={len(test_scaled):,}", flush=True)

    cfg = {
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
    }
    log_rows: list[dict[str, Any]] = []

    if args.mode in ("cv", "cv_predict"):
        split_masks = build_split_masks_v2(cache.train_features)
        split_results: dict[str, dict[str, float]] = {}

        for split_name, valid_mask in split_masks.items():
            train_mask = ~valid_mask
            print(f"\n{'='*50}", flush=True)
            print(f"[{split_name}] train={train_mask.sum()} valid={valid_mask.sum()}", flush=True)

            tr_df = train_scaled[train_mask].reset_index(drop=True)
            val_df = train_scaled[valid_mask].reset_index(drop=True)
            val_df_orig = cache.train_features[valid_mask].reset_index(drop=True)

            train_ds = ScenarioDataset(tr_df, feat_cols, has_target=True)
            val_ds = ScenarioDataset(val_df, feat_cols, has_target=True)

            print(f"  Training LSTM ...", flush=True)
            model, best_val_mae = train_lstm(train_ds, val_ds, len(feat_cols), cfg)
            print(f"  LSTM best val MAE = {best_val_mae:.5f}", flush=True)

            split_results[split_name] = {"lstm_mae": best_val_mae}
            log_rows.append({
                "run_id": run_id, "timestamp_utc": run_id,
                "variant": "lstm_only", "split_name": split_name,
                "mae": round(best_val_mae, 6), "rmse": "",
                "n_train": int(train_mask.sum()), "n_valid": int(valid_mask.sum()),
                "avg_mae": "", "notes": args.notes,
            })

        # avg_mae
        all_maes = [split_results[s]["lstm_mae"] for s in split_masks if s in split_results]
        avg_mae = float(np.mean(all_maes))
        print(f"\nLSTM avg_mae = {avg_mae:.5f}", flush=True)
        for r in log_rows:
            r["avg_mae"] = round(avg_mae, 6)

        append_log(args.experiment_dir / "lstm_log.csv", log_rows)
        write_yaml_summary(
            args.experiment_dir / "runs" / f"{run_id}_lstm_cv.yaml",
            {"run_id": run_id, "avg_mae": avg_mae, "cfg": cfg},
        )

    if args.mode in ("predict", "cv_predict"):
        bundle = load_bundle(args.data_dir, None, None)
        print("\nTraining LSTM on full train data ...", flush=True)

        full_ds = ScenarioDataset(train_scaled, feat_cols, has_target=True)
        # validation set = 마지막 20% 시나리오 (early stopping용)
        scen_ids = list(cache.train_features["scenario_id"].unique())
        n_val = max(1, int(len(scen_ids) * 0.1))
        val_ids = set(scen_ids[-n_val:])
        val_mask_full = cache.train_features["scenario_id"].isin(val_ids)
        tr_df_full = train_scaled[~val_mask_full].reset_index(drop=True)
        val_df_full = train_scaled[val_mask_full].reset_index(drop=True)
        full_train_ds = ScenarioDataset(tr_df_full, feat_cols, has_target=True)
        full_val_ds = ScenarioDataset(val_df_full, feat_cols, has_target=True)

        model_full, _ = train_lstm(full_train_ds, full_val_ds, len(feat_cols), cfg)

        test_ds = ScenarioDataset(test_scaled, feat_cols, has_target=False)
        lstm_preds = predict_lstm(model_full, test_ds)

        out_dir = args.output_dir / f"lstm_{run_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # LSTM 단독 submission
        sub_lstm = bundle.sample_submission.copy()
        sub_lstm[TARGET] = lstm_preds
        sub_lstm.to_csv(out_dir / "submission_lstm_only.csv", index=False)

        # LightGBM 최고 submission과 blend
        if args.lgbm_submission.exists():
            lgbm_sub = pd.read_csv(args.lgbm_submission)
            lgbm_preds = lgbm_sub[TARGET].to_numpy()
            print("\nBlending LSTM + LightGBM ...", flush=True)
            for w_lstm in args.blend_weights:
                blend = w_lstm * lstm_preds + (1 - w_lstm) * lgbm_preds
                sub_blend = bundle.sample_submission.copy()
                sub_blend[TARGET] = blend
                fname = f"submission_blend_lstm{int(w_lstm*100):02d}_lgbm{int((1-w_lstm)*100):02d}.csv"
                sub_blend.to_csv(out_dir / fname, index=False)
                print(f"  blend w_lstm={w_lstm:.1f} → {fname}", flush=True)

        summary = {
            "run_id": run_id, "lstm_cfg": cfg,
            "output_dir": str(out_dir.resolve()),
            "blend_weights_tested": args.blend_weights,
        }
        write_yaml_summary(args.experiment_dir / "runs" / f"{run_id}_lstm.yaml", summary)
        write_yaml_summary(args.experiment_dir / "current_lstm.yaml", summary)
        print(f"\nDone. Output → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
