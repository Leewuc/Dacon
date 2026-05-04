"""
Pseudo-labeling v4: congestion-weighted, using best submission as pseudo-labels.

Motivation:
  - Train distribution: congestion mean=9.99, test mean=12.70 (+27%)
  - Model extrapolates poorly beyond training range
  - Solution: add test rows WITH pseudo-labels to training so model sees test-distribution inputs

Design choices vs prior pseudo-label attempts:
  1. Use BEST submission (LB=10.1434) as pseudo-labels (not iterative)
  2. Congestion-weighted: high-congestion test rows get higher weight
       weight_test = base_weight * (1 + congestion_boost * congestion_relative_to_train)
  3. Low base weight (0.3) to prevent pseudo-label noise from dominating
  4. No CatBoost on combined data (CatBoost uses structural features that may not transfer)
  5. Run predict only (no CV — combined dataset breaks standard CV assumptions)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMRegressor, early_stopping, log_evaluation, reset_parameter
from sklearn.metrics import mean_absolute_error

from compare_models import (
    TARGET,
    build_lgbm_lr_schedule,
    build_split_masks_v2,
    default_feature_cache_dir,
    load_bundle,
    load_feature_cache,
    to_builtin,
    utc_now_string,
    write_yaml_summary,
)
from conditional_ensemble_experiments import (
    VARIANTS,
    apply_variant,
    fit_full_base_predictions_cached,
)
from ensemble_best_models import fit_lgbm, fit_catboost, load_params as load_base_params
from catboost_categorical_experiments import make_pool, prepare_catboost_frame
from layout_specialist_experiments import load_base_params as load_layout_base_params
from oof_threeway_blend_search import (
    fit_full_layout_predictions,
    fit_full_xgb_predictions,
)

BEST_SUBMISSION = Path(__file__).resolve().parent / (
    "submissions/conditional_manual_20260408T054845Z_triple_specialist_v3_late16_tail85"
    "/submission_triple_specialist_v3_late16_tail85.csv"
)
BEST_VARIANT = "triple_specialist_v3_late16_tail85"


def load_lgbm_params(yaml_path: Path) -> dict[str, Any]:
    obj = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    params = dict(obj["best_lgbm_params"])
    params.pop("device_type", None)
    params["n_jobs"] = int(params.get("n_jobs", 8))
    return params


def compute_test_weights(
    test_df: pd.DataFrame,
    train_congestion_mean: float,
    base_weight: float = 0.30,
    congestion_boost: float = 1.0,
) -> np.ndarray:
    """
    Weight each test row by how "hard" (high-congestion) it is.
    High-congestion rows get higher weight → model focuses on learning this regime.
    weight = base_weight * (1 + boost * max(0, (cong - train_mean) / train_mean))
    """
    cong = test_df["congestion_score"].to_numpy()
    relative_excess = np.maximum(0.0, (cong - train_congestion_mean) / train_congestion_mean)
    weights = base_weight * (1.0 + congestion_boost * relative_excess)
    weights = np.where(np.isnan(weights), base_weight, weights)  # fill NaN with base
    return weights.astype(np.float32)


def fit_lgbm_weighted(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    feature_cols: list[str],
    params: dict[str, Any],
) -> LGBMRegressor:
    model = LGBMRegressor(**params)
    lr_schedule = build_lgbm_lr_schedule(
        n_estimators=int(params["n_estimators"]),
        base_lr=float(params["learning_rate"]),
    )
    model.fit(
        x_train[feature_cols], y_train,
        sample_weight=sample_weight,
        callbacks=[log_evaluation(-1), reset_parameter(learning_rate=lr_schedule)],
    )
    return model


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",       type=Path, default=base_dir)
    p.add_argument("--experiment-dir", type=Path, default=base_dir / "experiments")
    p.add_argument("--output-dir",     type=Path, default=base_dir / "submissions")
    p.add_argument("--cache-name",     type=str,  default="features_v9_full")
    p.add_argument("--cache-dir",      type=Path, default=None)
    p.add_argument("--seed",           type=int,  default=42)
    p.add_argument("--base-weight",    type=float, default=0.30,
                   help="Base sample weight for test pseudo-labels (0.1~0.5)")
    p.add_argument("--congestion-boost", type=float, default=1.0,
                   help="Extra weight multiplier for high-congestion test rows")
    p.add_argument(
        "--lgbm-tuning-yaml", type=Path,
        default=base_dir / "experiments" / "runs" / "20260402T061536Z_tuning.yaml",
    )
    p.add_argument(
        "--base-ensemble-catboost-yaml", type=Path,
        default=base_dir / "experiments" / "runs" / "20260402T234826Z_cv_catboost_categorical.yaml",
    )
    p.add_argument(
        "--layout-base-tuning-yaml", type=Path,
        default=base_dir / "experiments" / "runs" / "20260402T061536Z_tuning.yaml",
    )
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    run_id = utc_now_string()

    cache_dir = args.cache_dir or default_feature_cache_dir(args.data_dir, args.cache_name)
    bundle    = load_bundle(args.data_dir, max_train_rows=None, max_test_rows=None)
    cache     = load_feature_cache(cache_dir)

    train_df  = cache.train_features.copy()
    test_df   = cache.test_features.copy()
    base_cols = cache.feature_columns

    lgbm_params, cat_params, cat_target_mode, cat_feature_columns, cat_categorical_columns = \
        load_base_params(args.lgbm_tuning_yaml, args.base_ensemble_catboost_yaml)
    layout_base_params = load_layout_base_params(args.layout_base_tuning_yaml)
    base_lgbm_weight   = 0.86

    # ── Step 1: Load pseudo-labels ────────────────────────────────────────────
    print(f"Step 1: Loading pseudo-labels from {BEST_SUBMISSION.name}...", flush=True)
    pseudo_sub  = pd.read_csv(BEST_SUBMISSION)
    pseudo_map  = pseudo_sub.set_index("ID")[TARGET]
    test_df[TARGET] = test_df["ID"].map(pseudo_map).to_numpy()

    train_cong_mean = float(train_df["congestion_score"].mean())
    test_cong_mean  = float(test_df["congestion_score"].mean())
    print(f"  Train congestion mean: {train_cong_mean:.3f}", flush=True)
    print(f"  Test  congestion mean: {test_cong_mean:.3f}", flush=True)

    # ── Step 2: Compute sample weights ───────────────────────────────────────
    print(f"\nStep 2: Computing sample weights (base={args.base_weight}, boost={args.congestion_boost})...", flush=True)
    test_weights  = compute_test_weights(test_df, train_cong_mean, args.base_weight, args.congestion_boost)
    train_weights = np.ones(len(train_df), dtype=np.float32)

    print(f"  Test weight stats: mean={test_weights.mean():.3f}  min={test_weights.min():.3f}  max={test_weights.max():.3f}", flush=True)
    print(f"  High-cong test rows (cong>12): {(test_df['congestion_score']>12).sum()} "
          f"avg_weight={(test_weights[test_df['congestion_score'].to_numpy()>12]).mean():.3f}", flush=True)

    # ── Step 3: Build combined train+test dataset ─────────────────────────────
    print(f"\nStep 3: Building combined dataset...", flush=True)
    combined_df      = pd.concat([train_df, test_df], ignore_index=True)
    combined_weights = np.concatenate([train_weights, test_weights])
    combined_y       = combined_df[TARGET].to_numpy()

    print(f"  Train rows: {len(train_df):,}  Test rows: {len(test_df):,}  "
          f"Total: {len(combined_df):,}", flush=True)

    # ── Step 4: Train LGBM on combined (with weights) ─────────────────────────
    print(f"\nStep 4: Training weighted LGBM on train+test...", flush=True)
    lgbm_model = fit_lgbm_weighted(
        combined_df, combined_y, combined_weights, base_cols, lgbm_params
    )
    lgbm_test_pred = np.clip(lgbm_model.predict(test_df[base_cols]), 0, None)
    print(f"  LGBM test pred: mean={lgbm_test_pred.mean():.3f}  std={lgbm_test_pred.std():.3f}", flush=True)

    # ── Step 5: CatBoost on train only (structural features, no pseudo-label) ─
    print(f"\nStep 5: Training CatBoost on train only (no pseudo-labels)...", flush=True)
    cat_train_df, _, _ = prepare_catboost_frame(bundle.train, bundle.layout)
    cat_test_df,  _, _ = prepare_catboost_frame(bundle.test,  bundle.layout)
    cat_model, _ = fit_catboost(
        cat_train_df, train_df[TARGET].to_numpy(), None, None,
        cat_feature_columns, cat_categorical_columns, cat_params, cat_target_mode,
    )
    cat_test_raw = cat_model.predict(
        make_pool(cat_test_df, None, cat_feature_columns, cat_categorical_columns)
    )
    if cat_target_mode == "log1p":
        cat_test_raw = np.expm1(cat_test_raw)
    cat_test_pred = np.clip(np.asarray(cat_test_raw), 0, None)
    base_test_pred = np.clip(
        base_lgbm_weight * lgbm_test_pred + (1.0 - base_lgbm_weight) * cat_test_pred, 0, None
    )

    # ── Step 6: Layout-XGB + Tail-XGB on combined (with weights) ──────────────
    print(f"\nStep 6: Training layout-XGB and tail-XGB on combined...", flush=True)
    y_train_only = train_df[TARGET].to_numpy()

    # These functions don't support sample_weight directly, use train-only for them
    layout_test_pred = fit_full_layout_predictions(
        train_df, test_df, bundle.layout,
        base_cols, y_train_only,
        layout_base_params, "layout_cluster_blend_w030", args.seed,
    )
    xgb_test_pred = fit_full_xgb_predictions(
        train_df, test_df, base_cols, y_train_only, "log1p_depth8_v1",
    )

    # ── Step 7: Apply best variant and save ───────────────────────────────────
    print(f"\nStep 7: Applying variant={BEST_VARIANT}...", flush=True)
    best_cfg = VARIANTS[BEST_VARIANT]
    pred     = apply_variant(test_df, base_test_pred, layout_test_pred, xgb_test_pred, best_cfg)

    sub = bundle.sample_submission.copy()
    sub[TARGET] = pred

    tag     = f"bw{args.base_weight:.2f}_cb{args.congestion_boost:.1f}"
    out_dir = args.output_dir / f"pseudo_v4_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    sub_path = out_dir / f"submission_pseudo_v4_{tag}.csv"
    sub.to_csv(sub_path, index=False)
    print(f"Saved → {sub_path}", flush=True)

    write_yaml_summary(
        args.experiment_dir / "runs" / f"{run_id}_pseudo_v4.yaml",
        {
            "run_id": run_id,
            "base_weight": args.base_weight,
            "congestion_boost": args.congestion_boost,
            "pseudo_label_source": str(BEST_SUBMISSION),
            "best_variant": BEST_VARIANT,
            "submission": str(sub_path),
        },
    )
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
