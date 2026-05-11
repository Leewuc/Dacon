"""
Normalized lead features: lead / scenario_mean (scale-invariant).

Problem with raw leads: congestion_lead1 ratio = 1.27 (test 27% higher),
causing distribution shift and extrapolation issues.

Fix: normalize by scenario mean so both train and test measure
"how does next step compare to THIS scenario's baseline?"
  lead1_rel = lead1 / scenario_mean  → ratio ≈ 1.0 for train and test
  lead_diff1 = lead1 - current       → forward change (complement to existing diff1)
  future_mean_rel = future_mean / scenario_mean
  future_max_rel  = future_max  / scenario_mean
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, log_evaluation, reset_parameter

from compare_models import (
    TARGET,
    build_lgbm_lr_schedule,
    default_feature_cache_dir,
    load_bundle,
    load_feature_cache,
    to_builtin,
    utc_now_string,
    write_yaml_summary,
)
from conditional_ensemble_experiments import VARIANTS, apply_variant
from ensemble_best_models import fit_catboost, load_params as load_base_params
from catboost_categorical_experiments import make_pool, prepare_catboost_frame
from layout_specialist_experiments import load_base_params as load_layout_base_params
from oof_threeway_blend_search import fit_full_layout_predictions, fit_full_xgb_predictions
from temporal_dynamics_ensemble import compute_temporal_dynamics

PSEUDO_SUBMISSION = Path(__file__).resolve().parent / (
    "submissions/temporal_pseudo_20260424T030628Z"
    "/submission_temporal_pseudo_bw0.20_cb1.0.csv"
)
BEST_VARIANT = "triple_specialist_v5_tail_xgb"
EPS = 1e-6

LEAD_FEATURES = [
    "congestion_score",
    "charge_queue_length",
    "order_inflow_15m",
    "avg_charge_wait",
    "robot_utilization",
]


def compute_norm_lead_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    For each key feature, compute scale-invariant lead features:
      lead1/2/3_rel  = lead_k / scenario_mean   (relative future level)
      lead_diff1/2   = lead_k - current          (forward difference)
      future_mean_rel = future_mean / scenario_mean
      future_max_rel  = future_max  / scenario_mean

    All have train/test ratios ≈ 1.0 since both are normalized by own scenario mean.
    """
    df = df.copy()
    new_cols: list[str] = []

    for feat in LEAD_FEATURES:
        if feat not in df.columns:
            continue

        grp      = df.groupby("scenario_id")[feat]
        scen_mean = grp.transform("mean")  # scenario-level mean for normalization

        # ── Relative lead values (lead / scenario_mean) ───────────────────────
        for k in [1, 2, 3]:
            raw_lead = grp.shift(-k)
            col_rel  = f"{feat}_lead{k}_rel"
            df[col_rel] = (raw_lead / (scen_mean.abs() + EPS)).astype(np.float32)
            new_cols.append(col_rel)

        # ── Relative suffix mean and max ─────────────────────────────────────
        def _suffix_mean(s: pd.Series) -> pd.Series:
            v = s.to_numpy(dtype=float)
            v_fill  = np.where(np.isnan(v), 0.0, v)
            valid   = (~np.isnan(v)).astype(int)
            tot_sum, tot_cnt = v_fill.sum(), valid.sum()
            sfx_sum = tot_sum - np.cumsum(v_fill)
            sfx_cnt = tot_cnt - np.cumsum(valid)
            result  = np.where(sfx_cnt > 0, sfx_sum / sfx_cnt, v)
            return pd.Series(result, index=s.index, dtype=np.float32)

        def _suffix_max(s: pd.Series) -> pd.Series:
            v = s.to_numpy(dtype=float)
            n = len(v)
            result = v.copy()
            run_max = -np.inf
            for i in range(n - 2, -1, -1):
                nxt = v[i + 1]
                if not np.isnan(nxt):
                    run_max = max(run_max, nxt)
                if run_max > -np.inf:
                    result[i] = run_max
            return pd.Series(result, index=s.index, dtype=np.float32)

        suffix_mean_vals = grp.transform(_suffix_mean)
        suffix_max_vals  = grp.transform(_suffix_max)

        col_fm = f"{feat}_future_mean_rel"
        df[col_fm] = (suffix_mean_vals / (scen_mean.abs() + EPS)).astype(np.float32)
        new_cols.append(col_fm)

        col_fmax = f"{feat}_future_max_rel"
        df[col_fmax] = (suffix_max_vals / (scen_mean.abs() + EPS)).astype(np.float32)
        new_cols.append(col_fmax)

    return df, new_cols


def compute_test_weights(
    test_df: pd.DataFrame,
    train_cong_mean: float,
    base_weight: float = 0.20,
    congestion_boost: float = 1.0,
) -> np.ndarray:
    cong = test_df["congestion_score"].to_numpy()
    rel  = np.maximum(0.0, (cong - train_cong_mean) / train_cong_mean)
    w    = base_weight * (1.0 + congestion_boost * rel)
    return np.where(np.isnan(w), base_weight, w).astype(np.float32)


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
    p.add_argument("--base-weight",    type=float, default=0.20)
    p.add_argument("--congestion-boost", type=float, default=1.0)
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

    train_df_raw = cache.train_features.copy()
    test_df_raw  = cache.test_features.copy()

    print("Computing normalized lead features...", flush=True)
    train_df, lead_cols = compute_norm_lead_features(train_df_raw)
    test_df,  _         = compute_norm_lead_features(test_df_raw)

    print(f"  {len(lead_cols)} lead features. Train/test alignment:", flush=True)
    for c in lead_cols:
        tr = float(train_df[c].mean())
        te = float(test_df[c].mean())
        ratio = te / (abs(tr) + EPS)
        print(f"  {c:<38s} tr={tr:7.4f}  te={te:7.4f}  ratio={ratio:.3f}", flush=True)

    print("\nComputing temporal dynamics features...", flush=True)
    train_df, dyn_cols = compute_temporal_dynamics(train_df)
    test_df,  _        = compute_temporal_dynamics(test_df)

    base_cols    = cache.feature_columns
    feature_cols = base_cols + lead_cols + dyn_cols
    print(f"\nTotal: {len(base_cols)} base + {len(lead_cols)} norm_lead + {len(dyn_cols)} dyn = {len(feature_cols)}", flush=True)

    lgbm_params, cat_params, cat_target_mode, cat_feature_columns, cat_categorical_columns = \
        load_base_params(args.lgbm_tuning_yaml, args.base_ensemble_catboost_yaml)
    layout_base_params = load_layout_base_params(args.layout_base_tuning_yaml)
    base_lgbm_weight   = 0.86

    print(f"\nLoading pseudo-labels from {PSEUDO_SUBMISSION.name}...", flush=True)
    pseudo_map = pd.read_csv(PSEUDO_SUBMISSION).set_index("ID")[TARGET]
    test_df[TARGET] = test_df["ID"].map(pseudo_map).to_numpy()

    train_cong_mean = float(train_df["congestion_score"].mean())
    test_weights    = compute_test_weights(test_df, train_cong_mean, args.base_weight, args.congestion_boost)
    train_weights   = np.ones(len(train_df), dtype=np.float32)
    print(f"  Test weights: mean={test_weights.mean():.3f}  max={test_weights.max():.3f}", flush=True)

    combined_df      = pd.concat([train_df, test_df], ignore_index=True)
    combined_weights = np.concatenate([train_weights, test_weights])
    combined_y       = combined_df[TARGET].to_numpy()
    print(f"  Combined: {len(train_df):,} + {len(test_df):,} = {len(combined_df):,}", flush=True)

    print("\nTraining LGBM...", flush=True)
    lgbm_model     = fit_lgbm_weighted(combined_df, combined_y, combined_weights, feature_cols, lgbm_params)
    lgbm_test_pred = np.clip(lgbm_model.predict(test_df[feature_cols]), 0, None)
    print(f"  LGBM pred: mean={lgbm_test_pred.mean():.3f}  std={lgbm_test_pred.std():.3f}", flush=True)

    print("\nTraining CatBoost on train only...", flush=True)
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
    cat_test_pred  = np.clip(np.asarray(cat_test_raw), 0, None)
    base_test_pred = np.clip(
        base_lgbm_weight * lgbm_test_pred + (1.0 - base_lgbm_weight) * cat_test_pred, 0, None
    )

    print("\nTraining layout/tail-XGB on train only...", flush=True)
    y_train = train_df[TARGET].to_numpy()
    layout_test_pred = fit_full_layout_predictions(
        train_df, test_df, bundle.layout, feature_cols, y_train,
        layout_base_params, "layout_cluster_blend_w030", args.seed,
    )
    xgb_test_pred = fit_full_xgb_predictions(
        train_df, test_df, feature_cols, y_train, "log1p_depth8_v1",
    )

    pred = apply_variant(test_df, base_test_pred, layout_test_pred, xgb_test_pred, VARIANTS[BEST_VARIANT])
    sub  = bundle.sample_submission.copy()
    sub[TARGET] = pred

    out_dir  = args.output_dir / f"lead_norm_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    sub_path = out_dir / f"submission_lead_norm_pseudo.csv"
    sub.to_csv(sub_path, index=False)
    print(f"\nSaved → {sub_path}", flush=True)

    write_yaml_summary(
        args.experiment_dir / "runs" / f"{run_id}_lead_norm.yaml",
        {
            "run_id": run_id,
            "lead_cols": lead_cols,
            "dyn_cols": dyn_cols,
            "total_features": len(feature_cols),
            "best_variant": BEST_VARIANT,
            "submission": str(sub_path),
        },
    )
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
