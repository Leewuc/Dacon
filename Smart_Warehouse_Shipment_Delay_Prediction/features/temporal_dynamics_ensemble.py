"""
Temporal dynamics features for conditional ensemble.

Key insight: ALL 25 time steps per scenario are visible at both train and test time.
This lets us compute WITHIN-SCENARIO temporal patterns:
  - slope (relative): how fast is congestion changing? → scale-invariant
  - coefficient of variation (CV): how volatile is this scenario? → scale-invariant
  - peak timing: when does the worst condition occur? → scale-invariant
  - early-vs-late ratio: is it getting better or worse? → scale-invariant

These differ from scen_agg (absolute means, shift-prone) and scen_rel (ratios, partial fix)
because they capture SHAPE not LEVEL — train/test scenarios with similar dynamics
should score similarly regardless of absolute congestion level.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

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
from conditional_ensemble_experiments import (
    VARIANTS,
    SPLIT_WEIGHTS,
    apply_variant,
    build_base_oof_predictions_cached,
    fit_full_base_predictions_cached,
    rmse,
    weighted_split_score,
)
from ensemble_best_models import load_params as load_base_params
from layout_specialist_experiments import load_base_params as load_layout_base_params
from oof_threeway_blend_search import (
    build_layout_oof_predictions,
    build_xgb_oof_predictions,
    fit_full_layout_predictions,
    fit_full_xgb_predictions,
)

# Key features to compute temporal dynamics for
DYNAMIC_FEATURES = [
    "congestion_score",
    "charge_queue_length",
    "order_inflow_15m",
    "avg_charge_wait",
    "robot_utilization",
]
EPS = 1e-6


def compute_temporal_dynamics(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    For each scenario, compute temporal dynamics across 25 time steps.
    Returns (df_with_new_cols, new_col_names).

    All features are scale-invariant by design:
      - rel_slope   = slope / |mean|  (relative rate of change)
      - cv          = std  / |mean|   (relative volatility)
      - peak_time   = argmax / 24     (normalized timing of peak, 0-1)
      - early_late  = mean(t<12) / mean(t>=12) - 1  (trend direction)
    """
    df = df.copy()
    new_cols: list[str] = []

    for feat in DYNAMIC_FEATURES:
        if feat not in df.columns:
            continue

        grp = df.groupby("scenario_id")[feat]
        feat_mean = grp.transform("mean")
        feat_std  = grp.transform("std")

        # 1. Coefficient of variation (volatility, scale-invariant)
        col_cv = f"{feat}_cv"
        df[col_cv] = (feat_std / (feat_mean.abs() + EPS)).astype(np.float32)
        new_cols.append(col_cv)

        # 2. Relative slope: (slope / |mean|) captures direction + speed of change
        # slope via least-squares on time_idx
        def _rel_slope(g: pd.DataFrame) -> pd.Series:
            t = g["time_idx"].to_numpy().astype(float)
            v = g[feat].to_numpy().astype(float)
            valid = ~np.isnan(v)
            if valid.sum() < 2:
                return pd.Series(0.0, index=g.index)
            t_v, v_v = t[valid], v[valid]
            t_c = t_v - t_v.mean()
            denom = (t_c ** 2).sum()
            slope = (t_c * v_v).sum() / denom if denom > 0 else 0.0
            m = np.nanmean(v)
            rel = slope / (abs(m) + EPS)
            return pd.Series(rel, index=g.index, dtype=np.float32)

        col_slope = f"{feat}_rel_slope"
        df[col_slope] = df.groupby("scenario_id", group_keys=False).apply(
            lambda g: _rel_slope(g)
        ).astype(np.float32)
        new_cols.append(col_slope)

        # 3. Normalized peak timing (when does max occur? 0=early, 1=late)
        def _peak_time(g: pd.DataFrame) -> pd.Series:
            v = g[feat].to_numpy()
            valid = ~np.isnan(v)
            if not valid.any():
                return pd.Series(0.5, index=g.index)
            peak_idx = int(np.nanargmax(v))
            peak_t   = float(g["time_idx"].iloc[peak_idx]) / 24.0
            return pd.Series(peak_t, index=g.index, dtype=np.float32)

        col_peak = f"{feat}_peak_time"
        df[col_peak] = df.groupby("scenario_id", group_keys=False).apply(
            lambda g: _peak_time(g)
        ).astype(np.float32)
        new_cols.append(col_peak)

        # (early_late removed — numerically unstable when late≈0)

    return df, new_cols


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",       type=Path, default=base_dir)
    p.add_argument("--experiment-dir", type=Path, default=base_dir / "experiments")
    p.add_argument("--output-dir",     type=Path, default=base_dir / "submissions")
    p.add_argument("--cache-name",     type=str,  default="features_v9_full")
    p.add_argument("--cache-dir",      type=Path, default=None)
    p.add_argument("--seed",           type=int,  default=42)
    p.add_argument("--mode", choices=["cv", "predict", "cv_predict"], default="cv_predict")
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

    base_cols = cache.feature_columns

    print("Computing temporal dynamics features...", flush=True)
    train_df_raw = cache.train_features.copy()
    test_df_raw  = cache.test_features.copy()

    train_df, new_cols = compute_temporal_dynamics(train_df_raw)
    test_df,  _        = compute_temporal_dynamics(test_df_raw)
    feature_cols = base_cols + new_cols

    print(f"Features: base={len(base_cols)}  +dynamics={len(new_cols)}  total={len(feature_cols)}", flush=True)
    print(f"\n{'Feature':<35} {'train_mean':>10}  {'test_mean':>9}  {'ratio':>6}", flush=True)
    for c in new_cols:
        tr = float(train_df[c].mean())
        te = float(test_df[c].mean())
        ratio = te / (abs(tr) + EPS)
        print(f"  {c:<33} {tr:10.4f}  {te:9.4f}  {ratio:6.3f}", flush=True)

    split_masks = build_split_masks_v2(train_df, seed=args.seed)
    y = train_df[TARGET].to_numpy()

    lgbm_params, cat_params, cat_target_mode, cat_feature_columns, cat_categorical_columns = \
        load_base_params(args.lgbm_tuning_yaml, args.base_ensemble_catboost_yaml)
    layout_base_params = load_layout_base_params(args.layout_base_tuning_yaml)
    base_lgbm_weight   = 0.86

    if args.mode in ("cv", "cv_predict"):
        print("\nBuilding OOF predictions...", flush=True)
        y_true_by_split, base_pred_by_split = build_base_oof_predictions_cached(
            train_df, bundle, split_masks, feature_cols,
            lgbm_params, cat_params, cat_target_mode,
            cat_feature_columns, cat_categorical_columns, base_lgbm_weight,
        )
        layout_pred_by_split = build_layout_oof_predictions(
            train_df, bundle, split_masks, feature_cols,
            layout_base_params, "layout_cluster_blend_w030", args.seed,
        )
        xgb_pred_by_split = build_xgb_oof_predictions(
            train_df, split_masks, feature_cols, "log1p_depth8_v1",
        )

        best_stress  = float("inf")
        best_variant = None
        variants_summary: dict[str, Any] = {}

        for variant_name, cfg in VARIANTS.items():
            split_metrics: dict[str, dict[str, float]] = {}
            for split_name, valid_mask in split_masks.items():
                valid_df = train_df.loc[valid_mask.to_numpy()]
                pred = apply_variant(
                    valid_df,
                    base_pred_by_split[split_name],
                    layout_pred_by_split[split_name],
                    xgb_pred_by_split[split_name],
                    cfg,
                )
                y_v = y_true_by_split[split_name]
                split_metrics[split_name] = {
                    "mae": float(mean_absolute_error(y_v, pred)),
                    "rmse": rmse(y_v, pred),
                }
            stress = weighted_split_score(split_metrics)
            variants_summary[variant_name] = {"stress_score": stress, "split_metrics": split_metrics}
            if stress < best_stress:
                best_stress  = stress
                best_variant = variant_name
            for sn, m in split_metrics.items():
                print(f"[dyn_ce] {variant_name:<40s} split={sn} mae={m['mae']:.6f} stress={stress:.6f}", flush=True)

        print(f"\n=== CV Summary ===", flush=True)
        for vn, r in sorted(variants_summary.items(), key=lambda x: x[1]["stress_score"]):
            tag = " ← BEST" if vn == best_variant else ""
            print(f"  {vn:<40s} stress={r['stress_score']:.5f}{tag}", flush=True)
        print(f"\n  Best stress={best_stress:.5f}  (baseline ~9.702)", flush=True)
        for sn, m in variants_summary[best_variant]["split_metrics"].items():
            print(f"    {sn:30s} {m['mae']:.5f}", flush=True)

    else:
        best_variant = "triple_specialist_v5_tail_xgb"
        best_stress  = float("inf")
        variants_summary = {}

    if args.mode in ("predict", "cv_predict"):
        print(f"\nStep 3: Full train → predict (variant={best_variant}) ===", flush=True)
        base_test_pred = fit_full_base_predictions_cached(
            train_df, test_df, bundle, feature_cols,
            lgbm_params, cat_params, cat_target_mode,
            cat_feature_columns, cat_categorical_columns, base_lgbm_weight,
        )
        layout_test_pred = fit_full_layout_predictions(
            train_df, test_df, bundle.layout, feature_cols, y,
            layout_base_params, "layout_cluster_blend_w030", args.seed,
        )
        xgb_test_pred = fit_full_xgb_predictions(
            train_df, test_df, feature_cols, y, "log1p_depth8_v1",
        )

        pred = apply_variant(test_df, base_test_pred, layout_test_pred, xgb_test_pred, VARIANTS[best_variant])
        sub = bundle.sample_submission.copy()
        sub[TARGET] = pred

        out_dir  = args.output_dir / f"temporal_dyn_{run_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        sub_path = out_dir / f"submission_temporal_dyn_{best_variant}.csv"
        sub.to_csv(sub_path, index=False)
        print(f"Saved → {sub_path}", flush=True)

        write_yaml_summary(
            args.experiment_dir / "runs" / f"{run_id}_temporal_dyn.yaml",
            {
                "run_id": run_id,
                "best_variant": best_variant,
                "best_stress": to_builtin(best_stress),
                "dynamic_features": new_cols,
                "feature_count": len(feature_cols),
                "submission": str(sub_path),
            },
        )

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
