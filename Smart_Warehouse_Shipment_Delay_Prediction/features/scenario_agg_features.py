"""
Scenario-level aggregate features.

Key insight: test scenarios are completely new (0 overlap with train), but
the test set contains ALL 25 time steps for each scenario. So we can compute
aggregates of INPUT features across all time steps of a scenario — these are
valid for both train and test without target leakage.

Example: scen_congestion_max = max congestion across all 25 steps of a scenario.
Even if we don't know the target, this tells us "this scenario had peak congestion
X, which predicts overall delay level."

Features added:
  - scen_{feat}_{agg}  : mean/max/min/std across all 25 steps (scenario-level)
  - scen_{feat}_t0     : value at time_idx=0 (initial state)
  - scen_{feat}_slope  : linear trend over time (positive = worsening)
  - rel_{feat}         : current value / scenario_mean  (within-scenario position)
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

SPLIT_WEIGHTS = {
    "scenario_holdout": 0.20,
    "layout_holdout": 0.20,
    "late_time_holdout": 0.15,
    "unseen_layout_heavy": 0.15,
    "congestion_tail_holdout": 0.30,
}

# Top correlated features to aggregate at scenario level
AGG_FEATURES = [
    "low_battery_ratio",
    "order_inflow_15m",
    "congestion_score",
    "charge_queue_length",
    "avg_charge_wait",
    "blocked_path_15m",
    "robot_utilization",
    "fault_count_15m",
    "battery_mean",
]


def load_lgbm_params(yaml_path: Path) -> dict[str, Any]:
    obj = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    params = dict(obj["best_lgbm_params"])
    params.pop("device_type", None)
    params["n_jobs"] = int(params.get("n_jobs", 8))
    return params


def add_scenario_agg_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Add scenario-level aggregate features. Uses all rows of each scenario
    (including 'future' time steps) — valid since we aggregate input features only.
    """
    new_cols: list[str] = []
    df = df.copy()

    for feat in AGG_FEATURES:
        if feat not in df.columns:
            continue

        grp = df.groupby("scenario_id")[feat]

        # Global aggs (all 25 steps)
        df[f"scen_{feat}_mean"] = grp.transform("mean").astype(np.float32)
        df[f"scen_{feat}_max"]  = grp.transform("max").astype(np.float32)
        df[f"scen_{feat}_min"]  = grp.transform("min").astype(np.float32)
        df[f"scen_{feat}_std"]  = grp.transform("std").fillna(0).astype(np.float32)
        new_cols += [f"scen_{feat}_mean", f"scen_{feat}_max",
                     f"scen_{feat}_min",  f"scen_{feat}_std"]

        # Initial state: value at time_idx=0
        t0_map = df.loc[df["time_idx"] == 0].set_index("scenario_id")[feat]
        df[f"scen_{feat}_t0"] = df["scenario_id"].map(t0_map).astype(np.float32)
        new_cols.append(f"scen_{feat}_t0")

        # Relative position: current / scenario_mean (deviation from typical)
        mean_col = f"scen_{feat}_mean"
        df[f"rel_{feat}"] = (df[feat] / (df[mean_col] + 1e-6)).astype(np.float32)
        new_cols.append(f"rel_{feat}")

    # Slope features: linear trend of top features over time_idx within scenario
    slope_feats = ["congestion_score", "charge_queue_length", "low_battery_ratio",
                   "order_inflow_15m", "battery_mean"]
    for feat in slope_feats:
        if feat not in df.columns:
            continue
        # slope = cov(time_idx, feat) / var(time_idx) within scenario
        def _slope(g: pd.Series) -> pd.Series:
            tidx = df.loc[g.index, "time_idx"].to_numpy().astype(float)
            vals = g.to_numpy().astype(float)
            if tidx.std() < 1e-9:
                return pd.Series(0.0, index=g.index)
            slope_val = np.polyfit(tidx, vals, 1)[0]
            return pd.Series(slope_val, index=g.index)

        slope_series = df.groupby("scenario_id")[feat].transform(
            lambda g: pd.Series(
                np.polyfit(
                    df.loc[g.index, "time_idx"].to_numpy().astype(float),
                    g.to_numpy().astype(float),
                    1
                )[0],
                index=g.index
            )
        ).astype(np.float32)
        df[f"scen_{feat}_slope"] = slope_series
        new_cols.append(f"scen_{feat}_slope")

    return df, new_cols


def fit_lgbm(
    x_tr: pd.DataFrame,
    y_tr: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    feature_cols: list[str],
    params: dict[str, Any],
) -> LGBMRegressor:
    model = LGBMRegressor(**params)
    lr_schedule = build_lgbm_lr_schedule(
        n_estimators=int(params["n_estimators"]),
        base_lr=float(params["learning_rate"]),
    )
    model.fit(
        x_tr[feature_cols], y_tr,
        eval_set=[(x_val[feature_cols], y_val)],
        callbacks=[
            early_stopping(80, verbose=False),
            log_evaluation(-1),
            reset_parameter(learning_rate=lr_schedule),
        ],
    )
    return model


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=base_dir)
    parser.add_argument("--cache-name", type=str, default="features_v9_full")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--experiment-dir", type=Path, default=base_dir / "experiments")
    parser.add_argument("--output-dir", type=Path, default=base_dir / "submissions")
    parser.add_argument(
        "--lgbm-tuning-yaml", type=Path,
        default=base_dir / "experiments" / "runs" / "20260402T061536Z_tuning.yaml",
    )
    parser.add_argument("--mode", choices=["cv", "predict", "cv_predict"], default="cv_predict")
    args = parser.parse_args()

    run_id = utc_now_string()
    cache_dir = args.cache_dir or default_feature_cache_dir(args.data_dir, args.cache_name)
    cache = load_feature_cache(cache_dir)
    bundle = load_bundle(args.data_dir, None, None)
    params = load_lgbm_params(args.lgbm_tuning_yaml)

    print("Adding scenario aggregate features...", flush=True)
    train_df, new_cols = add_scenario_agg_features(cache.train_features)
    test_df, _         = add_scenario_agg_features(cache.test_features)

    feature_cols = cache.feature_columns + new_cols
    print(f"Features: base={len(cache.feature_columns)}  +scen_agg={len(new_cols)}  total={len(feature_cols)}", flush=True)

    split_metrics: dict[str, float] = {}

    # ── CV ────────────────────────────────────────────────────────────────────
    if args.mode in ("cv", "cv_predict"):
        print("\n=== CV ===", flush=True)
        split_masks = build_split_masks_v2(train_df, seed=42)

        for split_name, valid_mask in split_masks.items():
            df_tr = train_df.loc[~valid_mask]
            df_val = train_df.loc[valid_mask]
            print(f"\n[{split_name}] train={len(df_tr):,}  valid={len(df_val):,}", flush=True)

            model = fit_lgbm(
                df_tr, df_tr[TARGET].to_numpy(),
                df_val, df_val[TARGET].to_numpy(),
                feature_cols, params,
            )
            pred = np.clip(model.predict(df_val[feature_cols]), 0, None)
            mae = float(mean_absolute_error(df_val[TARGET].to_numpy(), pred))
            print(f"  MAE = {mae:.5f}", flush=True)
            split_metrics[split_name] = mae

        avg_mae = float(np.mean(list(split_metrics.values())))
        stress = float(sum(SPLIT_WEIGHTS[k] * split_metrics[k] for k in SPLIT_WEIGHTS))
        print(f"\n=== CV Summary ===", flush=True)
        for k, v in split_metrics.items():
            print(f"  {k:30s} {v:.5f}", flush=True)
        print(f"  avg_mae  = {avg_mae:.6f}", flush=True)
        print(f"  stress   = {stress:.6f}  (baseline triple_spec ~9.702)", flush=True)

    # ── Predict ───────────────────────────────────────────────────────────────
    if args.mode in ("predict", "cv_predict"):
        print("\n=== Full train → test prediction ===", flush=True)
        model_full = LGBMRegressor(**params)
        lr_schedule = build_lgbm_lr_schedule(
            n_estimators=int(params["n_estimators"]),
            base_lr=float(params["learning_rate"]),
        )
        model_full.fit(
            train_df[feature_cols], train_df[TARGET].to_numpy(),
            callbacks=[log_evaluation(-1), reset_parameter(learning_rate=lr_schedule)],
        )

        pred_test = np.clip(model_full.predict(test_df[feature_cols]), 0, None)
        sub = bundle.sample_submission.copy()
        pred_map = pd.Series(pred_test, index=test_df["ID"].values)
        sub[TARGET] = sub["ID"].map(pred_map).to_numpy()

        out_dir = args.output_dir / f"scen_agg_{run_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        sub_path = out_dir / "submission_scen_agg.csv"
        sub.to_csv(sub_path, index=False)
        print(f"Saved → {sub_path}", flush=True)

        write_yaml_summary(
            args.experiment_dir / "runs" / f"{run_id}_scen_agg.yaml",
            {
                "run_id": run_id,
                "new_features": new_cols,
                "feature_count": len(feature_cols),
                "split_metrics": {k: to_builtin(v) for k, v in split_metrics.items()},
                "avg_mae": to_builtin(avg_mae) if split_metrics else None,
                "stress": to_builtin(stress) if split_metrics else None,
                "submission": str(sub_path),
            },
        )

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
