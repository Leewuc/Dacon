"""
Adversarial Validation CV: rebuild CV split to match test distribution.

Problem: current CV uses train-distribution scenarios as holdout → CV is too
optimistic vs LB because test has different layout/congestion distribution.

Fix:
  1. Train adversarial classifier (train=0, test=1) on all features.
  2. Score every TRAINING scenario by mean P(test-like).
  3. Hold out top 20% most test-like training scenarios as validation.
  4. These scenarios have layouts/congestion closest to test → CV should
     correlate better with LB.

Also compute a blended stress using both:
  - adv_holdout (50%): scenarios most similar to test
  - scenario_holdout (30%): random scenarios for stability
  - congestion_tail (20%): hardest congestion scenarios
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, mean_absolute_error

from compare_models import (
    TARGET,
    build_split_masks_v2,
    default_feature_cache_dir,
    load_bundle,
    load_feature_cache,
    split_by_group,
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

# Weights for the new adversarial-aligned stress score
ADV_SPLIT_WEIGHTS = {
    "adv_holdout":        0.50,  # most test-like scenarios → highest weight
    "scenario_holdout":   0.30,  # random held-out scenarios → stability
    "congestion_tail_holdout": 0.20,  # hardest scenarios
}


def build_adversarial_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    holdout_frac: float = 0.20,
    seed: int = 42,
) -> pd.Series:
    """
    Train-vs-test classifier → score each train scenario by P(test-like)
    → return boolean mask: True = most test-like (holdout) scenarios.
    """
    n_tr, n_te = len(train_df), len(test_df)
    X = pd.concat([train_df[feature_cols], test_df[feature_cols]], ignore_index=True)
    y = np.array([0] * n_tr + [1] * n_te)

    clf = LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=63,
        n_jobs=8, random_state=seed, verbose=-1,
    )
    clf.fit(X, y)
    auc = roc_auc_score(y, clf.predict_proba(X)[:, 1])
    print(f"  Adversarial AUC: {auc:.4f}", flush=True)

    train_prob = clf.predict_proba(train_df[feature_cols])[:, 1]  # P(test-like)

    tmp = train_df[["scenario_id"]].copy()
    tmp["_prob"] = train_prob
    scen_prob = tmp.groupby("scenario_id")["_prob"].mean()

    threshold = float(scen_prob.quantile(1.0 - holdout_frac))
    adv_scenarios = set(scen_prob[scen_prob >= threshold].index)
    adv_mask = train_df["scenario_id"].isin(adv_scenarios)

    n_scen = len(adv_scenarios)
    n_rows = int(adv_mask.sum())
    print(f"  Adversarial holdout: {n_scen} scenarios ({n_scen/10000*100:.1f}%)  {n_rows:,} rows", flush=True)
    print(f"  Adv-holdout congestion mean: {train_df.loc[adv_mask,'congestion_score'].mean():.3f}  "
          f"(test={test_df['congestion_score'].mean():.3f}  train_all={train_df['congestion_score'].mean():.3f})", flush=True)

    return pd.Series(adv_mask.values, index=train_df.index)


def adv_weighted_stress(split_metrics: dict[str, dict[str, float]]) -> float:
    """Compute stress using ADV_SPLIT_WEIGHTS (adv_holdout weighted highest)."""
    return float(sum(
        ADV_SPLIT_WEIGHTS[k] * split_metrics[k]["mae"]
        for k in ADV_SPLIT_WEIGHTS
        if k in split_metrics
    ))


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",       type=Path, default=base_dir)
    p.add_argument("--experiment-dir", type=Path, default=base_dir / "experiments")
    p.add_argument("--output-dir",     type=Path, default=base_dir / "submissions")
    p.add_argument("--cache-name",     type=str,  default="features_v9_full")
    p.add_argument("--cache-dir",      type=Path, default=None)
    p.add_argument("--seed",           type=int,  default=42)
    p.add_argument("--holdout-frac",   type=float, default=0.20)
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

    train_df  = cache.train_features.copy()
    test_df   = cache.test_features.copy()
    base_cols = cache.feature_columns

    lgbm_params, cat_params, cat_target_mode, cat_feature_columns, cat_categorical_columns = \
        load_base_params(args.lgbm_tuning_yaml, args.base_ensemble_catboost_yaml)
    layout_base_params = load_layout_base_params(args.layout_base_tuning_yaml)
    base_lgbm_weight   = 0.86
    y = train_df[TARGET].to_numpy()

    # ── Step 1: Build adversarial split ──────────────────────────────────────
    print("Step 1: Building adversarial validation split...", flush=True)
    adv_mask = build_adversarial_split(train_df, test_df, base_cols,
                                        args.holdout_frac, args.seed)

    # Build standard splits + inject adversarial holdout
    std_masks = build_split_masks_v2(train_df, seed=args.seed)
    split_masks = {
        "adv_holdout":             adv_mask,
        "scenario_holdout":        std_masks["scenario_holdout"],
        "congestion_tail_holdout": std_masks["congestion_tail_holdout"],
        # keep for reporting but not for new stress
        "layout_holdout":          std_masks["layout_holdout"],
        "late_time_holdout":       std_masks["late_time_holdout"],
        "unseen_layout_heavy":     std_masks["unseen_layout_heavy"],
    }

    # ── Step 2: CV ───────────────────────────────────────────────────────────
    if args.mode in ("cv", "cv_predict"):
        print("\nStep 2: Building OOF predictions...", flush=True)

        y_true_by_split, base_pred_by_split = build_base_oof_predictions_cached(
            train_df, bundle, split_masks,
            base_cols,
            lgbm_params, cat_params, cat_target_mode,
            cat_feature_columns, cat_categorical_columns,
            base_lgbm_weight,
        )
        layout_pred_by_split = build_layout_oof_predictions(
            train_df, bundle, split_masks,
            base_cols, layout_base_params, "layout_cluster_blend_w030", args.seed,
        )
        xgb_pred_by_split = build_xgb_oof_predictions(
            train_df, split_masks, base_cols, "log1p_depth8_v1",
        )

        best_adv_stress  = float("inf")
        best_orig_stress = float("inf")
        best_by_adv      = None
        variants_summary: dict[str, Any] = {}

        for variant_name, cfg in VARIANTS.items():
            split_metrics: dict[str, dict[str, float]] = {}
            for split_name, valid_mask in split_masks.items():
                valid_df = train_df.loc[valid_mask.to_numpy()]
                pred     = apply_variant(
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

            adv_stress  = adv_weighted_stress(split_metrics)
            orig_stress = weighted_split_score(split_metrics)
            variants_summary[variant_name] = {
                "adv_stress": adv_stress, "orig_stress": orig_stress,
                "split_metrics": split_metrics,
            }
            if adv_stress < best_adv_stress:
                best_adv_stress = adv_stress
                best_by_adv     = variant_name
            if orig_stress < best_orig_stress:
                best_orig_stress = orig_stress

            print(f"[adv_cv] {variant_name:<40s} adv={adv_stress:.5f}  orig={orig_stress:.5f}", flush=True)

        print(f"\n=== CV Summary ===", flush=True)
        print(f"  {'variant':<40s} {'adv_stress':>10}  {'orig_stress':>11}", flush=True)
        for vn, r in sorted(variants_summary.items(), key=lambda x: x[1]["adv_stress"]):
            tag = " ← BEST" if vn == best_by_adv else ""
            print(f"  {vn:<40s} {r['adv_stress']:10.5f}  {r['orig_stress']:11.5f}{tag}", flush=True)

        print(f"\n  Best by adv_stress: {best_by_adv}  adv={best_adv_stress:.5f}", flush=True)
        print(f"  adv_holdout weights: adv_holdout=0.50  scenario_holdout=0.30  congestion_tail=0.20", flush=True)
        print(f"\n  Per-split MAE (best variant={best_by_adv}):", flush=True)
        for sn, m in variants_summary[best_by_adv]["split_metrics"].items():
            w = ADV_SPLIT_WEIGHTS.get(sn, 0.0)
            print(f"    {sn:30s} {m['mae']:.5f}  (weight={w:.2f})", flush=True)

    else:
        best_by_adv     = "triple_specialist_v5_tail_xgb"
        best_adv_stress = float("inf")
        variants_summary = {}

    # ── Step 3: Predict ───────────────────────────────────────────────────────
    if args.mode in ("predict", "cv_predict"):
        print(f"\nStep 3: Full train → predict (variant={best_by_adv}) ===", flush=True)

        base_test_pred = fit_full_base_predictions_cached(
            train_df, test_df, bundle,
            base_cols,
            lgbm_params, cat_params, cat_target_mode,
            cat_feature_columns, cat_categorical_columns,
            base_lgbm_weight,
        )
        layout_test_pred = fit_full_layout_predictions(
            train_df, test_df, bundle.layout,
            base_cols, y,
            layout_base_params, "layout_cluster_blend_w030", args.seed,
        )
        xgb_test_pred = fit_full_xgb_predictions(
            train_df, test_df, base_cols, y, "log1p_depth8_v1",
        )

        best_cfg = VARIANTS[best_by_adv]
        pred     = apply_variant(test_df, base_test_pred, layout_test_pred, xgb_test_pred, best_cfg)

        sub = bundle.sample_submission.copy()
        sub[TARGET] = pred

        out_dir  = args.output_dir / f"adv_val_{run_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        sub_path = out_dir / f"submission_adv_val_{best_by_adv}.csv"
        sub.to_csv(sub_path, index=False)
        print(f"Saved → {sub_path}", flush=True)

        write_yaml_summary(
            args.experiment_dir / "runs" / f"{run_id}_adv_val.yaml",
            {
                "run_id": run_id,
                "best_variant": best_by_adv,
                "best_adv_stress": to_builtin(best_adv_stress),
                "submission": str(sub_path),
            },
        )

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
