from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from compare_models import (
    TARGET,
    build_and_save_feature_cache,
    build_split_masks_v2,
    default_feature_cache_dir,
    load_bundle,
    load_feature_cache,
    utc_now_string,
    write_yaml_summary,
)


LOG_COLUMNS = [
    "run_id",
    "timestamp_utc",
    "variant",
    "split_name",
    "mae",
    "rmse",
    "n_train",
    "n_valid",
    "avg_mae",
    "stress_score",
    "notes",
]

VARIANTS = {
    "raw_depth8_v1": {
        "target_mode": "raw",
        "params": {
            "n_estimators": 1500,
            "learning_rate": 0.03,
            "max_depth": 8,
            "min_child_weight": 6.0,
            "subsample": 0.8,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.2,
            "reg_lambda": 1.0,
            "gamma": 0.0,
        },
    },
    "log1p_depth8_v1": {
        "target_mode": "log1p",
        "params": {
            "n_estimators": 1500,
            "learning_rate": 0.03,
            "max_depth": 8,
            "min_child_weight": 6.0,
            "subsample": 0.8,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.2,
            "reg_lambda": 1.0,
            "gamma": 0.0,
        },
    },
    "raw_depth10_v2": {
        "target_mode": "raw",
        "params": {
            "n_estimators": 1800,
            "learning_rate": 0.02,
            "max_depth": 10,
            "min_child_weight": 8.0,
            "subsample": 0.75,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.5,
            "reg_lambda": 1.5,
            "gamma": 0.1,
        },
    },
    "log1p_depth10_v2": {
        "target_mode": "log1p",
        "params": {
            "n_estimators": 1800,
            "learning_rate": 0.02,
            "max_depth": 10,
            "min_child_weight": 8.0,
            "subsample": 0.75,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.5,
            "reg_lambda": 1.5,
            "gamma": 0.1,
        },
    },
}

SPLIT_WEIGHTS = {
    "scenario_holdout": 0.20,
    "layout_holdout": 0.20,
    "late_time_holdout": 0.15,
    "unseen_layout_heavy": 0.15,
    "congestion_tail_holdout": 0.30,
}

BASE_XGB_PARAMS = {
    "objective": "reg:absoluteerror",
    "eval_metric": "mae",
    "tree_method": "hist",
    "device": "cuda",
    "random_state": 42,
    "n_jobs": 8,
    "verbosity": 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--experiment-dir", type=Path, default=Path(__file__).resolve().parent / "experiments")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "submissions")
    parser.add_argument("--cache-name", type=str, default="features_v8_full")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["cv", "predict"], default="cv")
    parser.add_argument(
        "--fixed-variant",
        type=str,
        default=None,
        choices=[*VARIANTS.keys()],
        help="Skip variant search and use a fixed XGBoost variant for prediction.",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="xgboost gpu experiments on cv_pack_v2 splits",
    )
    return parser.parse_args()


def append_log(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=LOG_COLUMNS)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def weighted_split_score(split_metrics: dict[str, dict[str, float]]) -> float:
    return float(
        sum(SPLIT_WEIGHTS[name] * split_metrics[name]["mae"] for name in SPLIT_WEIGHTS if name in split_metrics)
    )


def build_model(params: dict[str, Any]) -> XGBRegressor:
    full = dict(BASE_XGB_PARAMS)
    full.update(params)
    return XGBRegressor(**full)


def fit_predict(
    x_train,
    y_train,
    x_valid,
    y_valid,
    params: dict[str, Any],
    target_mode: str,
) -> np.ndarray:
    model = build_model(params)
    y_train_fit = np.log1p(y_train) if target_mode == "log1p" else y_train
    y_valid_fit = np.log1p(y_valid) if target_mode == "log1p" else y_valid
    model.fit(
        x_train,
        y_train_fit,
        eval_set=[(x_valid, y_valid_fit)],
        verbose=False,
    )
    pred = model.predict(x_valid)
    if target_mode == "log1p":
        pred = np.expm1(pred)
    return np.clip(np.asarray(pred), 0, None)


def fit_predict_noeval(x_train, y_train, x_test, params: dict[str, Any], target_mode: str) -> np.ndarray:
    model = build_model(params)
    y_train_fit = np.log1p(y_train) if target_mode == "log1p" else y_train
    model.fit(x_train, y_train_fit, verbose=False)
    pred = model.predict(x_test)
    if target_mode == "log1p":
        pred = np.expm1(pred)
    return np.clip(np.asarray(pred), 0, None)


def main() -> None:
    args = parse_args()
    run_id = utc_now_string()
    cache_dir = args.cache_dir or default_feature_cache_dir(args.data_dir, args.cache_name)
    bundle = load_bundle(args.data_dir, args.max_train_rows, args.max_test_rows)
    if cache_dir.exists():
        cache = load_feature_cache(cache_dir)
    else:
        cache = build_and_save_feature_cache(bundle, cache_dir, args.cache_name, args.notes)

    train_df = cache.train_features
    x = train_df[cache.feature_columns]
    y = train_df[TARGET].to_numpy()
    x_test = cache.test_features[cache.feature_columns]
    split_masks = build_split_masks_v2(train_df, seed=args.seed)

    if args.mode == "predict" and args.fixed_variant:
        best_cfg = VARIANTS[args.fixed_variant]
        pred = fit_predict_noeval(x, y, x_test, best_cfg["params"], best_cfg["target_mode"])
        out_dir = args.output_dir / f"xgboost_{run_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        submission = bundle.sample_submission.copy()
        submission[TARGET] = pred
        submission_path = out_dir / f"submission_{args.fixed_variant}.csv"
        submission.to_csv(submission_path, index=False)
        summary = {
            "run_id": run_id,
            "timestamp_utc": run_id,
            "notes": args.notes,
            "cache_dir": str(cache_dir.resolve()),
            "fixed_variant": args.fixed_variant,
            "best_by_avg": args.fixed_variant,
            "best_avg_mae": None,
            "best_by_stress": args.fixed_variant,
            "best_stress_score": None,
            "variants": {
                args.fixed_variant: {
                    "config": best_cfg,
                }
            },
            "submission_path": str(submission_path.resolve()),
        }
        yaml_path = args.experiment_dir / "runs" / f"{run_id}_xgboost.yaml"
        write_yaml_summary(yaml_path, summary)
        write_yaml_summary(args.experiment_dir / "current_xgboost.yaml", summary)
        print(f"[xgboost] fixed_variant={args.fixed_variant} submission={submission_path}", flush=True)
        return

    rows: list[dict[str, Any]] = []
    variants_summary: dict[str, Any] = {}
    best_by_avg = args.fixed_variant
    best_avg_mae = float("inf")
    best_by_stress = args.fixed_variant
    best_stress_score = float("inf")

    variant_items = [(args.fixed_variant, VARIANTS[args.fixed_variant])] if args.fixed_variant else list(VARIANTS.items())
    for variant_name, cfg in variant_items:
        split_metrics: dict[str, dict[str, float]] = {}
        maes: list[float] = []
        for split_name, valid_mask in split_masks.items():
            train_mask = (~valid_mask).to_numpy()
            valid_mask_np = valid_mask.to_numpy()
            if int(train_mask.sum()) == 0 or int(valid_mask_np.sum()) == 0:
                continue
            pred = fit_predict(
                x.loc[train_mask],
                y[train_mask],
                x.loc[valid_mask_np],
                y[valid_mask_np],
                cfg["params"],
                cfg["target_mode"],
            )
            mae = float(mean_absolute_error(y[valid_mask_np], pred))
            split_metrics[split_name] = {
                "mae": mae,
                "rmse": rmse(y[valid_mask_np], pred),
                "n_train": int(train_mask.sum()),
                "n_valid": int(valid_mask_np.sum()),
            }
            maes.append(mae)
        avg_mae = float(np.mean(maes))
        stress_score = weighted_split_score(split_metrics)
        variants_summary[variant_name] = {
            "config": cfg,
            "avg_mae": avg_mae,
            "stress_score": stress_score,
            "split_metrics": split_metrics,
        }
        if avg_mae < best_avg_mae:
            best_avg_mae = avg_mae
            best_by_avg = variant_name
        if stress_score < best_stress_score:
            best_stress_score = stress_score
            best_by_stress = variant_name
        for split_name, metric in split_metrics.items():
            rows.append(
                {
                    "run_id": run_id,
                    "timestamp_utc": run_id,
                    "variant": variant_name,
                    "split_name": split_name,
                    "mae": round(metric["mae"], 6),
                    "rmse": round(metric["rmse"], 6),
                    "n_train": metric["n_train"],
                    "n_valid": metric["n_valid"],
                    "avg_mae": round(avg_mae, 6),
                    "stress_score": round(stress_score, 6),
                    "notes": args.notes,
                }
            )
            print(
                f"[xgboost] variant={variant_name} split={split_name} mae={metric['mae']:.6f} stress={stress_score:.6f}",
                flush=True,
            )

    summary = {
        "run_id": run_id,
        "timestamp_utc": run_id,
        "notes": args.notes,
        "cache_dir": str(cache_dir.resolve()),
        "fixed_variant": args.fixed_variant,
        "best_by_avg": best_by_avg,
        "best_avg_mae": best_avg_mae,
        "best_by_stress": best_by_stress,
        "best_stress_score": best_stress_score,
        "variants": variants_summary,
    }

    if args.mode == "predict":
        best_cfg = VARIANTS[best_by_stress]
        pred = fit_predict_noeval(x, y, x_test, best_cfg["params"], best_cfg["target_mode"])
        out_dir = args.output_dir / f"xgboost_{run_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        submission = bundle.sample_submission.copy()
        submission[TARGET] = pred
        submission_path = out_dir / f"submission_{best_by_stress}.csv"
        submission.to_csv(submission_path, index=False)
        summary["submission_path"] = str(submission_path.resolve())

    yaml_path = args.experiment_dir / "runs" / f"{run_id}_xgboost.yaml"
    write_yaml_summary(yaml_path, summary)
    write_yaml_summary(args.experiment_dir / "current_xgboost.yaml", summary)
    append_log(args.experiment_dir / "xgboost_log.csv", rows)


if __name__ == "__main__":
    main()
