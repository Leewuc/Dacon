from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from compare_models import (
    TARGET,
    append_csv_log,
    build_feature_frame,
    fit_catboost,
    fit_lightgbm,
    load_bundle,
    split_by_group,
    utc_now_string,
    write_yaml_summary,
)

CSV_COLUMNS = [
    "run_id",
    "timestamp_utc",
    "split_name",
    "weight_lgbm",
    "mae",
    "rmse",
    "n_train",
    "n_valid",
    "notes",
]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--experiment-dir", type=Path, default=Path(__file__).resolve().parent / "experiments")
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--weight-start", type=float, default=0.3)
    parser.add_argument("--weight-end", type=float, default=0.9)
    parser.add_argument("--weight-step", type=float, default=0.02)
    parser.add_argument(
        "--notes",
        type=str,
        default="blend weight search on fixed LightGBM/CatBoost predictions",
    )
    return parser.parse_args()


def append_rows(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    run_id = utc_now_string()

    bundle = load_bundle(args.data_dir, args.max_train_rows, None)
    train_features, feature_columns = build_feature_frame(bundle.train, bundle.layout)
    x = train_features[feature_columns]
    y = train_features[TARGET].to_numpy()
    split_masks = {
        "scenario_holdout": split_by_group(train_features, "scenario_id"),
        "layout_holdout": split_by_group(train_features, "layout_id"),
    }

    split_base_preds: dict[str, dict[str, np.ndarray]] = {}
    split_metrics: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    weights = np.arange(args.weight_start, args.weight_end + 1e-9, args.weight_step)

    for split_name, mask in split_masks.items():
        print(f"[blend-search] start split={split_name}", flush=True)
        x_train = x.loc[~mask]
        y_train = y[~mask]
        x_valid = x.loc[mask]
        y_valid = y[mask]

        _, lgbm_pred = fit_lightgbm(x_train, y_train, x_valid, y_valid)
        _, cat_pred = fit_catboost(x_train, y_train, x_valid, y_valid)
        split_base_preds[split_name] = {
            "lgbm": lgbm_pred,
            "cat": cat_pred,
        }

        best_weight = None
        best_mae = float("inf")
        best_rmse = None
        for weight in weights:
            pred = np.clip(weight * lgbm_pred + (1.0 - weight) * cat_pred, 0, None)
            current_mae = float(mean_absolute_error(y_valid, pred))
            if current_mae < best_mae:
                best_mae = current_mae
                best_rmse = rmse(y_valid, pred)
                best_weight = float(weight)
        split_metrics[split_name] = {
            "best_weight_lgbm": best_weight,
            "mae": best_mae,
            "rmse": best_rmse,
            "n_train": int(len(x_train)),
            "n_valid": int(len(x_valid)),
        }
        rows.append(
            {
                "run_id": run_id,
                "timestamp_utc": run_id,
                "split_name": split_name,
                "weight_lgbm": best_weight,
                "mae": round(best_mae, 6),
                "rmse": round(float(best_rmse), 6),
                "n_train": int(len(x_train)),
                "n_valid": int(len(x_valid)),
                "notes": args.notes,
            }
        )
        print(
            split_name,
            {"best_weight_lgbm": round(best_weight, 4), "mae": round(best_mae, 4), "rmse": round(float(best_rmse), 4)},
            flush=True,
        )

    scenario_weight = split_metrics["scenario_holdout"]["best_weight_lgbm"]
    layout_weight = split_metrics["layout_holdout"]["best_weight_lgbm"]
    averaged_weight = float(np.mean([scenario_weight, layout_weight]))
    summary = {
        "run_id": run_id,
        "timestamp_utc": run_id,
        "notes": args.notes,
        "max_train_rows": args.max_train_rows,
        "weight_range": {
            "start": args.weight_start,
            "end": args.weight_end,
            "step": args.weight_step,
        },
        "split_metrics": split_metrics,
        "recommended_weight_lgbm_mean": averaged_weight,
    }
    write_yaml_summary(args.experiment_dir / "runs" / f"{run_id}_blend_search.yaml", summary)
    append_rows(args.experiment_dir / "blend_search_log.csv", rows)

    print("recommended_weight_lgbm_mean", round(averaged_weight, 4), flush=True)


if __name__ == "__main__":
    main()
