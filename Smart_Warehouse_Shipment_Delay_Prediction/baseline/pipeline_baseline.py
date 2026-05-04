from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation
except ImportError as exc:
    raise SystemExit(
        "lightgbm is required. Run this script with the warehouse environment: "
        "/data/conda/envs/warehouse/bin/python pipeline_baseline.py ..."
    ) from exc

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import yaml


TARGET = "avg_delay_minutes_next_30m"
DROP_COLUMNS = {"ID", "scenario_id", "layout_id", "layout_type", TARGET}
PRIMARY_METRIC = "mae"
LAG_COLUMNS = [
    "order_inflow_15m",
    "unique_sku_15m",
    "battery_mean",
    "battery_std",
    "low_battery_ratio",
    "charge_queue_length",
    "avg_charge_wait",
    "congestion_score",
    "max_zone_density",
    "pack_utilization",
    "loading_dock_util",
    "network_latency_ms",
    "label_print_queue",
    "pick_list_length_avg",
    "staging_area_util",
]
MODEL_PARAMS = {
    "objective": "regression",
    "device_type": "gpu",
    "n_estimators": 1000,
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 40,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.5,
    "force_col_wise": True,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}
CSV_LOG_COLUMNS = [
    "run_id",
    "timestamp_utc",
    "mode",
    "split_name",
    "primary_metric",
    "mae",
    "rmse",
    "n_train",
    "n_valid",
    "target_transform",
    "model_name",
    "notes",
]


@dataclass
class DatasetBundle:
    train: pd.DataFrame
    test: pd.DataFrame
    layout: pd.DataFrame
    sample_submission: pd.DataFrame


def load_bundle(base_dir: Path) -> DatasetBundle:
    return DatasetBundle(
        train=pd.read_csv(base_dir / "train.csv"),
        test=pd.read_csv(base_dir / "test.csv"),
        layout=pd.read_csv(base_dir / "layout_info.csv"),
        sample_submission=pd.read_csv(base_dir / "sample_submission.csv"),
    )


def utc_now_string() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def to_builtin(value: Any) -> Any:
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def add_base_features(df: pd.DataFrame, layout: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(layout, on="layout_id", how="left")

    layout_type_map = {
        name: idx for idx, name in enumerate(sorted(layout["layout_type"].dropna().unique()))
    }

    eps = 1e-6
    time_idx = out.groupby("scenario_id").cumcount()
    observed_robot_total = out["robot_active"] + out["robot_idle"] + out["robot_charging"]
    derived = pd.DataFrame(
        {
            "time_idx": time_idx,
            "time_frac": time_idx / 24.0,
            "layout_type_code": out["layout_type"].map(layout_type_map).fillna(-1).astype("int32"),
            "observed_robot_total": observed_robot_total,
            "active_ratio_layout": out["robot_active"] / (out["robot_total"] + eps),
            "charging_ratio_layout": out["robot_charging"] / (out["robot_total"] + eps),
            "idle_ratio_layout": out["robot_idle"] / (out["robot_total"] + eps),
            "orders_per_active_robot": out["order_inflow_15m"] / (out["robot_active"] + 1),
            "orders_per_total_robot": out["order_inflow_15m"] / (out["robot_total"] + 1),
            "queue_per_charger": out["charge_queue_length"] / (out["charger_count"] + 1),
            "dock_pressure": out["loading_dock_util"] * out["order_inflow_15m"],
            "battery_stress": out["low_battery_ratio"] * out["robot_utilization"],
            "congestion_pressure": out["congestion_score"] * out["max_zone_density"],
            "missing_count": out.isna().sum(axis=1),
        },
        index=out.index,
    )
    return pd.concat([out, derived], axis=1)


def add_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    extra_frames = []
    grouped = df.groupby("scenario_id", sort=False)
    for column in LAG_COLUMNS:
        lag1 = grouped[column].shift(1)
        extra_frames.append(
            pd.DataFrame(
                {
                    f"{column}_lag1": lag1,
                    f"{column}_lag2": grouped[column].shift(2),
                    f"{column}_diff1": df[column] - lag1,
                    f"{column}_roll3": (
                        lag1.groupby(df["scenario_id"], sort=False)
                        .rolling(3, min_periods=1)
                        .mean()
                        .reset_index(level=0, drop=True)
                    ),
                    f"{column}_miss": df[column].isna().astype("int8"),
                }
            )
        )
    return pd.concat([df] + extra_frames, axis=1)


def build_feature_frame(df: pd.DataFrame, layout: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    features = add_base_features(df, layout)
    features = add_sequence_features(features)
    feature_columns = [col for col in features.columns if col not in DROP_COLUMNS]
    return features, feature_columns


def build_model() -> LGBMRegressor:
    return LGBMRegressor(**MODEL_PARAMS)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def original_scale_mae(eval_y_log: np.ndarray, pred_y_log: np.ndarray) -> tuple[str, float, bool]:
    mae = mean_absolute_error(np.expm1(eval_y_log), np.expm1(pred_y_log))
    return "mae_original_scale", float(mae), False


def split_by_group(
    df: pd.DataFrame, group_column: str, frac: float = 0.2, seed: int = 42
) -> pd.Series:
    unique_groups = np.array(sorted(df[group_column].unique()))
    rng = np.random.default_rng(seed)
    valid_groups = set(rng.choice(unique_groups, size=int(len(unique_groups) * frac), replace=False))
    return df[group_column].isin(valid_groups)


def build_run_metadata(args: argparse.Namespace) -> dict[str, Any]:
    timestamp = utc_now_string()
    return {
        "run_id": timestamp,
        "timestamp_utc": timestamp,
        "mode": args.mode,
        "target": TARGET,
        "primary_metric": PRIMARY_METRIC,
        "target_transform": "log1p",
        "model_name": "LightGBM",
        "model_params": MODEL_PARAMS,
        "lag_columns": LAG_COLUMNS,
        "notes": args.notes,
        "data_dir": str(args.data_dir.resolve()),
    }


def append_csv_log(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_LOG_COLUMNS)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: to_builtin(row.get(key, "")) for key in CSV_LOG_COLUMNS})


def write_yaml_summary(yaml_path: Path, payload: dict[str, Any]) -> None:
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with yaml_path.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(payload, fp, allow_unicode=True, sort_keys=False)


def run_cv(bundle: DatasetBundle, args: argparse.Namespace) -> list[dict[str, Any]]:
    train_features, feature_columns = build_feature_frame(bundle.train, bundle.layout)
    x = train_features[feature_columns]
    y = train_features[TARGET].to_numpy()

    scenario_mask = split_by_group(train_features, "scenario_id")
    layout_mask = split_by_group(train_features, "layout_id")

    rows = []
    for split_name, mask in [("scenario_holdout", scenario_mask), ("layout_holdout", layout_mask)]:
        model = build_model()
        model.fit(
            x.loc[~mask],
            np.log1p(y[~mask]),
            eval_set=[(x.loc[mask], np.log1p(y[mask]))],
            eval_metric=original_scale_mae,
            callbacks=[early_stopping(100), log_evaluation(0)],
        )
        pred = np.expm1(model.predict(x.loc[mask]))
        metrics = {
            "run_id": args.run_id,
            "timestamp_utc": args.timestamp_utc,
            "mode": "cv",
            "split_name": split_name,
            "primary_metric": PRIMARY_METRIC,
            "mae": round(float(mean_absolute_error(y[mask], pred)), 6),
            "rmse": round(rmse(y[mask], pred), 6),
            "n_train": int((~mask).sum()),
            "n_valid": int(mask.sum()),
            "target_transform": "log1p",
            "model_name": "LightGBM",
            "notes": args.notes,
        }
        rows.append(metrics)
        print(
            split_name,
            {
                "mae": round(metrics["mae"], 4),
                "rmse": round(metrics["rmse"], 4),
                "n_valid": metrics["n_valid"],
            },
        )
    return rows


def fit_predict(bundle: DatasetBundle, output_path: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    train_features, feature_columns = build_feature_frame(bundle.train, bundle.layout)
    test_features, _ = build_feature_frame(bundle.test, bundle.layout)

    model = build_model()
    y = train_features[TARGET].to_numpy()
    model.fit(train_features[feature_columns], np.log1p(y))
    pred = np.expm1(model.predict(test_features[feature_columns]))

    submission = bundle.sample_submission.copy()
    submission[TARGET] = pred
    submission.to_csv(output_path, index=False)
    print(f"saved submission to {output_path}")
    return [
        {
            "run_id": args.run_id,
            "timestamp_utc": args.timestamp_utc,
            "mode": "predict",
            "split_name": "full_train",
            "primary_metric": PRIMARY_METRIC,
            "mae": "",
            "rmse": "",
            "n_train": int(len(train_features)),
            "n_valid": int(len(test_features)),
            "target_transform": "log1p",
            "model_name": "LightGBM",
            "notes": args.notes,
        }
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing train.csv, test.csv, layout_info.csv, and sample_submission.csv",
    )
    parser.add_argument(
        "--mode",
        choices=["cv", "predict"],
        default="cv",
        help="Run holdout validation or fit on full train and write submission predictions",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "submission_baseline.csv",
        help="Output path for prediction mode",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "experiments",
        help="Directory used for run summaries and the experiment log CSV",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="baseline LightGBM with layout merge and scenario lag features",
        help="Short free-form note saved into the YAML summary and CSV log",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = build_run_metadata(args)
    args.run_id = metadata["run_id"]
    args.timestamp_utc = metadata["timestamp_utc"]
    bundle = load_bundle(args.data_dir)
    csv_path = args.experiment_dir / "experiment_log.csv"
    yaml_path = args.experiment_dir / "runs" / f"{args.run_id}_{args.mode}.yaml"
    if args.mode == "cv":
        rows = run_cv(bundle, args)
        write_yaml_summary(yaml_path, {**metadata, "results": rows})
        append_csv_log(csv_path, rows)
        return
    rows = fit_predict(bundle, args.output, args)
    write_yaml_summary(
        yaml_path,
        {
            **metadata,
            "output_path": str(args.output.resolve()),
            "results": rows,
        },
    )
    append_csv_log(csv_path, rows)


if __name__ == "__main__":
    main()
