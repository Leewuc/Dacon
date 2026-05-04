from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error


TARGET = "avg_delay_minutes_next_30m"
CSV_LOG_COLUMNS = [
    "run_id",
    "timestamp_utc",
    "mode",
    "split_name",
    "model_name",
    "target_mode",
    "profile",
    "bootstrap_mode",
    "feature_set",
    "mae",
    "rmse",
    "n_train",
    "n_valid",
    "notes",
]
CATBOOST_BASE_PARAMS = {
    "loss_function": "MAE",
    "eval_metric": "MAE",
    "task_type": "GPU",
    "devices": "0",
    "bootstrap_type": "Bernoulli",
    "gpu_ram_part": 0.2,
    "thread_count": 8,
    "iterations": 1600,
    "learning_rate": 0.03,
    "depth": 8,
    "l2_leaf_reg": 5.0,
    "subsample": 0.8,
    "random_strength": 0.5,
    "min_data_in_leaf": 40,
    "random_seed": 42,
    "allow_writing_files": False,
    "verbose": False,
}
CATBOOST_PROFILE_PRESETS = {
    "baseline_v1": {},
    "deep_log_v2": {
        "iterations": 2200,
        "learning_rate": 0.02,
        "depth": 10,
        "l2_leaf_reg": 8.0,
        "subsample": 0.85,
        "random_strength": 1.0,
        "min_data_in_leaf": 20,
    },
    "balanced_v2": {
        "iterations": 1800,
        "learning_rate": 0.025,
        "depth": 9,
        "l2_leaf_reg": 5.0,
        "subsample": 0.9,
        "random_strength": 0.5,
        "min_data_in_leaf": 20,
    },
    "deep_log_v3": {
        "iterations": 2800,
        "learning_rate": 0.016,
        "depth": 10,
        "l2_leaf_reg": 6.0,
        "subsample": 0.8,
        "random_strength": 1.5,
        "min_data_in_leaf": 10,
    },
}
BOOTSTRAP_PRESETS = {
    "bernoulli_v1": {
        "bootstrap_type": "Bernoulli",
        "subsample": 0.85,
    },
    "bayesian_soft": {
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 0.5,
    },
    "bayesian_hard": {
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 1.5,
    },
}
TEMPORAL_COLUMNS = [
    "order_inflow_15m",
    "pack_utilization",
    "loading_dock_util",
    "congestion_score",
    "max_zone_density",
    "low_battery_ratio",
    "charge_queue_length",
    "battery_mean",
]
STATE_BUCKET_COLUMNS = [
    "order_inflow_15m",
    "pack_utilization",
    "loading_dock_util",
    "congestion_score",
    "max_zone_density",
    "low_battery_ratio",
    "charge_queue_length",
    "avg_trip_distance",
    "orders_per_robot",
    "items_inflow_15m",
    "pack_pressure",
    "dock_pressure",
    "congestion_pressure",
    "battery_pressure",
]
SLIM_NUMERIC_COLUMNS = [
    "order_inflow_15m",
    "avg_items_per_order",
    "robot_total",
    "robot_utilization",
    "avg_trip_distance",
    "battery_mean",
    "low_battery_ratio",
    "charge_queue_length",
    "congestion_score",
    "max_zone_density",
    "pack_utilization",
    "loading_dock_util",
    "label_print_queue",
    "pick_list_length_avg",
    "items_inflow_15m",
    "orders_per_robot",
    "pack_pressure",
    "dock_pressure",
    "congestion_pressure",
    "battery_pressure",
    "layout_robot_pack_ratio",
    "robot_to_charger_ratio",
]
CAT_HEAVY_NUMERIC_COLUMNS = [
    "order_inflow_15m",
    "avg_items_per_order",
    "robot_total",
    "robot_utilization",
    "avg_trip_distance",
    "battery_mean",
    "low_battery_ratio",
    "charge_queue_length",
    "congestion_score",
    "max_zone_density",
    "pack_utilization",
    "loading_dock_util",
    "label_print_queue",
    "pick_list_length_avg",
    "items_inflow_15m",
    "orders_per_robot",
    "pack_pressure",
    "dock_pressure",
    "congestion_pressure",
    "battery_pressure",
    "layout_robot_pack_ratio",
    "robot_to_charger_ratio",
]
RANKED_CORE_NUMERIC_COLUMNS = [
    "avg_trip_distance",
    "congestion_score",
    "max_zone_density",
    "pack_utilization",
    "loading_dock_util",
    "low_battery_ratio",
    "order_inflow_15m",
    "battery_mean",
    "charge_queue_length",
    "pack_station_count",
    "layout_compactness",
    "layout_robot_pack_ratio",
]
RANKED_SUPPORT_NUMERIC_COLUMNS = [
    "label_print_queue",
    "pick_list_length_avg",
    "robot_total",
    "charger_count",
    "orders_per_robot",
    "items_inflow_15m",
    "robot_to_charger_ratio",
    "time_idx",
]
RANKED_STAGE_TEMPORAL_BASE = [
    "congestion_score",
    "pack_utilization",
    "loading_dock_util",
    "low_battery_ratio",
    "order_inflow_15m",
    "charge_queue_length",
]
RANKED_STAGE_TEMPORAL_PLUS_BASE = [
    "congestion_score",
    "pack_utilization",
    "loading_dock_util",
    "low_battery_ratio",
    "order_inflow_15m",
    "charge_queue_length",
    "max_zone_density",
    "battery_mean",
]
RANKED_STAGE_BUCKET_COLUMNS = [
    "congestion_score_bucket",
    "pack_utilization_bucket",
    "loading_dock_util_bucket",
    "low_battery_ratio_bucket",
    "order_inflow_15m_bucket",
    "avg_trip_distance_bucket",
    "pack_pressure_bucket",
    "congestion_pressure_bucket",
]
RANKED_STAGE_CROSS_CATS = [
    "layout_congestion_bucket_cat",
    "layout_pack_bucket_cat",
    "time_congestion_bucket_cat",
    "time_pack_bucket_cat",
    "layout_order_bucket_cat",
    "layout_time_bucket_cat",
    "layout_battery_bucket_cat",
    "time_order_bucket_cat",
]


def utc_now_string() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def split_by_group(df: pd.DataFrame, group_column: str, frac: float = 0.2, seed: int = 42) -> pd.Series:
    unique_groups = np.array(sorted(df[group_column].unique()))
    rng = np.random.default_rng(seed)
    valid_groups = set(rng.choice(unique_groups, size=max(1, int(len(unique_groups) * frac)), replace=False))
    return df[group_column].isin(valid_groups)


def write_yaml_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def append_csv_log(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_LOG_COLUMNS)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_bundle(base_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(base_dir / "train.csv")
    test = pd.read_csv(base_dir / "test.csv")
    layout = pd.read_csv(base_dir / "layout_info.csv")
    sample_submission = pd.read_csv(base_dir / "sample_submission.csv")
    return train, test, layout, sample_submission


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("scenario_id", sort=False)
    out = df.copy()
    for col in TEMPORAL_COLUMNS:
        lag1 = grouped[col].shift(1)
        lag2 = grouped[col].shift(2)
        lag3 = grouped[col].shift(3)
        roll3 = lag1.groupby(df["scenario_id"], sort=False).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
        roll3_std = lag1.groupby(df["scenario_id"], sort=False).rolling(3, min_periods=2).std().reset_index(level=0, drop=True)
        out[f"{col}_lag1"] = lag1
        out[f"{col}_lag2"] = lag2
        out[f"{col}_lag3"] = lag3
        out[f"{col}_diff1"] = out[col] - lag1
        out[f"{col}_diff2"] = lag1 - lag2
        out[f"{col}_roll3_mean"] = roll3
        out[f"{col}_roll3_std"] = roll3_std
    return out


def add_state_bucket_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    derived: dict[str, Any] = {}
    for col in STATE_BUCKET_COLUMNS:
        ranks = out[col].rank(method="average", pct=True)
        bucket = pd.cut(
            ranks,
            bins=[-0.001, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=["q1", "q2", "q3", "q4", "q5"],
        ).astype(str)
        derived[f"{col}_bucket"] = bucket
    bucket_frame = pd.DataFrame(derived, index=out.index)
    bucket_frame["layout_congestion_bucket_cat"] = (
        out["layout_type"].astype(str) + "__" + bucket_frame["congestion_score_bucket"]
    )
    bucket_frame["layout_pack_bucket_cat"] = (
        out["layout_type"].astype(str) + "__" + bucket_frame["pack_utilization_bucket"]
    )
    bucket_frame["time_congestion_bucket_cat"] = (
        out["time_idx_bucket"].astype(str) + "__" + bucket_frame["congestion_score_bucket"]
    )
    bucket_frame["time_pack_bucket_cat"] = (
        out["time_idx_bucket"].astype(str) + "__" + bucket_frame["pack_utilization_bucket"]
    )
    bucket_frame["time_battery_bucket_cat"] = (
        out["time_idx_bucket"].astype(str) + "__" + bucket_frame["low_battery_ratio_bucket"]
    )
    bucket_frame["layout_order_bucket_cat"] = (
        out["layout_type"].astype(str) + "__" + bucket_frame["order_inflow_15m_bucket"]
    )
    bucket_frame["layout_time_bucket_cat"] = (
        out["layout_id"].astype(str) + "__" + out["time_idx_bucket"].astype(str)
    )
    bucket_frame["layout_battery_bucket_cat"] = (
        out["layout_type"].astype(str) + "__" + bucket_frame["low_battery_ratio_bucket"]
    )
    bucket_frame["layout_dock_bucket_cat"] = (
        out["layout_type"].astype(str) + "__" + bucket_frame["loading_dock_util_bucket"]
    )
    bucket_frame["time_order_bucket_cat"] = (
        out["time_idx_bucket"].astype(str) + "__" + bucket_frame["order_inflow_15m_bucket"]
    )
    bucket_frame["layout_pack_pressure_bucket_cat"] = (
        out["layout_type"].astype(str) + "__" + bucket_frame["pack_pressure_bucket"]
    )
    bucket_frame["layout_congestion_pressure_bucket_cat"] = (
        out["layout_type"].astype(str) + "__" + bucket_frame["congestion_pressure_bucket"]
    )
    return pd.concat([out, bucket_frame], axis=1)


def prepare_catboost_frame(df: pd.DataFrame, layout: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    out = df.merge(layout, on="layout_id", how="left")
    derived = pd.DataFrame(
        {
            "time_idx": out.groupby("scenario_id").cumcount().astype(int),
        },
        index=out.index,
    )
    derived["time_idx_bucket"] = pd.cut(
        derived["time_idx"],
        bins=[-1, 4, 9, 14, 19, 24],
        labels=["t00_04", "t05_09", "t10_14", "t15_19", "t20_24"],
    ).astype(str)
    derived["shift_hour_cat"] = out["shift_hour"].astype("Int64").astype(str)
    derived["day_of_week_cat"] = out["day_of_week"].astype("Int64").astype(str)
    derived["layout_shift_cat"] = out["layout_type"].astype(str) + "__" + derived["shift_hour_cat"]
    derived["layout_day_cat"] = out["layout_type"].astype(str) + "__" + derived["day_of_week_cat"]
    derived["orders_per_robot"] = out["order_inflow_15m"] / (out["robot_total"] + 1)
    derived["items_inflow_15m"] = out["order_inflow_15m"] * out["avg_items_per_order"]
    derived["pack_pressure"] = out["pack_utilization"] * out["order_inflow_15m"]
    derived["dock_pressure"] = out["loading_dock_util"] * out["order_inflow_15m"]
    derived["congestion_pressure"] = out["congestion_score"] * out["max_zone_density"]
    derived["battery_pressure"] = out["low_battery_ratio"] * out["charge_queue_length"]
    derived["layout_robot_pack_ratio"] = out["robot_total"] / (out["pack_station_count"] + 1)
    derived["robot_to_charger_ratio"] = out["robot_total"] / (out["charger_count"] + 1)
    out = pd.concat([out, derived], axis=1)
    out = add_temporal_features(out)
    out = add_state_bucket_features(out)

    categorical_columns = [
        "layout_id",
        "layout_type",
        "day_of_week_cat",
        "shift_hour_cat",
        "time_idx_bucket",
        "layout_shift_cat",
        "layout_day_cat",
        "order_inflow_15m_bucket",
        "pack_utilization_bucket",
        "loading_dock_util_bucket",
        "congestion_score_bucket",
        "max_zone_density_bucket",
        "low_battery_ratio_bucket",
        "charge_queue_length_bucket",
        "avg_trip_distance_bucket",
        "layout_congestion_bucket_cat",
        "layout_pack_bucket_cat",
        "time_congestion_bucket_cat",
        "time_pack_bucket_cat",
        "time_battery_bucket_cat",
        "layout_order_bucket_cat",
        "orders_per_robot_bucket",
        "items_inflow_15m_bucket",
        "pack_pressure_bucket",
        "dock_pressure_bucket",
        "congestion_pressure_bucket",
        "battery_pressure_bucket",
        "layout_time_bucket_cat",
        "layout_battery_bucket_cat",
        "layout_dock_bucket_cat",
        "time_order_bucket_cat",
        "layout_pack_pressure_bucket_cat",
        "layout_congestion_pressure_bucket_cat",
    ]
    drop_columns = {"ID", "scenario_id", TARGET, "day_of_week", "shift_hour"}
    feature_columns = [col for col in out.columns if col not in drop_columns]

    for col in categorical_columns:
        out[col] = out[col].astype(str).fillna("missing")

    numeric_columns = [col for col in feature_columns if col not in categorical_columns]
    fill_values = out[numeric_columns].median(numeric_only=True)
    out[numeric_columns] = out[numeric_columns].fillna(fill_values).fillna(0.0)
    return out, feature_columns, categorical_columns


def select_feature_set(
    feature_columns: list[str],
    categorical_columns: list[str],
    feature_set: str,
) -> list[str]:
    if feature_set == "full":
        return feature_columns
    if feature_set == "ranked_s1_core":
        selected = {
            "layout_id",
            "layout_type",
            "time_idx_bucket",
            "shift_hour_cat",
            "day_of_week_cat",
            "layout_shift_cat",
            "layout_day_cat",
            *RANKED_CORE_NUMERIC_COLUMNS,
        }
        return [col for col in feature_columns if col in selected]
    if feature_set == "ranked_s2_support":
        selected = {
            "layout_id",
            "layout_type",
            "time_idx_bucket",
            "shift_hour_cat",
            "day_of_week_cat",
            "layout_shift_cat",
            "layout_day_cat",
            *RANKED_CORE_NUMERIC_COLUMNS,
            *RANKED_SUPPORT_NUMERIC_COLUMNS,
        }
        return [col for col in feature_columns if col in selected]
    if feature_set == "ranked_s3_temporal":
        selected = {
            "layout_id",
            "layout_type",
            "time_idx_bucket",
            "shift_hour_cat",
            "day_of_week_cat",
            "layout_shift_cat",
            "layout_day_cat",
            *RANKED_CORE_NUMERIC_COLUMNS,
            *RANKED_SUPPORT_NUMERIC_COLUMNS,
        }
        for col in RANKED_STAGE_TEMPORAL_BASE:
            selected.update({f"{col}_lag1", f"{col}_lag2", f"{col}_diff1", f"{col}_roll3_mean"})
        return [col for col in feature_columns if col in selected]
    if feature_set == "ranked_s3_temporal_plus":
        selected = {
            "layout_id",
            "layout_type",
            "time_idx_bucket",
            "shift_hour_cat",
            "day_of_week_cat",
            "layout_shift_cat",
            "layout_day_cat",
            *RANKED_CORE_NUMERIC_COLUMNS,
            *RANKED_SUPPORT_NUMERIC_COLUMNS,
        }
        for col in RANKED_STAGE_TEMPORAL_PLUS_BASE:
            selected.update(
                {
                    f"{col}_lag1",
                    f"{col}_lag2",
                    f"{col}_lag3",
                    f"{col}_diff1",
                    f"{col}_diff2",
                    f"{col}_roll3_mean",
                    f"{col}_roll3_std",
                }
            )
        return [col for col in feature_columns if col in selected]
    if feature_set == "ranked_s4_bucket":
        selected = {
            "layout_id",
            "layout_type",
            "time_idx_bucket",
            "shift_hour_cat",
            "day_of_week_cat",
            "layout_shift_cat",
            "layout_day_cat",
            *RANKED_CORE_NUMERIC_COLUMNS,
            *RANKED_SUPPORT_NUMERIC_COLUMNS,
            *RANKED_STAGE_BUCKET_COLUMNS,
        }
        for col in RANKED_STAGE_TEMPORAL_BASE:
            selected.update({f"{col}_lag1", f"{col}_lag2", f"{col}_diff1", f"{col}_roll3_mean"})
        return [col for col in feature_columns if col in selected]
    if feature_set == "ranked_s5_cross":
        selected = {
            "layout_id",
            "layout_type",
            "time_idx_bucket",
            "shift_hour_cat",
            "day_of_week_cat",
            "layout_shift_cat",
            "layout_day_cat",
            *RANKED_CORE_NUMERIC_COLUMNS,
            *RANKED_SUPPORT_NUMERIC_COLUMNS,
            *RANKED_STAGE_BUCKET_COLUMNS,
            *RANKED_STAGE_CROSS_CATS,
            "orders_per_robot_bucket",
            "items_inflow_15m_bucket",
            "time_battery_bucket_cat",
        }
        for col in RANKED_STAGE_TEMPORAL_BASE:
            selected.update({f"{col}_lag1", f"{col}_lag2", f"{col}_diff1", f"{col}_roll3_mean"})
        return [col for col in feature_columns if col in selected]
    if feature_set == "cat_heavy":
        selected = set(categorical_columns)
        selected.update(CAT_HEAVY_NUMERIC_COLUMNS)
        selected.update({f"{col}_lag1" for col in TEMPORAL_COLUMNS})
        selected.update({f"{col}_lag2" for col in TEMPORAL_COLUMNS})
        selected.update({f"{col}_diff1" for col in TEMPORAL_COLUMNS})
        selected.update({f"{col}_roll3_mean" for col in TEMPORAL_COLUMNS})
        return [col for col in feature_columns if col in selected]
    slim_candidates = set(categorical_columns)
    slim_candidates.update(SLIM_NUMERIC_COLUMNS)
    slim_candidates.update(
        {
            f"{col}_lag1" for col in TEMPORAL_COLUMNS
        }
    )
    slim_candidates.update(
        {
            f"{col}_diff1" for col in TEMPORAL_COLUMNS
        }
    )
    slim_candidates.update(
        {
            f"{col}_roll3_mean" for col in TEMPORAL_COLUMNS
        }
    )
    return [col for col in feature_columns if col in slim_candidates]


def build_params(profile: str, bootstrap_mode: str) -> dict[str, Any]:
    params = {**CATBOOST_BASE_PARAMS, **CATBOOST_PROFILE_PRESETS[profile], **BOOTSTRAP_PRESETS[bootstrap_mode]}
    if params.get("bootstrap_type") == "Bayesian":
        params.pop("subsample", None)
    else:
        params.pop("bagging_temperature", None)
    return params


def make_pool(
    df: pd.DataFrame,
    y: np.ndarray | None,
    feature_columns: list[str],
    categorical_columns: list[str],
) -> Pool:
    x = df[feature_columns].copy()
    cat_indices = [feature_columns.index(col) for col in categorical_columns]
    return Pool(x, label=y, cat_features=cat_indices)


def fit_predict(
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    valid_df: pd.DataFrame,
    y_valid: np.ndarray | None,
    feature_columns: list[str],
    categorical_columns: list[str],
    params: dict[str, Any],
    target_mode: str,
) -> np.ndarray:
    y_train_fit = np.log1p(y_train) if target_mode == "log1p" else y_train
    train_pool = make_pool(train_df, y_train_fit, feature_columns, categorical_columns)
    eval_set = None
    if y_valid is not None:
        y_valid_fit = np.log1p(y_valid) if target_mode == "log1p" else y_valid
        eval_set = make_pool(valid_df, y_valid_fit, feature_columns, categorical_columns)
    model = CatBoostRegressor(**params)
    model.fit(train_pool, eval_set=eval_set, use_best_model=y_valid is not None)
    pred = model.predict(make_pool(valid_df, None, feature_columns, categorical_columns))
    if target_mode == "log1p":
        pred = np.expm1(pred)
    return np.clip(np.asarray(pred), 0, None)


def build_metric_row(
    run_id: str,
    mode: str,
    split_name: str,
    model_name: str,
    target_mode: str,
    profile: str,
    bootstrap_mode: str,
    feature_set: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_train: int,
    n_valid: int,
    notes: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "timestamp_utc": run_id,
        "mode": mode,
        "split_name": split_name,
        "model_name": model_name,
        "target_mode": target_mode,
        "profile": profile,
        "bootstrap_mode": bootstrap_mode,
        "feature_set": feature_set,
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 6) if y_true.size else "",
        "rmse": round(rmse(y_true, y_pred), 6) if y_true.size else "",
        "n_train": n_train,
        "n_valid": n_valid,
        "notes": notes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--experiment-dir", type=Path, default=Path(__file__).resolve().parent / "experiments")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "submissions")
    parser.add_argument("--mode", choices=["cv", "predict"], default="cv")
    parser.add_argument("--target-mode", choices=["raw", "log1p"], default="raw")
    parser.add_argument(
        "--profile",
        choices=sorted(CATBOOST_PROFILE_PRESETS.keys()),
        default="baseline_v1",
    )
    parser.add_argument(
        "--bootstrap-mode",
        choices=sorted(BOOTSTRAP_PRESETS.keys()),
        default="bernoulli_v1",
    )
    parser.add_argument(
        "--feature-set",
        choices=[
            "full",
            "cat_heavy",
            "slim",
            "ranked_s1_core",
            "ranked_s2_support",
            "ranked_s3_temporal",
            "ranked_s3_temporal_plus",
            "ranked_s4_bucket",
            "ranked_s5_cross",
        ],
        default="full",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--notes", type=str, default="catboost categorical-first experiment")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = utc_now_string()
    params = build_params(args.profile, args.bootstrap_mode)
    train, test, layout, sample_submission = load_bundle(args.data_dir)
    train_df, feature_columns, categorical_columns = prepare_catboost_frame(train, layout)
    test_df, _, _ = prepare_catboost_frame(test, layout)
    selected_feature_columns = select_feature_set(feature_columns, categorical_columns, args.feature_set)
    selected_categorical_columns = [col for col in categorical_columns if col in selected_feature_columns]

    split_masks = {
        "scenario_holdout": split_by_group(train_df, "scenario_id", seed=args.seed),
        "layout_holdout": split_by_group(train_df, "layout_id", seed=args.seed),
    }

    rows: list[dict[str, Any]] = []
    for split_name, mask in split_masks.items():
        fold_train = train_df.loc[~mask].copy()
        fold_valid = train_df.loc[mask].copy()
        y_train = fold_train[TARGET].to_numpy()
        y_valid = fold_valid[TARGET].to_numpy()
        pred = fit_predict(
            fold_train,
            y_train,
            fold_valid,
            y_valid,
            selected_feature_columns,
            selected_categorical_columns,
            params,
            args.target_mode,
        )
        row = build_metric_row(
            run_id,
            args.mode,
            split_name,
            "catboost_categorical",
            args.target_mode,
            args.profile,
            args.bootstrap_mode,
            args.feature_set,
            y_valid,
            pred,
            len(fold_train),
            len(fold_valid),
            args.notes,
        )
        rows.append(row)
        print(split_name, "catboost_categorical", {"mae": row["mae"], "rmse": row["rmse"]}, flush=True)

    output_dir = args.output_dir / f"catboost_categorical_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "predict":
        pred = fit_predict(
            train_df,
            train_df[TARGET].to_numpy(),
            test_df,
            None,
            selected_feature_columns,
            selected_categorical_columns,
            params,
            args.target_mode,
        )
        submission = sample_submission.copy()
        submission[TARGET] = pred
        path = output_dir / f"submission_catboost_categorical_{args.target_mode}.csv"
        submission.to_csv(path, index=False)
        print(f"saved {path}", flush=True)

    summary = {
        "run_id": run_id,
        "timestamp_utc": run_id,
        "mode": args.mode,
        "target_mode": args.target_mode,
        "profile": args.profile,
        "bootstrap_mode": args.bootstrap_mode,
        "feature_set": args.feature_set,
        "feature_columns": selected_feature_columns,
        "categorical_columns": selected_categorical_columns,
        "feature_count": len(selected_feature_columns),
        "categorical_feature_count": len(selected_categorical_columns),
        "catboost_params": params,
        "results": rows,
        "notes": args.notes,
    }
    yaml_path = args.experiment_dir / "runs" / f"{run_id}_{args.mode}_catboost_categorical.yaml"
    write_yaml_summary(yaml_path, summary)
    write_yaml_summary(args.experiment_dir / "current_catboost_categorical.yaml", summary)
    append_csv_log(args.experiment_dir / "catboost_categorical_log.csv", rows)


if __name__ == "__main__":
    main()
