from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMRegressor, early_stopping, log_evaluation, reset_parameter
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error

from compare_models import (
    LIGHTGBM_PARAMS,
    TARGET,
    build_and_save_feature_cache,
    build_lgbm_lr_schedule,
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

CLUSTER_FEATURES = [
    "aisle_width_avg",
    "intersection_count",
    "one_way_ratio",
    "pack_station_count",
    "charger_count",
    "layout_compactness",
    "zone_dispersion",
    "robot_total",
    "building_age_years",
    "floor_area_sqm",
    "ceiling_height_m",
    "fire_sprinkler_count",
    "emergency_exit_count",
]

VARIANTS = {
    "global_only": {"mode": "global_only"},
    "layout_type_blend_w020": {"mode": "layout_type_blend", "expert_weight": 0.20},
    "layout_type_blend_w030": {"mode": "layout_type_blend", "expert_weight": 0.30},
    "layout_cluster_blend_w020": {"mode": "layout_cluster_blend", "expert_weight": 0.20},
    "layout_cluster_blend_w030": {"mode": "layout_cluster_blend", "expert_weight": 0.30},
}

SPLIT_WEIGHTS = {
    "scenario_holdout": 0.20,
    "layout_holdout": 0.20,
    "late_time_holdout": 0.15,
    "unseen_layout_heavy": 0.15,
    "congestion_tail_holdout": 0.30,
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
    parser.add_argument("--n-clusters", type=int, default=6)
    parser.add_argument(
        "--fixed-variant",
        type=str,
        default=None,
        choices=[*VARIANTS.keys()],
        help="Skip variant search and use a fixed specialist variant for prediction.",
    )
    parser.add_argument(
        "--base-tuning-yaml",
        type=Path,
        default=Path(__file__).resolve().parent / "experiments" / "runs" / "20260402T061536Z_tuning.yaml",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="layout specialist LGBM experiments on cv_pack_v2 splits",
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


def load_base_params(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    params = dict(LIGHTGBM_PARAMS)
    params.update(payload["best_lgbm_params"])
    return params


def build_model(base_params: dict[str, Any]) -> LGBMRegressor:
    params = dict(base_params)
    params["objective"] = "mae"
    params["metric"] = "mae"
    return LGBMRegressor(**params)


def weighted_split_score(split_metrics: dict[str, dict[str, float]]) -> float:
    return float(
        sum(SPLIT_WEIGHTS[name] * split_metrics[name]["mae"] for name in SPLIT_WEIGHTS if name in split_metrics)
    )


def fit_predict(x_train, y_train, x_valid, y_valid, base_params: dict[str, Any]) -> np.ndarray:
    model = build_model(base_params)
    lr_schedule = build_lgbm_lr_schedule(
        n_estimators=int(model.get_params()["n_estimators"]),
        base_lr=float(model.get_params()["learning_rate"]),
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        eval_metric="mae",
        callbacks=[reset_parameter(learning_rate=lr_schedule), early_stopping(80), log_evaluation(0)],
    )
    return np.clip(model.predict(x_valid), 0, None)


def fit_predict_noeval(x_train, y_train, x_test, base_params: dict[str, Any]) -> np.ndarray:
    model = build_model(base_params)
    lr_schedule = build_lgbm_lr_schedule(
        n_estimators=int(model.get_params()["n_estimators"]),
        base_lr=float(model.get_params()["learning_rate"]),
    )
    model.fit(x_train, y_train, callbacks=[reset_parameter(learning_rate=lr_schedule)])
    return np.clip(model.predict(x_test), 0, None)


def attach_layout_cluster(train_df: pd.DataFrame, test_df: pd.DataFrame, layout_df: pd.DataFrame, n_clusters: int, seed: int):
    layout_frame = layout_df[["layout_id", *CLUSTER_FEATURES]].copy()
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20)
    layout_frame["layout_cluster"] = kmeans.fit_predict(layout_frame[CLUSTER_FEATURES])
    cluster_map = layout_frame.set_index("layout_id")["layout_cluster"]
    train_out = train_df.copy()
    test_out = test_df.copy()
    train_out["layout_cluster"] = train_out["layout_id"].map(cluster_map).astype(int)
    test_out["layout_cluster"] = test_out["layout_id"].map(cluster_map).fillna(-1).astype(int)
    return train_out, test_out


def run_variant(train_df, x, y, split_masks, base_params, cfg):
    split_metrics: dict[str, dict[str, float]] = {}
    maes: list[float] = []
    for split_name, valid_mask in split_masks.items():
        train_mask = (~valid_mask).to_numpy()
        valid_mask_np = valid_mask.to_numpy()
        x_train = x.loc[train_mask]
        y_train = y[train_mask]
        x_valid = x.loc[valid_mask_np]
        y_valid = y[valid_mask_np]
        global_pred = fit_predict(x_train, y_train, x_valid, y_valid, base_params)
        pred = global_pred.copy()

        if cfg["mode"] != "global_only":
            if cfg["mode"] == "layout_type_blend":
                groups_train = train_df.loc[train_mask, "layout_type"].to_numpy()
                groups_valid = train_df.loc[valid_mask_np, "layout_type"].to_numpy()
            else:
                groups_train = train_df.loc[train_mask, "layout_cluster"].to_numpy()
                groups_valid = train_df.loc[valid_mask_np, "layout_cluster"].to_numpy()

            for group_value in pd.unique(groups_valid):
                stage_train_mask = groups_train == group_value
                stage_valid_mask = groups_valid == group_value
                if int(stage_train_mask.sum()) < 2000 or int(stage_valid_mask.sum()) == 0:
                    continue
                group_pred = fit_predict(
                    x_train.loc[stage_train_mask],
                    y_train[stage_train_mask],
                    x_valid.loc[stage_valid_mask],
                    y_valid[stage_valid_mask],
                    base_params,
                )
                w = float(cfg["expert_weight"])
                pred[stage_valid_mask] = np.clip((1.0 - w) * pred[stage_valid_mask] + w * group_pred, 0, None)

        mae = float(mean_absolute_error(y_valid, pred))
        split_metrics[split_name] = {
            "mae": mae,
            "rmse": rmse(y_valid, pred),
            "n_train": int(train_mask.sum()),
            "n_valid": int(valid_mask_np.sum()),
        }
        maes.append(mae)
    avg_mae = float(np.mean(maes))
    stress_score = weighted_split_score(split_metrics)
    return split_metrics, avg_mae, stress_score


def main() -> None:
    args = parse_args()
    run_id = utc_now_string()
    cache_dir = args.cache_dir or default_feature_cache_dir(args.data_dir, args.cache_name)
    bundle = load_bundle(args.data_dir, args.max_train_rows, args.max_test_rows)
    if cache_dir.exists():
        cache = load_feature_cache(cache_dir)
    else:
        cache = build_and_save_feature_cache(bundle, cache_dir, args.cache_name, args.notes)

    train_df, test_df = attach_layout_cluster(cache.train_features, cache.test_features, bundle.layout, args.n_clusters, args.seed)
    x = train_df[cache.feature_columns]
    y = train_df[TARGET].to_numpy()
    split_masks = build_split_masks_v2(train_df, seed=args.seed)
    base_params = load_base_params(args.base_tuning_yaml)

    rows: list[dict[str, Any]] = []
    variants_summary: dict[str, Any] = {}
    best_by_avg = args.fixed_variant
    best_avg_mae = float("inf")
    best_by_stress = args.fixed_variant
    best_stress_score = float("inf")

    variant_items = [(args.fixed_variant, VARIANTS[args.fixed_variant])] if args.fixed_variant else list(VARIANTS.items())
    for variant_name, cfg in variant_items:
        split_metrics, avg_mae, stress_score = run_variant(train_df, x, y, split_masks, base_params, cfg)
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
                f"[layout_specialist] variant={variant_name} split={split_name} "
                f"mae={metric['mae']:.6f} stress={stress_score:.6f}",
                flush=True,
            )

    summary = {
        "run_id": run_id,
        "timestamp_utc": run_id,
        "notes": args.notes,
        "cache_dir": str(cache_dir.resolve()),
        "base_tuning_yaml": str(args.base_tuning_yaml.resolve()),
        "n_clusters": args.n_clusters,
        "fixed_variant": args.fixed_variant,
        "cluster_features": CLUSTER_FEATURES,
        "best_by_avg": best_by_avg,
        "best_avg_mae": best_avg_mae,
        "best_by_stress": best_by_stress,
        "best_stress_score": best_stress_score,
        "variants": variants_summary,
    }

    if args.mode == "predict":
        best_cfg = VARIANTS[best_by_stress]
        x_test = test_df[cache.feature_columns]
        global_pred = fit_predict_noeval(x, y, x_test, base_params)
        pred = global_pred.copy()
        if best_cfg["mode"] != "global_only":
            if best_cfg["mode"] == "layout_type_blend":
                groups_train = train_df["layout_type"].to_numpy()
                groups_test = test_df["layout_type"].to_numpy()
            else:
                groups_train = train_df["layout_cluster"].to_numpy()
                groups_test = test_df["layout_cluster"].to_numpy()
            for group_value in pd.unique(groups_test):
                group_train_mask = groups_train == group_value
                group_test_mask = groups_test == group_value
                if int(group_train_mask.sum()) < 2000 or int(group_test_mask.sum()) == 0:
                    continue
                group_pred = fit_predict_noeval(x.loc[group_train_mask], y[group_train_mask], x_test.loc[group_test_mask], base_params)
                w = float(best_cfg["expert_weight"])
                pred[group_test_mask] = np.clip((1.0 - w) * pred[group_test_mask] + w * group_pred, 0, None)
        out_dir = args.output_dir / f"layout_specialist_{run_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        submission = bundle.sample_submission.copy()
        submission[TARGET] = pred
        submission_path = out_dir / f"submission_{best_by_stress}.csv"
        submission.to_csv(submission_path, index=False)
        summary["submission_path"] = str(submission_path.resolve())

    yaml_path = args.experiment_dir / "runs" / f"{run_id}_layout_specialist.yaml"
    write_yaml_summary(yaml_path, summary)
    write_yaml_summary(args.experiment_dir / "current_layout_specialist.yaml", summary)
    append_log(args.experiment_dir / "layout_specialist_log.csv", rows)


if __name__ == "__main__":
    main()
