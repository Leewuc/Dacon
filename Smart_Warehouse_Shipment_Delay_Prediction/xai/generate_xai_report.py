from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from compare_models import CATBOOST_PARAMS, LIGHTGBM_PARAMS, build_feature_frame, load_bundle, split_by_group


TARGET = "avg_delay_minutes_next_30m"


def utc_now_string() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def build_lightgbm() -> LGBMRegressor:
    return LGBMRegressor(**LIGHTGBM_PARAMS)


def build_catboost() -> CatBoostRegressor:
    return CatBoostRegressor(**CATBOOST_PARAMS)


def feature_family(name: str) -> str:
    lowered = name.lower()
    if any(token in lowered for token in ["battery", "charge", "charging"]):
        return "battery_and_charging"
    if any(token in lowered for token in ["congestion", "traffic", "intersection", "blocked", "collision"]):
        return "traffic_and_congestion"
    if any(token in lowered for token in ["order", "sku", "pick", "wave", "bulk", "package", "staging"]):
        return "order_and_pick_workload"
    if any(token in lowered for token in ["robot", "agv", "idle", "fleet", "firmware", "calibration"]):
        return "robot_operations"
    if any(token in lowered for token in ["layout", "aisle", "floor_area", "pack_station", "charger_count", "one_way"]):
        return "layout_structure"
    if any(token in lowered for token in ["dock", "truck", "conveyor", "forklift", "cross_dock"]):
        return "dock_and_material_flow"
    if any(token in lowered for token in ["temp", "humidity", "air_quality", "co2", "hvac", "noise", "vibration"]):
        return "environment"
    if any(token in lowered for token in ["latency", "wifi", "wms", "scanner", "barcode", "network"]):
        return "systems_and_it"
    if any(token in lowered for token in ["staff", "worker", "safety", "handover"]):
        return "labor_and_shift"
    if any(token in lowered for token in ["time_idx", "time_frac", "shift_hour", "day_of_week"]):
        return "time_context"
    if lowered.endswith("_miss"):
        return "missingness"
    if any(token in lowered for token in ["lag", "diff", "roll", "exp_"]):
        return "sequence_history"
    return "other"


def write_markdown_summary(
    output_path: Path,
    model_name: str,
    metrics: dict[str, float],
    top_gain: pd.DataFrame,
    top_perm: pd.DataFrame,
    family_df: pd.DataFrame,
) -> None:
    lines = [
        f"# XAI Insight Summary: {model_name}",
        "",
        "## Validation",
        f"- split: `{metrics['split_name']}`",
        f"- mae: `{metrics['mae']:.6f}`",
        f"- rmse: `{metrics['rmse']:.6f}`",
        f"- n_train: `{int(metrics['n_train'])}`",
        f"- n_valid: `{int(metrics['n_valid'])}`",
        "",
        "## Top Gain Features",
    ]
    for _, row in top_gain.head(15).iterrows():
        lines.append(f"- `{row['feature']}`: gain `{row['gain_importance']:.2f}`")
    lines.extend(["", "## Top Permutation Features"])
    for _, row in top_perm.head(15).iterrows():
        lines.append(
            f"- `{row['feature']}`: mean drop `{row['importance_mean']:.6f}` +/- `{row['importance_std']:.6f}`"
        )
    lines.extend(["", "## Family-Level Importance"])
    for _, row in family_df.head(10).iterrows():
        lines.append(f"- `{row['family']}`: total gain `{row['gain_importance']:.2f}`")
    lines.extend(
        [
            "",
            "## Modeling Direction",
            "- `battery_and_charging`, `traffic_and_congestion`, `order_and_pick_workload`, `sequence_history` 패밀리가 상위면 lag/rolling 강화가 계속 유효하다는 의미다.",
            "- `layout_structure` 패밀리 비중이 높으면 unseen layout 일반화가 중요하므로 layout cluster, layout target encoding, ratio-based normalization을 더 강화할 가치가 있다.",
            "- `systems_and_it`나 `missingness` 비중이 높으면 단순 값 자체보다 운영 이상 신호를 모델이 읽고 있다는 뜻이므로 missing indicator와 interaction을 유지하는 편이 낫다.",
            "- permutation importance에서 top feature가 gain importance와 다르면, split을 많이 탄 feature보다 실제 성능 기여 feature를 우선해서 정제해야 한다.",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def permutation_importance_manual(
    model: object,
    x_valid: pd.DataFrame,
    y_valid: np.ndarray,
    features: list[str],
    n_repeats: int = 5,
) -> pd.DataFrame:
    baseline_pred = np.clip(model.predict(x_valid), 0, None)
    baseline_mae = float(mean_absolute_error(y_valid, baseline_pred))
    rows = []
    rng = np.random.default_rng(42)
    for feature in features:
        drops = []
        for _ in range(n_repeats):
            shuffled = x_valid.copy()
            shuffled_values = shuffled[feature].to_numpy(copy=True)
            rng.shuffle(shuffled_values)
            shuffled[feature] = shuffled_values
            shuffled_pred = np.clip(model.predict(shuffled), 0, None)
            shuffled_mae = float(mean_absolute_error(y_valid, shuffled_pred))
            drops.append(shuffled_mae - baseline_mae)
        rows.append(
            {
                "feature": feature,
                "importance_mean": float(np.mean(drops)),
                "importance_std": float(np.std(drops)),
            }
        )
    return pd.DataFrame(rows).sort_values("importance_mean", ascending=False)


def build_feature_importance_df(model_name: str, model: object, feature_columns: list[str]) -> pd.DataFrame:
    if model_name == "lightgbm":
        importance_df = pd.DataFrame(
            {
                "feature": feature_columns,
                "gain_importance": model.booster_.feature_importance(importance_type="gain"),
                "split_importance": model.booster_.feature_importance(importance_type="split"),
            }
        )
    else:
        importance_df = pd.DataFrame(
            {
                "feature": feature_columns,
                "gain_importance": model.get_feature_importance(),
                "split_importance": np.nan,
            }
        )
    importance_df["family"] = importance_df["feature"].map(feature_family)
    return importance_df.sort_values("gain_importance", ascending=False)


def write_model_artifacts(
    model_name: str,
    model: object,
    x_valid: pd.DataFrame,
    y_valid: np.ndarray,
    feature_columns: list[str],
    metrics: dict[str, float],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    gain_importance = build_feature_importance_df(model_name, model, feature_columns)
    gain_importance.to_csv(output_dir / "feature_importance_gain.csv", index=False)
    gain_importance.sort_values("split_importance", ascending=False, na_position="last").to_csv(
        output_dir / "feature_importance_split.csv", index=False
    )

    family_df = (
        gain_importance.groupby("family", as_index=False)["gain_importance"]
        .sum()
        .sort_values("gain_importance", ascending=False)
    )
    family_df.to_csv(output_dir / "family_importance.csv", index=False)

    top_features = gain_importance.head(25)["feature"].tolist()
    valid_sample_n = min(8000, len(x_valid))
    valid_sample = x_valid.sample(n=valid_sample_n, random_state=42)
    valid_positions = x_valid.index.get_indexer(valid_sample.index)
    valid_sample_y = y_valid[valid_positions]

    perm_df = permutation_importance_manual(
        model=model,
        x_valid=valid_sample,
        y_valid=valid_sample_y,
        features=top_features,
        n_repeats=5,
    )
    perm_df.to_csv(output_dir / "permutation_importance.csv", index=False)

    metadata = {
        "run_id": metrics["run_id"],
        "split_name": metrics["split_name"],
        "metrics": metrics,
        "model_name": model_name,
        "model_params": LIGHTGBM_PARAMS if model_name == "lightgbm" else CATBOOST_PARAMS,
        "top_gain_features": gain_importance.head(20).to_dict(orient="records"),
        "top_permutation_features": perm_df.head(20).to_dict(orient="records"),
    }
    with (output_dir / "run_metadata.yaml").open("w", encoding="utf-8") as fp:
        yaml.safe_dump(metadata, fp, allow_unicode=True, sort_keys=False)

    write_markdown_summary(
        output_dir / "insight_summary.md",
        model_name=model_name,
        metrics=metrics,
        top_gain=gain_importance,
        top_perm=perm_df,
        family_df=family_df,
    )


def write_comparison_summary(xai_dir: Path) -> None:
    lgbm_gain = pd.read_csv(xai_dir / "lightgbm" / "feature_importance_gain.csv")
    cat_gain = pd.read_csv(xai_dir / "catboost" / "feature_importance_gain.csv")
    lgbm_perm = pd.read_csv(xai_dir / "lightgbm" / "permutation_importance.csv")
    cat_perm = pd.read_csv(xai_dir / "catboost" / "permutation_importance.csv")

    lines = [
        "# Model XAI Comparison",
        "",
        "## Top Gain Features",
        "### LightGBM",
    ]
    for _, row in lgbm_gain.head(10).iterrows():
        lines.append(f"- `{row['feature']}`: `{row['gain_importance']:.4f}`")
    lines.extend(["", "### CatBoost"])
    for _, row in cat_gain.head(10).iterrows():
        lines.append(f"- `{row['feature']}`: `{row['gain_importance']:.4f}`")

    lines.extend(["", "## Top Permutation Features", "### LightGBM"])
    for _, row in lgbm_perm.head(10).iterrows():
        lines.append(f"- `{row['feature']}`: `{row['importance_mean']:.6f}`")
    lines.extend(["", "### CatBoost"])
    for _, row in cat_perm.head(10).iterrows():
        lines.append(f"- `{row['feature']}`: `{row['importance_mean']:.6f}`")

    common = set(lgbm_perm.head(15)["feature"]) & set(cat_perm.head(15)["feature"])
    lines.extend(["", "## Common High-Value Features"])
    for feature in sorted(common):
        lines.append(f"- `{feature}`")

    (xai_dir / "model_compare_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    base_dir = Path("/data/AskFake/RAG/warehouse_shipment")
    xai_dir = base_dir / "xai"
    xai_dir.mkdir(parents=True, exist_ok=True)
    run_id = utc_now_string()

    bundle = load_bundle(base_dir, None, None)
    train_features, feature_columns = build_feature_frame(bundle.train, bundle.layout)
    x = train_features[feature_columns]
    y = train_features[TARGET].to_numpy()

    mask = split_by_group(train_features, "layout_id")
    split_name = "layout_holdout"
    x_train = x.loc[~mask]
    y_train = y[~mask]
    x_valid = x.loc[mask]
    y_valid = y[mask]

    lightgbm_model = build_lightgbm()
    lightgbm_model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        eval_metric="mae",
        callbacks=[early_stopping(100), log_evaluation(0)],
    )
    lightgbm_pred = np.clip(lightgbm_model.predict(x_valid), 0, None)
    lightgbm_metrics = {
        "run_id": run_id,
        "split_name": split_name,
        "mae": float(mean_absolute_error(y_valid, lightgbm_pred)),
        "rmse": rmse(y_valid, lightgbm_pred),
        "n_train": int(len(x_train)),
        "n_valid": int(len(x_valid)),
        "model_name": "LightGBM",
        "primary_metric": "mae",
    }
    write_model_artifacts(
        model_name="lightgbm",
        model=lightgbm_model,
        x_valid=x_valid,
        y_valid=y_valid,
        feature_columns=feature_columns,
        metrics=lightgbm_metrics,
        output_dir=xai_dir / "lightgbm",
    )

    catboost_model = build_catboost()
    catboost_model.fit(
        x_train,
        y_train,
        eval_set=(x_valid, y_valid),
        use_best_model=True,
    )
    catboost_pred = np.clip(catboost_model.predict(x_valid), 0, None)
    catboost_metrics = {
        "run_id": run_id,
        "split_name": split_name,
        "mae": float(mean_absolute_error(y_valid, catboost_pred)),
        "rmse": rmse(y_valid, catboost_pred),
        "n_train": int(len(x_train)),
        "n_valid": int(len(x_valid)),
        "model_name": "CatBoost",
        "primary_metric": "mae",
    }
    write_model_artifacts(
        model_name="catboost",
        model=catboost_model,
        x_valid=x_valid,
        y_valid=y_valid,
        feature_columns=feature_columns,
        metrics=catboost_metrics,
        output_dir=xai_dir / "catboost",
    )
    write_comparison_summary(xai_dir)
    print(f"saved xai artifacts to {xai_dir}")


if __name__ == "__main__":
    main()
