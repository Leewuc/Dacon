"""
Queuing Theory Feature Engineering (v11).

핵심 원리:
  창고는 M/M/c 대기행렬 시스템이다.
  ρ (이용률) → 대기시간 관계가 비선형이다:

    M/M/1: Lq = ρ² / (1 - ρ)   ← ρ=0.8이면 3.2, ρ=0.9이면 8.1, ρ=0.95이면 18!

  현재 모델은 ρ를 선형으로 사용 → 이 폭발적 증가를 못 잡음.
  v11에서는 이 비선형 변환을 직접 피처로 추가.

Sub-system별 분석:
  1. Packing stations (pack_utilization)
  2. Robots (robot_utilization, active_ratio)
  3. Loading dock (loading_dock_util)
  4. Charger (charge_queue_length / charger_count)
  5. WMS system (wms_response_time_ms — 높을수록 병목)

추가 피처:
  - q_{sys}: M/M/1 queue length proxy = ρ²/(1-ρ)
  - w_{sys}: M/M/1 wait time proxy = q/λ  (Little's Law)
  - near_sat_{sys}: ρ/(1-ρ)  — saturation 근접 강도
  - bottleneck_fraction: 어떤 시스템이 전체 병목의 몇 %인지
  - multi_bottleneck: 2개 이상 시스템이 동시에 포화 상태인지
  - queue_surge: 현재 q vs 시나리오 평균 q (급격한 상승 감지)
  - q_exp_mean / q_exp_max: 시나리오 내 누적 대기 압력

이 피처들을 features_v9_full 위에 추가하여 features_v11_full 생성.
(v10보다 v9 베이스 사용 - LB에서 v9가 더 좋았음)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from compare_models import default_feature_cache_dir, load_feature_cache, write_yaml_summary


# ── 서브시스템 이용률 컬럼 ──────────────────────────────────
UTILIZATION_SYSTEMS = {
    "pack": "pack_utilization",
    "robot": "robot_utilization",
    "dock": "loading_dock_util",
    "staging": "staging_area_util",
}

EPS = 1e-3  # 0으로 나누기 방지, ρ=1 폭발 방지


def mm1_queue_length(rho: np.ndarray) -> np.ndarray:
    """M/M/1 평균 대기열 길이: Lq = ρ² / (1-ρ)"""
    rho_c = np.clip(rho, 0.0, 1.0 - EPS)
    return rho_c ** 2 / (1.0 - rho_c)


def mm1_near_saturation(rho: np.ndarray) -> np.ndarray:
    """포화 근접 강도: ρ/(1-ρ)  (ρ=0.9 → 9, ρ=0.95 → 19)"""
    rho_c = np.clip(rho, 0.0, 1.0 - EPS)
    return rho_c / (1.0 - rho_c)


def add_v11_queuing_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    new_cols: dict[str, np.ndarray | pd.Series] = {}
    grp = df.groupby("scenario_id", sort=False)

    # ── 1. 각 서브시스템별 M/M/1 피처 ──────────────────────
    q_arrays: dict[str, np.ndarray] = {}
    for sys_name, col in UTILIZATION_SYSTEMS.items():
        if col not in df.columns:
            continue
        rho = df[col].fillna(0.0).to_numpy()
        q = mm1_queue_length(rho)
        sat = mm1_near_saturation(rho)

        new_cols[f"q_{sys_name}"] = q.astype("float32")
        new_cols[f"sat_{sys_name}"] = sat.astype("float32")

        # 이용률 > 임계값 플래그
        new_cols[f"near_sat_flag_{sys_name}"] = (rho > 0.85).astype("float32")
        new_cols[f"critical_flag_{sys_name}"] = (rho > 0.92).astype("float32")

        # Little's Law: W = Lq / λ  (λ = order_inflow_15m)
        if "order_inflow_15m" in df.columns:
            lam = df["order_inflow_15m"].fillna(0.0).to_numpy().clip(1.0)
            new_cols[f"wait_proxy_{sys_name}"] = (q / lam).astype("float32")

        q_arrays[sys_name] = q

    # ── 2. Charger 대기열 (M/M/c 근사) ─────────────────────
    if "charge_queue_length" in df.columns and "charger_count" in df.columns:
        cq = df["charge_queue_length"].fillna(0.0).to_numpy()
        cc = df["charger_count"].fillna(1.0).to_numpy().clip(1.0)
        charge_rho_eff = cq / cc  # 충전기당 평균 대기
        new_cols["charge_rho_eff"] = charge_rho_eff.astype("float32")
        new_cols["q_charge"] = mm1_queue_length(
            np.clip(charge_rho_eff / (charge_rho_eff + 1.0), 0, 1 - EPS)
        ).astype("float32")
        q_arrays["charge"] = new_cols["q_charge"]

    # ── 3. WMS 시스템 부하 (ms → 정규화) ────────────────────
    if "wms_response_time_ms" in df.columns:
        wms = df["wms_response_time_ms"].fillna(0.0).to_numpy()
        # 정규화: 100ms = 정상, 500ms = 심각
        wms_load = np.clip(wms / 500.0, 0.0, 1.0 - EPS)
        new_cols["wms_load_rho"] = wms_load.astype("float32")
        new_cols["q_wms"] = mm1_queue_length(wms_load).astype("float32")
        q_arrays["wms"] = new_cols["q_wms"]

    # ── 4. 복합 병목 피처 ────────────────────────────────────
    # 전체 시스템 대기 압력 합산
    all_q = list(q_arrays.values())
    if all_q:
        total_q = sum(all_q)
        new_cols["total_queue_pressure"] = total_q.astype("float32")

        # 병목 분율 (어느 서브시스템이 얼마나 기여하는지)
        denom = total_q + EPS
        for sys_name, q in q_arrays.items():
            new_cols[f"bottleneck_frac_{sys_name}"] = (q / denom).astype("float32")

        # 동시 포화: 2개 이상 서브시스템이 ρ > 0.85 인 카운트
        near_sat_flags = [
            v for k, v in new_cols.items() if k.startswith("near_sat_flag_")
        ]
        if near_sat_flags:
            multi_bottleneck = sum(near_sat_flags)
            new_cols["multi_bottleneck_count"] = multi_bottleneck.astype("float32")
            new_cols["multi_bottleneck_flag"] = (multi_bottleneck >= 2).astype("float32")

    # ── 5. Pack × Robot 상호작용 (두 핵심 시스템 동시 포화) ──
    if "q_pack" in new_cols and "q_robot" in new_cols:
        new_cols["q_pack_robot_joint"] = (
            new_cols["q_pack"] * new_cols["q_robot"]
        ).astype("float32")
        new_cols["q_pack_robot_max"] = np.maximum(
            new_cols["q_pack"], new_cols["q_robot"]
        ).astype("float32")

    # ── 6. 시나리오 내 누적 대기 압력 (expanding) ───────────
    for q_name in ["q_pack", "q_robot", "q_dock", "total_queue_pressure"]:
        if q_name not in new_cols:
            continue
        q_series = pd.Series(new_cols[q_name], index=df.index)
        shifted = q_series.groupby(df["scenario_id"], sort=False).shift(1)
        shifted_grp = shifted.groupby(df["scenario_id"], sort=False)

        exp_mean = shifted_grp.expanding().mean().reset_index(level=0, drop=True)
        exp_max = shifted_grp.expanding().max().reset_index(level=0, drop=True)

        new_cols[f"{q_name}_exp_mean"] = exp_mean.astype("float32")
        new_cols[f"{q_name}_exp_max"] = exp_max.astype("float32")

        # surge: 현재 - 평균  (급상승 감지)
        new_cols[f"{q_name}_surge"] = (
            new_cols[q_name] - exp_mean.to_numpy()
        ).astype("float32")

    # ── 7. Effective throughput gap ──────────────────────────
    # 실제로 처리되어야 할 것 vs 현재 처리 중인 것
    if "order_inflow_15m" in df.columns and "pack_utilization" in df.columns:
        lam = df["order_inflow_15m"].fillna(0.0).to_numpy()
        pack_util = df["pack_utilization"].fillna(0.0).to_numpy()
        pack_n = df["pack_station_count"].fillna(1.0).to_numpy().clip(1.0) \
            if "pack_station_count" in df.columns else np.ones(len(df))
        # 처리 용량 추정: pack_station_count * (1 - utilization)  ∝  여유 용량
        slack = pack_n * (1.0 - pack_util)
        new_cols["throughput_gap"] = (lam - slack).astype("float32")
        new_cols["throughput_gap_positive"] = np.maximum(
            new_cols["throughput_gap"], 0.0
        ).astype("float32")

    # ── 8. Jensen's inequality 피처 ─────────────────────────
    # E[q(ρ)] > q(E[ρ])  (볼록함수이므로)
    # 즉, 변동성이 있으면 평균 이용률로 예측한 것보다 실제 대기가 더 길다.
    for sys_name, col in UTILIZATION_SYSTEMS.items():
        if col not in df.columns:
            continue
        rho_series = df[col].fillna(0.0)
        shifted = rho_series.groupby(df["scenario_id"], sort=False).shift(1)
        shifted_grp = shifted.groupby(df["scenario_id"], sort=False)
        rho_exp_mean = shifted_grp.expanding().mean().reset_index(level=0, drop=True)

        q_from_mean = mm1_queue_length(
            np.clip(rho_exp_mean.to_numpy(), 0, 1 - EPS)
        )
        q_instant = new_cols.get(f"q_{sys_name}", mm1_queue_length(rho_series.to_numpy()))
        # Jensen gap: q_instant - q_from_mean > 0 이면 현재 순간이 평균보다 나쁨
        new_cols[f"jensen_gap_{sys_name}"] = (
            q_instant - q_from_mean
        ).astype("float32")

    added_names = list(new_cols.keys())
    result = pd.concat(
        [df, pd.DataFrame(new_cols, index=df.index)], axis=1
    )
    return result, added_names


# ── CLI: v9 캐시 → v11 캐시 변환 ──────────────────────────────
def build_v11_cache(
    src_cache_name: str = "features_v9_full",
    dst_cache_name: str = "features_v11_full",
    base_dir: Path | None = None,
) -> None:
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent

    src_dir = default_feature_cache_dir(base_dir, src_cache_name)
    dst_dir = default_feature_cache_dir(base_dir, dst_cache_name)

    print(f"Loading {src_cache_name} ...", flush=True)
    cache = load_feature_cache(src_dir)
    print(
        f"  train={len(cache.train_features):,}  "
        f"test={len(cache.test_features):,}  "
        f"features={len(cache.feature_columns)}",
        flush=True,
    )

    print("Adding v11 queuing theory features to train ...", flush=True)
    train_v11, new_names = add_v11_queuing_features(cache.train_features)
    print(f"  Added {len(new_names)} features", flush=True)

    print("Adding v11 queuing theory features to test ...", flush=True)
    test_v11, _ = add_v11_queuing_features(cache.test_features)

    new_feature_columns = cache.feature_columns + [
        c for c in new_names if c not in cache.feature_columns
    ]

    print(f"Saving to {dst_dir} ...", flush=True)
    dst_dir.mkdir(parents=True, exist_ok=True)
    train_v11.to_parquet(dst_dir / "train_features.parquet", index=False)
    test_v11.to_parquet(dst_dir / "test_features.parquet", index=False)

    metadata = dict(cache.metadata)
    metadata["cache_name"] = dst_cache_name
    metadata["source_cache"] = src_cache_name
    metadata["feature_count"] = len(new_feature_columns)
    metadata["feature_columns"] = new_feature_columns
    metadata["v11_new_features"] = new_names
    metadata["notes"] = (
        "v11: queuing theory features — M/M/1 queue length, "
        "near-saturation, bottleneck fractions, Jensen gap, "
        "expanding queue pressure, throughput gap"
    )
    write_yaml_summary(dst_dir / "metadata.yaml", metadata)

    print(f"\nDone.")
    print(f"  v9 features  : {len(cache.feature_columns)}")
    print(f"  v11 features : {len(new_feature_columns)}")
    print(f"  New features : {len(new_names)}")
    print(f"  Saved to     : {dst_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-cache", default="features_v9_full")
    parser.add_argument("--dst-cache", default="features_v11_full")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    build_v11_cache(args.src_cache, args.dst_cache, args.data_dir)
