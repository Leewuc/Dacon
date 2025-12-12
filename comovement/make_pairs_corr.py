import argparse
import numpy as np
import pandas as pd


def build_time_index(monthly: pd.DataFrame):
    """year, month 기반으로 전체 타임라인 인덱스 생성"""
    df = monthly[["year", "month"]].drop_duplicates().copy()
    df = df.sort_values(["year", "month"])
    time_index = list(zip(df["year"], df["month"]))
    ym2idx = {ym: i for i, ym in enumerate(time_index)}
    return time_index, ym2idx


def build_values_matrix(monthly: pd.DataFrame, time_index, ym2idx):
    """
    item_id 별 전체 타임라인에 맞는 value 시계열 생성

    반환:
      - values_matrix: (T, n_items)
      - item_ids: 열 순서와 대응되는 item_id 리스트
    """
    T = len(time_index)
    item_ids = sorted(monthly["item_id"].astype(str).unique().tolist())
    n_items = len(item_ids)

    values_matrix = np.zeros((T, n_items), dtype=np.float64)

    for j, item in enumerate(item_ids):
        sub = monthly[monthly["item_id"].astype(str) == item]
        for _, r in sub.iterrows():
            ym = (int(r["year"]), int(r["month"]))
            idx = ym2idx[ym]
            values_matrix[idx, j] = float(r["value"])

    return values_matrix, item_ids


def compute_best_lag_corr(sA: np.ndarray, sB: np.ndarray, max_lag: int = 6, min_overlap: int = 6):
    """
    A 선행, B 후행이라고 가정하고 lag ∈ [1, max_lag] 범위에서
    corr(A_{t-lag}, B_t)를 계산.
    - sA, sB: 같은 길이 T 의 numpy array
    - min_overlap: corr 계산에 사용할 최소 겹치는 시점 개수

    반환:
      best_lag, max_corr, n_valid
      (유효한 corr가 하나도 없으면 best_lag=0, max_corr=0.0, n_valid=0)
    """
    T = len(sA)
    best_lag = 0
    best_corr = 0.0
    best_abs_corr = 0.0
    best_n = 0

    for lag in range(1, max_lag + 1):
        if T - lag < min_overlap:
            # 겹치는 구간이 너무 짧으면 skip
            continue

        # A_{t-lag}, B_t → A를 과거로 shift
        a = sA[: T - lag]
        b = sB[lag:]

        # 둘 다 variance가 거의 0이면 corr 의미 없음
        if np.allclose(a, a.mean()) or np.allclose(b, b.mean()):
            continue

        c = np.corrcoef(a, b)[0, 1]
        if np.isnan(c):
            continue

        abs_c = abs(c)
        if abs_c > best_abs_corr:
            best_abs_corr = abs_c
            best_corr = c
            best_lag = lag
            best_n = len(a)

    return int(best_lag), float(best_corr), int(best_n)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--monthly", type=str, required=True,
                        help="monthly_agg.csv 경로 (year, month, item_id, value 포함)")
    parser.add_argument("--out_csv", type=str, required=True,
                        help="출력 pairs_corr.csv 경로")
    parser.add_argument("--max_lag", type=int, default=6,
                        help="A 선행 lag 최대 값 (1~max_lag)")
    parser.add_argument("--min_overlap", type=int, default=6,
                        help="corr 계산에 사용할 최소 겹치는 시점 수")

    args = parser.parse_args()

    print("▶ monthly_agg.csv 로드 중...")
    monthly = pd.read_csv(args.monthly)
    print("  - shape:", monthly.shape)

    monthly["item_id"] = monthly["item_id"].astype(str)

    print("▶ 타임라인 및 values_matrix 구성 중...")
    time_index, ym2idx = build_time_index(monthly)
    values_matrix, item_ids = build_values_matrix(monthly, time_index, ym2idx)
    T, n_items = values_matrix.shape
    print(f"  - time steps: {T}, items: {n_items}")

    item2col = {item_id: j for j, item_id in enumerate(item_ids)}

    rows = []

    print("▶ 모든 (A,B) 쌍에 대해 best_lag, max_corr 계산 중...")
    for i, A in enumerate(item_ids):
        sA = values_matrix[:, i]
        for j, B in enumerate(item_ids):
            if A == B:
                continue  # 자기 자신은 스킵

            sB = values_matrix[:, j]

            best_lag, max_corr, n_valid = compute_best_lag_corr(
                sA, sB,
                max_lag=args.max_lag,
                min_overlap=args.min_overlap
            )

            rows.append({
                "leading_item_id": A,
                "following_item_id": B,
                "best_lag": best_lag,
                "max_corr": max_corr,
                "corr_abs": abs(max_corr),
                "n_valid": n_valid,
            })

    pairs_corr = pd.DataFrame(rows)
    print("▶ 계산 완료. 예시 상위 10개:")
    print(pairs_corr.head(10))

    pairs_corr.to_csv(args.out_csv, index=False)
    print(f"▶ pairs_corr 저장 완료: {args.out_csv}")
    print("  - shape:", pairs_corr.shape)


if __name__ == "__main__":
    main()
