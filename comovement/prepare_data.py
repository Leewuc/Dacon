import pandas as pd
import numpy as np
from itertools import permutations

TRAIN_PATH = "train.csv"
N_TOP_PAIRS = 500
MAX_LAG = 6

OUT_MONTHLY_PATH = "monthly_agg.csv"
OUT_PAIRS_PATH = f"candidate_pairs_top{N_TOP_PAIRS}.csv"
OUT_REGDATA_PATH = f"regression_dataset_top{N_TOP_PAIRS}.csv"

def add_ym_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ym"] = df["year"] * 100 + df["month"]
    return df

def aggregate_monthly(train: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        train.groupby(["item_id", "year", "month"], as_index=False)
        .agg({
            "value": "sum",
            "weight": "sum",
            "quantity": "sum"
        })
    )

    item_info = (
        train.groupby("item_id", as_index=False)
        .agg({
            "type": "first",
            "hs4": "first"
        })
    )

    monthly = monthly.merge(item_info, on="item_id", how="left")
    monthly = add_ym_column(monthly)
    return monthly

def compute_lag1_correlation_pairs(monthly: pd.DataFrame) -> pd.DataFrame:
    pivot_val = monthly.pivot_table(index="ym", columns="item_id", values="value", aggfunc="sum").sort_index()
    pivot_val = pivot_val.fillna(0.0)
    items = pivot_val.columns.tolist()
    rows = []
    pivot_log = np.log1p(pivot_val)

    for a, b in permutations(items, 2):
        series_a = pivot_log[a].values.astype(float)
        series_b = pivot_log[b].values.astype(float)

        a_curr = series_a[:-1]
        b_next = series_b[1:]

        if np.std(a_curr) == 0 or np.std(b_next) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(a_curr, b_next)[0, 1]
        
        rows.append({
            "leading_item_id": a,
            "following_item_id": b,
            "lag": 1,
            "corr": corr
        })
    
    pair_scores = pd.DataFrame(rows)
    return pair_scores

def add_value_lags(monthly: pd.DataFrame, max_lag: int=6) -> pd.DataFrame:
    df = monthly.sort_values(["item_id", "year", "month"]).copy()
    for l in range(1, max_lag + 1):
        df[f"value_lag{l}"] = (
            df.groupby("item_id")["value"]
            .shift(1)
        )
    
    df["target_next_value"] = (
        df.groupby("item_id")["value"]
        .shift(-1)
    )

    return df

def build_regression_dataset(
        monthly_lagged: pd.DataFrame,
        candidate_pairs: pd.DataFrame,
        max_lag: int = 6,
) -> pd.DataFrame:
    rows = []
    base_cols = ["item_id", "year", "month", "ym", "type", "hs4", "target_next_value"]
    lag_cols = [f"value_lag{l}" for l in range(1, max_lag + 1)]
    use_cols = base_cols + lag_cols
    df_all = monthly_lagged[use_cols].copy()

    for _, row in candidate_pairs.iterrows():
        A = row["leading_item_id"]
        B = row["following_item_id"]

        df_B = df_all[df_all["item_id"] == B].copy()
        df_A = df_all[df_all["item_id"] == A].copy()

        b_cols = ["ym", "year", "month", "type", "hs4", "target_next_value"] + lag_cols
        df_B = df_B[b_cols]

        df_B = df_B.rename(columns={
            "type": "B_type",
            "hs4": "B_hs4",
            "target_next_value": "target",
            **{f"value_lag{l}": f"B_value_lag{l}" for l in range(1, max_lag + 1)}
        })

        a_cols = ["ym", "type", "hs4"] + lag_cols
        df_A = df_A[a_cols]
        df_A = df_A.rename(columns={
            "type": "A_type",
            "hs4": "A_hs4",
            **{f"value_lag{l}": f"A_value_lag{l}" for l in range(1, max_lag + 1)}
        })

        df_pair = df_B.merge(df_A, on="ym", how="inner")
        df_pair = df_pair[~df_pair["target"].isna()].copy()

        required_cols = ["target"] + [f"B_value_lag{l}" for l in range(1, max_lag + 1)] + [f"A_value_lag{l}" for l in range(1, max_lag + 1)]

        df_pair = df_pair.dropna(subset = required_cols)

        if df_pair.empty:
            continue

        df_pair["leading_item_id"] = A
        df_pair["following_item_id"] = B
        rows.append(df_pair)
    
    if not rows:
        print("후보쌍으로부터 생성된 회귀 데이터가 없습니다.")
        return pd.DataFrame()
    
    reg_df = pd.concat(rows, ignore_index=True)
    return reg_df

def main():
    # 1) train.csv 읽기
    print("▶ train.csv 읽는 중...")
    train = pd.read_csv(TRAIN_PATH)

    # 2) 월별 집계
    print("▶ 월별 집계 만드는 중...")
    monthly = aggregate_monthly(train)
    monthly.to_csv(OUT_MONTHLY_PATH, index=False)
    print(f"  - monthly 저장: {OUT_MONTHLY_PATH} (shape={monthly.shape})")

    # 3) lag-1 corr로 공행성 점수 계산
    print("▶ lag-1 cross-correlation 계산 중 (모든 (A,B) 쌍)...")
    pair_scores = compute_lag1_correlation_pairs(monthly)

    # corr 기준으로 상위 N개 후보쌍 선택 (양의 corr만 사용)
    print(f"▶ 공행성 상위 {N_TOP_PAIRS}개 후보쌍 선택 중...")
    candidate_pairs = (
        pair_scores
        .query("corr > 0")
        .sort_values("corr", ascending=False)
        .head(N_TOP_PAIRS)
        .reset_index(drop=True)
    )
    candidate_pairs.to_csv(OUT_PAIRS_PATH, index=False)
    print(f"  - 후보쌍 저장: {OUT_PAIRS_PATH} (shape={candidate_pairs.shape})")

    # 4) 회귀용 데이터셋 만들기 (lag feature + target)
    print("▶ 회귀용 lag feature 및 target 준비 중...")
    monthly_lagged = add_value_lags(monthly, max_lag=MAX_LAG)

    print("▶ 후보쌍 기반 회귀 데이터셋 생성 중...")
    reg_df = build_regression_dataset(
        monthly_lagged,
        candidate_pairs,
        max_lag=MAX_LAG
    )

    reg_df.to_csv(OUT_REGDATA_PATH, index=False)
    print(f"  - 회귀 데이터셋 저장: {OUT_REGDATA_PATH} (shape={reg_df.shape})")

    print("✅ 준비 완료! 이제 다른 파일에서 이 CSV들을 읽어서 모델 학습하면 됩니다.")


if __name__ == "__main__":
    main()
