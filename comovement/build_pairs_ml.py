"""
공행성 ML 기반 전체 pair score 생성기 (quantile pseudo label 버전)

1) monthly_agg.csv를 읽어서
2) 모든 (A,B) pair에 대해 feature 계산
3) 상관 기반 raw_score 계산
4) raw_score 분위수(quantile)로 pseudo label 생성
   - 상위 q_pos 이상 → label=1
   - 하위 q_neg 이하 → label=0
5) pseudo label(0/1)로 LightGBM 분류기 학습
6) 모든 pair에 대해 ml_score(공행성 확률) 추론
7) leading_item_id, following_item_id, ml_score 전체를 out_all에 저장

"""

import argparse
import itertools
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

eps = 1e-6


# ====================== 유틸 함수 ======================

def add_ym_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ym"] = df["year"] * 100 + df["month"]
    return df


def compute_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size != y.size or x.size == 0:
        return 0.0
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def compute_shifted_corr(x: np.ndarray, y: np.ndarray, lag: int) -> float:
    """
    lag > 0 : x(t-lag) vs y(t)
    예: lag=1 → x[:-1], y[1:]
    """
    n = len(x)
    if n <= lag:
        return 0.0
    if lag == 0:
        xs = x
        ys = y
    else:
        xs = x[:-lag]
        ys = y[lag:]
    return compute_corr(xs, ys)


# ====================== pair feature 생성 ======================

def build_pair_features(monthly: pd.DataFrame, max_lag: int = 3) -> pd.DataFrame:
    monthly = monthly.copy()
    monthly["item_id"] = monthly["item_id"].astype(str)

    monthly = add_ym_column(monthly)
    monthly = monthly.sort_values(["ym", "item_id"])

    pivot_val = monthly.pivot_table(
        index="ym", columns="item_id", values="value", aggfunc="sum"
    ).sort_index()
    pivot_val = pivot_val.fillna(0.0)

    ym_list = pivot_val.index.to_list()
    T = len(ym_list)

    pivot_log = np.log1p(pivot_val.values)  # (T, n_items)
    item_ids = list(pivot_val.columns.astype(str))
    n_items = len(item_ids)

    diff_log = np.diff(pivot_log, axis=0)
    diff_log = np.vstack([np.zeros((1, diff_log.shape[1])), diff_log])

    pct_log = np.zeros_like(pivot_log)
    pct_log[1:, :] = (pivot_log[1:, :] - pivot_log[:-1, :]) / (np.abs(pivot_log[:-1, :]) + eps)

    item_meta = (
        monthly.groupby("item_id", as_index=False)
        .agg({"type": "first", "hs4": "first"})
    )
    type_dict = dict(zip(item_meta["item_id"].astype(str), item_meta["type"]))
    hs4_dict = dict(zip(item_meta["item_id"].astype(str), item_meta["hs4"]))

    item2col = {item: idx for idx, item in enumerate(item_ids)}

    rows = []
    mid = T // 2

    total_pairs = n_items * (n_items - 1)
    print(f"▶ Pair feature 계산: items={n_items}, pairs={total_pairs}")

    for idx_pair, (a, b) in enumerate(itertools.permutations(item_ids, 2), start=1):
        col_a = item2col[a]
        col_b = item2col[b]

        seriesA_log = pivot_log[:, col_a]
        seriesB_log = pivot_log[:, col_b]

        seriesA_diff = diff_log[:, col_a]
        seriesB_diff = diff_log[:, col_b]

        seriesA_pct = pct_log[:, col_a]
        seriesB_pct = pct_log[:, col_b]

        feat = {
            "leading_item_id": a,
            "following_item_id": b,
        }

        # log값 lag 상관
        for lag in range(0, max_lag + 1):
            feat[f"corr_log_lag{lag}"] = compute_shifted_corr(seriesA_log, seriesB_log, lag)

        # diff 상관
        for lag in range(0, max_lag + 1):
            feat[f"corr_diff_lag{lag}"] = compute_shifted_corr(seriesA_diff, seriesB_diff, lag)

        # pct_change 상관
        feat["corr_pct"] = compute_corr(seriesA_pct[1:], seriesB_pct[1:])

        # 안정성: lag1 전체/앞/뒤
        if T > 4:
            corr_lag1_full = compute_shifted_corr(seriesA_log, seriesB_log, 1)

            seriesA_log_first = seriesA_log[:mid]
            seriesB_log_first = seriesB_log[:mid]
            corr_lag1_first = compute_shifted_corr(seriesA_log_first, seriesB_log_first, 1)

            seriesA_log_second = seriesA_log[mid:]
            seriesB_log_second = seriesB_log[mid:]
            corr_lag1_second = compute_shifted_corr(seriesA_log_second, seriesB_log_second, 1)

            feat["corr_lag1_full"] = corr_lag1_full
            feat["corr_lag1_first"] = corr_lag1_first
            feat["corr_lag1_second"] = corr_lag1_second
            feat["corr_lag1_stab"] = abs(corr_lag1_first - corr_lag1_second)
        else:
            feat["corr_lag1_full"] = 0.0
            feat["corr_lag1_first"] = 0.0
            feat["corr_lag1_second"] = 0.0
            feat["corr_lag1_stab"] = 1.0

        # 기본 통계
        feat["mean_logA"] = float(np.mean(seriesA_log))
        feat["mean_logB"] = float(np.mean(seriesB_log))
        feat["std_logA"] = float(np.std(seriesA_log))
        feat["std_logB"] = float(np.std(seriesB_log))
        feat["vol_ratio"] = float(feat["std_logB"] / (feat["std_logA"] + eps))

        typeA = type_dict.get(a, None)
        typeB = type_dict.get(b, None)
        hs4A = hs4_dict.get(a, None)
        hs4B = hs4_dict.get(b, None)

        feat["type_same"] = 1.0 if (typeA == typeB and typeA is not None) else 0.0
        feat["hs4_same"] = 1.0 if (hs4A == hs4B and hs4A is not None) else 0.0

        rows.append(feat)

    features_df = pd.DataFrame(rows)
    return features_df


# ====================== pseudo label & raw_score ======================

def add_pseudo_labels_and_raw_score(features_df: pd.DataFrame,
                                    q_pos: float = 0.85,
                                    q_neg: float = 0.30) -> pd.DataFrame:
    """
    raw_score 기반 pseudo label 생성:
      - raw_score 상위 q_pos 이상 → label=1
      - raw_score 하위 q_neg 이하 → label=0
      - 나머지는 -1 (unlabeled)
    """
    df = features_df.copy()

    corr_log_cols = [c for c in df.columns if c.startswith("corr_log_lag")]
    lag_cols_1to3 = [c for c in corr_log_cols
                     if any(c.endswith(f"lag{i}") for i in [1, 2, 3])]
    if len(lag_cols_1to3) == 0:
        lag_cols_1to3 = corr_log_cols

    df["best_corr_lag123"] = df[lag_cols_1to3].max(axis=1)

    # raw_score: 상관 + 안정성 + type/hs4 보너스
    raw = (
        0.6 * df["best_corr_lag123"] +
        0.3 * df["corr_lag1_full"] -
        0.2 * df["corr_lag1_stab"] +
        0.05 * df["type_same"] +
        0.05 * df["hs4_same"]
    )
    raw = raw.clip(-1.0, 1.0)
    df["raw_score"] = raw

    # 분위수 기반 threshold
    pos_thr = df["raw_score"].quantile(q_pos)
    neg_thr = df["raw_score"].quantile(q_neg)

    print(f"▶ raw_score quantile: q_pos={q_pos} → {pos_thr:.4f}, q_neg={q_neg} → {neg_thr:.4f}")

    df["pseudo_label"] = -1
    df.loc[df["raw_score"] >= pos_thr, "pseudo_label"] = 1
    df.loc[df["raw_score"] <= neg_thr, "pseudo_label"] = 0

    n_pos = int((df["pseudo_label"] == 1).sum())
    n_neg = int((df["pseudo_label"] == 0).sum())
    n_unl = int((df["pseudo_label"] == -1).sum())

    print(f"▶ Pseudo label 통계: pos={n_pos}, neg={n_neg}, unlabeled={n_unl}")
    return df


# ====================== LightGBM 학습 ======================

def train_lightgbm_classifier(features_labeled: pd.DataFrame,
                              feature_cols: list,
                              label_col: str = "pseudo_label"):
    df_labeled = features_labeled[features_labeled[label_col].isin([0, 1])].copy()
    X = df_labeled[feature_cols].values
    y = df_labeled[label_col].values

    print(f"▶ LightGBM 학습 데이터: {X.shape}, pos={(y==1).sum()}, neg={(y==0).sum()}")

    clf = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        min_child_samples=10,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )

    clf.fit(X, y)
    return clf


# ====================== main ======================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--monthly", type=str, required=True,
                        help="monthly_agg.csv 경로")
    parser.add_argument("--out_all", type=str, default="pair_scores_ml_all.csv",
                        help="모든 pair의 score를 저장할 파일")
    parser.add_argument("--max_lag", type=int, default=3,
                        help="feature 계산에 쓸 최대 lag")
    parser.add_argument("--q_pos", type=float, default=0.85,
                        help="raw_score 상위 quantile (pseudo 양성)  **(지금은 통계 확인용)**")
    parser.add_argument("--q_neg", type=float, default=0.30,
                        help="raw_score 하위 quantile (pseudo 음성)  **(지금은 통계 확인용)**")

    args = parser.parse_args()

    print("▶ monthly_agg.csv 로드 중...")
    monthly = pd.read_csv(args.monthly)
    print(f"  - shape: {monthly.shape}")

    print("▶ pair-level feature 계산 중...")
    features_df = build_pair_features(monthly, max_lag=args.max_lag)
    print(f"  - feature shape: {features_df.shape}")

    print("▶ raw_score + pseudo label 생성 중 (quantile 기반)...")
    features_labeled = add_pseudo_labels_and_raw_score(
        features_df,
        q_pos=args.q_pos,
        q_neg=args.q_neg
    )

    # 지금은 LightGBM 안 쓰고 raw_score만 normalize해서 ml_score로 사용
    s = features_labeled["raw_score"].values
    s_min, s_max = s.min(), s.max()
    print(f"▶ raw_score range: min={s_min:.4f}, max={s_max:.4f}")

    if s_max > s_min:
        ml_score = (s - s_min) / (s_max - s_min)
    else:
        ml_score = np.full_like(s, 1e-6, dtype=float)

    out_df = features_labeled[["leading_item_id", "following_item_id"]].copy()
    out_df["ml_score"] = ml_score
    out_df.to_csv(args.out_all, index=False)
    print(f"▶ 전체 pair score 저장 (raw_score normalize 기반): {args.out_all}")
    print(f"  - saved shape: {out_df.shape}")



if __name__ == "__main__":
    main()
