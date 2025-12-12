import argparse
import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pairs_corr", type=str, required=True,
                        help="make_pairs_corr.py 로 생성한 pairs_corr.csv 경로")
    parser.add_argument("--pseudo_stats", type=str, required=True,
                        help="analyze_pairs_pseudo.py 로 생성한 pair_pseudo_stats.csv 경로")
    parser.add_argument("--out_pairs", type=str, required=True,
                        help="최종 candidate_pairs_final.csv 출력 경로")

    parser.add_argument("--min_gain", type=float, default=0.0,
                        help="pseudo gain 최소값 (naive보다 얼마나 좋아야 하는지)")
    parser.add_argument("--min_corr", type=float, default=0.1,
                        help="|max_corr| 최소값")
    parser.add_argument("--min_n_samples", type=int, default=2,
                        help="pseudo 평가에서 최소 n_samples (pseudo_months 중 몇 개에서 관측됐는지)")
    parser.add_argument("--min_n_valid", type=int, default=6,
                        help="lag corr 계산에 사용된 최소 n_valid (겹치는 시점 수)")

    parser.add_argument("--max_per_following", type=int, default=3,
                        help="각 following_item_id(B) 당 최대 허용 leading_item_id(A) 수")
    parser.add_argument("--top_k", type=int, default=300,
                        help="전체에서 최종 선택할 최대 쌍 개수")

    args = parser.parse_args()

    print("▶ pairs_corr.csv 로드 중...")
    df_corr = pd.read_csv(args.pairs_corr)
    print("  - shape:", df_corr.shape)

    print("▶ pair_pseudo_stats.csv 로드 중...")
    df_pseudo = pd.read_csv(args.pseudo_stats)
    print("  - shape:", df_pseudo.shape)

    # 문자열 통일
    for c in ["leading_item_id", "following_item_id"]:
        df_corr[c] = df_corr[c].astype(str)
        df_pseudo[c] = df_pseudo[c].astype(str)

    # 1) 두 통계 merge
    print("▶ lag corr + pseudo gain 병합 중...")
    df = pd.merge(
        df_corr,
        df_pseudo,
        on=["leading_item_id", "following_item_id"],
        how="inner",  # 둘 다에 존재하는 쌍만 사용
        suffixes=("_corr", "_pseudo"),
    )
    print("  - merged shape:", df.shape)

    # 2) 기본 필터링
    cond = (
        (df["gain"] >= args.min_gain) &
        (df["corr_abs"] >= args.min_corr) &
        (df["n_samples"] >= args.min_n_samples) &
        (df["n_valid"] >= args.min_n_valid)
    )
    df_f = df[cond].copy()
    print(f"▶ 1차 필터링 후 쌍 개수: {len(df_f)}")

    if df_f.empty:
        print("  - 필터링 조건이 너무 빡셈 → 최소한 top_k만큼은 뽑을 수 있게 완화 필요")
        # 안전하게 df 전체에서 하나도 안 걸렀을 때 대비
        df_f = df.copy()

    # 3) 점수 계산: gain + corr_abs 가중합 (가중치는 취향대로 조정)
    #    예: score = gain + 0.5 * corr_abs
    alpha = 1.0
    beta = 0.5
    df_f["score"] = alpha * df_f["gain"] + beta * df_f["corr_abs"]

    # 4) score 기준 정렬
    df_f = df_f.sort_values(["score", "gain", "corr_abs"], ascending=False)

    # 5) following_item_id(B) 당 최대 max_per_following개까지만 허용
    keep_rows = []
    count_per_B = {}

    for _, row in df_f.iterrows():
        B = row["following_item_id"]
        cnt = count_per_B.get(B, 0)
        if cnt < args.max_per_following:
            keep_rows.append(True)
            count_per_B[B] = cnt + 1
        else:
            keep_rows.append(False)

    df_limited = df_f[keep_rows]
    print(f"▶ B당 최대 {args.max_per_following}개 제한 후 쌍 개수: {len(df_limited)}")

    # 6) 전체에서 top_k만 선택
    df_final = df_limited.head(args.top_k).copy()
    print(f"▶ 최종 top_k={args.top_k} 적용 후 쌍 개수: {len(df_final)}")

    # 7) 제출용 candidate_pairs_final.csv 생성 (leading, following 두 컬럼만)
    out_df = df_final[["leading_item_id", "following_item_id"]].copy()
    out_df.to_csv(args.out_pairs, index=False)
    print(f"▶ 최종 candidate_pairs 저장 완료: {args.out_pairs}")
    print("  - shape:", out_df.shape)

    # 디버깅용: 상위 10개 보여주기
    print("▶ 상위 10개 쌍 예시:")
    print(df_final[["leading_item_id", "following_item_id", "score", "gain", "corr_abs", "best_lag"]].head(10))


if __name__ == "__main__":
    main()
