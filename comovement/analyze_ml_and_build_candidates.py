import argparse, os
import pandas as pd
import numpy as np

def analyze_scores(df: pd.DataFrame):
    print("ml_score 통계")
    desc = df["ml_score"].describe()
    print(desc)

    qs = [0.99, 0.97, 0.95, 0.9, 0.85, 0.8, 0.7, 0.5]
    q_vals = df["ml_score"].quantile(qs)
    print("\n ml_score quantiles:")
    for q, v in zip(qs, q_vals):
        print(f" -q={q:.2f}: {v:.6f}")

def build_candidates_by_quantile(df: pd.DataFrame, out_dir: str):
    quantiles = [0.99, 0.97, 0.95, 0.90, 0.85, 0.80]
    print("\n Quantile 기반 candidate_pairs 생성")
    for q in quantiles:
        thr = df["ml_score"].quantile(q)
        cand = df[df["ml_score"] >= thr].copy()
        n = len(cand)
        if n == 0:
            print(f" -q={q:.2f}: threshold={thr:.6f}, N=0 -> skip")
        fname = f"candidate_pairs_ml_q{int(q*100):02d}.csv"
        out_path = os.path.join(out_dir, fname)
        cand.to_csv(out_path, index=False)
        print(f" - q={q:.2f}: threshold={thr:.6f}, N={n} -> {out_path}")

def build_candidates_by_topN(df: pd.DataFrame, out_dir: str):
    Ns = [1425]
    print("\n Top-N 기반 candidate_pairs 생성")
    total = len(df)
    for N in Ns:
        if N > total:
            print(f" -N = {N}: total={total} < N -> skip")
            continue
        cand = df.head(N).copy()
        fname = f"candidate_pairs_ml_top{N}.csv"
        out_path = os.path.join(out_dir, fname)
        cand.to_csv(out_path, index=False)
        print(f" - N={N}: saved {out_path} (shape={cand.shape})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scores",
        type=str,
        required=True,
        help="pair_scores_ml_all.csv 경로 (ml_score 포함)"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=".",
        help="candidate_pairs_*.csv 파일을 저장할 디렉토리"
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"▶ pair_scores_ml_all.csv 로드 중... ({args.scores})")
    df = pd.read_csv(args.scores)
    if "ml_score" not in df.columns:
        raise ValueError("입력 파일에 'ml_score' 컬럼이 없습니다.")

    # ml_score 기준 내림차순 정렬
    df = df.sort_values("ml_score", ascending=False).reset_index(drop=True)
    print(f"  - shape: {df.shape}")

    # 1) 분포 분석
    analyze_scores(df)

    # 2) quantile 기반 candidate 생성
    build_candidates_by_quantile(df, args.out_dir)

    # 3) top-N 기반 candidate 생성
    build_candidates_by_topN(df, args.out_dir)

    print("\n✅ 작업 완료")


if __name__ == "__main__":
    main()