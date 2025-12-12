import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ======================== 간단한 TS Transformer (6-feat 버전) ========================

class TimeSeriesTransformer(nn.Module):
    """
    train_multivar_ts_pred.py 에서 사용한 것과 동일한 구조:
    입력: (batch, seq_len, input_dim=6)
    출력: (batch,)  - log1p(B_next_value)
    """
    def __init__(self, input_dim=6, d_model=128, nhead=8,
                 num_layers=4, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.out_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (B, L, input_dim)
        h = self.input_proj(x)        # (B, L, d_model)
        h_enc = self.encoder(h)       # (B, L, d_model)
        last_token = h_enc[:, -1, :]  # (B, d_model)
        out = self.out_proj(last_token)  # (B, 1)
        return out.squeeze(-1)        # (B,)


# ======================== 유틸 함수들 ========================

def build_time_index(monthly: pd.DataFrame):
    """year, month 기반 time_index / ym2idx 생성."""
    df = monthly[["year", "month"]].drop_duplicates().copy()
    df = df.sort_values(["year", "month"])
    time_index = list(zip(df["year"], df["month"]))
    ym2idx = {ym: i for i, ym in enumerate(time_index)}
    return time_index, ym2idx


def build_values_matrix(monthly: pd.DataFrame, time_index, ym2idx):
    """
    item_id 별로 전체 타임라인에 맞는 value 시계열 생성.

    반환:
      - values_matrix: (T, n_items)
      - item_ids: 리스트 (열 순서와 대응)
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


def _parse_pseudo_months(pseudo_str: str):
    """
    "2024-12,2025-03,2025-06" -> [(2024,12), (2025,3), (2025,6)]
    """
    result = []
    if pseudo_str is None:
        return result
    for token in pseudo_str.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            y_str, m_str = token.split("-")
            y = int(y_str)
            m = int(m_str)
            result.append((y, m))
        except Exception:
            print(f"[WARN] 잘못된 pseudo month 형식: {token} (무시)")
    return result


def clipped_rel_err(y_true, y_pred, eps=1e-6):
    """
    대회 NMAE 정의와 비슷하게:
      |y - y_hat| / (|y| + eps), 최대 1로 클리핑
    """
    y_true = float(y_true)
    y_pred = float(y_pred)
    if np.isnan(y_true) or np.isnan(y_pred):
        return 1.0
    denom = abs(y_true) + eps
    if denom <= 0:
        # y_true == 0 인 경우, 오차 발생 시 상대오차 크게 보되 최대 1
        return min(abs(y_pred), 1.0)
    rel = abs(y_true - y_pred) / denom
    return float(min(rel, 1.0))


# ======================== 메인 분석 루틴 ========================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--monthly", type=str, required=True,
                        help="monthly_agg.csv 경로")
    parser.add_argument("--pairs", type=str, required=True,
                        help="candidate_pairs_topN.csv 경로 (기존 top301 등)")
    parser.add_argument("--ckpt", type=str, default="best_ts_transformer.pth",
                        help="학습된 TS Transformer 체크포인트 경로")
    parser.add_argument("--input_len", type=int, default=12,
                        help="TS 모델이 보는 과거 윈도우 길이")
    parser.add_argument(
        "--pseudo_months",
        type=str,
        required=True,
        help='가짜 테스트로 사용할 월 목록 (예: "2024-12,2025-03,2025-06")',
    )
    parser.add_argument(
        "--naive_k",
        type=int,
        default=3,
        help="naive 예측 시 B의 최근 k개월 평균을 사용할 때의 k",
    )
    parser.add_argument(
        "--out_stats",
        type=str,
        default="pair_pseudo_stats.csv",
        help="쌍별 pseudo 성능 통계 출력 csv 경로",
    )

    # 모델 구조 (train 스크립트와 동일하게 맞춰야 함)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"▶ Using device: {device}")

    # 1) 데이터 로드
    print("▶ monthly_agg.csv 로드 중...")
    monthly = pd.read_csv(args.monthly)
    print(f"  - monthly shape: {monthly.shape}")

    print("▶ candidate_pairs_topN.csv 로드 중...")
    pairs_df = pd.read_csv(args.pairs)
    print(f"  - pairs shape: {pairs_df.shape}")

    monthly["item_id"] = monthly["item_id"].astype(str)
    pairs_df["leading_item_id"] = pairs_df["leading_item_id"].astype(str)
    pairs_df["following_item_id"] = pairs_df["following_item_id"].astype(str)

    # 2) 타임라인 & 시계열
    print("▶ 타임라인 및 values_matrix 구성 중...")
    time_index, ym2idx = build_time_index(monthly)
    values_matrix, item_ids = build_values_matrix(monthly, time_index, ym2idx)
    T, n_items = values_matrix.shape
    print(f"  - time steps: {T}, items: {n_items}")

    # 3) 월 sin/cos, item2col
    months = np.array([m for (_, m) in time_index], dtype=np.float32)
    month_rad = 2 * np.pi * (months - 1) / 12.0
    month_sin = np.sin(month_rad)
    month_cos = np.cos(month_rad)

    item2col = {str(item_id): j for j, item_id in enumerate(item_ids)}

    # 4) 모델 로드
    print("▶ TimeSeriesTransformer 초기화 및 체크포인트 로드 중...")
    model = TimeSeriesTransformer(
        input_dim=6,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)

    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"  - checkpoint: {args.ckpt} 로드 완료")

    # 5) pseudo months 파싱
    pseudo_list = _parse_pseudo_months(args.pseudo_months)
    if not pseudo_list:
        print("▶ pseudo_months 가 비어있습니다. 종료합니다.")
        return

    print("▶ Pseudo months:", ", ".join(f"{y}-{m:02d}" for (y, m) in pseudo_list))

    # 6) 쌍별로 model_err / naive_err 기록용 딕셔너리
    # key: (A,B) 문자열 튜플
    stats = {}

    # 7) 각 pseudo month에 대해 루프
    with torch.no_grad():
        for (year, month) in pseudo_list:
            ym = (year, month)
            if ym not in ym2idx:
                print(f"  - [WARN] {year}-{month:02d} 는 time_index에 없음 → 스킵")
                continue

            k = ym2idx[ym]  # target index (B[k])

            if k < args.input_len:
                print(
                    f"  - [WARN] {year}-{month:02d}: "
                    f"과거 {args.input_len}개월이 없어 스킵 (idx={k})"
                )
                continue

            print(f"▶ [Pseudo {year}-{month:02d}] 쌍별 model vs naive 성능 계산 중...")

            for _, row in pairs_df.iterrows():
                A = str(row["leading_item_id"])
                B = str(row["following_item_id"])
                key = (A, B)

                if A not in item2col or B not in item2col:
                    # 데이터가 부족한 item이면 스킵
                    continue

                col_A = item2col[A]
                col_B = item2col[B]

                series_A = values_matrix[:, col_A]
                series_B = values_matrix[:, col_B]

                # log1p + diff
                logA = np.log1p(series_A)
                logB = np.log1p(series_B)

                diffA = np.zeros_like(logA)
                diffB = np.zeros_like(logB)
                diffA[1:] = logA[1:] - logA[:-1]
                diffB[1:] = logB[1:] - logB[:-1]

                # 입력 윈도우 범위
                start = k - args.input_len
                end = k

                window_A = logA[start:end]
                window_B = logB[start:end]
                window_dA = diffA[start:end]
                window_dB = diffB[start:end]
                window_sin = month_sin[start:end]
                window_cos = month_cos[start:end]

                # safety
                if (
                    len(window_A) < args.input_len
                    or len(window_B) < args.input_len
                    or len(window_dA) < args.input_len
                    or len(window_dB) < args.input_len
                    or len(window_sin) < args.input_len
                    or len(window_cos) < args.input_len
                ):
                    continue

                # 모델 입력
                window_feat = np.stack(
                    [window_A, window_B, window_dA, window_dB, window_sin, window_cos],
                    axis=-1
                ).astype(np.float32)  # (L, 6)

                x = torch.from_numpy(window_feat).unsqueeze(0).to(device)  # (1, L, 6)
                log_pred_next = model(x).item()
                model_pred = np.expm1(log_pred_next)
                if model_pred < 0:
                    model_pred = 0.0

                # true 값
                true_val = float(series_B[k])

                # naive: B의 최근 naive_k개월 평균
                if k >= args.naive_k:
                    naive_window = series_B[k-args.naive_k:k]  # k-1까지
                    naive_pred = float(np.mean(naive_window))
                else:
                    naive_pred = float(series_B[k-1]) if k > 0 else 0.0

                err_model = clipped_rel_err(true_val, model_pred)
                err_naive = clipped_rel_err(true_val, naive_pred)

                if key not in stats:
                    stats[key] = {
                        "leading_item_id": A,
                        "following_item_id": B,
                        "model_errs": [],
                        "naive_errs": [],
                    }
                stats[key]["model_errs"].append(err_model)
                stats[key]["naive_errs"].append(err_naive)

    # 8) 집계 → DataFrame
    rows = []
    for (A, B), d in stats.items():
        m_errs = d["model_errs"]
        n_errs = d["naive_errs"]
        if not m_errs:
            continue
        model_nmae = float(np.mean(m_errs))
        naive_nmae = float(np.mean(n_errs))
        gain = naive_nmae - model_nmae  # 양수면 모델이 naive보다 좋음
        n_samples = len(m_errs)
        rows.append({
            "leading_item_id": A,
            "following_item_id": B,
            "model_nmae": model_nmae,
            "naive_nmae": naive_nmae,
            "gain": gain,
            "n_samples": n_samples,
        })

    if not rows:
        print("▶ 쌍별 통계를 만들 수 있는 데이터가 없습니다.")
        return

    stats_df = pd.DataFrame(rows)
    stats_df = stats_df.sort_values(["gain", "n_samples"], ascending=[False, False])

    print("▶ 쌍별 pseudo 성능 통계 상위 10개 예시:")
    print(stats_df.head(10))

    stats_df.to_csv(args.out_stats, index=False)
    print(f"▶ pair_pseudo_stats 저장 완료: {args.out_stats}")
    print(f"  - shape: {stats_df.shape}")


if __name__ == "__main__":
    main()
