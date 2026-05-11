import argparse
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ======================== 평가 함수 ========================

def _validate_input(answer_df, submission_df):
    # ① 컬럼 개수·이름 일치 여부
    if len(answer_df.columns) != len(submission_df.columns) or not all(answer_df.columns == submission_df.columns):
        raise ValueError("The columns of the answer and submission dataframes do not match.")

    # ② 필수 컬럼에 NaN 존재 여부
    if submission_df.isnull().values.any():
        raise ValueError("The submission dataframe contains missing values.")

    # ③ pair 중복 여부
    pairs = list(zip(submission_df["leading_item_id"], submission_df["following_item_id"]))
    if len(pairs) != len(set(pairs)):
        raise ValueError("The submission dataframe contains duplicate (leading_item_id, following_item_id) pairs.")


def comovement_f1(answer_df, submission_df):
    """공행성쌍 F1 계산"""
    ans = answer_df[["leading_item_id", "following_item_id"]].copy()
    sub = submission_df[["leading_item_id", "following_item_id"]].copy()

    ans["pair"] = list(zip(ans["leading_item_id"], ans["following_item_id"]))
    sub["pair"] = list(zip(sub["leading_item_id"], sub["following_item_id"]))

    G = set(ans["pair"])
    P = set(sub["pair"])

    tp = len(G & P)
    fp = len(P - G)
    fn = len(G - P)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return f1


def comovement_nmae(answer_df, submission_df, eps=1e-6):
    """
    전체 U = G ∪ P에 대한 clipped NMAE 계산
    """
    ans = answer_df[["leading_item_id", "following_item_id", "value"]].copy()
    sub = submission_df[["leading_item_id", "following_item_id", "value"]].copy()

    ans["pair"] = list(zip(ans["leading_item_id"], ans["following_item_id"]))
    sub["pair"] = list(zip(sub["leading_item_id"], sub["following_item_id"]))

    G = set(ans["pair"])
    P = set(sub["pair"])
    U = G | P

    ans_val = dict(zip(ans["pair"], ans["value"]))
    sub_val = dict(zip(sub["pair"], sub["value"]))

    errors = []
    for pair in U:
        if pair in G and pair in P:
            # 정수 변환(반올림)
            y_true = int(round(float(ans_val[pair])))
            y_pred = int(round(float(sub_val[pair])))
            rel_err = abs(y_true - y_pred) / (abs(y_true) + eps)
            rel_err = min(rel_err, 1.0)  # 오차 100% 이상은 100%로 간주
        else:
            rel_err = 1.0  # FN, FP는 오차 100%
        errors.append(rel_err)

    return np.mean(errors) if errors else 1.0


def comovement_score(answer_df, submission_df):
    _validate_input(answer_df, submission_df)
    S1 = comovement_f1(answer_df, submission_df)
    nmae_full = comovement_nmae(answer_df, submission_df, 1e-6)
    S2 = 1 - nmae_full
    score = 0.6 * S1 + 0.4 * S2
    return score


# ======================== Pseudo competition 유틸 ========================

def _parse_pseudo_months(pseudo_str):
    """
    예: "2024-12,2025-03,2025-06" 형태 문자열을
    [(2024, 12), (2025, 3), (2025, 6)] 리스트로 변환
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


def run_pseudo_eval(
    model,
    values_matrix,
    item_ids,
    time_index,
    ym2idx,
    candidate_pairs,
    input_len,
    pseudo_months_str,
    device,
):
    """
    과거 여러 개월을 '가짜 8월'로 보고 NMAE를 계산하는 의사 대회 평가.

    - F1은 알 수 없으니, pair 집합은 candidate_pairs 그대로 두고
      answer_df / submission_df 모두 같은 pair 집합으로 맞춰서
      comovement_nmae만 본다.

    - 이때 Score_if_F1=1 = 0.6*1 + 0.4*(1-NMAE)도 같이 출력해서
      '회귀 성능이 좋아지면 Score가 얼마나 좋아질 수 있는지' 감만 본다.
    """
    target_ym_list = _parse_pseudo_months(pseudo_months_str)
    if not target_ym_list:
        print("▶ pseudo_months 가 지정되지 않아 의사 대회 평가를 건너뜁니다.")
        return

    print("▶ 의사 대회(pseudo competition) 평가 시작")
    print(f"  - 타겟 월 리스트: {', '.join(f'{y}-{m:02d}' for (y, m) in target_ym_list)}")

    # 공통 준비: 월 sin/cos, item2col
    T = len(time_index)
    months = np.array([m for (_, m) in time_index], dtype=np.float32)
    month_rad = 2 * np.pi * (months - 1) / 12.0
    month_sin = np.sin(month_rad)
    month_cos = np.cos(month_rad)

    item2col = {str(item_id): j for j, item_id in enumerate(item_ids)}

    model.eval()
    all_nmae = []

    with torch.no_grad():
        for (year, month) in target_ym_list:
            ym = (year, month)
            if ym not in ym2idx:
                print(f"  - [pseudo] {year}-{month:02d}: time_index에 없어 스킱")
                continue

            k = ym2idx[ym]  # target 시점 index

            if k < input_len:
                print(
                    f"  - [pseudo] {year}-{month:02d}: "
                    f"과거 {input_len}개월이 없어서 스킵 (idx={k})"
                )
                continue

            pred_rows = []
            ans_rows = []

            for _, row in candidate_pairs.iterrows():
                A = str(row["leading_item_id"])
                B = str(row["following_item_id"])

                if A not in item2col or B not in item2col:
                    continue

                col_A = item2col[A]
                col_B = item2col[B]

                series_A = values_matrix[:, col_A]
                series_B = values_matrix[:, col_B]

                logA = np.log1p(series_A)
                logB = np.log1p(series_B)

                # 전월 대비 변화량 (첫 시점은 0)
                diffA = np.zeros_like(logA)
                diffB = np.zeros_like(logB)
                diffA[1:] = logA[1:] - logA[:-1]
                diffB[1:] = logB[1:] - logB[:-1]

                # 입력 윈도우: [k-input_len .. k-1]
                start = k - input_len
                end = k

                window_A = logA[start:end]
                window_B = logB[start:end]
                window_dA = diffA[start:end]
                window_dB = diffB[start:end]
                window_sin = month_sin[start:end]
                window_cos = month_cos[start:end]

                if (
                    len(window_A) < input_len
                    or len(window_B) < input_len
                    or len(window_dA) < input_len
                    or len(window_dB) < input_len
                    or len(window_sin) < input_len
                    or len(window_cos) < input_len
                ):
                    continue

                window_feat = np.stack(
                    [window_A, window_B, window_dA, window_dB, window_sin, window_cos],
                    axis=-1,
                ).astype(np.float32)  # (input_len, 6)

                x = torch.from_numpy(window_feat).unsqueeze(0).to(device)  # (1, L, 6)
                log_pred_next = model(x).item()
                pred_value = np.expm1(log_pred_next)
                if pred_value < 0:
                    pred_value = 0.0

                pred_rows.append(
                    {
                        "leading_item_id": A,
                        "following_item_id": B,
                        "value": int(round(pred_value)),
                    }
                )

                # '정답'은 해당 월 B의 실제 value
                true_val = float(series_B[k])
                ans_rows.append(
                    {
                        "leading_item_id": A,
                        "following_item_id": B,
                        "value": true_val,
                    }
                )

            if not pred_rows:
                print(f"  - [pseudo] {year}-{month:02d}: 유효한 (A,B) 쌍이 없어 스킵")
            else:
                pred_df = pd.DataFrame(pred_rows)
                ans_df = pd.DataFrame(ans_rows)

                nmae = comovement_nmae(ans_df, pred_df, eps=1e-6)
                pseudo_score = 0.6 * 1.0 + 0.4 * (1.0 - nmae)

                print(
                    f"[Pseudo {year}-{month:02d}] "
                    f"NMAE={nmae:.6f}, Score_if_F1=1={pseudo_score:.6f}"
                )
                all_nmae.append(nmae)

    if all_nmae:
        mean_nmae = float(np.mean(all_nmae))
        mean_score = 0.6 * 1.0 + 0.4 * (1.0 - mean_nmae)
        print(
            f"▶ Pseudo 평가 요약: "
            f"mean NMAE={mean_nmae:.6f}, mean Score_if_F1=1={mean_score:.6f}"
        )
    else:
        print("▶ Pseudo 평가를 수행할 수 있는 월이 없었습니다.")


# ======================== Dataset ========================

class PairSeqDataset(Dataset):
    """
    (A,B) 쌍의 시계열 윈도우를 모아놓은 Dataset.
    X: [seq_len, input_dim]  (예: [input_len, 6])
    y: scalar (log1p(B_next_value))
    """
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ======================== 유틸 함수 ========================

def build_time_index(monthly):
    """
    year, month 를 기반으로 전체 타임라인 인덱스 생성.
    반환:
      - time_index: [ (year, month), ... ] 정렬된 리스트
      - ym2idx: { (year,month) -> idx } 매핑
    """
    df = monthly[["year", "month"]].drop_duplicates().copy()
    df = df.sort_values(["year", "month"])
    time_index = list(zip(df["year"], df["month"]))
    ym2idx = {ym: i for i, ym in enumerate(time_index)}
    return time_index, ym2idx


def build_pair_series(monthly, time_index, ym2idx):
    """
    월별 집계 데이터 monthly 로부터
    item_id 별로 전체 타임라인에 맞는 value 시계열 생성.
    """
    T = len(time_index)

    item_ids = sorted(monthly["item_id"].astype(str).unique().tolist())
    n_items = len(item_ids)

    values_matrix = np.zeros((T, n_items), dtype=np.float64)

    type_dict = {}
    hs4_dict = {}

    meta = (
        monthly
        .groupby("item_id", as_index=False)
        .agg({"type": "first", "hs4": "first"})
    )

    for _, row in meta.iterrows():
        item = str(row["item_id"])
        type_dict[item] = row["type"]
        hs4_dict[item] = row["hs4"]

    for j, item in enumerate(item_ids):
        sub = monthly[monthly["item_id"].astype(str) == item]
        for _, r in sub.iterrows():
            ym = (int(r["year"]), int(r["month"]))
            idx = ym2idx[ym]
            values_matrix[idx, j] = float(r["value"])

    return values_matrix, item_ids, type_dict, hs4_dict


def build_train_windows_for_pairs(values_matrix, item_ids, time_index,
                                  candidate_pairs, input_len=12):
    """
    모든 candidate (A,B) 쌍에 대해
    길이 input_len 의 윈도우로부터 B_next 를 예측하는 학습 샘플 생성.

    feature: [A_log, B_log, dA, dB, month_sin, month_cos]
    target: log1p(B_next_value)
    """
    T, n_items = values_matrix.shape

    item2col = {str(item_id): j for j, item_id in enumerate(item_ids)}

    months = np.array([m for (_, m) in time_index], dtype=np.float32)  # 1~12
    month_rad = 2 * np.pi * (months - 1) / 12.0
    month_sin = np.sin(month_rad)
    month_cos = np.cos(month_rad)

    X_list = []
    y_list = []
    time_idx_list = []

    for _, row in candidate_pairs.iterrows():
        A = str(row["leading_item_id"])
        B = str(row["following_item_id"])

        if A not in item2col or B not in item2col:
            continue

        col_A = item2col[A]
        col_B = item2col[B]

        series_A = values_matrix[:, col_A]
        series_B = values_matrix[:, col_B]

        logA = np.log1p(series_A)
        logB = np.log1p(series_B)

        diffA = np.zeros_like(logA)
        diffB = np.zeros_like(logB)
        diffA[1:] = logA[1:] - logA[:-1]
        diffB[1:] = logB[1:] - logB[:-1]

        for k in range(input_len, T):
            start = k - input_len
            end = k

            window_A = logA[start:end]
            window_B = logB[start:end]
            window_dA = diffA[start:end]
            window_dB = diffB[start:end]
            window_sin = month_sin[start:end]
            window_cos = month_cos[start:end]

            window_feat = np.stack(
                [window_A, window_B, window_dA, window_dB, window_sin, window_cos],
                axis=-1
            )  # (input_len, 6)

            target = logB[k]

            X_list.append(window_feat)
            y_list.append(target)
            time_idx_list.append(k)

    if not X_list:
        raise RuntimeError("No training windows were generated. Check input_len and data range.")

    X = np.stack(X_list, axis=0)  # (N, input_len, 6)
    y = np.array(y_list, dtype=np.float64)
    time_idx_arr = np.array(time_idx_list, dtype=np.int64)

    return X, y, time_idx_arr


# ======================== Transformer 모델 ========================

class TimeSeriesTransformer(nn.Module):
    """
    간단한 Transformer Encoder 기반 시계열 모델.
    입력: (batch, seq_len, input_dim)
    출력: scalar (batch,) - 마지막 토큰 representation 에서 예측
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
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.out_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        """
        x: (batch, seq_len, input_dim)
        """
        h = self.input_proj(x)          # (B, L, d_model)
        h_enc = self.encoder(h)         # (B, L, d_model)
        last_token = h_enc[:, -1, :]    # (B, d_model)
        out = self.out_proj(last_token) # (B, 1)
        return out.squeeze(-1)          # (B,)


# ======================== 학습 & 예측 루틴 ========================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--monthly", type=str, required=True,
                        help="monthly_agg.csv 경로")
    parser.add_argument("--pairs", type=str, required=True,
                        help="candidate_pairs_topN.csv 경로")
    parser.add_argument("--sample_submission", type=str, required=True,
                        help="sample_submission.csv 경로")
    parser.add_argument("--out_submission", type=str, default="submission_ts_transformer.csv",
                        help="출력 submission 파일 이름")
    parser.add_argument("--answer", type=str, default=None,
                        help="(선택) answer.csv 경로 - 있으면 comovement_score 계산")

    parser.add_argument(
        "--pseudo_months",
        type=str,
        default=None,
        help=(
            "과거 월을 가짜 8월로 보고 회귀 성능을 측정하기 위한 월 목록 "
            '(예: "2024-12,2025-03,2025-06")'
        ),
    )

    parser.add_argument("--input_len", type=int, default=12,
                        help="윈도우 길이 (과거 몇 개월을 볼지)")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--valid_last_n_steps", type=int, default=6,
                        help="마지막 time index 기준 몇 개를 validation 으로 둘지")
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
    candidate_pairs = pd.read_csv(args.pairs)
    print(f"  - pairs shape: {candidate_pairs.shape}")

    monthly["item_id"] = monthly["item_id"].astype(str)
    candidate_pairs["leading_item_id"] = candidate_pairs["leading_item_id"].astype(str)
    candidate_pairs["following_item_id"] = candidate_pairs["following_item_id"].astype(str)

    # 2) 타임라인 & 시계열 매트릭스 생성
    print("▶ 타임라인 및 item 시계열 구성 중...")
    time_index, ym2idx = build_time_index(monthly)
    values_matrix, item_ids, type_dict, hs4_dict = build_pair_series(
        monthly, time_index, ym2idx
    )
    print(f"  - time steps: {len(time_index)}, items: {len(item_ids)}")

    # 3) 학습 윈도우 생성
    print("▶ 학습용 윈도우 생성 중...")
    X, y, time_idx_arr = build_train_windows_for_pairs(
        values_matrix, item_ids, time_index,
        candidate_pairs,
        input_len=args.input_len
    )
    print(f"  - total train windows: {X.shape[0]}  (seq_len={X.shape[1]}, feat_dim={X.shape[2]})")

    # 4) train / valid split (time 기반)
    max_tidx = time_idx_arr.max()
    valid_threshold = max_tidx - args.valid_last_n_steps + 1

    train_mask = time_idx_arr < valid_threshold
    valid_mask = time_idx_arr >= valid_threshold

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_valid = X[valid_mask]
    y_valid = y[valid_mask]

    print(f"  - train windows: {X_train.shape[0]}, valid windows: {X_valid.shape[0]}")

    train_dataset = PairSeqDataset(X_train, y_train)
    valid_dataset = PairSeqDataset(X_valid, y_valid)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False
    )

    # 5) 모델 생성
    model = TimeSeriesTransformer(
        input_dim=X.shape[2],
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout
    ).to(device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    best_valid_loss = float("inf")
    patience = 20
    no_improve = 0

    # 6) 학습 루프
    print("▶ 학습 시작...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        n_train = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = xb.size(0)
            train_loss_sum += loss.item() * bs
            n_train += bs

        train_loss = train_loss_sum / max(n_train, 1)

        model.eval()
        valid_loss_sum = 0.0
        n_valid = 0
        with torch.no_grad():
            for xb, yb in valid_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                bs = xb.size(0)
                valid_loss_sum += loss.item() * bs
                n_valid += bs

        valid_loss = valid_loss_sum / max(n_valid, 1)

        print(f"[Epoch {epoch:03d}] train_loss={train_loss:.5f}  valid_loss={valid_loss:.5f}")

        if valid_loss < best_valid_loss - 1e-4:
            best_valid_loss = valid_loss
            no_improve = 0
            torch.save(model.state_dict(), "best_ts_transformer.pth")
        else:
            no_improve += 1
            if no_improve >= patience:
                print("▶ Early stopping triggered.")
                break

    print(f"▶ Best valid loss: {best_valid_loss:.5f}")
    model.load_state_dict(torch.load("best_ts_transformer.pth", map_location=device))
    model.eval()

    # (선택) Pseudo 평가
    if args.pseudo_months is not None and args.pseudo_months.strip():
        run_pseudo_eval(
            model=model,
            values_matrix=values_matrix,
            item_ids=item_ids,
            time_index=time_index,
            ym2idx=ym2idx,
            candidate_pairs=candidate_pairs,
            input_len=args.input_len,
            pseudo_months_str=args.pseudo_months,
            device=device,
        )

    # 7) 2025-08 예측
    print("▶ 2025-08 예측용 윈도우 생성 중...")

    T = len(time_index)
    months = np.array([m for (_, m) in time_index], dtype=np.float32)
    month_rad = 2 * np.pi * (months - 1) / 12.0
    month_sin = np.sin(month_rad)
    month_cos = np.cos(month_rad)

    item2col = {str(item_id): j for j, item_id in enumerate(item_ids)}

    pred_rows = []

    for _, row in candidate_pairs.iterrows():
        A = str(row["leading_item_id"])
        B = str(row["following_item_id"])

        if A not in item2col or B not in item2col:
            continue

        col_A = item2col[A]
        col_B = item2col[B]

        series_A = values_matrix[:, col_A]
        series_B = values_matrix[:, col_B]

        logA = np.log1p(series_A)
        logB = np.log1p(series_B)

        diffA = np.zeros_like(logA)
        diffB = np.zeros_like(logB)
        diffA[1:] = logA[1:] - logA[:-1]
        diffB[1:] = logB[1:] - logB[:-1]

        if T < args.input_len:
            continue

        start = T - args.input_len
        end = T

        window_A = logA[start:end]
        window_B = logB[start:end]
        window_dA = diffA[start:end]
        window_dB = diffB[start:end]
        window_sin = month_sin[start:end]
        window_cos = month_cos[start:end]

        window_feat = np.stack(
            [window_A, window_B, window_dA, window_dB, window_sin, window_cos],
            axis=-1
        )

        x = torch.from_numpy(window_feat.astype(np.float32)).unsqueeze(0).to(device)

        with torch.no_grad():
            log_pred_next = model(x).item()

        pred_value = np.expm1(log_pred_next)
        if pred_value < 0:
            pred_value = 0.0

        pred_rows.append({
            "leading_item_id": A,
            "following_item_id": B,
            "value": int(round(pred_value))
        })

    pred_df = pd.DataFrame(pred_rows)
    print(f"  - 예측된 pair 수: {pred_df.shape[0]}")

    # ===== fallback 값 계산 (최근 3개월 평균 기반) =====
    k = 3
    if values_matrix.shape[0] >= k:
        global_fallback_value = float(values_matrix[-k:, :].mean())
        itemwise_recent_mean = values_matrix[-k:, :].mean(axis=0)
    else:
        global_fallback_value = float(values_matrix.mean())
        itemwise_recent_mean = values_matrix.mean(axis=0)

    fallback_dict = {
        str(item_id): float(v)
        for item_id, v in zip(item_ids, itemwise_recent_mean)
    }

    print(f"▶ global_fallback_value (최근 {k}개월 전체 평균): {global_fallback_value:.2f}")

    # 8) 최종 제출 파일 생성
    print("▶ sample_submission 기반 최종 제출 파일 생성 중...")
    sub = pd.read_csv(args.sample_submission)


    pair2val = {
        (str(r["leading_item_id"]), str(r["following_item_id"])): int(r["value"])
        for _, r in pred_df.iterrows()
    }

    vals = []
    for a, b in zip(sub["leading_item_id"].values, sub["following_item_id"].values):
        key = (str(a), str(b))
        if key in pair2val:
            vals.append(pair2val[key])
        else:
            vals.append(0)

    sub = pred_df[["leading_item_id", "following_item_id", "value"]].copy()
    sub["leading_item_id"] = sub["leading_item_id"].astype(str)
    sub["following_item_id"] = sub["leading_item_id"].astype(str)
    sub["value"] = sub["value"].astype(int)
    sub.to_csv(args.out_submission, index=False)
    print(f"▶ 최종 제출 파일 저장 완료: {args.out_submission}")
    print(f"  - shape: {sub.shape}")

    # 9) answer.csv 있으면 score 계산
    if args.answer is not None:
        print("▶ answer.csv 기반 comovement_score 계산 중...")
        answer_df = pd.read_csv(args.answer)
        try:
            score = comovement_score(answer_df, sub)
            f1 = comovement_f1(answer_df, sub)
            nmae = comovement_nmae(answer_df, sub)
            print(f"  - F1   : {f1:.6f}")
            print(f"  - NMAE : {nmae:.6f}")
            print(f"  - Score: {score:.6f}")
        except Exception as e:
            print(f"  - 점수 계산 중 오류: {e}")


if __name__ == "__main__":
    main()
