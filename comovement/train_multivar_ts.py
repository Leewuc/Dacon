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


# ======================== Dataset ========================

class PairSeqDataset(Dataset):
    """
    (A,B) 쌍의 시계열 윈도우를 모아놓은 Dataset.
    X: [seq_len, input_dim]  (예: [input_len, 4])  (A,B 값 + 월 sin/cos)
    y: scalar (log1p(B_next_value))
    """
    def __init__(self, X, y):
        # X: (N, seq_len, input_dim)
        # y: (N,)
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

    반환:
      - values_matrix: shape (T, n_items), (year,month 순) value (log1p 변환 전의 원값)
      - item_ids: column에 해당하는 item_id 리스트
      - type_dict: { item_id -> type }
      - hs4_dict: { item_id -> hs4 }
    """
    T = len(time_index)

    # item_id 목록 (문자열로 가정)
    item_ids = sorted(monthly["item_id"].astype(str).unique().tolist())
    n_items = len(item_ids)

    # 값 매트릭스 및 메타정보
    values_matrix = np.zeros((T, n_items), dtype=np.float64)

    type_dict = {}
    hs4_dict = {}

    # item별 meta 먼저 모으기
    meta = (monthly
            .groupby("item_id", as_index=False)
            .agg({"type": "first", "hs4": "first"}))

    for _, row in meta.iterrows():
        item = str(row["item_id"])
        type_dict[item] = row["type"]
        hs4_dict[item] = row["hs4"]

    # 실제 value 채우기
    # (item_id, year, month) 기준으로 value 합산은 이미 돼 있다고 가정
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

    - feature: [A_value, B_value, month_sin, month_cos]
      (모두 log1p 변환해서 모델에 넣음)
    - target: log1p(B_next_value)

    반환:
      - X: shape (N_samples, input_len, 4)
      - y: shape (N_samples,)
      - time_idx_arr: 각 샘플의 target이 위치한 time index (일부 split 용)
    """
    T, n_items = values_matrix.shape

    # item_id -> column index 맵핑 (문자열 키)
    item2col = {str(item_id): j for j, item_id in enumerate(item_ids)}

    # month of year를 기반으로 sin/cos feature 생성
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

        # log1p 변환
        logA = np.log1p(series_A)
        logB = np.log1p(series_B)

        # k: target 시점 index (B_next)
        # window: [k-input_len .. k-1] 사용
        # k는 최소 input_len, 최대 T-1 까지 (T-1은 마지막 관측)
        for k in range(input_len, T):
            # 입력 윈도우 범위
            start = k - input_len
            end = k  # k-1 까지

            # feature 시퀀스 구성: [A, B, sin, cos]
            window_A = logA[start:end]             # (input_len,)
            window_B = logB[start:end]             # (input_len,)
            window_sin = month_sin[start:end]      # (input_len,)
            window_cos = month_cos[start:end]      # (input_len,)

            # stack -> (input_len, 4)
            window_feat = np.stack(
                [window_A, window_B, window_sin, window_cos],
                axis=-1
            )

            # target: B_next (시점 k)의 log1p(B)
            target = logB[k]  # scalar

            X_list.append(window_feat)
            y_list.append(target)
            time_idx_list.append(k)

    if not X_list:
        raise RuntimeError("No training windows were generated. Check input_len and data range.")

    X = np.stack(X_list, axis=0)  # (N, input_len, 4)
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
    def __init__(self, input_dim=4, d_model=128, nhead=8,
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

    # 🔧 item_id 계열은 전부 문자열로 통일
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

    # 3) 모든 (A,B) 후보쌍에 대해 학습 윈도우 생성
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

    criterion = nn.L1Loss()  # MAE 기반
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
        # ---- train ----
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

        # ---- valid ----
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

        # early stopping
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
    # best 모델 로드
    model.load_state_dict(torch.load("best_ts_transformer.pth", map_location=device))
    model.eval()

    # 7) 2025-08 예측 (각 (A,B) 후보쌍)
    print("▶ 2025-08 예측용 윈도우 생성 중...")

    T = len(time_index)
    months = np.array([m for (_, m) in time_index], dtype=np.float32)
    month_rad = 2 * np.pi * (months - 1) / 12.0
    month_sin = np.sin(month_rad)
    month_cos = np.cos(month_rad)

    # item -> column index (문자열 키)
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

        # 마지막 input_len 개월 (끝 index = T-1) 사용
        if T < args.input_len:
            continue

        start = T - args.input_len
        end = T  # T-1 까지

        window_A = logA[start:end]
        window_B = logB[start:end]
        window_sin = month_sin[start:end]
        window_cos = month_cos[start:end]

        window_feat = np.stack(
            [window_A, window_B, window_sin, window_cos],
            axis=-1
        )  # (input_len, 4)

        x = torch.from_numpy(window_feat.astype(np.float32)).unsqueeze(0).to(device)  # (1, L, 4)

        with torch.no_grad():
            log_pred_next = model(x).item()  # log1p(pred_value)

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
        # 전체 평균 (global fallback)
        global_fallback_value = float(values_matrix[-k:, :].mean())
        # item별 최근 k개월 평균
        itemwise_recent_mean = values_matrix[-k:, :].mean(axis=0)
    else:
        global_fallback_value = float(values_matrix.mean())
        itemwise_recent_mean = values_matrix.mean(axis=0)

    fallback_dict = {
        str(item_id): float(v)
        for item_id, v in zip(item_ids, itemwise_recent_mean)
    }

    print(f"▶ global_fallback_value (최근 {k}개월 전체 평균): {global_fallback_value:.2f}")

    # 8) sample_submission 기반 최종 제출 파일 생성
    print("▶ sample_submission 기반 최종 제출 파일 생성 중...")
    sub = pd.read_csv(args.sample_submission)

    # 🔧 문자열로 통일
    sub["leading_item_id"] = sub["leading_item_id"].astype(str)
    sub["following_item_id"] = sub["following_item_id"].astype(str)

    # 예측 dict (문자열 키)
    pair2val = {
        (str(r["leading_item_id"]), str(r["following_item_id"])): int(r["value"])
        for _, r in pred_df.iterrows()
    }

    vals = []
    for a, b in zip(sub["leading_item_id"].values, sub["following_item_id"].values):
        key = (str(a), str(b))
        if key in pair2val:
            # candidate_pairs에 있었고 Transformer가 예측한 쌍
            vals.append(pair2val[key])
        else:
            # candidate_pairs에는 없지만 정답에 있을 수도 있는 쌍
            fid = str(b)
            if fid in fallback_dict:
                vals.append(int(round(fallback_dict[fid])))
            else:
                vals.append(int(round(global_fallback_value)))

    sub["value"] = vals

    sub.to_csv(args.out_submission, index=False)
    print(f"▶ 최종 제출 파일 저장 완료: {args.out_submission}")
    print(f"  - shape: {sub.shape}")

    # 9) answer.csv 가 있다면 스코어 계산
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
