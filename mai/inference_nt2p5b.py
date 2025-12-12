# inference_nt2p5b.py
import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from models_nt import DNAEmbeddingModel, load_tokenizer, OUT_EMB_DIM  # MODEL_NAME은 안 씀

# HF 2.5B 모델 이름
MODEL_NAME_2P5B = "InstaDeepAI/nucleotide-transformer-2.5b-1000g"


# -------------------------
# Dataset
# -------------------------
class TestDNADataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.ids = df["ID"].tolist()
        self.seqs = df["seq"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return {
            "id": self.ids[idx],
            "seq": self.seqs[idx],
        }


def collate_fn(batch, tokenizer, max_length):
    seqs = [x["seq"] for x in batch]
    ids = [x["id"] for x in batch]

    tok = tokenizer(
        seqs,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    return {
        "ids": ids,
        "input_ids": tok["input_ids"],
        "attention_mask": tok["attention_mask"]
    }


def make_chunk_df(test_df, max_len=512, stride=256):
    rows = []
    for row in test_df.itertuples(index=False):
        seq_id = row.ID
        seq = row.seq

        if not isinstance(seq, str):
            seq = str(seq)

        L = len(seq)
        if L <= max_len:
            rows.append({"ID": seq_id, "seq": seq})
        else:
            start = 0
            while start < L:
                sub = seq[start:start + max_len]
                if not sub:
                    break
                rows.append({"ID": seq_id, "seq": sub})
                start += stride
    return pd.DataFrame(rows)


# -------------------------
# Inference
# -------------------------
def main():
    test_csv_path = ""
    sample_sub_path = ""

    # 2.5B 모델 사용
    model_ckpt_dir = MODEL_NAME_2P5B

    output_path = ""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    test_df = pd.read_csv(test_csv_path)
    sample_sub = pd.read_csv(sample_sub_path)
    emb_cols = [c for c in sample_sub.columns if c.startswith("emb_")]
    emb_cols_sorted = sorted(emb_cols)
    assert len(emb_cols_sorted) == OUT_EMB_DIM

    chunk_df = make_chunk_df(test_df, max_len=512, stride=256)

    # Load tokenizer & model
    tokenizer = load_tokenizer(model_ckpt_dir)
    base_model = DNAEmbeddingModel.from_pretrained(model_ckpt_dir, device=device)
    base_model.eval()

    max_length = 512

    dataset = TestDNADataset(chunk_df, tokenizer, max_length)
    loader = DataLoader(
        dataset,
        batch_size=8,   # ← 2.5B라서 16 → 8로 줄임
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda x: collate_fn(x, tokenizer, max_length),
    )

    all_ids = []
    all_embs = []

    # Inference loop
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference 2.5B"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            embs = base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )  # (B, 768)

            all_ids.extend(batch["ids"])
            all_embs.append(embs.cpu().numpy())

    all_embs = np.concatenate(all_embs, axis=0)  # (num_chunks, 768)

    # ID별로 chunk 임베딩 평균
    id_to_embs = {}
    for sid, emb in zip(all_ids, all_embs):
        if sid not in id_to_embs:
            id_to_embs[sid] = []
        id_to_embs[sid].append(emb)

    sub_df = sample_sub.copy()
    emb_cols_sorted = sorted([c for c in sub_df.columns if c.startswith("emb_")])
    sub_df[emb_cols_sorted] = sub_df[emb_cols_sorted].astype("float32")

    final_ids = sub_df["ID"].tolist()
    final_embs = []

    for sid in final_ids:
        assert sid in id_to_embs, f"ID {sid} not found in id_to_embs"
        emb_list = id_to_embs[sid]
        arr = np.stack(emb_list, axis=0)
        mean_emb = arr.mean(axis=0)
        final_embs.append(mean_emb)

    final_embs = np.stack(final_embs, axis=0)  # (N, 768)

    sub_df.loc[:, emb_cols_sorted] = final_embs.astype("float32")

    sub_df.to_csv(output_path, index=False)
    print(f"Saved submission to {output_path}")


if __name__ == "__main__":
    main()
