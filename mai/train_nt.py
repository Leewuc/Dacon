# train_nt.py
import os, math
import torch
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
from models_nt import MODEL_NAME

def clean_seq(s: str) -> str:
    s = str(s).upper()
    allowed = set(['A', 'C', 'G', 'T', 'N'])
    out = []
    for c in s:
        if c in allowed:
            out.append(c)
        else:
            out.append('N')
    return ''.join(out)

def load_fasta_sequences(fasta_paths):
    sequences = []
    for path in fasta_paths:
        with open(path, "r") as f:
            cur = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if cur:
                        seq = clean_seq(''.join(cur))
                        if len(seq) > 0:
                            sequences.append(seq)
                        cur = []
                else:
                    cur.append(line)
            if cur:
                seq = clean_seq(''.join(cur))
                if len(seq) > 0:
                    sequences.append(seq)
    return sequences

def make_windows_from_sequences(sequences, window_size=512, stride=512, max_windows_per_seq=None):
    windows = []
    for seq in sequences:
        L = len(seq)
        if L < window_size:
            continue
        count=0
        for start in range(0, L - window_size + 1, stride):
            windows.append(seq[start:start + window_size])
            count += 1
            if max_windows_per_seq is not None and count >= max_windows_per_seq:
                break
    return windows

class GenomeMLMDataset(Dataset):
    def __init__(self, windows, tokenizer, max_length=512):
        self.windows = windows
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        seq = self.windows[idx]
        encoding = self.tokenizer(
            seq,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)
        return item
    

def main():
    fasta_paths = [
        "",
        "",
    ]

    output_dir = ""
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    max_length = 512
    batch_size = 4
    num_epochs = 5
    learning_rate = 5e-5
    weight_decay = 0.01
    warmup_ratio = 0.05
    gradient_accumulation_steps = 1

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device)

    vocab_size = model.get_input_embeddings().weight.shape[0]
    print(f"Vocab size from model embeddings: {vocab_size}")

    print("Loading sequences from FASTA files...")
    sequences = load_fasta_sequences(fasta_paths)
    print(f"Loaded {len(sequences)} sequences.")

    print("Making windows from sequences...")
    windows = make_windows_from_sequences(sequences, window_size=max_length, stride=max_length, max_windows_per_seq=5000)
    print(f"Created {len(windows)} windows.")

    train_dataset = GenomeMLMDataset(windows, tokenizer, max_length)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=data_collator,
        pin_memory=True,
    )

    first_batch = next(iter(train_dataloader))
    print("Sanity check - first batch:")
    for k, v in first_batch.items():
        print(f"{k}: shape={tuple(v.shape)}, "
            f"min={v.min().item()}, max={v.max().item()}"
            )
    
    if first_batch["input_ids"].max().item() >= vocab_size:
        print(f"[WARN] input_ids >= vocab_size({vocab_size}) detected in first batch.")
    
    if "labels" in first_batch:
        pos_labels = first_batch["labels"][first_batch["labels"] >= 0]
        if pos_labels.numel() > 0 and pos_labels.max().item() >= vocab_size:
            print(f"[WARN] labels >= vocab_size({vocab_size}) detected in first batch.")
    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    max_train_steps = num_epochs * num_update_steps_per_epoch
    num_warmup_steps = int(max_train_steps * warmup_ratio)

    def get_lr(step):
        if step < num_warmup_steps:
            return learning_rate * float(step) / float(max(1, num_warmup_steps))
        return learning_rate * max(
            0.0,
            float(max_train_steps - step) / float(max(1, max_train_steps - num_warmup_steps)),
        )

    global_step = 0

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_steps = 0

        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for step, batch in enumerate(pbar):
            ids = batch["input_ids"]
            over = ids >= vocab_size
            if over.any():
                unk_id = tokenizer.unk_token_id
                if unk_id is None:
                    unk_id = tokenizer.pad_token_id
                    if unk_id is None:
                        unk_id = 0
                ids[over] = unk_id
            batch["input_ids"] = ids

            if "labels" in batch:
                labels = batch["labels"]
                mask = labels >= vocab_size
                if mask.any():
                    labels[mask] = -100
                batch["labels"] = labels
            
            batch = {k: v.to(device) for k, v in batch.items()}

            lr = get_lr(global_step)
            for param_group in optimizer.grouped_parameters if hasattr(optimizer, "grouped_parameters") else optimizer.param_groups:
                param_group["lr"] = lr
            
            outputs = model(**batch)
            loss = outputs.loss

            loss = loss / gradient_accumulation_steps
            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                total_loss += loss.item()
                total_steps += 1

                if global_step % 100 == 0:
                    avg_loss = total_loss / max(1, total_steps)
                    print(
                        f"Epoch [{epoch+1}/{num_epochs}], "
                        f"Step [{global_step}], "
                        f"LR: {lr:.6f}, "
                        f"Loss: {loss.item():.4f}, "
                        f"Avg Loss: {avg_loss:.4f}"
                    )
        
        print(f"Epoch {epoch+1} completed.")
        epoch_dir = os.path.join(output_dir, f"epoch_{epoch+1}")
        os.makedirs(epoch_dir, exist_ok=True)
        model.save_pretrained(epoch_dir)
        tokenizer.save_pretrained(epoch_dir)
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Training completed and model saved.")

if __name__ == "__main__":
    main()