import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM, AutoTokenizer
from typing import Optional

# Hugging Face NT500M 모델명
MODEL_NAME = "InstaDeepAI/nucleotide-transformer-500m-1000g"

# 출력 차원 (Dacon sample_submission 기준)
OUT_EMB_DIM = 768


class DNAEmbeddingModel(nn.Module):
    """
    개선 버전:
      - 마지막 4개 레이어 평균 (last 4 layers mean)
      - CLS token embedding
      - Mean pooling embedding
      - CLS + mean pooling 결합 (concat → projection)

    최종 출력: 768-d vector
    """

    def __init__(self, base_model, emb_dim: int = OUT_EMB_DIM):
        super().__init__()
        self.base_model = base_model
        hidden_size = base_model.config.hidden_size
        self.hidden_size = hidden_size

        # Projection: CLS(768) + mean(768) -> 1536 -> 768
        self.proj = nn.Linear(hidden_size * 2, emb_dim)

    @classmethod
    def from_pretrained(cls, model_name: str = MODEL_NAME, device: str = "cuda"):
        base = AutoModelForMaskedLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            output_hidden_states=True
        )
        model = cls(base)
        return model.to(device)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # hidden_states = tuple: (layer0,...,layerN)
        hidden_states = outputs.hidden_states

        # ---- (1) last 4 layers mean pooling across layers ----
        last4 = torch.stack(
            [hidden_states[-1], hidden_states[-2], hidden_states[-3], hidden_states[-4]],
            dim=0
        )  # shape: (4, B, L, H)
        last4_mean = last4.mean(0)            # (B, L, H)

        # ---- (2) CLS embedding ----
        cls_vec = last4_mean[:, 0, :]         # (B, H)

        # ---- (3) Mean pooling over non-padding tokens ----
        mask = attention_mask.unsqueeze(-1)   # (B, L, 1)
        mean_vec = (last4_mean * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)

        # ---- (4) concat → projection ----
        combo = torch.cat([cls_vec, mean_vec], dim=-1)  # (B, 2H)
        final_emb = self.proj(combo)                    # (B, OUT_EMB_DIM)

        return final_emb


def load_tokenizer(model_name: str = MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    return tokenizer
