"""
Embedding model wrapper. Default: facebook/contriever, matching the paper's
choice ("Memory embeddings are generated with Contriever to ensure fair
comparison with prior work", §4.1). Swap `model_name` in config to try
BGE-M3, MPNet, etc. for the cross-backbone / cross-embedding ablations.
"""
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from config import EmbeddingConfig
from utils import cosine_sim  # noqa: F401  (re-exported for backward compatibility)


def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * mask, dim=1)
    counted = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counted


class EmbeddingModel:
    def __init__(self, cfg: EmbeddingConfig):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model = AutoModel.from_pretrained(cfg.model_name).to(cfg.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts, batch_size: int = None) -> np.ndarray:
        """Encode a string or list of strings into L2-normalized vectors."""
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        bs = batch_size or self.cfg.batch_size

        all_vecs = []
        for i in range(0, len(texts), bs):
            batch = texts[i:i + bs]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=self.cfg.max_length, return_tensors="pt",
            ).to(self.cfg.device)
            outputs = self.model(**inputs)
            pooled = mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            all_vecs.append(pooled.cpu().numpy())

        vecs = np.concatenate(all_vecs, axis=0)
        return vecs[0] if single else vecs
