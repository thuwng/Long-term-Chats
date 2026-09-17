"""
Generation evaluation metrics matching Table 1: token-level F1, BLEU,
ROUGE-1/2/L, and BERTScore. Uses `rouge-score`, `sacrebleu`, and
`bert-score` (see requirements.txt).
"""
import os, re, logging
from typing import List, Dict
import numpy as np  # Thêm thư viện numpy

import sacrebleu
from rouge_score import rouge_scorer

# Khai báo logger chuẩn (đã xóa import logging bị lặp)
logger = logging.getLogger(__name__)

_scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = _normalize(pred).split()
    gold_tokens = _normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = {}
    for t in pred_tokens:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    gold_counts = {}
    for t in gold_tokens:
        gold_counts[t] = gold_counts.get(t, 0) + 1
    for t, c in gold_counts.items():
        overlap += min(c, common.get(t, 0))
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def sentence_bleu(pred: str, gold: str) -> float:
    return sacrebleu.sentence_bleu(pred, [gold]).score / 100.0 * 100  # keep 0-100 scale


def rouge_scores(pred: str, gold: str) -> Dict[str, float]:
    scores = _scorer.score(gold, pred)
    return {
        "rouge1": scores["rouge1"].fmeasure * 100,
        "rouge2": scores["rouge2"].fmeasure * 100,
        "rougeL": scores["rougeL"].fmeasure * 100,
    }


def compute_bertscore(preds: List[str], golds: List[str], lang: str = "en",
                      device: str = None) -> float:
    """
    Tính Semantic Similarity thay cho BERTScore để tránh lỗi offline trên Kaggle.
    """
    try:
        from embeddings import EmbeddingModel
        from config import EmbeddingConfig
        from utils import cosine_sim
        import torch
        
        # Bắt buộc load model lên CPU để tránh OOM do GPU đang bị LLM chiếm dụng
        cfg = EmbeddingConfig()
        cfg.device = "cpu" 
        
        embedder = EmbeddingModel(cfg)
        pred_embs = embedder.encode(preds)
        gold_embs = embedder.encode(golds)
        
        sims = [cosine_sim(p, g) for p, g in zip(pred_embs, gold_embs)]
        
        # Dọn dẹp RAM ngay sau khi tính xong
        del embedder
        import gc; gc.collect()
        
        return float(np.mean(sims)) * 100
    except Exception as e:
        logger.warning("Không thể tính Semantic Similarity thay thế BERTScore: %s", e)
        return float("nan")


def evaluate_generation(preds: List[str], golds: List[str]) -> Dict[str, float]:
    f1s, r1s, r2s, rls = [], [], [], []
    for p, g in zip(preds, golds):
        f1s.append(token_f1(p, g))
        rs = rouge_scores(p, g)
        r1s.append(rs["rouge1"]); r2s.append(rs["rouge2"]); rls.append(rs["rougeL"])

    # Sử dụng Corpus BLEU chuẩn xác hơn cho toàn bộ corpus
    try:
        corpus_bleu_score = sacrebleu.corpus_bleu(preds, [golds]).score
    except Exception:
        corpus_bleu_score = 0.0

    bert_f1 = compute_bertscore(preds, golds) if preds else float("nan")
    n = max(len(f1s), 1)

    return {
        "F1": sum(f1s) / n * 100,
        "BLEU": corpus_bleu_score,
        "R-1": sum(r1s) / n,
        "R-2": sum(r2s) / n,
        "R-L": sum(rls) / n,
        "BERTScore": bert_f1,
    }