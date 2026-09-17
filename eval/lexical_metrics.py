"""
Generation evaluation metrics matching Table 1: token-level F1, BLEU,
ROUGE-1/2/L, and BERTScore. Uses `rouge-score`, `sacrebleu`, and
`bert-score` (see requirements.txt).
"""
import os, re, logging
from typing import List, Dict

import sacrebleu
from rouge_score import rouge_scorer


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
    """Batched BERTScore (F1), returns the corpus-level average * 100.

    Imports `bert_score` lazily (it pulls in torch/transformers) so that
    modules/tests only needing token_f1/BLEU/ROUGE don't pay that cost.
    """
    logger = logging.getLogger(__name__)
    if device is None:
        device = os.environ.get("MEMORAI_BERTSCORE_DEVICE", "cpu")

    from bert_score import score as bertscore_score
    try:
        P, R, F1 = bertscore_score(preds, golds, lang=lang, verbose=False, device=device)
        return float(F1.mean()) * 100
    except Exception as e:  # noqa: BLE001
        # Previously this silently swallowed the traceback, which is why a
        # 100% BERTScore failure rate (both memorai and dense_baseline
        # returning NaN in a full run) went unnoticed. Log the full
        # traceback at ERROR level so it's impossible to miss in Kaggle's
        # cell output; still return NaN so a flaky BERTScore run doesn't
        # crash the whole pipeline and lose the other metrics/records.
        logger.error(
            "BERTScore failed (device=%s, n_preds=%d): %r. Common causes on "
            "Kaggle: Internet is OFF (can't download the roberta-large "
            "baseline model), or a CUDA OOM. Returning NaN for BERTScore "
            "only - all other metrics/records are unaffected.",
            device, len(preds), e, exc_info=True,
        )
        return float("nan")


def evaluate_generation(preds: List[str], golds: List[str]) -> Dict[str, float]:
    f1s, r1s, r2s, rls = [], [], [], []
    for p, g in zip(preds, golds):
        f1s.append(token_f1(p, g))
        rs = rouge_scores(p, g)
        r1s.append(rs["rouge1"]); r2s.append(rs["rouge2"]); rls.append(rs["rougeL"])

    # Sử dụng Corpus BLEU chuẩn xác hơn cho toàn bộ corpus thay vì trung bình sentence BLEU
    corpus_bleu_score = sacrebleu.corpus_bleu(preds, [golds]).score

    bert_f1 = compute_bertscore(preds, golds) if preds else float("nan")
    n = max(len(f1s), 1)

    return {
        "F1": sum(f1s) / n * 100,
        "BLEU": corpus_bleu_score,  # Đã là thang điểm 0-100 từ sacrebleu
        "R-1": sum(r1s) / n,
        "R-2": sum(r2s) / n,
        "R-L": sum(rls) / n,
        "BERTScore": bert_f1,
    }
