"""
Retrieval evaluation: Recall@k (k in {3,5,10}) at both session-level and
turn-level granularity, matching Tables 2 and 3.

Recall@k = |retrieved_top_k ∩ gold| / |gold|   (0 if gold is empty -> skipped)
"""
from typing import List, Dict
import numpy as np


def recall_at_k(retrieved_ranked: List[str], gold: List[str], k: int) -> float:
    if not gold:
        return None  # caller should skip QA pairs with no gold labels
    top_k = set(retrieved_ranked[:k])
    hit = len(top_k.intersection(set(gold)))
    return hit / len(set(gold))


def evaluate_retrieval(all_retrieved_ranked: List[List[str]], all_gold: List[List[str]],
                        ks=(3, 5, 10)) -> Dict[str, float]:
    """
    all_retrieved_ranked[i]: ranked list of retrieved turn_ids or session_ids for QA i
    all_gold[i]: gold turn_ids or session_ids for QA i
    Returns {"R@3": x, "R@5": y, "R@10": z} averaged over QA pairs with
    non-empty gold sets (matches standard practice for these benchmarks).
    """
    results = {}
    for k in ks:
        scores = [
            recall_at_k(ret, gold, k)
            for ret, gold in zip(all_retrieved_ranked, all_gold)
            if gold
        ]
        results[f"R@{k}"] = float(np.mean(scores)) * 100 if scores else float("nan")
    return results


def rank_nodes_by_score(pr_scores: Dict[str, float], node_ids: List[str]) -> List[str]:
    """Utility: sort a list of node ids by descending PageRank score."""
    return sorted(node_ids, key=lambda n: pr_scores.get(n, 0.0), reverse=True)
