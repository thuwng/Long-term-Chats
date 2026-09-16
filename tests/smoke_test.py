"""
Lightweight smoke test: exercises the full pipeline (segmentation -> filter
-> summarize -> triplet/entity extraction -> graph build -> seeding -> DWPR
-> retrieve -> generate -> eval) using a scripted fake LLM and a random
hash-based fake embedder, so it runs with zero external dependencies
(no GPU, no API keys, no torch/transformers download).

Run: python tests/smoke_test.py
"""
import sys
import os
import json
import hashlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import AblationConfig, RetrievalConfig
from indexing.build_graph import build_conversation_graph, graph_complexity_stats
from retrieval.seeding import get_seed_subgraph
from retrieval.dwpr import dynamic_weighted_pagerank
from retrieval.retrieve import select_top_turns, enrich_with_triplets, format_context
from generation.generate import generate_answer
from eval.recall import evaluate_retrieval
from eval.lexical_metrics import token_f1


class FakeEmbedder:
    """Deterministic pseudo-embeddings so cosine similarity is stable across calls."""
    dim = 32

    def encode(self, text):
        if isinstance(text, list):
            return np.stack([self._encode_one(t) for t in text])
        return self._encode_one(text)

    def _encode_one(self, text):
        h = hashlib.sha256(text.encode("utf-8")).digest()
        vec = np.frombuffer(h, dtype=np.uint8).astype(np.float32)[: self.dim]
        vec = vec - vec.mean()
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


class FakeLLM:
    """
    Scripted responses keyed by which prompt template is being used
    (detected via distinctive substrings), so the pipeline exercises real
    parsing logic (JSON / pipe-delimited) end to end.
    """

    def generate(self, prompt, **kwargs):
        if "TASK DEFINITION" in prompt and "topic segmentation" in prompt:
            return "[[0,1],[2,3]]"
        if "Data Curator" in prompt:
            return "[0,1,2,3]"
        if "Summarize the following conversation segment" in prompt:
            return "The user talked about their job and their pet."
        if "knowledge graph extractor" in prompt:
            return (
                "User|works at|Acme Corp|0\n"
                "User|has pet|Luna|2\n"
            )
        if "provide a brief description for each entity" in prompt:
            return (
                "User | Works at Acme Corp as an engineer | 0\n"
                "Acme Corp | The user's employer | 0\n"
                "Luna | The user's pet cat | 2\n"
            )
        if "Based on the provided conversation context" in prompt:
            return "Acme Corp"
        raise ValueError(f"FakeLLM got an unexpected prompt:\n{prompt[:200]}")


def build_fake_conversation():
    turns = [
        {"turn_id": "t0", "speaker": "User", "text": "I work at Acme Corp as an engineer.",
         "session_id": "s0"},
        {"turn_id": "t1", "speaker": "Assistant", "text": "That's great, how long have you been there?",
         "session_id": "s0"},
        {"turn_id": "t2", "speaker": "User", "text": "I also have a pet cat named Luna.",
         "session_id": "s1"},
        {"turn_id": "t3", "speaker": "Assistant", "text": "Cats are wonderful companions.",
         "session_id": "s1"},
    ]
    qas = [{
        "question": "Where does the user work?",
        "answer": "Acme Corp",
        "gold_turn_ids": ["t0"],
        "gold_session_ids": ["s0"],
    }]
    return {"conv_id": "fake_conv_1", "turns": turns, "qas": qas}


def main():
    llm = FakeLLM()
    embedder = FakeEmbedder()
    ablation = AblationConfig()
    retrieval_cfg = RetrievalConfig()

    conv = build_fake_conversation()

    print(">> Phase 1+2: building graph...")
    G = build_conversation_graph(llm, embedder, conv["conv_id"], conv["turns"], ablation)
    stats = graph_complexity_stats(G)
    print(f"   graph stats: {stats}")
    assert stats["n_nodes"] > 0, "graph should not be empty"

    node_types = {}
    for _, data in G.nodes(data=True):
        node_types[data["type"]] = node_types.get(data["type"], 0) + 1
    print(f"   node type counts: {node_types}")
    assert set(node_types) == {"segment", "turn", "entity"}, "missing a node type"

    qa = conv["qas"][0]
    q_emb = embedder.encode(qa["question"])

    print(">> Phase 3.1: seeding subgraph...")
    Gq = get_seed_subgraph(G, q_emb, k=retrieval_cfg.top_k_seed)
    print(f"   subgraph nodes: {Gq.number_of_nodes()}, edges: {Gq.number_of_edges()}")
    assert Gq.number_of_nodes() > 0

    print(">> Phase 3.2: Dynamic Weighted PageRank...")
    pr_scores = dynamic_weighted_pagerank(Gq, q_emb, d=retrieval_cfg.damping,
                                           iters=retrieval_cfg.pagerank_iters)
    assert len(pr_scores) == Gq.number_of_nodes()
    assert all(np.isfinite(v) for v in pr_scores.values())

    print(">> Phase 3.3: top-m turns + triplet enrichment...")
    top_turns = select_top_turns(Gq, pr_scores, m=retrieval_cfg.top_m_turns)
    triplets = enrich_with_triplets(Gq, top_turns)
    context = format_context(Gq, top_turns, triplets, use_triplets=True)
    print(f"   top turns: {[Gq.nodes[n]['turn_id'] for n in top_turns]}")
    print(f"   triplets: {triplets}")
    print(f"   context:\n{context}")

    print(">> Generation...")
    answer = generate_answer(llm, context, qa["question"])
    print(f"   answer: {answer}")
    f1 = token_f1(answer, qa["answer"])
    print(f"   token F1 vs reference: {f1:.2f}")
    assert f1 > 0.5, "sanity check: fake pipeline should retrieve the right fact"

    print(">> Retrieval eval sanity check (Recall@k)...")
    all_turn_nodes_ranked = select_top_turns(Gq, pr_scores, m=len(Gq.nodes))
    ranked_turn_ids = [Gq.nodes[n]["turn_id"] for n in all_turn_nodes_ranked]
    metrics = evaluate_retrieval([ranked_turn_ids], [qa["gold_turn_ids"]], ks=(3, 5, 10))
    print(f"   turn-level recall: {metrics}")

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
