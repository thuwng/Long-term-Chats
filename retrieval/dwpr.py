"""
Section 3.3.2 - Dynamic Weighted PageRank (DW-PR), Equations 1-3.

Edge weight w(u -> v) (Eq. 1):
  - u = entity, v = turn        : sim(q, e.desc)
  - u --r--> v (entity-entity)  : sim(q, r.desc)
  - u = turn,   v = segment     : mean_{e in tau} sim(q, e.desc)

seed(v) = sim(q, v's own embedding: desc / text / summary)

PR_{t+1}(v) = (1-d) * seed(v) + d * S(v)                          (Eq. 2)
S(v) = sum_{u->v} [ w(u->v) / sum_{u->*} w(u->*) ] * PR_t(u)      (Eq. 3)
"""
from typing import Dict

import networkx as nx
import numpy as np

from utils import cosine_sim


def _entity_neighbors_of_turn(G: nx.MultiDiGraph, turn_node: str):
    for _, v, data in G.out_edges(turn_node, data=True):
        if data.get("etype") == "entity_turn" and G.nodes[v].get("type") == "entity":
            yield v


def compute_edge_weight(G: nx.MultiDiGraph, u: str, v: str, data: dict, q_emb: np.ndarray) -> float:
    """Implements the 3 cases of Eq. 1 based on edge type / node types."""
    u_type = G.nodes[u].get("type")
    v_type = G.nodes[v].get("type")
    etype = data.get("etype")

    if etype == "entity_turn" and u_type == "entity" and v_type == "turn":
        return max(cosine_sim(q_emb, G.nodes[u]["emb"]), 0.0)

    if etype == "rel" and u_type == "entity" and v_type == "entity":
        return max(cosine_sim(q_emb, data["emb"]), 0.0)

    if etype == "turn_segment" and u_type == "turn" and v_type == "segment":
        ent_sims = [
            cosine_sim(q_emb, G.nodes[e]["emb"]) for e in _entity_neighbors_of_turn(G, u)
        ]
        return max(float(np.mean(ent_sims)), 0.0) if ent_sims else 1e-6

    # Fallback for edge directions/types not covered by Eq. 1 (e.g. the
    # reverse turn<-segment or entity<-turn edges we added for traversal
    # convenience): small constant weight so the graph stays connected
    # without dominating the query-conditioned signal.
    return 1e-6


def dynamic_weighted_pagerank(Gq: nx.MultiDiGraph, q_emb: np.ndarray,
                               d: float = 0.85, iters: int = 20,
                               eps: float = 1e-6, uniform: bool = False) -> Dict[str, float]:
    """
    Runs DW-PR (or, with uniform=True, the 'Uniform (w=1)' ablation from
    Table 5) over the given subgraph. Returns {node_id: score}.
    """
    nodes = list(Gq.nodes())
    if not nodes:
        print("⚠️ [DW-PR] Subgraph rỗng (0 nodes), không thể chạy PageRank.")
        return {}
    
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    seed = np.array([max(cosine_sim(q_emb, Gq.nodes[v]["emb"]), 1e-8) for v in nodes])
    seed_sum = seed.sum()
    seed = seed / seed_sum if seed_sum > 0 else np.full(n, 1.0 / n)

    W = np.zeros((n, n))
    for u, v, data in Gq.edges(data=True):
        w = 1.0 if uniform else compute_edge_weight(Gq, u, v, data, q_emb)
        W[idx[u], idx[v]] += w

    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    W_norm = W / row_sums

    pr = seed.copy()
    for _ in range(iters):
        new_pr = (1 - d) * seed + d * (W_norm.T @ pr)
        if np.abs(new_pr - pr).sum() < eps:
            pr = new_pr
            break
        pr = new_pr

    scores = {nodes[i]: float(pr[i]) for i in range(n)}
    
    return scores