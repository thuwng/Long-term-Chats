"""
Section 3.3.1 - Query-Focused Subgraph Retrieval.

Multi-aspect parallel retrieval identifies top-k seed nodes:
  - segment nodes (via summary embeddings)
  - entity nodes (via description embeddings)
  - relation edges (via triplet/relation description embeddings)
followed by one-hop neighborhood expansion to include all directly
connected turns, entities, and segments -> sparse query-focused
subgraph G_q = (V_q, E_q).
"""
from typing import List, Tuple

import networkx as nx
import numpy as np

from utils import cosine_sim


def _top_k_nodes(G, node_type, q_emb, k, degree_penalty=True):
    scored = []
    for n, data in G.nodes(data=True):
        if data.get("type") != node_type: 
            continue
            
        sim = cosine_sim(q_emb, data["emb"])
        
        # THÊM LOGIC PHẠT HUB ENTITY
        if degree_penalty and node_type == "entity":
            deg = G.out_degree(n)  # Số lượng edge (turn) nối tới
            if deg > 0:
                sim = sim / np.log(deg + np.e)   # Giảm điểm các hub xuất hiện quá nhiều
                
        scored.append((n, sim))
        
    scored.sort(key=lambda x: x[1], reverse=True)
    return [n for n, _ in scored[:k]]


def _top_k_relation_edges(G: nx.MultiDiGraph, q_emb: np.ndarray, k: int
                           ) -> List[Tuple[str, str, int]]:
    scored = []
    for u, v, key, data in G.edges(keys=True, data=True):
        if data.get("etype") == "rel":
            scored.append(((u, v, key), cosine_sim(q_emb, data["emb"])))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [edge for edge, _ in scored[:k]]


def get_seed_subgraph(G: nx.MultiDiGraph, q_emb: np.ndarray, k: int = 3) -> nx.MultiDiGraph:
    """
    Returns the query-focused subgraph G_q: seed nodes (segments, entities,
    relation endpoints) plus their one-hop neighborhood (turns, entities,
    segments directly connected to the seeds).
    """
    seed_segments = _top_k_nodes(G, "segment", q_emb, k)
    seed_entities = _top_k_nodes(G, "entity", q_emb, k)
    seed_relation_edges = _top_k_relation_edges(G, q_emb, k)

    seed_nodes = set(seed_segments) | set(seed_entities)
    for u, v, _key in seed_relation_edges:
        seed_nodes.add(u)
        seed_nodes.add(v)

    
    for es in seed_entities:
        e_data = G.nodes[es]

    if not seed_nodes:
        print("⚠️ [Seeding] Không tìm thấy seed node nào phù hợp (Graph trống hoặc cold start)!")
        return nx.MultiDiGraph()

    expanded = set(seed_nodes)
    for n in seed_nodes:
        expanded.update(G.successors(n))
        expanded.update(G.predecessors(n))

    Gq = G.subgraph(expanded).copy()
    Gq.graph["seed_nodes"] = seed_nodes
    
    return Gq


def get_full_graph_as_subgraph(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Ablation: 'Full Graph' - skip subgraph retrieval, rank over the entire graph."""
    Gq = G.copy()
    Gq.graph["seed_nodes"] = set(G.nodes())
    
    return Gq