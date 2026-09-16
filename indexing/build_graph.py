"""
Orchestrates Phase 1 (segmentation + selective filtering + summarization)
and Phase 2 (triplet + entity extraction) into the heterogeneous,
provenance-enriched knowledge graph G = (V, E) described in Section 3.2.

Node types: entity (VE), turn (VT), segment (VS)
Edge types: entity-relation-entity (rel), entity-turn (entity_turn),
            turn-segment (turn_segment)

Embeddings: entity <- description, turn <- text, segment <- summary,
relation <- relation string (needed for query-conditioned edge weighting
in retrieval/dwpr.py, Eq. 1).
"""
import logging
from typing import List, Dict, Any

import networkx as nx

from config import AblationConfig
from indexing.segment import segment_conversation, no_segmentation_baseline
from indexing.filter import selective_filter, no_selective_filter_baseline
from indexing.summarize import summarize_segment
from indexing.extract_triplets import extract_triplets
from indexing.extract_entities import extract_entity_descriptions, collect_entity_names

logger = logging.getLogger(__name__)


def entity_node_id(name: str) -> str:
    return f"ent::{name.strip().lower()}"


def turn_node_id(turn_id: str) -> str:
    return f"turn::{turn_id}"


def segment_node_id(conv_id: str, segment_idx: int) -> str:
    return f"seg::{conv_id}::{segment_idx}"


def build_conversation_graph(llm, embedder, conv_id: str, turns: List[Dict[str, Any]],
                              ablation: AblationConfig = None) -> nx.MultiDiGraph:
    """
    Builds the full knowledge graph for a single conversation.
    Toggle `ablation.use_topic_segmentation` / `use_selective_filtering`
    to reproduce Table 4's ablation rows.
    """
    ablation = ablation or AblationConfig()
    G = nx.MultiDiGraph()
    stats = {"structured_output_errors": 0, "structured_output_total": 0}

    # ---- Phase 1.1: segmentation ----
    if ablation.use_topic_segmentation:
        segments = segment_conversation(llm, turns)
    else:
        segments = no_segmentation_baseline(turns)

    for seg_idx, segment_turns in enumerate(segments):
        seg_node = segment_node_id(conv_id, seg_idx)

        # ---- Phase 1.2: selective filtering ----
        if ablation.use_selective_filtering:
            filtered = selective_filter(llm, segment_turns)
        else:
            filtered = no_selective_filter_baseline(segment_turns)

        # ---- Phase 1.3: segment summary (always over the FULL segment) ----
        summary = summarize_segment(llm, segment_turns)
        G.add_node(seg_node, type="segment", summary=summary, conv_id=conv_id,
                   segment_idx=seg_idx, emb=embedder.encode(summary or " "))

        if not filtered:
            continue

        # turn nodes + turn-segment edges
        for t in filtered:
            t_node = turn_node_id(t["turn_id"])
            if t_node not in G:
                G.add_node(t_node, type="turn", text=t["text"], turn_id=t["turn_id"],
                           speaker=t["speaker"], segment_idx=seg_idx, conv_id=conv_id,
                           emb=embedder.encode(t["text"] or " "))
            G.add_edge(t_node, seg_node, etype="turn_segment")
            G.add_edge(seg_node, t_node, etype="turn_segment")

        # ---- Phase 2: triplet extraction (C.6) ----
        triplets, n_err = extract_triplets(llm, filtered)
        stats["structured_output_errors"] += n_err
        stats["structured_output_total"] += n_err + len(triplets)

        entity_names = collect_entity_names(triplets)

        # ---- Phase 2: entity description extraction (C.4) ----
        entities, n_err2 = extract_entity_descriptions(llm, filtered, entity_names)
        stats["structured_output_errors"] += n_err2
        stats["structured_output_total"] += n_err2 + len(entities)

        for e in entities:
            e_node = entity_node_id(e["name"])
            if e_node not in G:
                G.add_node(e_node, type="entity", name=e["name"], description=e["description"],
                           emb=embedder.encode(e["description"] or e["name"]))
            else:
                # merge descriptions across segments (append new context)
                old = G.nodes[e_node]["description"]
                if e["description"] and e["description"] not in old:
                    merged = f"{old} {e['description']}".strip()
                    G.nodes[e_node]["description"] = merged
                    G.nodes[e_node]["emb"] = embedder.encode(merged)
            for tid in e["turn_ids"]:
                t_node = turn_node_id(tid)
                if t_node in G:
                    G.add_edge(e_node, t_node, etype="entity_turn")
                    G.add_edge(t_node, e_node, etype="entity_turn")

        for tr in triplets:
            h_node, t_node_ent = entity_node_id(tr["head"]), entity_node_id(tr["tail"])
            for node_id, name in ((h_node, tr["head"]), (t_node_ent, tr["tail"])):
                if node_id not in G:
                    G.add_node(node_id, type="entity", name=name, description=name,
                               emb=embedder.encode(name))
            G.add_edge(h_node, t_node_ent, etype="rel", relation=tr["relation"],
                       source_turns=tr["source_turns"], emb=embedder.encode(tr["relation"]))

    G.graph["stats"] = stats
    return G


def build_graph_for_dataset(llm, embedder, conversations: List[Dict[str, Any]],
                             ablation: AblationConfig = None) -> Dict[str, nx.MultiDiGraph]:
    """Builds one graph per conversation; returns {conv_id: graph}."""
    graphs = {}
    for conv in conversations:
        logger.info("Building graph for conversation %s (%d turns)",
                    conv["conv_id"], len(conv["turns"]))
        graphs[conv["conv_id"]] = build_conversation_graph(
            llm, embedder, conv["conv_id"], conv["turns"], ablation
        )
    return graphs


def graph_complexity_stats(G: nx.MultiDiGraph) -> Dict[str, int]:
    """Avg #Nodes / #Edges style stats, matching Figure 3."""
    return {"n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges()}
