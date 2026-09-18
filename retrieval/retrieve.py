"""
Section 3.3.2 (end) / 3.3.3 - top-m turn selection after DW-PR convergence,
plus triplet enrichment: for each retrieved turn, include all
entity-relation triplets whose source_turns cite it. Formats the final
provenance-aware context passed to generation (Appendix C.5).
"""
from typing import Dict, List, Tuple

import networkx as nx


def select_top_turns(Gq: nx.MultiDiGraph, pr_scores: Dict[str, float], m: int = 3) -> List[str]:
    """Ranks turn nodes by final DW-PR score, returns top-m turn node ids."""
    turn_nodes = [n for n in Gq.nodes if Gq.nodes[n].get("type") == "turn"]
    turn_nodes.sort(key=lambda n: pr_scores.get(n, 0.0), reverse=True)
    top_m = turn_nodes[:m]
    for n in top_m:
        data = Gq.nodes[n]
    return top_m


def enrich_with_triplets(Gq: nx.MultiDiGraph, top_turn_nodes: List[str]
                          ) -> List[Tuple[str, str, str]]:
    """
    Ablation-aware: paper §3.3 - 'For each retrieved turn, we also include
    all entity-relation triplets that cite it (source_turns).'
    Returns list of (head, relation, tail) triples.
    """
    cited_turn_ids = {Gq.nodes[n]["turn_id"] for n in top_turn_nodes}
    triplets = []
    seen = set()
    for u, v, data in Gq.edges(data=True):
        if data.get("etype") != "rel":
            continue
        if cited_turn_ids.intersection(data.get("source_turns", [])):
            head, tail = Gq.nodes[u]["name"], Gq.nodes[v]["name"]
            key = (head, data["relation"], tail)
            if key not in seen:
                seen.add(key)
                triplets.append(key)
                
    return triplets


def format_context(Gq: nx.MultiDiGraph, top_turn_nodes: List[str],
                    triplets: List[Tuple[str, str, str]], use_triplets: bool = True) -> str:
    """
    Builds the provenance-aware prompt context. Turns are ordered by
    (segment_idx, turn position) to keep temporal/dialogue order readable
    for the generator, matching the paper's emphasis on temporal grounding.
    """
    turns_sorted = sorted(
        top_turn_nodes,
        key=lambda n: (Gq.nodes[n].get("segment_idx", 0), Gq.nodes[n]["turn_id"]),
    )
    lines = ["Relevant conversation turns:"]
    for n in turns_sorted:
        data = Gq.nodes[n]
        date_str = f" ({data['date']})" if data.get('date') else ""
        lines.append(f"- [{data['turn_id']}{date_str}] {data['speaker']}: {data['text']}")

    if use_triplets and triplets:
        lines.append("\nSupporting facts (entity - relation - entity):")
        for h, r, t in triplets:
            lines.append(f"- ({h}, {r}, {t})")

    context_str = "\n".join(lines)
    return context_str
