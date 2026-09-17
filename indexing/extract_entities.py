"""
Phase 2 - Provenance-Enriched Graph Construction: entity description
extraction. Paper reference: Section 3.2, prompt Appendix C.4.

Produces fine-grained natural-language descriptions per entity (shown in
the paper to preserve semantic detail better than uniform node summaries),
plus the turn_ids each description is grounded in.

Output line format from the LLM: entity | description | turn_index1,turn_index2,...
"""
from typing import List, Dict, Any, Tuple, Set

from prompts import load_prompt
from utils import parse_pipe_lines
from llm_client import scaled_max_tokens


def _format_segment_for_extraction(turns: List[Dict[str, Any]]) -> str:
    return "\n".join(f"Message {i}: {t['speaker']}: {t['text']}" for i, t in enumerate(turns))


def collect_entity_names(triplets: List[Dict[str, Any]]) -> Set[str]:
    names = set()
    for t in triplets:
        names.add(t["head"])
        names.add(t["tail"])
    return names


def extract_entity_descriptions(llm, filtered_turns: List[Dict[str, Any]],
                                 entity_names: Set[str]) -> Tuple[List[Dict[str, Any]], int]:
    """
    Returns (entities, n_parse_errors) where each entity is:
      {"name": str, "description": str, "turn_ids": [turn_id, ...]}
    """
    if not filtered_turns or not entity_names:
        return [], 0

    prompt_template = load_prompt("c4_entity_description")
    prompt = prompt_template.format(
        segment=_format_segment_for_extraction(filtered_turns),
        entity_list=", ".join(sorted(entity_names)),
    )
    cap = getattr(getattr(llm, "cfg", None), "max_tokens_extraction_cap", 4000)
    max_tokens = scaled_max_tokens(len(entity_names), per_item=20, base=200, cap=cap)
    raw_output = llm.generate(prompt, max_tokens=max_tokens)
    rows, n_errors = parse_pipe_lines(raw_output, expected_fields=3)

    entities = []
    for name, description, turn_idx_str in rows:
        turn_indices = [int(x) for x in turn_idx_str.replace(" ", "").split(",") if x.isdigit()]
        turn_ids = [filtered_turns[i]["turn_id"] for i in turn_indices if 0 <= i < len(filtered_turns)]
        if not name or not description:
            n_errors += 1
            continue
            
        if not turn_ids:
            turn_ids = [t["turn_id"] for t in filtered_turns if name.lower() in t["text"].lower()]
            
        if not turn_ids:
            n_errors += 1
            continue

        entities.append({
            "name": name,
            "description": description,
            "turn_ids": turn_ids,
        })

    covered = {e["name"] for e in entities}
    for missing in entity_names - covered:
        # Fallback tìm turn cho các entity bị sót
        fallback_turns = [t["turn_id"] for t in filtered_turns if missing.lower() in t["text"].lower()]
        if fallback_turns:
            entities.append({
                "name": missing, "description": missing,
                "turn_ids": fallback_turns,
            })

    return entities, n_errors
