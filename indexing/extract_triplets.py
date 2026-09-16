"""
Phase 2 - Provenance-Enriched Graph Construction: entity-relation triplet
extraction. Paper reference: Section 3.2, prompt Appendix C.6.

Output line format from the LLM: entity1|relation|entity2|message_indices
`message_indices` are indices into the *filtered segment turn list* passed
in, which we map back to true turn_ids for provenance (source_turns).
"""
from typing import List, Dict, Any, Tuple

from prompts import load_prompt
from utils import parse_pipe_lines


def _format_segment_for_extraction(turns: List[Dict[str, Any]]) -> str:
    return "\n".join(f"Message {i}: {t['speaker']}: {t['text']}" for i, t in enumerate(turns))


def extract_triplets(llm, filtered_turns: List[Dict[str, Any]]
                      ) -> Tuple[List[Dict[str, Any]], int]:
    """
    Returns (triplets, n_parse_errors) where each triplet is:
      {"head": str, "relation": str, "tail": str, "source_turns": [turn_id, ...]}
    `n_parse_errors` feeds the structured-output-error-rate metric (Table 14).
    """
    if not filtered_turns:
        return [], 0

    prompt_template = load_prompt("c6_triplet_extraction")
    prompt = prompt_template.format(segment_text=_format_segment_for_extraction(filtered_turns))
    raw_output = llm.generate(prompt)

    rows, n_errors = parse_pipe_lines(raw_output, expected_fields=4)

    triplets = []
    for head, relation, tail, msg_indices_str in rows:
        msg_indices = _parse_indices(msg_indices_str)
        source_turns = [
            filtered_turns[i]["turn_id"] for i in msg_indices if 0 <= i < len(filtered_turns)
        ]
        if not head or not relation or not tail:
            n_errors += 1
            continue
        triplets.append({
            "head": head, "relation": relation, "tail": tail,
            "source_turns": source_turns or [t["turn_id"] for t in filtered_turns],
        })

    return triplets, n_errors


def _parse_indices(s: str) -> List[int]:
    out = []
    for part in s.replace(" ", "").split(","):
        if part.isdigit():
            out.append(int(part))
    return out
