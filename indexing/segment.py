"""
Phase 1.1 - Session Segmentation & Selective Compression, segmentation step.
Paper reference: Section 3.1, prompt Appendix C.1.

Decomposes raw multi-session conversation turns into semantically
coherent topical segments S_i = {t_1, ..., t_m} via LLM prompting.
"""
from typing import List, Dict, Any

from prompts import load_prompt
from utils import safe_json_loads


def _format_numbered_messages(turns: List[Dict[str, Any]]) -> str:
    lines = []
    for i, t in enumerate(turns):
        lines.append(f"Message {i}: {t['speaker']}: {t['text']}")
    return "\n".join(lines)


def segment_conversation(llm, turns: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """
    Returns a list of segments, each a list of turn dicts (in original order).
    Falls back to a single segment containing everything if the LLM output
    is malformed or fails validation, so downstream phases never crash.
    """
    if not turns:
        return []

    prompt_template = load_prompt("c1_segmentation")
    numbered = _format_numbered_messages(turns)
    prompt = prompt_template.format(numbered_messages_str=numbered)

    raw_output = llm.generate(prompt)
    segment_indices = safe_json_loads(raw_output, default=None)

    if not _is_valid_segmentation(segment_indices, len(turns)):
        # Fallback: single segment with all turns (== "w/o Topic Seg" ablation
        # behavior is actually intentionally disabled via ablation flag,
        # this fallback only triggers on genuine parse failure)
        return [turns]

    segments = []
    for idx_list in segment_indices:
        seg = [turns[i] for i in idx_list if 0 <= i < len(turns)]
        if seg:
            segments.append(seg)
    return segments if segments else [turns]


def _is_valid_segmentation(segment_indices, n_turns: int) -> bool:
    if not isinstance(segment_indices, list) or not segment_indices:
        return False
    seen = set()
    for seg in segment_indices:
        if not isinstance(seg, list):
            return False
        for i in seg:
            if not isinstance(i, int) or i in seen:
                return False
            seen.add(i)
    return True


def no_segmentation_baseline(turns: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Ablation: 'w/o Topic Seg' - treat the whole conversation as one segment."""
    return [turns] if turns else []
