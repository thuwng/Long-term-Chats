"""
Phase 1.2 - Selective Memory Filtering (the "memory gate").
Paper reference: Section 3.1, prompt Appendix C.2.

For each segment, retains only messages containing user-specific episodic
content (personal facts, preferences, commitments, identity markers),
producing a filtered set M_i subseteq S_i.
"""
from typing import List, Dict, Any

from prompts import load_prompt
from utils import safe_json_loads
from llm_client import scaled_max_tokens


def _format_conv(turns: List[Dict[str, Any]]) -> str:
    return "\n".join(f"[{i}] {t['speaker']}: {t['text']}" for i, t in enumerate(turns))


def selective_filter(llm, segment_turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Returns the subset of `segment_turns` deemed user-relevant."""
    if not segment_turns:
        return []

    prompt_template = load_prompt("c2_selective_filter")
    prompt = prompt_template.format(formatted_conv=_format_conv(segment_turns))

    cap = getattr(getattr(llm, "cfg", None), "max_tokens_filter_cap", 4000)
    max_tokens = scaled_max_tokens(len(segment_turns), per_item=4, base=200, cap=cap)
    raw_output = llm.generate(prompt, max_tokens=max_tokens)
    keep_indices = safe_json_loads(raw_output, default=None)

    if not isinstance(keep_indices, list) or not all(isinstance(i, int) for i in keep_indices):
        # Malformed output -> fail open (keep everything) rather than
        # silently dropping the whole segment's content.
        return segment_turns

    kept = [segment_turns[i] for i in keep_indices if 0 <= i < len(segment_turns)]
    return kept


def no_selective_filter_baseline(segment_turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ablation: 'w/o Selective' - keep all turns in the segment."""
    return segment_turns
