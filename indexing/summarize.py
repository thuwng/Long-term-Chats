"""
Phase 1.3 - Segment-level summary generation (global contextual anchor).
Paper reference: Section 3.1, prompt Appendix C.3.

Generated over the FULL segment (not just the filtered subset) to preserve
global context from discarded messages, per the dual-layer compression
design (retains M_i for fine-grained content + sigma_i for global anchor).
"""
from typing import List, Dict, Any

from prompts import load_prompt


def _format_segment(turns: List[Dict[str, Any]]) -> str:
    return "\n".join(f"{t['speaker']}: {t['text']}" for t in turns)


def summarize_segment(llm, full_segment_turns: List[Dict[str, Any]]) -> str:
    """Returns a 2-3 sentence summary sigma_i of the whole (unfiltered) segment."""
    if not full_segment_turns:
        return ""

    prompt_template = load_prompt("c3_summarize")
    prompt = prompt_template.format(segment_content=_format_segment(full_segment_turns))
    return llm.generate(prompt).strip()
