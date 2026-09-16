"""Loads prompt templates copied verbatim from Appendix C of the MemORAI paper."""
import os

_PROMPTS_DIR = os.path.dirname(os.path.abspath(__file__))

_FILES = {
    "c1_segmentation": "c1_segmentation.txt",
    "c2_selective_filter": "c2_selective_filter.txt",
    "c3_summarize": "c3_summarize.txt",
    "c4_entity_description": "c4_entity_description.txt",
    "c5_answer_generation": "c5_answer_generation.txt",
    "c6_triplet_extraction": "c6_triplet_extraction.txt",
    "c7_gpt4_judge": "c7_gpt4_judge.txt",
}

_cache = {}


def load_prompt(name: str) -> str:
    if name not in _FILES:
        raise KeyError(f"Unknown prompt '{name}'. Available: {list(_FILES)}")
    if name not in _cache:
        path = os.path.join(_PROMPTS_DIR, _FILES[name])
        with open(path, "r", encoding="utf-8") as f:
            _cache[name] = f.read()
    return _cache[name]
