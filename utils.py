"""Shared helpers: robust JSON extraction from LLM text, IO utilities."""
import json
import os
import re
from typing import Any

import numpy as np


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors (safe against zero norms).

    Lives here (not in embeddings.py) so retrieval modules don't need to
    import torch/transformers just to compute similarities.
    """
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def strip_code_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers some LLMs add."""
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def safe_json_loads(text: str, default: Any = None) -> Any:
    """Parse JSON robustly: strips fences, falls back to first [...]/{...} match."""
    cleaned = strip_code_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # fallback: find first bracketed JSON substring
    for pattern in (r"\[.*\]", r"\{.*\}"):
        m = re.search(pattern, cleaned, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                continue
    return default if default is not None else []


def parse_pipe_lines(text: str, expected_fields: int):
    """
    Parse '|' delimited output lines (used by C.4 entity and C.6 triplet
    extraction prompts). Skips malformed lines instead of raising, and
    reports the count of malformed lines for the structured-output-error
    metric described in Appendix B.2 / Table 14.
    """
    rows, n_errors = [], 0
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.lower().startswith(("output", "entity1|relation", "entity |")):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != expected_fields:
            n_errors += 1
            continue
        rows.append(parts)
    return rows, n_errors


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
