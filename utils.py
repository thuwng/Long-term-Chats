"""Shared helpers: robust JSON extraction from LLM text, IO utilities."""
import json
import os
import re
from typing import Any

import numpy as np


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors (safe against zero norms)."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def strip_code_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers some LLMs add."""
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def safe_json_loads(text: str, default: Any = None) -> Any:
    """Parse JSON robustly: strips fences, falls back to first [...]/{...} match, and fixes truncation."""
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
                
    # fallback 2: Xử lý lỗi LLM sinh mảng JSON bị cắt cụt (thiếu dấu ']') ở cuối
    if cleaned.startswith("[") and not cleaned.endswith("]"):
        # Cắt lùi đến dấu phẩy cuối cùng và đóng mảng
        fixed_cleaned = cleaned.rsplit(",", 1)[0] + "]"
        try:
            return json.loads(fixed_cleaned)
        except json.JSONDecodeError:
            pass

    return default if default is not None else []


def parse_pipe_lines(text: str, expected_fields: int):
    """
    Parse '|' delimited output lines. 
    Tolerates Markdown tables, list bullets, and conversational LLM fillers.
    """
    rows, n_errors = [], 0
    text = strip_code_fences(text)
    
    for line in text.strip().splitlines():
        line = line.strip()
        
        # Bỏ qua các dòng trống, đường kẻ ngang của bảng Markdown hoặc các câu mào đầu của LLM
        if not line or line.startswith("---") or line.startswith("==="):
            continue
        if line.lower().startswith(("output", "entity1|relation", "entity |", "here are", "entity 1 |")):
            continue
            
        # Xóa các gạch đầu dòng (bullet points) của Markdown (- / * / 1.)
        line = re.sub(r"^[-*\d+.]\s+", "", line)
        
        # Xóa các cạnh viền của bảng Markdown nếu có (vd: | Entity | Desc | Index |)
        if line.startswith("|"): 
            line = line[1:]
        if line.endswith("|"): 
            line = line[:-1]
            
        line = line.strip()
        
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