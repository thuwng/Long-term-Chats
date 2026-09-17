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
                
    # fallback 2: Xử lý lỗi LLM sinh JSON bị cắt cụt giữa chừng (hết max_tokens).
    # Hỗ trợ cả JSON lồng nhau (vd. list-of-lists của C.1 segmentation), không
    # chỉ list phẳng: cắt lùi tới dấu phẩy hợp lệ cuối cùng NGOÀI mọi chuỗi,
    # rồi đóng lại đúng số ngoặc [ ] { } còn đang mở theo thứ tự LIFO.
    repaired = _repair_truncated_json(cleaned)
    if repaired is not None:
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

    return default if default is not None else []


def _repair_truncated_json(text: str):
    """
    Best-effort repair for JSON truncated mid-stream (ran out of max_tokens).
    Walks the string tracking an open-bracket stack (ignoring brackets inside
    string literals), trims back to the last safely-closable point, and
    appends the missing closing brackets in the correct order. Returns None
    if the text doesn't look like a JSON array/object at all.
    """
    text = text.strip()
    if not text or text[0] not in "[{":
        return None

    stack = []
    in_string = False
    escape = False
    last_safe_idx = -1  # index right after the last comma/opening bracket at depth-consistent point

    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch in "[{":
            stack.append(ch)
        elif ch in "]}":
            if stack:
                stack.pop()
        elif ch == "," and stack:
            last_safe_idx = i  # safe to cut right before this comma

    if not stack:
        return None  # already balanced; the earlier json.loads would have worked

    # Cut off any trailing incomplete element (after the last safe comma),
    # unless the whole thing is still balanced without cutting (e.g. it was
    # only missing closing brackets, no dangling partial element).
    trimmed = text if last_safe_idx == -1 else text[:last_safe_idx]

    # Recompute the open-bracket stack for the trimmed text.
    stack2 = []
    in_string = False
    escape = False
    for ch in trimmed:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "[{":
            stack2.append(ch)
        elif ch in "]}":
            if stack2:
                stack2.pop()

    closers = {"[": "]", "{": "}"}
    return trimmed + "".join(closers[b] for b in reversed(stack2))


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