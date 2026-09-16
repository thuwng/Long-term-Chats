"""
Dataset loaders for LOCOMO-10 (Maharana et al., 2024) and LongMemEval-s
(Wu et al., 2025). These datasets are NOT redistributed here - download
them yourself and point `path` at the JSON/JSONL files:

  LOCOMO-10:      https://github.com/snap-stanford/locomo
  LongMemEval-s:  https://github.com/xiaowu0162/LongMemEval

Both loaders normalize the raw format into a common internal schema:

Conversation = {
    "conv_id": str,
    "turns": [ {"turn_id": str, "speaker": str, "text": str, "date": str|None}, ... ],
    "qas": [ {
        "question": str,
        "answer": str,
        "category": str|None,
        "gold_turn_ids": [str, ...],     # turn-level retrieval ground-truth
        "gold_session_ids": [str, ...],  # session-level retrieval ground-truth
    }, ... ]
}

NOTE: exact field names in the raw dumps can change between dataset
releases. If `load_locomo` / `load_longmemeval` raise a KeyError, open one
sample and adjust the field-name constants marked "ADAPT ME" below.
"""
import json
from typing import List, Dict, Any


def _read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_locomo(path: str) -> List[Dict[str, Any]]:
    """
    Parses the backup LOCOMO release format where each row in the JSON is a QA instance
    containing conversation sessions and specific question/answer/evidence fields.
    """
    raw = _read_json(path)
    
    # Gom nhóm các QA theo conv_id vì một hội thoại có thể có nhiều câu hỏi
    conv_map = {}

    for row_idx, row in enumerate(raw):
        # 1. Parse trường 'conversation' nếu nó đang là chuỗi string JSON
        conv_data = row.get("conversation", {})
        if isinstance(conv_data, str):
            try:
                conv_data = json.loads(conv_data)
            except json.JSONDecodeError:
                conv_data = {}

        conv_id = row.get("conv_id", f"locomo_{row_idx}")
        
        if conv_id not in conv_map:
            turns = []
            session_of_turn = {}
            
            # Tìm tất cả các session bắt đầu bằng session_
            session_keys = sorted(
                [k for k in conv_data.keys() if k.startswith("session_") and not k.endswith("date_time")],
                key=lambda k: int(k.split("_")[1]) if k.split("_")[1].isdigit() else 0,
            )

            for sess_key in session_keys:
                sess_num = sess_key.split("_")[1]
                date_key = f"{sess_key}_date_time"
                date = conv_data.get(date_key)
                
                # Duyệt qua các turn trong session
                sess_turns = conv_data.get(sess_key, [])
                if isinstance(sess_turns, list):
                    for turn_idx, turn in enumerate(sess_turns):
                        # Lấy turn_id chuẩn (ưu tiên dia_id nếu có, không thì tự sinh D{sess_num}:{turn_idx})
                        turn_id = turn.get("dia_id", f"D{sess_num}:{turn_idx}")
                        turns.append({
                            "turn_id": turn_id,
                            "speaker": turn.get("speaker", "unknown"),
                            "text": turn.get("text", turn.get("clean_text", "")),
                            "date": date,
                            "session_id": f"session_{sess_num}",
                        })
                        session_of_turn[turn_id] = f"session_{sess_num}"

            conv_map[conv_id] = {
                "conv_id": conv_id,
                "turns": turns,
                "session_of_turn": session_of_turn,
                "qas": []
            }

        # 2. Trích xuất thông tin QA từ row hiện tại
        session_of_turn = conv_map[conv_id]["session_of_turn"]
        evidence = row.get("evidence", [])
        if isinstance(evidence, str):
            try:
                evidence = json.loads(evidence)
            except json.JSONDecodeError:
                evidence = []

        gold_sessions = sorted({session_of_turn.get(e, e.split(":")[0] if ":" in e else "session_1") for e in evidence})
        
        qa_item = {
            "question": row.get("question", ""),
            "answer": str(row.get("answer", "")),
            "category": row.get("category"),
            "gold_turn_ids": evidence,
            "gold_session_ids": gold_sessions,
        }
        conv_map[conv_id]["qas"].append(qa_item)

    # Loại bỏ trường phụ và trả về danh sách conversation chuẩn
    conversations = []
    for c_id, c_val in conv_map.items():
        conversations.append({
            "conv_id": c_id,
            "turns": c_val["turns"],
            "qas": c_val["qas"]
        })

    return conversations


def load_longmemeval(path: str) -> List[Dict[str, Any]]:
    """
    Parses LongMemEval-s: one JSONL file where each row is a QA instance
    with a `haystack_sessions` list of sessions (each a list of
    {role, content} turns), `haystack_session_ids`, `answer_session_ids`
    (gold session-level provenance), `question`, and `answer`.

    Since LongMemEval is QA-instance-centric (not conversation-centric),
    we treat each row as its own single "conversation" containing the
    full haystack, matching the paper's inference-only per-QA setting.
    """
    rows = _read_jsonl(path) if path.endswith(".jsonl") else _read_json(path)
    conversations = []

    for row_idx, row in enumerate(rows):
        conv_id = row.get("question_id", f"longmem_{row_idx}")
        turns = []
        session_ids = row.get("haystack_session_ids", [])
        sessions = row.get("haystack_sessions", [])

        for s_idx, (sess_id, session) in enumerate(zip(session_ids, sessions)):
            for t_idx, turn in enumerate(session):
                turn_id = f"{sess_id}:{t_idx}"
                turns.append({
                    "turn_id": turn_id,
                    "speaker": turn.get("role", turn.get("speaker", "unknown")),
                    "text": turn.get("content", turn.get("text", "")),
                    "date": row.get("haystack_dates", [None] * len(sessions))[s_idx]
                    if row.get("haystack_dates") else None,
                    "session_id": sess_id,
                })

        gold_sessions = row.get("answer_session_ids", [])
        qas = [{
            "question": row.get("question", ""),
            "answer": str(row.get("answer", "")),
            "category": row.get("question_type"),
            "gold_turn_ids": [],  # LongMemEval ground-truth is session-level
            "gold_session_ids": gold_sessions,
        }]

        conversations.append({"conv_id": conv_id, "turns": turns, "qas": qas})

    return conversations


def load_dataset(name: str, path: str) -> List[Dict[str, Any]]:
    name = name.lower()
    if name in ("locomo", "locomo-10", "locomo10"):
        return load_locomo(path)
    if name in ("longmemeval", "longmemeval-s", "longmem"):
        return load_longmemeval(path)
    raise ValueError(f"Unknown dataset '{name}'. Use 'locomo' or 'longmemeval'.")
