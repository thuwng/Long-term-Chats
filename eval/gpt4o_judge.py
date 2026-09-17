"""
LLM-as-a-judge evaluation (Local vLLM alternative to GPT-4o).
Paper reference: prompt Appendix C.7.
"""
import os
import re
import sys
import json
import argparse
from typing import List

# Thêm thư mục gốc vào sys.path để gọi được thư mục prompts và llm_client
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompts import load_prompt
from llm_client import LLMClient


def judge_answer(judge_client, question: str, reference_answer: str, model_response: str) -> bool:
    """Returns True if the judge marks the response correct ([[yes]])."""
    prompt_template = load_prompt("c7_gpt4_judge")
    prompt = prompt_template.format(
        question=question, answer=reference_answer, response=model_response
    )
    raw = judge_client.generate(prompt).strip().lower()
    if "[[yes]]" in raw:
        return True
    if "[[no]]" in raw:
        return False
    # fallback: loose match if brackets are missing
    match = re.search(r"\byes\b|\bno\b", raw)
    return match is not None and match.group(0) == "yes"


def gpt4o_judge_score(judge_client, questions: List[str], references: List[str],
                       responses: List[str]) -> float:
    """Returns the % of QA pairs judged correct (GPT4o-J), scale 0-100."""
    if not questions:
        return float("nan")
    verdicts = [
        judge_answer(judge_client, q, r, resp)
        for q, r, resp in zip(questions, references, responses)
    ]
    return sum(verdicts) / len(verdicts) * 100


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Đường dẫn tới file output của pipeline")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Tên model đang chạy trên vLLM")
    args = parser.parse_args()

    # 1. Khởi tạo LLMClient trỏ về vLLM nội bộ (cổng 8000)
    judge_client = LLMClient(
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",  # vLLM local không cần API key
        model_name=args.model,
        temperature=0.0   # Set temp=0.0 để kết quả chấm điểm ổn định, không bị random
    )

    # 2. Đọc file kết quả đầu vào
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions, references, responses = [], [], []

    # 3. Trích xuất QAs từ file kết quả
    for item in data:
        # Tùy thuộc vào việc run_pipeline sinh ra danh sách hội thoại hay danh sách QA phẳng
        if "qas" in item:
            for qa in item["qas"]:
                questions.append(qa.get("question", ""))
                references.append(str(qa.get("answer", "")))
                responses.append(str(qa.get("generated_answer", qa.get("model_answer", ""))))
        else:
            questions.append(item.get("question", ""))
            references.append(str(item.get("answer", item.get("reference", ""))))
            responses.append(str(item.get("generated_answer", item.get("model_answer", ""))))

    print(f"\n[Judge] Đang chấm {len(questions)} câu hỏi bằng mô hình nội bộ: {args.model}...")
    score = gpt4o_judge_score(judge_client, questions, references, responses)
    print(f"✅ Điểm LLM-as-a-Judge (Thay thế GPT4o-J): {score:.2f}%\n")