"""
LLM-as-a-judge evaluation. Paper reference: prompt Appendix C.7 ("we
additionally employ GPT-4o as a judge"). By default this CLI now calls the
REAL OpenAI API via JudgeClient, matching the paper. `--local` is available
for cheap local sanity-checking only and is explicitly labeled as NOT a
faithful reproduction of the paper's GPT4o-J metric.
"""
import os
import re
import sys
import json
import argparse
from typing import List

# Thêm thư mục gốc vào sys.path để gọi được thư mục prompts, config, llm_client
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompts import load_prompt
from config import DEFAULT_CONFIG, LLMConfig, JudgeConfig
from llm_client import LLMClient, JudgeClient


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


def _load_records(path: str):
    """
    Loads QA records from either:
      - the `*_raw.jsonl` file written by run_pipeline.py (one JSON object
        per line, keys: "question", "reference", "prediction", ...) - this
        is the actual output format run_pipeline.py produces, and
      - a generic summary .json with a top-level list of {"qas": [...]}
        or a flat list of QA dicts, for compatibility with other formats.
    Returns (questions, references, responses).
    """
    questions, references, responses = [], [], []

    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                questions.append(item.get("question", ""))
                references.append(str(item.get("reference", item.get("answer", ""))))
                responses.append(str(item.get("prediction", item.get("generated_answer", item.get("model_answer", "")))))
        return questions, references, responses

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        if "qas" in item:
            for qa in item["qas"]:
                questions.append(qa.get("question", ""))
                references.append(str(qa.get("answer", qa.get("reference", ""))))
                responses.append(str(qa.get("generated_answer", qa.get("prediction", qa.get("model_answer", "")))))
        else:
            questions.append(item.get("question", ""))
            references.append(str(item.get("answer", item.get("reference", ""))))
            responses.append(str(item.get("generated_answer", item.get("prediction", item.get("model_answer", "")))))
    return questions, references, responses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                         help="Path to run_pipeline.py's *_raw.jsonl (preferred) or a summary .json")
    parser.add_argument("--local", action="store_true",
                         help="Use a local OpenAI-compatible endpoint (e.g. the same vLLM server) "
                              "as a CHEAP PROXY judge instead of real GPT-4o. NOTE: the paper's "
                              "GPT4o-J metric specifically uses GPT-4o via the OpenAI API; scores "
                              "from --local are NOT comparable to Table 1 and should be labeled "
                              "separately (e.g. 'Local-Judge') in any report.")
    parser.add_argument("--local_base_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--local_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--judge_model", type=str, default=None,
                         help="Override the real GPT-4o judge model name (default from config/env)")
    args = parser.parse_args()

    if args.local:
        judge_client = LLMClient(LLMConfig(
            api_base=args.local_base_url, api_key="EMPTY",
            model_name=args.local_model, temperature=0.0, max_tokens=10,
        ))
        judge_label = f"LOCAL PROXY judge ({args.local_model}) - NOT the paper's GPT-4o-J"
    else:
        judge_cfg = DEFAULT_CONFIG.judge
        if args.judge_model:
            judge_cfg = JudgeConfig(model_name=args.judge_model, api_key=judge_cfg.api_key)
        if not judge_cfg.api_key:
            sys.exit(
                "ERROR: OPENAI_API_KEY is not set. The paper's GPT4o-J metric requires the real "
                "OpenAI API (Appendix C.7). Set OPENAI_API_KEY, or pass --local for a cheap, "
                "NON-comparable local proxy judge instead."
            )
        judge_client = JudgeClient(judge_cfg)
        judge_label = f"GPT-4o judge ({judge_cfg.model_name})"

    questions, references, responses = _load_records(args.input)

    print(f"\n[Judge] Scoring {len(questions)} QA pairs with {judge_label}...")
    score = gpt4o_judge_score(judge_client, questions, references, responses)
    print(f"✅ GPT4o-J score: {score:.2f}%\n")