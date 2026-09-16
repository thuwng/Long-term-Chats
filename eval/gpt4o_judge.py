"""
GPT-4o-as-judge evaluation, matching the GPT4o-J column in Table 1.
Paper reference: prompt Appendix C.7. Uses the real OpenAI API (GPT-4o),
separate from the local backbone LLM used for indexing/retrieval/generation.
"""
import re
from typing import List

from prompts import load_prompt


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
