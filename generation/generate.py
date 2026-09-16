"""
Section 3.3 (end) - final personalized response generation.
Paper reference: prompt Appendix C.5.

Formats the provenance-aware context (turns + supporting triplets) into
the answer-generation prompt and calls the backbone LLM.
"""
from prompts import load_prompt


def generate_answer(llm, context: str, query: str) -> str:
    prompt_template = load_prompt("c5_answer_generation")
    prompt = prompt_template.format(context=context, query=query)
    return llm.generate(prompt).strip()
