"""
Section 3.3 (end) - final personalized response generation.
Paper reference: prompt Appendix C.5.

Formats the provenance-aware context (turns + supporting triplets) into
the answer-generation prompt and calls the backbone LLM.
"""
import re
from prompts import load_prompt

def generate_answer(llm, context: str, query: str) -> str:
    prompt_template = load_prompt("c5_answer_generation")
    prompt = prompt_template.format(context=context, query=query)
    ans = llm.generate(prompt).strip()
    
    # Hậu xử lý (Post-processing): Xóa bỏ các tag mã nguồn bị lọt vào (VD: [D1:3])
    ans = re.sub(r'\[D\d+:\d+\]', '', ans).strip()
    return ans
