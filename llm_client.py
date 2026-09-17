"""
Thin wrapper around an OpenAI-compatible chat completion endpoint.

The paper runs `openai/gpt-oss-20b` locally (e.g. served via vLLM or
Ollama with an OpenAI-compatible API) for ALL LLM-dependent modules:
segmentation, filtering, summarization, triplet/entity extraction,
and final answer generation. Point `api_base` at your local server.

GPT-4o-as-judge (Appendix C.7) uses the real OpenAI API instead.
"""
import re, time
import logging
from typing import Optional

from openai import OpenAI

from config import LLMConfig, JudgeConfig

logger = logging.getLogger(__name__)

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

def strip_thinking(text: str) -> str:
    """Removes a leaked <think>...</think> reasoning block, if present."""
    if not text:
        return text
    cleaned = _THINK_BLOCK_RE.sub("", text).strip()
    if "<think>" in cleaned.lower() and "</think>" not in cleaned.lower():
        return ""
    return cleaned

def scaled_max_tokens(n_items: int, per_item: int = 4, base: int = 200, cap: int = 4000) -> int:
    """
    Estimates a safe max_tokens budget for prompts that must enumerate one
    output unit (an index, a pipe-delimited row, ...) per input item -
    e.g. C.1 segmentation and C.2 selective filtering, which need to emit
    EVERY message index. A fixed 512-token budget silently truncates these
    on long conversations, producing invalid JSON that falls back to
    degenerate behavior (whole-conversation-as-one-segment, keep-everything).
    """
    return max(base, min(cap, base + per_item * max(n_items, 0)))


class LLMClient:
    """Generic chat-completion client with retries, used for backbone LLM calls."""

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.client = OpenAI(base_url=cfg.api_base, api_key=cfg.api_key)

    def generate(self, prompt: str, system: Optional[str] = None,
                 max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        last_err = None
        for attempt in range(self.cfg.retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.cfg.model_name,
                    messages=messages,
                    temperature=temperature if temperature is not None else self.cfg.temperature,
                    max_tokens=max_tokens or self.cfg.max_tokens,
                    timeout=self.cfg.timeout,
                    extra_body={"chat_template_kwargs": {"enable_thinking": self.cfg.enable_thinking}},
                )
                raw = resp.choices[0].message.content or ""
                return strip_thinking(raw)
            except Exception as e:  # noqa: BLE001
                last_err = e
                logger.warning("LLM call failed (attempt %d/%d): %s",
                               attempt + 1, self.cfg.retries, e)
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"LLM generation failed after {self.cfg.retries} attempts: {last_err}")


class JudgeClient:
    """GPT-4o judge client (Appendix C.7), separate from the backbone LLM."""

    def __init__(self, cfg: JudgeConfig):
        self.cfg = cfg
        self.client = OpenAI(api_key=cfg.api_key)

    def generate(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.cfg.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
        )
        return resp.choices[0].message.content or ""
