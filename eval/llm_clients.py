"""Thin, uniform wrappers around the OpenAI and Gemini SDKs for the Phase 0
judge harness. Each call returns a JudgeResult with enough detail (latency,
token usage, raw text) to compute the accuracy/latency/cost table without
re-calling the API.
"""

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from tenacity import retry, stop_after_attempt, wait_exponential

_openai_client = None


def get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI

        _openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _openai_client


@dataclass
class JudgeResult:
    verdict: Optional[str]  # "correct" | "incorrect" | None if unparseable
    confidence: Optional[float]
    reasoning: Optional[str]
    latency_s: float
    input_tokens: int
    output_tokens: int
    raw_text: str
    error: Optional[str] = None
    extra: dict = field(default_factory=dict)


def _parse_verdict_json(text: str) -> tuple[Optional[str], Optional[float], Optional[str]]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None, None, None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None, None, None
    verdict = obj.get("verdict")
    if verdict not in ("correct", "incorrect"):
        verdict = None
    confidence = obj.get("confidence")
    try:
        confidence = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence = None
    return verdict, confidence, obj.get("reasoning")


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=30), reraise=True)
def call_openai(model: str, system_prompt: str, user_prompt: str, reasoning_effort: Optional[str] = None) -> JudgeResult:
    client = get_openai_client()
    start = time.monotonic()
    kwargs = {}
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **kwargs,
        )
    except Exception as e:  # surfaced to caller; retry decorator handles transient ones
        latency = time.monotonic() - start
        return JudgeResult(None, None, None, latency, 0, 0, "", error=str(e))
    latency = time.monotonic() - start
    text = resp.choices[0].message.content or ""
    verdict, confidence, reasoning = _parse_verdict_json(text)
    usage = resp.usage
    details = getattr(usage, "completion_tokens_details", None)
    reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0
    return JudgeResult(
        verdict=verdict,
        confidence=confidence,
        reasoning=reasoning,
        latency_s=latency,
        input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        raw_text=text,
        extra={"reasoning_tokens": reasoning_tokens},
    )


CALLERS = {"openai": call_openai}
