"""Model + pricing config for the Phase 0 baseline.

Pricing is $ per 1M tokens. Figures that can't be verified are not hardcoded as
current — fill these in from your provider's live pricing page before
trusting the cost-per-call numbers in the output table. Everything still
runs with zeros; the accuracy/latency numbers are unaffected.
"""

import json
import os
from pathlib import Path

DEFAULT_MODELS = {
    "openai-nano": {"provider": "openai", "model": "gpt-5-nano", "reasoning_effort": "low"},
    "openai-mini": {"provider": "openai", "model": "gpt-5-mini", "reasoning_effort": "low"},
}

PRICING_PATH = Path(__file__).parent / "pricing.json"


def load_pricing() -> dict:
    if PRICING_PATH.exists():
        return json.loads(PRICING_PATH.read_text())
    return {}


def load_models() -> dict:
    """Allow overriding model IDs via env vars without touching code, since
    exact model name strings (e.g. "gpt-5-nano" vs "gpt-5-nano-2025-08-07")
    depend on what's live on your account."""
    models = json.loads(json.dumps(DEFAULT_MODELS))  # deep copy
    if os.getenv("OPENAI_NANO_MODEL"):
        models["openai-nano"]["model"] = os.getenv("OPENAI_NANO_MODEL")
    if os.getenv("OPENAI_MINI_MODEL"):
        models["openai-mini"]["model"] = os.getenv("OPENAI_MINI_MODEL")
    return models
