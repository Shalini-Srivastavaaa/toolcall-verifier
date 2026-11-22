"""Build Phase 0 (and later Phase 2) candidate examples from tau2-bench's
precomputed simulation results.

tau2-bench ships full agent trajectories (`messages`) plus, per simulation,
a gold `action_checks` list: the required actions for that task and whether
each was ever performed (`action_match`). That's a *trajectory-level*
"was this required action hit" signal, not a per-call correctness label, so
it can't be zipped positionally against the agent's actual tool calls (see
docs/decision_log.md for the verification that ruled this out).

What we extract instead, per simulation, is a single (context, candidate)
pair built from the LAST tool call the agent made:

  - exact match (name + args) against a gold action  -> label "correct"
  - same tool name as a gold action, different args   -> label "incorrect"
    (wrong_argument_value)
  - reward == 0.0 and no matching gold action at all  -> label "incorrect"
    (task failed and the final action wasn't a recognized correct one)
  - reward == 1.0 and no matching gold action at all  -> skipped (ambiguous:
    usually a benign trailing read call after the task was already satisfied)

`transfer_to_human_agents` is excluded: its argument is free-form NL summary
text, and tau2's own exact-match grading of it produces false "incorrect"
labels for calls that are actually fine (see docs/decision_log.md).

This is a coarse proxy, deliberately so — Phase 0 is a go/no-go kill test,
not the final labeled dataset. Phase 2/4 build the real per-error-class
labels by hand.
"""

import argparse
import glob
import hashlib
import json
import random
from pathlib import Path

from tool_schemas import FREE_TEXT_TOOLS, extract_tool_schemas, format_schemas_for_prompt

RESULTS_DIR = Path(__file__).parent / "raw" / "tau2-bench" / "data" / "tau2" / "results" / "final"
DOMAINS = ("retail", "airline", "telecom")


def domain_of(filename: str) -> str | None:
    """Map a results filename to one of our three domains, excluding the
    telecom-workflow / -op variants (different task format, out of scope here).
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) < 2:
        return None
    domain_field = parts[1]  # e.g. "retail", "airline", "telecom", "telecom-workflow"
    return domain_field if domain_field in DOMAINS else None


def get_tool_calls(sim: dict) -> list[dict]:
    calls = []
    for m in sim["messages"]:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            calls.extend(m["tool_calls"])
    return calls


def render_turns(messages: list[dict]) -> list[dict]:
    """Compact the raw tau2 message log into {role, content} turns for the prompt."""
    turns = []
    for m in messages:
        role = m["role"]
        if role == "tool":
            turns.append({"role": "tool", "content": str(m.get("content"))[:2000]})
        elif role == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                turns.append(
                    {"role": "assistant_tool_call", "name": tc["name"], "arguments": tc["arguments"]}
                )
        elif m.get("content"):
            turns.append({"role": role, "content": m["content"]})
    return turns


def label_last_call(sim: dict) -> tuple[str | None, str | None, dict | None]:
    """Return (label, tier, matched_gold_action) for the sim's last tool call."""
    calls = get_tool_calls(sim)
    if not calls:
        return None, "no_calls", None
    last = calls[-1]
    if last["name"] in FREE_TEXT_TOOLS:
        return None, "free_text_tool", None

    checks = (sim.get("reward_info") or {}).get("action_checks") or []
    exact = [c for c in checks if c["action"]["name"] == last["name"] and c["action"]["arguments"] == last["arguments"]]
    if exact:
        return ("correct" if exact[0]["action_match"] else "incorrect"), "exact", exact[0]["action"]

    name_matches = [c for c in checks if c["action"]["name"] == last["name"]]
    if name_matches:
        return "incorrect", "wrong_argument_value", name_matches[0]["action"]

    reward = (sim.get("reward_info") or {}).get("reward")
    if reward == 0.0:
        return "incorrect", "reward_fallback", None
    return None, "skip_ambiguous", None


def build_candidate(sim: dict, domain: str, source_file: str, tool_schema_text: str) -> dict | None:
    calls_with_idx = [
        (i, m) for i, m in enumerate(sim["messages"]) if m.get("role") == "assistant" and m.get("tool_calls")
    ]
    if not calls_with_idx:
        return None
    last_idx, last_msg = calls_with_idx[-1]
    candidate_tool_call = last_msg["tool_calls"][-1]

    label, tier, gold_action = label_last_call(sim)
    if label is None:
        return None

    return {
        "trajectory_id": f"{Path(source_file).stem}::{sim['id']}",
        "domain": domain,
        "task_id": sim.get("task_id"),
        "tool_schemas_text": tool_schema_text,
        "turns": render_turns(sim["messages"][:last_idx]),
        "candidate_tool_call": {"name": candidate_tool_call["name"], "arguments": candidate_tool_call["arguments"]},
        "label": label,
        "label_tier": tier,
        "gold_action": gold_action,
        "source_model": Path(source_file).stem.split("_")[0],
    }


def _content_key(cand: dict) -> str:
    """Hash of (conversation, candidate call) — the actual labeling content.
    Distinct `task_id`s in telecom can realize the same conversation (see
    docs/decision_log.md), so dedup on this rather than on trajectory_id (always
    unique, a UUID) or task_id (would also drop legitimately distinct
    failures from different agents on the same underlying task)."""
    payload = json.dumps(cand["turns"], sort_keys=True) + json.dumps(cand["candidate_tool_call"], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def dedupe_pool(cands: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for c in cands:
        key = _content_key(c)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-domain", type=int, default=70, help="target candidates per domain before balancing")
    ap.add_argument("--out", default="derived/phase0_candidates.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    schema_text = {d: format_schemas_for_prompt(extract_tool_schemas(d)) for d in DOMAINS}

    by_domain: dict[str, list[dict]] = {d: [] for d in DOMAINS}
    tier_counts: dict[str, int] = {}

    for f in sorted(glob.glob(str(RESULTS_DIR / "*.json"))):
        domain = domain_of(f)
        if domain is None:
            continue
        data = json.load(open(f))
        for sim in data.get("simulations", []):
            label, tier, _ = label_last_call(sim)
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            if label is None:
                continue
            cand = build_candidate(sim, domain, f, schema_text[domain])
            if cand:
                by_domain[domain].append(cand)

    print("Label tier counts across all simulations scanned:")
    for t, c in sorted(tier_counts.items()):
        print(f"  {t}: {c}")

    for domain in by_domain:
        before = len(by_domain[domain])
        by_domain[domain] = dedupe_pool(by_domain[domain])
        removed = before - len(by_domain[domain])
        if removed:
            print(f"{domain}: removed {removed} content-duplicate candidate(s) from the pool")

    out_path = Path(__file__).parent / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    selected = []
    for domain, cands in by_domain.items():
        correct = [c for c in cands if c["label"] == "correct"]
        incorrect = [c for c in cands if c["label"] == "incorrect"]
        rng.shuffle(correct)
        rng.shuffle(incorrect)
        n_each = min(args.per_domain // 2, len(correct), len(incorrect))
        chosen = correct[:n_each] + incorrect[:n_each]
        rng.shuffle(chosen)
        selected.extend(chosen)
        print(f"{domain}: pool correct={len(correct)} incorrect={len(incorrect)} -> selected {len(chosen)}")

    rng.shuffle(selected)
    with open(out_path, "w") as fh:
        for c in selected:
            fh.write(json.dumps(c) + "\n")

    n_correct = sum(1 for c in selected if c["label"] == "correct")
    print(f"\nWrote {len(selected)} candidates to {out_path} ({n_correct} correct / {len(selected) - n_correct} incorrect)")


if __name__ == "__main__":
    main()
