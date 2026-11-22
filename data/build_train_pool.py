"""Phase 2: Build the full training pool from tau2-bench.

The Phase 0 script (prepare_tau2.py) produced a small balanced sample
(70/domain) for the kill test.  This script extracts a much larger pool for
Phase 3 (synthetic negative generation) and Phase 5/6 (model training).

Key differences from prepare_tau2.py:
  - No per-domain cap (take everything extractable).
  - CORRECT-ONLY by default (see docs/decision_log.md D-008): the heuristic
    "incorrect" labels were ~75% benchmark artifacts in the Phase 1 hand
    review, so every negative in training comes from Phase 3 synthetic
    corruption instead. Pass --include-incorrect for the full heuristic pool.
  - Include all source models and all three core domains.
  - Exclude Phase 0 examples by trajectory_id AND by (domain, task_id), so
    other agents'/trials' runs of the same task can't leak into training.
  - Add a `split` field: "train" for all examples here.
  - Save as data/derived/train_pool.jsonl.

Output format per line matches the Phase 3 corruption contract:
{
  "trajectory_id": str,       # unique; format <source_file_stem>::<sim_uuid>
  "domain": "retail|airline|telecom",
  "task_id": str|int,
  "source_model": str,
  "tool_schemas_text": str,   # formatted tool schema for this domain
  "turns": [...],             # same compact format as prepare_tau2.py
  "candidate_tool_call": {"name": ..., "arguments": ...},
  "label": "correct",         # "incorrect" only with --include-incorrect
  "label_tier": "exact",       # wrong_argument_value|reward_fallback only with --include-incorrect
  "gold_action": dict|null,
  "split": "train"
}
"""

import argparse
import glob
import json
import random
import sys
from pathlib import Path

# Allow running from data/ directory or project root
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from tool_schemas import FREE_TEXT_TOOLS, extract_tool_schemas, format_schemas_for_prompt

RESULTS_DIR = _HERE / "raw" / "tau2-bench" / "data" / "tau2" / "results" / "final"
DOMAINS = ("retail", "airline", "telecom")


# ── helpers (same as prepare_tau2.py) ──────────────────────────────────────

def domain_of(filename: str) -> str | None:
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) < 2:
        return None
    domain_field = parts[1]
    return domain_field if domain_field in DOMAINS else None


def get_tool_calls(sim: dict) -> list[dict]:
    calls = []
    for m in sim["messages"]:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            calls.extend(m["tool_calls"])
    return calls


def render_turns(messages: list[dict]) -> list[dict]:
    turns = []
    for m in messages:
        role = m["role"]
        if role == "tool":
            turns.append({"role": "tool", "content": str(m.get("content", ""))[:2000]})
        elif role == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                turns.append({
                    "role": "assistant_tool_call",
                    "name": tc["name"],
                    "arguments": tc["arguments"],
                })
        elif m.get("content"):
            turns.append({"role": role, "content": m["content"]})
    return turns


def label_last_call(sim: dict) -> tuple[str | None, str | None, dict | None]:
    """Return (label, tier, matched_gold_action) for the sim's last non-free-text tool call."""
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
        (i, m) for i, m in enumerate(sim["messages"])
        if m.get("role") == "assistant" and m.get("tool_calls")
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
        "source_model": Path(source_file).stem.split("_")[0],
        "tool_schemas_text": tool_schema_text,
        "turns": render_turns(sim["messages"][:last_idx]),
        "candidate_tool_call": {
            "name": candidate_tool_call["name"],
            "arguments": candidate_tool_call["arguments"],
        },
        "label": label,
        "label_tier": tier,
        "gold_action": gold_action,
        "split": "train",
    }


# ── deduplication ───────────────────────────────────────────────────────────

import hashlib

def _content_key(cand: dict) -> str:
    payload = json.dumps(cand["turns"], sort_keys=True) + json.dumps(cand["candidate_tool_call"], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def dedupe_pool(cands: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out = []
    for c in cands:
        key = _content_key(c)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


# ── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--exclude-ids",
        default="derived/phase0_candidates.jsonl",
        help="Path to Phase 0 file; those trajectory_ids are excluded from train pool",
    )
    ap.add_argument("--out", default="derived/train_pool.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--max-per-domain", type=int, default=0,
        help="If >0, cap examples per domain to this many (for quick tests)",
    )
    ap.add_argument(
        "--include-incorrect", action="store_true",
        help="Keep heuristic 'incorrect' examples too (default: correct-only, see docs/decision_log.md D-008)",
    )
    args = ap.parse_args()

    # Load Phase 0 IDs and (domain, task_id) pairs to exclude. task_ids like
    # "17" repeat across retail/airline, so key on the pair, not task_id alone.
    exclude_ids: set[str] = set()
    exclude_tasks: set[tuple[str, str]] = set()
    excl_path = _HERE / args.exclude_ids
    if excl_path.exists():
        with open(excl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    exclude_ids.add(obj["trajectory_id"])
                    exclude_tasks.add((obj["domain"], str(obj.get("task_id"))))
        print(
            f"Excluding {len(exclude_ids)} Phase 0 trajectory IDs and "
            f"{len(exclude_tasks)} Phase 0 (domain, task_id) pairs from train pool."
        )
    else:
        print(f"Warning: exclude file not found at {excl_path}; proceeding without exclusions.")

    rng = random.Random(args.seed)
    schema_text = {d: format_schemas_for_prompt(extract_tool_schemas(d)) for d in DOMAINS}

    by_domain: dict[str, list[dict]] = {d: [] for d in DOMAINS}
    tier_counts: dict[str, int] = {}
    skipped_excluded = 0
    skipped_task_excluded = 0

    for f in sorted(glob.glob(str(RESULTS_DIR / "*.json"))):
        domain = domain_of(f)
        if domain is None:
            continue
        print(f"  Loading {Path(f).name} ...", end=" ", flush=True)
        with open(f) as fh:
            data = json.load(fh)
        n_added = 0
        for sim in data.get("simulations", []):
            label, tier, _ = label_last_call(sim)
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            if label is None:
                continue
            cand = build_candidate(sim, domain, f, schema_text[domain])
            if cand is None:
                continue
            if cand["trajectory_id"] in exclude_ids:
                skipped_excluded += 1
                continue
            if (domain, str(cand["task_id"])) in exclude_tasks:
                skipped_task_excluded += 1
                continue
            by_domain[domain].append(cand)
            n_added += 1
        print(f"{n_added} candidates")

    print(f"\nSkipped {skipped_excluded} Phase 0 examples by trajectory_id (train/test separation).")
    print(f"Skipped {skipped_task_excluded} more by Phase 0 (domain, task_id) (same task, other agent/trial).")

    print("\nLabel tier totals across all scanned simulations:")
    for t, c in sorted(tier_counts.items()):
        print(f"  {t}: {c}")

    total_before_dedup = sum(len(v) for v in by_domain.values())
    for domain in by_domain:
        before = len(by_domain[domain])
        by_domain[domain] = dedupe_pool(by_domain[domain])
        removed = before - len(by_domain[domain])
        if removed:
            print(f"  {domain}: removed {removed} content-duplicate(s)")

    print("\nPool sizes after dedup (before correct-only filter):")
    for domain, cands in by_domain.items():
        correct = sum(1 for c in cands if c["label"] == "correct")
        incorrect = sum(1 for c in cands if c["label"] == "incorrect")
        print(f"  {domain}: {correct} correct, {incorrect} incorrect  (total {len(cands)})")

    if args.include_incorrect:
        print("\n--include-incorrect set: keeping heuristic 'incorrect' examples (known ~75% artifact rate).")
    else:
        n_dropped = 0
        for domain in by_domain:
            kept = [c for c in by_domain[domain] if c["label"] == "correct"]
            n_dropped += len(by_domain[domain]) - len(kept)
            by_domain[domain] = kept
        print(
            f"\nCorrect-only (default): dropped {n_dropped} heuristic 'incorrect' examples. "
            "All negatives must come from Phase 3 synthetic corruption (docs/decision_log.md D-008)."
        )

    # Apply optional per-domain cap (for tests / quick dev runs)
    selected = []
    for domain, cands in by_domain.items():
        if args.max_per_domain > 0:
            correct = [c for c in cands if c["label"] == "correct"]
            incorrect = [c for c in cands if c["label"] == "incorrect"]
            rng.shuffle(correct)
            rng.shuffle(incorrect)
            if args.include_incorrect:
                n_each = min(args.max_per_domain // 2, len(correct), len(incorrect))
                chosen = correct[:n_each] + incorrect[:n_each]
            else:
                chosen = correct[: args.max_per_domain]
        else:
            chosen = cands
        selected.extend(chosen)

    rng.shuffle(selected)

    out_path = _HERE / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for c in selected:
            fh.write(json.dumps(c) + "\n")

    n_correct = sum(1 for c in selected if c["label"] == "correct")
    n_incorrect = len(selected) - n_correct
    print(f"\nWrote {len(selected)} candidates to {out_path}")
    print(f"  correct: {n_correct}  incorrect: {n_incorrect}")
    if args.include_incorrect:
        print(f"  balance ratio: {n_incorrect / max(n_correct, 1):.2f} incorrect per correct")
    else:
        print("  incorrect excluded by default; negatives come from Phase 3 synthetic corruption")

    # Source model breakdown
    from collections import Counter
    model_counts = Counter(c["source_model"] for c in selected)
    print("\nSource model breakdown:")
    for m, cnt in sorted(model_counts.items()):
        print(f"  {m}: {cnt}")


if __name__ == "__main__":
    main()
