#!/usr/bin/env python3
"""Standalone CLI for Phase 1 manual taxonomy labeling.

Loads the failing trajectories (label == "incorrect") from a candidates
file, shows them one at a time, and asks you to assign one of the seven
taxonomy classes (data/taxonomy.md) or type your own short label if none
fit. Labels are appended to data/labels_manual.jsonl as you go, so killing
the process (Ctrl+C, closing the terminal, whatever) never loses progress —
rerunning just picks up at the first still-unlabeled trajectory.

Usage:
    python data/review.py
    python data/review.py --candidates data/derived/phase0_candidates.jsonl --labels-out data/labels_manual.jsonl
"""

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# Keep in sync with data/taxonomy.md (the canonical labeling guide) if the
# seven classes ever get revised during Phase 1.
TAXONOMY = {
    1: "wrong_argument_value",
    2: "stale_state_ignored_correction",
    3: "wrong_tool_selection",
    4: "repeated_call_duplicate_side_effect",
    5: "failure_to_propagate_tool_output",
    6: "ignoring_tool_error",
    7: "missing_precondition_wrong_order",
}
TAXONOMY_TITLES = {
    1: "Wrong Argument Value",
    2: "Stale State / Ignored Correction / Wrong Turn",
    3: "Wrong Tool Selection",
    4: "Repeated Call / Duplicate Side Effect",
    5: "Failure to Propagate Tool Output",
    6: "Ignoring a Tool Error",
    7: "Policy Violation",
}

DEFAULT_CANDIDATES = Path(__file__).parent / "derived" / "phase0_candidates.jsonl"
DEFAULT_LABELS_OUT = Path(__file__).parent / "labels_manual.jsonl"


def render_turns(turns: list[dict]) -> str:
    lines = []
    for t in turns:
        role = t.get("role")
        if role == "assistant_tool_call":
            lines.append(f"  [agent called]  {t['name']}({t['arguments']})")
        elif role == "tool":
            content = str(t.get("content", ""))
            if len(content) > 300:
                content = content[:300] + " ...(truncated)"
            lines.append(f"  [tool result]   {content}")
        elif role == "user":
            lines.append(f"  [user]          {t.get('content', '')}")
        elif role == "assistant":
            lines.append(f"  [agent]         {t.get('content', '')}")
        else:
            lines.append(f"  [{role}]  {t.get('content', '')}")
    return "\n".join(lines)


def load_candidates(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"Candidates file not found: {path}")
    return [json.loads(line) for line in open(path) if line.strip()]


def load_existing_labels(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    labels = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        labels[rec["trajectory_id"]] = rec
    return labels


def print_menu():
    print()
    for i in sorted(TAXONOMY_TITLES):
        print(f"    {i}) {TAXONOMY_TITLES[i]}")
    print("    Or type your own short label if none of these fit.")
    print("    [c] no error: the flagged call is actually fine (saved, won't reappear)")
    print("    [s]kip for now (NOT saved, reappears next run)   [q]uit (progress is already saved)")


def show_candidate(cand: dict, idx: int, total: int) -> None:
    print("\n" + "=" * 78)
    print(f"Failing trajectory {idx}/{total}   domain={cand['domain']}   task_id={cand.get('task_id')}")
    print(f"id: {cand['trajectory_id']}")
    print(f"source agent: {cand.get('source_model', '?')}   heuristic tier: {cand.get('label_tier', '?')}")
    print("-" * 78)
    print("CONVERSATION:")
    print(render_turns(cand["turns"]))
    print("-" * 78)
    call = cand["candidate_tool_call"]
    print(f">>> CANDIDATE TOOL CALL (flagged incorrect): {call['name']}({call['arguments']})")
    gold = cand.get("gold_action")
    if gold:
        print(f"    reference — a gold/expected action for this task: {gold['name']}({gold['arguments']})")
    else:
        print("    (no specific gold action on record — the task failed overall; judge from context)")


def prompt_label() -> tuple[int | None, str, str] | None:
    """Returns (class_id_or_None, label_name, raw_input), or None for skip/quit signal handled by caller."""
    raw = input("\nYour label (1-7, c=no error, custom text, s=skip, q=quit): ").strip()
    if raw.lower() in ("q", "quit"):
        return "QUIT"
    if raw.lower() in ("s", "skip", ""):
        return "SKIP"
    if raw.lower() in ("c", "ok", "no error"):
        return (None, "no_error", raw)
    if raw.isdigit() and int(raw) in TAXONOMY:
        cid = int(raw)
        return (cid, TAXONOMY[cid], raw)
    return (None, raw, raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    ap.add_argument("--labels-out", type=Path, default=DEFAULT_LABELS_OUT)
    args = ap.parse_args()

    all_candidates = load_candidates(args.candidates)
    failing = [c for c in all_candidates if c.get("label") == "incorrect"]
    existing = load_existing_labels(args.labels_out)
    todo = [c for c in failing if c["trajectory_id"] not in existing]

    print(f"Loaded {len(failing)} failing trajectories from {args.candidates}")
    print(f"{len(existing)} already labeled in {args.labels_out}, {len(todo)} remaining")

    if not todo:
        print("Nothing left to review.")
        print_tally(args.labels_out)
        return

    print_menu()

    args.labels_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.labels_out, "a") as out:
        for i, cand in enumerate(todo):
            show_candidate(cand, i + 1, len(todo))
            try:
                result = prompt_label()
            except (EOFError, KeyboardInterrupt):
                print("\n\nStopped. Progress is saved — rerun to resume where you left off.")
                return

            if result == "QUIT":
                print("Progress is saved — rerun to resume where you left off.")
                break
            if result == "SKIP":
                print("Skipped (will show again next run).")
                continue

            class_id, label_name, raw = result
            record = {
                "trajectory_id": cand["trajectory_id"],
                "domain": cand["domain"],
                "task_id": cand.get("task_id"),
                "source_model": cand.get("source_model"),
                "taxonomy_class": class_id,
                "label": label_name,
                "raw_input": raw,
                "labeled_at": datetime.now(timezone.utc).isoformat(),
            }
            out.write(json.dumps(record) + "\n")
            out.flush()
            tag = f"class {class_id} ({label_name})" if class_id else f"custom label: '{label_name}'"
            print(f"Saved: {tag}")
        else:
            print("\nAll failing trajectories reviewed!")

    print_tally(args.labels_out)


def print_tally(labels_path: Path) -> None:
    if not labels_path.exists():
        return
    labels = [json.loads(line) for line in open(labels_path) if line.strip()]
    if not labels:
        return
    counts = Counter(r["label"] for r in labels)
    print(f"\n{len(labels)} total labels in {labels_path}:")
    for label, n in counts.most_common():
        print(f"  {n:>3}  {label}")


if __name__ == "__main__":
    main()
