"""Phase 4: build data/derived/test_real.jsonl from every hand-labeled trajectory.

Takes every row in labels_manual.jsonl, joins it to its full record in
phase0_candidates.jsonl, reconciles field names with the training schema,
verifies there is no overlap with any training file, and writes the held-out
test set. Exits non-zero (writing nothing) if any overlap is found.

  label != "no_error"  -> label "incorrect" (genuine error, with an error class)
  label == "no_error"  -> label "correct"   (hand-verified: the flagged call is fine;
                                             error_class / error_class_name are null)

Both halves come from the same pool (calls the Phase 0 heuristic flagged as
incorrect), so the correct half is "suspicious-looking but fine" calls, not
ordinary correct calls. Rows keep the order of labels_manual.jsonl.

Naming reconciliation (manual labels -> training schema):
  manual `label` (class name)       -> `human_label`; `label` becomes "incorrect"/"correct"
  manual `taxonomy_class` (int|None)-> `error_class` (int); a free-text label that
                                       starts with a class digit ("7 - ...") is
                                       mapped to that class and the rest kept as
                                       `human_note`
  class name                        -> `error_class_name`, using corrupt_pool.ERROR_CLASSES
  candidate's old heuristic tier    -> `heuristic_tier`; `label_tier` = "hand_labeled"
"""

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from corrupt_pool import ERROR_CLASSES  # noqa: E402

D = Path(__file__).parent / "derived"
LABELS = Path(__file__).parent / "labels_manual.jsonl"
CANDIDATES = D / "phase0_candidates.jsonl"
OUT = D / "test_real.jsonl"
TRAIN_FILES = ["train_pool.jsonl", "bfcl_pool.jsonl", "negatives_pool_final.jsonl", "positives_class2_controls.jsonl"]

HUMAN_TO_CLASS = {
    "wrong_argument_value": 1,
    "stale_state_ignored_correction": 2,
    "wrong_tool_selection": 3,
    "repeated_call_duplicate_side_effect": 4,
    "failure_to_propagate_tool_output": 5,
    "ignoring_tool_error": 6,
    "missing_precondition_wrong_order": 7,
}
SUFFIX = re.compile(r"::(corrupt\d+p?|ctrl)$")


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path) if line.strip()]


def turns_key(turns: list) -> str:
    return hashlib.sha256(json.dumps(turns, sort_keys=True).encode()).hexdigest()


def content_key(r: dict) -> str:
    return hashlib.sha256((json.dumps(r["turns"], sort_keys=True) + json.dumps(r["candidate_tool_call"], sort_keys=True)).encode()).hexdigest()


def resolve_class(lab: dict) -> tuple[int, str | None]:
    """Return (error_class, human_note) for a genuine-error label row."""
    if lab["taxonomy_class"] is not None:
        cls = int(lab["taxonomy_class"])
        assert HUMAN_TO_CLASS.get(lab["label"]) == cls, f"class/name mismatch: {lab['label']} vs {cls}"
        return cls, None
    m = re.match(r"^\s*([1-7])\b[\s\-:.]*(.*)$", lab["label"])
    if not m:
        raise SystemExit(f"Cannot map free-text label to a class: {lab['label']!r}")
    return int(m.group(1)), (m.group(2).strip() or None)


def main():
    labels = load(LABELS)
    n_err = sum(r["label"] != "no_error" for r in labels)
    print(f"labels_manual.jsonl: {len(labels)} rows = {n_err} genuine errors + {len(labels) - n_err} no_error")

    cands = {r["trajectory_id"]: r for r in load(CANDIDATES)}
    missing = [r["trajectory_id"] for r in labels if r["trajectory_id"] not in cands]
    if missing:
        raise SystemExit(f"Labeled rows missing from phase0_candidates.jsonl: {missing}")

    records, dropped = [], []
    for lab in labels:
        base = cands[lab["trajectory_id"]]
        is_error = lab["label"] != "no_error"
        if not base["turns"] or not base["candidate_tool_call"]:
            dropped.append((lab["trajectory_id"], "empty turns or candidate call"))
            continue
        rec = {k: base[k] for k in ("trajectory_id", "domain", "task_id", "source_model",
                                    "tool_schemas_text", "turns", "candidate_tool_call", "gold_action")}
        if is_error:
            cls, note = resolve_class(lab)
            error_fields = {"error_class": cls, "error_class_name": ERROR_CLASSES[cls]}
        else:
            note, error_fields = None, {"error_class": None, "error_class_name": None}
        rec.update({
            "label": "incorrect" if is_error else "correct",
            "label_tier": "hand_labeled",
            "split": "test",
            **error_fields,
            "human_label": lab["label"],
            "heuristic_tier": base["label_tier"],
            "labeled_at": lab["labeled_at"],
        })
        if note:
            rec["human_note"] = note
        records.append(rec)

    ids = [r["trajectory_id"] for r in records]
    assert len(ids) == len(set(ids)), "duplicate ids in test set"
    ck = [content_key(r) for r in records]
    assert len(ck) == len(set(ck)), "duplicate content in test set"

    # ── overlap checks against every training file ───────────────────────────
    test_ids = set(ids)
    test_tasks = {(r["domain"], str(r["task_id"])) for r in records}
    test_turns = {turns_key(r["turns"]): r["trajectory_id"] for r in records}
    overlaps = []
    for name in TRAIN_FILES:
        rows = load(D / name)
        id_hits = task_hits = turn_hits = 0
        for r in rows:
            base_id = SUFFIX.sub("", r["trajectory_id"])
            if r["trajectory_id"] in test_ids or base_id in test_ids:
                id_hits += 1
                overlaps.append((name, "trajectory_id", r["trajectory_id"]))
            if (r["domain"], str(r["task_id"])) in test_tasks:
                task_hits += 1
                overlaps.append((name, "(domain, task_id)", f"{r['domain']}/{r['task_id']}"))
            if turns_key(r["turns"]) in test_turns:
                turn_hits += 1
                overlaps.append((name, "identical conversation turns", r["trajectory_id"]))
        print(f"  overlap vs {name:32s} rows={len(rows):5d}  id={id_hits}  (domain,task_id)={task_hits}  identical-turns={turn_hits}")

    if overlaps:
        print(f"\nOVERLAPS FOUND ({len(overlaps)}); nothing written:")
        for o in overlaps:
            print("  ", o)
        sys.exit(1)

    with open(OUT, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    print(f"\nWrote {len(records)} examples to {OUT.name}; dropped {len(dropped)}")
    for d in dropped:
        print("  dropped:", d)
    by_label = Counter(r["label"] for r in records)
    print("Labels:", dict(by_label))
    print("\nError class distribution (incorrect rows):")
    counts = Counter(r["error_class"] for r in records if r["label"] == "incorrect")
    for c in range(1, 8):
        print(f"  Class {c} ({ERROR_CLASSES[c]}): {counts.get(c, 0)}")
    for lab in ("incorrect", "correct"):
        sub = [r for r in records if r["label"] == lab]
        print(f"\n[{lab}] n={len(sub)}  domains={dict(Counter(r['domain'] for r in sub))}")
        print(f"[{lab}] agents={dict(Counter(r['source_model'] for r in sub))}")
        print(f"[{lab}] heuristic tiers={dict(Counter(r['heuristic_tier'] for r in sub))}")
    print("\nDistinct (domain, task_id) pairs overall:", len(test_tasks))


if __name__ == "__main__":
    main()
