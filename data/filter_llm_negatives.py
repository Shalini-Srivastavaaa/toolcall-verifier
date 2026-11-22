"""Phase 3: filter the raw LLM negatives down to examples that satisfy their class
definition, then merge with the programmatic negatives.

The raw LLM pass (corrupt_pool.py) produced many mislabeled examples for
Classes 1, 2 and 5 (see docs/decision_log.md D-009). Keep only:

  Class 1: every NEW value in the corrupted call appears in the conversation.
  Class 2: the context contains a real user correction ("actually", "instead", ...).
  Class 5: tau2-bench only (BFCL has no tool-result turns), and every NEW value
           appears in the conversation (not invented).
  Class 3, 6: unchanged by the context filters above.

Quality filters (from the manual review of Class 1 and 3 samples, D-009), applied to the LLM Class 1 and 3 rows
of negatives_pool_final.jsonl after the shuffle, so every other row keeps its position (negatives_llm.jsonl, the
intermediate, is not affected):
  Class 1: drop a call that adds an argument the source call did not have (strict superset of the
           source's argument keys), a call identical to the source (cannot happen: the generator
           rejects it), and a self-exchange (item_ids == new_item_ids).
  Class 3: BFCL only, using the benchmark's own gold calls (raw possible_answer files; a row only
           holds the first gold call of its turn, so it cannot be used): drop a negative whose tool
           is also a gold call of the SAME turn (a valid call, so a mislabeled negative) or is a gold
           call of NO turn of the task (an unrelated tool, too easy). Kept: tools that are gold calls
           of a different turn (wrong timing). tau2-bench has no per-turn gold, so no Class 3 check.

Files (all under data/derived/):
  negatives_llm_raw.jsonl     raw LLM output (gitignored, never committed)
  negatives_llm.jsonl         filtered LLM negatives (rewritten from raw each run)
  negatives_pool.jsonl        programmatic negatives (classes 4, 7; class 7 is excluded below)
  negatives_class2_class5.jsonl  programmatic Class 2 and 5 (corrupt_pool.py --classes 2 5),
                              appended after the shuffled block if present
  negatives_class1_context_swap.jsonl  second-pass LLM Class 1 rows (corrupt_pool.py --class1-context-swap,
                              tau2-bench only, wrong value copied from the conversation); re-checked here
                              and appended last, after the Class 2/5 rows, so earlier rows keep their positions
  negatives_pool_final.jsonl  shuffled(filtered LLM + programmatic 4/7, seed 42), then the
                              Class 2/5 programmatic rows appended. LLM Class 2/5 rows are
                              dropped from the shuffled block afterward: the programmatic
                              generators replace them (see docs/decision_log.md D-009).
  positives_class2_controls.jsonl  one paired positive per programmatic Class 2 negative:
                              same turns (including the injected correction) but the
                              correct value in the call, label "correct"

On first run, an existing negatives_llm.jsonl is moved to negatives_llm_raw.jsonl.
"""

import copy
import hashlib
import json
import random
import re
import shutil
from collections import Counter
from pathlib import Path

D = Path(__file__).parent / "derived"
RAW = D / "negatives_llm_raw.jsonl"
FILTERED = D / "negatives_llm.jsonl"
PROG = D / "negatives_pool.jsonl"
EXTRA = D / "negatives_class2_class5.jsonl"
CLASS1_V2 = D / "negatives_class1_context_swap.jsonl"
FINAL = D / "negatives_pool_final.jsonl"
CONTROLS = D / "positives_class2_controls.jsonl"
# Class 7 (missing precondition) is not trained on: the removal generator makes negatives shorter
# than their positives, and prerequisite rules do not fix that (D-009). negatives_pool.jsonl still
# holds the generator's Class 7 rows; they are dropped here.
EXCLUDED_CLASSES = {7}

CORRECTION_CUE = re.compile(
    r"\b(actually|instead|change (that|it)|i meant|wait|never ?mind|make it|correction)\b", re.I
)


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path) if line.strip()]


def leaves(v) -> set[str]:
    """Flatten a JSON value into its scalar leaves as strings (bool/None ignored)."""
    if isinstance(v, dict):
        return set().union(*(leaves(x) for x in v.values())) if v else set()
    if isinstance(v, (list, tuple)):
        return set().union(*(leaves(x) for x in v)) if v else set()
    if v is None or isinstance(v, bool):
        return set()
    return {str(v)}


def context_text(turns: list[dict]) -> str:
    parts = []
    for t in turns:
        parts.append(str(t.get("content", "")))
        if t.get("arguments"):
            parts.append(json.dumps(t["arguments"], ensure_ascii=False))
    return "\n".join(parts)


def in_context(value: str, ctx: str) -> bool:
    # alphanumeric boundaries so "2" doesn't match inside "1234"
    return re.search(r"(?<![A-Za-z0-9])" + re.escape(value) + r"(?![A-Za-z0-9])", ctx) is not None


def new_values_all_in_context(neg: dict, original_call: dict) -> bool:
    ctx = context_text(neg["turns"])
    o, n = original_call["arguments"], neg["candidate_tool_call"]["arguments"]
    new_leaves = set()
    for k in set(o) | set(n):
        if o.get(k) != n.get(k):
            new_leaves |= leaves(n.get(k)) - leaves(o.get(k))
    return all(in_context(v, ctx) for v in new_leaves)


def make_control(neg: dict) -> dict:
    """Paired positive for a programmatic Class 2 negative: identical context
    (including the injected correction turn), correct value restored in the call."""
    d = neg["corruption_detail"]
    ctrl = copy.deepcopy(neg)
    ctrl["candidate_tool_call"]["arguments"][d["arg"]] = d["correct_value"]
    ctrl["label"] = "correct"
    ctrl["label_tier"] = "synthetic_class2_control"
    ctrl["trajectory_id"] = neg["trajectory_id"].removesuffix("::corrupt2p") + "::ctrl"
    ctrl.pop("error_class")
    ctrl.pop("error_class_name")
    ctrl["corruption_method"] = "control"
    ctrl["paired_negative_id"] = neg["trajectory_id"]
    return ctrl


BFCL_GOLD_DIR = (Path(__file__).parent / "raw" / "bfcl-gorilla" / "berkeley-function-call-leaderboard"
                 / "bfcl_eval" / "data" / "possible_answer")


def load_bfcl_gold() -> dict[str, list[set[str]]]:
    """task id -> for each turn, the set of tool names the benchmark's gold calls use."""
    files = sorted(BFCL_GOLD_DIR.glob("BFCL_v4_multi_turn_*.json"))
    if not files:
        raise SystemExit(f"BFCL gold files not found in {BFCL_GOLD_DIR}; fetch them first (see README)")
    gold = {}
    for f in files:
        for line in open(f):
            if line.strip():
                o = json.loads(line)
                gold[o["id"]] = [{m.group(1) for c in turn if (m := re.match(r"\s*(\w+)\(", c))} for turn in o["ground_truth"]]
    return gold


def quality_reasons(r: dict, orig: dict, gold: dict) -> list[str]:
    """Why a Class 1 or Class 3 negative should be dropped; empty if it is fine."""
    call, reasons = r["candidate_tool_call"], []
    if r["error_class"] == 1:
        if set(call["arguments"]) > set(orig["arguments"]):
            reasons.append("adds an argument the source call did not have")
        if call["arguments"] == orig["arguments"]:
            reasons.append("arguments identical to the source call")
        if call["arguments"].get("item_ids") and call["arguments"].get("item_ids") == call["arguments"].get("new_item_ids"):
            reasons.append("self-exchange (item_ids == new_item_ids)")
    elif r["error_class"] == 3 and r.get("source_dataset") == "bfcl":
        _, qid, t = r["trajectory_id"].split("::")[:3]          # bfcl_<category>::<task id>::t<turn>
        turns, turn = gold[qid], int(t[1:])
        if call["name"] in turns[turn]:
            reasons.append("BFCL: tool is also a gold call of the same turn")
        elif not any(call["name"] in names for names in turns):
            reasons.append("BFCL: tool is a gold call of no turn of the task")
    return reasons


def has_user_correction(turns: list[dict]) -> bool:
    return any(t.get("role") == "user" and CORRECTION_CUE.search(t.get("content", "")) for t in turns)


def main():
    if not RAW.exists():
        if not FILTERED.exists():
            raise SystemExit(f"Neither {RAW.name} nor {FILTERED.name} exists; run corrupt_pool.py first.")
        shutil.move(FILTERED, RAW)
        print(f"Moved unfiltered {FILTERED.name} -> {RAW.name}")

    pool = {}
    for name in ("train_pool.jsonl", "bfcl_pool.jsonl"):
        for r in load(D / name):
            pool[r["trajectory_id"]] = r

    raw = load(RAW)
    kept, dropped = [], Counter()
    for r in raw:
        c = r["error_class"]
        base = pool[r["trajectory_id"].rsplit("::corrupt", 1)[0]]
        orig = base["candidate_tool_call"]
        is_bfcl = r.get("source_dataset") == "bfcl"

        if c == 1:
            ok = new_values_all_in_context(r, orig)
            reason = "new value not in context"
        elif c == 2:
            ok = has_user_correction(r["turns"])
            reason = "no user correction in context"
        elif c == 5:
            if is_bfcl:
                ok, reason = False, "BFCL (no tool-result turns)"
            else:
                ok = new_values_all_in_context(r, orig)
                reason = "new value not in context"
        else:
            ok, reason = True, ""

        if ok:
            kept.append(r)
        else:
            dropped[(c, reason)] += 1

    print(f"Raw LLM negatives: {len(raw)}  ->  kept {len(kept)}")
    print("Raw per class:     ", dict(sorted(Counter(r["error_class"] for r in raw).items())))
    print("Kept per class:    ", dict(sorted(Counter(r["error_class"] for r in kept).items())))
    for (c, reason), n in sorted(dropped.items()):
        print(f"  dropped class {c}: {n}  ({reason})")

    with open(FILTERED, "w") as fh:
        for r in kept:
            fh.write(json.dumps(r) + "\n")

    prog = load(PROG)
    merged = prog + kept
    random.Random(42).shuffle(merged)
    # Dropped after the shuffle so every remaining row keeps its position.
    n_before = len(merged)
    merged = [r for r in merged if not (r["corruption_method"] == "llm" and r["error_class"] in (2, 5))]
    n_llm_25_dropped = n_before - len(merged)
    n_before = len(merged)
    merged = [r for r in merged if r["error_class"] not in EXCLUDED_CLASSES]
    n_excluded = n_before - len(merged)
    gold, quality_dropped, kept_rows = load_bfcl_gold(), Counter(), []
    for r in merged:
        if r["corruption_method"] == "llm" and r["error_class"] in (1, 3):
            reasons = quality_reasons(r, pool[r["trajectory_id"].rsplit("::corrupt", 1)[0]]["candidate_tool_call"], gold)
            if reasons:
                quality_dropped[(r["error_class"], reasons[0])] += 1
                continue
        kept_rows.append(r)
    merged = kept_rows
    extra = load(EXTRA) if EXTRA.exists() else []
    merged += extra
    c1v2_in = load(CLASS1_V2) if CLASS1_V2.exists() else []
    c1v2 = [r for r in c1v2_in
            if not quality_reasons(r, pool[r["trajectory_id"].rsplit("::corrupt", 1)[0]]["candidate_tool_call"], gold)
            and new_values_all_in_context(r, pool[r["trajectory_id"].rsplit("::corrupt", 1)[0]]["candidate_tool_call"])]
    merged += c1v2

    # Two generators can corrupt the same source into the same call (an LLM Class 1 row and a
    # programmatic Class 5 row did). Keep one row per (turns, candidate): the programmatic one
    # if there is a choice, otherwise the first. Surviving rows keep their positions.
    def _content_key(r):
        return hashlib.sha256((json.dumps(r["turns"], sort_keys=True) + json.dumps(r["candidate_tool_call"], sort_keys=True)).encode()).hexdigest()

    by_key = {}
    for i, r in enumerate(merged):
        by_key.setdefault(_content_key(r), []).append(i)
    drop = set()
    for idxs in by_key.values():
        if len(idxs) > 1:
            keep = next((i for i in idxs if merged[i]["corruption_method"] == "programmatic"), idxs[0])
            drop |= {i for i in idxs if i != keep}
            print(f"  content duplicate: keeping {merged[keep]['trajectory_id'][-45:]} (class {merged[keep]['error_class']}), "
                  f"dropping {[(merged[i]['error_class'], merged[i]['corruption_method']) for i in idxs if i != keep]}")
    n_content_dups = len(drop)
    merged = [r for i, r in enumerate(merged) if i not in drop]

    ids = [r["trajectory_id"] for r in merged]
    assert len(ids) == len(set(ids)), "duplicate trajectory_ids in merged output"
    with open(FINAL, "w") as fh:
        for r in merged:
            fh.write(json.dumps(r) + "\n")

    controls = [make_control(r) for r in merged if r["error_class"] == 2 and r["corruption_method"] == "programmatic"]
    control_ids = [c["trajectory_id"] for c in controls]
    assert len(control_ids) == len(set(control_ids)) and not set(control_ids) & set(ids), "control id collision"
    with open(CONTROLS, "w") as fh:
        for c in controls:
            fh.write(json.dumps(c) + "\n")

    print(f"\nWrote {len(kept)} filtered LLM negatives to {FILTERED.name}")
    print(f"Dropped {n_llm_25_dropped} LLM Class 2/5 rows (replaced by programmatic generators)")
    print(f"Excluded {n_excluded} rows of class(es) {sorted(EXCLUDED_CLASSES)} (not trained on, see D-009)")
    for (c, q), n in sorted(quality_dropped.items()):
        print(f"Quality filter dropped {n} class {c} rows: {q}")
    print(f"Dropped {n_content_dups} content-duplicate negatives (same turns + candidate as another row)")
    print(f"Wrote {len(merged)} merged negatives to {FINAL.name} "
          f"({len(prog)} programmatic + {len(kept) - n_llm_25_dropped} LLM, then {len(extra)} appended Class 2/5)")
    if c1v2_in:
        print(f"Appended {len(c1v2)} of {len(c1v2_in)} Class 1 context-swap rows from {CLASS1_V2.name}")
    print(f"Wrote {len(controls)} paired Class 2 controls to {CONTROLS.name}")
    print("\nFinal count per class:")
    final_counts = Counter((r["error_class"], r["error_class_name"], r["corruption_method"]) for r in merged)
    for (c, name, method), n in sorted(final_counts.items()):
        print(f"  Class {c} ({name}, {method}): {n}")


if __name__ == "__main__":
    main()
