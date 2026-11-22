"""Phase 5 prep: paired control positives for the Class 4 and Class 6 negatives.

Both classes inject a synthetic artifact into the context that no positive has:
  Class 4: a copy of the candidate call + a fake {"status": "success"} result
  Class 6: a fabricated failing predecessor call + an error result
Without matching positives, a model can flag those negatives by the artifact alone.
Each control keeps the artifact but removes the error, so the label depends on the
reasoning, not the pattern (same idea as positives_class2_controls.jsonl).

  Class 4 control: same position, same fake success result, but the injected call is a
                   DIFFERENT call that already appears in the turns, so the candidate is
                   not a duplicate.
  Class 6 control: same fabricated predecessor call at the same position, but with a
                   success result, so proceeding is correct.

Reads  data/derived/negatives_pool_final.jsonl, train_pool.jsonl, bfcl_pool.jsonl,
       positives_class2_controls.jsonl
Writes data/derived/positives_class4_controls.jsonl, positives_class6_controls.jsonl
Deterministic (seed 42).
"""

import copy
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

D = Path(__file__).parent / "derived"
SUCCESS_4 = '{"status": "success"}'  # exactly what corrupt_class4 injects
ERROR_6 = '{"status": "error", "message": "Operation failed: prerequisite not met"}'  # corrupt_llm class 6
# success twin of ERROR_6, same shape and wording so only the outcome differs
SUCCESS_6 = '{"status": "success", "message": "Operation completed: prerequisite met"}'


def load(name: str) -> list[dict]:
    return [json.loads(line) for line in open(D / name) if line.strip()]


def same_call(a: dict, b: dict) -> bool:
    return a["name"] == b["name"] and a["arguments"] == b["arguments"]


def content_key(r: dict) -> str:
    return hashlib.sha256((json.dumps(r["turns"], sort_keys=True) + json.dumps(r["candidate_tool_call"], sort_keys=True)).encode()).hexdigest()


def to_control(neg: dict, error_class: int, suffix: str, tier: str, detail: dict) -> dict:
    ctrl = copy.deepcopy(neg)
    ctrl["label"] = "correct"
    ctrl["label_tier"] = tier
    ctrl["trajectory_id"] = neg["trajectory_id"].removesuffix(f"::corrupt{error_class}") + suffix
    ctrl["paired_negative_id"] = neg["trajectory_id"]
    ctrl.pop("error_class")
    ctrl.pop("error_class_name")
    ctrl["corruption_method"] = "control"
    ctrl["corruption_detail"] = detail
    return ctrl


def locate_class4_injection(neg: dict) -> int:
    t, cand = neg["turns"], neg["candidate_tool_call"]
    idx = [
        i for i in range(len(t) - 1)
        if t[i].get("role") == "assistant_tool_call" and same_call(t[i], cand)
        and t[i + 1].get("role") == "tool" and t[i + 1].get("content") == SUCCESS_4
    ]
    assert len(idx) == 1, f"expected exactly one injected pair in {neg['trajectory_id']}"
    return idx[0]


def main():
    rng = random.Random(42)
    negs = load("negatives_pool_final.jsonl")
    orig = {r["trajectory_id"]: r for r in load("train_pool.jsonl") + load("bfcl_pool.jsonl")}
    existing_ids = {r["trajectory_id"] for r in load("positives_class2_controls.jsonl")}

    # ---------------- Class 4 ----------------
    c4, skips4 = [], Counter()
    for neg in (r for r in negs if r["error_class"] == 4):
        base = orig[neg["trajectory_id"].removesuffix("::corrupt4")]
        i = locate_class4_injection(neg)
        assert neg["turns"][:i] + neg["turns"][i + 2:] == base["turns"], "negative is not original + injected pair"
        cand = base["candidate_tool_call"]
        calls = [t for t in base["turns"] if t.get("role") == "assistant_tool_call"]
        others = [t for t in calls if not same_call(t, cand)]
        if len(others) < len(calls):
            skips4["source positive already contains its own candidate call earlier"] += 1
            continue
        if not others:
            skips4["no other tool call in the turns to inject"] += 1
            continue
        pick = rng.choice(others)
        ctrl = to_control(neg, 4, "::ctrl4", "synthetic_class4_control",
                          {"injected_call": {"name": pick["name"], "arguments": pick["arguments"]}, "injected_at_turn": i})
        ctrl["turns"][i]["name"] = pick["name"]
        ctrl["turns"][i]["arguments"] = copy.deepcopy(pick["arguments"])
        if ctrl["trajectory_id"] in existing_ids:
            skips4["trajectory_id already in positives_class2_controls.jsonl"] += 1
            continue
        c4.append(ctrl)

    # ---------------- Class 6 ----------------
    c6, skips6 = [], Counter()
    for neg in (r for r in negs if r["error_class"] == 6):
        base = orig[neg["trajectory_id"].removesuffix("::corrupt6")]
        t = neg["turns"]
        assert t[:-2] == base["turns"] and t[-2].get("role") == "assistant_tool_call" and t[-1] == {"role": "tool", "content": ERROR_6}, \
            f"unexpected class 6 structure in {neg['trajectory_id']}"
        ctrl = to_control(neg, 6, "::ctrl6", "synthetic_class6_control",
                          {"predecessor_call": {"name": t[-2]["name"], "arguments": t[-2]["arguments"]}, "predecessor_result": "success"})
        ctrl["turns"][-1] = {"role": "tool", "content": SUCCESS_6}
        if ctrl["trajectory_id"] in existing_ids:
            skips6["trajectory_id already in positives_class2_controls.jsonl"] += 1
            continue
        c6.append(ctrl)

    # ---------------- checks across everything that will be training positives ----------------
    all_ids = [r["trajectory_id"] for r in c4 + c6] + list(existing_ids) + list(orig)
    assert len(all_ids) == len(set(all_ids)), "trajectory_id collision among positives"
    keys = Counter(content_key(r) for r in c4 + c6 + load("positives_class2_controls.jsonl") + list(orig.values()))
    dup_content = sum(v - 1 for v in keys.values() if v > 1)

    for name, rows in (("positives_class4_controls.jsonl", c4), ("positives_class6_controls.jsonl", c6)):
        with open(D / name, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")

    print(f"Class 4 controls: {len(c4)} written of {sum(1 for r in negs if r['error_class'] == 4)} negatives")
    for reason, n in skips4.items():
        print(f"  skipped {n}: {reason}")
    print(f"Class 6 controls: {len(c6)} written of {sum(1 for r in negs if r['error_class'] == 6)} negatives")
    for reason, n in skips6.items():
        print(f"  skipped {n}: {reason}")
    if not skips4.get("trajectory_id already in positives_class2_controls.jsonl") and not skips6:
        print("Dedup vs positives_class2_controls.jsonl: 0 skipped (class 2 controls end '::ctrl'; these end '::ctrl4' / '::ctrl6', so ids cannot collide)")
    print(f"Content-identical (turns + candidate) rows across all positives: {dup_content}")
    print("Class 4 controls by source:", dict(Counter("bfcl" if r["domain"] == "bfcl" else "tau2" for r in c4)))
    print("Class 6 controls by source:", dict(Counter("bfcl" if r["domain"] == "bfcl" else "tau2" for r in c6)))


if __name__ == "__main__":
    main()
