"""Apply queue 5 by rule instead of by hand (decided 2026-11-15, D-022 update). Nothing here touches review_queue_decisions.jsonl.

    python data/apply_queue5_rules.py apply [--rebuild]    # writes data/derived/review_queue_rule_decisions.jsonl (source: "rule")
    python data/apply_queue5_rules.py undo                 # deletes that file: every rule decision is gone, the human decisions were never touched

Rules (one record per queue-5 item, in the same `q5::<original id>` shape as a human queue-5 decision):
  A  queue-2 item marked wrong as Class 7 on a read-only call      -> correct; tag parallel_lookup when the call was issued with an earlier call's
                                                                      result still outstanding (calls directly before it, no tool result between), else plain
  B  the queue-1 Class 7 item on a read-only call (can_send_mms)    -> the same (not an error; no correct example is added for an error candidate)
  C  read-only item marked wrong with another class (1, 3, 4, ...) -> the original decision is kept
  D  Class 7 decision on a write call (queues 1 and 2, test_real)    -> stands; class7_reason = the heuristic pre-fill, class7_reason_source "heuristic"
                                                                      ("unlabeled" when the heuristic found no signal)
A human queue-5 decision (a key pressed in `review --queue 5`) always wins over the rule record of the same item (data/review_decisions.py).
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DER = HERE / "derived"
sys.path.insert(0, str(HERE))
from review_decisions import RULES_NAME  # noqa: E402

RULES = DER / RULES_NAME


def outstanding_before(it: dict) -> int:
    """Calls directly before the judged call with no tool result between: their results are still outstanding when the call is issued."""
    src = it["source_item"]
    c = src["candidate"] if src["kind"] == "error" else src
    win, turns = c["window"], c["turns"]
    n = 0
    for t in reversed(turns[: win["flagged_offset"]]):
        if t.get("role") != "assistant_tool_call":
            break
        n += 1
    return n


def decide(it: dict) -> dict:
    od, ro, pre = it["orig_decision"], it["call"]["read_only"], it["prefill"]
    cls = od.get("human_class")
    if ro and cls == 7:
        out_n = outstanding_before(it)
        rule = "A" if it["orig_queue"] == 2 else "B"
        rec = {"decision": "correct", "human_label": "correct", "human_class": None, "tag": "parallel_lookup" if out_n else None, "rule": rule,
               "human_note": f"rule {rule}: read-only call is never Class 7" + (f"; issued with {out_n} earlier call(s) outstanding (parallel_lookup)" if out_n else "")}
    elif ro:
        rec = {"decision": "wrong", "human_label": "incorrect", "human_class": cls, "rule": "C", "human_note": f"rule C: read-only, class {cls}: the original decision is kept"}
    else:
        reason = pre["reason"] if pre["verdict"] == "class7" and pre["reason"] else "unlabeled"
        rec = {"decision": "wrong", "human_label": "incorrect", "human_class": 7, "class7_reason": reason, "class7_reason_source": "heuristic", "rule": "D",
               "human_note": f"rule D: the Class 7 decision on a write call stands; reason from the heuristic pre-fill ({reason})"}
    return {"item_id": it["item_id"], "queue": 5, "origin": it["origin"], "kind": "recheck", "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "supersedes": it["orig_item_id"],
            "orig_queue": it["orig_queue"], "orig_kind": it["orig_kind"], "proposed_class": None, "source": "rule",
            "prefill": {k: pre[k] for k in ("verdict", "reason", "tag")}, **rec}


def cmd_apply(args):
    if RULES.exists() and not args.rebuild:
        sys.exit(f"{RULES} exists; use --rebuild to rewrite it (or `undo` to remove it)")
    items = [json.loads(l) for l in open(DER / "review_queue.jsonl") if l.strip()]
    q5 = [i for i in items if i["queue"] == 5]
    recs = [decide(i) for i in q5]
    RULES.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))
    human = {}
    for l in open(DER / "review_queue_decisions.jsonl"):
        d = json.loads(l)
        if d["item_id"].startswith("q5::"):
            human[d["item_id"]] = d
    print(f"wrote {RULES}: {len(recs)} rule decisions for {len(q5)} queue-5 items")
    byg = Counter((r["rule"], r["decision"], r.get("tag"), r.get("class7_reason")) for r in recs)
    for k, v in sorted(byg.items(), key=str):
        print(f"   rule {k[0]}: {k[1]:7s} tag={k[2]} class7_reason={k[3]}: {v}")
    hd = [k for k, d in human.items() if d["decision"] != "skip"]
    print(f"{len(hd)} item(s) already have a human queue-5 decision; those win over the rule record")
    for k in hd:
        r = next(x for x in recs if x["item_id"] == k)
        h = human[k]
        print(f"   {k[-34:]}: human {h['decision']}/{h.get('human_class')}/{h.get('class7_reason')}  vs rule {r['decision']}/{r.get('human_class')}/{r.get('class7_reason')}")


def cmd_undo(args):
    if RULES.exists():
        RULES.unlink()
        print(f"removed {RULES}: no rule decision is applied any more")
    else:
        print("no rule decisions file")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["apply", "undo"])
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    {"apply": cmd_apply, "undo": cmd_undo}[a.stage](a)
