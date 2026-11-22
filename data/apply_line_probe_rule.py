"""Apply the line-probe rule (2026-11-16; docs/labeling_guide.md "Line lookups", D-022) to every `get_details_by_id` call on a line id (L1001 ...) in
queues 1, 2 and 4 and in test_real.jsonl. Nothing here touches review_queue_decisions.jsonl.

    python data/apply_line_probe_rule.py apply [--rebuild]   # writes data/derived/review_queue_lineprobe_decisions.jsonl (ids lp::<original id>, source: "rule")
    python data/apply_line_probe_rule.py undo                # deletes that file: the rule is reversed

The judgement itself is data/line_probe_rule.py. Items it decides (correct, or Class 1 for a lookup of the wrong line) get a rule record; repeated lookups
(out of scope) keep their original decision; items it cannot decide confidently go to queue 6 (`review_queues.py add-queue6`). Never Class 3.
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DER = HERE / "derived"
sys.path.insert(0, str(HERE))
from line_probe_rule import judge  # noqa: E402
from review_decisions import LINEPROBE_NAME, load_raw  # noqa: E402

OUT = DER / LINEPROBE_NAME


def collect() -> list[dict]:
    """Every in-scope-or-not line-id lookup with the rule's judgement: dicts with item (queue item or test_real row), kind, call, turns-free judge result."""
    from mine_real_errors import render_full, runs_by_file

    items = [json.loads(l) for l in open(DER / "review_queue.jsonl") if l.strip()]
    sel = []
    for i in items:
        if i["queue"] in (1, 2, 4):
            c = i["candidate"] if i["kind"] == "error" else i["positive"]
            fc = c["flagged_call"] if i["kind"] == "error" else c
            if fc["name"] == "get_details_by_id" and re.fullmatch(r"L\d+", str(fc["arguments"].get("id", ""))):
                sel.append((i, c, fc))
    keys = {c["run_key"] for i, c, fc in sel if i["origin"] == "natural"}
    msgs = {f"{s['file']}::{s['sim_id']}": s["messages"] for s in runs_by_file(keys)}
    for r in (json.loads(l) for l in open(HERE / "raw" / "induced" / "induced_runs.jsonl")):
        msgs[r["run_id"]] = r["messages"]
    out = []
    for i, c, fc in sel:
        turns, ct = render_full(msgs[c["run_key"]])
        t = ct[fc["call_index"]]
        call = {"name": turns[t]["name"], "arguments": turns[t]["arguments"]}
        out.append({"kind": "queue", "item": i, "orig_id": i["item_id"], "orig_kind": i["kind"], "origin": i["origin"], "domain": c["domain"], "task_id": c["task_id"],
                    "call": call, "judge": judge(turns[:t], call)})
    for r in (json.loads(l) for l in open(DER / "test_real.jsonl")):
        c = r["candidate_tool_call"]
        if c["name"] == "get_details_by_id" and re.fullmatch(r"L\d+", str(c["arguments"].get("id", ""))):
            out.append({"kind": "test_real", "item": r, "orig_id": f"test_real::{r['trajectory_id']}", "orig_kind": "test_real", "origin": "natural", "domain": r["domain"],
                        "task_id": r["task_id"], "call": {"name": c["name"], "arguments": c["arguments"]}, "judge": judge(r["turns"], c)})
    return out


def record(x: dict) -> dict:
    j = x["judge"]
    rec = {"item_id": "lp::" + x["orig_id"], "queue": "lp", "origin": x["origin"], "kind": "recheck", "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "supersedes": x["orig_id"],
           "orig_kind": x["orig_kind"], "proposed_class": None, "source": "rule", "rule": "line_probe", "judge": {k: j[k] for k in ("verdict", "target", "why", "evidence")},
           "human_note": f"line-probe rule: {j['why']}"}
    if j["verdict"] == "correct":
        rec.update(decision="correct", human_label="correct", human_class=None)
    else:
        rec.update(decision="wrong", human_label="incorrect", human_class=1)
    if x["kind"] == "test_real":
        r = x["item"]
        rec["orig_base"] = {"item_id": x["orig_id"], "queue": 0, "origin": "natural", "kind": "test_real", "decision": "hand_label", "human_label": r["label"],
                            "human_class": int(r["error_class"]) if r["label"] == "incorrect" and r.get("error_class") else None, "human_note": "", "proposed_class": 7}
    return rec


def cmd_apply(args):
    if OUT.exists() and not args.rebuild:
        sys.exit(f"{OUT} exists; use --rebuild to rewrite it (or `undo` to remove it)")
    xs = collect()
    recs = [record(x) for x in xs if x["judge"]["verdict"] in ("correct", "class1")]
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))
    dec = load_raw(DER / "review_queue_decisions.jsonl")
    print(f"line-id lookups found: {len(xs)} ({sum(x['kind'] == 'queue' for x in xs)} in queues 1/2/4, {sum(x['kind'] == 'test_real' for x in xs)} in test_real)")
    print(f"wrote {OUT}: {len(recs)} rule decisions")
    c = Counter()
    for x in xs:
        d = dec.get(x["orig_id"])
        before = ("test_real " + x["item"]["label"] + (f" class {x['item'].get('error_class')}" if x["item"]["label"] == "incorrect" else "")) if x["kind"] == "test_real" else \
            f"queue {x['item']['queue']} {d['decision']}" + (f" class {d['human_class']}" if d and d.get("human_class") else "")
        c[(before, x["judge"]["verdict"])] += 1
    print("original decision -> rule verdict:")
    for (b, v), n in sorted(c.items()):
        print(f"   {n:3d}  {b:32s} -> {v}")
    und = [x for x in xs if x["judge"]["verdict"] == "undecided"]
    print(f"undecided (queue 6): {len(und)}; out of scope (repeats, original decision stays): {sum(x['judge']['verdict'] == 'out_of_scope' for x in xs)}")


def cmd_undo(args):
    if OUT.exists():
        OUT.unlink()
        print(f"removed {OUT}: the line-probe rule is no longer applied")
    else:
        print("no line-probe decisions file")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["apply", "undo"])
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    {"apply": cmd_apply, "undo": cmd_undo}[a.stage](a)
