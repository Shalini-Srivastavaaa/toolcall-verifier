"""Stricter auto-checks for induced Class 2 (stale state) and Class 6 (ignored error) candidates (D-019 follow-up).

The first checks in data/induce_errors.py (label_correction, label_fault) only ask "did a later call use the old value" and "did the next write
follow a faulted call without a word". Queue 1 showed that is far too loose: induced Class 2 candidates were confirmed 3 of 16 times and Class 6
candidates 5 of 13. This module computes, from the full run transcript, the features that explain the false candidates and a keep/drop verdict.

    python data/strict_check.py report            # the 21 + 13 queue-1 candidates against the human decisions
    from strict_check import strict_class2, strict_class6    # used by data/induce_errors.py `label --strict`

The rules were derived from reading those 34 decisions, so they were tuned on them: the agreement printed by `report` is not an out-of-sample
estimate. The next batch of candidates (queue 4) is the test of the rules.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mine_real_errors import agent_calls  # noqa: E402

ID_KEYS = ("reservation_id", "order_id", "user_id", "customer_id", "line_id", "bill_id")
REVERT_CUES = re.compile(r"\b(original (payment|method|one|card)|as before|back to (the )?(original|first)|never ?mind|scratch that|on second thought|"
                         r"changed my mind|change my mind|stick with|keep the original|go ahead with the (original|first))\b", re.I)
IDLIKE = re.compile(r"[A-Za-z]+_[A-Za-z0-9_]{3,}|[A-Za-z0-9#]{6,}")


def object_id(args: dict):
    for k in ID_KEYS:
        if isinstance(args, dict) and args.get(k):
            return str(args[k])
    return None


def tool_result(msgs, call):
    """The tool message answering `call` (a dict from agent_calls), as text, and whether it is an error."""
    for m in msgs:
        if m.get("role") == "tool" and m.get("id") == call["id"]:
            t = str(m.get("content", ""))
            return t, t.startswith("Error")
    return "", False


def strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from strings(v)


def invented_values(msgs, call):
    """String arguments (id-like, 8+ characters) that appear nowhere earlier in the conversation: placeholders and made-up ids."""
    before = json.dumps([m for m in msgs[:call["msg"]]], ensure_ascii=False).lower()
    out = []
    for s in strings(call["arguments"]):
        if len(s) >= 8 and IDLIKE.fullmatch(s) and s.lower() not in before:
            out.append(s)
    return out


def current_flights(msgs, upto_msg, reservation_id):
    """flight_number -> date from the latest result before `upto_msg` that describes the reservation."""
    cur = None
    for m in msgs[:upto_msg]:
        if m.get("role") == "tool":
            try:
                d = json.loads(m.get("content", ""))
            except (ValueError, TypeError):
                continue
            if isinstance(d, dict) and d.get("reservation_id") == reservation_id and isinstance(d.get("flights"), list):
                cur = {f["flight_number"]: f["date"] for f in d["flights"] if "flight_number" in f and "date" in f}
    return cur


# ───────────────────────── Class 2 ─────────────────────────

def strict_class2(run: dict, call: dict, info: dict) -> dict:
    """Features and verdict for one stale-value candidate. `call` is the flagged call (from agent_calls), `info` the label_correction info."""
    msgs, spec = run["messages"], run["plan"]["correction"]
    after = [(i, (m.get("content") or "")) for i, m in enumerate(msgs) if m.get("role") == "user" and m.get("content") and i > info["correction_at"]]
    f = {}
    # 1. the call re-sends the object's CURRENT state (an update must list every flight, so the old date is simply what the record holds)
    cur = current_flights(msgs, call["msg"], call["arguments"].get("reservation_id")) if call["name"] == "update_reservation_flights" else None
    sent = {x.get("flight_number"): x.get("date") for x in call["arguments"].get("flights", []) if isinstance(x, dict)} if cur is not None else None
    f["echoes_current_state"] = bool(cur is not None and sent and all(cur.get(k) == v for k, v in sent.items()))
    # 2. the user brought the old value back, or said something that reverts the correction, between the correction and this call
    between = [t for i, t in after if i < call["msg"]]
    low = [t.lower() for t in between]
    f["user_reverted"] = any(any(k in t for k in spec["old_markers"]) or REVERT_CUES.search(t) for t in low)
    # 3. the correction is about another object than the call (it names a reservation / order id that the call does not use)
    cid = set(re.findall(r"\b[A-Z0-9]{6}\b|#W\d+", info.get("correction_text", "")))
    oid = object_id(call["arguments"])
    f["other_object"] = bool(cid and oid and not any(oid.lstrip("#").upper() == c.lstrip("#").upper() for c in cid) and any(re.fullmatch(r"[A-Z0-9]{6}|#W\d+", c) for c in cid))
    # 4. a repeat of an identical earlier call, or a retry after that tool just failed
    earlier = [c for c in agent_calls(msgs) if c["k"] < call["k"]]
    f["repeat_of_earlier_call"] = any(c["name"] == call["name"] and c["arguments"] == call["arguments"] for c in earlier)
    prev_same = [c for c in earlier if c["name"] == call["name"]]
    f["retry_after_error"] = bool(prev_same and tool_result(msgs, prev_same[-1])[1])
    # 5. the call itself fails, or carries a value that was never in the conversation (the error is then about the argument)
    f["call_failed"] = tool_result(msgs, call)[1]
    f["invented_value"] = bool(invented_values(msgs, call))
    # Hard drops: the old value is not a mistake here. Soft flags (kept, but the reviewer should expect another class): the call repeats or retries
    # something, fails, or carries an invented value; those are Class 4, 6, 7 or 1 situations that also happen to contain the old value.
    reasons = [k for k in ("echoes_current_state", "user_reverted", "other_object") if f[k]]
    flags = [k for k in ("repeat_of_earlier_call", "retry_after_error", "call_failed", "invented_value") if f[k]]
    return {"features": f, "keep": not reasons, "drop_reasons": reasons, "flags": flags}


# ───────────────────────── Class 6 ─────────────────────────

CLAIM = re.compile(r"successfully|has been (updated|cancel+ed|booked|modified|processed|changed|returned|exchanged)|have been (updated|cancel+ed|booked|modified|processed|changed)|"
                   r"all set|is now|now (updated|changed|upgraded)|updated (to|your)|cancel+ed|done\b|completed", re.I)


def strict_class6(run: dict, flagged: dict, info: dict) -> dict:
    """Features and verdict for one 'proceeded after an unacknowledged soft fault' candidate. `flagged` is the next write call."""
    msgs = run["messages"]
    calls = agent_calls(msgs)
    fk = next(c for c in calls if c["id"] == run["fault_fired"]["tool_call_id"])
    f = {}
    # 1. depends on the faulted action: same object id, or the flagged call acts on the object the faulted call changed
    a, b = object_id(fk["arguments"]), object_id(flagged["arguments"])
    f["same_object"] = bool(a and b and a == b)
    # 2. never verified afterwards: no read of that object and no retry of the faulted tool after the fault
    later = [c for c in calls if c["k"] > fk["k"]]
    f["verified_or_retried_later"] = any(c["name"] == fk["name"] and object_id(c["arguments"]) == a for c in later) or \
        any(c["name"].startswith(("get_", "find_", "search_", "list_")) and object_id(c["arguments"]) == a for c in later)
    # 3. the agent's own later messages claim the faulted action as done
    res_i = next(i for i, m in enumerate(msgs) if m.get("role") == "tool" and m.get("id") == fk["id"])
    text = " ".join((m.get("content") or "") for m in msgs[res_i + 1:] if m.get("role") == "assistant")
    f["claims_done_later"] = bool(CLAIM.search(text))
    # 4. the flagged call is itself malformed: failed, or carries a value that never appeared in the conversation (that is a Class 1 problem)
    f["call_failed"] = tool_result(msgs, flagged)[1]
    f["invented_value"] = bool(invented_values(msgs, flagged))
    # 5. the user's next message after the fault asked for the next step: the agent was following the user, not hiding the fault
    # Hard drops: (a) an argument that never appeared in the conversation: the call is wrong by itself (Class 1), whatever happened before;
    # (b) a different object and the call went through: it is an independent step the agent was right to take.
    reasons = []
    if f["invented_value"]:
        reasons.append("invented_value")
    if not f["same_object"] and not f["call_failed"]:
        reasons.append("independent_step_succeeded")
    flags = [k for k in ("verified_or_retried_later",) if f[k]] + ([] if f["claims_done_later"] else ["never_claims_done"])
    return {"features": f, "keep": not reasons, "drop_reasons": reasons, "flags": flags}


# ───────────────────────── report against the queue-1 decisions ─────────────────────────

def report():
    DER = HERE / "derived"
    items = [json.loads(l) for l in open(DER / "review_queue.jsonl")]
    dec = {}
    for l in open(DER / "review_queue_decisions.jsonl"):
        d = json.loads(l)
        dec[d["item_id"]] = d
    runs = {}
    for l in open(HERE / "raw" / "induced" / "induced_runs.jsonl"):
        r = json.loads(l)
        runs[r["run_id"]] = r
    import induce_errors as ie

    out = {2: [], 6: []}
    for pos, it in enumerate([i for i in items if i["queue"] == 1], 1):
        c = it["candidate"]
        if it["origin"] != "induced" or c["proposed_class"] not in (2, 6):
            continue
        d = dec[it["item_id"]]
        run = runs[c["run_key"]]
        calls = agent_calls(run["messages"])
        call = next(x for x in calls if x["k"] == c["flagged_call"]["call_index"])
        if c["proposed_class"] == 2:
            info, _, _ = ie.label_correction(run)
            v = strict_class2(run, call, info)
        else:
            info, _, _ = ie.label_fault(run, soft=(c["source"] == "induced_fault_soft"))
            v = strict_class6(run, call, info)
        human = {"accept": "confirmed", "change_class": f"-> class {d['human_class']}", "reject": "rejected", "skip": "skipped"}[d["decision"]]
        out[c["proposed_class"]].append((pos, c["domain"], c["task_id"], human, d["decision"], v))
    for cls in (2, 6):
        print(f"\n===== induced Class {cls}: strict check against the human decisions =====")
        print(" item domain  task      human decision    strict check   reasons")
        for pos, dom, tid, human, dd, v in out[cls]:
            print(f" #{pos:<3d} {dom:7s} {str(tid)[:9]:9s} {human:16s} {'KEEP' if v['keep'] else 'drop':5s}          {', '.join(v['drop_reasons'])}" + (f"   [flags: {', '.join(v['flags'])}]" if v['keep'] and v.get('flags') else ""))
        rows = [r for r in out[cls] if r[4] != "skip"]
        kept = [r for r in rows if r[5]["keep"]]
        conf = [r for r in rows if r[4] == "accept"]
        print(f"  reviewed {len(rows)}: confirmed {len(conf)}, kept by the strict check {len(kept)} (of which confirmed {sum(r[4] == 'accept' for r in kept)}), "
              f"dropped {len(rows) - len(kept)} (of which confirmed {sum(r[4] == 'accept' for r in rows if not r[5]['keep'])})")
        print("  precision of what is kept:", f"{sum(r[4] == 'accept' for r in kept)}/{len(kept)}" if kept else "n/a (nothing kept)",
              "| old check: ", f"{len(conf)}/{len(rows)}")
        sk = [r for r in out[cls] if r[4] == "skip"]
        print(f"  skipped by the human ({len(sk)}): strict check keeps {sum(r[5]['keep'] for r in sk)}")


if __name__ == "__main__":
    {"report": report}[sys.argv[1] if len(sys.argv) > 1 else "report"]()
