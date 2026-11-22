"""Heuristic pre-fill for Class 7 (policy violation) decisions (docs/labeling_guide.md). It reads STRUCTURE only: it never reads the policy, so it only
suggests; the reviewer confirms or changes it (queue 5 of data/review_queues.py), and a row stores `class7_reason_source` = human or heuristic.

    from class7_reasons import suggest, is_read_only
    suggest(turns, {"name": ..., "arguments": {...}}, domain)  ->  {"verdict": "class7" | "correct" | "class4" | None, "reason": str | None,
                                                                    "tag": "parallel_lookup" | None, "signals": [str, ...]}

`turns` are the turns BEFORE the call in the training-row format (user, assistant, assistant_tool_call, tool), with the earlier calls of the same
assistant message at the end (data/mine_real_errors.render_full and data/build_final_dataset.history both give that).
"""

import json
import re

REASONS = ["no_confirmation", "ineligible_state", "missing_step", "precondition_error", "parallel_write", "other"]
READ_PREFIXES = ("get_", "search_", "find_", "list_")
READ_EXACT = {"calculate"}
# telecom diagnostics read the device and change nothing (labeling guide, rule 1)
DIAG_PREFIXES = ("check_", "can_")
DIAG_EXACT = {"run_speed_test"}
NO_CONFIRMATION_NEEDED = {"transfer_to_human_agents", "think"}   # not database updates
AFFIRM = re.compile(r"\b(yes|yep|yeah|yup|sure|confirm(ed)?|go ahead|proceed|please do|do it|go for it|ok(ay)?|sounds good|that'?s (right|correct|fine)|"
                    r"that is (right|correct|fine)|approve[d]?|correct)\b", re.I)
ID_KEYS = ("reservation_id", "order_id", "user_id", "customer_id", "line_id", "bill_id")


def is_read_only(name: str) -> bool:
    return name.startswith(READ_PREFIXES) or name in READ_EXACT or name.startswith(DIAG_PREFIXES) or name in DIAG_EXACT


def _outstanding(turns):
    """The calls at the end of `turns` with no tool result in between: issued in the same assistant message as the call being judged."""
    out = []
    for t in reversed(turns):
        if t.get("role") == "assistant_tool_call":
            out.append(t)
        else:
            break
    return out


def _json(text):
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return None


def _latest_object(turns, key, value):
    """Latest tool result (parsed) before the call that describes the object (key == value). Falls back to a regex on text cut by a length cap."""
    best = None
    for t in turns:
        if t.get("role") != "tool":
            continue
        d = _json(t.get("content", ""))
        if isinstance(d, dict) and str(d.get(key)) == str(value):
            best = d
        elif d is None and f'"{key}": "{value}"' in str(t.get("content", "")):
            m = re.search(r'"status": "([^"]+)"', str(t["content"]))
            best = {key: value, **({"status": m.group(1)} if m else {})}
    return best


def _object_seen(turns, value):
    v = str(value)
    for t in turns:
        if t.get("role") == "tool" and v in str(t.get("content", "")):
            return True
        if t.get("role") == "assistant_tool_call" and is_read_only(t["name"]) and v in json.dumps(t.get("arguments", {})):
            return True
    return False


def suggest(turns: list[dict], call: dict, domain: str) -> dict:
    name, args = call["name"], call.get("arguments") or {}
    sig, tag = [], None
    out = {"verdict": None, "reason": None, "tag": None, "signals": sig}
    ro = is_read_only(name)
    # identical earlier call
    for i, t in enumerate(turns):
        if t.get("role") == "assistant_tool_call" and t["name"] == name and t.get("arguments") == args:
            sig.append(f"an identical call {name}(...) was made earlier (turn {i} of the history)")
            break
    outs = _outstanding(turns)
    if ro:
        if outs and all(is_read_only(o["name"]) for o in outs):
            tag = "parallel_lookup"
            sig.append(f"issued in the same assistant message as {len(outs)} other read-only call(s): parallel lookup")
        if any("identical call" in s for s in sig):
            out.update(verdict="class4")
        else:
            out.update(verdict="correct", tag=tag)
            sig.append("read-only call: never Class 7 unless the policy forbids this specific lookup (guide, rule 1)")
        return out
    # ---- a write call
    if outs:
        sig.append(f"issued in the same assistant message as {len(outs)} call(s) whose results were still outstanding: {', '.join(o['name'] for o in outs)}")
        out.update(verdict="class7", reason="parallel_write")
        return out
    user = [t for t in turns if t.get("role") == "user"]
    if domain in ("airline", "retail") and name not in NO_CONFIRMATION_NEEDED:
        last = user[-1]["content"] if user else ""
        if not user or not AFFIRM.search(last):
            sig.append("no affirmative user message (yes, confirm, go ahead, ...) before this write; the last user message is " + (f"{last[:90]!r}" if user else "missing"))
            out.update(verdict="class7", reason="no_confirmation")
            return out
    # eligibility from the latest lookup of the same object
    if domain == "retail" and "order_id" in args:
        o = _latest_object(turns, "order_id", args["order_id"])
        st = (o or {}).get("status")
        need = "pending" if name.startswith(("modify_pending_order", "cancel_pending_order")) else "delivered" if name.endswith("delivered_order_items") else None
        if st and need and st != need:
            sig.append(f"the lookup of order {args['order_id']} says status {st!r}; {name} needs {need!r}")
            out.update(verdict="class7", reason="ineligible_state")
            return out
    if domain == "airline" and name == "update_reservation_flights" and "reservation_id" in args:
        o = _latest_object(turns, "reservation_id", args["reservation_id"]) or {}
        have = {(f.get("flight_number"), f.get("date")) for f in o.get("flights", []) if isinstance(f, dict)}
        want = {(f.get("flight_number"), f.get("date")) for f in args.get("flights", []) if isinstance(f, dict)}
        if o.get("cabin") == "basic_economy" and have and want and have != want:
            sig.append(f"reservation {args['reservation_id']} is basic_economy and the call changes its flights (basic economy flights cannot be modified)")
            out.update(verdict="class7", reason="ineligible_state")
            return out
    if domain == "airline" and name == "cancel_reservation" and "reservation_id" in args:
        o = _latest_object(turns, "reservation_id", args["reservation_id"]) or {}
        if o.get("cabin") and o["cabin"] != "business" and o.get("insurance") == "no" and str(o.get("created_at", "9999")) < "2024-05-14T15:00:00":
            sig.append(f"reservation {args['reservation_id']}: {o['cabin']}, no insurance, booked {o['created_at']} (more than 24 hours before the policy's 'now'); "
                       "cancellation needs an airline-cancelled flight, so check that")
            out.update(verdict="class7", reason="ineligible_state")
            return out
    # was the object ever looked up, and was the user identified
    for k in ("reservation_id", "order_id"):
        if args.get(k) and not _object_seen(turns, args[k]):
            sig.append(f"{k} {args[k]} appears in no earlier lookup or tool result")
            out.update(verdict="class7", reason="missing_step")
            return out
    if domain in ("airline", "retail") and not any(t.get("role") == "assistant_tool_call" and t["name"] in ("get_user_details", "find_user_id_by_email", "find_user_id_by_name_zip") for t in turns):
        sig.append("no user lookup (get_user_details / find_user_id_*) before this write: the user was never authenticated")
        out.update(verdict="class7", reason="missing_step")
        return out
    sig.append("no structural signal: choose the reason from the policy")
    return out
