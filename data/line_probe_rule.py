"""The line-probe rule (2026-11-16, labeling guide section "Line lookups" and D-022): how to judge `get_details_by_id` calls on a line id (L1001 ...).

    from line_probe_rule import judge
    judge(turns_before_call, call) -> {"verdict": "correct" | "class1" | "undecided" | "out_of_scope", "target": line id or None, "evidence": [str, ...], "why": str}

The rule (structural, no model):
  * KNOWABLE: the target line was identifiable before the call, because (a) the user named a line id (L1002) or the customer has a single line, or (b) the user stated a phone number and an
    earlier tool result tied that number to a line id (a line record with both `line_id` and `phone_number`), or (c) an earlier tool result other than the
    customer's `line_ids` list names exactly one line id. If the agent looked up a different line it is Class 1; the same line, correct.
  * NOT KNOWABLE: the customer record lists line ids with no phone numbers, so a lookup to find out which line is affected is correct (not an error).
  * NEVER Class 3: no other tool should have been used.
  * UNDECIDED (goes to queue 6): the structure does not decide it confidently: several distinct phone numbers or line ids were stated, or a stated number
    matches two line records, or the tied line differs between sources.
  * OUT OF SCOPE: the same lookup was already made earlier in the conversation (a repeated call, Class 4 territory), or the id is not a line id.
"""

import json
import re

LINE = re.compile(r"\bL\d{3,}\b")
PHONE = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")


def norm_phone(s: str) -> str:
    return re.sub(r"\D", "", s)


def _records(content):
    try:
        d = json.loads(content)
    except (ValueError, TypeError):
        return []
    return [d] if isinstance(d, dict) else [x for x in d if isinstance(x, dict)] if isinstance(d, list) else []


def judge(turns: list[dict], call: dict) -> dict:
    ev, x = [], str((call.get("arguments") or {}).get("id", ""))
    out = {"verdict": "out_of_scope", "target": None, "evidence": ev, "why": ""}
    if call["name"] != "get_details_by_id" or not re.fullmatch(r"L\d+", x):
        out["why"] = "not a lookup of a line id"
        return out
    earlier = [t for t in turns if t.get("role") == "assistant_tool_call" and t["name"] == "get_details_by_id" and str((t.get("arguments") or {}).get("id")) == x]
    if earlier:
        out["why"] = f"{x} was already looked up earlier in the conversation (a repeated call, not governed by this rule)"
        return out
    user_text = " ".join(t["content"] for t in turns if t.get("role") == "user" and t.get("content"))
    stated_lines = sorted(set(LINE.findall(user_text)))
    stated_phones = sorted({norm_phone(p) for p in PHONE.findall(user_text)})
    ev.append(f"user stated line ids {stated_lines or 'none'}, phone numbers {stated_phones or 'none'}")
    # line records seen in earlier tool results: phone -> line ids
    by_phone, tied, only_line = {}, set(), set()
    for t in turns:
        if t.get("role") != "tool":
            continue
        for rec in _records(t.get("content", "")):
            if "line_id" in rec and "phone_number" in rec:
                by_phone.setdefault(norm_phone(str(rec["phone_number"])), set()).add(rec["line_id"])
            elif "line_id" in rec:                       # a bill or similar record naming one line
                tied.add(rec["line_id"])
            if isinstance(rec.get("line_ids"), list) and len(rec["line_ids"]) == 1:   # the customer has a single line: it is the target
                only_line.add(rec["line_ids"][0])
    if by_phone:
        ev.append("earlier line records tie phone numbers to lines: " + ", ".join(f"{p}->{sorted(v)}" for p, v in by_phone.items()))
    if tied:
        ev.append(f"earlier records name line ids {sorted(tied)}")
    targets = set()
    if len(stated_lines) == 1:
        targets.add(stated_lines[0])
    elif len(stated_lines) > 1:
        out.update(verdict="undecided", why=f"the user named several line ids {stated_lines}")
        return out
    if len(stated_phones) > 1:
        # the customer's own number is often repeated; only a conflict between line-mapped numbers matters
        mapped = {p for p in stated_phones if p in by_phone}
        if len(mapped) > 1:
            out.update(verdict="undecided", why="the user stated several phone numbers that match different lines")
            return out
        stated_phones = sorted(mapped) or stated_phones
    for p in stated_phones:
        if p in by_phone:
            if len(by_phone[p]) > 1:
                out.update(verdict="undecided", why=f"phone number {p} matches several line records {sorted(by_phone[p])}")
                return out
            targets |= by_phone[p]
    if not targets and len(only_line) == 1:
        targets |= only_line
        ev.append(f"the customer record lists a single line {sorted(only_line)}")
    if not targets and len(tied) == 1:
        targets |= tied
    if len(targets) > 1:
        out.update(verdict="undecided", why=f"the sources point at different lines {sorted(targets)}")
        return out
    if not targets:
        if tied and len(tied) > 1:
            out.update(verdict="undecided", why=f"earlier records name several line ids {sorted(tied)} and none is tied to the reported problem")
            return out
        out.update(verdict="correct", why="not knowable: nothing before the call tied the reported problem to one line, so the lookup is a reasonable way to find out which line is affected")
        return out
    t = next(iter(targets))
    out["target"] = t
    if x == t:
        out.update(verdict="correct", why=f"knowable: the target line is {t} and this call looks up {t}")
    else:
        out.update(verdict="class1", why=f"knowable: the target line is {t} (from the evidence above) but the call looks up {x}")
    return out
