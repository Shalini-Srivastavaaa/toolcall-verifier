"""Rule-based Class 7 (policy violation; was "missing precondition / wrong order") checker, for inference time.

Class 7 is not learned by the ML classifier (D-009): a violated precondition is a fact about the
call history, so it is checked with the PREREQUISITES table in data/tool_schemas.py instead.

A candidate write call is flagged when none of the lookup tools its rule names has been called
successfully earlier in the conversation. "Successfully" means the tool's result contains none of
"error", "failed", "not found" (case-insensitive).

Our rows store tool results as `{"role": "tool", "content": ...}` with no tool name, and the call
as a separate `{"role": "assistant_tool_call", "name": ..., "arguments": ...}` turn. So a result is
attributed to its call by order: results answer the pending calls first-in, first-out (a message
with parallel calls is rendered as consecutive call turns followed by consecutive result turns).

Rules that name no lookup tool (e.g. "explicit user confirmation") cannot be checked against the call
history and are skipped; a tool whose only rules are of that kind gets reason "no_checkable_rule".

    python eval/class7_checker.py             # score test_real.jsonl
"""

import json
import sys
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data"))
from tool_schemas import ALL_SOURCES, PREREQUISITES  # noqa: E402

ERROR_SIGNALS = ("error", "failed", "not found")


def _candidate_name(row: dict) -> str | None:
    """The candidate tool name. Rows use `candidate_tool_call`; `candidate` is accepted too."""
    cand = row.get("candidate_tool_call", row.get("candidate"))
    return cand.get("name") if isinstance(cand, dict) else cand


def successful_tool_calls(turns: list[dict]) -> list[str]:
    """Names of earlier tool calls whose result carries no error signal, in call order."""
    pending: deque[str] = deque()
    ok = []
    for t in turns:
        if t["role"] == "assistant_tool_call":
            pending.append(t["name"])
        elif t["role"] == "tool" and pending:
            name = pending.popleft()
            if not any(sig in str(t.get("content", "")).lower() for sig in ERROR_SIGNALS):
                ok.append(name)
    return ok


def _rules(domain: str | None, tool: str, sources) -> list | None:
    """Checkable clauses (those that name lookup tools) for `tool`; None if the tool has no entry."""
    domains = [domain] if domain in PREREQUISITES else list(PREREQUISITES)
    for d in domains:
        if tool in PREREQUISITES[d]:
            return [c for c in PREREQUISITES[d][tool] if c.source in sources and c.any_of]
    return None


def check_class7(row: dict, mode: str = "any", sources=ALL_SOURCES) -> dict:
    """Check whether the candidate's preconditions appear in the history.

    row:     an example dict with `turns` and the candidate call (`candidate_tool_call`, or `candidate`)
    mode:    "any"          flag unless at least one required lookup tool succeeded earlier (the default)
             "all_clauses"  flag unless every checkable clause of the rule is met by a successful call
    sources: which clause sources to use; ("policy",) is the clean setting (see D-010 caveat 5)
    returns: {"flagged": bool, "reason": str, ...}
    """
    assert mode in ("any", "all_clauses")
    tool = _candidate_name(row)
    clauses = _rules(row.get("domain"), tool, sources)
    if clauses is None:
        return {"flagged": False, "reason": "no_rule"}
    if not clauses:
        return {"flagged": False, "reason": "no_checkable_rule"}

    done = set(successful_tool_calls(row["turns"]))
    required = sorted({t for c in clauses for t in c.any_of})
    if mode == "any":
        met = bool(done & set(required))
    else:
        met = all(done & set(c.any_of) for c in clauses)
    if met:
        return {"flagged": False, "reason": "precondition_met", "satisfied_by": sorted(done & set(required))}
    return {"flagged": True, "reason": "missing_precondition", "required_one_of": required, "candidate": tool,
            "succeeded_before": sorted(done)}


def load_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in open(path) if line.strip()]


def evaluate(rows: list[dict], **kw) -> dict:
    flags = [check_class7(r, **kw) for r in rows]
    bad = [(r, f) for r, f in zip(rows, flags) if r["label"] == "incorrect"]
    good = [(r, f) for r, f in zip(rows, flags) if r["label"] == "correct"]
    c7 = [(r, f) for r, f in bad if r["error_class"] == 7]
    return {
        "flags": flags, "n_flagged": sum(f["flagged"] for f in flags),
        "false_flags": sum(f["flagged"] for _, f in good), "n_correct": len(good),
        "other_incorrect_flagged": sum(f["flagged"] for r, f in bad if r["error_class"] != 7),
        "n_other_incorrect": len(bad) - len(c7),
        "c7_flagged": sum(f["flagged"] for _, f in c7), "c7": c7,
    }


def main():
    rows = load_jsonl(ROOT / "data" / "derived" / "test_real.jsonl")
    res = evaluate(rows)
    print(f"test_real: {len(rows)} rows ({res['n_correct']} correct, {len(rows) - res['n_correct']} incorrect)\n")
    print("=== Overall (default rule: at least one required lookup succeeded; all rule sources) ===")
    print(f"flagged {res['n_flagged']}/{len(rows)}")
    print(f"false-flag rate on correct examples: {res['false_flags']}/{res['n_correct']} = {res['false_flags'] / res['n_correct']:.1%}")
    print(f"flagged among other incorrect classes: {res['other_incorrect_flagged']}/{res['n_other_incorrect']}")
    print(f"\n=== Class 7 rows only ===\nrecall: {res['c7_flagged']}/{len(res['c7'])} = {res['c7_flagged'] / len(res['c7']):.1%}")
    missed = [(r, f) for r, f in res["c7"] if not f["flagged"]]
    print(f"\n=== Class 7 rows NOT flagged ({len(missed)}) ===")
    for r, f in missed:
        note = f"; hand note: {r['human_note']}" if r.get("human_note") else ""
        extra = f" (satisfied by {f['satisfied_by']})" if "satisfied_by" in f else ""
        print(f"- {r['domain']:8s} candidate={_candidate_name(r)}  reason={f['reason']}{extra}{note}")

    print("\n=== Variants (rule strictness x which clause sources) ===")
    print(f"{'variant':44s} {'C7 recall':>10s} {'false-flag':>11s}")
    for label, kw in [("any lookup, all sources (default)", {}),
                      ("any lookup, policy-only sources", {"sources": ("policy",)}),
                      ("every clause, all sources", {"mode": "all_clauses"}),
                      ("every clause, policy-only sources", {"mode": "all_clauses", "sources": ("policy",)})]:
        v = evaluate(rows, **kw)
        print(f"{label:44s} {v['c7_flagged']:>4d}/{len(v['c7'])}      {v['false_flags']:>4d}/{v['n_correct']}")


if __name__ == "__main__":
    main()
