"""Categorize the Class 1 (wrong argument value) negatives by how the wrong value was made.

Each negative is compared with its source positive (trajectory_id minus the "::corrupt1" suffix, looked up
in train_pool.jsonl / bfcl_pool.jsonl). For every argument whose value changed, the new value is checked
against the conversation (the negative's turns) and against the old value:

  context_swap       new value appears elsewhere in the conversation and has the same shape and type as
                     the old one (another order ID, another date, another name)
  type_mismatch      new value appears in the conversation but has a different shape or type from the old
                     one (a name or number in an ID slot)
  plausible_mutation new value is NOT in the conversation but is a small change to the old one
                     (similar string, number within a factor of 2)
  random             new value is not in the conversation and not close to the old one

"Same shape" collapses digit runs to 9 and letter runs to a ("#W1234567" -> "#a9"), so it is a syntactic
test, not a semantic one: type_mismatch only catches values of a visibly different kind. A row with several
changed arguments takes the category of the first changed one. Run: python data/analyze_class1.py [path]
"""

import difflib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

D = Path(__file__).parent / "derived"
CATEGORIES = ("context_swap", "plausible_mutation", "type_mismatch", "random")


def load(path) -> list[dict]:
    return [json.loads(line) for line in open(path) if line.strip()]


def leaves(v) -> list:
    if isinstance(v, dict):
        return [x for e in v.values() for x in leaves(e)]
    if isinstance(v, (list, tuple)):
        return [x for e in v for x in leaves(e)]
    return [] if v is None or isinstance(v, bool) else [v]


def shape(v) -> str:
    return re.sub(r"[A-Za-z]+", "a", re.sub(r"\d+", "9", str(v)))


def in_text(value, text: str) -> bool:
    return re.search(r"(?<![A-Za-z0-9])" + re.escape(str(value)) + r"(?![A-Za-z0-9])", text) is not None


def similar(old, new) -> bool:
    try:
        a, b = float(old), float(new)
        return a != 0 and 0.5 <= b / a <= 2
    except (TypeError, ValueError):
        return difflib.SequenceMatcher(None, str(old), str(new)).ratio() >= 0.7


def where_from(value, turns: list[dict]) -> str:
    """Which kind of turn mentions the value first: user, tool result, or an earlier call."""
    for t in turns:
        if t["role"] == "user" and in_text(value, str(t.get("content", ""))):
            return "user turn"
    for t in turns:
        if t["role"] == "tool" and in_text(value, str(t.get("content", ""))):
            return "tool result"
    for t in turns:
        if t["role"] == "assistant_tool_call" and in_text(value, json.dumps(t.get("arguments", ""))):
            return "earlier call"
    for t in turns:
        if t["role"] == "assistant" and in_text(value, str(t.get("content", ""))):
            return "agent message"
    return "nowhere"


def classify(neg: dict, orig: dict) -> tuple[str, str, object, object]:
    """(category, changed argument, old value, new value) for the first changed argument."""
    o, n = orig["arguments"], neg["candidate_tool_call"]["arguments"]
    ctx = "\n".join(str(t.get("content", "")) + " " + json.dumps(t.get("arguments", "")) for t in neg["turns"])
    for k in o:  # a changed or removed argument; added ones are dropped by the quality filter
        if k in n and n[k] != o[k]:
            old_l, new_l = leaves(o[k]), [x for x in leaves(n[k]) if x not in leaves(o[k])]
            old = old_l[0] if old_l else o[k]
            new = new_l[0] if new_l else n[k]
            if in_text(new, ctx):
                same = type(old) is type(new) and shape(old) == shape(new)
                return ("context_swap" if same else "type_mismatch"), k, old, new
            return ("plausible_mutation" if similar(old, new) else "random"), k, old, new
    return "random", "?", None, None


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else D / "negatives_pool_final.jsonl"
    pool = {r["trajectory_id"]: r for name in ("train_pool.jsonl", "bfcl_pool.jsonl") for r in load(D / name)}
    rows = [r for r in load(path) if r["error_class"] == 1]

    by_cat, by_src, origin, examples = Counter(), defaultdict(Counter), Counter(), defaultdict(list)
    multi = 0
    for r in rows:
        orig = pool[r["trajectory_id"].rsplit("::corrupt", 1)[0]]["candidate_tool_call"]
        cat, key, old, new = classify(r, orig)
        src = "BFCL" if r.get("source_dataset") == "bfcl" else "tau2"
        by_cat[cat] += 1
        by_src[src][cat] += 1
        multi += sum(r["candidate_tool_call"]["arguments"].get(k) != v for k, v in orig["arguments"].items()) > 1
        if cat in ("context_swap", "type_mismatch"):
            origin[(cat, where_from(new, r["turns"]))] += 1
        if len(examples[cat]) < 3:
            examples[cat].append(f"{src:5s} {r['candidate_tool_call']['name']}.{key}: {old!r} -> {new!r}")

    print(f"Class 1 negatives in {path.name}: {len(rows)}\n")
    print(f"{'category':20s} {'all':>5s} {'tau2':>6s} {'BFCL':>6s}")
    for c in CATEGORIES:
        print(f"{c:20s} {by_cat[c]:>5d} {by_src['tau2'][c]:>6d} {by_src['BFCL'][c]:>6d}")
    print(f"{'total':20s} {len(rows):>5d} {sum(by_src['tau2'].values()):>6d} {sum(by_src['BFCL'].values()):>6d}")
    print(f"\nrows with more than one changed argument: {multi}")
    print("\nwhere the wrong value first appears (context_swap and type_mismatch rows):")
    for (c, w), n in sorted(origin.items()):
        print(f"  {c:14s} {w:14s} {n}")
    print("\nexamples:")
    for c in CATEGORIES:
        for e in examples[c]:
            print(f"  [{c}] {e}")


if __name__ == "__main__":
    main()
