"""The human decisions as the builders should read them (D-024).

data/derived/review_queue_decisions.jsonl is append-only. Queues 1 to 4 write one record per item, keyed by the item id; the latest record wins.
Later layers write records keyed `<layer>::<original item id>` with `supersedes` set to the original id; the original record stays in the file, unchanged.
`effective()` returns one record per ORIGINAL item id in the original shape (`decision`, `human_label`, `human_class`, `queue` = the original queue),
with the layers applied in this order, each over the previous result:
    q5   queue 5, the Class 7 / read-only re-review (a human key press in `review --queue 5`, or the rule records of data/apply_queue5_rules.py in
         data/derived/review_queue_rule_decisions.jsonl, `source: "rule"`)
    lp   the line-probe rule (data/apply_line_probe_rule.py, data/derived/review_queue_lineprobe_decisions.jsonl, `source: "rule"`)
    q6   queue 6, the line-probe items the rule could not decide (a human key press in `review --queue 6`)
A human record (not a skip) in the main file wins over the rule record of the same id; deleting a rule file undoes that rule. Extra fields on the
effective record:
    class7_reason, class7_reason_source   when the effective class is 7 ("human" or "heuristic")
    tag                   "parallel_lookup" for a read-only call judged correct that was issued beside other read-only calls
    superseded_by         the last layer applied; superseded_chain = every layer applied, in order
    applied_by            "rule" or "human" for the last layer
    earlier               the original record's decision / class
A skip in any layer changes nothing. The Class 7 rows and the line-ID lookups of data/derived/test_real.jsonl are items too (orig_kind "test_real", original
id `test_real::<trajectory id>`, queue 0): their effective record is built from the layer record, which carries `orig_base` (the phase-4 label).
"""

import json
from pathlib import Path

Q5 = "q5::"
LAYERS = ("q5", "lp", "q6")
RULES_NAME = "review_queue_rule_decisions.jsonl"            # queue 5 by rule
LINEPROBE_NAME = "review_queue_lineprobe_decisions.jsonl"   # the line-probe rule


def load_raw(path: Path) -> dict:
    """item id -> latest record (layer records under their own `<layer>::` ids)."""
    out = {}
    for line in open(path):
        if line.strip():
            d = json.loads(line)
            out[d["item_id"]] = d
    return out


def effective(raw: dict) -> dict:
    out = {k: dict(v) for k, v in raw.items() if k.split("::", 1)[0] not in LAYERS}
    for layer in LAYERS:
        for k, v in raw.items():
            if k.startswith(layer + "::") and v["decision"] != "skip":
                _apply(out, v, layer)
    return out


def _apply(out: dict, v: dict, layer: str) -> None:
    o = v["supersedes"]
    base = out.get(o)
    if base is None and v["orig_kind"] == "test_real":   # a test_real label has no decision record: build one from the label the record carries
        base = out[o] = dict(v.get("orig_base") or {"item_id": o, "queue": 0, "origin": v["origin"], "kind": "test_real", "decision": "hand_label",
                                                     "human_label": "incorrect", "human_class": 7, "human_note": "", "proposed_class": 7})
    if base is None:                       # a record without its original: nothing to supersede
        return
    base.setdefault("earlier", {"decision": base["decision"], "human_class": base.get("human_class")})
    cls = v.get("human_class")
    if v["orig_kind"] == "error":          # the original was a candidate: the call is a test error with class `cls`, or it is not an error
        if v["decision"] == "correct":
            base.update(decision="reject", human_label="reject", human_class=None)
        else:
            base.update(decision="accept" if cls == base.get("proposed_class") else "change_class", human_label="accept", human_class=cls)
    else:                                  # the original was a correct example or a test_real row
        if v["decision"] == "correct":
            base.update(decision="correct", human_label="correct", human_class=None)
        else:
            base.update(decision="wrong", human_label="incorrect", human_class=cls)
    base.update(class7_reason=v.get("class7_reason") if cls == 7 else None, class7_reason_source=v.get("class7_reason_source", "human") if cls == 7 else None,
                tag=(v.get("tag", base.get("tag")) if v["decision"] == "correct" else None), superseded_by=layer,
                superseded_chain=base.get("superseded_chain", []) + [layer], q5_time=v.get("time"), applied_by=v.get("source", "human"))
    if v.get("human_note"):
        base["human_note"] = ((base.get("human_note") or "") + f" [{layer}: " + v["human_note"] + "]").strip()


def load_effective(path: Path, rules: bool = True) -> dict:
    """Human decisions with the later layers applied; rule-applied verdicts fill the items no human decided (only for the default file name)."""
    path = Path(path)
    raw = load_raw(path)
    if rules and path.name == "review_queue_decisions.jsonl":
        for name in (RULES_NAME, LINEPROBE_NAME):
            rp = path.parent / name
            if rp.exists():
                for k, v in load_raw(rp).items():
                    h = raw.get(k)
                    if h is None or h["decision"] == "skip":
                        raw[k] = v
    return effective(raw)
