"""Class 7 state-flip negatives (D-021): the lookup shows the action is not allowed, the agent does it anyway.

Real Class 7 errors are policy violations: the agent looked the thing up, the result says the action is not allowed,
and the call is made regardless. The old synthetic Class 7 data deleted lookups, which made the negatives shorter than
their positives (a length shortcut, D-009). Here the call and the conversation stay as they are and ONE value in the most
recent relevant lookup result is changed so that the unchanged call now breaks a rule of the domain policy.

    python data/make_class7_stateflip.py             # writes data/derived/class7_stateflip.jsonl and prints the report
    python data/make_class7_stateflip.py --check     # re-verifies the written file against train_pool.jsonl

How a row is made
  source     a correct tau2-bench row of data/derived/train_pool.jsonl whose task is in the TRAIN side of
             data/derived/real_split.json (D-018) and whose candidate is a write call a rule covers
  edit       the lookup result(s) named by the rule are parsed as JSON, one value is changed, and the result is written back
             with json.dumps; a result is only edited if json.dumps(json.loads(text)) == text, so nothing else changes
  paired     the unedited row stays in train_pool.jsonl as the paired correct example (`paired_correct_id`)
  consistency  if the same lookup was made more than once, every result that carries the old value is changed the same way, so
             the conversation does not contradict itself. Rows are skipped when a user or agent message states the old value
             (it would contradict the edited lookup), and when the original call is not clearly compliant in the lookup.

Rules come ONLY from the policy documents (data/raw/tau2-bench/data/tau2/domains/{retail,airline}/policy.md and
telecom/main_policy.md); each row records the rule's document, section and verbatim text. No rule was taken from a
test_real row or a review candidate. The telecom SIM-lock example is not in main_policy.md, so it is not used.
"""

import argparse
import ast
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).parent
DER = HERE / "derived"
DOMAINS = HERE / "raw" / "tau2-bench" / "data" / "tau2" / "domains"
OUT = DER / "class7_stateflip.jsonl"
OUT_CONDITIONAL = DER / "class7_stateflip_conditional.jsonl"  # rules phrased as an instruction ("if X, do Y"), not a prohibition; not merged (D-023)
SEED = 20261112

POLICY_FILES = {"retail": DOMAINS / "retail" / "policy.md", "airline": DOMAINS / "airline" / "policy.md",
                "telecom": DOMAINS / "telecom" / "main_policy.md"}
CURRENT_TIME = {"airline": "2024-05-15T15:00:00", "telecom": "2025-02-25"}  # from the first line of each policy


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


POLICY_TEXT = {d: norm(p.read_text()) for d, p in POLICY_FILES.items()}


@dataclass
class Rule:
    rule_id: str
    domain: str
    tools: tuple
    section: str
    text: str            # verbatim from the policy document (checked below)
    strength: str = "prohibition"   # "prohibition": the policy forbids the action. "conditional": the policy tells the agent to act only if a condition holds

    def __post_init__(self):
        assert norm(self.text) in POLICY_TEXT[self.domain], f"{self.rule_id}: rule text is not in the policy document"

    @property
    def doc(self):
        return POLICY_FILES[self.domain].name


RULES = {r.rule_id: r for r in [
    Rule("retail_cancel_status", "retail", ("cancel_pending_order",), "Cancel pending order",
         "An order can only be cancelled if its status is 'pending', and you should check its status before taking the action."),
    Rule("retail_modify_status", "retail", ("modify_pending_order_address", "modify_pending_order_items", "modify_pending_order_payment"),
         "Modify pending order", "An order can only be modified if its status is 'pending', and you should check its status before taking the action."),
    Rule("retail_return_status", "retail", ("return_delivered_order_items",), "Return delivered order",
         "An order can only be returned if its status is 'delivered', and you should check its status before taking the action."),
    Rule("retail_exchange_status", "retail", ("exchange_delivered_order_items",), "Exchange delivered order",
         "An order can only be exchanged if its status is 'delivered', and you should check its status before taking the action."),
    Rule("airline_modify_basic_economy", "airline", ("update_reservation_flights",), "Modify flight: Change flights",
         "Basic economy flights cannot be modified."),
    Rule("airline_payment_in_profile_book", "airline", ("book_reservation",), "Book flight: Payment",
         "All payment methods must already be in user profile for safety reasons."),
    Rule("airline_payment_in_profile_modify", "airline", ("update_reservation_flights",), "Modify flight: Payment",
         "If the flights are changed, the user needs to provide a single gift card or credit card for payment or refund method. "
         "The payment method must already be in user profile for safety reasons."),
    Rule("airline_passenger_count", "airline", ("update_reservation_passengers",), "Modify flight: Change passengers",
         "The user can modify passengers but cannot modify the number of passengers."),
    Rule("airline_remove_bags", "airline", ("update_reservation_baggages",), "Modify flight: Change baggage and insurance",
         "The user can add but not remove checked bags."),
    Rule("airline_cancel_conditions", "airline", ("cancel_reservation",), "Cancel flight",
         "Otherwise, flight can be cancelled if any of the following is true: - The booking was made within the last 24 hrs "
         "- The flight is cancelled by airline - It is a business flight - The user has travel insurance and the reason for cancellation is covered by insurance."),
    Rule("telecom_roaming_already_enabled", "telecom", ("enable_roaming",), "Data Roaming",
         "If a user is traveling outside their home network, you should check if the line is roaming enabled. If it is not, you should enable it at no cost for the user.",
         "conditional"),
    Rule("telecom_resume_not_suspended", "telecom", ("resume_line",), "Line Suspension",
         "You are allowed to lift the suspension after the user has paid all their overdue bills.", "conditional"),
    Rule("telecom_resume_contract_ended", "telecom", ("resume_line",), "Line Suspension",
         "You are not allowed to lift the suspension if the line's contract end date is in the past, even if the user has paid all their overdue bills."),
    Rule("telecom_resume_overdue_bill", "telecom", ("resume_line",), "Line Suspension",
         "You are allowed to lift the suspension after the user has paid all their overdue bills."),
]}


# ───────────────────────────── conversation helpers ─────────────────────────────

def parse_args(a):
    if isinstance(a, dict):
        return a
    try:
        v = ast.literal_eval(a)
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


def lookups(turns: list[dict]) -> list[dict]:
    """Every tool call paired with its result, by order (results answer the pending calls first-in, first-out)."""
    pending, out = [], []
    for i, t in enumerate(turns):
        if t["role"] == "assistant_tool_call":
            pending.append((i, t["name"], parse_args(t.get("arguments"))))
        elif t["role"] == "tool" and pending:
            ci, name, args = pending.pop(0)
            out.append({"call": ci, "result": i, "name": name, "args": args, "content": str(t.get("content", ""))})
    return out


def as_json(content: str):
    """The parsed result, only if writing it back with json.dumps reproduces the text exactly."""
    try:
        obj = json.loads(content)
    except Exception:
        return None
    return obj if json.dumps(obj) == content else None


def mentioned(turns: list[dict], pattern: str) -> bool:
    rx = re.compile(pattern, re.I)
    return any(t["role"] in ("user", "assistant") and rx.search(str(t.get("content", ""))) for t in turns)


def states_status(turns: list[dict], order_id: str, status: str) -> bool:
    """A user or agent message says THIS order has `status`: it names the order and the status in the same message, or
    says 'is / was / currently / status ... <status>'. Such a message would contradict an edited lookup."""
    w = re.escape(status)
    near = re.compile(rf"{re.escape(order_id)}[^.\n]{{0,80}}\b{w}\b|\b{w}\b[^.\n]{{0,80}}{re.escape(order_id)}", re.I)
    verb = re.compile(rf"\b(is|are|was|were|currently|status(?: of| is)?|marked as|shows?)\W+(?:\w+\W+){{0,2}}{w}\b", re.I)
    return any(t["role"] in ("user", "assistant") and (near.search(str(t.get("content", ""))) or verb.search(str(t.get("content", ""))))
               for t in turns)


class Skip(Exception):
    pass


def pick(options, row, salt):
    h = int(hashlib.sha256(f"{SEED}|{row['trajectory_id']}|{salt}".encode()).hexdigest(), 16)
    return options[h % len(options)]


# ───────────────────────────── the rules ─────────────────────────────
# Each function returns {"edits": {turn_index: new_content}, "field": ..., "old": ..., "new": ..., "lookup": ..., "lookup_args": ...}
# or raises Skip(reason).

def _results(row, name, key, value):
    return [p for p in lookups(row["turns"]) if p["name"] == name and p["args"].get(key) == value and as_json(p["content"]) is not None]


def _edit_all(ps, mutate, same_old):
    """Apply `mutate(obj)` to every lookup result whose old value matches `same_old(obj)`; return {turn: new_content}."""
    edits = {}
    for p in ps:
        obj = as_json(p["content"])
        if same_old(obj):
            mutate(obj)
            edits[p["result"]] = json.dumps(obj)
    return edits


def flip_order_status(row, cand, old_status, new_options):
    ps = _results(row, "get_order_details", "order_id", cand["arguments"].get("order_id"))
    if not ps:
        raise Skip("no lookup of the order")
    last = as_json(ps[-1]["content"])
    if last.get("status") != old_status:
        raise Skip(f"the lookup does not show the expected status {old_status!r}")
    new = pick(new_options, row, "status")
    if states_status(row["turns"], cand["arguments"].get("order_id", ""), old_status):
        raise Skip("a message states the old status of this order")
    def mutate(o):
        # In every unedited lookup "pending" means no fulfillments and any other status means there are some (checked on all
        # 2,326 order lookups of train_pool), so the status is changed together with the fulfillment info. A bare status edit
        # leaves an inconsistent order that a model could spot instead of reading the policy.
        o["status"] = new
        if new == "pending":
            o["fulfillments"] = []
        elif not o.get("fulfillments"):
            tid = str(int(hashlib.sha256(f"{o['order_id']}|tracking".encode()).hexdigest(), 16))[:12]
            o["fulfillments"] = [{"tracking_id": [tid], "item_ids": [it["item_id"] for it in o["items"]]}]
    edits = _edit_all(ps, mutate, lambda o: o.get("status") == old_status)
    return {"edits": edits, "field": "status+fulfillments", "old": old_status, "new": new, "lookup": "get_order_details", "lookup_args": ps[-1]["args"]}


def flip_airline_basic_economy(row, cand):
    rid = cand["arguments"].get("reservation_id")
    ps = _results(row, "get_reservation_details", "reservation_id", rid)
    if not ps:
        raise Skip("no lookup of the reservation")
    last = as_json(ps[-1]["content"])
    have = {(f["flight_number"], f["date"]) for f in last.get("flights", [])}
    want = {(f.get("flight_number"), f.get("date")) for f in cand["arguments"].get("flights", [])}
    if have == want:
        raise Skip("cabin-only change: the rule is about changing flights")
    if last.get("cabin") == "basic_economy":
        raise Skip("already basic economy")
    old = last["cabin"]
    if mentioned(row["turns"], r"(?<!basic )(?<!basic_)\b" + re.escape(old.replace("_", " ")) + r"\b"):
        raise Skip("a message states the old cabin")
    edits = _edit_all(ps, lambda o: o.__setitem__("cabin", "basic_economy"), lambda o: o.get("cabin") == old)
    return {"edits": edits, "field": "cabin", "old": old, "new": "basic_economy", "lookup": "get_reservation_details", "lookup_args": ps[-1]["args"]}


def _payment_ids(cand):
    a = cand["arguments"]
    if cand["name"] == "book_reservation":
        return [p.get("payment_id") for p in a.get("payment_methods", [])]
    return [a.get("payment_id")] if a.get("payment_id") else []


def flip_payment_not_in_profile(row, cand, needs_changed_flights):
    if needs_changed_flights:
        rid = cand["arguments"].get("reservation_id")
        res = _results(row, "get_reservation_details", "reservation_id", rid)
        if not res:
            raise Skip("no lookup of the reservation")
        last = as_json(res[-1]["content"])
        have = {(f["flight_number"], f["date"]) for f in last.get("flights", [])}
        if have == {(f.get("flight_number"), f.get("date")) for f in cand["arguments"].get("flights", [])}:
            raise Skip("cabin-only change: the payment rule is about changing flights")
    ids = _payment_ids(cand)
    users = [p for p in lookups(row["turns"]) if p["name"] == "get_user_details" and as_json(p["content"]) is not None]
    if not ids or not users:
        raise Skip("no payment method or no user lookup")
    last = as_json(users[-1]["content"])
    uid = users[-1]["args"].get("user_id")
    target = next((i for i in ids if i in last.get("payment_methods", {})), None)
    if target is None:
        raise Skip("the call's payment method is not in the profile to begin with")
    pm = last["payment_methods"][target]
    if mentioned(row["turns"], re.escape(target) + (r"|" + re.escape(str(pm["last_four"])) if pm.get("last_four") else "")):
        raise Skip("a message mentions the removed payment method")
    ps = [p for p in users if p["args"].get("user_id") == uid]
    edits = _edit_all(ps, lambda o: o["payment_methods"].pop(target), lambda o: target in o.get("payment_methods", {}))
    return {"edits": edits, "field": f"payment_methods.{target}", "old": pm, "new": "(removed from the profile)", "lookup": "get_user_details", "lookup_args": {"user_id": uid}}


def flip_passenger_count(row, cand):
    rid = cand["arguments"].get("reservation_id")
    ps = _results(row, "get_reservation_details", "reservation_id", rid)
    if not ps:
        raise Skip("no lookup of the reservation")
    last = as_json(ps[-1]["content"])
    n = len(last.get("passengers", []))
    if n < 2 or n != len(cand["arguments"].get("passengers", [])):
        raise Skip("needs a reservation with 2+ passengers and a call that keeps the count")
    gone = last["passengers"][-1]
    if mentioned(row["turns"], re.escape(gone["first_name"]) + r"\s+" + re.escape(gone["last_name"])):
        raise Skip("a message names the removed passenger")
    edits = _edit_all(ps, lambda o: o["passengers"].pop(), lambda o: len(o.get("passengers", [])) == n)
    return {"edits": edits, "field": "passengers", "old": f"{n} passengers", "new": f"{n - 1} passengers (the call keeps {n})", "lookup": "get_reservation_details", "lookup_args": ps[-1]["args"]}


def flip_remove_bags(row, cand):
    rid = cand["arguments"].get("reservation_id")
    ps = _results(row, "get_reservation_details", "reservation_id", rid)
    if not ps:
        raise Skip("no lookup of the reservation")
    last = as_json(ps[-1]["content"])
    new_total = cand["arguments"].get("total_baggages")
    if not isinstance(new_total, int) or last.get("total_baggages") is None or last["total_baggages"] >= new_total:
        raise Skip("the call does not add bags to begin with")
    old = last["total_baggages"]
    edits = _edit_all(ps, lambda o: o.__setitem__("total_baggages", new_total + 1), lambda o: o.get("total_baggages") == old)
    return {"edits": edits, "field": "total_baggages", "old": old, "new": new_total + 1, "lookup": "get_reservation_details", "lookup_args": ps[-1]["args"]}


def flip_cancel_conditions(row, cand):
    rid = cand["arguments"].get("reservation_id")
    ps = _results(row, "get_reservation_details", "reservation_id", rid)
    if not ps:
        raise Skip("no lookup of the reservation")
    last = as_json(ps[-1]["content"])
    if any(f["date"] < CURRENT_TIME["airline"][:10] for f in last.get("flights", [])):
        raise Skip("a segment is already flown")
    if mentioned(row["turns"], r"airline (has )?cancel|cancel+ed (my |the |our )?flight|flight (was|has been|is) cancel|health|medical|illness|sick|weather|storm"):
        raise Skip("the stated reason could itself allow the cancellation")
    if any(p["name"] == "get_flight_status" and re.search(r"cancel", p["content"], re.I) for p in lookups(row["turns"])):
        raise Skip("a flight status says cancelled")
    changes = {}
    if last.get("cabin") == "business":
        changes["cabin"] = ("business", "economy")
    if last.get("insurance") == "yes":
        changes["insurance"] = ("yes", "no")
    created = last.get("created_at", "")
    if created and created > "2024-05-14T15:00:00":
        changes["created_at"] = (created, "2024-05-02" + created[10:])
    if not changes:
        raise Skip("no condition holds in the lookup, so the original cancellation is not explained by it")
    def mutate(o):
        for k, (_, new) in changes.items():
            o[k] = new
    edits = _edit_all(ps, mutate, lambda o: o.get("created_at") == created and o.get("cabin") == last["cabin"] and o.get("insurance") == last.get("insurance"))
    return {"edits": edits, "field": "+".join(changes), "old": {k: v[0] for k, v in changes.items()}, "new": {k: v[1] for k, v in changes.items()},
            "lookup": "get_reservation_details", "lookup_args": ps[-1]["args"]}


def flip_contract_ended(row, cand):
    lid = cand["arguments"].get("line_id")
    ps = _results(row, "get_details_by_id", "id", lid)
    if not ps:
        raise Skip("no lookup of the line")
    last = as_json(ps[-1]["content"])
    old = last.get("contract_end_date")
    if not old or old < CURRENT_TIME["telecom"]:
        raise Skip("the contract end date is not in the future in the lookup")
    if mentioned(row["turns"], re.escape(old)):
        raise Skip("a message states the contract end date")
    new = "2025-01-15"
    edits = _edit_all(ps, lambda o: o.__setitem__("contract_end_date", new), lambda o: o.get("contract_end_date") == old)
    return {"edits": edits, "field": "contract_end_date", "old": old, "new": new, "lookup": "get_details_by_id", "lookup_args": ps[-1]["args"]}


def flip_bill_overdue(row, cand):
    """Most recent lookup that shows a bill as Paid (a bill the user had to pay): show it as Overdue instead."""
    cands = []
    for p in lookups(row["turns"]):
        obj = as_json(p["content"])
        if p["name"] == "get_details_by_id" and isinstance(obj, dict) and obj.get("bill_id") and obj.get("status") == "Paid":
            cands.append((p, obj["bill_id"]))
        elif p["name"] == "get_bills_for_customer" and isinstance(obj, list):
            cands += [(p, b["bill_id"]) for b in obj if b.get("status") == "Paid" and b.get("total_due", 0) > 0]
    if not cands:
        raise Skip("no lookup shows a paid bill")
    p, bill = cands[-1]
    if mentioned(row["turns"], r"\b(paid|payment (was|has been) (made|processed))\b"):
        raise Skip("a message says the bill was paid")
    obj = as_json(p["content"])
    if isinstance(obj, dict):
        obj["status"] = "Overdue"
    else:
        next(b for b in obj if b["bill_id"] == bill)["status"] = "Overdue"
    return {"edits": {p["result"]: json.dumps(obj)}, "field": f"bill {bill} status", "old": "Paid", "new": "Overdue", "lookup": p["name"], "lookup_args": p["args"]}


def flip_roaming_enabled(row, cand):
    lid = cand["arguments"].get("line_id")
    ps = _results(row, "get_details_by_id", "id", lid)
    if not ps:
        raise Skip("no lookup of the line")
    last = as_json(ps[-1]["content"])
    if last.get("roaming_enabled") is not False:
        raise Skip("the lookup does not show roaming switched off")
    if mentioned(row["turns"], r"roaming[^.]{0,40}(disabled|off|not (enabled|on|active))|(disabled|off)[^.]{0,20}roaming"):
        raise Skip("a message states that roaming is off")
    edits = _edit_all(ps, lambda o: o.__setitem__("roaming_enabled", True), lambda o: o.get("roaming_enabled") is False)
    return {"edits": edits, "field": "roaming_enabled", "old": False, "new": True, "lookup": "get_details_by_id", "lookup_args": ps[-1]["args"]}


def flip_line_not_suspended(row, cand):
    lid = cand["arguments"].get("line_id")
    ps = _results(row, "get_details_by_id", "id", lid)
    if not ps:
        raise Skip("no lookup of the line")
    last = as_json(ps[-1]["content"])
    if last.get("status") != "Suspended":
        raise Skip("the lookup does not show a suspended line")
    if mentioned(row["turns"], r"suspend"):
        raise Skip("a message mentions the suspension")
    edits = _edit_all(ps, lambda o: (o.__setitem__("status", "Active"), o.__setitem__("suspension_start_date", None)),
                      lambda o: o.get("status") == "Suspended")
    return {"edits": edits, "field": "status+suspension_start_date", "old": "Suspended", "new": "Active", "lookup": "get_details_by_id", "lookup_args": ps[-1]["args"]}


APPLY = {
    "telecom_roaming_already_enabled": flip_roaming_enabled,
    "telecom_resume_not_suspended": flip_line_not_suspended,
    "retail_cancel_status": lambda r, c: flip_order_status(r, c, "pending", ["processed", "delivered"]),
    "retail_modify_status": lambda r, c: flip_order_status(r, c, "pending", ["processed", "delivered"]),
    "retail_return_status": lambda r, c: flip_order_status(r, c, "delivered", ["pending", "processed"]),
    "retail_exchange_status": lambda r, c: flip_order_status(r, c, "delivered", ["pending", "processed"]),
    "airline_modify_basic_economy": flip_airline_basic_economy,
    "airline_payment_in_profile_book": lambda r, c: flip_payment_not_in_profile(r, c, False),
    "airline_payment_in_profile_modify": lambda r, c: flip_payment_not_in_profile(r, c, True),
    "airline_passenger_count": flip_passenger_count,
    "airline_remove_bags": flip_remove_bags,
    "airline_cancel_conditions": flip_cancel_conditions,
    "telecom_resume_contract_ended": flip_contract_ended,
    "telecom_resume_overdue_bill": flip_bill_overdue,
}


# ───────────────────────────── build ─────────────────────────────

def load(p):
    return [json.loads(line) for line in open(p) if line.strip()]


def train_tasks():
    s = json.load(open(DER / "real_split.json"))["tasks"]
    return {d: {str(t) for t in s[d]["train"]} for d in s}


def build():
    train = train_tasks()
    pool = [r for r in load(DER / "train_pool.jsonl") if str(r["task_id"]) in train.get(r["domain"], set())]
    print(f"train_pool rows on train-split tasks: {len(pool)} of 1,185")
    by_tool = defaultdict(list)
    for rule in RULES.values():
        for tool in rule.tools:
            by_tool[(rule.domain, tool)].append(rule)
    out, stats, reasons = [], Counter(), Counter()
    for row in pool:
        key = (row["domain"], row["candidate_tool_call"]["name"])
        if key not in by_tool:
            continue
        stats[("eligible", *key)] += 1
        options = []
        for rule in by_tool[key]:
            try:
                options.append((rule, APPLY[rule.rule_id](row, row["candidate_tool_call"])))
            except Skip as e:
                reasons[(rule.rule_id, str(e))] += 1
        if not options:
            continue
        options = [(r_, f_) for r_, f_ in options if f_["edits"]]
        if not options:
            continue
        strong = [(r_, f_) for r_, f_ in options if r_.strength == "prohibition"]
        chosen = []
        if strong:
            first = pick(strong, row, "rule")           # the same choice as in D-021 ...
            chosen = [first] + [x for x in strong if x[0] is not first[0]]   # ... plus one more negative for every other applicable rule (D-023)
        chosen += [x for x in options if x[0].strength != "prohibition"]
        for n_flip, (rule, flip) in enumerate(chosen):
            neg = json.loads(json.dumps(row))
            for i, content in flip["edits"].items():
                neg["turns"][i]["content"] = content
            neg.update({
                "trajectory_id": f"{row['trajectory_id']}::flip7" if n_flip == 0 else f"{row['trajectory_id']}::flip7::{rule.rule_id}",
                "label": "incorrect", "error_class": 7, "error_class_name": "missing_precondition",
                "label_tier": "synthetic_class7_stateflip", "corruption_method": "state_flip",
                "source": "synthetic_stateflip", "origin": "synthetic", "split": "train",
                "paired_correct_id": row["trajectory_id"],
                "policy_rule": {"id": rule.rule_id, "doc": rule.doc, "section": rule.section, "text": rule.text, "strength": rule.strength},
                "flip": {"lookup": flip["lookup"], "lookup_args": flip["lookup_args"], "field": flip["field"], "old": flip["old"], "new": flip["new"],
                         "edited_turns": sorted(flip["edits"]), "rules_applicable": [r.rule_id for r, _ in options]},
            })
            out.append(neg)
            stats[("flipped" if rule.strength == "prohibition" else "conditional", *key)] += 1
    return pool, out, stats, reasons


def report(pool, out, stats, reasons):
    print("\nstate-flip negatives by domain and write tool (eligible = train-split correct rows with that candidate tool)")
    print(f"   {'domain':8s} {'tool':34s} {'eligible':>8s} {'flipped':>8s}")
    for (kind, d, t), n in sorted(stats.items()):
        if kind == "eligible":
            print(f"   {d:8s} {t:34s} {n:8d} {stats[('flipped', d, t)]:8d}")
    print(f"   total flipped {len(out)} | by rule {dict(Counter(r['policy_rule']['id'] for r in out))}")
    by_dom = Counter(r["domain"] for r in out)
    print(f"   by domain {dict(by_dom)} ({by_dom['retail'] / len(out):.0%} retail)")
    print("\nrows not flipped, by reason")
    for (rid, why), n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"   {n:4d}  {rid}: {why}")
    rng = random.Random(1)
    shown = set()
    print("\nexamples (before -> after of the edited lookup result)")
    for r in rng.sample(out, len(out)):
        if r["policy_rule"]["id"] in shown or len(shown) >= 5:
            continue
        shown.add(r["policy_rule"]["id"])
        orig = next(p for p in pool if p["trajectory_id"] == r["paired_correct_id"])
        i = r["flip"]["edited_turns"][-1]
        print(f"\n[{r['domain']}] {r['candidate_tool_call']['name']}({json.dumps(r['candidate_tool_call']['arguments'])[:150]})")
        print(f"  rule ({r['policy_rule']['doc']}, {r['policy_rule']['section']}): {r['policy_rule']['text'][:200]}")
        print(f"  lookup {r['flip']['lookup']}{json.dumps(r['flip']['lookup_args'])}: {r['flip']['field']}: {r['flip']['old']!r} -> {r['flip']['new']!r}")
        print(f"  before: {_window(orig['turns'][i]['content'], r['flip']['field'])}")
        print(f"  after : {_window(r['turns'][i]['content'], r['flip']['field'])}")


def _window(content: str, field: str) -> str:
    key = "payment_methods" if field.startswith("payment_methods") else (field.split("+")[0].split(" ")[-1] if field else "")
    key = "status" if key == "status" else key
    m = re.search(re.escape(f'"{key}"'), content)
    s = max((m.start() if m else 0) - 40, 0)
    return "…" + content[s: s + 170].replace("\n", " ") + "…"


def check(pool_by_id, rows):
    """Every negative differs from its paired positive only in the edited lookup results, and there only in the flipped value(s)."""
    bad = 0
    for r in rows:
        o = pool_by_id[r["paired_correct_id"]]
        for k in o:
            if k in ("turns", "trajectory_id", "label", "label_tier", "split", "error_class", "error_class_name", "source", "origin"):
                continue
            if o[k] != r[k]:
                print("field differs:", k, r["trajectory_id"])
                bad += 1
        assert len(o["turns"]) == len(r["turns"])
        diff_turns = [i for i, (a, b) in enumerate(zip(o["turns"], r["turns"])) if a != b]
        if diff_turns != r["flip"]["edited_turns"]:
            print("turn diff mismatch", r["trajectory_id"], diff_turns, r["flip"]["edited_turns"])
            bad += 1
        for i in diff_turns:
            if {k: v for k, v in o["turns"][i].items() if k != "content"} != {k: v for k, v in r["turns"][i].items() if k != "content"}:
                bad += 1
            if o["turns"][i]["content"] == r["turns"][i]["content"]:
                bad += 1
    return bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify the written file against train_pool.jsonl and exit")
    args = ap.parse_args()
    if args.check:
        rows = load(OUT) + load(OUT_CONDITIONAL)
        bad = check({r["trajectory_id"]: r for r in load(DER / "train_pool.jsonl")}, rows)
        print(f"{len(rows)} rows checked (both files), {bad} problems")
        sys.exit(1 if bad else 0)
    pool, out, stats, reasons = build()
    strong = [r for r in out if r["policy_rule"]["strength"] == "prohibition"]
    cond = [r for r in out if r["policy_rule"]["strength"] != "prohibition"]
    report(pool, strong, stats, reasons)
    print(f"\nconditional-rule negatives (an instruction 'if X, do Y', not a prohibition; kept apart and not merged): {len(cond)} by rule {dict(Counter(r['policy_rule']['id'] for r in cond))}")
    for path, rs in ((OUT, strong), (OUT_CONDITIONAL, cond)):
        with open(path, "w") as fh:
            for r in rs:
                fh.write(json.dumps(r) + "\n")
        print(f"wrote {len(rs)} rows to {path}")
