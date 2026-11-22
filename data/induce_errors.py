"""Induce real Class 2 (stale value) and Class 6 (ignored error) situations in tau2-bench runs (D-019).

Natural failed runs almost never contain these (0 and 1 of 273 proposals, D-018): users rarely change their
minds and tools rarely fail. So we create the situation and let a real agent react. We change the
ENVIRONMENT, never the agent's calls:

  fault       one assistant tool call (a lookup or a write, chosen with a fixed seed) returns the domain's own
              kind of error instead of running (the call is not executed, so state is unchanged)
  correction  the user simulator's instructions get one added mid-conversation change of mind (address, payment
              method, item, date, quantity or passenger detail); old value, new value and timing are recorded

Only tasks listed in data/derived/real_split.json are used, so a task's induced rows inherit its split.
Every row carries source "induced_fault" or "induced_correction". Auto-labels come from what we injected and are
only pre-filled for a human to confirm (data/review_real.py --candidates ... --decisions ...).

Run with the tau2 environment (Python 3.11, see D-019):  data/raw/.venv-tau2/bin/python3 data/induce_errors.py <stage>

    plan      choose the pilot tasks and write data/derived/induced_plan.jsonl
    estimate  print the cost estimate (stops above --max-usd)
    run       run the plan (cached in data/raw/induced/induced_runs.jsonl (gitignored); --limit N for a smoke test)
    label     auto-label the cached runs -> induced_candidates_review.jsonl + induced_positive_candidates.jsonl + report
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
DER = HERE / "derived"
DOMAINS_DIR = HERE / "raw" / "tau2-bench" / "data" / "tau2" / "domains"
PLAN_PATH = DER / "induced_plan.jsonl"
RUNS_PATH = HERE / "raw" / "induced" / "induced_runs.jsonl"  # 37 MB of full transcripts: gitignored (data/raw/), not versioned
REVIEW_PATH = DER / "induced_candidates_review.jsonl"
POSITIVES_PATH = DER / "induced_positive_candidates.jsonl"
REPORT_PATH = DER / "induced_report.json"
SEED = 20261112
AGENT_MODEL, AGENT_ARGS = "gpt-5-nano", {"reasoning_effort": "low"}
DEFAULT_USER_MODEL = "gpt-4.1-mini"
MAX_STEPS = {"retail": 50, "airline": 50, "telecom": 60}
NOT_FAULTED = {"calculate", "transfer_to_human_agents", "think"}

# ───────────────────────── the domains' own error styles ─────────────────────────
# Every string below is a message the domain tools already raise (ValueError text), except the generic
# "temporarily unavailable" one, which the task asked for. {x} fields are filled from the call's own arguments.
TEMP = "Service temporarily unavailable. Please try again later."
ERRORS = {
    "retail": {
        "get_order_details": ["Order not found", TEMP], "get_user_details": ["User not found", TEMP],
        "find_user_id_by_email": ["User not found"], "find_user_id_by_name_zip": ["User not found"],
        "get_product_details": ["Product not found", TEMP], "get_item_details": ["Item not found"],
        "list_all_product_types": [TEMP],
        "modify_pending_order_address": ["Non-pending order cannot be modified", TEMP],
        "modify_pending_order_items": ["Non-pending order cannot be modified", "Some item not found", TEMP],
        "modify_pending_order_payment": ["Payment method not found", "Non-pending order cannot be modified", TEMP],
        "modify_user_address": ["User not found", TEMP], "cancel_pending_order": ["Non-pending order cannot be cancelled", TEMP],
        "return_delivered_order_items": ["Non-delivered order cannot be returned", TEMP],
        "exchange_delivered_order_items": ["Non-delivered order cannot be exchanged", "New item {new_item_ids} not found or available", TEMP],
    },
    "airline": {
        "get_reservation_details": ["Reservation {reservation_id} not found", TEMP], "get_user_details": ["User {user_id} not found", TEMP],
        "search_direct_flight": [TEMP], "search_onestop_flight": [TEMP], "list_all_airports": [TEMP],
        "get_flight_status": ["Flight {flight_number} not found on date {date}"],
        "book_reservation": ["Payment method not found", "Gift card balance is not enough", TEMP],
        "cancel_reservation": ["Reservation {reservation_id} not found", TEMP],
        "update_reservation_baggages": ["Reservation {reservation_id} not found", "Gift card balance is not enough", TEMP],
        "update_reservation_flights": ["Not enough seats on flight {flight_number}", "Payment method {payment_id} not found", TEMP],
        "update_reservation_passengers": ["Number of passengers does not match", TEMP], "send_certificate": ["Too many certificates", TEMP],
    },
    "telecom": {
        "get_customer_by_phone": ["Customer with phone number {phone_number} not found", TEMP],
        "get_customer_by_id": ["Customer with ID {customer_id} not found"], "get_customer_by_name": ["Customer not found", TEMP],
        "get_details_by_id": ["Unknown ID format or type: {id}", "Line with ID {id} not found", TEMP],
        "get_bills_for_customer": ["Customer {customer_id} not found", TEMP], "get_data_usage": ["Line {line_id} not found for customer {customer_id}", TEMP],
        "suspend_line": ["Line must be active to suspend", TEMP], "resume_line": ["Line must be suspended to resume", TEMP],
        "send_payment_request": ["Bill {bill_id} not found for customer {customer_id}", TEMP],
        "enable_roaming": ["Line with ID {line_id} not found", TEMP], "disable_roaming": ["Line with ID {line_id} not found", TEMP],
        "refuel_data": ["Line {line_id} not found for customer {customer_id}", TEMP],
    },
}


class _Safe(dict):
    def __missing__(self, k):
        raise KeyError(k)


def fault_message(domain: str, tool: str, args: dict, rng: random.Random) -> str:
    pool = ERRORS.get(domain, {}).get(tool) or [TEMP]
    msg = rng.choice(pool)
    try:
        return msg.format_map(_Safe({k: (v if not isinstance(v, (list, dict)) else json.dumps(v)) for k, v in (args or {}).items()}))
    except KeyError:
        return TEMP


# ───────────────────────── fault injection (environment wrapper) ─────────────────────────

_FAULT: dict = {}  # per-process, set by the worker before each run: {"spec": {...}, "n": 0, "fired": None}

# ---- soft faults (D-019 follow-up): the call is NOT executed and the response looks normal but shows the action did not happen.
# The response uses the tool's own return schema: the record exactly as it still is (a write that changed nothing), or the
# tool's own kind of empty result.
SOFT_RECORD = {  # (domain, write tool) -> (read tool, id argument): the unchanged record is what the write would have returned
    ("retail", "modify_pending_order_address"): ("get_order_details", "order_id"), ("retail", "modify_pending_order_items"): ("get_order_details", "order_id"),
    ("retail", "modify_pending_order_payment"): ("get_order_details", "order_id"), ("retail", "cancel_pending_order"): ("get_order_details", "order_id"),
    ("retail", "return_delivered_order_items"): ("get_order_details", "order_id"), ("retail", "exchange_delivered_order_items"): ("get_order_details", "order_id"),
    ("retail", "modify_user_address"): ("get_user_details", "user_id"),
    ("airline", "update_reservation_baggages"): ("get_reservation_details", "reservation_id"), ("airline", "update_reservation_flights"): ("get_reservation_details", "reservation_id"),
    ("airline", "update_reservation_passengers"): ("get_reservation_details", "reservation_id"), ("airline", "cancel_reservation"): ("get_reservation_details", "reservation_id"),
}
SOFT_EMPTY = {("airline", "search_direct_flight"): [], ("airline", "search_onestop_flight"): [], ("retail", "list_all_product_types"): "{}"}
SOFT_TELECOM = {"suspend_line", "resume_line", "refuel_data"}


def soft_response(env, domain: str, message):
    """(response object, description) for a soft fault on this call, or None when this call cannot be faulted softly."""
    key, args = (domain, message.name), message.arguments or {}
    if key in SOFT_EMPTY:
        return SOFT_EMPTY[key], "empty result"
    if key in SOFT_RECORD:
        read, idarg = SOFT_RECORD[key]
        if idarg not in args:
            return None
        try:
            return env.make_tool_call(read, requestor="assistant", **{idarg: args[idarg]}), "record still in its old state"
        except Exception:  # noqa: BLE001
            return None
    if domain == "telecom" and message.name in SOFT_TELECOM and "customer_id" in args and "line_id" in args:
        try:
            line = env.tools._get_target_line(args["customer_id"], args["line_id"])
        except Exception:  # noqa: BLE001
            return None
        if message.name == "refuel_data":
            return {"message": f"No data was added for line {args['line_id']}.", "new_data_refueling_gb": line.data_refueling_gb, "charge": 0.0}, "no change reported"
        return {"message": "No change was made to the line.", "line": line}, "record still in its old state"
    return None


# ---- propagation (value changed): one lookup reports that the id the agent is working with has been replaced
NEW_ID = {"order_split": lambda db, rng: "#W" + "".join(rng.choice("0123456789") for _ in range(7)),
          "reservation_reissued": lambda db, rng: "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") for _ in range(6))}


def _contains(obj, value) -> bool:
    if isinstance(obj, dict):
        return any(_contains(v, value) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_contains(v, value) for v in obj)
    return isinstance(obj, str) and obj == value


def _swap(obj, old, new):
    if isinstance(obj, dict):
        return {k: _swap(v, old, new) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_swap(v, old, new) for v in obj]
    return new if isinstance(obj, str) and obj == old else obj


def _prop_response(env, original, message, st, spec):
    """Before the reveal: behave normally until the first lookup of the old id, whose result reports the new id.
    After it: the new id is accepted (and translated to the stored record), the old id is 'not found', and every later
    result shows the new id. The agent's calls are never changed except for that translation of the new id back."""
    import json as _json

    from tau2.data_model.message import ToolMessage

    old, new = spec["old"], spec["new"]
    if st.get("revealed") is None:
        resp = original(message)
        if message.name == spec["reveal_tool"] and (message.arguments or {}).get(spec["id_arg"]) == old and not resp.error:
            try:
                data = _json.loads(resp.content)
            except ValueError:
                return resp
            rest = {k: v for k, v in data.items() if k != spec["id_field"]}
            data = {spec["id_field"]: new, spec["prev_field"]: old, **rest}
            st["revealed"] = {"tool_call_id": message.id, "name": message.name, "arguments": message.arguments}
            return ToolMessage(id=message.id, content=_json.dumps(data), requestor=message.requestor, role="tool", error=False)
        return resp
    args = message.arguments or {}
    if _contains(args, old):
        st.setdefault("old_value_calls", []).append(message.id)
        return ToolMessage(id=message.id, content=f"Error: {spec['not_found']}", requestor=message.requestor, role="tool", error=True)
    if _contains(args, new):
        m2 = message.model_copy(deep=True)
        m2.arguments = _swap(args, new, old)
        resp = original(m2)
    else:
        resp = original(message)
    resp.content = (resp.content or "").replace(old, new)
    return resp


def make_fault_constructor(base_constructor, domain: str):
    """A domain constructor whose environment perturbs ONE thing, chosen by _FAULT['spec']['mode']:
       hard (default)  one assistant call returns the domain's own error and is not executed
       soft            one assistant call returns a normal-looking response showing nothing happened, and is not executed
       prop            one lookup reports that the id the agent works with was replaced by a new one (see _prop_response)"""
    import types

    from tau2.data_model.message import ToolMessage

    def constructor(**kwargs):
        env = base_constructor(**kwargs)
        original = env.get_response

        def get_response(self, message):
            st = _FAULT
            spec = st.get("spec")
            if not spec or message.requestor != "assistant":
                return original(message)
            mode = spec.get("mode", "hard")
            if mode == "prop":
                return _prop_response(self, original, message, st, spec)
            if st.get("fired") is None and message.name not in NOT_FAULTED:
                kind = "write" if self._is_mutating_tool(message.name) else "lookup"
                if mode == "soft":
                    capable = (domain, message.name) in SOFT_RECORD or (domain, message.name) in SOFT_EMPTY or (domain == "telecom" and message.name in SOFT_TELECOM)
                    if capable and kind == spec["kind"]:
                        got = soft_response(self, domain, message)
                        if got is not None:
                            st["n"] = st.get("n", 0) + 1
                            if st["n"] == spec["nth"]:
                                resp, how = got
                                content = self.to_json_str(resp)
                                st["fired"] = {"tool_call_id": message.id, "name": message.name, "arguments": message.arguments, "kind": kind, "nth_of_kind": st["n"],
                                               "message": how, "soft": True, "response_head": (content if isinstance(content, str) else str(content))[:300]}
                                return ToolMessage(id=message.id, content=content, requestor=message.requestor, role="tool", error=False)
                elif kind == spec["kind"]:
                    st["n"] = st.get("n", 0) + 1
                    if st["n"] == spec["nth"]:
                        msg = fault_message(domain, message.name, message.arguments, random.Random(spec["msg_seed"]))
                        st["fired"] = {"tool_call_id": message.id, "name": message.name, "arguments": message.arguments, "kind": kind,
                                       "nth_of_kind": st["n"], "message": msg}
                        return ToolMessage(id=message.id, content=f"Error: {msg}", requestor=message.requestor, role="tool", error=True)
            return original(message)

        env.get_response = types.MethodType(get_response, env)
        return env

    return constructor


# ───────────────────────── mind-change (correction) specs ─────────────────────────

LEADS = ["Actually, ", "Wait, sorry, ", "Hold on, let me correct myself: ", "On second thought, ", "Oops, I gave you the wrong one. ", "One correction: ", "Scratch that, "]
TIMINGS = {  # every one of them is BEFORE the agent has changed anything, otherwise the correction cannot be tested
    "after_repeat": "after the agent has repeated back or confirmed what you first asked for, and before it makes any change",
    "before_confirm": "when the agent asks you to confirm before it makes the change",
    "after_lookup": "after the agent has started looking up your account or order, but before it makes any change",
    "right_after": "in your very next message after the agent first reacts to what you asked for, before it makes any change",
}
SOURCE_DESC = {"credit_card": "credit card", "gift_card": "gift card", "paypal": "PayPal account", "certificate": "travel certificate"}
CUE = re.compile(r"actually|instead|rather|wait|sorry|mean|change|switch|scratch|hold on|oops|correct|second thought|mistake|not that|no,", re.I)


def _db(domain):
    return json.load(open(DOMAINS_DIR / domain / "db.json"))


def find_user(domain: str, instr: str, db: dict):
    users = db["users"]
    m = re.search(r"\b([a-z]+_[a-z]+_\d{3,5})\b", instr)
    if m and m.group(1) in users:
        return users[m.group(1)]
    m = re.search(r"You are ([A-Z][\w'.-]*(?: [A-Z][\w'.-]*)+?)(?: in zip code | with zip code |,|\.| and )\D*?(\d{5})?", instr)
    if m:
        first, _, last = m.group(1).partition(" ")
        for u in users.values():
            if u["name"]["first_name"] == first and u["name"]["last_name"] == last.strip() and (not m.group(2) or u["address"]["zip"] == m.group(2)):
                return u
    return None


def _shift_date(d: str, days: int) -> str:
    return (dt.date.fromisoformat(d) + dt.timedelta(days=days)).isoformat()


def _pretty_date(d: str) -> str:
    x = dt.date.fromisoformat(d)
    return f"{x.strftime('%B')} {x.day}, {x.year}"


def build_correction(domain: str, task, db: dict | None, rng: random.Random):
    """All correction specs that this task supports (one per type); the caller picks one."""
    acts = (task.evaluation_criteria.actions if task.evaluation_criteria else None) or []
    instr = str(task.user_scenario.instructions)
    user = find_user(domain, instr, db) if db else None
    out = []
    for a in acts:
        args = a.arguments or {}
        # ---- payment method (a different method of a different kind, both unique in the user's profile)
        pid = args.get("payment_method_id") or args.get("payment_id")
        if pid and user and pid in user["payment_methods"]:
            pms = user["payment_methods"]
            kinds = Counter(m["source"] for m in pms.values())
            old = pms[pid]
            alts = sorted((m for m in pms.values() if m["source"] != old["source"] and kinds[m["source"]] == 1 and m["source"] in SOURCE_DESC), key=lambda m: m["id"])
            if kinds[old["source"]] == 1 and old["source"] in SOURCE_DESC and alts:
                new = rng.choice(alts)
                out.append({"type": "payment", "tool": a.name, "paths": ["payment_method_id", "payment_id"], "old": pid, "new": new["id"],
                            "old_desc": f"your {SOURCE_DESC[old['source']]}", "new_desc": f"your {SOURCE_DESC[new['source']]}",
                            "old_markers": [SOURCE_DESC[old["source"]].split()[0].lower()], "new_markers": [SOURCE_DESC[new["source"]].split()[0].lower()]})
        # ---- address
        if a.name in ("modify_pending_order_address", "modify_user_address") and db:
            others = sorted({json.dumps(u["address"], sort_keys=True) for u in db["users"].values() if u["address"]["zip"] != args.get("zip")})
            new = json.loads(rng.choice(others))
            fmt = lambda d: f"{d['address1']}, {d['address2']}, {d['city']}, {d['state']} {d['zip']}".replace(", ,", ",")
            out.append({"type": "address", "tool": a.name, "paths": ["address1"], "old": args["address1"], "new": new["address1"],
                        "old_desc": fmt(args), "new_desc": fmt(new), "old_markers": [args["address1"].lower()], "new_markers": [new["address1"].lower()]})
        # ---- quantity: checked bags
        if a.name == "update_reservation_baggages" and isinstance(args.get("total_baggages"), int) and args["total_baggages"] >= 1:
            n = args["total_baggages"]
            out.append({"type": "baggage", "tool": a.name, "paths": ["total_baggages"], "old": n, "new": n + 1,
                        "old_desc": f"{n} checked bag{'s' if n != 1 else ''} in total", "new_desc": f"{n + 1} checked bags in total",
                        "old_markers": [str(n)], "new_markers": [str(n + 1)]})
        # ---- passenger detail (date of birth)
        if a.name == "update_reservation_passengers" and args.get("passengers") and args["passengers"][0].get("dob"):
            d = args["passengers"][0]["dob"]
            nd = _shift_date(d, 10)
            out.append({"type": "passenger_dob", "tool": a.name, "paths": ["passengers.0.dob"], "old": d, "new": nd,
                        "old_desc": f"a date of birth of {_pretty_date(d)}", "new_desc": f"a date of birth of {_pretty_date(nd)}",
                        "old_markers": [d, _pretty_date(d).lower()], "new_markers": [nd, _pretty_date(nd).lower()]})
        # ---- date
        if a.name in ("update_reservation_flights", "book_reservation") and args.get("flights") and args["flights"][0].get("date"):
            d = args["flights"][0]["date"]
            nd = _shift_date(d, 1)
            out.append({"type": "flight_date", "tool": a.name, "paths": ["flights.0.date"], "old": d, "new": nd,
                        "old_desc": f"the first flight on {_pretty_date(d)}", "new_desc": f"the first flight on {_pretty_date(nd)}",
                        "old_markers": [d, f"{dt.date.fromisoformat(d).strftime('%B').lower()} {dt.date.fromisoformat(d).day}"],
                        "new_markers": [nd, f"{dt.date.fromisoformat(nd).strftime('%B').lower()} {dt.date.fromisoformat(nd).day}"]})
        # ---- quantity: data to refuel
        if a.name == "refuel_data" and isinstance(args.get("gb_amount"), (int, float)):
            x = float(args["gb_amount"])
            nx = 1.0 if x > 1 else x + 1.0
            out.append({"type": "refuel_amount", "tool": a.name, "paths": ["gb_amount"], "old": x, "new": nx,
                        "old_desc": f"{x:g} GB of extra data", "new_desc": f"{nx:g} GB of extra data",
                        "old_markers": [f"{x:g} gb", f"{x:g}gb", f"{int(x)} gb"], "new_markers": [f"{nx:g} gb", f"{nx:g}gb", f"{int(nx)} gb"]})
        # ---- item (return): another item from the same order
        if a.name == "return_delivered_order_items" and db and len(args.get("item_ids", [])) == 1:
            order = db["orders"].get(args.get("order_id"))
            if order:
                items = order["items"]
                cur = next((i for i in items if i["item_id"] == args["item_ids"][0]), None)
                alts = [i for i in items if cur and i["name"] != cur["name"]]
                if cur and alts and sum(i["name"] == cur["name"] for i in items) == 1:
                    new = rng.choice(alts)
                    out.append({"type": "return_item", "tool": a.name, "paths": ["item_ids"], "old": cur["item_id"], "new": new["item_id"],
                                "old_desc": f"the {cur['name']}", "new_desc": f"the {new['name']}",
                                "old_markers": [cur["name"].lower()], "new_markers": [new["name"].lower()]})
    return out


def correction_addendum(spec: dict, rng: random.Random) -> tuple[str, dict]:
    lead = rng.choice(LEADS)
    timing_id = rng.choice(sorted(TIMINGS))
    form = rng.randrange(4)
    o, n, t = spec["old_desc"], spec["new_desc"], TIMINGS[timing_id]
    verb = {"payment": "pay with", "address": "use the address", "baggage": "have", "passenger_dob": "use", "flight_date": "travel for",
            "refuel_amount": "buy", "return_item": "return"}[spec["type"]]
    forms = [
        f"CHANGE OF MIND (this overrides anything above about it): at first, tell the agent you want to {verb} {o}. Then, {t}, correct yourself and say you actually want to {verb} {n} instead. "
        f"Say it in your own words, for example starting with \"{lead.strip()}\" and state the new detail in full.",
        f"UPDATE TO YOUR SCENARIO: your first choice, which you state first, is {o}. {t.capitalize()}, you change your mind and give {n} instead, because you realised the first one was wrong. "
        f"Make the correction sound natural, e.g. \"{lead.strip()}\" and then the new detail. Do not mention that you were told to change it.",
        f"ONE MID-CONVERSATION CORRECTION: begin by asking for {o}. Later, {t}, tell the agent you made a mistake and the right one is {n}. "
        f"A natural opening could be \"{lead.strip()}\". Give the new detail completely so the agent does not have to guess.",
        f"IMPORTANT: you will change your mind exactly once. First you want {o}. {t.capitalize()}, you correct it to {n}. "
        f"Do not repeat the first choice after correcting it; use words like \"{lead.strip()}\" or your own.",
    ]
    return forms[form], {"timing": timing_id, "lead": lead.strip(), "wording": form}


# ───────────────────────── plan ─────────────────────────

def load_split() -> dict:
    return json.load(open(DER / "real_split.json"))


def cmd_plan(args):
    from tau2.runner.helpers import get_tasks

    split = load_split()
    rows = []
    rng0 = random.Random(SEED)
    for dom in ("retail", "airline", "telecom"):
        in_split = {t: ("test" if t in set(split["tasks"][dom]["test"]) else "train") for t in split["tasks"][dom]["test"] + split["tasks"][dom]["train"]}
        tasks = {str(t.id): t for t in get_tasks(dom) if str(t.id) in in_split}
        db = _db(dom) if dom in ("retail", "airline") else None
        # --- corrections: tasks that support at least one type; pick the type by seed
        elig = []
        for tid in sorted(tasks):
            specs = build_correction(dom, tasks[tid], db, random.Random(f"{SEED}-spec-{dom}-{tid}"))
            if specs:
                elig.append((tid, specs))
        rng = random.Random(f"{SEED}-corr-{dom}")
        rng.shuffle(elig)
        # prefer variety of types: round-robin over types while taking tasks
        by_type = defaultdict(list)
        for tid, specs in elig:
            by_type[random.Random(f"{SEED}-type-{dom}-{tid}").choice(specs)["type"]].append((tid, specs))
        chosen, types = [], sorted(by_type)
        while len(chosen) < args.per_kind and any(by_type.values()):
            for ty in types:
                if by_type[ty] and len(chosen) < args.per_kind:
                    chosen.append((by_type[ty].pop(0), ty))
        n_corr_eligible = len(elig)
        for (tid, specs), ty in chosen:
            spec = next(s for s in specs if s["type"] == ty)
            text, meta = correction_addendum(spec, random.Random(f"{SEED}-word-{dom}-{tid}"))
            rows.append({"run_id": f"ind::{dom}::{tid}::correction", "source": "induced_correction", "domain": dom, "task_id": tid, "split": in_split[tid],
                         "correction": {**spec, **meta, "addendum": text}})
        # --- faults: any task with a lookup or write in its gold actions
        pool = [tid for tid in sorted(tasks) if (tasks[tid].evaluation_criteria and tasks[tid].evaluation_criteria.actions)]
        random.Random(f"{SEED}-fault-{dom}").shuffle(pool)
        for tid in pool[: args.per_kind]:
            r = random.Random(f"{SEED}-faultspec-{dom}-{tid}")
            rows.append({"run_id": f"ind::{dom}::{tid}::fault", "source": "induced_fault", "domain": dom, "task_id": tid, "split": in_split[tid],
                         "fault": {"kind": r.choice(["lookup", "write"]), "nth": r.choice([1, 2]), "msg_seed": r.randrange(10**9)}})
        print(f"{dom}: {len(tasks)} split tasks | correction-eligible tasks {n_corr_eligible} (types {dict(Counter(ty for _, ty in chosen))}) | faults {min(args.per_kind, len(pool))}")
    PLAN_PATH.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print("wrote", PLAN_PATH, len(rows), "runs;", dict(Counter((r["source"], r["split"]) for r in rows)))


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p)] if p.exists() else []


def _interleave(*lists):
    out, i = [], 0
    while any(i < len(l) for l in lists):
        out += [l[i] for l in lists if i < len(l)]
        i += 1
    return out


def _split_map(split: dict, dom: str) -> dict:
    t = set(split["tasks"][dom]["test"])
    return {x: ("test" if x in t else "train") for x in split["tasks"][dom]["test"] + split["tasks"][dom]["train"]}


def cmd_plan_extra(args):
    """Plans for the follow-up steps (D-019 follow-up). Same split rules as the pilot: only tasks in real_split.json."""
    from tau2.runner.helpers import get_tasks

    split, name = load_split(), args.stage.replace("plan-", "")
    rows = []
    if name == "usersim":  # the first 5 retail and 5 airline pilot corrections, rerun with another user model
        pilot = [r for r in read_jsonl(PLAN_PATH) if r["source"] == "induced_correction"]
        rows = [r for d in ("retail", "airline") for r in [x for x in pilot if x["domain"] == d][:5]]
    elif name == "class2":
        per_dom = {}
        for dom in ("retail", "airline"):
            sp = _split_map(split, dom)
            tasks = {str(t.id): t for t in get_tasks(dom) if str(t.id) in sp}
            db = _db(dom)
            pilot_ids = {r["run_id"] for r in read_jsonl(PLAN_PATH)}
            ents = []
            for tid in sorted(tasks):
                base = build_correction(dom, tasks[tid], db, random.Random(f"{SEED}-spec-{dom}-{tid}"))
                if not base:
                    continue
                types = sorted({s["type"] for s in base})
                first = random.Random(f"{SEED}-type-{dom}-{tid}").choice(base)["type"]
                order = [first] + [t for t in types if t != first]
                for rep in range(args.max_rep + 1):
                    specs = build_correction(dom, tasks[tid], db, random.Random(f"{SEED}-spec-{dom}-{tid}" + (f"-r{rep}" if rep else "")))
                    ty = order[rep % len(order)]
                    spec = next(s for s in specs if s["type"] == ty)
                    rid = f"ind::{dom}::{tid}::correction" + (f"::r{rep}" if rep else "")
                    if rid in pilot_ids:
                        continue
                    text, meta = correction_addendum(spec, random.Random(f"{SEED}-word-{dom}-{tid}-r{rep}"))
                    ents.append({"run_id": rid, "source": "induced_correction", "domain": dom, "task_id": tid, "split": sp[tid], "rep": rep, "correction": {**spec, **meta, "addendum": text}})
            per_dom[dom] = ents
        for sp_ in ("test", "train"):  # test first; inside a split: by repetition, then spread across types
            lists = []
            for dom in ("retail", "airline"):
                e = sorted((x for x in per_dom[dom] if x["split"] == sp_), key=lambda x: (x["rep"], random.Random(f"{SEED}-o-{x['run_id']}").random()))
                lists.append(e)
            rows += _interleave(*lists)
    elif name == "soft":  # test split only, all three domains
        lists = []
        for dom in ("retail", "airline", "telecom"):
            sp = _split_map(split, dom)
            tasks = [t for t in get_tasks(dom) if sp.get(str(t.id)) == "test"]
            ents = []
            for t in sorted(tasks, key=lambda t: str(t.id)):
                acts = [a for a in ((t.evaluation_criteria.actions if t.evaluation_criteria else None) or []) if getattr(a, "requestor", "assistant") == "assistant"]
                cap = lambda n: (dom, n) in SOFT_RECORD or (dom, n) in SOFT_EMPTY or (dom == "telecom" and n in SOFT_TELECOM)
                writes = [a for a in acts if cap(a.name) and ((dom, a.name) in SOFT_RECORD or a.name in SOFT_TELECOM)]
                lookups = [a for a in acts if cap(a.name) and (dom, a.name) in SOFT_EMPTY]
                kind, pool = ("write", writes) if writes else ("lookup", lookups)
                if not pool:
                    continue
                r = random.Random(f"{SEED}-soft-{dom}-{t.id}")
                nth0 = min(r.choice([1, 1, 2]), len(pool))
                if args.rep:  # a second pass over the same test tasks with the other occurrence (skip tasks that offer no other one)
                    nth = 2 if nth0 == 1 else 1
                    if len(pool) < 2 or dom == "telecom":
                        continue
                    ents.append({"run_id": f"ind::{dom}::{t.id}::soft::r{args.rep}", "source": "induced_fault_soft", "domain": dom, "task_id": str(t.id), "split": "test", "rep": args.rep,
                                 "fault": {"mode": "soft", "kind": kind, "nth": nth}})
                    continue
                ents.append({"run_id": f"ind::{dom}::{t.id}::soft", "source": "induced_fault_soft", "domain": dom, "task_id": str(t.id), "split": "test",
                             "fault": {"mode": "soft", "kind": kind, "nth": nth0}})
            random.Random(f"{SEED}-softorder-{dom}-{args.rep}").shuffle(ents)
            lists.append(ents[: args.telecom_cap] if dom == "telecom" else ents)
        rows = _interleave(*lists)
    elif name == "soft2":
        # Class 6 gap-filling (D-019 update, 2026-11-14). Test split, retail and airline. The soft plans of D-019 already used 86 of the 105 faultable gold
        # writes, so this plan has (1) the 19 (task, nth) pairs no earlier run used, and (2) reruns (r1, r2) of the pairs whose run produced a Class 6
        # candidate that passes the strict check of data/strict_check.py (same plan, another sample of the agent and user models; the candidates of one
        # task are not independent of each other, which the review keeps in mind). Rows carry strict=True.
        import strict_check as sc

        runs_old = [r for r in read_jsonl(RUNS_PATH) if r["plan"]["source"] == "induced_fault_soft"]
        used = {(r["plan"]["domain"], str(r["plan"]["task_id"]), r["plan"]["fault"]["nth"]) for r in runs_old}
        productive = []
        for r in runs_old:
            info, nxt, verdict = label_fault(r, soft=True)
            if verdict == "proceeded" and sc.strict_class6(r, nxt, info)["keep"]:
                productive.append((r["plan"]["domain"], str(r["plan"]["task_id"]), r["plan"]["fault"]["nth"]))
        fresh, reruns = [], []
        for dom in ("retail", "airline"):
            sp = _split_map(split, dom)
            for t in sorted(get_tasks(dom), key=lambda t: str(t.id)):
                if sp.get(str(t.id)) != "test":
                    continue
                n_cap = len([a for a in ((t.evaluation_criteria.actions if t.evaluation_criteria else None) or [])
                             if getattr(a, "requestor", "assistant") == "assistant" and (dom, a.name) in SOFT_RECORD])
                for nth in range(1, n_cap + 1):
                    if (dom, str(t.id), nth) not in used:
                        fresh.append({"run_id": f"ind::{dom}::{t.id}::soft2::n{nth}", "source": "induced_fault_soft", "domain": dom, "task_id": str(t.id), "split": "test", "rep": 2,
                                      "strict": True, "fault": {"mode": "soft", "kind": "write", "nth": nth}})
        for k in (1, 2):
            for dom, tid, nth in sorted(set(productive)):
                reruns.append({"run_id": f"ind::{dom}::{tid}::soft2::n{nth}::r{k}", "source": "induced_fault_soft", "domain": dom, "task_id": tid, "split": "test", "rep": 2,
                               "strict": True, "fault": {"mode": "soft", "kind": "write", "nth": nth}})
        random.Random(f"{SEED}-soft2").shuffle(fresh)
        rows = fresh + reruns
        print(f"soft2: {len(fresh)} unused (task, nth) pairs + {len(reruns)} reruns of {len(set(productive))} productive pairs")
    elif name == "prop":
        lists = {"test": [], "train": []}
        for dom in ("retail", "airline"):
            sp = _split_map(split, dom)
            db = _db(dom)
            ents = {"test": [], "train": []}
            for t in sorted(get_tasks(dom), key=lambda t: str(t.id)):
                if str(t.id) not in sp:
                    continue
                acts = (t.evaluation_criteria.actions if t.evaluation_criteria else None) or []
                key, typ, tool, argn, idf, prev, nf = (("order_id", "order_split", "get_order_details", "order_id", "order_id", "previous_order_id", "Order not found") if dom == "retail" else
                                                       ("reservation_id", "reservation_reissued", "get_reservation_details", "reservation_id", "reservation_id", "previous_reservation_id", None))
                old = next(((a.arguments or {}).get(key) for a in acts if (a.arguments or {}).get(key)), None)
                book = db["orders"] if dom == "retail" else db["reservations"]
                if not old or old not in book:
                    continue
                r = random.Random(f"{SEED}-prop-{dom}-{t.id}")
                new = NEW_ID[typ](db, r)
                while new in book:
                    new = NEW_ID[typ](db, r)
                ents[sp[str(t.id)]].append({"run_id": f"ind::{dom}::{t.id}::prop", "source": "induced_propagation", "domain": dom, "task_id": str(t.id), "split": sp[str(t.id)],
                                            "prop": {"mode": "prop", "type": typ, "old": old, "new": new, "reveal_tool": tool, "id_arg": argn, "id_field": idf, "prev_field": prev,
                                                     "not_found": nf or f"Reservation {old} not found"}})
            for s_ in ents:
                random.Random(f"{SEED}-proporder-{dom}-{s_}").shuffle(ents[s_])
                lists[s_].append(ents[s_])
        rows = _interleave(*lists["test"]) + _interleave(*lists["train"])
    path = DER / (f"induced_plan_{name}_r{args.rep}.jsonl" if args.rep else f"induced_plan_{name}.jsonl")
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print("wrote", path, len(rows), "runs;", dict(Counter((r["domain"], r["split"], r["source"]) for r in rows)))
    if name == "class2":
        print("types:", dict(Counter((r["domain"], r["correction"]["type"]) for r in rows)))


# ───────────────────────── spend tracking ─────────────────────────
SPEND_PATH = DER / "api_spend.json"
HARD_LIMIT_USD = 6.14  # was 4.50 (spent $3.64 by 2026-11-12); raised by the $2.50 cap of the 2026-11-14 gap-filling prompt, approved by the user


def record_spend(step: str, usd: float, runs: int, note: str = "") -> float:
    d = json.load(open(SPEND_PATH)) if SPEND_PATH.exists() else {"hard_limit_usd": HARD_LIMIT_USD, "scope": "the follow-up prompts of 2026-11-12 and 2026-11-14; earlier spend is in D-018/D-019", "entries": []}
    d["entries"].append({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "step": step, "runs_or_calls": runs, "usd": round(usd, 4), "note": note})
    d["total_usd"] = round(sum(e["usd"] for e in d["entries"]), 4)
    d["by_step"] = {}
    for e in d["entries"]:
        d["by_step"][e["step"]] = round(d["by_step"].get(e["step"], 0) + e["usd"], 4)
    SPEND_PATH.write_text(json.dumps(d, indent=1))
    return d["total_usd"]


def spent_total() -> float:
    return json.load(open(SPEND_PATH))["total_usd"] if SPEND_PATH.exists() else 0.0


# ───────────────────────── running ─────────────────────────


def user_args_for(model: str) -> dict:
    return {"reasoning_effort": "low"} if model.startswith("gpt-5") else {"temperature": 0.0}


def run_one(row: dict, user_model: str, agent_model: str = AGENT_MODEL) -> dict:
    import copy

    from tau2.data_model.simulation import TextRunConfig
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.registry import registry
    from tau2.runner.helpers import get_tasks
    from tau2.runner import run_single_task

    dom = row["domain"]
    task = copy.deepcopy(next(t for t in get_tasks(dom) if str(t.id) == row["task_id"]))
    domain_name = dom
    if row["source"] == "induced_correction":
        task.user_scenario.instructions.task_instructions = (task.user_scenario.instructions.task_instructions or "") + "\n\n" + row["correction"]["addendum"]
    else:
        domain_name = f"{dom}__fault"
        _FAULT.clear()
        _FAULT.update({"spec": row.get("fault") or row.get("prop"), "n": 0, "fired": None, "revealed": None})
        if domain_name not in registry.get_domains():
            registry.register_domain(make_fault_constructor(registry.get_env_constructor(dom), dom), domain_name)
    cfg = TextRunConfig(domain=domain_name, agent="llm_agent", user="user_simulator", llm_agent=agent_model, llm_args_agent=dict(AGENT_ARGS),
                        llm_user=user_model, llm_args_user=user_args_for(user_model), max_steps=MAX_STEPS[dom], max_errors=10, seed=300)
    t0 = time.time()
    sim = run_single_task(cfg, task, seed=300, evaluation_type=EvaluationType.ACTION)
    out = {"run_id": row["run_id"], "plan": row, "user_model": user_model, "agent_model": agent_model, "agent_cost": sim.agent_cost, "user_cost": sim.user_cost,
           "termination": str(sim.termination_reason), "duration_s": round(time.time() - t0, 1), "messages": json.loads(sim.model_dump_json())["messages"],
           "fault_fired": dict(_FAULT["fired"]) if row["source"] in ("induced_fault", "induced_fault_soft") and _FAULT.get("fired") else None,
           "prop_revealed": dict(_FAULT["revealed"]) if row["source"] == "induced_propagation" and _FAULT.get("revealed") else None,
           "prop_old_value_calls": list(_FAULT.get("old_value_calls", [])) if row["source"] == "induced_propagation" else None}
    return out


def _worker(args):
    row, user_model = args
    try:
        return run_one(row, user_model)
    except Exception as e:  # noqa: BLE001 - recorded in the summary, a rerun retries it
        return {"run_id": row["run_id"], "error": f"{type(e).__name__}: {str(e)[:300]}"}


def mean_run_cost(domains) -> dict:
    """Mean recorded cost per run (agent + user) of the pilot, per domain: the basis for the pre-run estimates."""
    by = defaultdict(list)
    for r in read_jsonl(RUNS_PATH):
        by[r["plan"]["domain"]].append((r["agent_cost"] or 0) + (r["user_cost"] or 0))
    return {d: (sum(by[d]) / len(by[d]) if by[d] else 0.01) for d in domains}


def progress(kind: str) -> dict:
    """Counts after each batch (labels every run in induced_runs.jsonl). kind: class2 | soft | prop."""
    runs = {}
    for r in read_jsonl(RUNS_PATH):
        runs[r["run_id"]] = r
    out = Counter()
    for run in runs.values():
        plan = run["plan"]
        if kind == "class2" and plan["source"] == "induced_correction" and plan["domain"] != "telecom":
            info, stale, ok = label_correction(run)
            out["runs"] += 1
            out["delivered"] += info["status"] == "delivered"
            out[f"cand_{plan['split']}"] += len(stale)
            out["with_relevant_call"] += bool(stale or ok)
            out["with_stale"] += bool(stale)
        elif kind == "soft" and plan["source"] == "induced_fault_soft":
            info, nxt, verdict = label_fault(run, soft=True)
            out["runs"] += 1
            out["triggered"] += info["status"] == "triggered"
            bad = verdict == "proceeded" or info.get("end") == "claimed_success_without_a_call"
            out["failures"] += bool(bad)
            out["cand_test"] += verdict == "proceeded"
        elif kind == "prop" and plan["source"] == "induced_propagation":
            info, stale, ok = label_prop(run)
            out["runs"] += 1
            out["revealed"] += info["status"] == "revealed"
            out["with_relevant_call"] += bool(stale or ok)
            out["with_stale"] += bool(stale)
            out[f"cand_{plan['split']}"] += min(len(stale), 1)
    return dict(out)


def strict_progress() -> int:
    """Soft-fault candidates from the strict plans (plan rows with strict=True) that pass the strict check of data/strict_check.py."""
    import strict_check as sc

    n = 0
    for run in read_jsonl(RUNS_PATH):
        if run["plan"].get("strict") and run["plan"]["source"] == "induced_fault_soft":
            info, nxt, verdict = label_fault(run, soft=True)
            n += verdict == "proceeded" and sc.strict_class6(run, nxt, info)["keep"]
    return n


def stage_rate(kind: str, p: dict) -> tuple[float, int]:
    if kind == "soft":
        return p.get("failures", 0) / max(p.get("triggered", 0), 1), p.get("triggered", 0)
    if kind == "prop":
        return p.get("with_stale", 0) / max(p.get("with_relevant_call", 0), 1), p.get("with_relevant_call", 0)
    return p.get("with_stale", 0) / max(p.get("with_relevant_call", 0), 1), p.get("with_relevant_call", 0)


def cmd_run(args):
    import multiprocessing as mp

    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    assert os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY missing"  # never printed
    plan_path = Path(args.plan) if args.plan else PLAN_PATH
    runs_path = Path(args.runs) if args.runs else RUNS_PATH
    plan = read_jsonl(plan_path)
    if args.only:
        plan = [r for r in plan if args.only in r["run_id"] or r["source"] == args.only or r["domain"] == args.only]
    done = {r["run_id"] for r in read_jsonl(runs_path) if r.get("user_model") == args.user_model}
    todo = [r for r in plan if r["run_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    means = mean_run_cost({"retail", "airline", "telecom"})
    per_run = lambda r: means[r["domain"]] * args.cost_factor
    est = sum(per_run(r) for r in todo[: args.max_runs or len(todo)])
    print(f"{len(todo)} runs in the plan ({len(done)} cached), user model {args.user_model}, {args.workers} workers; estimated cost of the whole list ${est:.2f}; "
          f"cap ${args.cap_usd:.2f}, batch {args.batch}", flush=True)
    spent, errors, n_done, stopped = 0.0, 0, 0, ""
    ntest_target = args.test_target
    with mp.get_context("fork").Pool(args.workers) as pool, open(runs_path, "a") as f:
        while todo and not stopped:
            if args.max_runs and n_done >= args.max_runs:
                stopped = f"max runs {args.max_runs} reached"
                break
            if args.kind == "class2" and progress("class2").get("cand_test", 0) >= ntest_target:
                todo = [r for r in todo if r["split"] != "test"]  # the test target is met: fill the training split
            batch, todo = todo[: args.batch], todo[args.batch:]
            if not batch:
                break
            b_est = sum(per_run(r) for r in batch)
            if spent + b_est > args.cap_usd:
                stopped = f"cap: spent ${spent:.2f} + next batch estimate ${b_est:.2f} would exceed ${args.cap_usd:.2f}"
                break
            if spent_total() + b_est > HARD_LIMIT_USD:
                stopped = f"HARD LIMIT ${HARD_LIMIT_USD}: total spend so far ${spent_total():.2f}"
                break
            bspent = 0.0
            for res in pool.imap_unordered(_worker, [(r, args.user_model) for r in batch]):
                if "error" in res:
                    errors += 1
                    print(f"  {res['run_id']} FAILED {res['error']}", flush=True)
                    continue
                c = (res["agent_cost"] or 0) + (res["user_cost"] or 0)
                bspent += c
                f.write(json.dumps(res) + "\n")
                f.flush()
            spent += bspent
            n_done += len(batch)
            tot = record_spend(args.step, bspent, len(batch), f"user={args.user_model}")
            msg = f"batch of {len(batch)}: ${bspent:.3f} (step total ${spent:.2f}, all-prompt total ${tot:.2f})"
            if args.kind:
                p = progress(args.kind) if runs_path == RUNS_PATH else {}
                msg += f" | {p}"
                if args.kind in ("soft", "prop") and p.get("runs", 0) >= args.early_stop_after:
                    rate, den = stage_rate(args.kind, p)
                    if rate < args.min_rate:
                        stopped = f"EARLY STOP: after {p['runs']} runs the failure rate is {rate:.1%} ({den} usable runs), under {args.min_rate:.0%}"
                if args.kind in ("soft", "prop") and p.get("cand_test", 0) >= ntest_target:
                    stopped = stopped or f"target reached: {p['cand_test']} test candidates"
            if args.strict_target and runs_path == RUNS_PATH:
                n_strict = strict_progress()
                msg += f" | strict class-6 candidates so far: {n_strict}/{args.strict_target}"
                if n_strict >= args.strict_target:
                    stopped = stopped or f"target reached: {n_strict} strict candidates"
            print(msg, flush=True)
    print(f"done: ${spent:.3f} spent in this call, {errors} errors; {stopped or 'plan finished'}")


def cmd_estimate(args):
    """Calibrate from the tau2 runs we already have (agent_cost / user_cost per simulation, gpt-4.1 agent and user),
    scaled by the price ratio of the models we will use, plus a safety factor for gpt-5-nano's reasoning tokens."""
    res = ROOT / "data" / "raw" / "tau2-bench" / "data" / "tau2" / "results" / "final"
    by = defaultdict(list)
    for f in sorted(res.glob("*gpt-4.1-2025-04-14_*_gpt-4.1-2025-04-14_4trials.json")):
        d = json.load(open(f))
        dom = f.stem.split("_")[1].split("-")[0]
        for s in d["simulations"]:
            if s.get("agent_cost") is not None and s.get("user_cost") is not None:
                steps = min(len(s["messages"]), MAX_STEPS[dom]) / max(len(s["messages"]), 1)
                by[dom].append((s["agent_cost"] * steps, s["user_cost"] * steps))
    price = {"gpt-4.1": (2.0, 8.0), "gpt-5-nano": (0.05, 0.40), "gpt-4.1-mini": (0.40, 1.60), "gpt-4.1-nano": (0.10, 0.40), "gpt-4o-mini": (0.15, 0.60)}
    ratio = lambda m: (price[m][0] * 0.7 + price[m][1] * 0.3) / (price["gpt-4.1"][0] * 0.7 + price["gpt-4.1"][1] * 0.3)
    plan = read_jsonl(PLAN_PATH)
    total, rows = 0.0, []
    for dom in ("retail", "airline", "telecom"):
        n = sum(r["domain"] == dom for r in plan)
        a = sum(x[0] for x in by[dom]) / len(by[dom])
        u = sum(x[1] for x in by[dom]) / len(by[dom])
        agent = a * ratio(AGENT_MODEL) * args.reasoning_factor
        user = u * ratio(args.user_model)
        rows.append((dom, n, a, u, agent, user))
        total += n * (agent + user)
    print(f"calibration: mean gpt-4.1 agent / user cost per simulation (capped at max_steps {MAX_STEPS})")
    for dom, n, a, u, agent, user in rows:
        print(f"  {dom:8s} runs {n:3d}  4.1 agent ${a:.3f} user ${u:.3f}  ->  {AGENT_MODEL} ${agent:.4f} + {args.user_model} ${user:.4f} per run")
    print(f"ESTIMATED COST for {len(plan)} runs: ${total:.2f}  (limit ${args.max_usd:.2f}; reasoning factor {args.reasoning_factor})")
    if total > args.max_usd:
        sys.exit("STOP: above the limit; nothing was run.")
    return total


# ───────────────────────── auto-labeling ─────────────────────────

def _lookup_path(args: dict, path: str):
    cur = args
    for p in path.split("."):
        if isinstance(cur, list):
            try:
                cur = cur[int(p)]
            except (ValueError, IndexError):
                return None
        elif isinstance(cur, dict):
            cur = cur.get(p)
        else:
            return None
    return cur


def _eq(a, b) -> bool:
    if isinstance(b, (int, float)) and not isinstance(b, bool):
        try:
            return float(a) == float(b)
        except (TypeError, ValueError):
            return False
    if isinstance(b, list):
        return a == b or (isinstance(a, list) and a == b)
    return str(a).strip().lower() == str(b).strip().lower()


def uses_value(args: dict, paths: list[str], value, list_ok: bool = False) -> bool:
    for p in paths:
        v = _lookup_path(args or {}, p)
        if v is None:
            continue
        if isinstance(v, list) and not isinstance(value, list):
            if any(_eq(x, value) for x in v):
                return True
        elif _eq(v, value):
            return True
    return False


def label_correction(run: dict):
    """Returns (info, stale_calls, correct_calls): calls after the user's correction that use the old / the new value."""
    from mine_real_errors import agent_calls

    spec = run["plan"]["correction"]
    msgs = run["messages"]
    users = [(i, (m.get("content") or "").lower()) for i, m in enumerate(msgs) if m.get("role") == "user" and m.get("content")]
    old_i = next((i for i, t in users if any(k in t for k in spec["old_markers"])), None)
    if old_i is None:
        return {"status": "old_value_never_stated"}, [], []
    new_i = next((i for i, t in users if i > old_i and any(k in t for k in spec["new_markers"]) and CUE.search(t)), None)
    if new_i is None:
        return {"status": "correction_not_delivered", "old_stated_at": old_i}, [], []
    calls = agent_calls(msgs)
    stale, ok = [], []
    for c in calls:
        if c["msg"] <= new_i:
            continue
        uses_old = uses_value(c["arguments"], spec["paths"], spec["old"])
        uses_new = uses_value(c["arguments"], spec["paths"], spec["new"])
        if uses_old and not uses_new:
            stale.append(c)
        elif uses_new and not uses_old:
            ok.append(c)
    again = any(i > new_i and any(k in t for k in spec["old_markers"]) for i, t in users)  # the user brought the old value back: scrutinise
    return {"status": "delivered", "old_stated_at": old_i, "correction_at": new_i, "correction_text": msgs[new_i]["content"][:300],
            "user_mentions_old_value_again_later": again}, stale, ok


SOFT_ACK = re.compile(r"unchanged|not (yet )?(been )?(updated|changed|applied|processed|modified|cancel+ed|completed|go(ne)? through)|did(n.t| not) (go through|work|update|change|apply|process)|"
                      r"doesn.t (show|reflect|appear)|does not (show|reflect|appear)|still (shows?|showing|pending|the old|reads?|has)|no change|nothing (changed|was changed)|"
                      r"retry|try again|verify|double.?check|re.?check|let me (check|verify|look|confirm|re)|seems (that|like) (the|it)|appears (that|the)|something went wrong|error|problem|issue|sorry|apolog", re.I)
CLAIM = re.compile(r"successfully|has been (updated|cancelled|canceled|booked|modified|processed|returned|exchanged|changed)|have been (updated|cancelled|canceled|booked|modified|processed|returned|exchanged|changed)|all set|is (now )?(updated|changed)|done", re.I)


def label_fault(run: dict, soft: bool = False):
    """Returns (info, next_call, verdict): verdict is 'proceeded' (Class 6), 'correct' or 'none' (no next call)."""
    from mine_real_errors import ACK, READ_ONLY, agent_calls

    fired = run.get("fault_fired")
    msgs = run["messages"]
    if not fired:
        return {"status": "fault_not_triggered"}, None, None
    calls = agent_calls(msgs)
    fk = next((c for c in calls if c["id"] == fired["tool_call_id"]), None)
    if fk is None:
        return {"status": "faulted_call_not_found"}, None, None
    res_i = next((i for i, m in enumerate(msgs) if m.get("role") == "tool" and m.get("id") == fk["id"]), None)
    nxt = next((c for c in calls if c["k"] > fk["k"]), None)
    end = nxt["msg"] if nxt else len(msgs)
    between = " ".join((m.get("content") or "") for m in msgs[res_i + 1:end] if m.get("role") == "assistant")
    info = {"status": "triggered", "mode": "soft" if soft else "hard",
            "faulted_call": {"call_index": fk["k"], "name": fk["name"], "arguments": fk["arguments"], "kind": fired["kind"], "message": fired["message"], **({"response_head": fired.get("response_head")} if soft else {})},
            "between_text": between[:400]}
    if soft:
        ack = bool(SOFT_ACK.search(between))
    else:
        ack = bool(ACK.search(between)) or "?" in between or bool(re.search(r"try again|retry|check|verify|confirm", between, re.I))
    if nxt is None:
        claimed = bool(CLAIM.search(between)) and not ack
        info["end"] = "claimed_success_without_a_call" if claimed else "stopped_or_informed"
        return info, None, None
    if ack:
        info["how"] = "acknowledged_before_next_call"
        return info, nxt, "correct"
    if nxt["name"] == fk["name"]:
        info["how"] = "retried_same_tool"
        return info, nxt, "correct"
    if READ_ONLY.match(nxt["name"]) or nxt["name"] in NOT_FAULTED:
        info["how"] = "different_lookup_after_error" if not soft else "looked_something_up_next"
        return info, nxt, "correct"
    info["how"] = "proceeded_with_a_write_without_acknowledging"
    return info, nxt, "proceeded"


def label_prop(run: dict):
    """Returns (info, stale_calls, ok_calls): calls after the reveal that use the old id / the new id."""
    from mine_real_errors import agent_calls

    spec = run["plan"]["prop"]
    msgs = run["messages"]
    rev = run.get("prop_revealed")
    if not rev:
        return {"status": "never_revealed"}, [], []
    res_i = next((i for i, m in enumerate(msgs) if m.get("role") == "tool" and m.get("id") == rev["tool_call_id"]), None)
    calls = agent_calls(msgs)
    stale, ok = [], []
    for c in calls:
        if c["msg"] <= res_i:
            continue
        uo, un = _contains(c["arguments"], spec["old"]), _contains(c["arguments"], spec["new"])
        if uo and not un:
            stale.append(c)
        elif un and not uo:
            ok.append(c)
    users = [(i, (m.get("content") or "")) for i, m in enumerate(msgs) if m.get("role") == "user" and i > res_i]
    again = any(spec["old"].lower() in t.lower() for i, t in users)
    return {"status": "revealed", "revealed_at": res_i, "reveal_call": rev["name"], "old": spec["old"], "new": spec["new"], "type": spec["type"], "user_mentions_old_value_after": again}, stale, ok


def cmd_label(args):
    from mine_real_errors import CLASS_NAMES, agent_calls, render_full, RESULT_CAP

    runs = {}
    for r in read_jsonl(RUNS_PATH):
        runs[r["run_id"]] = r
    runs = list(runs.values())
    review, positives = [], []
    stats = defaultdict(Counter)
    for run in runs:
        plan = run["plan"]
        dom, src = plan["domain"], plan["source"]
        S = stats[(dom, src)]
        S["runs"] += 1
        S["cost_usd"] += (run["agent_cost"] or 0) + (run["user_cost"] or 0)
        S[f"termination_{run['termination'].split('.')[-1]}"] += 1
        calls = agent_calls(run["messages"])
        turns, call_turn = render_full(run["messages"])
        picks = []  # (call, label, class, reason, induced-details)
        if src == "induced_correction":
            info, stale, ok = label_correction(run)
            S[f"correction_{info['status']}"] += 1
            if info["status"] != "delivered":
                continue
            S["correction_runs_with_relevant_call"] += bool(stale or ok)
            S["runs_with_stale_call"] += bool(stale)
            spec = plan["correction"]
            flags = []
            policy_note = ""
            if dom == "airline" and spec["type"] == "payment" and str(spec["new"]).startswith("certificate"):
                policy_note = " [airline policy: a travel certificate cannot be used to update a reservation, so the old payment method may be correct here]"
                flags.append("policy_may_make_old_value_correct")
            if info.get("user_mentions_old_value_again_later"):
                flags.append("user_mentioned_old_value_again")
            for c in stale:
                picks.append((c, "incorrect", 2, policy_note.strip() + (" " if policy_note else "") + f"uses the old value {spec['old']!r} after the user's correction to {spec['new']!r} ({spec['type']})" + (" [the user mentioned the old value again later: check it was not re-requested]" if info.get("user_mentions_old_value_again_later") else ""), {"correction": {**{k: spec[k] for k in ("type", "old", "new", "old_desc", "new_desc", "timing", "wording")}, **info}}, flags))
            for c in ok:
                picks.append((c, "correct", None, f"uses the new value {spec['new']!r} after the correction", {"correction": {**{k: spec[k] for k in ("type", "old", "new", "timing")}, **info}}, flags))
        elif src == "induced_propagation":
            info, stale, ok = label_prop(run)
            S[f"prop_{info['status']}"] += 1
            if info["status"] != "revealed":
                continue
            S["prop_runs_with_relevant_call"] += bool(stale or ok)
            S["runs_with_stale_call"] += bool(stale)
            spec = plan["prop"]
            flags = ["user_repeated_old_value"] if info["user_mentions_old_value_after"] else []
            for c in stale[:1]:  # the first stale use is the example; later ones repeat the same mistake
                picks.append((c, "incorrect", 5, f"after {info['reveal_call']} reported the new id {spec['new']!r} (replacing {spec['old']!r}) the agent still uses the old id ({spec['type']})", {"propagation": {**{k: spec[k] for k in ("type", "old", "new")}, **info}}, flags))
            for c in ok[:2]:
                picks.append((c, "correct", None, f"uses the new id {spec['new']!r} reported by {info['reveal_call']}", {"propagation": {**{k: spec[k] for k in ("type", "old", "new")}, **info}}, flags))
        else:
            soft = src == "induced_fault_soft"
            info, nxt, verdict = label_fault(run, soft=soft)
            strict_flags = []
            if plan.get("strict") and verdict == "proceeded":  # the stricter check (D-019 update): a call that fails it is neither a candidate nor a positive
                import strict_check as sc

                sv = sc.strict_class6(run, nxt, info)
                S["strict_dropped" if not sv["keep"] else "strict_kept"] += 1
                if not sv["keep"]:
                    continue
                strict_flags = ["strict_" + f for f in sv["flags"]]
            S[f"fault_{info['status']}"] += 1
            if info["status"] == "triggered":
                S[f"fault_next_{verdict or info.get('end')}"] += 1
                S[f"fault_kind_{info['faulted_call']['kind']}"] += 1
            if nxt is not None:
                what = "reported nothing happened (" + info["faulted_call"]["message"] + ")" if soft else repr(info["faulted_call"]["message"])
                reason = f"after the faulted {info['faulted_call']['name']} ({what}) the agent {info['how'].replace('_', ' ')}"
                flags = (["assistant_text_between_calls"] if info.get("between_text", "").strip() else []) + strict_flags
                picks.append((nxt, "incorrect" if verdict == "proceeded" else "correct", 6 if verdict == "proceeded" else None, reason, {"fault": info}, flags))
        for c, lab, cls, reason, extra, flags in picks:
            S[f"{lab}_calls"] += 1
            cid = f"{plan['run_id']}::C{c['k']}"
            if lab == "incorrect":
                t = call_turn[c["k"]]
                lo, hi = max(0, t - 20), min(len(turns), t + 5)
                window = []
                for x in turns[lo:hi]:
                    x = dict(x)
                    if x.get("role") == "tool" and len(x["content"]) > RESULT_CAP:
                        x["content"] = x["content"][:RESULT_CAP] + f"…[truncated, {len(x['content'])} chars]"
                    window.append(x)
                review.append({"candidate_id": cid, "run_key": plan["run_id"], "domain": dom, "task_id": plan["task_id"], "source_model": run["agent_model"], "split": plan["split"],
                               "source": src, "origin": "induced", "proposal_source": "auto_label", "auto_flags": flags,
                               "auto_status": ("test_needs_human_review" if plan["split"] == "test" else ("needs_human_review" if flags else "auto_accepted")), "proposed_class": cls, "proposed_class_name": CLASS_NAMES[cls], "evidence": "[auto-label] " + reason,
                               "flagged_call": {"call_index": c["k"], "turn_index": t, "name": c["name"], "arguments": c["arguments"]},
                               "window": {"start": lo, "end": hi, "flagged_offset": t - lo, "total_turns": len(turns)}, "turns": window, "induced": extra,
                               "heuristic_class6": None, "heuristic_class2": None, "reward_summary": {}, "human_label": "", "human_class": "", "human_note": ""})
            else:
                positives.append({"positive_id": cid, "run_key": plan["run_id"], "source": src, "origin": "induced", "auto_flags": flags, "domain": dom, "task_id": plan["task_id"], "split": plan["split"],
                                  "call_index": c["k"], "turn_index": call_turn[c["k"]], "name": c["name"], "arguments": c["arguments"], "auto_label": "correct",
                                  "reason": reason, "induced": extra, "status": "pending"})
    review.sort(key=lambda r: (r["proposed_class"], r["domain"], r["candidate_id"]))
    REVIEW_PATH.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in review))
    POSITIVES_PATH.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in positives))
    rep = {"runs": len(runs), "cost_usd": round(sum(S["cost_usd"] for S in stats.values()), 3),
           "by_domain_and_source": {f"{d}/{s}": {k: (round(v, 3) if k == "cost_usd" else v) for k, v in S.items()} for (d, s), S in sorted(stats.items())},
           "candidates": len(review), "candidates_by_class": dict(Counter(r["proposed_class"] for r in review)), "candidates_by_domain": dict(Counter(r["domain"] for r in review)),
           "candidates_by_split": dict(Counter(r["split"] for r in review)), "correct_handling_positives": len(positives)}
    REPORT_PATH.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["plan", "estimate", "run", "label", "plan-usersim", "plan-class2", "plan-soft", "plan-soft2", "plan-prop"])
    ap.add_argument("--per-kind", type=int, default=25, help="runs per domain for each perturbation")
    ap.add_argument("--user-model", default=DEFAULT_USER_MODEL)
    ap.add_argument("--max-usd", type=float, default=5.0)
    ap.add_argument("--reasoning-factor", type=float, default=2.0, help="extra cost of gpt-5-nano reasoning tokens over a non-reasoning agent")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--plan", default="", help="plan file for run (default: the pilot plan)")
    ap.add_argument("--runs", default="", help="runs cache file (default data/raw/induced/induced_runs.jsonl)")
    ap.add_argument("--step", default="pilot", help="label for the spend log")
    ap.add_argument("--kind", default="", choices=["", "class2", "soft", "prop"], help="stage kind for batch progress / early stop")
    ap.add_argument("--cap-usd", type=float, default=99.0, help="stop starting new batches when this step would exceed the cap")
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--max-runs", type=int, default=0)
    ap.add_argument("--cost-factor", type=float, default=1.0, help="multiplier on the pilot's mean cost per run (estimate safety margin)")
    ap.add_argument("--test-target", type=int, default=20)
    ap.add_argument("--early-stop-after", type=int, default=60)
    ap.add_argument("--min-rate", type=float, default=0.05)
    ap.add_argument("--strict-target", type=int, default=0, help="stop when this many strict-check Class 6 candidates exist among the strict plan rows")
    ap.add_argument("--max-rep", type=int, default=3)
    ap.add_argument("--rep", type=int, default=0, help="plan-soft: a repetition pass over the same test tasks")
    ap.add_argument("--telecom-cap", type=int, default=15)
    args = ap.parse_args()
    if args.stage.startswith("plan-"):
        return cmd_plan_extra(args)
    {"plan": cmd_plan, "estimate": cmd_estimate, "run": cmd_run, "label": cmd_label}[args.stage](args)


if __name__ == "__main__":
    main()
