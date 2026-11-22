"""Extract tool signatures from tau2-bench domain source via static AST parsing.

We deliberately do NOT `pip install -e` tau2-bench and import it at runtime:
the package requires Python >=3.12 (we're on 3.11) and pulls in a heavy
dependency tree (litellm, fastapi, uvicorn, ...) that we don't need just to
read tool signatures. Parsing the source with `ast` is dependency-free,
version-independent, and sufficient: we only need tool name + params +
description as text to put in a judge prompt, not a machine-strict schema.
"""

import ast
import re
from pathlib import Path
from typing import NamedTuple

TAU2_ROOT = Path(__file__).parent / "raw" / "tau2-bench" / "src" / "tau2" / "domains"

# Tools whose primary/only argument is free-form natural language. tau2's own
# grader compares these with fuzzy/semantic matching that isn't captured by
# the precomputed `action_checks` exact-match we use for labeling (see
# prepare_tau2.py). We exclude them from candidate extraction rather than
# label them off a comparison method that doesn't apply to them.
FREE_TEXT_TOOLS = {"transfer_to_human_agents"}


def _parse_docstring_args(docstring: str) -> dict[str, str]:
    """Pull `param: description` pairs out of a Google-style Args: section."""
    if not docstring:
        return {}
    lines = docstring.splitlines()
    args_start = None
    for i, line in enumerate(lines):
        if line.strip().rstrip(":") == "Args":
            args_start = i + 1
            break
    if args_start is None:
        return {}
    out: dict[str, str] = {}
    current = None
    for line in lines[args_start:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.rstrip(":") in ("Returns", "Raises", "Yields"):
            break
        m = re.match(r"^(\w+):\s*(.*)$", stripped)
        if m and line.startswith((" " * 8, "\t\t")) is False or m:
            # New "name: description" line (Google style is typically indented
            # one level deeper than "Args:"; we don't rely on exact indent).
            if m and (line[:1] in (" ", "\t")):
                current = m.group(1)
                out[current] = m.group(2)
                continue
        if current is not None:
            out[current] += " " + stripped
    return out


def _summary(docstring: str) -> str:
    if not docstring:
        return ""
    return docstring.strip().splitlines()[0].strip()


def _annotation_to_str(node) -> str:
    if node is None:
        return "Any"
    try:
        return ast.unparse(node)
    except Exception:
        return "Any"


def extract_tool_schemas(domain: str) -> dict[str, dict]:
    """Return {tool_name: {description, parameters: {name: {type, description}}}}."""
    tools_file = TAU2_ROOT / domain / "tools.py"
    tree = ast.parse(tools_file.read_text())

    schemas: dict[str, dict] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for item in node.body:
            if not isinstance(item, ast.FunctionDef):
                continue
            is_tool_decorated = any(
                isinstance(d, ast.Call)
                and isinstance(d.func, ast.Name)
                and d.func.id == "is_tool"
                for d in item.decorator_list
            )
            if not is_tool_decorated:
                continue
            if item.name.startswith("_"):
                continue
            if item.name in FREE_TEXT_TOOLS:
                continue

            docstring = ast.get_docstring(item) or ""
            arg_docs = _parse_docstring_args(docstring)

            params = {}
            fn_args = item.args
            defaults_offset = len(fn_args.args) - len(fn_args.defaults)
            for i, a in enumerate(fn_args.args):
                if a.arg == "self":
                    continue
                required = i < defaults_offset
                params[a.arg] = {
                    "type": _annotation_to_str(a.annotation),
                    "description": arg_docs.get(a.arg, ""),
                    "required": required,
                }

            schemas[item.name] = {
                "description": _summary(docstring),
                "parameters": params,
            }
    return schemas


class Clause(NamedTuple):
    """One prerequisite of a tool.

    text    shown to the model as part of `# Requires: ...`
    any_of  tool names, ANY of which called earlier satisfies the clause; empty means
            the clause is not checkable against the call history (e.g. user confirmation)
    source  where the clause came from, so evaluations can exclude the contaminated ones:
              "policy"        the benchmark's own policy / workflow documents
              "user_example"  suggested by the project owner, mapped to real tool names
              "test_informed" added after looking at the real Class 7 rows in test_real.jsonl
    """

    text: str
    any_of: tuple = ()
    source: str = "policy"


def _c(text: str, any_of=(), source: str = "policy") -> Clause:
    return Clause(text, tuple(any_of), source)


_CONFIRM = _c("explicit user confirmation (yes) of the action details")
_R_AUTH = _c("user authenticated via find_user_id_by_email or find_user_id_by_name_zip",
             ["find_user_id_by_email", "find_user_id_by_name_zip"])
_R_PENDING = _c("get_order_details to confirm the status is 'pending'", ["get_order_details"])
_R_AVAIL = ["get_product_details", "get_item_details"]
_A_SEARCH = ["search_direct_flight", "search_onestop_flight"]
_T_CUST = _c("customer identified via get_customer_by_phone, get_customer_by_id or get_customer_by_name",
             ["get_customer_by_phone", "get_customer_by_id", "get_customer_by_name"])

# NOT embedded in the stored datasets (see D-009); for inference time and a rule-based layer only.
# Prerequisites per write tool, from the τ²-bench policy documents (domains/*/policy.md,
# telecom main_policy.md and tech_support_workflow_solo.md). Only tools that really exist:
# there is no unlock_sim_pin, get_plan_info or check_upgrade_fee. A PIN-locked SIM cannot be
# fixed by a tool (user_tools.lock_sim_card); the workflow says to check it with the user-side
# check_sim_status and escalate.
PREREQUISITES: dict[str, dict[str, list[Clause]]] = {
    "retail": {
        "cancel_pending_order": [_R_AUTH, _c("get_order_details to confirm the status is 'pending'", ["get_order_details"], "user_example"), _CONFIRM],
        "modify_pending_order_address": [_R_AUTH, _R_PENDING, _CONFIRM],
        "modify_pending_order_items": [
            _R_AUTH, _c("get_order_details to confirm the order is still pending", ["get_order_details"], "user_example"),
            _c("get_product_details or get_item_details to confirm the new items are available", _R_AVAIL), _CONFIRM],
        "modify_pending_order_payment": [
            _R_AUTH, _R_PENDING, _c("get_user_details to confirm the payment method and any gift-card balance", ["get_user_details"]), _CONFIRM],
        "modify_user_address": [_R_AUTH, _CONFIRM],
        "return_delivered_order_items": [
            _R_AUTH, _c("get_order_details to confirm delivery status and return eligibility", ["get_order_details"], "user_example"), _CONFIRM],
        "exchange_delivered_order_items": [
            _R_AUTH, _c("get_order_details to confirm delivery", ["get_order_details"], "user_example"),
            _c("get_product_details or get_item_details to confirm item availability", _R_AVAIL, "user_example"), _CONFIRM],
    },
    "airline": {
        "book_reservation": [
            _c("get_user_details to confirm membership and that the payment methods are in the profile", ["get_user_details"]),
            _c("search_direct_flight or search_onestop_flight to confirm availability", _A_SEARCH, "user_example"), _CONFIRM],
        "cancel_reservation": [
            _c("get_reservation_details to check cancellation eligibility (booking time, flown segments, cabin, insurance)", ["get_reservation_details"])],
        "send_certificate": [
            _c("get_flight_status to confirm the flight was delayed or cancelled", ["get_flight_status"], "user_example"),
            _c("get_user_details to confirm the membership level", ["get_user_details"]),
            _c("get_reservation_details to confirm insurance or cabin class", ["get_reservation_details"])],
        "update_reservation_baggages": [
            _c("get_reservation_details to confirm the current bags and cabin", ["get_reservation_details"]),
            _c("get_user_details to confirm the membership level", ["get_user_details"]), _CONFIRM],
        "update_reservation_flights": [
            _c("get_reservation_details to check the reservation can be modified (not basic economy, no flown segments)", ["get_reservation_details"]),
            _c("search_direct_flight or search_onestop_flight for the new flights", _A_SEARCH),
            _c("calculate for the price difference before a cabin change", ["calculate"], "user_example"), _CONFIRM],
        "update_reservation_passengers": [_c("get_reservation_details to confirm the reservation", ["get_reservation_details"]), _CONFIRM],
    },
    "telecom": {
        "resume_line": [
            _T_CUST,
            _c("get_details_by_id to confirm the line is Suspended and its contract end date is not in the past", ["get_details_by_id"]),
            _c("get_bills_for_customer to confirm all overdue bills are paid", ["get_bills_for_customer"], "user_example"),
            _c("check_sim_status to confirm the SIM is not PIN/PUK-locked (a locked SIM cannot be fixed with a tool)", ["check_sim_status"], "test_informed")],
        "suspend_line": [
            _T_CUST, _c("get_details_by_id to confirm the line is Active", ["get_details_by_id"]),
            _c("get_bills_for_customer or get_details_by_id to confirm an overdue bill or an expired contract", ["get_bills_for_customer", "get_details_by_id"])],
        "enable_roaming": [_T_CUST, _c("get_details_by_id to check whether the line is already roaming-enabled", ["get_details_by_id"], "user_example")],
        "disable_roaming": [_T_CUST, _c("get_details_by_id to check whether the line is roaming-enabled", ["get_details_by_id"])],
        "refuel_data": [
            _T_CUST, _c("get_data_usage or get_details_by_id to confirm the usage exceeds the plan limit and the refuel price",
                        ["get_data_usage", "get_details_by_id"], "user_example")],
        "send_payment_request": [_T_CUST, _c("get_bills_for_customer to confirm the bill is overdue and its amount", ["get_bills_for_customer"])],
    },
}
USER_SIDE_TOOLS = {"telecom": {"check_sim_status"}}  # device tools the agent also calls in no-user tasks
ALL_SOURCES = ("policy", "user_example", "test_informed")


def requires_text(domain: str, tool: str, sources=ALL_SOURCES) -> str | None:
    clauses = [c for c in PREREQUISITES.get(domain, {}).get(tool, []) if c.source in sources]
    return "; ".join(c.text for c in clauses) if clauses else None


def format_schemas_for_prompt(schemas: dict[str, dict], domain: str | None = None, include_requires: bool = False) -> str:
    """Render tool schemas as compact text for a judge prompt.

    By default the text carries NO prerequisite rules, and that is what the stored datasets hold
    (D-009: the rules do not help a trained classifier and the real Class 7 rows informed one of
    them). Pass `domain` and `include_requires=True` for inference time or a rule-based layer; each
    tool that has prerequisites then gets a second line right after it:
        "  # Requires: <clause>; <clause>; ..."
    """
    lines = []
    for name, spec in sorted(schemas.items()):
        params = spec["parameters"]
        param_str = ", ".join(
            f"{p}: {info['type']}" + ("" if info["required"] else " (optional)")
            for p, info in params.items()
        )
        lines.append(f"- {name}({param_str}): {spec['description']}")
        req = requires_text(domain, name) if (include_requires and domain) else None
        if req:
            lines.append(f"  # Requires: {req}")
    return "\n".join(lines)


if __name__ == "__main__":
    for domain in ("retail", "airline", "telecom"):
        schemas = extract_tool_schemas(domain)
        print(f"=== {domain}: {len(schemas)} tools ===")
        print(format_schemas_for_prompt(schemas, domain, include_requires=True))
        print()
