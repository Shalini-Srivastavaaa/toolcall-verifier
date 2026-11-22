"""Policy text per write tool, for the optional POLICY: block of format_input_v2 (D-021).

Each snippet is copied verbatim (whitespace collapsed) from the benchmark's policy documents:
  retail   data/raw/tau2-bench/data/tau2/domains/retail/policy.md
  airline  data/raw/tau2-bench/data/tau2/domains/airline/policy.md
  telecom  data/raw/tau2-bench/data/tau2/domains/telecom/main_policy.md
A tool with no section of its own (retail modify_user_address, the telecom device tools) has no snippet. The snippet
depends on the tool only, never on the row, so a correct example and its edited twin carry the same text.

    python data/policy_snippets.py            # prints the snippets and writes data/derived/policy_snippets.json
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).parent
DOMAINS = HERE / "raw" / "tau2-bench" / "data" / "tau2" / "domains"
FILES = {"retail": DOMAINS / "retail" / "policy.md", "airline": DOMAINS / "airline" / "policy.md",
         "telecom": DOMAINS / "telecom" / "main_policy.md"}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def sections(text: str) -> dict[str, str]:
    """Heading text -> the body up to the next heading (of any level)."""
    out, name, buf = {}, None, []
    for line in text.splitlines():
        m = re.match(r"^#{2,3}\s+(.*\S)\s*$", line)
        if m:
            if name is not None:
                out[name] = "\n".join(buf)
            name, buf = m.group(1), []
        elif name is not None:
            buf.append(line)
    if name is not None:
        out[name] = "\n".join(buf)
    return out


def block(body: str, label: str) -> str:
    """The labeled block `Label:` of a section (up to the next line that looks like `Another label:`)."""
    lines, out, on = body.splitlines(), [], False
    for line in lines:
        is_label = re.match(r"^[A-Z][A-Za-z ]+:\s*$", line) is not None
        if is_label and line.strip() == label + ":":
            on = True
        elif is_label and on:
            break
        if on:
            out.append(line)
    assert out, f"block {label!r} not found"
    return "\n".join(out)


def build() -> dict[str, str]:
    S = {d: sections(p.read_text()) for d, p in FILES.items()}
    r, a, t = S["retail"], S["airline"], S["telecom"]
    modify_top = r["Modify pending order"]
    snippets = {
        "retail::cancel_pending_order": r["Cancel pending order"],
        "retail::modify_pending_order_address": modify_top,
        "retail::modify_pending_order_payment": modify_top + "\n" + r["Modify payment"],
        "retail::modify_pending_order_items": modify_top + "\n" + r["Modify items"],
        "retail::return_delivered_order_items": r["Return delivered order"],
        "retail::exchange_delivered_order_items": r["Exchange delivered order"],
        "airline::book_reservation": a["Book flight"],
        "airline::update_reservation_flights": "\n".join(block(a["Modify flight"], x) for x in ("Change flights", "Change cabin", "Payment")),
        "airline::update_reservation_baggages": block(a["Modify flight"], "Change baggage and insurance"),
        "airline::update_reservation_passengers": block(a["Modify flight"], "Change passengers"),
        "airline::cancel_reservation": a["Cancel flight"],
        "airline::send_certificate": a["Refunds and Compensation"],
        "telecom::resume_line": t["Line Suspension"],
        "telecom::suspend_line": t["Line Suspension"],
        "telecom::send_payment_request": t["Overdue Bill Payment"],
        "telecom::refuel_data": t["Data Refueling"],
        "telecom::enable_roaming": t["Data Roaming"],
        "telecom::disable_roaming": t["Data Roaming"],
    }
    out = {}
    for k, v in snippets.items():
        text = norm(v)
        assert text, k
        # every piece is a slice of the policy file (a section body or a labeled block), so it is verbatim up to whitespace
        assert all(norm(piece) in norm(FILES[k.split("::")[0]].read_text()) for piece in re.split(r"\n(?=\S)", v) if piece.strip() and len(piece) > 40), f"{k}: not verbatim"
        out[k] = text
    return out


if __name__ == "__main__":
    sn = build()
    for k, v in sn.items():
        print(f"{k:42s} {len(v):5d} chars")
    (HERE / "derived" / "policy_snippets.json").write_text(json.dumps(sn, indent=1, ensure_ascii=False) + "\n")
    print("wrote data/derived/policy_snippets.json")
