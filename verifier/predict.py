"""Score an agent's proposed tool call before it runs, with the fine-tuned ModernBERT v3b verifier (D-031; D-numbers refer to docs/decision_log.md).

    from verifier.predict import load, score
    load("models/final_v3b")
    v = score(conversation, {"name": "cancel_pending_order", "arguments": {"order_id": "#W1", "reason": "no longer needed"}}, domain="retail")
    print(v.probability, v.flagged)          # probability that the call is incorrect; flagged = probability >= the chosen threshold

`conversation` is the list of turns before the call, in the format of the dataset: {"role": "user" | "assistant" | "assistant_tool_call" | "tool", "content": ...}, where an
`assistant_tool_call` turn has "name" and "arguments" instead of "content" (a tool result is a "tool" turn). `turns_from_openai` converts an OpenAI-style message list.
`tool_schemas` is the text of the tool list in the dataset's format ("- name(p: type, ...): description" per line), or a list of function descriptions
(OpenAI or plain JSON-schema style), or None: for the three tau2-bench domains (retail, airline, telecom) the text used in training is then filled in.

The input text is built exactly as in training (`format_input_v2`, D-017, with the policy block of D-021 for the 18 write tools of the three tau2 domains); the model is the
one in `models/final_v3b/` and reads at most 4,096 tokens. No network access is needed once the model folder exists. Needs torch and transformers.

Operating points (data/thresholds.json): "fa05" and "fa10" are the thresholds at which 5% and 10% of the correct calls of the 363-row real test set were flagged (recall on
those rows 0.229 and 0.329), "default" is 0.5 (recall 0.494, 19.7% of correct calls flagged). **They were computed on the test set**, so they are in-sample: on other traffic the
share of correct calls flagged will differ.

CLI:  python -m verifier.predict --model-dir models/final_v3b --input call.json [--operating-point fa10] [--show-input]
      python -m verifier.predict --example        # prints a small input file
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
TOKEN_BUDGET = 4096     # the ladder below tightens the text until it fits (TOKEN_BUDGET_V2 of the training notebook)
CAND_ARG_CHARS = 300    # candidate arguments are shown almost in full
MIN_NEEDLE = 3          # candidate string values shorter than this are not searched for in tool results
POLICY_SNIPPETS = json.load(open(DATA / "policy_snippets.json"))
TAU2_SCHEMAS = json.load(open(DATA / "tau2_tool_schemas.json"))
THRESHOLDS = json.load(open(DATA / "thresholds.json"))["operating_points"]

# Tightening ladder (D-017), applied only while the text is over the budget: older non-evidence tool results, then messages, then the number of turns kept.
LEVELS = (
    [dict(n_turns=30, tool_chars=t, msg_chars=500, earlier=20, arg_chars=100) for t in (200, 100, 50, 20)]
    + [dict(n_turns=30, tool_chars=20, msg_chars=m, earlier=20, arg_chars=100) for m in (300, 200, 120)]
    + [dict(n_turns=n, tool_chars=20, msg_chars=120, earlier=20, arg_chars=100) for n in (20, 12, 8, 5, 3)]
)


# ----------------------------------------------------------------------------------------------- building the model input (the notebook's code, copied)

def _norm(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def _clip(s, n):
    s = _norm(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def _val(v, n):
    return _clip(v if isinstance(v, str) else json.dumps(v, ensure_ascii=False), n)


def _call(name, args, n):
    return f"{name}(" + ", ".join(f"{k}={_val(v, n)}" for k, v in (args or {}).items()) + ")"


def _split_top(s):
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur)
    return parts


def parse_requires(text):
    """{tool: prerequisite rule} from the "  # Requires: ..." line after a tool in the tau2 schema text."""
    rules, cur = {}, None
    for line in text.splitlines():
        if line.startswith("- ") and "(" in line:
            cur = line[2: line.index("(")].strip()
        elif cur and line.startswith("  # Requires:"):
            rules[cur] = line[len("  # Requires:"):].strip()
    return rules


def parse_schema(text):
    """{tool: [parameter names]} from the tau2 text format ("- name(p: type, ...): description") or the BFCL format ("Function: name" / "    p (type) ...")."""
    tools, cur = {}, None
    for line in text.splitlines():
        if line.startswith("- ") and "(" in line:
            name = line[2: line.index("(")].strip()
            depth, end = 0, None
            for i in range(line.index("("), len(line)):
                depth += (line[i] == "(") - (line[i] == ")")
                if depth == 0:
                    end = i
                    break
            inner = line[line.index("(") + 1: end] if end else ""
            tools[name] = [p.split(":")[0].strip() for p in _split_top(inner) if p.strip()]
        elif line.startswith("Function: "):
            cur = line[len("Function: "):].strip()
            tools[cur] = []
        elif cur and re.match(r"^    \w+ \(", line):
            tools[cur].append(line.split("(")[0].strip())
    return tools


def _needles(args):
    out = []

    def walk(v):
        if isinstance(v, str):
            if len(v) >= MIN_NEEDLE:
                out.append(v)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)

    walk(args or {})
    return out


def _format(ex, n_turns, tool_chars, msg_chars, earlier, arg_chars):
    turns, cand = ex["turns"], ex["candidate_tool_call"]
    start = max(0, len(turns) - n_turns)
    needles = _needles(cand.get("arguments"))
    tool_idx = [i for i, t in enumerate(turns) if t.get("role") == "tool"]
    recent2 = set(tool_idx[-2:])

    schema = parse_schema(ex.get("tool_schemas_text", ""))
    wanted = []
    for t in turns[-5:]:
        if t.get("role") == "assistant_tool_call" and t["name"] not in wanted:
            wanted.append(t["name"])
    if cand["name"] not in wanted:
        wanted.append(cand["name"])
    tool_strs = [f"{n}({', '.join(schema[n])})" for n in wanted if n in schema]
    rule = parse_requires(ex.get("tool_schemas_text", "")).get(cand["name"])
    if rule:
        lines = ["TOOLS:"] + [f"  {s}" + (f"  # Requires: {rule}" if s.startswith(cand["name"] + "(") else "") for s in tool_strs]
    else:
        lines = ["TOOLS: " + ("; ".join(tool_strs) if tool_strs else "(none listed)")]

    policy = POLICY_SNIPPETS.get(f"{ex.get('domain')}::{cand['name']}")
    if policy:
        lines.append("POLICY: " + policy)

    before = [t for t in turns[:start] if t.get("role") == "assistant_tool_call"]
    if before and earlier:
        lines.append("EARLIER_CALLS: " + "; ".join(_call(t["name"], t["arguments"], arg_chars) for t in before[-earlier:]))

    lines.append("HISTORY:")
    for i in range(start, len(turns)):
        t = turns[i]
        role = t.get("role")
        if role == "user":
            lines.append("USER: " + _clip(t.get("content", ""), msg_chars))
        elif role == "assistant":
            lines.append("AGENT: " + _clip(t.get("content", ""), msg_chars))
        elif role == "assistant_tool_call":
            lines.append("AGENT_CALL: " + _call(t["name"], t["arguments"], arg_chars))
        elif role == "tool":
            raw = str(t.get("content", ""))
            full = i in recent2 or any(n in raw for n in needles)   # evidence rule: a result that contains a value of the candidate call is shown in full
            lines.append("TOOL_RESULT: " + (_norm(raw) if full else _clip(raw, tool_chars)))
        else:
            lines.append(f"{str(role).upper()}: " + _clip(t.get("content", ""), msg_chars))
    lines.append("CANDIDATE: " + _call(cand["name"], cand["arguments"], CAND_ARG_CHARS))
    return "\n".join(lines)


def format_input(example, n_tokens):
    """The model input text (`format_input_v2` with the policy block). `example` has turns, candidate_tool_call, tool_schemas_text and domain."""
    for lvl in LEVELS:
        text = _format(example, **lvl)
        if n_tokens(text) <= TOKEN_BUDGET:
            break
    return text


# ----------------------------------------------------------------------------------------------- converting what a caller has

def schemas_to_text(tools):
    """A list of function descriptions (OpenAI {"type": "function", "function": {...}} or plain {"name", "description", "parameters"}) to the dataset's text format."""
    lines = []
    for t in tools:
        f = t.get("function", t)
        props = (f.get("parameters") or {}).get("properties") or {}
        params = ", ".join(f"{k}: {(v or {}).get('type', 'any')}" if isinstance(v, dict) else f"{k}: any" for k, v in props.items())
        lines.append(f"- {f['name']}({params}): {_norm(f.get('description', ''))}")
    return "\n".join(lines)


def turns_from_openai(messages):
    """OpenAI-style chat messages to the dataset's turns. System messages are dropped; an assistant message with tool_calls becomes one assistant_tool_call turn per call
    (plus an assistant turn for its text), a tool message becomes a tool turn."""
    turns = []
    for m in messages:
        role = m.get("role")
        if role == "user":
            turns.append({"role": "user", "content": m.get("content") or ""})
        elif role == "assistant":
            if m.get("content"):
                turns.append({"role": "assistant", "content": m["content"]})
            for c in m.get("tool_calls") or []:
                f = c.get("function", c)
                args = f.get("arguments", {})
                turns.append({"role": "assistant_tool_call", "name": f["name"], "arguments": json.loads(args) if isinstance(args, str) else args})
        elif role == "tool":
            turns.append({"role": "tool", "content": m.get("content") or ""})
    return turns


# ----------------------------------------------------------------------------------------------- the model

@dataclass
class Verdict:
    probability: float        # probability that the call is incorrect (the model's score; not calibrated)
    flagged: bool             # probability >= threshold
    threshold: float
    operating_point: str      # "fa05", "fa10", "default" or "custom"
    input_text: str           # exactly what the model read (including the POLICY line, if the tool has one)
    tokens: int


class Verifier:
    def __init__(self, model_dir, device=None):
        import torch
        from transformers import AutoModelForSequenceClassification, PreTrainedTokenizerFast

        model_dir = Path(model_dir)
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(model_dir / "tokenizer.json"), pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]",
                                                 mask_token="[MASK]", unk_token="[UNK]")
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(self.device).eval()
        labels = {int(k): v for k, v in self.model.config.id2label.items()}
        assert labels.get(1) == "incorrect", f"expected label 1 = 'incorrect', got {labels}"

    def n_tokens(self, text):
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def score(self, conversation, candidate_call, tool_schemas=None, domain=None, operating_point="fa10"):
        text_schemas = tool_schemas if isinstance(tool_schemas, str) else (schemas_to_text(tool_schemas) if tool_schemas else TAU2_SCHEMAS.get(domain, ""))
        example = {"turns": conversation, "candidate_tool_call": {"name": candidate_call["name"], "arguments": candidate_call.get("arguments", {})},
                   "tool_schemas_text": text_schemas, "domain": domain}
        text = format_input(example, self.n_tokens)
        enc = self.tokenizer(text, truncation=True, max_length=TOKEN_BUDGET, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            p = float(self.torch.softmax(self.model(**enc).logits.float(), dim=-1)[0, 1])
        if isinstance(operating_point, (int, float)):
            thr, name = float(operating_point), "custom"
        else:
            thr, name = THRESHOLDS[operating_point]["threshold"], operating_point
        return Verdict(p, p >= thr, thr, name, text, int(enc["input_ids"].shape[1]))


_default = None


def load(model_dir="models/final_v3b", device=None):
    """Load the model and make it the default for `score`. Returns the Verifier."""
    global _default
    _default = Verifier(model_dir, device)
    return _default


def score(conversation, candidate_call, tool_schemas=None, domain=None, operating_point="fa10", verifier=None):
    """Score one candidate call. `operating_point`: "fa05" (5% of correct calls flagged on the test set), "fa10", "default" (0.5) or a number."""
    v = verifier or _default
    if v is None:
        raise RuntimeError("call load(model_dir) first")
    return v.score(conversation, candidate_call, tool_schemas, domain, operating_point)


# ----------------------------------------------------------------------------------------------- command line

EXAMPLE = {
    "domain": "retail",
    "conversation": [
        {"role": "user", "content": "Hi, I want to cancel order #W2378156, I ordered it by mistake."},
        {"role": "assistant_tool_call", "name": "get_order_details", "arguments": {"order_id": "#W2378156"}},
        {"role": "tool", "content": "{\"order_id\": \"#W2378156\", \"status\": \"pending\", \"items\": []}"},
        {"role": "assistant", "content": "The order is pending, so I can cancel it. Please confirm the order id and the reason (ordered by mistake)."},
        {"role": "user", "content": "Yes, #W2378156, ordered by mistake. Please cancel it."},
    ],
    "candidate_call": {"name": "cancel_pending_order", "arguments": {"order_id": "#W2378156", "reason": "ordered by mistake"}},
}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Score one tool call with the ModernBERT v3b verifier.")
    ap.add_argument("--model-dir", default="models/final_v3b")
    ap.add_argument("--input", help="JSON file: {conversation, candidate_call, domain, tool_schemas (optional)}")
    ap.add_argument("--operating-point", default="fa10", help="fa05, fa10, default (0.5) or a number")
    ap.add_argument("--show-input", action="store_true", help="also print the text the model read")
    ap.add_argument("--example", action="store_true", help="print an example input file and exit")
    a = ap.parse_args(argv)
    if a.example:
        print(json.dumps(EXAMPLE, indent=1))
        return 0
    if not a.input:
        ap.error("--input is required (or --example)")
    req = json.load(open(a.input))
    op = float(a.operating_point) if re.fullmatch(r"[0-9.]+", a.operating_point) else a.operating_point
    load(a.model_dir)
    v = score(req["conversation"], req["candidate_call"], req.get("tool_schemas"), req.get("domain"), op)
    out = {"probability_incorrect": round(v.probability, 4), "flagged": v.flagged, "threshold": round(v.threshold, 4), "operating_point": v.operating_point, "tokens": v.tokens}
    print(json.dumps(out, indent=1))
    if a.show_input:
        print("\n--- text the model read ---\n" + v.input_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
