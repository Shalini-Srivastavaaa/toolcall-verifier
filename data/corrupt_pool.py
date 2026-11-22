"""Phase 3: Generate synthetic negative examples by corrupting correct examples.

Reads from:
  data/derived/train_pool.jsonl   (tau2-bench correct examples)
  data/derived/bfcl_pool.jsonl    (BFCL v4 correct examples)

For each correct example we attempt to produce one negative per applicable
error class.  Two corruption strategies are used:

  Programmatic (no LLM, zero cost):
    Class 2 — Stale State / Ignored Correction (inject "Actually, make that X not Y.")
    Class 4 — Repeated Call / Duplicate Side Effect
    Class 5 — Failure to Propagate Tool Output (tau2-bench only; swap in another context value)
    Class 7 — Missing Precondition / Wrong Order

  LLM-assisted (OpenAI API, GPT-5-nano or GPT-5-mini):
    Class 1 — Wrong Argument Value
    Class 3 — Wrong Tool Selection
    Class 6 — Ignoring a Tool Error

Output: data/derived/negatives_pool.jsonl
Each output line has the same schema as train_pool.jsonl plus:
  "error_class":      int (1-7)
  "error_class_name": str  (snake_case)
  "corruption_method": "programmatic" | "llm"
  "label":            "incorrect"

Design notes:
- Each correct example generates at most ONE negative per class (hard cap to
  avoid overwhelming one class).
- For programmatic corruptions: we directly mutate the candidate_tool_call
  or inject/remove turns in a rule-based way.
- For LLM corruptions: we send the full context + candidate + a class-specific
  system prompt, and ask the model to return a mutated candidate_tool_call.
  We validate the response (must be same tool name except for Class 3, must
  differ from the original call, must parse as valid JSON).
- Class 4 programmatic: inject a copy of the candidate call earlier in the
  turns list (as an assistant_tool_call turn), signaling the call was already
  made. This requires >=1 user turn preceding the candidate so there's
  a natural injection point.
- Class 7 programmatic: from BFCL miss_func/miss_param examples we already
  know when NO call should happen. For tau2-bench we look for adjacent
  tool calls in the turns that are "setup" calls (get/fetch/check/list prefix)
  and produce a version that *skips* the setup: we remove the last setup
  turn-pair so the context looks like the agent skipped the prerequisite.

Run:
  python data/corrupt_pool.py [options]

Options:
  --in-pools       space-separated paths (default: derived/train_pool.jsonl derived/bfcl_pool.jsonl)
  --out            output path (default: derived/negatives_pool.jsonl)
  --classes        which error classes to generate (default: 1 2 3 4 5 6 7)
  --llm-model      OpenAI reasoning model for LLM corruptions (default: gpt-5-nano)
  --max-llm        max number of LLM calls (default: 1000, ~$1 at nano pricing)
  --max-per-class  max negatives per error class (default: 200)
  --dry-run        show stats only, don't call LLM or write output
  --seed           random seed (default: 42)
  --workers        parallel LLM calls (default: 8)
  --api-key        OpenAI API key (default: env OPENAI_API_KEY)

Cost estimate at defaults:
  ~1000 LLM calls at gpt-5-nano pricing ($0.05/$0.40 per 1M in/out) = ~$1 or less
"""

import argparse
import copy
import json
import os
import random
import re
import sys
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

# ── error class registry ──────────────────────────────────────────────────────

ERROR_CLASSES = {
    1: "wrong_arg_value",
    2: "stale_state",
    3: "wrong_tool_selection",
    4: "repeated_call",
    5: "failure_to_propagate",
    6: "ignored_error",
    7: "missing_precondition",
}

PROGRAMMATIC_CLASSES = {2, 4, 5, 7}
LLM_CLASSES = {1, 3, 6}

# Setup-looking tool name prefixes — used by Class 7 corruption.
# A call whose name starts with any of these is considered a "lookup" that
# would be a natural prerequisite.
LOOKUP_PREFIXES = (
    "get_", "fetch_", "check_", "list_", "search_", "find_", "look_",
    "retrieve_", "query_", "read_", "show_",
)

# ── LLM prompt templates ──────────────────────────────────────────────────────

_LLM_SYSTEM = """\
You are a tool-call corruption engine for a training data pipeline.
You receive a multi-turn conversation context and a CORRECT tool call.
Your job is to return a CORRUPTED version of that tool call that illustrates
a specific error class. The corrupted call must be plausible-looking — it should
NOT be obviously broken. It should look like a realistic agent mistake.

Rules:
- Return ONLY a JSON object: {"name": "...", "arguments": {...}}
- Do NOT include any explanation, markdown, or extra text.
- The returned call must differ from the original.
- The corruption should match the error class described in the user prompt.
- For all classes except Class 3, keep the same tool name.
"""

_CLASS_PROMPTS = {
    1: """\
Error Class 1 — Wrong Argument Value
The tool name must stay the same. Change exactly ONE argument to a plausible
but wrong value. The wrong value must come from somewhere in the conversation
context (a different number, name, ID, or date that appears elsewhere in the
turns). Do NOT invent values that aren't mentioned anywhere in the context.

Conversation context (earlier turns):
{turns_text}

Available tools:
{schema_text}

Correct tool call:
{candidate_json}

Return the corrupted tool call JSON:""",

    3: """\
Error Class 3 — Wrong Tool Selection
Change the tool name to a different tool from the schema that is superficially
plausible for the user's intent but is NOT the right tool. The arguments
must be adjusted to be valid for the wrong tool (reuse values from the context
where possible). Pick a tool that shares some purpose with the correct one —
e.g., if the correct tool modifies items, pick a tool that modifies a related
entity (like payment method instead of items).

Conversation context (earlier turns):
{turns_text}

Available tools:
{schema_text}

Correct tool call:
{candidate_json}

Return the corrupted tool call JSON (different tool name, plausible arguments):""",

    6: """\
Error Class 6 — Ignoring a Tool Error
Return the candidate call JSON UNCHANGED (copy it exactly). Do not modify the tool name or any arguments. The error in this class is represented by the context mutation already applied — a failing predecessor call was injected into the turns. Your job is only to echo the original call back.

Correct tool call:
{candidate_json}

Return (copy exactly):""",
}

# Class 1, second pass ("context swap"): the wrong value must be a real value from elsewhere in the same
# conversation. Used by --class1-context-swap; the original prompt above produced the first 146 negatives.
_CLASS1_CONTEXT_SWAP_PROMPT = """\
Error Class 1 — Wrong Argument Value (context swap)
Keep the tool name the same. Change exactly ONE argument so that it carries a WRONG value, and that
wrong value must be a real value that appears elsewhere in this conversation: a different order ID, a
different customer ID, a different item ID, a date or amount from a previous turn, a different address or
name, and so on. The mistake is the agent picking the wrong one of several values that are all present
in the context.

Requirements:
- Copy the replacement value exactly as it appears in the conversation (same spelling and format). Do not
  invent, edit, round or combine values.
- It must be the same kind of value as the one it replaces (an order ID for an order ID, a date for a date),
  so the call still looks well-formed.
- It must be genuinely wrong for this call: a value that belongs to a different order, item, person or date,
  not another spelling of the same thing.
- Do not add or remove arguments and do not change the tool name.

Examples (abbreviated):

Context: the user asks about two orders, #W1111111 (the order to cancel) and #W2222222 (a different order they
only want to check).
Correct call: {{"name": "cancel_pending_order", "arguments": {{"order_id": "#W1111111", "reason": "no longer needed"}}}}
Corrupted:    {{"name": "cancel_pending_order", "arguments": {{"order_id": "#W2222222", "reason": "no longer needed"}}}}

Context: a tool result lists reservations ABC123 (the one the user wants to change) and XYZ789 (an earlier
booking for a different trip).
Correct call: {{"name": "update_reservation_baggages", "arguments": {{"reservation_id": "ABC123", "total_baggages": 2}}}}
Corrupted:    {{"name": "update_reservation_baggages", "arguments": {{"reservation_id": "XYZ789", "total_baggages": 2}}}}

Context: the user first mentions 2024-05-15 as the original flight date, then says to move the flight to 2024-05-22.
Correct call: {{"name": "search_direct_flight", "arguments": {{"origin": "JFK", "destination": "SEA", "date": "2024-05-22"}}}}
Corrupted:    {{"name": "search_direct_flight", "arguments": {{"origin": "JFK", "destination": "SEA", "date": "2024-05-15"}}}}

Conversation context (earlier turns):
{turns_text}

Available tools:
{schema_text}

Correct tool call:
{candidate_json}

Return the corrupted tool call JSON:"""


# ── programmatic corruptions ──────────────────────────────────────────────────

def corrupt_class4(ex: dict, rng: random.Random) -> Optional[dict]:
    """Class 4: Repeated Call — inject the candidate call earlier in the turns.

    We insert a copy of the candidate tool call as an assistant_tool_call turn
    earlier in the conversation, plus a synthetic tool result confirming it
    succeeded. This makes the conversation look like the call was already made.
    """
    turns = ex["turns"]
    candidate = ex["candidate_tool_call"]

    # Find the last user/system turn index to inject BEFORE
    # We need at least one user turn so the duplication is non-trivial
    user_turn_indices = [i for i, t in enumerate(turns) if t.get("role") == "user"]
    if not user_turn_indices:
        return None

    # Pick an injection point: somewhere in the middle of the turns
    # We want the duplicate to appear a bit before the end so it reads
    # as "already done earlier in the session"
    injection_pool = user_turn_indices[:-1] if len(user_turn_indices) > 1 else user_turn_indices
    inject_after = rng.choice(injection_pool)

    # Build the injected turns: tool call + synthetic success result
    injected_call = {
        "role": "assistant_tool_call",
        "name": candidate["name"],
        "arguments": copy.deepcopy(candidate["arguments"]),
    }
    injected_result = {
        "role": "tool",
        "content": '{"status": "success"}',
    }

    new_turns = (
        list(turns[: inject_after + 1])
        + [injected_call, injected_result]
        + list(turns[inject_after + 1 :])
    )

    corrupted = copy.deepcopy(ex)
    corrupted["turns"] = new_turns
    # The candidate_tool_call stays the same — it's now a duplicate
    return corrupted


def corrupt_class7(ex: dict, rng: random.Random) -> Optional[dict]:
    """Class 7: Missing Precondition — remove the last lookup/setup call from turns.

    We look for the last assistant_tool_call in the turns whose name matches
    a LOOKUP_PREFIXES pattern. We remove that call and its following tool result
    (if any). This simulates an agent that skipped the prerequisite lookup
    and jumped straight to the action.

    Only valid when there are at least 2 distinct tool calls in the turns
    (otherwise removing one leaves no history at all, which isn't realistic).
    """
    turns = ex["turns"]

    # Find assistant_tool_call turns with lookup-prefixed names
    call_indices = [
        i for i, t in enumerate(turns)
        if t.get("role") == "assistant_tool_call"
        and any(t.get("name", "").startswith(p) for p in LOOKUP_PREFIXES)
    ]

    if len(call_indices) < 1:
        return None  # No lookup calls to remove

    # We need at least one user turn so the context is not just empty turns
    user_turns = [t for t in turns if t.get("role") == "user"]
    if not user_turns:
        return None

    # Remove the last lookup call (and its following tool result if present)
    remove_idx = call_indices[-1]
    indices_to_remove = {remove_idx}

    # If the next turn is a tool result, remove it too
    if remove_idx + 1 < len(turns) and turns[remove_idx + 1].get("role") == "tool":
        indices_to_remove.add(remove_idx + 1)

    new_turns = [t for i, t in enumerate(turns) if i not in indices_to_remove]

    if not new_turns:
        return None

    corrupted = copy.deepcopy(ex)
    corrupted["turns"] = new_turns
    return corrupted


# ── programmatic Class 2 (stale state) and Class 5 (failure to propagate) ────

_IDLIKE = re.compile(r"^\S{3,60}$")
_TOKEN = re.compile(r"[#A-Za-z0-9_\-.@:]{3,}")
_NUMERIC_STR = re.compile(r"^\d+(\.\d+)?$")


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_idlike(v) -> bool:
    return isinstance(v, str) and bool(_IDLIKE.match(v)) and any(ch.isdigit() for ch in v)


def _norm(v) -> str:
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def _appears(value: str, text: str) -> bool:
    return re.search(r"(?<![A-Za-z0-9])" + re.escape(value) + r"(?![A-Za-z0-9])", text) is not None


def _shape(s: str) -> str:
    return re.sub(r"[0-9]", "0", re.sub(r"[A-Za-z]", "a", s))


def _stale_number(v, rng: random.Random):
    factors = [1.5, 2.0, 0.5]
    rng.shuffle(factors)
    for f in factors:
        old = int(round(v * f))
        if old != v:
            return float(old) if isinstance(v, float) else old
    return None


def _stale_id(s: str, rng: random.Random) -> str:
    last = s[-1]
    if rng.random() < 0.5 or not last.isalnum():
        return s + str(rng.randint(0, 9))
    if last.isdigit():
        return s[:-1] + rng.choice([d for d in "0123456789" if d != last])
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if last.isupper() else "abcdefghijklmnopqrstuvwxyz"
    return s[:-1] + rng.choice([c for c in letters if c != last])


def corrupt_class2(ex: dict, rng: random.Random) -> Optional[dict]:
    """Class 2: Stale State — the user corrects old -> new, the call still uses old.

    Pick one scalar argument (numbers preferred, then ID-like strings), fabricate
    a plausible stale value, insert a user turn "Actually, make that <new> not
    <old>." near the end of the context, and make the candidate call use <old>.
    """
    args = ex["candidate_tool_call"]["arguments"]
    nums = [k for k, v in args.items() if _is_number(v)]
    ids = [k for k, v in args.items() if _is_idlike(v)]
    rng.shuffle(nums)
    rng.shuffle(ids)

    chosen = None
    for k in nums + ids:
        new = args[k]
        old = _stale_number(new, rng) if _is_number(new) else _stale_id(new, rng)
        if old is not None and old != new:
            chosen = (k, new, old)
            break
    if chosen is None:
        return None
    k, new, old = chosen

    turns = list(ex["turns"])
    n = len(turns)
    # Insert within the last few turns, never between a tool call and its result.
    positions = [
        i for i in range(max(0, n - 5), n + 1)
        if not (i > 0 and turns[i - 1].get("role") == "assistant_tool_call")
    ]
    if not positions:
        return None
    turns.insert(rng.choice(positions), {"role": "user", "content": f"Actually, make that {new} not {old}."})

    corrupted = copy.deepcopy(ex)
    corrupted["turns"] = turns
    corrupted["candidate_tool_call"]["arguments"][k] = old
    corrupted["corruption_detail"] = {"arg": k, "correct_value": new, "stale_value": old}
    return corrupted


def _flatten(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _flatten(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _flatten(v)
    elif obj is not None and not isinstance(obj, bool):
        yield obj


def _tool_values(content: str) -> set[str]:
    """String forms of the values in a tool result (JSON leaves plus raw tokens,
    so plain-text and 2000-char-truncated results still yield values)."""
    out: set[str] = set()
    try:
        out |= {_norm(v) for v in _flatten(json.loads(content))}
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    out |= {t.strip(".,:;") for t in _TOKEN.findall(content)}
    return out


def corrupt_class5(ex: dict, rng: random.Random) -> Optional[dict]:
    """Class 5: Failure to Propagate — the agent had the right value from the
    last tool result but the call uses a different one from elsewhere in context.

    tau2-bench only (BFCL examples have no tool-result turns). The original
    argument value must come from the last tool result and must NOT also appear
    in a user turn, so its provenance really is the tool output.
    """
    if ex.get("source_dataset") == "bfcl" or ex.get("domain") == "bfcl":
        return None
    turns = ex["turns"]
    tool_idx = [i for i, t in enumerate(turns) if t.get("role") == "tool"]
    if not tool_idx:
        return None

    last_vals = _tool_values(str(turns[tool_idx[-1]].get("content", "")))
    user_text = "\n".join(str(t.get("content", "")) for t in turns if t.get("role") == "user")
    args = ex["candidate_tool_call"]["arguments"]

    target = None
    for k, v in args.items():
        elems = list(enumerate(v)) if isinstance(v, list) else [(None, v)]
        for idx, e in elems:
            if not (_is_idlike(e) or (_is_number(e) and abs(e) >= 10)):
                continue
            s = _norm(e)
            if s in last_vals and not _appears(s, user_text):
                target = (k, idx, e, s)
                break
        if target:
            break
    if target is None:
        return None
    k, idx, e, s = target

    earlier_vals: set[str] = set()
    for i in tool_idx[:-1]:
        earlier_vals |= _tool_values(str(turns[i].get("content", "")))
    user_vals = {t.strip(".,:;") for t in _TOKEN.findall(user_text)}
    current = {_norm(x) for v in args.values() for x in (v if isinstance(v, list) else [v])}

    shape = _shape(s)
    cands = sorted(
        c for c in (earlier_vals | user_vals)
        if c != s and c not in current and c not in last_vals and _shape(c) == shape
        and ((_is_number(e) and _NUMERIC_STR.match(c)) or (isinstance(e, str) and _is_idlike(c)))
    )
    if not cands:
        return None
    repl = rng.choice(cands)
    if _is_number(e):
        repl = float(repl) if isinstance(e, float) else int(float(repl))

    corrupted = copy.deepcopy(ex)
    cargs = corrupted["candidate_tool_call"]["arguments"]
    if idx is None:
        cargs[k] = repl
    else:
        cargs[k][idx] = repl
    corrupted["corruption_detail"] = {
        "arg": k, "index": idx, "tool_result_value": s, "replaced_with": repl,
    }
    return corrupted


# ── LLM corruption ────────────────────────────────────────────────────────────

def _format_turns_for_prompt(turns: list[dict], max_chars: int = 4000) -> str:
    """Render turns as a readable text block for the LLM prompt."""
    lines = []
    for t in turns:
        role = t.get("role", "?")
        if role == "user":
            lines.append(f"User: {t.get('content', '')}")
        elif role == "assistant":
            lines.append(f"Assistant: {t.get('content', '')}")
        elif role == "assistant_tool_call":
            args_str = json.dumps(t.get("arguments", {}))
            lines.append(f"Tool call: {t.get('name', '')}({args_str})")
        elif role == "tool":
            lines.append(f"Tool result: {t.get('content', '')[:400]}")
        elif role == "system":
            lines.append(f"System: {t.get('content', '')[:400]}")
    text = "\n".join(lines)
    if len(text) > max_chars:
        # Keep the start and end, truncate the middle
        half = max_chars // 2
        text = text[:half] + "\n... [truncated] ...\n" + text[-half:]
    return text


def _call_openai(
    client,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout: float = 30.0,
) -> Optional[str]:
    """Call OpenAI chat completions. Returns response text or None on error."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            reasoning_effort="low",
            timeout=timeout,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"    [LLM error] {e}", file=sys.stderr)
        return None


def _parse_llm_response(text: Optional[str]) -> Optional[dict]:
    """Parse LLM response as JSON tool call {name, arguments}."""
    if not text:
        return None
    # Strip markdown code fences if present
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last fence lines
        lines = [l for l in lines if not l.startswith("```")]
        text = "\n".join(lines).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
            if isinstance(obj["arguments"], dict):
                return obj
        return None
    except json.JSONDecodeError:
        return None


def corrupt_llm(
    ex: dict,
    error_class: int,
    client,
    model: str,
    rng: random.Random,
    template: Optional[str] = None,
) -> Optional[dict]:
    """Corrupt using the LLM. Returns a new example dict or None on failure.
    `template` replaces the class's default prompt (used for the Class 1 context-swap pass)."""
    turns_text = _format_turns_for_prompt(ex["turns"])
    schema_text = ex.get("tool_schemas_text", "(schema not available)")
    candidate_json = json.dumps(ex["candidate_tool_call"], indent=2)

    template = template or _CLASS_PROMPTS.get(error_class)
    if template is None:
        return None

    user_prompt = template.format(
        turns_text=turns_text,
        schema_text=schema_text[:3000],
        candidate_json=candidate_json,
    )

    response_text = _call_openai(client, model, _LLM_SYSTEM, user_prompt)
    corrupted_call = _parse_llm_response(response_text)
    if corrupted_call is None:
        return None

    # Validate: must differ from original (except Class 6, whose prompt asks
    # the model to echo the call back unchanged; its corruption is in the context)
    if error_class != 6 and corrupted_call == ex["candidate_tool_call"]:
        return None  # LLM returned original unchanged

    # For non-Class3: tool name must stay the same
    if error_class != 3 and corrupted_call["name"] != ex["candidate_tool_call"]["name"]:
        return None

    # For Class 3: tool name must differ
    if error_class == 3 and corrupted_call["name"] == ex["candidate_tool_call"]["name"]:
        return None

    corrupted = copy.deepcopy(ex)

    # Class 6 special case: the mutation is in context (we inject a failed preceding call)
    if error_class == 6:
        corrupted = _inject_error_context(corrupted, rng)
        if corrupted is None:
            return None
        # Keep candidate_tool_call as-is (the "ignore" is the error)
        # but mark that the context was mutated
    else:
        corrupted["candidate_tool_call"] = corrupted_call

    return corrupted


def _inject_error_context(ex: dict, rng: random.Random) -> Optional[dict]:
    """Class 6: inject a synthetic failed tool call just before the candidate.

    We fabricate a preceding call that returns an error. The candidate call
    then proceeds as if it succeeded — illustrating 'ignored error'.
    The failing call is one that would logically precede the candidate
    (same prefix but a 'setup' variant, or same name with different args
    to simulate a partial failure).
    """
    turns = list(ex["turns"])
    candidate = ex["candidate_tool_call"]

    # Determine a plausible "failing predecessor" tool name.
    # Strategy: use the candidate's name but with the _first_ arg changed to
    # something that would plausibly fail (e.g., an ID that looks wrong).
    # Simpler: just emit a "check_" or "verify_" prefixed variant.
    cname = candidate["name"]
    # Build a plausible failure call name
    for prefix in LOOKUP_PREFIXES:
        if cname.startswith(prefix):
            fail_name = "verify_" + cname[len(prefix):]
            break
    else:
        fail_name = "check_" + cname

    # Inject at the very end of turns (just before the candidate would be called)
    fail_call = {
        "role": "assistant_tool_call",
        "name": fail_name,
        "arguments": copy.deepcopy(candidate["arguments"]),
    }
    fail_result = {
        "role": "tool",
        "content": '{"status": "error", "message": "Operation failed: prerequisite not met"}',
    }

    new_turns = turns + [fail_call, fail_result]
    corrupted = copy.deepcopy(ex)
    corrupted["turns"] = new_turns
    return corrupted


# ── deduplication ─────────────────────────────────────────────────────────────

def _content_key(turns: list, candidate: dict) -> str:
    payload = json.dumps(turns, sort_keys=True) + json.dumps(candidate, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


# ── main pipeline ─────────────────────────────────────────────────────────────

def load_pool(path: Path) -> list[dict]:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def finalize_negative(base: dict, error_class: int, method: str, id_suffix: Optional[str] = None) -> dict:
    """Add error_class fields and set label=incorrect on a corrupted example."""
    ex = copy.deepcopy(base)
    ex["label"] = "incorrect"
    ex["error_class"] = error_class
    ex["error_class_name"] = ERROR_CLASSES[error_class]
    ex["corruption_method"] = method
    # Overwrite trajectory_id to signal synthetic origin
    ex["trajectory_id"] = ex.get("trajectory_id", "unknown") + (id_suffix or f"::corrupt{error_class}")
    ex["label_tier"] = f"synthetic_class{error_class}"
    return ex


def run_programmatic(
    examples: list[dict],
    classes: set[int],
    rng: random.Random,
    max_per_class: int,
) -> list[dict]:
    """Generate all programmatic (non-LLM) negatives."""
    # Shuffle so the per-class cap samples across tau2-bench and BFCL rather
    # than filling entirely from whichever pool file was loaded first.
    examples = list(examples)
    rng.shuffle(examples)
    results = []
    counts = {c: 0 for c in PROGRAMMATIC_CLASSES if c in classes}

    # Classes 2 and 5 get a "p" id suffix so they can't collide with the
    # earlier LLM-generated Class 2/5 negatives built from the same base examples.
    generators = {
        2: (corrupt_class2, "::corrupt2p"),
        4: (corrupt_class4, None),
        5: (corrupt_class5, "::corrupt5p"),
        7: (corrupt_class7, None),
    }
    for ex in examples:
        if ex.get("label") != "correct":
            continue
        for cls, (fn, suffix) in generators.items():
            if cls in counts and counts[cls] < max_per_class:
                neg = fn(ex, rng)
                if neg is not None:
                    results.append(finalize_negative(neg, cls, "programmatic", suffix))
                    counts[cls] += 1

    for c, n in counts.items():
        print(f"  Class {c} ({ERROR_CLASSES[c]}): {n} programmatic negatives")
    return results


def run_llm_corruptions(
    examples: list[dict],
    classes: set[int],
    rng: random.Random,
    client,
    model: str,
    max_llm_total: int,
    max_per_class: int,
    workers: int,
) -> list[dict]:
    """Generate LLM-assisted negatives with parallel API calls."""

    llm_classes = sorted(LLM_CLASSES & classes)
    if not llm_classes:
        return []

    # Build work queue: (example, error_class) pairs
    # We spread budget evenly across classes
    budget_per_class = min(max_per_class, max_llm_total // max(len(llm_classes), 1))
    print(f"  LLM budget: {budget_per_class} calls per class × {len(llm_classes)} classes")

    # Shuffle examples for each class independently
    work_items = []
    for cls in llm_classes:
        pool = [ex for ex in examples if ex.get("label") == "correct"]
        rng.shuffle(pool)
        for ex in pool[:budget_per_class * 2]:  # request 2× in case of failures
            work_items.append((ex, cls))
    rng.shuffle(work_items)

    # Deduplicate across the run
    seen_keys: set[str] = set()
    counts = {c: 0 for c in llm_classes}
    results = []
    llm_calls_made = 0

    def do_corrupt(item):
        ex, cls = item
        return corrupt_llm(ex, cls, client, model, rng)

    print(f"  Submitting up to {len(work_items)} LLM tasks ({workers} workers)...")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(do_corrupt, item): item for item in work_items}
        for future in as_completed(futures):
            ex_orig, cls = futures[future]
            llm_calls_made += 1
            if llm_calls_made % 100 == 0:
                done = sum(counts.values())
                print(f"    {llm_calls_made} LLM calls done | {done} negatives so far")

            if counts.get(cls, 0) >= max_per_class:
                continue
            try:
                neg = future.result()
            except Exception as e:
                print(f"    [future error cls={cls}] {e}", file=sys.stderr)
                neg = None

            if neg is None:
                continue

            key = _content_key(neg["turns"], neg["candidate_tool_call"])
            if key in seen_keys:
                continue
            seen_keys.add(key)

            results.append(finalize_negative(neg, cls, "llm"))
            counts[cls] += 1

    for c in llm_classes:
        print(f"  Class {c} ({ERROR_CLASSES[c]}): {counts.get(c, 0)} LLM negatives "
              f"(from {llm_calls_made} calls)")
    return results


# ── Class 1 context-swap pass ─────────────────────────────────────────────────

def _shape(v) -> str:
    return re.sub(r"[A-Za-z]+", "a", re.sub(r"\d+", "9", str(v)))


def _scalar_leaves(v) -> list:
    if isinstance(v, dict):
        return [x for e in v.values() for x in _scalar_leaves(e)]
    if isinstance(v, (list, tuple)):
        return [x for e in v for x in _scalar_leaves(e)]
    return [] if v is None or isinstance(v, bool) else [v]


def _in_context(value, ctx: str) -> bool:
    return re.search(r"(?<![A-Za-z0-9])" + re.escape(str(value)) + r"(?![A-Za-z0-9])", ctx) is not None


def _context_text(turns: list[dict]) -> str:
    return "\n".join(str(t.get("content", "")) + " " + json.dumps(t.get("arguments", "")) for t in turns)


def has_swap_candidate(ex: dict) -> bool:
    """True if some argument value of the call has a same-shaped, different value elsewhere in the context,
    i.e. a context swap is possible. Spares LLM calls on conversations that offer nothing to swap in."""
    ctx = _context_text(ex["turns"])
    tokens = set(re.findall(r"[A-Za-z0-9#_@.\-]+", ctx))
    for v in _scalar_leaves(ex["candidate_tool_call"]["arguments"]):
        if len(str(v)) < 2:
            continue
        if any(t != str(v) and _shape(t) == _shape(v) for t in tokens):
            return True
    return False


def valid_context_swap(neg: dict, orig: dict) -> bool:
    """Strict check of a Class 1 context-swap negative against its source call: same tool, same argument
    names, exactly one argument changed, every new value copied from the context and shaped like a value it
    replaced."""
    o, n = orig["arguments"], neg["candidate_tool_call"]["arguments"]
    if neg["candidate_tool_call"]["name"] != orig["name"] or set(o) != set(n):
        return False
    changed = [k for k in o if o[k] != n[k]]
    if len(changed) != 1:
        return False
    old_l = _scalar_leaves(o[changed[0]])
    new_l = [x for x in _scalar_leaves(n[changed[0]]) if x not in old_l]
    if not new_l:
        return False
    ctx = _context_text(neg["turns"])
    return all(_in_context(x, ctx) and any(type(x) is type(y) and _shape(x) == _shape(y) for y in old_l)
               for x in new_l)


def run_class1_context_swap(
    examples: list[dict],
    exclude_sources: set[str],
    exclude_keys: set[str],
    rng: random.Random,
    client,
    model: str,
    target: int,
    max_calls: int,
    workers: int,
) -> list[dict]:
    """Generate Class 1 context-swap negatives from tau2-bench positives with the improved prompt.

    Sources that already have a Class 1 negative are skipped (no second negative from the same positive),
    as are sources with no possible swap. Stops once `target` valid rows exist or `max_calls` are made."""
    pool = [e for e in examples if e.get("label") == "correct" and e.get("source_dataset") != "bfcl"
            and e["trajectory_id"] not in exclude_sources and has_swap_candidate(e)]
    rng.shuffle(pool)
    pool = pool[:max_calls]
    print(f"  {len(pool)} tau2-bench positives with a possible swap and no existing Class 1 negative (capped at {max_calls})")

    results, seen, failed = [], set(exclude_keys), {"no_response": 0, "invalid": 0, "duplicate": 0}

    def work(ex):
        return ex, corrupt_llm(ex, 1, client, model, rng, template=_CLASS1_CONTEXT_SWAP_PROMPT)

    with ThreadPoolExecutor(max_workers=workers) as pool_ex:
        futures = [pool_ex.submit(work, ex) for ex in pool]
        for i, fut in enumerate(as_completed(futures), 1):
            if i % 50 == 0:
                print(f"    {i} calls done | {len(results)} valid so far")
            if len(results) >= target:
                for f in futures:
                    f.cancel()
                break
            try:
                ex, neg = fut.result()
            except Exception as e:
                print(f"    [future error] {e}", file=sys.stderr)
                continue
            if neg is None:
                failed["no_response"] += 1
            elif not valid_context_swap(neg, ex["candidate_tool_call"]):
                failed["invalid"] += 1
            else:
                key = _content_key(neg["turns"], neg["candidate_tool_call"])
                if key in seen:
                    failed["duplicate"] += 1
                    continue
                seen.add(key)
                row = finalize_negative(neg, 1, "llm", "::corrupt1v2")
                row["prompt_version"] = "class1_context_swap_v2"
                results.append(row)
    print(f"  Class 1 context-swap: {len(results)} negatives; rejected {failed}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in-pools",
        nargs="+",
        default=["derived/train_pool.jsonl", "derived/bfcl_pool.jsonl"],
    )
    ap.add_argument("--out", default="derived/negatives_pool.jsonl")
    ap.add_argument(
        "--classes",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5, 6, 7],
        help="Error classes to generate (1-7)",
    )
    ap.add_argument(
        "--llm-model",
        default="gpt-5-nano",
        help="OpenAI reasoning model for LLM corruptions (must accept reasoning_effort)",
    )
    ap.add_argument(
        "--max-llm",
        type=int,
        default=1000,
        help="Max total LLM API calls (~$1 at nano pricing)",
    )
    ap.add_argument(
        "--max-per-class",
        type=int,
        default=200,
        help="Max negatives per error class",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Stats only, no LLM calls, no output written",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--api-key", default=None, help="OpenAI API key (env OPENAI_API_KEY)")
    ap.add_argument(
        "--class1-context-swap",
        action="store_true",
        help="Only generate Class 1 context-swap negatives (improved prompt, tau2-bench only); writes --out. "
             "--max-per-class is the target row count and --max-llm the call cap.",
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)
    _HERE = Path(__file__).parent
    classes = set(args.classes)

    # Load all input pools
    all_examples = []
    for pool_path in args.in_pools:
        p = _HERE / pool_path
        if not p.exists():
            print(f"  Pool not found: {p} — skipping")
            continue
        exs = load_pool(p)
        correct = sum(1 for e in exs if e.get("label") == "correct")
        print(f"Loaded {len(exs)} examples from {p.name}  ({correct} correct)")
        all_examples.extend(exs)

    correct_examples = [e for e in all_examples if e.get("label") == "correct"]
    print(f"\nTotal correct examples available: {len(correct_examples)}")

    if args.class1_context_swap:
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            sys.exit("OPENAI_API_KEY not set")
        from openai import OpenAI
        existing = load_pool(_HERE / "derived" / "negatives_pool_final.jsonl")
        used = {e["trajectory_id"].rsplit("::corrupt", 1)[0] for e in existing if e["error_class"] == 1}
        keys = {_content_key(e["turns"], e["candidate_tool_call"]) for e in existing + all_examples}
        print(f"\n=== Class 1 context-swap pass ({len(used)} sources already have a Class 1 negative) ===")
        rows = run_class1_context_swap(all_examples, used, keys, rng, OpenAI(api_key=api_key), args.llm_model,
                                       args.max_per_class, args.max_llm, args.workers)
        out_path = _HERE / args.out
        with open(out_path, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(f"Wrote {len(rows)} rows to {out_path}")
        return

    if args.dry_run:
        print("\n[dry-run] Would generate negatives for classes:", sorted(classes))
        print("[dry-run] Programmatic classes:", sorted(PROGRAMMATIC_CLASSES & classes))
        print("[dry-run] LLM classes:", sorted(LLM_CLASSES & classes))
        # Quick estimate of programmatic yield
        for cls, fn in ((2, corrupt_class2), (4, corrupt_class4), (5, corrupt_class5), (7, corrupt_class7)):
            n = sum(1 for e in correct_examples if fn(e, rng) is not None)
            print(f"Programmatic yield estimate (class {cls}): ~{n}")
        return

    # ── Programmatic corruptions ──────────────────────────────────────────────
    print("\n=== Programmatic corruptions ===")
    prog_negatives = run_programmatic(
        correct_examples, classes & PROGRAMMATIC_CLASSES, rng, args.max_per_class
    )

    # ── LLM corruptions ───────────────────────────────────────────────────────
    llm_negatives = []
    if LLM_CLASSES & classes:
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("\n[WARNING] OPENAI_API_KEY not set — skipping LLM corruptions.")
            print("  Set OPENAI_API_KEY or pass --api-key to enable LLM corruptions.")
        else:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
            except ImportError:
                print("[ERROR] openai package not installed. Run: pip install openai")
                client = None
            if client is not None:
                print("\n=== LLM corruptions ===")
                llm_negatives = run_llm_corruptions(
                    correct_examples,
                    classes & LLM_CLASSES,
                    rng,
                    client,
                    args.llm_model,
                    args.max_llm,
                    args.max_per_class,
                    args.workers,
                )

    # ── Combine and write ─────────────────────────────────────────────────────
    all_negatives = prog_negatives + llm_negatives
    rng.shuffle(all_negatives)

    print(f"\nTotal negatives generated: {len(all_negatives)}")
    from collections import Counter
    class_counts = Counter(e["error_class"] for e in all_negatives)
    for c in sorted(class_counts):
        n = class_counts[c]
        print(f"  Class {c} ({ERROR_CLASSES[c]}): {n}")

    out_path = _HERE / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for ex in all_negatives:
            fh.write(json.dumps(ex) + "\n")
    print(f"\nWrote {len(all_negatives)} negatives to {out_path}")
    print(
        f"Correct pool: {len(correct_examples)}   "
        f"Negatives: {len(all_negatives)}   "
        f"Ratio: {len(all_negatives) / max(len(correct_examples), 1):.2f}:1"
    )


if __name__ == "__main__":
    main()
