"""Phase 2b: Extract multi-turn examples from BFCL v4.

BFCL's multi-turn categories provide a different signal than tau2-bench:
rather than full agent trajectories with tool results, each test case is
a multi-turn *task specification* with:
  - question: list of per-turn user messages
  - ground_truth: list of per-turn correct call sequences (as Python-ish strings)
  - initial_config: simulated environment state (GorillaFileSystem, TwitterAPI, etc.)
  - involved_classes + path: which tools are relevant across the task

Categories extracted:
  - multi_turn_base (200): standard multi-turn tasks
  - multi_turn_long_context (200): same tasks with more context
  - multi_turn_miss_func (200): tasks where one turn has NO correct function to call
  - multi_turn_miss_param (200): tasks where one turn's ground truth is empty (missing param)

Extraction approach (positives only, see docs/decision_log.md D-008):
  For each (task, turn_idx) pair with a non-empty ground truth, we produce
  one example:
  - Build the conversation context up to that turn (prior turns + their gold calls)
  - The candidate call is the FIRST gold action at that turn, labeled "correct"
  Turns with an empty ground_truth (miss_func/miss_param) produce no example.
  No negatives are generated here: all incorrect examples come from the
  Phase 3 synthetic corruptions.

Output format matches train_pool.jsonl with two additions:
  "source_dataset": "bfcl",
  "bfcl_category": "base|long_context|miss_func|miss_param"

Note: BFCL ground_truth calls are formatted as Python-ish strings
  (e.g. "mv(source='x', destination='y')"). We parse them into
  {"name": ..., "arguments": {...}} format to match tau2-bench.
"""

import argparse
import json
import re
import random
import hashlib
from pathlib import Path

BFCL_DIR = (
    Path(__file__).parent
    / "raw"
    / "bfcl-gorilla"
    / "berkeley-function-call-leaderboard"
    / "bfcl_eval"
    / "data"
)

MULTI_TURN_FILES = {
    "base": "BFCL_v4_multi_turn_base.json",
    "long_context": "BFCL_v4_multi_turn_long_context.json",
    "miss_func": "BFCL_v4_multi_turn_miss_func.json",
    "miss_param": "BFCL_v4_multi_turn_miss_param.json",
}

FUNC_DOC_DIR = BFCL_DIR / "multi_turn_func_doc"


# ── call string parser ────────────────────────────────────────────────────────

def parse_call_string(call_str: str) -> dict | None:
    """Parse 'func_name(arg1=val1, arg2=val2)' into {name, arguments} dict.

    Returns None if the string can't be parsed reliably.
    BFCL ground-truth strings are Python-ish: string values use single quotes,
    boolean/int/None are Python literals. We use a regex approach + eval with
    ast.literal_eval for safety.
    """
    import ast

    call_str = call_str.strip()
    # Extract function name
    m = re.match(r'^(\w+)\((.*)\)$', call_str, re.DOTALL)
    if not m:
        return None
    func_name = m.group(1)
    args_str = m.group(2).strip()

    if not args_str:
        return {"name": func_name, "arguments": {}}

    # Try to parse the args string via AST.
    # BFCL uses keyword arguments almost exclusively, but some calls like
    # sort('filename') use positional args. Handle both.
    try:
        parsed = ast.parse(f"_f({args_str})", mode='eval')
        call_node = parsed.body
        if not isinstance(call_node, ast.Call):
            return None

        result = {}

        # Positional args → store as "__arg0", "__arg1", ...
        for i, arg_node in enumerate(call_node.args):
            try:
                val = ast.literal_eval(arg_node)
            except (ValueError, TypeError):
                val = ast.unparse(arg_node)
            result[f"__arg{i}"] = val

        # Keyword args
        for keyword in call_node.keywords:
            key = keyword.arg
            try:
                val = ast.literal_eval(keyword.value)
            except (ValueError, TypeError):
                val = ast.unparse(keyword.value)
            result[key] = val

        return {"name": func_name, "arguments": result}
    except (SyntaxError, AttributeError):
        return None


def format_call_for_turns(call: dict) -> dict:
    """Format a parsed call as an assistant_tool_call turn."""
    return {
        "role": "assistant_tool_call",
        "name": call["name"],
        "arguments": call["arguments"],
    }


# ── schema loading ────────────────────────────────────────────────────────────

def load_func_docs(class_names: list[str]) -> str:
    """Load and format tool schemas for the given BFCL API class names."""
    # Map class names to doc files
    class_to_file = {
        "GorillaFileSystem": "gorilla_file_system.json",
        "TwitterAPI": "posting_api.json",  # posting_api has tweet-like tools
        "MathAPI": "math_api.json",
        "MemoryKV": "memory_kv.json",
        "MemoryRecSum": "memory_rec_sum.json",
        "MemoryVector": "memory_vector.json",
        "MessageAPI": "message_api.json",
        "PostAPI": "posting_api.json",
        "TicketAPI": "ticket_api.json",
        "TradingBot": "trading_bot.json",
        "TravelBooking": "travel_booking.json",
        "VehicleControl": "vehicle_control.json",
        "WebSearch": "web_search.json",
    }

    schemas = []
    seen_files = set()
    for cls in class_names:
        fname = class_to_file.get(cls)
        if fname and fname not in seen_files:
            fpath = FUNC_DOC_DIR / fname
            if fpath.exists():
                seen_files.add(fname)
                try:
                    with open(fpath) as f:
                        tools = [json.loads(line) for line in f if line.strip()]
                    for t in tools:
                        schemas.append(t)
                except (json.JSONDecodeError, OSError):
                    pass

    if not schemas:
        return "(no schema available)"

    lines = []
    for t in schemas:
        params = t.get("parameters", {}).get("properties", {})
        required = t.get("parameters", {}).get("required", [])
        param_strs = []
        for pname, pinfo in params.items():
            req = " [required]" if pname in required else ""
            ptype = pinfo.get("type", "any")
            param_strs.append(f"    {pname} ({ptype}){req}: {pinfo.get('description','')}")
        lines.append(f"Function: {t['name']}")
        lines.append(f"  Description: {t.get('description','')}")
        if param_strs:
            lines.append("  Parameters:")
            lines.extend(param_strs)
        lines.append("")
    return "\n".join(lines)


# ── turn rendering ────────────────────────────────────────────────────────────

def render_bfcl_context(
    question: list[list[dict]],
    ground_truth: list[list[str]],
    up_to_turn: int,
) -> list[dict]:
    """Build the conversation context for all turns BEFORE up_to_turn.

    Each turn consists of:
      1. The user message(s) for that turn
      2. The assistant's gold tool calls for that turn (simulated as if they happened)
    We include the tool call but NOT a simulated tool result (BFCL doesn't supply them).
    """
    turns = []
    for t in range(up_to_turn):
        # User messages for this turn
        for msg in question[t]:
            if msg.get("content"):
                turns.append({"role": msg["role"], "content": msg["content"]})
        # Gold calls for this turn (if any)
        for call_str in (ground_truth[t] or []):
            parsed = parse_call_string(call_str)
            if parsed:
                turns.append(format_call_for_turns(parsed))
    return turns


# ── content dedup ─────────────────────────────────────────────────────────────

def _content_key(turns: list[dict], candidate: dict) -> str:
    payload = json.dumps(turns, sort_keys=True) + json.dumps(candidate, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


# ── extraction ────────────────────────────────────────────────────────────────

def extract_from_category(
    category: str,
    question_file: Path,
    answer_file: Path,
) -> list[dict]:
    """Extract (context, gold candidate_call) positives from one BFCL category."""

    # Load questions and answers indexed by id
    questions: dict[str, dict] = {}
    with open(question_file) as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                questions[obj["id"]] = obj

    answers: dict[str, dict] = {}
    with open(answer_file) as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                answers[obj["id"]] = obj

    examples = []

    for qid, q in questions.items():
        gt_entry = answers.get(qid)
        if gt_entry is None:
            continue

        question_turns: list[list[dict]] = q["question"]  # list[list[{role, content}]]
        ground_truth: list[list[str]] = gt_entry["ground_truth"]  # list[list[str]]
        involved_classes: list[str] = q.get("involved_classes", [])
        schema_text = load_func_docs(involved_classes)

        # Ensure lengths match
        n_turns = min(len(question_turns), len(ground_truth))

        for turn_idx in range(n_turns):
            user_msgs = question_turns[turn_idx]
            gold_calls_strs = ground_truth[turn_idx] or []

            # Context = everything before this turn
            context_turns = render_bfcl_context(question_turns, ground_truth, turn_idx)

            # Add this turn's user message(s) to context
            for msg in user_msgs:
                if msg.get("content"):
                    context_turns.append({"role": msg["role"], "content": msg["content"]})

            # --- Positive examples: first gold call at this turn ---
            if gold_calls_strs:
                first_gold = parse_call_string(gold_calls_strs[0])
                if first_gold:
                    examples.append({
                        "trajectory_id": f"bfcl_{category}::{qid}::t{turn_idx}",
                        "domain": "bfcl",
                        "task_id": qid,
                        "source_model": "gold",
                        "tool_schemas_text": schema_text,
                        "turns": context_turns,
                        "candidate_tool_call": first_gold,
                        "label": "correct",
                        "label_tier": "bfcl_gold",
                        "gold_action": first_gold,
                        "split": "train",
                        "source_dataset": "bfcl",
                        "bfcl_category": category,
                    })

    return examples


# ── dedup ─────────────────────────────────────────────────────────────────────

def dedupe(examples: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out = []
    for ex in examples:
        key = _content_key(ex["turns"], ex["candidate_tool_call"])
        if key in seen:
            continue
        seen.add(key)
        out.append(ex)
    return out


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="derived/bfcl_pool.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--categories", nargs="+",
                    default=["base", "long_context", "miss_func", "miss_param"],
                    help="Which BFCL multi-turn categories to include")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    _HERE = Path(__file__).parent

    all_examples = []

    for category in args.categories:
        fname = MULTI_TURN_FILES.get(category)
        if not fname:
            print(f"  Unknown category {category}, skipping")
            continue
        question_file = BFCL_DIR / fname
        answer_file = BFCL_DIR / "possible_answer" / fname
        if not question_file.exists():
            print(f"  Question file not found: {question_file}")
            continue
        if not answer_file.exists():
            print(f"  Answer file not found: {answer_file}")
            continue
        print(f"Processing {category} ...", end=" ", flush=True)
        exs = extract_from_category(category, question_file, answer_file)
        print(f"{len(exs)} examples")
        all_examples.extend(exs)

    print(f"\nTotal before dedup: {len(all_examples)}")
    all_examples = dedupe(all_examples)
    print(f"Total after dedup:  {len(all_examples)}")

    print(f"\nAll {len(all_examples)} examples are gold positives (label=correct); "
          "negatives come from Phase 3 synthetic corruption.")

    from collections import Counter
    tier_counts = Counter(e["label_tier"] for e in all_examples)
    print("\nLabel tier breakdown:")
    for t, c in sorted(tier_counts.items()):
        print(f"  {t}: {c}")

    rng.shuffle(all_examples)

    out_path = _HERE / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for ex in all_examples:
            fh.write(json.dumps(ex) + "\n")
    print(f"\nWrote {len(all_examples)} examples to {out_path}")


if __name__ == "__main__":
    main()
