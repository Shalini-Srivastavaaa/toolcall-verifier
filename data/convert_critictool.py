"""Convert CriticTool's external-error data to the project's row format (D-019). Training split only.

Source: https://huggingface.co/datasets/chocckaka/CriticTool-Dataset (dataset card: Apache-2.0; paper arXiv 2506.13977,
CC BY 4.0), the four files under external_error/ downloaded to data/raw/critictool/ (gitignored, never executed).

Each record is a tool-use context that ENDS in an environment error (rate limit, connection reset, timeout, daily
quota, ...) plus `ground_truth`: the list of correct next actions. In all 799 records the first ground-truth action
is a retry of the failed call and the second is the next step of the task. The dataset stores no model outputs, so
there are no recorded cases of an agent mishandling the error. Both sides are built from the ground truth:

  correct   candidate = ground_truth[0], the retry of the failed call (label correct)
  Class 6   candidate = ground_truth[1], the next step taken as if the failed call had worked (label incorrect,
            error_class 6). It is the benchmark's own definition of a wrong reaction (anything but a retry), and the
            next step often needs the failed call's output, but it is a constructed negative, not a model's mistake.

Both rows share the same context (paired through `paired_id`). Duplicated contexts across the four files are removed.
Rows are written to data/derived/external_critictool.jsonl and are NOT merged into any training file.

    python data/convert_critictool.py
"""

import ast
import hashlib
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw" / "critictool" / "external_error"
OUT = HERE / "derived" / "external_critictool.jsonl"
FILES = ("base_data_wo_cot", "base_data_w_cot", "evol_data_wo_cot", "evol_data_w_cot")


def literal(x):
    """Parse a python-literal string (safe: literals only); None if it is not one."""
    if not isinstance(x, str):
        return x
    try:
        return ast.literal_eval(x.strip())
    except (ValueError, SyntaxError):
        return None


def schema_text(system: str) -> str:
    """The system prompt holds a python-literal list of tool dicts; render it like the tau2 schema text
    ('- name(p: type, q: type (optional)): description') so the notebook's parse_schema can read it."""
    start = system.index("[")
    tools = literal(system[start: system.rindex("]") + 1]) or []
    lines = []
    for t in tools:
        params = [f"{p['name']}: {p.get('type', 'STRING')}" for p in t.get("required_parameters", [])]
        params += [f"{p['name']}: {p.get('type', 'STRING')} (optional)" for p in t.get("optional_parameters", [])]
        lines.append(f"- {t['name']}({', '.join(params)}): {t.get('description', '')}")
    return "Available tools:\n" + "\n".join(lines)


def turns_of(query_prompt: list) -> list[dict]:
    turns = []
    for m in query_prompt[1:]:
        role, content = m["role"], m["content"]
        if role == "assistant":
            call = literal(content)
            if isinstance(call, dict) and "name" in call:
                turns.append({"role": "assistant_tool_call", "name": call["name"], "arguments": call.get("args", {})})
            else:
                turns.append({"role": "assistant", "content": str(content)})
        elif role == "function":
            turns.append({"role": "tool", "content": str(content).strip()})
        else:
            turns.append({"role": "user", "content": str(content)})
    return turns


def main():
    seen, rows, stats = {}, [], Counter()
    for fname in FILES:
        for key, rec in json.load(open(RAW / f"{fname}.json")).items():
            stats["records"] += 1
            q, gt = rec["query_prompt"], rec["ground_truth"]
            turns = turns_of(q)
            h = hashlib.sha256(json.dumps([turns, gt[:2]], sort_keys=True).encode()).hexdigest()
            if h in seen:
                stats["duplicate_contexts_removed"] += 1
                continue
            seen[h] = key
            if turns[-1]["role"] != "tool" or turns[-2]["role"] != "assistant_tool_call":
                stats["skipped_not_ending_in_a_tool_error"] += 1
                continue
            retry = {"name": gt[0]["name"], "arguments": gt[0].get("args", {})}
            base = {"domain": "critictool", "task_id": key, "source_model": "critictool_ground_truth", "source": "critictool",
                    "tool_schemas_text": schema_text(q[0]["content"]), "turns": turns, "split": "train",
                    "critictool": {"file": fname, "data_type": rec["data_type"], "error_type": rec["error_type"], "error_text": turns[-1]["content"][:300]}}
            tid = f"critictool::{fname}::{key}"
            rows.append({**base, "trajectory_id": tid + "::retry", "candidate_tool_call": retry, "label": "correct", "label_tier": "critictool_gt_retry",
                         "gold_action": retry, "error_class": None, "error_class_name": None, "corruption_method": "external_dataset", "paired_id": tid + "::next"})
            stats["correct_rows"] += 1
            if len(gt) >= 2 and (gt[1]["name"], gt[1].get("args")) != (gt[0]["name"], gt[0].get("args")):
                nxt = {"name": gt[1]["name"], "arguments": gt[1].get("args", {})}
                rows.append({**base, "trajectory_id": tid + "::next", "candidate_tool_call": nxt, "label": "incorrect", "label_tier": "critictool_gt_next_step_after_error",
                             "gold_action": retry, "error_class": 6, "error_class_name": "ignored_error", "corruption_method": "external_dataset", "paired_id": tid + "::retry"})
                stats["class6_rows"] += 1
            else:
                rows.pop()  # no negative to pair with: drop the lone correct row so the two sides stay comparable
                stats["correct_rows"] -= 1
                stats["dropped_no_distinct_next_step"] += 1
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(dict(stats), "->", OUT, f"({len(rows)} rows)")
    print("by data_type:", dict(Counter(r["critictool"]["data_type"] for r in rows if r["label"] == "incorrect")))


if __name__ == "__main__":
    main()
