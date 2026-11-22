"""Independent second labeler for natural TRAIN-split error candidates (D-018 follow-up).

The first labeler was gpt-5-mini (low reasoning, prompt 'localize_v1') or a heuristic. The second is gpt-5-nano at MEDIUM reasoning
with a freshly written prompt, a different class order and a different answer format, and it is never shown the first labeler's answer
(the candidate file is not read into the prompt; only the run itself is). A candidate becomes

    auto_accepted   when both labelers name the same call index AND the same class
    auto_rejected   otherwise (a different call, a different class, "none", or an unparseable answer)

Test-split candidates are never sent here and never auto-accepted: they stay "test_needs_human_review".
Results are cached (data/derived/real_second_opinion_cache.jsonl), cost is estimated first and capped, and every call is logged to
data/derived/api_spend.json.

    .venv/bin/python data/second_opinion.py estimate
    .venv/bin/python data/second_opinion.py run --cap-usd 0.40
    .venv/bin/python data/second_opinion.py apply      # write second_opinion / auto_status into real_candidates_review.jsonl, print agreement
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from induce_errors import HARD_LIMIT_USD, record_spend, spent_total  # noqa: E402
from mine_real_errors import (CLASS_NAMES, POLICY, RESULT_CAP, agent_calls, is_error, reward_summary, runs_by_file)  # noqa: E402

DER = HERE / "derived"
REVIEW = DER / "real_candidates_review.jsonl"
CACHE = DER / "real_second_opinion_cache.jsonl"
MODEL, EFFORT, VERSION = "gpt-5-nano", "medium", "second_v1"
PRICE_IN, PRICE_OUT = 0.05, 0.40  # USD per 1M tokens

SYSTEM = """You audit the tool calls of a customer-service AI agent. You get the company policy, the result of an automatic grader, and the full transcript. Every call the agent made is numbered (C0, C1, ...).

Task: decide whether the agent made a mistake in one of its tool calls and, if so, which call was the EARLIEST mistaken one and what kind of mistake it was. Think step by step through the calls in order, and for each call ask these questions in this order; the first one answered "yes" is the mistake type:

 a. Failure ignored: did an earlier tool message report an error or a failure (including a result that quietly shows the action did not happen), and does this call carry on as though that action had worked?   -> 6
 b. Missing prerequisite: does the policy (or the normal order of steps) require something that has not been done, or has not succeeded, before this call?   -> 7
 c. Duplicate: is this the same side-effecting action that was already carried out successfully before?   -> 4
 d. Wrong tool: is there another available tool that fits what the user asked for much better?   -> 3
 e. Outdated value: does an argument use a value the user gave earlier but later corrected or replaced?   -> 2
 f. Result not used: did an earlier tool result contain the right value (or a warning or changed value), while this call uses a different one or goes ahead without taking it into account?   -> 5
 g. Wrong value: right tool and right goal, but an argument is wrong although the correct value was available in the conversation?   -> 1

Be strict about fairness. The grader's list of expected actions can be arbitrary or noisy; a call that differs from it but is sensible and allowed by the policy is NOT a mistake. If the run failed because the agent skipped a step it never took, because of the user's behaviour, because of the final message, or because of a quirk of the benchmark, there is no mistaken call. Quote the transcript in your reason.

Reply with one JSON object and nothing else:
{"mistaken_call": <number N meaning CN, or null>, "mistake_type": <1-7 or null>, "reason": "<one sentence quoting the transcript>"}"""


def render(s: dict) -> str:
    msgs = s["messages"]
    calls = agent_calls(msgs)
    by_id = {c["id"]: c["k"] for c in calls}
    lines = []
    for m in msgs:
        role = m["role"]
        if role == "tool":
            body = str(m.get("content", ""))
            if len(body) > RESULT_CAP:
                body = body[:RESULT_CAP] + f"...[cut, {len(body)} chars]"
            who = f"C{by_id[m['id']]}" if m.get("requestor") in (None, "assistant") and m.get("id") in by_id else "the user's device"
            lines.append(f"[result of {who}{' - FAILED' if is_error(m) else ''}] {body}")
            continue
        if m.get("content"):
            lines.append(f"[{role}] {m['content']}")
        for tc in m.get("tool_calls") or []:
            lines.append(f"[agent call C{by_id[tc['id']]}] {tc['name']} {json.dumps(tc['arguments'], ensure_ascii=False)}" if role == "assistant"
                         else f"[user's device call] {tc['name']} {json.dumps(tc['arguments'], ensure_ascii=False)}")
    rs = reward_summary(s["reward_info"])
    grader = {k: v for k, v in rs.items() if v not in (None, [], 0) or k in ("reward", "db_match")}
    return (f"POLICY\n{POLICY[s['domain']].read_text()}\n\nGRADER RESULT (reward below 1 means the run failed; expected actions may be noisy)\n"
            f"{json.dumps(grader, ensure_ascii=False)}\n\nTRANSCRIPT\n" + "\n".join(lines) + f"\n\nThe agent made {len(calls)} calls, C0 to C{len(calls) - 1}.")


def parse(text: str, n_calls: int):
    try:
        obj = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
    except (AttributeError, json.JSONDecodeError):
        return None
    k, c = obj.get("mistaken_call"), obj.get("mistake_type")
    if k is None or (isinstance(k, str) and k.strip().lower() in ("none", "null")):
        return {"call": None, "class": None, "reason": str(obj.get("reason", ""))}
    if isinstance(k, str) and re.fullmatch(r"[Cc]?\d+", k.strip()):
        k = int(k.strip().lstrip("Cc"))
    if not (isinstance(k, int) and 0 <= k < n_calls) or c not in range(1, 8):
        return None
    return {"call": k, "class": int(c), "reason": str(obj.get("reason", ""))}


def candidates() -> list[dict]:
    rows = [json.loads(l) for l in open(REVIEW) if l.strip()]
    return [r for r in rows if r["split"] == "train" and r.get("origin", "natural") == "natural"]


def cache_rows() -> dict:
    out = {}
    if CACHE.exists():
        for l in open(CACHE):
            r = json.loads(l)
            out[(r["run_key"], r["version"])] = r
    return out


def todo_runs() -> list[str]:
    done = cache_rows()
    return sorted({c["run_key"] for c in candidates() if (c["run_key"], VERSION) not in done})


def estimate(quiet=False) -> float:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    keys = set(todo_runs())
    tin, n = 0, 0
    sys_tok = len(tok(SYSTEM)["input_ids"])
    for s in runs_by_file(keys):
        tin += sys_tok + len(tok(render(s))["input_ids"])
        n += 1
    out_tok = n * 3500  # medium reasoning on gpt-5-nano: generous; the first labeler (low) averaged about 900
    cost = tin * PRICE_IN / 1e6 + out_tok * PRICE_OUT / 1e6
    if not quiet:
        print(f"{n} runs to label, {tin:,} input tokens, assumed {out_tok:,} output tokens -> ESTIMATED COST ${cost:.3f}")
    return cost


def run(cap: float):
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(HERE.parent / ".env")
    assert os.environ.get("OPENAI_API_KEY")  # never printed
    est = estimate()
    if est > cap:
        sys.exit(f"STOP: estimate ${est:.2f} is above the cap ${cap:.2f}; nothing was run.")
    if spent_total() + est > HARD_LIMIT_USD:
        sys.exit("STOP: the hard limit would be exceeded.")
    client = OpenAI()
    keys = set(todo_runs())
    sims = list(runs_by_file(keys))
    spent = 0.0

    def one(s):
        key = f"{s['file']}::{s['sim_id']}"
        n_calls = len(agent_calls(s["messages"]))
        for attempt in range(3):
            try:
                r = client.chat.completions.create(model=MODEL, reasoning_effort=EFFORT, response_format={"type": "json_object"},
                                                   messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": render(s)}])
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    return {"run_key": key, "version": VERSION, "error": str(e)[:200], "cost_usd": 0.0}
                time.sleep(3 * (attempt + 1))
        u = r.usage
        cost = u.prompt_tokens * PRICE_IN / 1e6 + u.completion_tokens * PRICE_OUT / 1e6
        text = r.choices[0].message.content or ""
        return {"run_key": key, "version": VERSION, "model": MODEL, "effort": EFFORT, "parsed": parse(text, n_calls), "raw": text[:600],
                "in": u.prompt_tokens, "out": u.completion_tokens, "cost_usd": round(cost, 5)}

    with ThreadPoolExecutor(8) as ex, open(CACHE, "a") as f:
        for i, res in enumerate(ex.map(one, sims), 1):
            spent += res["cost_usd"]
            if "error" not in res:
                f.write(json.dumps(res) + "\n")
                f.flush()
            if spent > cap:
                print(f"cap ${cap} reached after {i} runs")
                break
    total = record_spend("5_second_opinion", spent, len(sims), f"{MODEL} {EFFORT}")
    print(f"done: {len(sims)} runs, ${spent:.3f} (all-prompt total ${total:.2f})")


def apply():
    cache = cache_rows()
    rows = [json.loads(l) for l in open(REVIEW) if l.strip()]
    agree = defaultdict(Counter)
    for r in rows:
        r.setdefault("origin", "natural")
        if r["origin"] != "natural":
            continue
        if r["split"] == "test":
            r["auto_status"] = "test_needs_human_review"  # never auto-accepted
            continue
        c = cache.get((r["run_key"], VERSION))
        if c is None:
            r["auto_status"] = "second_opinion_pending"
            continue
        p = c.get("parsed")
        r["second_opinion"] = {"model": MODEL, "effort": EFFORT, "version": VERSION, "call": p and p["call"], "class": p and p["class"], "reason": p and p["reason"]}
        same_call = bool(p) and p["call"] == r["flagged_call"]["call_index"]
        same_class = bool(p) and p["class"] == r["proposed_class"]
        r["auto_status"] = "auto_accepted" if (same_call and same_class) else "auto_rejected"
        a = agree[r["proposed_class"]]
        a["n"] += 1
        a["same_call"] += same_call
        a["same_call_and_class"] += same_call and same_class
        a["second_says_none"] += bool(p) and p["call"] is None
        a["unparsed"] += p is None
    with open(REVIEW, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("agreement by the first labeler's class:")
    tot = Counter()
    for k in sorted(agree):
        a = agree[k]
        tot.update(a)
        print(f"  class {k} {CLASS_NAMES[k]:22s} n={a['n']:3d}  same call {a['same_call']:3d}  same call+class {a['same_call_and_class']:3d} ({a['same_call_and_class'] / a['n']:.0%})  second says none {a['second_says_none']}  unparsed {a['unparsed']}")
    if tot["n"]:
        print(f"  all: n={tot['n']}  auto_accepted {tot['same_call_and_class']} ({tot['same_call_and_class'] / tot['n']:.0%}), auto_rejected {tot['n'] - tot['same_call_and_class']}")
    print("auto_status counts (natural):", dict(Counter(r["auto_status"] for r in rows if r.get("origin") == "natural")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["estimate", "run", "apply"])
    ap.add_argument("--cap-usd", type=float, default=0.40)
    a = ap.parse_args()
    {"estimate": lambda: estimate(), "run": lambda: run(a.cap_usd), "apply": apply}[a.stage]()
