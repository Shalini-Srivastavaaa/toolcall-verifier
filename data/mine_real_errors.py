"""Stage 1 of mining real agent errors from tau2-bench's failed runs (D-018). Review file only.

Nothing here creates a training or test file: it produces a candidate list for hand review. Stages:

    split     lock the task-level test/train split (data/derived/real_split.json; refuses to overwrite)
    pick      failed runs -> dedup -> cap 3 per (task, agent model) -> heuristics -> data/derived/real_run_index.jsonl
    estimate  build the localization prompts for the first N runs per domain and print the cost estimate
    localize  ask gpt-5-mini (low reasoning) to name the first incorrect tool call; every response is cached
    build     real_candidates_review.jsonl + real_positive_candidates.jsonl (pending) + a counts report

    python data/mine_real_errors.py split|pick|estimate|localize|build [--per-domain 150] [--max-usd 4]

The LLM is an offline labeling aid for humans to check. It is never part of the model or its evaluation.
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from build_train_pool import render_turns as render_turns_d007  # same render the D-007 dedup hash used
from tool_schemas import FREE_TEXT_TOOLS

RESULTS = HERE / "raw" / "tau2-bench" / "data" / "tau2" / "results" / "final"
DOMAINS_DIR = HERE / "raw" / "tau2-bench" / "data" / "tau2" / "domains"
POLICY = {"airline": DOMAINS_DIR / "airline" / "policy.md", "retail": DOMAINS_DIR / "retail" / "policy.md",
          "telecom": DOMAINS_DIR / "telecom" / "main_policy.md"}
DER = HERE / "derived"
SPLIT_PATH = DER / "real_split.json"
INDEX_PATH = DER / "real_run_index.jsonl"
SUMMARY_PATH = DER / "real_pick_summary.json"
PLAN_PATH = DER / "real_localize_plan.json"
CACHE_PATH = DER / "real_localization_cache.jsonl"
REVIEW_PATH = DER / "real_candidates_review.jsonl"
POSITIVES_PATH = DER / "real_positive_candidates.jsonl"
REPORT_PATH = DER / "real_candidates_report.json"
TRAIN_MORE_PATH = DER / "real_train_more_keys.json"  # runs localized by `localize-train` (D-028): train-split failed runs not localized before
MORE_PATH = DER / "real_class3_more_keys.json"  # runs localized by `localize-more` (D-019 update): `build` keeps only their Class 3 proposals

SEED = 20261111
TEST_FRACTION = 0.30
RUNS_PER_TASK_MODEL = 3
MODEL = "gpt-5-mini"
REASONING_EFFORT = "low"
PRICE_IN, PRICE_OUT = 0.25, 2.00  # USD per 1M tokens, eval/pricing.json (gpt-5-mini)
PROMPT_VERSION = "localize_v1"
RESULT_CAP = 3000  # chars of one tool result shown to the labeling model and in the review window
CLASS_NAMES = {1: "wrong_arg_value", 2: "stale_state", 3: "wrong_tool_selection", 4: "repeated_call",
               5: "failure_to_propagate", 6: "ignored_error", 7: "missing_precondition"}


# ───────────────────────── reading the results files ─────────────────────────

def domain_of(stem: str) -> str:
    return stem.split("_")[1].split("-")[0]  # "telecom-workflow" is still the telecom domain


def result_files() -> list[Path]:
    return sorted(RESULTS.glob("*.json"))


def iter_sims(files=None):
    """Yield every simulation with the needed fields (one file in memory at a time)."""
    for f in files or result_files():
        d = json.load(open(f))
        for s in d["simulations"]:
            yield {"file": f.stem, "domain": domain_of(f.stem), "source_model": f.stem.split("_")[0],
                   "task_id": str(s["task_id"]), "sim_id": s["id"], "trial": s.get("trial"),
                   "reward_info": s.get("reward_info") or {}, "messages": s["messages"],
                   "termination": s.get("termination_reason")}


def run_key(s: dict) -> str:
    return f"{s['file']}::{s['sim_id']}"  # the trajectory_id format used everywhere else


def is_failed(s: dict) -> bool:
    r = s["reward_info"].get("reward")
    return r is None or r < 1


def is_error(m: dict) -> bool:
    return m.get("role") == "tool" and (m.get("error") in (True, "True") or str(m.get("content", "")).lstrip().lower().startswith("error"))


def agent_calls(messages: list[dict]) -> list[dict]:
    """The agent's tool calls in order, numbered C0, C1, ... (user-side device calls in telecom are not numbered)."""
    out = []
    for mi, m in enumerate(messages):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                out.append({"k": len(out), "msg": mi, "id": tc.get("id"), "name": tc["name"], "arguments": tc["arguments"]})
    return out


def render_full(messages: list[dict]) -> tuple[list[dict], dict]:
    """Compact turns like the training schema, but keeping the assistant text beside a call and the user's
    own device calls. Returns (turns, {call index k: turn index})."""
    turns, call_turn, k = [], {}, 0
    for m in messages:
        role = m["role"]
        if role == "tool":
            t = {"role": "tool", "content": str(m.get("content", "")), "requestor": m.get("requestor")}
            if is_error(m):
                t["error"] = True
            turns.append(t)
        elif m.get("tool_calls"):
            if m.get("content"):
                turns.append({"role": role, "content": m["content"]})
            for tc in m["tool_calls"]:
                if role == "assistant":
                    call_turn[k] = len(turns)
                    k += 1
                turns.append({"role": "assistant_tool_call" if role == "assistant" else "user_tool_call",
                              "name": tc["name"], "arguments": tc["arguments"]})
        elif m.get("content"):
            turns.append({"role": role, "content": m["content"]})
    return turns, call_turn


def content_hash(messages: list[dict]) -> str:
    """D-007 method: sha256 of the sorted-key JSON of the rendered turns (+ the final call)."""
    turns = render_turns_d007(messages)
    last = agent_calls(messages)
    cand = {"name": last[-1]["name"], "arguments": last[-1]["arguments"]} if last else None
    return hashlib.sha256((json.dumps(turns, sort_keys=True) + json.dumps(cand, sort_keys=True)).encode()).hexdigest()


# ───────────────────────── reward_info summary ─────────────────────────

def reward_summary(ri: dict) -> dict:
    ac = ri.get("action_checks") or []
    return {
        "reward": ri.get("reward"), "reward_basis": ri.get("reward_basis"),
        "db_match": (ri.get("db_check") or {}).get("db_match"),
        "missed_actions": [{"name": c["action"]["name"], "arguments": c["action"]["arguments"]} for c in ac if not c.get("action_match")],
        "n_action_checks": len(ac),
        "nl_unmet": [a["nl_assertion"] for a in ri.get("nl_assertions") or [] if not a.get("met")],
        "comm_unmet": [a["info"] for a in ri.get("communicate_checks") or [] if not a.get("met")],
    }


# ───────────────────────── heuristics (review order only) ─────────────────────────

ACK = re.compile(r"error|sorry|unable|unfortunately|apolog|cannot|can't|couldn't|failed|not able|issue|problem|invalid|not found|doesn't|does not|not allowed|unsuccessful|insufficient|not enough|didn.t|did not|wasn.t|unavailable|declin|reject|denied|fail|unable|no longer|invalid|expired|restrict", re.I)
READ_ONLY = re.compile(r"^(get_|list_|search_|find_|check_|calculate|think|run_speed|can_)")
CORRECTION = re.compile(r"\b(actually|instead|rather|i meant|scratch that|never ?mind|on second thought|make that|change (it|that|them|this)|wait,)", re.I)
TOKEN = re.compile(r"[#A-Za-z]*\d[\w\-#:.]*")
REVERT = re.compile(r"\b(back to|change (it|that|them|this) back|revert|previous|original|what it was|as before)\b", re.I)  # asking to undo a change is not a stale value


def _tokens(text: str) -> set[str]:
    return {t.strip(".,:;)") for t in TOKEN.findall(text) if len(t.strip(".,:;)")) >= 3}


def _shape(t: str) -> str:
    return re.sub(r"\d", "9", re.sub(r"[A-Za-z]", "a", t))


def heuristic_class6(messages: list[dict], calls: list[dict]):
    """A tool returned an error and the agent's next move treats it as if it had worked.
    Returns (reason, index of the agent call it flags) or None."""
    by_id = {c["id"]: c for c in calls}
    for i, m in enumerate(messages):
        if not (is_error(m) and m.get("requestor") in (None, "assistant")):
            continue
        err_call = by_id.get(m.get("id"))
        for n in messages[i + 1:]:
            if n.get("role") != "assistant":
                continue
            if n.get("tool_calls"):
                for tc in n["tool_calls"]:
                    if err_call and tc["name"] != err_call["name"] and not READ_ONLY.match(tc["name"]) and tc["name"] not in FREE_TEXT_TOOLS:
                        return f"after the error from {err_call['name']} the agent called {tc['name']} (a different write tool)", by_id[tc["id"]]["k"]
            elif n.get("content") and not ACK.search(n["content"]):
                nxt = next((c for c in calls if c["msg"] > messages.index(n)), None)  # the agent call that follows the unacknowledged error
                if nxt is not None and not READ_ONLY.match(nxt["name"]) and nxt["name"] not in FREE_TEXT_TOOLS:
                    return f"after the error from {err_call['name'] if err_call else 'a tool'} the next agent message does not mention it and the agent calls {nxt['name']}", nxt["k"]
            break
    return None


def heuristic_class2(messages: list[dict], calls: list[dict]):
    """The user changed a value and a later call still uses the old one. Returns (reason, call index) or None."""
    users = [(i, m["content"]) for i, m in enumerate(messages) if m.get("role") == "user" and m.get("content")]
    for ui, (mi, text) in enumerate(users):
        if ui == 0 or not CORRECTION.search(text) or REVERT.search(text):
            continue
        earlier = set().union(*[_tokens(t) for _, t in users[:ui]])
        new = _tokens(text) - earlier
        old = {o for o in earlier - _tokens(text) if any(_shape(o) == _shape(n) for n in new)}
        if not new or not old:
            continue
        for c in calls:
            if c["msg"] > mi:
                args = json.dumps(c["arguments"])
                hit = [o for o in old if o in args]
                if hit and not any(n in args for n in new):
                    return f"after the user's correction (message {mi}) call C{c['k']} {c['name']} still uses {hit[0]} instead of {sorted(new)[0]}", c["k"]
    return None


# ───────────────────────── stage: split ─────────────────────────

def test_real_tasks() -> set[tuple[str, str]]:
    return {(r["domain"], str(r["task_id"])) for r in map(json.loads, open(DER / "test_real.jsonl"))}


def current_training_tasks() -> Counter:
    tasks = Counter()
    for fn in ("train_pool", "positives_class2_controls", "positives_class4_controls", "positives_class6_controls", "negatives_pool_final"):
        for line in open(DER / f"{fn}.jsonl"):
            r = json.loads(line)
            if r["domain"] != "bfcl":
                tasks[(r["domain"], str(r["task_id"]))] += 1
    return tasks


def cmd_split(args):
    if SPLIT_PATH.exists():
        sys.exit(f"{SPLIT_PATH} already exists and must never be regenerated (it is locked).")
    all_tasks = defaultdict(set)
    for f in result_files():
        d = json.load(open(f))
        for s in d["simulations"]:
            all_tasks[domain_of(f.stem)].add(str(s["task_id"]))
    in_test_real = test_real_tasks()
    training = current_training_tasks()
    split = {"seed": SEED, "created": time.strftime("%Y-%m-%d"), "test_fraction_of_remaining": TEST_FRACTION,
             "rule": "Every (domain, task_id) in test_real.jsonl is test. The remaining tasks of each domain are sorted, shuffled with "
                     "random.Random(f'{seed}-{domain}') and the first round(0.3*n) go to test, the rest to train. Never regenerate.",
             "tasks": {}, "overlap_with_current_training": {}}
    for dom in ("retail", "airline", "telecom"):
        tasks = sorted(all_tasks[dom])
        fixed_test = [t for t in tasks if (dom, t) in in_test_real]
        rest = [t for t in tasks if (dom, t) not in in_test_real]
        random.Random(f"{SEED}-{dom}").shuffle(rest)
        n_test = round(TEST_FRACTION * len(rest))
        new_test, train = sorted(rest[:n_test]), sorted(rest[n_test:])
        split["tasks"][dom] = {"test": sorted(fixed_test + new_test), "train": train, "test_real": fixed_test, "new_test": new_test}
        clash = [t for t in new_test if (dom, t) in training]
        split["overlap_with_current_training"][dom] = {
            "new_test_tasks_that_appear_in_current_training_files": sorted(clash),
            "training_rows_on_those_tasks": sum(training[(dom, t)] for t in clash)}
        print(f"{dom}: {len(tasks)} tasks | test_real {len(fixed_test)} | new test {len(new_test)} | train {len(train)} | "
              f"new-test tasks already in training files: {len(clash)} ({split['overlap_with_current_training'][dom]['training_rows_on_those_tasks']} rows)")
    SPLIT_PATH.write_text(json.dumps(split, indent=1))
    print("wrote", SPLIT_PATH)


def split_of(dom: str, task: str, split: dict) -> str:
    return "test" if task in set(split["tasks"][dom]["test"]) else "train"


# ───────────────────────── stage: pick ─────────────────────────

def cmd_pick(args):
    split = json.load(open(SPLIT_PATH))
    in_test_real = test_real_tasks()
    stats = Counter()
    ri_stats = Counter()
    kept, seen = [], {}
    for s in iter_sims():
        stats["sims"] += 1
        if not is_failed(s):
            continue
        stats["failed"] += 1
        ri = s["reward_info"]
        basis = ri.get("reward_basis") or []
        for b in basis:
            ri_stats[f"basis_{b}"] += 1
        summ = reward_summary(ri)
        ri_stats["db_mismatch"] += summ["db_match"] is False
        ri_stats["some_action_missed"] += bool(summ["missed_actions"])
        ri_stats["all_actions_matched_and_db_ok"] += (not summ["missed_actions"]) and summ["db_match"] is not False
        ri_stats["nl_unmet"] += bool(summ["nl_unmet"])
        ri_stats["comm_unmet"] += bool(summ["comm_unmet"])
        ri_stats["no_action_checks"] += summ["n_action_checks"] == 0
        stats[f"failed_{s['domain']}"] += 1
        if (s["domain"], s["task_id"]) in in_test_real:
            stats["excluded_task_in_test_real"] += 1
            continue
        h = content_hash(s["messages"])
        if h in seen:
            stats["content_duplicates_removed"] += 1
            continue
        calls = agent_calls(s["messages"])
        seen[h] = True
        h6, h2 = heuristic_class6(s["messages"], calls), heuristic_class2(s["messages"], calls)
        h6, h2 = h6 or (None, None), h2 or (None, None)
        kept.append({"run_key": run_key(s), "domain": s["domain"], "task_id": s["task_id"], "source_model": s["source_model"],
                     "file": s["file"], "trial": s["trial"], "split": split_of(s["domain"], s["task_id"], split),
                     "n_messages": len(s["messages"]), "n_agent_calls": len(calls), "termination": s["termination"],
                     "has_tool_error": any(is_error(m) for m in s["messages"]), "hash": h,
                     "heuristic_class6": h6[0], "heuristic_class6_call": h6[1], "heuristic_class2": h2[0], "heuristic_class2_call": h2[1],
                     "reward_summary": summ})
    stats["after_dedup"] = len(kept)
    groups = defaultdict(list)
    for r in kept:
        groups[(r["domain"], r["task_id"], r["source_model"])].append(r)
    capped = []
    for key, rs in sorted(groups.items()):
        rs = sorted(rs, key=lambda r: r["run_key"])
        if len(rs) > RUNS_PER_TASK_MODEL:
            stats["dropped_by_cap"] += len(rs) - RUNS_PER_TASK_MODEL
            rs = random.Random(f"{SEED}-{key}").sample(rs, RUNS_PER_TASK_MODEL)
        capped += rs
    capped.sort(key=lambda r: r["run_key"])
    stats["selected"] = len(capped)
    INDEX_PATH.write_text("".join(json.dumps(r) + "\n" for r in capped))
    summary = {"counts": dict(stats), "reward_info": dict(ri_stats),
               "selected_by_domain": dict(Counter(r["domain"] for r in capped)),
               "selected_by_split": dict(Counter(r["split"] for r in capped)),
               "selected_failed_tasks_by_domain": {d: len({r["task_id"] for r in capped if r["domain"] == d}) for d in ("retail", "airline", "telecom")},
               "selected_with_tool_error": sum(r["has_tool_error"] for r in capped),
               "selected_flagged_class6": sum(bool(r["heuristic_class6"]) for r in capped),
               "selected_flagged_class2": sum(bool(r["heuristic_class2"]) for r in capped)}
    SUMMARY_PATH.write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


# ───────────────────────── stage: estimate / localize ─────────────────────────

SYSTEM = """You are an expert reviewer of AI customer-service agents. You are shown the policy an agent must follow, the outcome of an automatic checker, and a full conversation in which the agent made numbered tool calls (C0, C1, ...). Your job is to find the FIRST agent tool call that was incorrect, if any, and classify the mistake.

A tool call is incorrect if it is the first agent call where one of these holds (use this priority when several apply: 6, 7, 4, 3, 2, 5, 1):
 6 ignored_error: an earlier tool result was an error or failure and this call proceeds as if that operation had succeeded.
 7 missing_precondition: a step the policy or workflow requires before this call has not happened (or has not succeeded).
 4 repeated_call: the same action was already performed successfully earlier (a duplicate side effect).
 3 wrong_tool_selection: a different available tool matches what the user asked for.
 2 stale_state: an argument uses a value the user had earlier given but then corrected or overridden.
 5 failure_to_propagate: the right value or signal was in an earlier tool result and the call uses something else, or acts without checking it.
 1 wrong_arg_value: the tool and intent are right but an argument value is wrong, although the right value was available in the conversation.

Rules: judge each call against the conversation, the tool results and the policy. The checker's expected actions are often noisy or arbitrary (dates, card numbers or alternative valid paths), so treat them as hints, not truth. A call that is merely different from the expected action, but reasonable and permitted by the policy, is NOT incorrect. If the run failed because the agent never made a call it should have, or because of the user's behaviour, the final message, or a benchmark artifact, answer "none". Quote the conversation in your evidence.

Answer with JSON only: {"first_incorrect_call": <integer N for call CN, or "none">, "class": <1-7, or null if none>, "evidence": "<one sentence that quotes the conversation or a tool result>"}"""


def render_prompt(s: dict) -> str:
    msgs = s["messages"]
    calls = agent_calls(msgs)
    by_id = {c["id"]: c["k"] for c in calls}
    lines = []
    for m in msgs:
        role = m["role"]
        if role == "tool":
            body = str(m.get("content", ""))
            if len(body) > RESULT_CAP:
                body = body[:RESULT_CAP] + f"…[truncated, {len(body)} chars]"
            who = f"C{by_id[m['id']]}" if m.get("requestor") in (None, "assistant") and m.get("id") in by_id else "user device"
            lines.append(f"TOOL_RESULT for {who}{' [ERROR]' if is_error(m) else ''}: {body}")
            continue
        if m.get("content"):
            lines.append(f"{role.upper()}: {m['content']}")
        for tc in m.get("tool_calls") or []:
            if role == "assistant":
                lines.append(f"AGENT_CALL C{by_id[tc['id']]}: {tc['name']}({json.dumps(tc['arguments'], ensure_ascii=False)})")
            else:
                lines.append(f"USER_DEVICE_CALL: {tc['name']}({json.dumps(tc['arguments'], ensure_ascii=False)})")
    rs = reward_summary(s["reward_info"])
    outcome = {k: v for k, v in rs.items() if v not in (None, [], 0) or k in ("reward", "db_match")}
    policy = POLICY[s["domain"]].read_text()
    return (f"## Policy the agent must follow\n{policy}\n\n## Outcome of the automatic checker (reward 0 = the run failed; expected actions may be noisy)\n"
            f"{json.dumps(outcome, ensure_ascii=False)}\n\n## Conversation\n" + "\n".join(lines) +
            f"\n\nThe agent made {len(calls)} tool calls (C0 to C{len(calls) - 1}). Return the JSON now.")


def load_index() -> list[dict]:
    return [json.loads(l) for l in open(INDEX_PATH)]


def choose_runs(per_domain: int) -> list[dict]:
    """First N runs per domain: heuristic-flagged runs (Class 2 / 6) first, then a seeded random fill."""
    chosen = []
    for dom in ("retail", "airline", "telecom"):
        rs = [r for r in load_index() if r["domain"] == dom]
        flagged = sorted((r for r in rs if r["heuristic_class6"] or r["heuristic_class2"]), key=lambda r: r["run_key"])
        rest = sorted((r for r in rs if not (r["heuristic_class6"] or r["heuristic_class2"])), key=lambda r: r["run_key"])
        random.Random(f"{SEED}-pick-{dom}").shuffle(rest)
        if len(flagged) > per_domain:
            flagged = random.Random(f"{SEED}-flag-{dom}").sample(flagged, per_domain)
        chosen += (flagged + rest)[:per_domain]
    return chosen


def runs_by_file(keys: set[str]):
    files = sorted({k.split("::")[0] for k in keys})
    for f in files:
        d = json.load(open(RESULTS / f"{f}.json"))
        for s in d["simulations"]:
            k = f"{f}::{s['id']}"
            if k in keys:
                yield {"file": f, "domain": domain_of(f), "source_model": f.split("_")[0], "task_id": str(s["task_id"]), "sim_id": s["id"],
                       "reward_info": s.get("reward_info") or {}, "messages": s["messages"]}


def cache_rows() -> dict:
    if not CACHE_PATH.exists():
        return {}
    rows = {}
    for line in open(CACHE_PATH):
        r = json.loads(line)
        rows[(r["run_key"], r["prompt_version"])] = r
    return rows


def cmd_estimate(args, quiet=False):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")  # a stand-in tokenizer: slightly over-counts vs OpenAI's
    chosen = choose_runs(args.per_domain)
    PLAN_PATH.write_text(json.dumps([r["run_key"] for r in chosen], indent=0))
    done = cache_rows()
    todo = {r["run_key"] for r in chosen if (r["run_key"], PROMPT_VERSION) not in done}
    sys_tok = len(tok(SYSTEM)["input_ids"])
    per_domain_cost, total_in = Counter(), 0
    for s in runs_by_file(todo):
        n = sys_tok + len(tok(render_prompt(s), truncation=False)["input_ids"])
        total_in += n
        per_domain_cost[s["domain"]] += n
    out_tok = len(todo) * args.out_tokens
    cost = total_in * PRICE_IN / 1e6 + out_tok * PRICE_OUT / 1e6
    print(f"runs to label: {len(todo)} ({len(chosen) - len(todo)} already cached) | input tokens {total_in:,} "
          f"({', '.join(f'{d} {t:,}' for d, t in per_domain_cost.items())}) | assumed output {args.out_tokens} tokens/run")
    print(f"ESTIMATED COST: ${cost:.2f}  (limit ${args.max_usd:.2f}; gpt-5-mini ${PRICE_IN}/${PRICE_OUT} per 1M tokens)")
    return cost


def call_model(client, prompt: str) -> dict:
    from tenacity import retry, stop_after_attempt, wait_exponential

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=2, max=40), reraise=True)
    def go():
        return client.chat.completions.create(model=MODEL, reasoning_effort=REASONING_EFFORT, response_format={"type": "json_object"},
                                              messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}])
    t0 = time.monotonic()
    resp = go()
    u = resp.usage
    det = getattr(u, "completion_tokens_details", None)
    return {"text": resp.choices[0].message.content or "", "in": u.prompt_tokens, "out": u.completion_tokens,
            "reasoning": getattr(det, "reasoning_tokens", 0) or 0, "latency_s": round(time.monotonic() - t0, 1)}


def parse_answer(text: str, n_calls: int):
    try:
        obj = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
    except (AttributeError, json.JSONDecodeError):
        return None
    k, c = obj.get("first_incorrect_call"), obj.get("class")
    if isinstance(k, str) and k.strip().lower() == "none":
        k = "none"
    elif isinstance(k, str) and re.fullmatch(r"[Cc]?\d+", k.strip()):
        k = int(k.strip().lstrip("Cc"))
    if k != "none" and not (isinstance(k, int) and 0 <= k < n_calls):
        return None
    if k != "none" and c not in range(1, 8):
        return None
    return {"first_incorrect_call": k, "class": None if k == "none" else int(c), "evidence": str(obj.get("evidence", ""))}


def cmd_localize(args):
    cost = cmd_estimate(args)
    if cost > args.max_usd:
        sys.exit(f"STOP: estimated cost ${cost:.2f} is above the ${args.max_usd:.2f} limit. Nothing was sent.")
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(ROOT / ".env")
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])  # never printed
    done = cache_rows()
    keys = json.load(open(PLAN_PATH))
    todo = {k for k in keys if (k, PROMPT_VERSION) not in done}
    if not todo:
        print("everything is cached")
        return
    jobs = [(s["file"] + "::" + s["sim_id"], len(agent_calls(s["messages"])), render_prompt(s)) for s in runs_by_file(todo)]
    spent, t0 = 0.0, time.monotonic()
    with open(CACHE_PATH, "a") as cache, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(call_model, client, p): (k, n) for k, n, p in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            k, n = futs[f]
            try:
                r = f.result()
            except Exception as e:  # noqa: BLE001 - recorded, not cached, so a rerun retries it
                print(f"[{i}/{len(jobs)}] {k[-12:]} FAILED: {type(e).__name__}: {str(e)[:120]}")
                continue
            c = (r["in"] * PRICE_IN + r["out"] * PRICE_OUT) / 1e6
            spent += c
            row = {"run_key": k, "prompt_version": PROMPT_VERSION, "model": MODEL, "reasoning_effort": REASONING_EFFORT,
                   "response_text": r["text"], "parsed": parse_answer(r["text"], n), "n_agent_calls": n, "usage": {"in": r["in"], "out": r["out"], "reasoning": r["reasoning"]},
                   "cost_usd": round(c, 5), "latency_s": r["latency_s"]}
            cache.write(json.dumps(row) + "\n")
            cache.flush()
            if i % 25 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] spent ${spent:.2f} in {time.monotonic() - t0:.0f}s")
    print(f"done: ${spent:.2f} spent this run")


def cmd_localize_more(args):
    """Localize test-split failed runs that were not localized yet (D-019 update, 2026-11-14), same gpt-5-mini prompt as before, within --max-usd.
    Only the Class 3 proposals of these runs become candidates (`build` drops the rest: the user reviews every Class 3 proposal, whose precision was 27%).
    The cost per run is estimated from the prompt's token count (ModernBERT tokenizer as a stand-in) and the mean output of the earlier calls; the
    spend is stopped by the real cost, checked after every batch of workers, and written to data/derived/api_spend.json."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    idx = load_index()
    done = cache_rows()
    order = sorted((r["run_key"] for r in idx if r["split"] == "test" and (r["run_key"], PROMPT_VERSION) not in done))
    random.Random(f"{SEED}-more").shuffle(order)
    mean_out = sum(r["usage"]["out"] for r in done.values()) / max(len(done), 1)
    sys_tok = len(tok(SYSTEM)["input_ids"])
    jobs, est = [], 0.0
    for s in runs_by_file(set(order)):
        k = s["file"] + "::" + s["sim_id"]
        n_in = sys_tok + len(tok(render_prompt(s), truncation=False)["input_ids"])
        jobs.append((k, len(agent_calls(s["messages"])), render_prompt(s), (n_in * PRICE_IN + mean_out * PRICE_OUT) / 1e6))
    rank = {k: i for i, k in enumerate(order)}
    jobs.sort(key=lambda j: rank[j[0]])
    chosen = []
    for j in jobs:
        if est + j[3] > args.max_usd:
            break
        chosen.append(j)
        est += j[3]
    print(f"{len(jobs)} test-split runs not localized yet; {len(chosen)} fit the ${args.max_usd:.2f} cap by the estimate (${est:.2f}; mean output {mean_out:.0f} tokens/run from {len(done)} earlier calls)")
    if args.estimate_only or not chosen:
        return
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(ROOT / ".env")
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])  # never printed
    spent, new_keys, t0 = 0.0, [], time.monotonic()
    with open(CACHE_PATH, "a") as cache, ThreadPoolExecutor(max_workers=args.workers) as ex:
        for b in range(0, len(chosen), args.workers * 2):
            batch = chosen[b:b + args.workers * 2]
            if spent + sum(j[3] for j in batch) > args.max_usd:
                batch = batch[: max(0, int((args.max_usd - spent) / (est / len(chosen))))]
                if not batch:
                    print(f"cap: spent ${spent:.3f}, next batch would exceed ${args.max_usd:.2f}")
                    break
            futs = {ex.submit(call_model, client, j[2]): j for j in batch}
            for f in as_completed(futs):
                k, n = futs[f][0], futs[f][1]
                try:
                    r = f.result()
                except Exception as e:  # noqa: BLE001 - not cached, a rerun retries it
                    print(f"  {k[-12:]} FAILED: {type(e).__name__}: {str(e)[:120]}")
                    continue
                c = (r["in"] * PRICE_IN + r["out"] * PRICE_OUT) / 1e6
                spent += c
                cache.write(json.dumps({"run_key": k, "prompt_version": PROMPT_VERSION, "model": MODEL, "reasoning_effort": REASONING_EFFORT, "response_text": r["text"],
                                        "parsed": parse_answer(r["text"], n), "n_agent_calls": n, "usage": {"in": r["in"], "out": r["out"], "reasoning": r["reasoning"]},
                                        "cost_usd": round(c, 5), "latency_s": r["latency_s"]}) + "\n")
                cache.flush()
                new_keys.append(k)
            print(f"  {len(new_keys)}/{len(chosen)} localized, ${spent:.3f} spent in {time.monotonic() - t0:.0f}s", flush=True)
    more = set(json.load(open(MORE_PATH))) if MORE_PATH.exists() else set()
    MORE_PATH.write_text(json.dumps(sorted(more | set(new_keys))))
    sp = json.load(open(DER / "api_spend.json"))
    sp["entries"].append({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "step": "7_class3_localize", "runs_or_calls": len(new_keys), "usd": round(spent, 4), "note": f"gpt-5-mini {REASONING_EFFORT}, test-split runs not localized before"})
    sp["total_usd"] = round(sum(e["usd"] for e in sp["entries"]), 4)
    sp["by_step"] = {}
    for e in sp["entries"]:
        sp["by_step"][e["step"]] = round(sp["by_step"].get(e["step"], 0) + e["usd"], 4)
    (DER / "api_spend.json").write_text(json.dumps(sp, indent=1))
    print(f"done: ${spent:.3f} spent, {len(new_keys)} runs localized; all-prompt total ${sp['total_usd']:.2f} of ${sp['hard_limit_usd']}")


def cmd_localize_train(args):
    """Localize the train-split failed runs that were not localized yet (D-028), with the same gpt-5-mini prompt (prompt version localize_v1, so the cache rows are
    interchangeable with the earlier ones), within --max-usd. Telecom first (it yields about four times more usable errors per dollar), a seeded shuffle inside each
    domain. The cost per run is estimated from the prompt's token count (ModernBERT tokenizer as a stand-in, which over-counts by about 9%) and the mean output of the
    earlier calls; the spend is stopped by the REAL cost, checked after every batch, and written to data/derived/api_spend.json."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    idx = load_index()
    done = cache_rows()
    todo = [r for r in idx if r["split"] == "train" and (r["run_key"], PROMPT_VERSION) not in done]
    order = []
    for dom in ("telecom", "retail", "airline"):
        ks = sorted(r["run_key"] for r in todo if r["domain"] == dom)
        random.Random(f"{SEED}-train-{dom}").shuffle(ks)
        order += ks
    mean_out = sum(r["usage"]["out"] for r in done.values()) / max(len(done), 1)
    sys_tok = len(tok(SYSTEM)["input_ids"])
    jobs = []
    for s in runs_by_file(set(order)):
        k = s["file"] + "::" + s["sim_id"]
        n_in = sys_tok + len(tok(render_prompt(s), truncation=False)["input_ids"])
        jobs.append((k, len(agent_calls(s["messages"])), render_prompt(s), (n_in * PRICE_IN + mean_out * PRICE_OUT) / 1e6))
    rank = {k: i for i, k in enumerate(order)}
    jobs.sort(key=lambda j: rank[j[0]])
    chosen, est = [], 0.0
    for j in jobs:
        if est + j[3] > args.max_usd:
            break
        chosen.append(j)
        est += j[3]
    by_dom = Counter(rank_dom for rank_dom in (k.split("_")[1].split("-")[0] for k, *_ in chosen))
    print(f"{len(jobs)} train-split runs not localized yet; {len(chosen)} fit the ${args.max_usd:.2f} cap by the estimate (${est:.2f} for them, ${sum(j[3] for j in jobs):.2f} for all; mean output {mean_out:.0f} tokens/run from {len(done)} earlier calls) {dict(by_dom)}")
    if args.estimate_only or not chosen:
        return
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(ROOT / ".env")
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])  # never printed
    spent, new_keys, t0 = 0.0, [], time.monotonic()
    per_run = est / len(chosen)
    with open(CACHE_PATH, "a") as cache, ThreadPoolExecutor(max_workers=args.workers) as ex:
        for b in range(0, len(chosen), args.workers * 2):
            batch = chosen[b:b + args.workers * 2]
            if spent + len(batch) * per_run * 0.9 > args.max_usd:     # the estimate over-counts by about 9%; stop before the cap is passed
                batch = batch[: max(0, int((args.max_usd - spent) / (per_run * 0.9)))]
                if not batch:
                    print(f"cap: spent ${spent:.3f}, the next batch could pass ${args.max_usd:.2f}")
                    break
            futs = {ex.submit(call_model, client, j[2]): j for j in batch}
            for f in as_completed(futs):
                k, n = futs[f][0], futs[f][1]
                try:
                    r = f.result()
                except Exception as e:  # noqa: BLE001 - not cached, a rerun retries it
                    print(f"  {k[-12:]} FAILED: {type(e).__name__}: {str(e)[:120]}")
                    continue
                c = (r["in"] * PRICE_IN + r["out"] * PRICE_OUT) / 1e6
                spent += c
                cache.write(json.dumps({"run_key": k, "prompt_version": PROMPT_VERSION, "model": MODEL, "reasoning_effort": REASONING_EFFORT, "response_text": r["text"],
                                        "parsed": parse_answer(r["text"], n), "n_agent_calls": n, "usage": {"in": r["in"], "out": r["out"], "reasoning": r["reasoning"]},
                                        "cost_usd": round(c, 5), "latency_s": r["latency_s"]}) + "\n")
                cache.flush()
                new_keys.append(k)
            print(f"  {len(new_keys)}/{len(chosen)} localized, ${spent:.3f} spent in {time.monotonic() - t0:.0f}s", flush=True)
            more = set(json.load(open(TRAIN_MORE_PATH))) if TRAIN_MORE_PATH.exists() else set()
            TRAIN_MORE_PATH.write_text(json.dumps(sorted(more | set(new_keys))))
    sp = json.load(open(DER / "api_spend.json"))
    sp["entries"].append({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "step": "9_train_localize", "runs_or_calls": len(new_keys), "usd": round(spent, 4), "note": f"gpt-5-mini {REASONING_EFFORT}, train-split failed runs not localized before (D-028)"})
    sp["total_usd"] = round(sum(e["usd"] for e in sp["entries"]), 4)
    sp["by_step"] = {}
    for e in sp["entries"]:
        sp["by_step"][e["step"]] = round(sp["by_step"].get(e["step"], 0) + e["usd"], 4)
    (DER / "api_spend.json").write_text(json.dumps(sp, indent=1))
    print(f"done: ${spent:.3f} spent, {len(new_keys)} runs localized; all-prompt total ${sp['total_usd']:.2f} of ${sp['hard_limit_usd']}")


def cmd_append_train(args):
    """Turn the localizations of `localize-train` into candidates and pending positives, APPENDED to the review files with `batch: train_more` (nothing existing is
    rewritten: `build` must not be rerun, it drops the later fields of the enriched rows). The first labeler's proposal is the only label (auto_status first_labeler_only)."""
    cache = cache_rows()
    keys = set(json.load(open(TRAIN_MORE_PATH)))
    idx = {r["run_key"]: r for r in load_index()}
    have = {json.loads(l)["candidate_id"] for l in open(REVIEW_PATH)}
    review, positives, outcome = [], [], Counter()
    for s in runs_by_file(keys):
        k = run_key(s)
        ix = idx[k]
        p, why = proposals_for(k, ix, cache)
        outcome[why] += 1
        if p is None or p["class"] not in range(1, 8):
            continue
        calls = agent_calls(s["messages"])
        turns, call_turn = render_full(s["messages"])
        ci = p["first_incorrect_call"]
        cid = f"{k}::C{ci}"
        if cid in have:
            outcome["already present"] += 1
            continue
        t = call_turn[ci]
        lo, hi = max(0, t - 20), min(len(turns), t + 5)
        window = []
        for x in turns[lo:hi]:
            x = dict(x)
            if x.get("role") == "tool" and len(x["content"]) > RESULT_CAP:
                x["content"] = x["content"][:RESULT_CAP] + f"…[truncated, {len(x['content'])} chars]"
            window.append(x)
        review.append({
            "candidate_id": cid, "run_key": k, "domain": s["domain"], "task_id": s["task_id"], "source_model": s["source_model"], "split": ix["split"],
            "proposal_source": p["source"], "proposed_class": p["class"], "proposed_class_name": CLASS_NAMES[p["class"]], "evidence": p["evidence"],
            "flagged_call": {"call_index": ci, "turn_index": t, "name": calls[ci]["name"], "arguments": calls[ci]["arguments"]},
            "window": {"start": lo, "end": hi, "flagged_offset": t - lo, "total_turns": len(turns)}, "turns": window,
            "heuristic_class6": ix["heuristic_class6"], "heuristic_class2": ix["heuristic_class2"], "reward_summary": ix["reward_summary"],
            "human_label": "", "human_class": "", "human_note": "", "origin": "natural", "auto_status": "first_labeler_only", "batch": "train_more"})
        err_ids = {m.get("id") for m in s["messages"] if is_error(m)}
        for c in calls[:ci]:
            if c["name"] in FREE_TEXT_TOOLS or c["id"] in err_ids:
                continue
            positives.append({"positive_id": f"{k}::C{c['k']}", "run_key": k, "linked_candidate_id": cid, "call_index": c["k"], "turn_index": call_turn[c["k"]], "name": c["name"],
                              "arguments": c["arguments"], "domain": s["domain"], "task_id": s["task_id"], "split": ix["split"], "status": "pending", "batch": "train_more"})
    with open(REVIEW_PATH, "a") as f:
        f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in review))
    with open(POSITIVES_PATH, "a") as f:
        f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in positives))
    print(f"appended {len(review)} candidates and {len(positives)} pending positives; outcomes {dict(outcome)}")
    print("   candidates by (domain, proposed class):", dict(sorted(Counter((r["domain"], r["proposed_class"]) for r in review).items())))


# ───────────────────────── stage: build ─────────────────────────

def proposals_for(k: str, ix: dict, cache: dict) -> tuple[dict | None, str]:
    """One proposal per run: the LLM's if it was labeled (and parsed), else the heuristic's if one fired.
    Returns (proposal or None, outcome) where outcome is one of llm_none, llm_unparsed, llm, heuristic, nothing."""
    row = cache.get((k, PROMPT_VERSION))
    if row is not None:
        p = row["parsed"]
        if p is None:
            return None, "llm_unparsed"
        if p["first_incorrect_call"] == "none":
            return None, "llm_none"
        return {**p, "source": "llm"}, "llm"
    if ix["heuristic_class6"] and ix["heuristic_class6_call"] is not None:
        return {"first_incorrect_call": ix["heuristic_class6_call"], "class": 6, "evidence": "[heuristic] " + ix["heuristic_class6"], "source": "heuristic"}, "heuristic"
    if ix["heuristic_class2"] and ix["heuristic_class2_call"] is not None:
        return {"first_incorrect_call": ix["heuristic_class2_call"], "class": 2, "evidence": "[heuristic] " + ix["heuristic_class2"], "source": "heuristic"}, "heuristic"
    return None, "nothing"


def cmd_build(args):
    cache = cache_rows()
    MORE_KEYS = set(json.load(open(MORE_PATH))) if MORE_PATH.exists() else set()
    idx = {r["run_key"]: r for r in load_index()}
    plan = set(json.load(open(PLAN_PATH)))
    # runs to consider: everything the LLM labeled, plus every heuristic-flagged run (so the review file is useful before / without the LLM)
    todo = {k for k in idx if (k, PROMPT_VERSION) in cache} | {k for k in idx if idx[k]["heuristic_class6"] or idx[k]["heuristic_class2"]}
    review, positives = [], []
    outcome = Counter()
    for s in runs_by_file(todo):
        k = run_key(s)
        ix = idx[k]
        p, why = proposals_for(k, ix, cache)
        outcome[why] += 1
        if p is None:
            continue
        if k in MORE_KEYS and p["class"] != 3:  # runs localized by `localize-more`: only Class 3 proposals become candidates
            outcome["more_runs_not_class3_dropped"] += 1
            continue
        calls = agent_calls(s["messages"])
        turns, call_turn = render_full(s["messages"])
        ci = p["first_incorrect_call"]
        t = call_turn[ci]
        lo, hi = max(0, t - 20), min(len(turns), t + 5)
        window = []
        for x in turns[lo:hi]:
            x = dict(x)
            if x.get("role") == "tool" and len(x["content"]) > RESULT_CAP:
                x["content"] = x["content"][:RESULT_CAP] + f"…[truncated, {len(x['content'])} chars]"
            window.append(x)
        cid = f"{k}::C{ci}"
        review.append({
            "candidate_id": cid, "run_key": k, "domain": s["domain"], "task_id": s["task_id"], "source_model": s["source_model"], "split": ix["split"],
            "proposal_source": p["source"], "proposed_class": p["class"], "proposed_class_name": CLASS_NAMES[p["class"]], "evidence": p["evidence"],
            "flagged_call": {"call_index": ci, "turn_index": t, "name": calls[ci]["name"], "arguments": calls[ci]["arguments"]},
            "window": {"start": lo, "end": hi, "flagged_offset": t - lo, "total_turns": len(turns)}, "turns": window,
            "heuristic_class6": ix["heuristic_class6"], "heuristic_class2": ix["heuristic_class2"], "reward_summary": ix["reward_summary"],
            "human_label": "", "human_class": "", "human_note": ""})
        err_ids = {m.get("id") for m in s["messages"] if is_error(m)}
        for c in calls[:ci]:
            if c["name"] in FREE_TEXT_TOOLS or c["id"] in err_ids:
                continue
            positives.append({"positive_id": f"{k}::C{c['k']}", "run_key": k, "linked_candidate_id": cid, "call_index": c["k"],
                              "turn_index": call_turn[c["k"]], "name": c["name"], "arguments": c["arguments"], "domain": s["domain"],
                              "task_id": s["task_id"], "split": ix["split"], "status": "pending"})
    rare = {2, 3, 6, 7}

    def order(r):  # rare classes and airline first, then by class
        return (-(int(r["proposed_class"] in rare or bool(r["heuristic_class6"] or r["heuristic_class2"])) + int(r["domain"] == "airline")),
                r["proposed_class"], r["domain"], r["candidate_id"])
    review.sort(key=order)
    REVIEW_PATH.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in review))
    POSITIVES_PATH.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in positives))
    llm_keys = [k for k in idx if (k, PROMPT_VERSION) in cache]
    rep = {"llm_runs_labeled": len(llm_keys), "llm_cost_usd": round(sum(cache[(k, PROMPT_VERSION)]["cost_usd"] for k in llm_keys), 3),
           "runs_considered": len(todo), "outcomes": dict(outcome), "candidates": len(review), "pending_positives": len(positives),
           "by_proposal_source": dict(Counter(r["proposal_source"] for r in review)),
           "by_domain": dict(Counter(r["domain"] for r in review)), "by_split": dict(Counter(r["split"] for r in review)),
           "by_proposed_class": {f"{c} {CLASS_NAMES[c]}": n for c, n in sorted(Counter(r["proposed_class"] for r in review).items())},
           "by_agent_model": dict(Counter(r["source_model"] for r in review)),
           "llm_none_by_domain": dict(Counter(idx[k]["domain"] for k in llm_keys if cache[(k, PROMPT_VERSION)]["parsed"] and cache[(k, PROMPT_VERSION)]["parsed"]["first_incorrect_call"] == "none")),
           "planned_llm_runs_not_yet_labeled": len(plan - set(llm_keys))}
    REPORT_PATH.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["split", "pick", "estimate", "localize", "localize-more", "localize-train", "append-train", "build"])
    ap.add_argument("--estimate-only", action="store_true")
    ap.add_argument("--per-domain", type=int, default=150)
    ap.add_argument("--max-usd", type=float, default=4.0)
    ap.add_argument("--out-tokens", type=int, default=900, help="assumed output tokens per run (reasoning + JSON) for the estimate")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    {"split": cmd_split, "pick": cmd_pick, "estimate": cmd_estimate, "localize": cmd_localize, "localize-more": cmd_localize_more, "localize-train": cmd_localize_train, "append-train": cmd_append_train, "build": cmd_build}[args.stage](args)


if __name__ == "__main__":
    main()
