"""Three review queues for the real / induced candidates (D-018, D-019). Nothing here writes a training or test file.

    queue 1  every TEST-split error candidate, natural and induced. Rare classes first (2, 5, 6), then 3 and 1, then 4 and 7.
    queue 2  a random 150 TEST-split correct examples (natural and induced), seeded and stratified by domain.
    queue 3  a random 40 'auto_accepted' TRAIN candidates (natural and induced): this measures how often auto-acceptance is right.
    queue 4  new test-split candidates found after queue 1 (strict Class 6, extra Class 3).
    queue 5  re-review under docs/labeling_guide.md: every queue-2 "wrong" on a read-only call, and every Class 7 decision on a write call (queues 1 and 2),
             then (at the end) the 8 Class 7 rows of test_real.jsonl.
    queue 6  line lookups (`get_details_by_id` on a line id) that the line-probe rule (data/line_probe_rule.py) cannot decide confidently; ids `q6::<original id>`.
             Item ids are `q5::<original id>`; the original decision is never changed, the queue-5 one supersedes it (data/review_decisions.py).

    python data/review_queues.py build            # freeze the three queues into data/derived/review_queue.jsonl (refuses to overwrite; --rebuild to redo)
    python data/review_queues.py sizes            # sizes and estimated time (1 min per error, 20 s per correct example)
    python data/review_queues.py review           # work through the queues in order; saves after every decision; resumes where you stopped
    python data/review_queues.py review --queue 2 # only one queue
    python data/review_queues.py stats            # progress, and the label precision measured by queue 3
    python data/review_queues.py merge            # copy the human decisions back into the candidate / positive files

Decisions are appended to data/derived/review_queue_decisions.jsonl (fsynced); every record carries `origin` (natural | induced), the queue and
the item kind, so natural and induced labels never mix. At the prompt:
  error items   a accept | c N accept with class N | r reject | s skip | n note | f full text | q quit
  correct items y the call is correct | w [N] the call is actually wrong (optionally with a class) | s skip | o note | f full text | q quit
"""

import argparse
import json
import os
import random
import shutil
import sys
import textwrap
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from review_real import CLASS_NAMES, clip  # noqa: E402
from review_decisions import Q5, effective, load_effective  # noqa: E402
Q6 = "q6::"
from class7_reasons import REASONS, is_read_only, suggest  # noqa: E402

DER = HERE / "derived"
QUEUE = DER / "review_queue.jsonl"
DECISIONS = DER / "review_queue_decisions.jsonl"
SRC = {"natural": {"cands": DER / "real_candidates_review.jsonl", "pos": DER / "real_positive_candidates.jsonl"},
       "induced": {"cands": DER / "induced_candidates_review.jsonl", "pos": DER / "induced_positive_candidates.jsonl"}}
SEED = 20261113
N_CORRECT, N_AUDIT = 150, 40
GROUP = {2: 0, 5: 0, 6: 0, 3: 1, 1: 1, 4: 2, 7: 2}
MINUTES = {"error": 1.0, "correct": 20 / 60, "recheck": 0.5}
LINE_RULE_BOX = """  RULE (docs/labeling_guide.md, "Line lookups")  a get_details_by_id call on a LINE id (L1001 ...):
    KNOWABLE  the target line was identifiable before the call (the user named a line, the customer has one line, or an earlier result tied the stated phone number
              or the problem to one line): looking up a DIFFERENT line is Class 1 (press 1); looking up THAT line is correct (y)
    NOT KNOWABLE  a lookup to find out which line is affected is correct (y), not an error
    NEVER Class 3: no other tool should have been used.   The rule could not decide this item: the evidence below shows why."""
RARE_FIRST = [2, 6, 3, 4, 5, 1, 7]  # by the number of real test errors per class after queue 1 (D-022)


def rj(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p) if l.strip()] if p.exists() else []


def origin_of(r: dict) -> str:
    return r.get("origin") or ("induced" if str(r.get("source", "")).startswith("induced") else "natural")


# ───────────────────────── build ─────────────────────────

def positive_window(p: dict, cache: dict):
    """Turns around a correct-example call (20 before, 5 after) built from the raw run."""
    from mine_real_errors import RESULT_CAP, render_full

    key = p["run_key"]
    msgs = cache[key]
    turns, call_turn = render_full(msgs)
    t = call_turn[p["call_index"]]
    lo, hi = max(0, t - 20), min(len(turns), t + 5)
    win = []
    for x in turns[lo:hi]:
        x = dict(x)
        if x.get("role") == "tool" and len(x["content"]) > RESULT_CAP:
            x["content"] = x["content"][:RESULT_CAP] + f"…[truncated, {len(x['content'])} chars]"
        win.append(x)
    return {"start": lo, "end": hi, "flagged_offset": t - lo, "total_turns": len(turns)}, win


def cmd_build(args):
    if QUEUE.exists() and not args.rebuild:
        sys.exit(f"{QUEUE} exists: the queues are frozen so a resumed review stays stable. Use --rebuild to redo them (decisions are keyed by item id).")
    items = []
    cand_by_origin = {o: rj(s["cands"]) for o, s in SRC.items()}
    # ---- queue 1: all test-split error candidates
    q1 = []
    for o, rows in cand_by_origin.items():
        for c in rows:
            if c["split"] == "test":
                q1.append({"item_id": f"{o}::{c['candidate_id']}", "queue": 1, "kind": "error", "origin": o, "candidate": c})
    q1.sort(key=lambda x: (GROUP[x["candidate"]["proposed_class"]], x["candidate"]["proposed_class"], x["origin"], x["candidate"]["domain"], x["candidate"]["candidate_id"]))
    # ---- queue 2: 150 test-split correct examples, stratified by domain, seeded
    pool = defaultdict(list)
    for o, s in SRC.items():
        for p in rj(s["pos"]):
            if p["split"] == "test" and p.get("auto_label", "correct") == "correct":
                pool[p["domain"]].append((o, p))
    total = sum(len(v) for v in pool.values())
    take = {d: min(len(v), round(N_CORRECT * len(v) / total)) for d, v in pool.items()}
    while sum(take.values()) > N_CORRECT:
        take[max(take, key=take.get)] -= 1
    while sum(take.values()) < N_CORRECT and any(take[d] < len(pool[d]) for d in pool):
        d = max((d for d in pool if take[d] < len(pool[d])), key=lambda d: len(pool[d]) - take[d])
        take[d] += 1
    picked = []
    for d in sorted(pool):
        rows = sorted(pool[d], key=lambda x: x[1]["positive_id"])
        random.Random(f"{SEED}-q2-{d}").shuffle(rows)
        picked += rows[: take[d]]
    # raw runs for the natural ones
    from mine_real_errors import runs_by_file

    nat_keys = {p["run_key"] for o, p in picked if o == "natural"}
    msgs = {f"{s['file']}::{s['sim_id']}": s["messages"] for s in runs_by_file(nat_keys)}
    for r in rj(HERE / "raw" / "induced" / "induced_runs.jsonl"):
        msgs[r["run_id"]] = r["messages"]
    q2 = []
    for o, p in sorted(picked, key=lambda x: (x[1]["domain"], x[1]["positive_id"])):
        win, turns = positive_window(p, msgs)
        q2.append({"item_id": f"{o}::{p['positive_id']}", "queue": 2, "kind": "correct", "origin": o,
                   "positive": {**{k: p[k] for k in ("positive_id", "run_key", "domain", "task_id", "split", "call_index", "name", "arguments")}, "auto_label": p.get("auto_label", "pending natural: assumed correct, before the first flagged call"),
                                "reason": p.get("reason", ""), "induced": p.get("induced"), "linked_candidate_id": p.get("linked_candidate_id")}, "window": win, "turns": turns})
    # ---- queue 3: 40 auto_accepted training candidates
    acc = [{"item_id": f"{o}::{c['candidate_id']}", "queue": 3, "kind": "error", "origin": o, "candidate": c}
           for o, rows in cand_by_origin.items() for c in rows if c["split"] == "train" and c.get("auto_status") == "auto_accepted"]
    acc.sort(key=lambda x: x["item_id"])
    random.Random(f"{SEED}-q3").shuffle(acc)
    q3 = acc[:N_AUDIT]
    items = q1 + q2 + q3
    QUEUE.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items))
    print(f"wrote {QUEUE}")
    cmd_sizes(args)
    print("queue 2 allocation by domain:", dict(take), "| pool sizes:", {d: len(v) for d, v in pool.items()})
    print("queue 3 pool (auto_accepted train):", len(acc), "->", dict(Counter(i["origin"] for i in q3)))


def cmd_add_queue4(args):
    """Queue 4 (D-019 / D-022 update, 2026-11-14): every test-split error candidate that is in no queue yet (new induced Class 6 candidates from the strict
    plan, new natural Class 3 proposals from the extra localization), rare classes first. Queues 1 to 3 are not touched: their lines stay as they are,
    only the lines of queue 4 are (re)written."""
    raw = [l for l in open(QUEUE) if l.strip()]
    keep = [l for l in raw if json.loads(l)["queue"] != 4]
    seen = {json.loads(l)["item_id"] for l in keep}
    if len(keep) != len(raw) and not args.rebuild:
        sys.exit("queue 4 already exists; use --rebuild to rewrite it (queues 1 to 3 stay untouched)")
    rarity = {k: i for i, k in enumerate(RARE_FIRST)}
    q4 = []
    for o, src in SRC.items():
        for c in rj(src["cands"]):
            iid = f"{o}::{c['candidate_id']}"
            if c["split"] == "test" and iid not in seen:
                q4.append({"item_id": iid, "queue": 4, "kind": "error", "origin": o, "candidate": c})
    q4.sort(key=lambda x: (rarity[x["candidate"]["proposed_class"]], x["origin"], x["candidate"]["domain"] != "airline", x["candidate"]["domain"], x["candidate"]["candidate_id"]))
    tmp = QUEUE.with_suffix(".tmp")
    tmp.write_text("".join(keep) + "".join(json.dumps(i, ensure_ascii=False) + "\n" for i in q4))
    os.replace(tmp, QUEUE)
    print(f"queue 4: {len(q4)} items added (rare classes first: {RARE_FIRST}); queues 1 to 3 untouched ({len(keep)} lines)")
    print("   by (origin, proposed class, domain):", dict(sorted(Counter((i['origin'], i['candidate']['proposed_class'], i['candidate']['domain']) for i in q4).items(), key=lambda kv: str(kv[0]))))


POLICY_FILE = {"airline": "data/raw/tau2-bench/data/tau2/domains/airline/policy.md", "retail": "data/raw/tau2-bench/data/tau2/domains/retail/policy.md",
               "telecom": "data/raw/tau2-bench/data/tau2/domains/telecom/main_policy.md"}


def q5_test_real_items() -> list:
    """Queue-5 items for the Class 7 rows of data/derived/test_real.jsonl (hand-labeled in phase 4, before the labeling guide). They are not in queues 1 to 4, so
    each item carries a display copy built from the raw run; the call is found in the run by tool and arguments (unique for all 8 rows today)."""
    from mine_real_errors import agent_calls, render_full, runs_by_file

    rows = [r for r in rj(DER / "test_real.jsonl") if r["label"] == "incorrect" and str(r.get("error_class")) == "7"]
    runs = {f"{s['file']}::{s['sim_id']}": s for s in runs_by_file({r["trajectory_id"] for r in rows})}
    out = []
    for r in rows:
        key, cc = r["trajectory_id"], r["candidate_tool_call"]
        s = runs[key]
        hits = [c["k"] for c in agent_calls(s["messages"]) if c["name"] == cc["name"] and c["arguments"] == cc["arguments"]]
        if len(hits) != 1:
            sys.exit(f"{key}: {len(hits)} calls in the run match {cc['name']} with these arguments; cannot place the test_real call")
        idx = hits[0]
        win, window_turns = positive_window({"run_key": key, "call_index": idx}, {key: s["messages"]})
        turns, call_turn = render_full(s["messages"])
        t = call_turn[idx]
        call = {"name": turns[t]["name"], "arguments": turns[t]["arguments"]}
        cand = {"candidate_id": key, "run_key": key, "domain": r["domain"], "task_id": str(r["task_id"]), "source_model": r.get("source_model"), "split": "test",
                "source": "human_labeled_phase4", "origin": "natural", "proposal_source": "hand label in phase 4 (test_real.jsonl)", "auto_flags": [],
                "proposed_class": 7, "proposed_class_name": "policy_violation",
                "evidence": f"hand-labeled before the labeling guide: {r.get('human_label') or '(no text)'}" + (f" / note: {r['human_note']}" if r.get("human_note") else ""),
                "flagged_call": {"call_index": idx, "name": call["name"], "arguments": call["arguments"]}, "window": win, "turns": window_turns}
        src = {"item_id": f"test_real::{key}", "queue": 0, "kind": "error", "origin": "natural", "candidate": cand}
        out.append({"item_id": Q5 + src["item_id"], "queue": 5, "kind": "recheck", "origin": "natural", "orig_item_id": src["item_id"], "orig_queue": 0,
                    "orig_kind": "test_real", "group": "test_real_class7", "domain": r["domain"], "task_id": str(r["task_id"]), "run_key": key, "call_index": idx,
                    "call": {**call, "read_only": is_read_only(call["name"])},
                    "orig_decision": {"decision": "hand label", "human_label": "incorrect", "human_class": 7, "human_note": r.get("human_label") or "", "time": r.get("labeled_at", "")},
                    "prefill": suggest(turns[:t], call, r["domain"]), "source_item": src})
    return out


def cmd_add_queue5_testreal(args):
    """Append the test_real Class 7 items to queue 5. Every existing line (queues 1 to 5) stays as it is, in the same order; items already there are skipped."""
    raw = [l for l in open(QUEUE) if l.strip()]
    have = {json.loads(l)["item_id"] for l in raw}
    new = [i for i in q5_test_real_items() if i["item_id"] not in have]
    tmp = QUEUE.with_suffix(".tmp")
    tmp.write_text("".join(raw) + "".join(json.dumps(i, ensure_ascii=False) + "\n" for i in new))
    os.replace(tmp, QUEUE)
    print(f"queue 5: {len(new)} test_real Class 7 items appended at the end ({len(raw)} existing lines untouched)")
    for i in new:
        p = i["prefill"]
        print(f"   {i['domain']:8s} {i['call']['name']:26s} pre-fill {p['verdict']}/{p['reason']}  {'; '.join(s[:80] for s in p['signals'][:1])}")


def cmd_add_queue5(args):
    """Queue 5 (2026-11-15, docs/labeling_guide.md): re-review of (a) every queue-2 item marked wrong whose call is read-only, whatever class it was given, and
    (b) every Class 7 decision of queue 1 or 2 on a write call (and, as the guide's rule 1 makes them suspect, on a read-only call). Each item carries the original decision, a heuristic pre-fill of the sub-reason and a copy of
    the original item for display. Queues 1 to 4 and every decision stay untouched; the queue-5 decision supersedes the original in the builders
    (data/review_decisions.py)."""
    from mine_real_errors import agent_calls, render_full, runs_by_file  # noqa: F401

    raw = [l for l in open(QUEUE) if l.strip()]
    keep = [l for l in raw if json.loads(l)["queue"] != 5]
    if len(keep) != len(raw) and not args.rebuild:
        sys.exit("queue 5 already exists; use --rebuild to rewrite it (queues 1 to 4 and all decisions stay untouched)")
    items, dec = [json.loads(l) for l in keep], latest()
    picks = []
    for it in items:
        if it["queue"] not in (1, 2):
            continue
        d = dec.get(it["item_id"])
        if not d or d["decision"] == "skip":
            continue
        call = it["candidate"]["flagged_call"] if it["kind"] == "error" else it["positive"]
        ro = is_read_only(call["name"])
        if it["queue"] == 2 and d["decision"] == "wrong" and ro:
            group = "q2_wrong_read_only"
        elif d.get("human_class") == 7 and d["decision"] in ("accept", "change_class", "wrong"):
            group = "class7_write" if not ro else "class7_on_read_only"   # a Class 7 label on a read-only call breaks guide rule 1: it is re-reviewed too
        else:
            continue
        rec = it["candidate"] if it["kind"] == "error" else it["positive"]
        picks.append((it, d, group, rec))
    keys = {rec["run_key"] for it, d, g, rec in picks if it["origin"] == "natural"}
    msgs = {f"{s['file']}::{s['sim_id']}": s["messages"] for s in runs_by_file(keys)}
    for r in rj(HERE / "raw" / "induced" / "induced_runs.jsonl"):
        msgs[r["run_id"]] = r["messages"]
    q5 = []
    for it, d, group, rec in picks:
        idx = (rec["flagged_call"] if it["kind"] == "error" else rec)["call_index"]
        turns, call_turn = render_full(msgs[rec["run_key"]])
        t = call_turn[idx]
        call = {"name": turns[t]["name"], "arguments": turns[t]["arguments"]}
        pre = suggest(turns[:t], call, rec["domain"])
        q5.append({"item_id": Q5 + it["item_id"], "queue": 5, "kind": "recheck", "origin": it["origin"], "orig_item_id": it["item_id"], "orig_queue": it["queue"],
                   "orig_kind": it["kind"], "group": group, "domain": rec["domain"], "task_id": rec["task_id"], "run_key": rec["run_key"], "call_index": idx,
                   "call": {**call, "read_only": is_read_only(call["name"])},
                   "orig_decision": {k: d.get(k) for k in ("decision", "human_label", "human_class", "human_note", "time")},
                   "prefill": pre, "source_item": it})
    q5.sort(key=lambda x: (x["domain"], x["run_key"], x["call_index"]))   # the same conversation stays together
    q5 += q5_test_real_items()                                           # then the test_real Class 7 rows, at the end
    tmp = QUEUE.with_suffix(".tmp")
    tmp.write_text("".join(keep) + "".join(json.dumps(i, ensure_ascii=False) + "\n" for i in q5))
    os.replace(tmp, QUEUE)
    print(f"queue 5: {len(q5)} items added; queues 1 to 4 untouched ({len(keep)} lines)")
    print("   by group:", dict(Counter(i["group"] for i in q5)), "| by domain:", dict(Counter(i["domain"] for i in q5)), "| by origin:", dict(Counter(i["origin"] for i in q5)))
    print("   original verdicts:", dict(Counter((i["orig_queue"], i["orig_decision"]["decision"], i["orig_decision"]["human_class"]) for i in q5)))
    print("   pre-fill:", dict(sorted(Counter((i["prefill"]["verdict"], i["prefill"]["reason"], i["prefill"]["tag"]) for i in q5).items(), key=str)))


def cmd_add_queue6(args):
    """Queue 6: the line-probe items the rule could not decide. Queues 1 to 5 and every decision stay untouched; only the lines of queue 6 are (re)written."""
    from apply_line_probe_rule import collect

    raw = [l for l in open(QUEUE) if l.strip()]
    keep = [l for l in raw if json.loads(l)["queue"] != 6]
    if len(keep) != len(raw) and not args.rebuild:
        sys.exit("queue 6 already exists; use --rebuild to rewrite it (queues 1 to 5 and all decisions stay untouched)")
    dec = latest()
    q6 = []
    for x in collect():
        if x["judge"]["verdict"] != "undecided":
            continue
        j, od = x["judge"], dec.get(x["orig_id"]) or {}
        src = x["item"] if x["kind"] == "queue" else None
        if src is None:
            continue   # test_real rows have no display copy; none was undecided when this was written
        q6.append({"item_id": Q6 + x["orig_id"], "queue": 6, "kind": "recheck", "origin": x["origin"], "orig_item_id": x["orig_id"], "orig_queue": src["queue"], "orig_kind": src["kind"],
                   "group": "line_probe_undecided", "domain": x["domain"], "task_id": x["task_id"], "run_key": (src["candidate"] if src["kind"] == "error" else src["positive"])["run_key"],
                   "call_index": (src["candidate"]["flagged_call"] if src["kind"] == "error" else src["positive"])["call_index"],
                   "call": {**x["call"], "read_only": True}, "rule_box": LINE_RULE_BOX,
                   "orig_decision": {k: od.get(k) for k in ("decision", "human_label", "human_class", "human_note", "time")},
                   "prefill": {"verdict": None, "reason": None, "tag": None, "signals": [f"the rule could not decide: {j['why']}"] + j["evidence"]}, "source_item": src})
    tmp = QUEUE.with_suffix(".tmp")
    tmp.write_text("".join(keep) + "".join(json.dumps(i, ensure_ascii=False) + "\n" for i in q6))
    os.replace(tmp, QUEUE)
    print(f"queue 6: {len(q6)} items (line lookups the rule could not decide); queues 1 to 5 untouched ({len(keep)} lines)")


def cmd_sizes(args):
    items = rj(QUEUE)
    for q in (1, 2, 3, 4, 5, 6):
        its = [i for i in items if i["queue"] == q]
        mins = sum(MINUTES[i["kind"]] for i in its)
        by = Counter((i["origin"], i["candidate"]["proposed_class"] if i["kind"] == "error" else i["positive"]["domain"] if i["kind"] == "correct" else i["group"]) for i in its)
        name = {1: "test-split error candidates", 2: "test-split correct examples", 3: "auto_accepted train audit", 4: "new test-split error candidates (rare classes first)",
                5: "re-review under the labeling guide (Class 7 writes, read-only 'wrong')", 6: "line lookups the line-probe rule could not decide"}[q]
        if not its:
            continue
        print(f"queue {q}: {len(its):4d} items  ~{mins:5.0f} min ({mins / 60:.1f} h)  {name}")
        print("          by (origin, class or domain):", dict(sorted(by.items(), key=lambda kv: str(kv[0]))))
    tot = sum(MINUTES[i["kind"]] for i in items)
    print(f"total {len(items)} items, about {tot:.0f} min ({tot / 60:.1f} h)")


# ───────────────────────── review ─────────────────────────

def latest() -> dict:
    return {d["item_id"]: d for d in rj(DECISIONS)}


BACKUP_EVERY = 25
BACKUP_DIR = Path(os.environ.get("TCV_BACKUP_DIR", Path.home() / "tcv_backups"))  # outside the repo


def backup_decisions() -> None:
    """Timestamped copy of the decisions file outside the repository: the decisions are human work that cannot be regenerated."""
    try:
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        n = sum(1 for _ in open(DECISIONS))
        dst = BACKUP_DIR / f"{DECISIONS.stem}_{time.strftime('%Y%m%dT%H%M%S')}_{n:04d}.jsonl"
        shutil.copy2(DECISIONS, dst)
        print(f"  (backup of {n} decisions: {dst})")
    except OSError as e:  # a failed backup must never lose or block a decision
        print(f"  (backup FAILED: {e})")


def append(rec: dict) -> None:
    with open(DECISIONS, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    if sum(1 for _ in open(DECISIONS)) % BACKUP_EVERY == 0:
        backup_decisions()


def injection_block(c: dict) -> list[str]:
    """What was injected into the run, for induced candidates."""
    ind = c.get("induced") or {}
    out = []
    if "correction" in ind:
        x = ind["correction"]
        out.append(f"INJECTION (mind-change, {x.get('type')}, timing {x.get('timing')}): the user first wanted {x.get('old_desc', x.get('old'))!r}, then corrected to {x.get('new_desc', x.get('new'))!r}")
        out.append(f"   the correction as the user actually said it: {clip(x.get('correction_text', ''), 260)}")
    if "fault" in ind:
        x = ind["fault"]
        f = x.get("faulted_call", {})
        out.append(f"INJECTION ({x.get('mode', 'hard')} fault): {f.get('name')}({clip(json.dumps(f.get('arguments'), ensure_ascii=False), 160)}) was not executed; it returned: "
                   f"{f.get('message')!r}" + (f" -> {clip(str(f.get('response_head', '')), 200)}" if f.get("response_head") else ""))
        out.append(f"   how the agent reacted: {x.get('how', x.get('end'))}; text between the result and its next call: {clip(x.get('between_text', '') or '(none)', 200)}")
    if "propagation" in ind:
        x = ind["propagation"]
        out.append(f"INJECTION (value changed, {x.get('type')}): {x.get('reveal_call')} reported {x.get('old')!r} was replaced by {x.get('new')!r}; the environment accepts the new id and answers 'not found' to the old one")
    return out


def show_item(it: dict, pos: int, total: int, full: bool) -> None:
    w = textwrap.TextWrapper(width=110, initial_indent="    ", subsequent_indent="    ")
    print("\n" + "=" * 112)
    if it["kind"] == "error":
        c = it["candidate"]
        win, turns = c["window"], c["turns"]
        print(f"[queue {it['queue']}  {pos}/{total}]  ORIGIN: {it['origin'].upper()}  {c['domain']} task {c['task_id']}  split={c['split']}  {c['source_model']}")
        print(f"proposed class {c['proposed_class']} ({CLASS_NAMES[c['proposed_class']]}) by {c['proposal_source']}" + (f"   auto_status: {c.get('auto_status')}" if c.get("auto_status") else ""))
        print(w.fill("evidence: " + c["evidence"]))
        if c.get("auto_flags"):
            print(w.fill("FLAGS (check these): " + ", ".join(c["auto_flags"])))
        for line in injection_block(c):
            print(w.fill(line))
        if c.get("second_opinion"):
            so = c["second_opinion"]
            print(w.fill(f"second labeler ({so['model']} {so['effort']}): call {so['call']}, class {so['class']} - {clip(so.get('reason') or '', 200)}"))
        rs = c.get("reward_summary") or {}
        if rs:
            missed = "; ".join(f"{m['name']}({clip(json.dumps(m['arguments']), 80)})" for m in rs.get("missed_actions", [])[:3])
            print(w.fill(f"checker: reward {rs.get('reward')}, db_match {rs.get('db_match')}, missed actions: {missed or 'none'}"))
        mark_label = "the flagged call"
    else:
        p = it["positive"]
        win, turns = it["window"], it["turns"]
        print(f"[queue {it['queue']}  {pos}/{total}]  ORIGIN: {it['origin'].upper()}  {p['domain']} task {p['task_id']}  split={p['split']}  CORRECT-EXAMPLE CHECK")
        print(w.fill(f"auto label: {p.get('auto_label')}. {p.get('reason', '')}"))
        for line in injection_block(p):
            print(w.fill(line))
        print(w.fill("Question: is the marked call a CORRECT next step here (right tool, right arguments, nothing ignored, nothing stale)?"))
        mark_label = "the call to judge"
    print(f"--- turns {win['start']}-{win['end'] - 1} of {win['total_turns']} ({mark_label} is marked >>>) ---")
    for i, t in enumerate(turns):
        mark = ">>>" if i == win["flagged_offset"] else "   "
        role = t["role"]
        if role in ("assistant_tool_call", "user_tool_call"):
            body = f"{role.replace('_tool_call', '').upper()} CALL {t['name']}({clip(json.dumps(t['arguments'], ensure_ascii=False), 600)})"
        elif role == "tool":
            body = ("TOOL ERROR: " if t.get("error") else "TOOL: ") + (t["content"] if full else clip(t["content"], 500))
        else:
            body = f"{role.upper()}: " + (t["content"] if full else clip(t["content"], 700))
        print(f"{mark} {win['start'] + i:3d} {body}")


RULE_BOX = """  RULE (docs/labeling_guide.md)  Class 7 = POLICY VIOLATION: the call performs or starts an action the domain policy does not allow at that point.
    reasons, first that applies:  w parallel_write | n no_confirmation | e ineligible_state | m missing_step | p precondition_error | o other (+ note)
    a read-only call (get_ search_ find_ list_ calculate; telecom check_ can_ run_speed_test) is NEVER Class 7 unless the policy forbids that lookup
    parallel read-only lookups are CORRECT with tag parallel_lookup (deliberately not tau2's one-call-at-a-time rule); a write beside an open call is wrong"""
KEYS5 = {"n": "no_confirmation", "e": "ineligible_state", "m": "missing_step", "p": "precondition_error", "w": "parallel_write", "o": "other"}


def show_q5(it: dict, pos: int, total: int, full: bool) -> None:
    w = textwrap.TextWrapper(width=110, initial_indent="    ", subsequent_indent="        ")
    src = {**it["source_item"], "queue": 5}
    show_item(src, pos, total, full)
    od, pre, call = it["orig_decision"], it["prefill"], it["call"]
    print("\n" + (it.get("rule_box") or RULE_BOX))
    print(f"  policy for this item: {POLICY_FILE[it['domain']]}   (domain: {it['domain']})")
    print(f"  YOUR ORIGINAL DECISION: {'the test_real set (phase-4 hand label)' if it['orig_queue'] == 0 else 'queue ' + str(it['orig_queue'])}, {od['decision'].upper()}" + (f" class {od['human_class']}" if od.get("human_class") else "")
          + (f"  note: {od['human_note']}" if od.get("human_note") else "") + f"   [{od.get('time', '')}]")
    print(w.fill(f"CALL: {call['name']}({clip(json.dumps(call['arguments'], ensure_ascii=False), 300)})   [{'READ-ONLY' if call['read_only'] else 'WRITE'}]"))
    if pre["verdict"] == "class7":
        fill = f"Class 7 / {pre['reason']}"
    elif pre["verdict"] == "correct":
        fill = "correct" + (f" + tag {pre['tag']}" if pre["tag"] else "")
    elif pre["verdict"] == "class4":
        fill = "Class 4 (repeated call)"
    else:
        fill = "none"
    print(w.fill(f"PRE-FILL (heuristic, structure only): {fill}"))
    for s in pre["signals"]:
        print(w.fill("- " + s))


def review_queue5(its: list, dec: dict, args) -> str:
    todo = [i for i in its if i["item_id"] not in dec or (args.include_skipped and dec[i["item_id"]]["decision"] == "skip")]
    qn = its[0]["queue"]
    print(f"\nqueue {qn}: {len(its) - len(todo)}/{len(its)} done, {len(todo)} to go (about {sum(MINUTES[i['kind']] for i in todo):.0f} min)")
    n, last, history = 0, "", []
    while n < len(todo):
        it = todo[n]
        full, note, redraw = False, "", True
        pre = it["prefill"]
        while True:
            if redraw:
                show_q5(it, len(its) - len(todo) + n + 1, len(its), full)
                if last:
                    print(f"\n(last: {last})")
                redraw = False
            ans = ""
            try:
                if it["queue"] == 6:
                    ans = input("\n[y] correct (not an error)   [1] Class 1 (a different line than the target)   [2-6] other class   [s] skip (original stays)   [u] undo   [t] note   [f] full text   [g] guide   [q] quit > ").strip().lower()
                else:
                    ans = input("\n[k] keep pre-fill   Class 7 reason: [n]o_confirmation [e]ineligible_state [m]issing_step [p]recondition_error [w]parallel_write [o]ther\n"
                                "[y] correct   [l] correct + parallel_lookup   [1-6] other class   [s] skip (original stays)   [u] undo   [t] note   [f] full text   [g] guide   [q] quit > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nstopped; decisions so far are saved.")
                return "quit"
            cmd, _, arg = ans.partition(" ")
            if cmd[:1] == "c" and len(cmd) == 2 and cmd[1].isdigit():
                cmd = cmd[1]
            if cmd == "q":
                print("stopped; decisions so far are saved.")
                return "quit"
            if cmd == "f":
                full, redraw = not full, True
                continue
            if cmd == "g":
                print(open(HERE.parent / "docs" / "labeling_guide.md").read())
                continue
            if cmd == "u":
                if not history:
                    print("  nothing to undo in this session")
                    continue
                n = history.pop()
                last = "undo: deciding this item again (the new answer replaces the old one)"
                break
            if cmd == "t":
                note = input("note: ").strip()
                print("  note kept; now give the decision")
                continue
            rec = None
            if cmd == "k":
                if pre["verdict"] == "class7" and pre["reason"]:
                    rec = {"decision": "wrong", "human_label": "incorrect", "human_class": 7, "class7_reason": pre["reason"]}
                elif pre["verdict"] == "correct":
                    rec = {"decision": "correct", "human_label": "correct", "human_class": None, "tag": pre["tag"]}
                elif pre["verdict"] == "class4":
                    rec = {"decision": "wrong", "human_label": "incorrect", "human_class": 4}
                else:
                    print("  there is no pre-fill for this item: choose a reason or a verdict")
                    continue
            elif cmd in KEYS5:
                if it["call"]["read_only"]:
                    print("  this call is read-only: it is not Class 7 unless the policy forbids this lookup. Confirm with [x] (then give the reason in a note), or choose [y] / [l] / a class.")
                    continue
                if cmd == "o" and not note:
                    note = input("note (required for 'other'): ").strip()
                    if not note:
                        print("  'other' needs a note; nothing saved")
                        continue
                rec = {"decision": "wrong", "human_label": "incorrect", "human_class": 7, "class7_reason": KEYS5[cmd]}
            elif cmd == "x" and it["call"]["read_only"]:
                note = note or input("note (which policy rule forbids this lookup?): ").strip()
                if not note:
                    print("  nothing saved")
                    continue
                rec = {"decision": "wrong", "human_label": "incorrect", "human_class": 7, "class7_reason": "other"}
            elif cmd == "y":
                rec = {"decision": "correct", "human_label": "correct", "human_class": None, "tag": None}
            elif cmd == "l":
                rec = {"decision": "correct", "human_label": "correct", "human_class": None, "tag": "parallel_lookup"}
            elif cmd.isdigit() and cmd in "123456" and len(cmd) == 1:
                rec = {"decision": "wrong", "human_label": "incorrect", "human_class": int(cmd)}
            elif cmd == "7":
                print("  Class 7 needs a reason: press n, e, m, p, w or o")
                continue
            elif cmd == "s":
                rec = {"decision": "skip", "human_label": "skip", "human_class": None}
            if rec is None:
                print("  not understood (nothing saved); try k, n, e, m, p, w, o, y, l, 1-6, s, u")
                continue
            same = rec["decision"] == "wrong" and pre["verdict"] == "class7" and rec.get("class7_reason") == pre["reason"] or \
                rec["decision"] == "correct" and pre["verdict"] == "correct" and rec.get("tag") == pre["tag"]
            rec = {"item_id": it["item_id"], "queue": it["queue"], "origin": it["origin"], "kind": "recheck", "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "human_note": note,
                   "supersedes": it["orig_item_id"], "orig_queue": it["orig_queue"], "orig_kind": it["orig_kind"], "proposed_class": None,
                   "prefill": {k: pre[k] for k in ("verdict", "reason", "tag")}, "prefill_kept": bool(same), **rec}
            append(rec)
            dec[it["item_id"]] = rec
            what = {"wrong": "wrong" + (f", class {rec['human_class']}" if rec["human_class"] else "") + (f" / {rec['class7_reason']}" if rec.get("class7_reason") else ""),
                    "correct": "correct" + (f" + {rec['tag']}" if rec.get("tag") else ""), "skip": "skipped (original decision stays)"}[rec["decision"]]
            last = f"saved: item {len(its) - len(todo) + n + 1}/{len(its)} {what}"
            history.append(n)
            n += 1
            break
    print(f"\nqueue {qn} finished.")
    return "done"


def cmd_review(args):
    items = rj(QUEUE)
    if not items:
        sys.exit("no queue yet: run `build` first")
    dec = latest()
    order = [args.queue] if args.queue else [1, 2, 3, 4, 5, 6]
    for q in order:
        its = [i for i in items if i["queue"] == q]
        if q in (5, 6):
            if its and review_queue5(its, dec, args) == "quit":
                return
            continue
        todo = [i for i in its if i["item_id"] not in dec or (args.include_skipped and dec[i["item_id"]]["decision"] == "skip")]
        print(f"\nqueue {q}: {len(its) - len(todo)}/{len(its)} done, {len(todo)} to go (about {sum(MINUTES[i['kind']] for i in todo):.0f} min)")
        legend = "  ".join(f"{k}={CLASS_NAMES[k]}" for k in range(1, 8))
        n, last, history = 0, "", []        # n: position in todo; history: positions decided this session (for undo)
        while n < len(todo):
            it = todo[n]
            full, note, redraw = False, "", True
            while True:
                if redraw:
                    show_item(it, len(its) - len(todo) + n + 1, len(its), full)
                    if last:
                        print(f"\n(last: {last})")
                    redraw = False
                if it["kind"] == "error":
                    prompt = (f"\nclasses: {legend}\n"
                              f"[a] accept proposed ({it['candidate']['proposed_class']})   [1-7] wrong, but it is THAT class (e.g. 7 or c7)   [r] reject   "
                              f"[s] skip   [u] undo previous   [n] note   [f] full text   [q] quit > ")
                else:
                    prompt = ("\n[y] correct   [w] wrong (class optional: w or w3 or 3)   [s] skip   [u] undo previous   [o] note   [f] full text   [q] quit > ")
                try:
                    ans = input(prompt).strip().lower()
                except (EOFError, KeyboardInterrupt):
                    print("\nstopped; decisions so far are saved.")
                    return
                cmd, _, arg = ans.partition(" ")
                arg = arg.strip()
                if cmd[:1] in "cw" and len(cmd) == 2 and cmd[1].isdigit():      # "c7" / "w3" without the space
                    cmd, arg = cmd[0], cmd[1]
                elif cmd.isdigit() and len(cmd) == 1:                           # a bare digit: the class
                    cmd, arg = ("c" if it["kind"] == "error" else "w"), cmd
                if cmd == "q":
                    print("stopped; decisions so far are saved.")
                    return
                if cmd == "f":
                    full, redraw = not full, True
                    continue
                if cmd == "u":
                    if not history:
                        print("  nothing to undo in this session")
                        continue
                    n = history.pop()
                    last = "undo: deciding this item again (the new answer replaces the old one)"
                    break
                if cmd in ("n", "o") and not (it["kind"] == "correct" and cmd == "n"):
                    note = input("note: ").strip()
                    print("  note kept; now give the decision")
                    continue
                base = {"item_id": it["item_id"], "queue": q, "origin": it["origin"], "kind": it["kind"], "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "human_note": note,
                        "proposed_class": it["candidate"]["proposed_class"] if it["kind"] == "error" else None}
                rec, cls = None, (int(arg) if arg.isdigit() and 1 <= int(arg) <= 7 else None)
                if it["kind"] == "error":
                    if cmd == "a":
                        rec = {"decision": "accept", "human_label": "accept", "human_class": base["proposed_class"]}
                    elif cmd == "c" and cls:
                        if cls == base["proposed_class"]:
                            rec = {"decision": "accept", "human_label": "accept", "human_class": cls}
                        else:
                            rec = {"decision": "change_class", "human_label": "accept", "human_class": cls, "human_note": (note + " " if note else "") + f"[class changed {base['proposed_class']} -> {cls}]"}
                    elif cmd == "r":
                        rec = {"decision": "reject", "human_label": "reject", "human_class": None}
                else:
                    if cmd == "y":
                        rec = {"decision": "correct", "human_label": "correct", "human_class": None}
                    elif cmd == "w":
                        rec = {"decision": "wrong", "human_label": "incorrect", "human_class": cls}
                if cmd == "s":
                    rec = {"decision": "skip", "human_label": "skip", "human_class": None}
                if rec is None:
                    print("  not understood (nothing saved); try again, e.g. a, 7, c7, r, s, u")
                    continue
                rec = {**base, **rec}
                append(rec)
                dec[it["item_id"]] = rec
                what = {"accept": "accepted", "change_class": "accepted as class " + str(rec["human_class"]), "reject": "rejected", "skip": "skipped",
                        "correct": "marked correct", "wrong": "marked wrong" + (f" (class {rec['human_class']})" if rec["human_class"] else "")}[rec["decision"]]
                last = f"saved: item {len(its) - len(todo) + n + 1}/{len(its)} {what}"
                history.append(n)
                n += 1
                break
    print("\nall queues in the selection are finished.")
    cmd_stats(args)


def _tally(rows, dec):
    """rows: queue items (kind error). Returns counts of human outcomes against the labeler's proposal."""
    t = Counter()
    for i in rows:
        d = dec.get(i["item_id"])
        if not d or d["decision"] == "skip":
            continue
        t["reviewed"] += 1
        if d["human_label"] == "reject":
            t["rejected"] += 1
        elif d["human_class"] == i["candidate"]["proposed_class"]:
            t["same_call_same_class"] += 1
        else:
            t["same_call_other_class"] += 1
    return t


def _line(label, t):
    n = t["reviewed"]
    p = f"{t['same_call_same_class'] / n:4.0%}" if n else "  - "
    return f"   {label:44s} reviewed {n:3d} | confirmed same call+class {t['same_call_same_class']:3d} ({p}) | same call, other class {t['same_call_other_class']:3d} | rejected {t['rejected']:3d}"


def precision_report(items, dec):
    """After human review: how often each labeler's proposals were confirmed (test split = queue 1; the train audit = queue 3)."""
    err = [i for i in items if i["kind"] == "error"]
    print("\n=== labeler precision against the human decisions (no decision yet = not counted) ===")
    print("'same call' is implied by an accept: the reviewer judges the flagged call; the call is never changed.")
    for title, q in (("TEST split (queue 1, every candidate)", 1), ("TEST split (queue 4, new candidates)", 4), ("TRAIN audit (queue 3, only auto_accepted candidates)", 3)):
        rows = [i for i in err if i["queue"] == q]
        if not any(i["item_id"] in dec for i in rows):
            print(f"\n{title}: nothing reviewed yet")
            continue
        print(f"\n{title}")
        print("  first labeler, by class (natural = gpt-5-mini or the Class 5 heuristic, induced = the auto-label)")
        rec = {}
        for o in ("natural", "induced"):
            for k in range(1, 8):
                sub = [i for i in rows if i["origin"] == o and i["candidate"]["proposed_class"] == k]
                t = _tally(sub, dec)
                if not t["reviewed"]:
                    continue
                print(_line(f"{o} class {k} {CLASS_NAMES[k]}", t))
                rec[(o, k)] = t
        print("  first labeler, by source")
        for src in sorted({i["candidate"].get("proposal_source") if i["origin"] == "natural" else i["candidate"].get("source") for i in rows}):
            sub = [i for i in rows if (i["candidate"].get("proposal_source") if i["origin"] == "natural" else i["candidate"].get("source")) == src]
            t = _tally(sub, dec)
            if t["reviewed"]:
                print(_line(f"{src}", t))
        if q == 1:
            print("  recommendation per class (first labeler's confirmed precision >= 70% with >= 8 reviewed -> auto-accept train labels)")
            for (o, k), t in sorted(rec.items(), key=lambda kv: (kv[0][1], kv[0][0])):
                ok = t["reviewed"] >= 8 and t["same_call_same_class"] / t["reviewed"] >= 0.70
                why = f"{t['same_call_same_class']}/{t['reviewed']} confirmed" + ("" if t["reviewed"] >= 8 else ", fewer than 8 reviewed")
                print(f"   {o:8s} class {k} {CLASS_NAMES[k]:22s} -> {'auto-accept train labels' if ok else 'human review or drop'}   ({why})")
            missing = [k for k in range(1, 8) if not any(kk[1] == k for kk in rec)]
            if missing:
                print(f"   no reviewed test candidates for class {missing}: -> human review or drop")
    # the second labeler: only natural TRAIN candidates were sent to it, and only the ones it agreed on were drawn into queue 3
    sec = [i for i in err if i["origin"] == "natural" and i["candidate"].get("second_opinion") and i["item_id"] in dec]
    print("\nsecond labeler (gpt-5-nano, medium): where it has labels AND a human decision")
    if not sec:
        print("   none yet. It labeled only natural train candidates, and queue 3 draws only the ones it agreed on, so this can measure the precision of")
        print("   agreement, never of its disagreements (those 163 'auto_rejected' rows are not in any queue).")
    else:
        for k in range(1, 8):
            sub = [i for i in sec if i["candidate"]["second_opinion"]["class"] == k]
            t = Counter()
            for i in sub:
                d = dec[i["item_id"]]
                if d["decision"] == "skip":
                    continue
                t["reviewed"] += 1
                t["rejected" if d["human_label"] == "reject" else ("same_call_same_class" if d["human_class"] == k else "same_call_other_class")] += 1
            if t["reviewed"]:
                print(_line(f"second labeler said class {k} {CLASS_NAMES[k]}", t))
    # induced auto-labels (all splits that were reviewed)
    print("\ninduced auto-labels by source (queue 1 test + queue 3 train audit together)")
    for src in sorted({i["candidate"].get("source") for i in err if i["origin"] == "induced"}):
        t = _tally([i for i in err if i["origin"] == "induced" and i["candidate"].get("source") == src], dec)
        if t["reviewed"]:
            print(_line(src, t))
    q2 = [i for i in items if i["queue"] == 2 and i["item_id"] in dec and dec[i["item_id"]]["decision"] != "skip"]
    for o in ("natural", "induced"):
        sub = [i for i in q2 if i["origin"] == o]
        if sub:
            ok = sum(dec[i["item_id"]]["human_label"] == "correct" for i in sub)
            print(f"correct examples ({o}, queue 2): {ok}/{len(sub)} confirmed correct ({ok / len(sub):.0%})")


def cmd_stats(args):
    items, dec = rj(QUEUE), latest()
    for q in (1, 2, 3, 4, 5, 6):
        its = [i for i in items if i["queue"] == q]
        if q in (4, 5, 6) and not its:
            continue
        c = Counter(dec[i["item_id"]]["decision"] for i in its if i["item_id"] in dec)
        print(f"queue {q}: {sum(c.values())}/{len(its)} decided {dict(c)}")
    precision_report([i for i in items if i["queue"] != 5], load_effective(DECISIONS))


def cmd_merge(args):
    dec = load_effective(DECISIONS)
    for o, s in SRC.items():
        rows = rj(s["cands"])
        n = 0
        for c in rows:
            for q in (1, 3, 4):
                d = dec.get(f"{o}::{c['candidate_id']}")
                if d and d["queue"] == q and d["decision"] != "skip":
                    c.update({"human_label": d["human_label"], "human_class": d["human_class"], "human_note": d["human_note"], "human_queue": q, "human_origin": o})
                    if d.get("class7_reason"):
                        c["class7_reason"] = d["class7_reason"]
                    n += 1
        if rows:
            tmp = s["cands"].with_suffix(".tmp")
            tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
            os.replace(tmp, s["cands"])
        pos = rj(s["pos"])
        m = 0
        for p in pos:
            d = dec.get(f"{o}::{p['positive_id']}")
            if d and d["queue"] == 2 and d["decision"] != "skip":
                p["status"] = "accepted" if d["human_label"] == "correct" else "rejected"
                p["human_class"], p["human_note"], p["human_origin"] = d["human_class"], d["human_note"], o
                p["class7_reason"], p["tag"] = d.get("class7_reason"), d.get("tag")
                m += 1
        if pos:
            tmp = s["pos"].with_suffix(".tmp")
            tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in pos))
            os.replace(tmp, s["pos"])
        print(f"{o}: {n} candidate decisions and {m} correct-example decisions merged")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "add-queue4", "add-queue5", "add-queue5-testreal", "add-queue6", "sizes", "review", "stats", "merge"])
    ap.add_argument("--queue", type=int, default=0)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--include-skipped", action="store_true")
    ap.add_argument("--decisions", default="", help="decisions file (default review_queue_decisions.jsonl); useful for a dry run")
    a = ap.parse_args()
    if a.decisions:
        DECISIONS = Path(a.decisions)
    {"build": cmd_build, "add-queue4": cmd_add_queue4, "add-queue5": cmd_add_queue5, "add-queue5-testreal": cmd_add_queue5_testreal, "add-queue6": cmd_add_queue6, "sizes": cmd_sizes, "review": cmd_review, "stats": cmd_stats, "merge": cmd_merge}[a.stage](a)
