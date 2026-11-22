"""Natural Class 5 (failure to propagate) candidates from tau2-bench runs that were not localized yet (D-019 follow-up). No API.

Class 5: a value returned by an earlier tool result that a later, related call should use, but does not. The checker's own
missed actions tell us what the call should have used, so no model is involved:

  for an agent call C of tool T in a failed run, take a checker-missed action M of the same tool T. If C's arguments differ from M's
  in a scalar string field (path p, agent value va, expected value vg, both >= 3 chars and id-like) and

     tier A  va appeared in an EARLIER tool result and vg in a LATER tool result, both before C
             (the agent read a value, a later result gave another one, and the call kept the first one)
     tier B  vg is in one of the 3 most recent tool results before C, and va appears only in older context
             (the user's words or the agent's own earlier calls), not in any tool result

  it is a Class 5 candidate. Tier C (vg somewhere in an earlier result, va in no result: the agent just picked a wrong value) is
  Class 1 territory and is not proposed. One candidate per run (the earliest C), tier A before B.

Candidates are appended to data/derived/real_candidates_review.jsonl with proposal_source "heuristic_class5" (existing rows are untouched;
rerunning replaces only the rows this script wrote) and origin "natural". Linked pending positives go to real_positive_candidates.jsonl.

    python data/mine_class5.py
"""

import json
import re
from collections import Counter
from pathlib import Path

from mine_real_errors import (CLASS_NAMES, FREE_TEXT_TOOLS, PROMPT_VERSION, RESULT_CAP, agent_calls, cache_rows, is_error, load_index,
                              render_full, runs_by_file)

HERE = Path(__file__).resolve().parent
DER = HERE / "derived"
REVIEW = DER / "real_candidates_review.jsonl"
POSITIVES = DER / "real_positive_candidates.jsonl"
REPORT = DER / "real_class5_report.json"
IDLIKE = re.compile(r"(\d|_)")
MAX_PER_SPLIT = {"test": 80, "train": 80}


def flat(obj, prefix=""):
    """{path: scalar string} of a call's arguments (lists get index paths)."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flat(v, f"{prefix}{k}."))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(flat(v, f"{prefix}{i}."))
    elif isinstance(obj, str):
        out[prefix.rstrip(".")] = obj
    return out


def idlike(v: str) -> bool:
    return len(v) >= 3 and bool(IDLIKE.search(v)) and len(v) <= 60


def analyse(messages: list[dict], missed: list[dict]):
    """Best (tier, distance, call, path, va, vg, result_idx_vg) for this run, or None."""
    calls = agent_calls(messages)
    results = [(i, str(m.get("content", ""))) for i, m in enumerate(messages) if m.get("role") == "tool" and not is_error(m)]
    best = None
    for c in calls:
        if c["name"] in FREE_TEXT_TOOLS:
            continue
        gold_args = [flat(m["arguments"]) for m in missed if m["name"] == c["name"]]
        if not gold_args:
            continue
        mine = flat(c["arguments"])
        prior = [(i, t) for i, t in results if i < c["msg"]]
        recent = [t for _, t in prior[-3:]]
        older_ctx = " ".join(str(m.get("content", "")) for m in messages[: c["msg"]] if m.get("role") in ("user", "assistant")) + " " + \
            " ".join(json.dumps(x["arguments"]) for x in calls if x["k"] < c["k"])
        for g in gold_args:
            diffs = [(p, mine[p], g[p]) for p in mine if p in g and mine[p] != g[p]]
            if not diffs or len(diffs) > 2:
                continue
            for p, va, vg in diffs:
                if not (idlike(va) and idlike(vg)):
                    continue
                va_res = [i for i, t in prior if va in t]
                vg_res = [i for i, t in prior if vg in t]
                if not vg_res:
                    continue
                if va_res and max(vg_res) > max(va_res):
                    tier = "A"
                elif not va_res and any(vg in t for t in recent) and va in older_ctx:
                    tier = "B"
                else:
                    continue
                cand = (tier, c["msg"] - max(vg_res), c, p, va, vg, max(vg_res))
                if best is None or (cand[0], cand[1]) < (best[0], best[1]):
                    best = cand
        if best is not None:
            break  # the earliest call with a hit
    return best


def main():
    cache = cache_rows()
    idx = [r for r in load_index() if (r["run_key"], PROMPT_VERSION) not in cache]
    by_key = {r["run_key"]: r for r in idx}
    stats = Counter()
    found = []
    for s in runs_by_file(set(by_key)):
        k = f"{s['file']}::{s['sim_id']}"
        ix = by_key[k]
        stats["unlocalized_runs_scanned"] += 1
        missed = ix["reward_summary"].get("missed_actions") or []
        if not missed:
            stats["no_missed_actions"] += 1
            continue
        hit = analyse(s["messages"], missed)
        if hit is None:
            continue
        stats[f"hit_tier_{hit[0]}"] += 1
        found.append((s, ix, hit))
    # best first, capped per split
    found.sort(key=lambda x: (x[2][0], x[2][1], x[0]["domain"], x[0]["sim_id"]))
    kept, per_split = [], Counter()
    for s, ix, hit in found:
        if per_split[ix["split"]] < MAX_PER_SPLIT[ix["split"]]:
            per_split[ix["split"]] += 1
            kept.append((s, ix, hit))
    review_old = [json.loads(l) for l in open(REVIEW) if l.strip()]
    review_old = [r for r in review_old if r.get("proposal_source") != "heuristic_class5"]
    pos_old = [json.loads(l) for l in open(POSITIVES) if l.strip()]
    pos_old = [p for p in pos_old if not str(p.get("linked_candidate_id", "")).endswith("::h5") and p.get("origin_tag") != "heuristic_class5"]
    new_review, new_pos = [], []
    for s, ix, (tier, dist, c, path, va, vg, vg_idx) in kept:
        k = f"{s['file']}::{s['sim_id']}"
        calls = agent_calls(s["messages"])
        turns, call_turn = render_full(s["messages"])
        ci = c["k"]
        t = call_turn[ci]
        lo, hi = max(0, t - 20), min(len(turns), t + 5)
        window = []
        for x in turns[lo:hi]:
            x = dict(x)
            if x.get("role") == "tool" and len(x["content"]) > RESULT_CAP:
                x["content"] = x["content"][:RESULT_CAP] + f"…[truncated, {len(x['content'])} chars]"
            window.append(x)
        cid = f"{k}::C{ci}"
        why = ("an earlier result showed " + repr(va) + " and a later result (message %d) gave " % vg_idx + repr(vg)) if tier == "A" else \
              ("a recent result (message %d) gave " % vg_idx + repr(vg) + ", while " + repr(va) + " only appears in older context")
        new_review.append({
            "candidate_id": cid, "run_key": k, "domain": s["domain"], "task_id": s["task_id"], "source_model": s["source_model"], "split": ix["split"], "origin": "natural",
            "proposal_source": "heuristic_class5", "heuristic_tier": tier, "proposed_class": 5, "proposed_class_name": CLASS_NAMES[5],
            "evidence": f"[heuristic class 5, tier {tier}] call C{ci} {c['name']} uses {path} = {va!r} but the checker expected {vg!r}; {why}.",
            "flagged_call": {"call_index": ci, "turn_index": t, "name": c["name"], "arguments": c["arguments"]},
            "window": {"start": lo, "end": hi, "flagged_offset": t - lo, "total_turns": len(turns)}, "turns": window,
            "heuristic_class6": ix["heuristic_class6"], "heuristic_class2": ix["heuristic_class2"], "reward_summary": ix["reward_summary"],
            "human_label": "", "human_class": "", "human_note": ""})
        err_ids = {m.get("id") for m in s["messages"] if is_error(m)}
        for q in calls[:ci]:
            if q["name"] in FREE_TEXT_TOOLS or q["id"] in err_ids:
                continue
            new_pos.append({"positive_id": f"{k}::C{q['k']}", "run_key": k, "linked_candidate_id": cid, "call_index": q["k"], "turn_index": call_turn[q["k"]],
                            "name": q["name"], "arguments": q["arguments"], "domain": s["domain"], "task_id": s["task_id"], "split": ix["split"], "status": "pending",
                            "origin_tag": "heuristic_class5"})
    existing_ids = {r["candidate_id"] for r in review_old}
    new_review = [r for r in new_review if r["candidate_id"] not in existing_ids]  # a run already proposed some other way keeps that proposal
    keep_ids = {r["candidate_id"] for r in new_review}
    new_pos = [p for p in new_pos if p["linked_candidate_id"] in keep_ids]
    with open(REVIEW, "w") as f:
        for r in review_old + new_review:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(POSITIVES, "w") as f:
        for p in pos_old + new_pos:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    rep = {"unlocalized_runs_scanned": stats["unlocalized_runs_scanned"], "runs_without_missed_actions": stats["no_missed_actions"],
           "hits_before_cap": {"A": stats["hit_tier_A"], "B": stats["hit_tier_B"]}, "added_candidates": len(new_review),
           "by_split": dict(Counter(r["split"] for r in new_review)), "by_domain": dict(Counter(r["domain"] for r in new_review)),
           "by_tier": dict(Counter(r["heuristic_tier"] for r in new_review)), "by_split_and_tier": dict(Counter(f"{r['split']}/{r['heuristic_tier']}" for r in new_review)),
           "by_agent_model": dict(Counter(r["source_model"] for r in new_review)), "added_pending_positives": len(new_pos), "cap_per_split": MAX_PER_SPLIT}
    REPORT.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
