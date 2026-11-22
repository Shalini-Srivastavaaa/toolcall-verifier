"""Review real-error candidates mined from tau2-bench's failed runs (D-018).

Shows one candidate at a time: the turns around the flagged call, the proposed class and the evidence
(from gpt-5-mini, or from a heuristic when no LLM label exists). You record a decision; it is appended to
data/derived/real_review_decisions.jsonl and fsynced before the next candidate appears, so interrupting
(Ctrl+C, closing the terminal) never loses work and rerunning resumes at the first undecided candidate.

    python data/review_real.py                    # review (resumes where you stopped)
    python data/review_real.py --include-skipped  # also revisit the ones you skipped
    python data/review_real.py --stats            # progress only
    python data/review_real.py --merge            # write human_label / human_class / human_note into the candidates file
                                                  # and flip the linked positives from "pending" to accepted / rejected

At the prompt:
    a          accept: the flagged call is wrong and the proposed class is right
    c <1-7>    accept, but with a different class (for example "c 3"); the note records the change
    r          reject: the flagged call is fine (a benchmark artifact or a reasonable call)
    s          skip for now
    n          add a one-line note to this decision, then answer again
    f          show the full text of every tool result in the window
    q          quit

Classes: 1 wrong arg, 2 stale state, 3 wrong tool, 4 repeated call, 5 failure to propagate, 6 ignored error, 7 policy violation.
Nothing here writes a training or test file.
"""

import argparse
import json
import os
import sys
import textwrap
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DER = HERE / "derived"
CLASS_NAMES = {1: "wrong arg", 2: "stale state", 3: "wrong tool", 4: "repeated call", 5: "failure to propagate",
               6: "ignored error", 7: "policy violation"}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def latest_decisions(path: Path) -> dict:
    return {d["candidate_id"]: d for d in load_jsonl(path)}  # the last line for a candidate wins


def append_decision(path: Path, rec: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def clip(s: str, n: int) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def show(c: dict, pos: int, total: int, full: bool) -> None:
    w = textwrap.TextWrapper(width=110, initial_indent="    ", subsequent_indent="    ")
    print("\n" + "=" * 112)
    print(f"[{pos}/{total}]  {c['domain']}  task {c['task_id']}  {c['source_model']}  split={c['split']}")
    print(f"proposed class {c['proposed_class']} ({CLASS_NAMES[c['proposed_class']]}) by {c['proposal_source']}")
    print(w.fill("evidence: " + c["evidence"]))
    rs = c.get("reward_summary") or {}
    missed = "; ".join(f"{m['name']}({clip(json.dumps(m['arguments']), 80)})" for m in rs.get("missed_actions", [])[:3])
    print(w.fill(f"checker: reward {rs.get('reward')}, db_match {rs.get('db_match')}, missed actions: {missed or 'none'}"
                 + (f"; unmet: {clip('; '.join(rs.get('nl_unmet', [])), 160)}" if rs.get("nl_unmet") else "")))
    win = c["window"]
    print(f"--- turns {win['start']}-{win['end'] - 1} of {win['total_turns']} (the flagged call is marked >>>) ---")
    for i, t in enumerate(c["turns"]):
        mark = ">>>" if i == win["flagged_offset"] else "   "
        role = t["role"]
        if role in ("assistant_tool_call", "user_tool_call"):
            body = f"{role.replace('_tool_call', '').upper()} CALL {t['name']}({clip(json.dumps(t['arguments'], ensure_ascii=False), 600)})"
        elif role == "tool":
            body = ("TOOL ERROR: " if t.get("error") else "TOOL: ") + (t["content"] if full else clip(t["content"], 500))
        else:
            body = f"{role.upper()}: " + (t["content"] if full else clip(t["content"], 700))
        print(f"{mark} {win['start'] + i:3d} {body}")


def progress(cands: list[dict], decisions: dict) -> str:
    n = {"accept": 0, "reject": 0, "skip": 0}
    for c in cands:
        d = decisions.get(c["candidate_id"])
        if d:
            n[d["human_label"]] += 1
    return f"{len(cands)} candidates: {n['accept']} accepted, {n['reject']} rejected, {n['skip']} skipped, {len(cands) - sum(n.values())} undecided"


def merge(cands_path: Path, decisions_path: Path, positives_path: Path) -> None:
    cands, decisions = load_jsonl(cands_path), latest_decisions(decisions_path)
    for c in cands:
        d = decisions.get(c["candidate_id"])
        if d:
            c["human_label"], c["human_class"], c["human_note"] = d["human_label"], d["human_class"], d["human_note"]
    tmp = cands_path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cands))
    os.replace(tmp, cands_path)
    status = {"accept": "accepted", "reject": "rejected"}
    positives = load_jsonl(positives_path)
    for p in positives:
        d = decisions.get(p["linked_candidate_id"])
        p["status"] = status.get(d["human_label"], "pending") if d else "pending"
    if positives:
        tmp = positives_path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in positives))
        os.replace(tmp, positives_path)
    from collections import Counter
    print("merged.", progress(cands, decisions), "| positives:", dict(Counter(p["status"] for p in positives)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=str(DER / "real_candidates_review.jsonl"))
    ap.add_argument("--decisions", default=str(DER / "real_review_decisions.jsonl"))
    ap.add_argument("--positives", default=str(DER / "real_positive_candidates.jsonl"))
    ap.add_argument("--include-skipped", action="store_true")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    cpath, dpath, ppath = Path(args.candidates), Path(args.decisions), Path(args.positives)

    if args.merge:
        return merge(cpath, dpath, ppath)
    cands = load_jsonl(cpath)
    decisions = latest_decisions(dpath)
    print(progress(cands, decisions))
    if args.stats:
        return

    def pending(c):
        d = decisions.get(c["candidate_id"])
        return d is None or (args.include_skipped and d["human_label"] == "skip")

    queue = [c for c in cands if pending(c)]
    if not queue:
        print("nothing left to review.")
        return
    for pos, c in enumerate(queue, 1):
        full, note = False, ""
        while True:
            show(c, len(cands) - len(queue) + pos, len(cands), full)
            try:
                ans = input("\n[a]ccept  [c N] change class  [r]eject  [s]kip  [n]ote  [f]ull  [q]uit > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nstopped; decisions so far are saved.")
                return
            cmd, _, arg = ans.partition(" ")
            if cmd == "q":
                print("stopped; decisions so far are saved.")
                return
            if cmd == "f":
                full = not full
                continue
            if cmd == "n":
                note = input("note: ").strip()
                continue
            rec = {"candidate_id": c["candidate_id"], "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "proposed_class": c["proposed_class"],
                   "human_note": note}
            if cmd == "a":
                rec |= {"human_label": "accept", "human_class": c["proposed_class"], "decision": "accept"}
            elif cmd == "c" and arg.strip().isdigit() and 1 <= int(arg) <= 7:
                new = int(arg)
                rec |= {"human_label": "accept", "human_class": new, "decision": "change_class",
                        "human_note": (note + " " if note else "") + f"[class changed {c['proposed_class']} -> {new}]"}
            elif cmd == "r":
                rec |= {"human_label": "reject", "human_class": None, "decision": "reject"}
            elif cmd == "s":
                rec |= {"human_label": "skip", "human_class": None, "decision": "skip"}
            else:
                print("  not understood: use a, c <1-7>, r, s, n, f or q")
                continue
            append_decision(dpath, rec)
            decisions[c["candidate_id"]] = rec
            break
    print("\nqueue finished.", progress(cands, decisions))


if __name__ == "__main__":
    main()
