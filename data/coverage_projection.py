"""Real test errors per class from the human decisions, and the projection once the next review queue is done (D-019 / D-022). No API.

Real test errors per class = the incorrect rows of test_real.jsonl + the test-split candidates a human accepted or reclassified (queues 1 and 4 of
data/derived/review_queue_decisions.jsonl) + the queue-2 items a human marked wrong with a class (found_in queue2), split into natural (unperturbed
benchmark runs) and induced (real agent behaviour in a perturbed run). Queue-5 decisions (the Class 7 / read-only re-review) supersede the earlier ones
(data/review_decisions.py); a queue-2 error that is in queue 5 and not yet re-reviewed is held out of the count and shown apart.

Projection = what is confirmed today
           + candidates still waiting for a human (skipped items and candidates in queue 4 that are not decided yet)
           + planned new candidates (--planned origin:class:count, repeatable),
each waiting or planned candidate being turned into classes with the transition rates measured on the decisions already made for the same
(origin, proposed class): share confirmed as that class, share moved to each other class, share rejected. A pair with fewer than 5 decided
items falls back to all origins of that proposed class, then to 65% confirmed. The planned induced Class 2 and 6 candidates come from the
stricter checks of data/strict_check.py, whose precision is only known in-sample (D-019), so they are also shown with the old measured rates.

    .venv/bin/python data/coverage_projection.py [--min 10] [--planned induced:2:25 --planned induced:6:15 --planned natural:3:20]
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from review_decisions import load_effective  # noqa: E402

DER = Path(__file__).resolve().parent / "derived"
NAMES = {1: "wrong arg", 2: "stale state", 3: "wrong tool", 4: "repeated call", 5: "failure to propagate", 6: "ignored error", 7: "policy violation"}
POOLS = {"natural": "real_candidates_review.jsonl", "induced": "induced_candidates_review.jsonl"}


def rj(p):
    return [json.loads(l) for l in open(p) if l.strip()] if p.exists() else []


def decided_items():
    """(origin, proposed class, outcome) for every non-skipped test-candidate decision of queues 1 and 4; outcome = final class or 'reject'."""
    dec = load_effective(DER / "review_queue_decisions.jsonl")
    done, waiting = [], Counter()
    for o, f in POOLS.items():
        for c in rj(DER / f):
            if c["split"] != "test":
                continue
            d = dec.get(f"{o}::{c['candidate_id']}")
            p = int(c["proposed_class"])
            if d and d["queue"] in (1, 4) and d["decision"] != "skip":
                done.append((o, p, "reject" if d["human_label"] == "reject" else int(d["human_class"])))
            else:
                waiting[(o, p)] += 1
    return done, waiting


def queue2_summary():
    """Queue-2 decisions: errors per (origin, class) confirmed, errors held for the queue-5 re-review, correct examples per origin."""
    items = rj(DER / "review_queue.jsonl")
    dec = load_effective(DER / "review_queue_decisions.jsonl")
    q5 = {i["orig_item_id"] for i in items if i["queue"] == 5}
    held_ids = q5 - {k for k, d in dec.items() if "q5" in d.get("superseded_chain", [])}
    err, held, ok = Counter(), Counter(), Counter()
    for i in items:
        if i["queue"] != 2:
            continue
        d = dec.get(i["item_id"])
        if not d or d["decision"] == "skip":
            continue
        if d["human_label"] == "correct":
            ok[i["origin"]] += 1
        elif d["human_class"]:
            (held if i["item_id"] in held_ids else err)[(i["origin"], int(d["human_class"]))] += 1
    return err, held, ok, len(held_ids)


def transition(done, o, p):
    rows = [x for x in done if x[0] == o and x[1] == p]
    src = f"{o} class {p}"
    if len(rows) < 5:
        rows = [x for x in done if x[1] == p]
        src = f"all origins, class {p}"
    if len(rows) < 5:
        return {p: 0.65, "reject": 0.35}, "default 65%"
    c = Counter(x[2] for x in rows)
    return {k: v / len(rows) for k, v in c.items()}, f"{src}, {len(rows)} decided"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min", type=int, default=10)
    ap.add_argument("--planned", action="append", default=[], help="origin:class:count of new candidates, e.g. induced:2:25")
    a = ap.parse_args()
    dec_tr = load_effective(DER / "review_queue_decisions.jsonl")
    test_rows = []                      # test_real after the queue-5 verdicts on its Class 7 rows: (label, class)
    for r in rj(DER / "test_real.jsonl"):
        d = dec_tr.get(f"test_real::{r['trajectory_id']}")
        if d and d.get("superseded_by"):
            test_rows.append(("correct", None) if d["human_label"] == "correct" else ("incorrect", int(d["human_class"])))
        else:
            test_rows.append((r["label"], int(r["error_class"]) if r["label"] == "incorrect" and r.get("error_class") else None))
    test_real = Counter(c for lab, c in test_rows if lab == "incorrect" and c)
    done, waiting = decided_items()
    conf = {"natural": Counter(), "induced": Counter()}
    for o, p, out in done:
        if out != "reject":
            conf[o][out] += 1
    q2err, q2held, q2ok, n_held = queue2_summary()
    q1 = {k: Counter() for k in ("natural", "induced")}
    q4 = {k: Counter() for k in ("natural", "induced")}
    dec_all = load_effective(DER / "review_queue_decisions.jsonl")
    for o, f in POOLS.items():
        for c in rj(DER / f):
            d = dec_all.get(f"{o}::{c['candidate_id']}")
            if c["split"] == "test" and d and d["queue"] in (1, 4) and d["decision"] != "skip" and d["human_label"] != "reject":
                (q1 if d["queue"] == 1 else q4)[o][int(d["human_class"])] += 1
    c2 = {o: Counter({k: v for (oo, k), v in q2err.items() if oo == o}) for o in ("natural", "induced")}
    h2 = Counter()
    for (o, k), v in q2held.items():
        h2[k] += v
    tot = {k: test_real[k] + sum(x[o][k] for x in (q1, c2, q4) for o in ("natural", "induced")) for k in range(1, 8)}
    nat = {k: test_real[k] + q1["natural"][k] + c2["natural"][k] + q4["natural"][k] for k in range(1, 8)}
    ind = {k: q1["induced"][k] + c2["induced"][k] + q4["induced"][k] for k in range(1, 8)}
    print("REAL TEST ERRORS PER CLASS from the human decisions (queue-5 verdicts applied over the earlier ones)\n")
    print(f"{'class':24s} {'test_real':>9s} {'queue 1':>8s} {'queue 2':>8s} {'queue 4':>8s} {'TOTAL':>6s}   {'natural':>7s} {'induced':>7s}   {'queue-2 held for queue 5':>25s}")
    for k in range(1, 8):
        print(f"{k} {NAMES[k]:22s} {test_real[k]:9d} {sum(q1[o][k] for o in q1):8d} {sum(c2[o][k] for o in c2):8d} {sum(q4[o][k] for o in q4):8d} {tot[k]:6d}   {nat[k]:7d} {ind[k]:7d}   {h2[k]:25d}"
              + (f"   <-- below {a.min}" if tot[k] < a.min else ""))
    print(f"{'all':24s} {sum(test_real.values()):9d} {sum(sum(q1[o].values()) for o in q1):8d} {sum(sum(c2[o].values()) for o in c2):8d} {sum(sum(q4[o].values()) for o in q4):8d} {sum(tot.values()):6d}   {sum(nat.values()):7d} {sum(ind.values()):7d}   {sum(h2.values()):25d}")
    tr_ok = sum(1 for lab, c in test_rows if lab == "correct")
    print(f"\nCORRECT EXAMPLES IN TEST: {tr_ok + sum(q2ok.values())} = test_real {tr_ok} + human-confirmed queue 2 {q2ok['natural'] + q2ok['induced']} (natural {q2ok['natural']}, induced {q2ok['induced']}); "
          f"only human-confirmed correct calls are used (D-022)")
    print(f"   {sum(h2.values())} queue-2 errors are in queue 5 and not yet re-reviewed: held out of the table; a queue-5 'correct' verdict on a read-only call adds a correct example")
    conf = {"natural": Counter({k: nat[k] - test_real[k] for k in nat}), "induced": Counter(ind)}
    print(f"\n(decided candidates: {len(done)}; waiting for a decision: {sum(waiting.values())} {dict(sorted(waiting.items()))})")

    planned = []
    for s in a.planned:
        o, k, n = s.split(":")
        planned.append((o, int(k), int(n)))

    def project(extra_rates=None):
        proj = Counter({k: float(tot[k]) for k in range(1, 8)})
        byo = {"natural": Counter({k: float(nat[k]) for k in range(1, 8)}), "induced": Counter({k: float(ind[k]) for k in range(1, 8)})}
        notes = []
        for (o, p), n in list(waiting.items()) + [((o, k), n) for o, k, n in planned]:
            if n == 0:
                continue
            tr, src = transition(done, o, p) if not (extra_rates and (o, p) in extra_rates) else (extra_rates[(o, p)], "strict-check precision (in-sample)")
            for k, share in tr.items():
                if k != "reject":
                    proj[k] += n * share
                    byo[o][k] += n * share
            notes.append((o, p, n, src))
        project.by_origin = byo
        return proj, notes

    base, notes = project()
    base_by = project.by_origin
    print("\nTRANSITION RATES used (measured on the decisions already made):")
    for o, p, n, src in sorted(set(notes)):
        tr, _ = transition(done, o, p)
        print(f"   {o:8s} proposed class {p}: {n:3d} candidates x {{{', '.join(f'{k}: {v:.0%}' for k, v in sorted(tr.items(), key=lambda kv: str(kv[0])))}}}   ({src})")
    cols = [("today", {k: float(tot[k]) for k in range(1, 8)}), ("projected", base)]
    pending = {k for k, n in waiting.items() if n} | {(o, k) for o, k, n in planned if n}
    strict = {k: v for k, v in {("induced", 2): {2: 0.75, "reject": 0.25}, ("induced", 6): {6: 0.57, "reject": 0.43}}.items() if k in pending}
    if strict:  # the induced Class 2 / 6 candidates now come from the stricter checks, whose precision (3/4 and 4/7) was measured in-sample only (D-019)
        hi, _ = project(strict)
        cols.append(("projected, strict-check rates", hi))
    print("\nPROJECTION once the waiting and planned candidates are reviewed" + (f" (planned: {', '.join(f'{o} class {k} x{n}' for o, k, n in planned)})" if planned else ""))
    print(f"{'class':24s}" + "".join(f"{name:>30s}" for name, _ in cols))
    for k in range(1, 8):
        print(f"{k} {NAMES[k]:22s}" + "".join(f"{c[k]:30.1f}" + ("" if c[k] >= a.min else "") for _, c in cols) + ("   <-- below %d" % a.min if base[k] < a.min else ""))
    print("\nQUEUE 4 PROJECTED, natural vs induced (the 54 queue-4 candidates and the 17 skipped queue-1 ones turned into classes with the measured rates; confirmed errors from queue 2 that wait for queue 5 are not included)")
    print(f"{'class':24s} {'natural now':>12s} {'natural proj.':>14s} {'induced now':>12s} {'induced proj.':>14s} {'total now':>10s} {'total proj.':>12s}")
    for k in range(1, 8):
        print(f"{k} {NAMES[k]:22s} {nat[k]:12d} {base_by['natural'][k]:14.1f} {ind[k]:12d} {base_by['induced'][k]:14.1f} {tot[k]:10d} {base[k]:12.1f}")
    print("\nnote: induced candidates are real agent behaviour in a perturbed environment; natural ones come from unperturbed benchmark runs. Report them separately when quoting recall.")


main()
