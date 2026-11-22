"""Independent checks of the written final dataset (data/derived/final/*.jsonl), run after `build_final_dataset.py --write` (D-022 / D-023).

    .venv/bin/python data/check_final_dataset.py [--dir data/derived/final]

Prints counts by split x class x origin x domain, the correct-example count per split, a leakage check (no (domain, task_id) in train/val and in
test or test_synth), a duplicate check within and across splits (trajectory ids, identical content, identical run + call index), and the length-only
and source-only baseline AUROC on validation and test (trained on the training split, same features as the notebooks' final report).
Exits with status 1 when a check fails.
"""

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
SPLITS = ("train", "val", "test", "test_synth")


def load(d: Path):
    return {s: [json.loads(l) for l in open(d / f"{s}.jsonl") if l.strip()] for s in SPLITS}


def cls(r):
    return f"class {r['error_class']}" if r["label"] == "incorrect" else "correct"


def counts(rows):
    print("=== counts by split x class x origin x domain ===")
    for s in SPLITS:
        c = Counter((cls(r), r["origin"], r["domain"]) for r in rows[s])
        cols = sorted({(o, d) for _, o, d in c})
        print(f"\n{s}: {len(rows[s])} rows ({sum(r['label'] == 'incorrect' for r in rows[s])} incorrect, {sum(r['label'] == 'correct' for r in rows[s])} correct)")
        print(f"   {'':10s}" + "".join(f"{o[:4]}:{d[:5]:>6s}" for o, d in cols) + f"{'total':>8s}")
        for k in sorted({k for k, _, _ in c}, key=lambda x: (x == "correct", x)):
            vals = [c[(k, o, d)] for o, d in cols]
            print(f"   {k:10s}" + "".join(f"{v:11d}" if v else f"{'.':>11s}" for v in vals) + f"{sum(vals):8d}")
        tot = [sum(c[(k, o, d)] for k in {k for k, _, _ in c}) for o, d in cols]
        print(f"   {'total':10s}" + "".join(f"{v:11d}" for v in tot) + f"{sum(tot):8d}")
    print("\n=== correct examples per split ===")
    for s in SPLITS:
        print(f"   {s:11s} {sum(r['label'] == 'correct' for r in rows[s]):5d} correct of {len(rows[s]):5d}")
    t = [r for r in rows["test"] if r["label"] == "incorrect"]
    print("\ntest errors per class, natural vs induced:")
    for k in range(1, 8):
        print(f"   class {k}: natural {sum(r['error_class'] == k and r['origin'] == 'natural' for r in t):3d}  induced {sum(r['error_class'] == k and r['origin'] == 'induced' for r in t):3d}")


def groups(rs):
    return {(r["domain"], str(r["task_id"])) for r in rs}


def leakage(rows):
    print("\n=== leakage: a (domain, task_id) must not be on both sides ===")
    train, val = groups(rows["train"]), groups(rows["val"])
    ok = True
    for name, a, b in (("train / val", rows["train"], rows["val"]), ("train+val / test", rows["train"] + rows["val"], rows["test"]),
                       ("train+val / test_synth", rows["train"] + rows["val"], rows["test_synth"]), ("test / test_synth", rows["test"], rows["test_synth"])):
        both = groups(a) & groups(b)
        real = {g for g in both if g[0] != "critictool"}
        print(f"   {name:24s}: {len(both)} shared groups" + (f" ({len(both - real)} are critictool)" if both - real else ""))
        if name != "train / val" and real:
            ok = False
            print("      FAIL", sorted(real)[:6])
        if name == "train / val" and real:
            ok = False
            print("      FAIL", sorted(real)[:6])
    return ok


def content_key(r):
    return hashlib.sha1(json.dumps([r["turns"], r["candidate_tool_call"]], sort_keys=True, default=str).encode()).hexdigest()


def duplicates(rows):
    print("\n=== duplicates within and across splits ===")
    ok = True
    allr = [(s, r) for s in SPLITS for r in rows[s]]
    ids = Counter(r["trajectory_id"] for _, r in allr)
    d = {k: v for k, v in ids.items() if v > 1}
    print(f"   trajectory ids: {len(allr)} rows, {len(d)} ids used more than once")
    ok &= not d
    by = defaultdict(list)
    for s, r in allr:
        by[content_key(r)].append((s, r))
    dup = {k: v for k, v in by.items() if len(v) > 1}
    within = {k: v for k, v in dup.items() if len({s for s, _ in v}) == 1}
    across = {k: v for k, v in dup.items() if len({s for s, _ in v}) > 1}
    print(f"   identical content (turns + candidate call): {len(dup)} groups; within one split {len(within)}, across splits {len(across)}")
    for tag, g in (("within", within), ("across", across)):
        for k, v in list(g.items())[:8]:
            print(f"      {tag}: " + " | ".join(f"{s} {r['trajectory_id'][-45:]} ({cls(r)})" for s, r in v))
    conflict = [v for v in dup.values() if len({cls(r) for _, r in v}) > 1]
    print(f"   of those, with conflicting labels: {len(conflict)}")
    ok &= not dup
    rk = Counter((r.get("run_key"), r.get("call_index")) for s, r in allr if r.get("run_key") and r.get("call_index") is not None)
    rd = {k: v for k, v in rk.items() if v > 1}
    print(f"   same run + call index: {len(rd)} repeated")
    ok &= not rd
    return ok


def length(r):
    return float(np.log1p(len(json.dumps(r["turns"])) + len(json.dumps(r["candidate_tool_call"]))))


def src(r):
    s = str(r.get("source", ""))
    return "induced" if s.startswith("induced") else "critictool" if s == "critictool" else "bfcl" if r["domain"] == "bfcl" else "tau2"


def baselines(rows):
    print("\n=== baselines (trained on the training split) ===")
    names = ["tau2", "bfcl", "critictool", "induced"]
    y = np.array([r["label"] == "incorrect" for r in rows["train"]]).astype(int)
    out = {}
    for name, feat in (("length only", lambda r: [length(r)]), ("source only", lambda r: [float(src(r) == n) for n in names])):
        clf = LogisticRegression(max_iter=1000).fit(np.array([feat(r) for r in rows["train"]]), y)
        for s in ("val", "test", "test_synth"):
            yy = np.array([r["label"] == "incorrect" for r in rows[s]]).astype(int)
            sc = clf.predict_proba(np.array([feat(r) for r in rows[s]]))[:, 1]
            out[(name, s)] = roc_auc_score(yy, sc) if yy.min() != yy.max() else float("nan")
    for name in ("length only", "source only"):
        print(f"   {name:12s}: validation AUROC {out[(name, 'val')]:.3f}   test AUROC {out[(name, 'test')]:.3f}   test_synth AUROC {out[(name, 'test_synth')]:.3f}")
    print("   (0.5 = no information; a model has to beat these on the same rows)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(HERE / "derived" / "final"))
    a = ap.parse_args()
    rows = load(Path(a.dir))
    counts(rows)
    ok = leakage(rows)
    ok = duplicates(rows) and ok
    baselines(rows)
    print("\nALL CHECKS PASSED" if ok else "\nCHECKS FAILED")
    sys.exit(0 if ok else 1)
