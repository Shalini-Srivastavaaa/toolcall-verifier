"""Training composition before and after the D-026 data fix: real-origin (natural, induced) versus synthetic or external, per class, for errors and for correct rows.

    .venv/bin/python3 eval/train_composition.py [--before DIR] [--after DIR]      # DIR holds train.jsonl and val.jsonl (default after: data/derived/final)
"""

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAMES = {1: "wrong arg", 2: "stale state", 3: "wrong tool", 4: "repeated call", 5: "failure to propagate", 6: "ignored error", 7: "policy violation"}


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def kind(r):
    s, d = str(r.get("source", "")), r["domain"]
    if s.startswith("induced"):
        return "real: induced"
    if s in ("tau2_real_failed_run", "tau2_real_correct_call"):
        return "real: natural"
    if s == "critictool":
        return "CriticTool"
    if s == "synthetic_stateflip":
        return "state flips"
    if s in ("tau2_gold", "bfcl_gold"):
        return "gold (successful runs)"
    return "BFCL synthetic" if d == "bfcl" else "tau2 synthetic"


def table(name, rows):
    errs = [r for r in rows if r["label"] == "incorrect"]
    cor = [r for r in rows if r["label"] == "correct"]
    real = lambda r: kind(r).startswith("real")
    print(f"\n{name}: {len(rows)} rows = {len(errs)} errors + {len(cor)} correct")
    print(f"   {'class':24s} {'real natural':>13s} {'real induced':>13s} {'synthetic/ext':>14s} {'total':>7s}   real share")
    for k in range(1, 8):
        c = Counter(kind(r) for r in errs if r["error_class"] == k)
        n, i = c["real: natural"], c["real: induced"]
        tot = sum(c.values())
        print(f"   {k} {NAMES[k]:22s} {n:13d} {i:13d} {tot - n - i:14d} {tot:7d}   {(n + i) / max(tot, 1):6.1%}")
    c = Counter(kind(r) for r in errs)
    tot = len(errs)
    print(f"   {'ALL ERRORS':24s} {c['real: natural']:13d} {c['real: induced']:13d} {tot - c['real: natural'] - c['real: induced']:14d} {tot:7d}   {(c['real: natural'] + c['real: induced']) / tot:6.1%}")
    c = Counter(kind(r) for r in cor)
    print(f"   correct rows by kind: " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())) + f"  | real-origin correct {sum(real(r) for r in cor)}")
    print(f"   real-origin rows (errors + their correct rows): {sum(real(r) for r in rows)} of {len(rows)} ({sum(real(r) for r in rows) / len(rows):.1%})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default="")
    ap.add_argument("--after", default=str(ROOT / "data" / "derived" / "final"))
    a = ap.parse_args()
    for label, d in (("BEFORE (final_v1 data)", a.before), ("AFTER (D-026)", a.after)):
        if not d:
            continue
        print("=" * 100 + f"\n{label}: {d}\n" + "=" * 100)
        table("train", load(Path(d) / "train.jsonl"))
        table("validation", load(Path(d) / "val.jsonl"))
