"""Real-origin training errors per class and domain for dataset versions (D-028): natural and induced errors from real agent runs, their correct rows, and the synthetic errors for contrast.

    .venv/bin/python3 eval/real_errors_by_domain.py final final_v3a final_v3b        # folders under data/derived/; the first is 'before', the last 'after'
"""

import json
import sys
from collections import Counter
from pathlib import Path

DER = Path(__file__).resolve().parent.parent / "data" / "derived"
NAMES = {1: "wrong arg", 2: "stale state", 3: "wrong tool", 4: "repeated call", 5: "failure to propagate", 6: "ignored error", 7: "policy violation"}
DOMS = ("airline", "retail", "telecom")


def load(d):
    return [json.loads(l) for l in open(DER / d / "train.jsonl") if l.strip()]


def main():
    dirs = sys.argv[1:] or ["final", "final_v3a", "final_v3b"]
    data = {d: load(d) for d in dirs}
    for d in dirs:
        rows = data[d]
        real_e = [r for r in rows if r["label"] == "incorrect" and r.get("real_origin")]
        real_c = [r for r in rows if r["label"] == "correct" and r.get("real_origin")]
        print(f"\n{d}: {len(rows)} training rows; real-origin errors {len(real_e)} (natural {sum(r['origin'] == 'natural' for r in real_e)}, induced {sum(r['origin'] == 'induced' for r in real_e)}), real-origin correct rows {len(real_c)}")
        print(f"   {'class':24s} " + "".join(f"{x:>9s}" for x in DOMS) + f"{'real':>7s} {'synthetic/ext':>14s}")
        for k in range(1, 8):
            c = Counter(r["domain"] for r in real_e if r["error_class"] == k)
            syn = sum(1 for r in rows if r["label"] == "incorrect" and r["error_class"] == k and not r.get("real_origin"))
            print(f"   {k} {NAMES[k]:22s} " + "".join(f"{c[x]:9d}" for x in DOMS) + f"{sum(c.values()):7d} {syn:14d}")
        c = Counter(r["domain"] for r in real_e)
        cc = Counter(r["domain"] for r in real_c)
        print(f"   {'all real errors':24s} " + "".join(f"{c[x]:9d}" for x in DOMS) + f"{len(real_e):7d}")
        print(f"   {'real correct rows':24s} " + "".join(f"{cc[x]:9d}" for x in DOMS) + f"{len(real_c):7d}")
    a, b = data[dirs[0]], data[dirs[-1]]
    print(f"\nCLASS 4 AND TELECOM, {dirs[0]} -> {dirs[-1]}:")
    for lab, f in (("real Class 4 errors", lambda r: r["label"] == "incorrect" and r["error_class"] == 4 and r.get("real_origin")),
                   ("real Class 4 errors in telecom", lambda r: r["label"] == "incorrect" and r["error_class"] == 4 and r.get("real_origin") and r["domain"] == "telecom"),
                   ("real telecom errors, all classes", lambda r: r["label"] == "incorrect" and r.get("real_origin") and r["domain"] == "telecom"),
                   ("real telecom correct rows", lambda r: r["label"] == "correct" and r.get("real_origin") and r["domain"] == "telecom"),
                   ("real Class 7 errors", lambda r: r["label"] == "incorrect" and r["error_class"] == 7 and r.get("real_origin")),
                   ("real Class 1 errors", lambda r: r["label"] == "incorrect" and r["error_class"] == 1 and r.get("real_origin"))):
        print(f"   {lab:36s} {sum(bool(f(r)) for r in a):5d} -> {sum(bool(f(r)) for r in b):5d}")


if __name__ == "__main__":
    main()
