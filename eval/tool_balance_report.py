"""Per-tool balance of the training split and the tool-only baseline, for two or more dataset versions (D-028).

    .venv/bin/python3 eval/tool_balance_report.py final final_v3a [final_v3b]       # folders under data/derived/

Per tool: errors against correct rows in the training split, before (the first folder) and after (the last). Tool-only baseline: predict the label from the tool name alone
(the training error rate of the tool, smoothed toward the overall rate with a prior of 2 rows; an unseen tool gets the overall rate) and report AUROC on validation and on test.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

DER = Path(__file__).resolve().parent.parent / "data" / "derived"


def load(d, s):
    return [json.loads(l) for l in open(DER / d / f"{s}.jsonl") if l.strip()]


def counts(train):
    E, C = Counter(), Counter()
    for r in train:
        (E if r["label"] == "incorrect" else C)[r["candidate_tool_call"]["name"]] += 1
    return E, C


def tool_only(train, rows, prior=2.0):
    E, C = counts(train)
    base = sum(E.values()) / (sum(E.values()) + sum(C.values()))
    y = np.array([r["label"] == "incorrect" for r in rows]).astype(int)
    s = np.array([(E[r["candidate_tool_call"]["name"]] + prior * base) / (E[r["candidate_tool_call"]["name"]] + C[r["candidate_tool_call"]["name"]] + prior) for r in rows])
    return roc_auc_score(y, s)


def main():
    dirs = sys.argv[1:] or ["final", "final_v3a"]
    tr = {d: load(d, "train") for d in dirs}
    Eb, Cb = counts(tr[dirs[0]])
    Ea, Ca = counts(tr[dirs[-1]])
    tools = sorted(set(Eb) | set(Cb) | set(Ea) | set(Ca), key=lambda t: -(Eb[t] - Cb[t]))
    print(f"per-tool balance of the training split, {dirs[0]} -> {dirs[-1]} (tools with more errors than correct rows before, or whose counts changed):")
    print(f"   {'tool':36s} {'errors':>6s} {'correct before':>14s} {'correct after':>13s} {'errors after':>12s}  ratio correct/errors before -> after")
    for t in tools:
        if Eb[t] > Cb[t] or (Ca[t], Ea[t]) != (Cb[t], Eb[t]):
            print(f"   {t:36s} {Eb[t]:6d} {Cb[t]:14d} {Ca[t]:13d} {Ea[t]:12d}  {Cb[t] / max(Eb[t], 1):.2f} -> {Ca[t] / max(Ea[t], 1):.2f}")
    left = [t for t in tools if Ea[t] > Ca[t]]
    print(f"\ntools with more errors than correct rows after: {len(left)} ({sum(Ea[t] - Ca[t] for t in left)} rows of deficit); before: {sum(Eb[t] > Cb[t] for t in tools)} tools ({sum(Eb[t] - Cb[t] for t in tools if Eb[t] > Cb[t])} rows)")
    print(f"   after, the largest deficits: " + ", ".join(f"{t} {Ea[t]}/{Ca[t]}" for t in sorted(left, key=lambda t: -(Ea[t] - Ca[t]))[:8]))
    print("\nTOOL-ONLY BASELINE (the label predicted from the tool name alone; AUROC):")
    print("   (the tool-name statistics come from the training split of the version; 'train, in-sample' is how well they separate the training rows themselves, i.e. the strength of the shortcut inside training)")
    print(f"   {'version':12s} {'train rows':>10s} {'train, in-sample':>17s} {'validation':>11s} {'test':>7s} {'test, natural rows':>19s} {'test, tau2 domains':>19s}")
    for d in dirs:
        va, te = load(d, "val"), load(d, "test")
        nat = [r for r in te if r["origin"] == "natural"]
        print(f"   {d:12s} {len(tr[d]):10d} {tool_only(tr[d], tr[d]):17.3f} {tool_only(tr[d], va):11.3f} {tool_only(tr[d], te):7.3f} {tool_only(tr[d], nat):19.3f} {tool_only(tr[d], [r for r in te if r['domain'] in ('airline', 'retail', 'telecom')]):19.3f}")
    te = load(dirs[-1], "test")
    print("\ntest rows of the tools that were balanced (correct / errors in the test set): " + ", ".join(f"{t} {sum(r['label'] == 'correct' for r in te if r['candidate_tool_call']['name'] == t)}/{sum(r['label'] == 'incorrect' for r in te if r['candidate_tool_call']['name'] == t)}" for t in ("cancel_reservation", "exchange_delivered_order_items", "modify_pending_order_items", "modify_pending_order_payment", "return_delivered_order_items", "resume_line")))


if __name__ == "__main__":
    main()
