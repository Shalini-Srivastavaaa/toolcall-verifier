"""How the ML classifier and the rule-based Class 7 checker combine at inference.

LEGACY (first-round experiment on the 102-row `test_real`): the rule checker found 0 of 8 real Class 7 errors (D-014) and is not part of the final model; the final model learns policy
violations from the policy text in its input (D-021, D-024). The Colab paths below are those of the first round.

    final prediction = incorrect  if  the ML probability >= threshold  OR  the Class 7 checker flags

ML scores come from a saved predictions file (one JSON line per test_real row with `trajectory_id`
and `p_incorrect`, written by the training notebook's evaluation cell); the model is not re-run.

REQUIRES the Phase 5 predictions file from Drive at checkpoints/phase5/test_real_predictions.jsonl
(MyDrive/toolcall-verifier/checkpoints/phase5/). It is not in the repo (checkpoints/ is gitignored): copy
it there manually, or pass --predictions PATH.
The rule checker runs on the rows in data/derived/test_real.jsonl.

AUROC is ML-only: the rule output is a binary flag, not a score, so the combined predictor has no
AUROC of its own and the ML AUROC is printed once. F1, recall and false-flag rate are computed
for ML alone, the rule alone and the combination, all at the same threshold.

    python eval/combined_predict.py [--predictions PATH] [--threshold 0.5]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from class7_checker import ROOT, check_class7, load_jsonl  # noqa: E402

DEFAULT_PREDICTIONS = ROOT / "checkpoints" / "phase5" / "test_real_predictions.jsonl"
CLASSES = (1, 3, 4, 5, 7)


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    tp, fp, fn = int((y & pred).sum()), int((~y & pred).sum()), int((y & ~pred).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"flagged": int(pred.sum()), "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0, "false_flag": fp / int((~y).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    pred_path = Path(args.predictions)
    if not pred_path.exists():
        sys.exit(f"ML predictions not found: {pred_path}\nCopy the notebook's test_real_predictions.jsonl "
                 f"there (it is saved to MyDrive/toolcall-verifier/checkpoints/phase5/) or pass --predictions.")
    scores = {r["trajectory_id"]: r["p_incorrect"] for r in load_jsonl(pred_path)}
    rows = load_jsonl(ROOT / "data" / "derived" / "test_real.jsonl")
    missing = [r["trajectory_id"] for r in rows if r["trajectory_id"] not in scores]
    if missing or len(scores) != len(rows):
        sys.exit(f"{pred_path} does not match test_real.jsonl ({len(missing)} rows missing, {len(scores)} scored)")

    y = np.array([r["label"] == "incorrect" for r in rows])
    cls = np.array([r["error_class"] or 0 for r in rows])
    p = np.array([scores[r["trajectory_id"]] for r in rows])
    ml = p >= args.threshold
    rule = np.array([check_class7(r)["flagged"] for r in rows])
    both = ml | rule

    print(f"ML predictions: {pred_path}")
    print(f"test_real: {len(rows)} rows, {int(y.sum())} incorrect; threshold {args.threshold}")
    print(f"ML AUROC (unchanged by the rule layer): {roc_auc_score(y, p):.3f}\n")

    cols = [("ML alone", ml), ("Rule alone", rule), ("Combined", both)]
    print(f"{'':26s}" + "".join(f"{n:>12s}" for n, _ in cols))
    ms = [metrics(y, v) for _, v in cols]
    for key, label in [("flagged", "rows flagged"), ("precision", "precision"), ("recall", "recall (all errors)"),
                       ("f1", "F1"), ("false_flag", "false-flag rate (correct)")]:
        fmt = (lambda v: f"{v:>12d}") if key == "flagged" else (lambda v: f"{v:>12.3f}")
        print(f"{label:26s}" + "".join(fmt(m[key]) for m in ms))
    for c in CLASSES:
        m = cls == c
        print(f"{f'Class {c} recall (n={int(m.sum())})':26s}" + "".join(f"{f'{int(v[m].sum())}/{int(m.sum())}':>12s}" for _, v in cols))

    gained = int((both & ~ml & y).sum())
    added_fp = int((both & ~ml & ~y).sum())
    print(f"\nRule layer adds {gained} true catch(es) and {added_fp} extra false flag(s) over ML alone.")
    c7 = cls == 7
    if not rule[c7].any():
        print("The rule checker flags none of the Class 7 rows, so Class 7 recall is unchanged.")


if __name__ == "__main__":
    main()
