"""Score an OLD model's predictions on the new test set, separating rows its training data could have seen (D-023).

The v2 checkpoint of today was trained on the old 7,580 rows, which include 1,309 rows of 37 tasks that are now TEST tasks (D-018).
Rows of those tasks in the new test set are therefore not a clean test for that model. This script reads the model's
`test_predictions.jsonl` (written by the notebook in DATASET = "final" mode, with row ids) and prints AUROC with a 95% bootstrap
interval on: all rows, the CLEAN rows (tasks the old model never saw: the 74 tasks of test_real and every new test task that is not in the
overlap list of data/derived/real_split.json) and the OVERLAPPING rows. Only the clean number isolates what the new data adds.

    python eval/score_old_model_on_new_test.py PREDICTIONS.jsonl [--test data/derived/final/test.jsonl]
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

DER = Path(__file__).resolve().parent.parent / "data" / "derived"


def load(p):
    return [json.loads(line) for line in open(p) if line.strip()]


def boot(y, p, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            v.append(roc_auc_score(y[i], p[i]))
    return np.percentile(v, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions")
    ap.add_argument("--test", default=str(DER / "final" / "test.jsonl"))
    a = ap.parse_args()
    ov = json.load(open(DER / "real_split.json"))["overlap_with_current_training"]
    seen = {d: {str(t) for t in v["new_test_tasks_that_appear_in_current_training_files"]} for d, v in ov.items()}
    pred = {r["trajectory_id"]: r["p_incorrect"] for r in load(a.predictions)}
    rows = [r for r in load(a.test) if r["trajectory_id"] in pred]
    print(f"{len(rows)} test rows have a prediction")
    for name, keep in (("all rows", lambda r: True),
                       ("CLEAN rows (tasks the old model never saw)", lambda r: str(r["task_id"]) not in seen.get(r["domain"], ())),
                       ("OVERLAPPING rows (task was in the old training data)", lambda r: str(r["task_id"]) in seen.get(r["domain"], ()))):
        part = [r for r in rows if keep(r)]
        y = np.array([r["label"] == "incorrect" for r in part]).astype(int)
        s = np.array([pred[r["trajectory_id"]] for r in part])
        if len(y) == 0 or y.min() == y.max():
            print(f"  {name:55s} {len(y):4d} rows, {int(y.sum()):3d} incorrect: AUROC not defined")
            continue
        lo, hi = boot(y, s)
        print(f"  {name:55s} {len(y):4d} rows, {int(y.sum()):3d} incorrect: AUROC {roc_auc_score(y, s):.3f}  95% CI {lo:.3f}-{hi:.3f}")
    print("\nnote: the synthetic held-out set (test_synth) is NOT a held-out result for the old model: its task groups were part of the old training data.")


if __name__ == "__main__":
    main()
