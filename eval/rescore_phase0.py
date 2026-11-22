"""Re-score the Phase 0 judge outputs against the hand labels in test_real.jsonl (D-006 correction).

D-006 scored the GPT-5 judges against the Phase 0 heuristic labels, which D-010 showed were only
24.5% precise. test_real.jsonl holds the hand labels for 102 of those candidates (joined on
trajectory_id), so this recomputes every number against them.

Score for AUROC: pred_confidence if pred_verdict == "incorrect", else 1 - pred_confidence.
A row whose verdict could not be parsed (1 of 102 for nano) gets the neutral score 0.5 so all 102 rows
stay in the AUROC; dropping it instead moves nano from 0.709 to 0.714. For the threshold metrics it counts
as not flagged ("did the judge flag it" is false).

    python eval/rescore_phase0.py [--n-boot 1000] [--seed 0] [--json results/phase0_rescored.json]
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "phase0_raw"
MODELS = {"GPT-5-nano": "openai-nano", "GPT-5-mini": "openai-mini"}
CLASS_NAMES = {1: "wrong_arg", 3: "wrong_tool", 4: "repeated", 5: "propagate", 7: "precondition"}


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s))


def bootstrap_ci(y, s, n_boot: int, seed: int, mask=None) -> tuple[float, float]:
    """Percentile 95% interval over row resamples; resamples with one class only are skipped."""
    if mask is not None:
        y, s = y[mask], s[mask]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        vals.append(auroc(y[i], s[i]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def score_model(rows: list[dict], preds: dict, n_boot: int, seed: int) -> dict:
    y = np.array([r["label"] == "incorrect" for r in rows])
    cls = np.array([r["error_class"] or 0 for r in rows])
    dom = np.array([r["domain"] for r in rows])
    ps = [preds[r["trajectory_id"]] for r in rows]
    flag = np.array([p["pred_verdict"] == "incorrect" for p in ps])
    ok = np.array([p["pred_verdict"] in ("correct", "incorrect") and p["pred_confidence"] is not None for p in ps])
    s = np.array([
        (p["pred_confidence"] if p["pred_verdict"] == "incorrect" else 1 - p["pred_confidence"]) if o else 0.5
        for p, o in zip(ps, ok)
    ])
    all_rows = np.ones(len(rows), dtype=bool)
    no7 = cls != 7

    tp, fp, fn, tn = (int((y & flag).sum()), int((~y & flag).sum()), int((y & ~flag).sum()), int((~y & ~flag).sum()))
    out = {
        "n": len(rows), "n_unparsed": int((~ok).sum()),
        "auroc": auroc(y, s), "auroc_ci": bootstrap_ci(y, s, n_boot, seed, all_rows),
        "auroc_no_class7": auroc(y[no7], s[no7]), "auroc_no_class7_ci": bootstrap_ci(y, s, n_boot, seed, no7),
        "accuracy": (tp + tn) / len(rows),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn), "false_flag_rate": fp / (fp + tn),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "recall_by_class": {}, "by_domain": {},
    }
    for c in sorted(CLASS_NAMES):
        m = cls == c
        out["recall_by_class"][c] = (int((flag & m).sum()), int(m.sum()))
    for d in sorted(set(dom)):
        m = dom == d
        pos, neg = m & y, m & ~y
        out["by_domain"][d] = {
            "recall": (int((flag & pos).sum()), int(pos.sum())),
            "false_flag": (int((flag & neg).sum()), int(neg.sum())),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None, help="optional path to save the numbers")
    args = ap.parse_args()

    rows = load_jsonl(ROOT / "data" / "derived" / "test_real.jsonl")
    print(f"test_real: {len(rows)} rows, {sum(r['label'] == 'incorrect' for r in rows)} incorrect\n")
    results = {}
    for name, key in MODELS.items():
        raw = load_jsonl(RAW / f"{key}.jsonl")
        preds = {}
        for p in raw:  # the raw files can hold a trajectory twice (210 rows, 207 unique); keep the last
            preds[p["trajectory_id"]] = p
        missing = [r["trajectory_id"] for r in rows if r["trajectory_id"] not in preds]
        assert not missing, f"{key}: {len(missing)} test_real rows have no judge output"
        results[name] = r = score_model(rows, preds, args.n_boot, args.seed)

        print(f"== {name} ({r['n']} rows, {r['n_unparsed']} unparsed scored 0.5)")
        print(f"AUROC                {r['auroc']:.3f}   95% CI {r['auroc_ci'][0]:.2f}-{r['auroc_ci'][1]:.2f}")
        print(f"AUROC without Cl. 7  {r['auroc_no_class7']:.3f}   95% CI "
              f"{r['auroc_no_class7_ci'][0]:.2f}-{r['auroc_no_class7_ci'][1]:.2f}")
        print(f"accuracy {r['accuracy']:.1%}   precision {r['precision']:.1%}   recall {r['recall']:.1%} "
              f"({r['tp']}/{r['tp'] + r['fn']})   false-flag {r['false_flag_rate']:.1%} ({r['fp']}/{r['fp'] + r['tn']})")
        print("recall by class: " + "   ".join(
            f"{c} {CLASS_NAMES[c]} {a}/{b}" for c, (a, b) in r["recall_by_class"].items()))
        print("by domain:       " + "   ".join(
            f"{d} recall {v['recall'][0]}/{v['recall'][1]}, false-flag {v['false_flag'][0]}/{v['false_flag'][1]}"
            for d, v in r["by_domain"].items()))
        print()
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
