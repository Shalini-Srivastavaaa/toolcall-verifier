"""Run the Phase 0 judges (gpt-5-nano and gpt-5-mini, low reasoning effort) on the FINAL test set, so the judges and the model are
compared on identical rows (D-023). The prompt is the Phase 0 one (eval/prompts.py, unchanged).

    python eval/judge_new_test.py                  # ESTIMATE ONLY: expected cost now and with the projected review additions; no API call
    python eval/judge_new_test.py --run            # call the API (cost cap, caching), then print the report
    python eval/judge_new_test.py --report-only    # the report from the cache, no API call
    options: --test PATH (default data/derived/final/test.jsonl; --preview uses data/derived/final_preview/test.jsonl)
             --models openai-nano openai-mini   --cap-usd 2.50   --workers 4   --with-synth (also the synthetic test_synth rows)
             --compare PRED.jsonl  (the model's test_predictions.jsonl from the notebook: its scores are printed next to the judges' on the same rows)

Cost control: before any call the script estimates the cost of the rows that are not cached yet (input tokens from the prompt length,
calibrated on the recorded Phase 0 runs; output tokens from their recorded mean) and refuses to start when that exceeds --cap-usd
(--force overrides); while running it adds up the real token usage and stops when the cap is reached, keeping everything done so far.
Caching: every answer is stored in results/judge_new_test/<model>.jsonl keyed by trajectory id and a hash of the exact prompt, so a rerun
only pays for new or changed rows. The 102 `test_real` rows already have Phase 0 answers (results/phase0_raw/) made with the same prompt;
they are reused (assumption: the rows are the same as in Phase 0; --no-phase0-cache turns this off).

Score for AUROC: confidence if the verdict is "incorrect", else 1 - confidence; an unparsed answer scores 0.5 (as in eval/rescore_phase0.py).
The report has the same sections as the notebooks' final report: overall (AUROC with a 95% bootstrap interval, precision, recall, F1,
false-flag rate, recall at 5% and 10% false alarms), per class by origin (a cell with fewer than 10 examples is not shown as a rate),
per domain, and the synthetic test set separately.
"""

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import load_models, load_pricing  # noqa: E402
from prompts import JUDGE_SYSTEM_PROMPT, build_user_prompt  # noqa: E402

DER = ROOT / "data" / "derived"
CACHE_DIR = ROOT / "results" / "judge_new_test"
PHASE0_DIR = ROOT / "results" / "phase0_raw"
REVIEW_ADDITIONS = {"error_candidates": 92 + 57, "correct_examples": 150}   # queue 1 (natural + induced) and queue 2 of data/review_queues.py


def load(p: Path) -> list[dict]:
    return [json.loads(line) for line in open(p) if line.strip()] if p.exists() else []


def prompt_key(row: dict) -> str:
    return hashlib.sha256((JUDGE_SYSTEM_PROMPT + "\n" + build_user_prompt(row)).encode()).hexdigest()[:16]


def read_cache(model_key: str, rows: list[dict], use_phase0: bool) -> dict:
    """trajectory_id -> record, only if the stored prompt hash equals the current one (Phase 0 records have no hash: reused for test_real rows)."""
    want = {r["trajectory_id"]: prompt_key(r) for r in rows}
    out = {}
    for rec in load(CACHE_DIR / f"{model_key}.jsonl"):
        if want.get(rec["trajectory_id"]) == rec.get("prompt_key") and rec.get("pred_verdict") is not None:
            out[rec["trajectory_id"]] = rec
    if use_phase0:
        for rec in load(PHASE0_DIR / f"{model_key}.jsonl"):
            if rec["trajectory_id"] in want and rec["trajectory_id"] not in out and rec.get("pred_verdict") is not None:
                out[rec["trajectory_id"]] = {**rec, "prompt_key": "phase0", "cost_usd": 0.0}
    return out


def calibration(model_key: str):
    """(characters per input token, mean output tokens) from the recorded Phase 0 runs on the same prompt."""
    by_id = {r["trajectory_id"]: r for r in load(DER / "test_real.jsonl")}
    ch = tok = 0
    outs = []
    for rec in load(PHASE0_DIR / f"{model_key}.jsonl"):
        r = by_id.get(rec["trajectory_id"])
        if r and rec.get("input_tokens"):
            ch += len(JUDGE_SYSTEM_PROMPT) + len(build_user_prompt(r))
            tok += rec["input_tokens"]
            outs.append(rec["output_tokens"])
    if not tok:
        return 3.5, 400.0
    return ch / tok, float(np.mean(outs))


def row_cost(row, model_key, pricing, cal):
    chars_per_tok, out_mean = cal
    n_in = (len(JUDGE_SYSTEM_PROMPT) + len(build_user_prompt(row))) / chars_per_tok
    p = pricing[model_key]
    return (n_in * p["input_per_1m"] + out_mean * p["output_per_1m"]) / 1e6, n_in


def estimate(rows, models, pricing, use_phase0, cap):
    print(f"test rows: {len(rows)}")
    total_new = 0.0
    for key in models:
        cal = calibration(key)
        cache = read_cache(key, rows, use_phase0)
        todo = [r for r in rows if r["trajectory_id"] not in cache]
        costs = [row_cost(r, key, pricing, cal) for r in todo]
        c = sum(x for x, _ in costs)
        total_new += c
        allc = [row_cost(r, key, pricing, cal)[0] for r in rows]
        mean_all = float(np.mean(allc)) if allc else 0.0
        print(f"  {key:12s} model {models[key]['model']:12s} cached {len(cache):4d}, to run {len(todo):4d}: about ${c:.3f} "
              f"(input {sum(n for _, n in costs):,.0f} tokens; {cal[0]:.2f} characters per token and {cal[1]:.0f} output tokens per call, both from Phase 0)")
        print(f"  {'':12s} mean cost per row for this test set ${mean_all:.4f}")
    print(f"  expected cost of the rows still to run, both judges: ${total_new:.3f}  (cap ${cap:.2f})")
    cur = len(rows)
    print("\nprojection once the review adds rows (queue 1: up to 149 error candidates; queue 2: 150 correct examples; the cached rows cost nothing):")
    per_row = {k: float(np.mean([row_cost(r, k, pricing, calibration(k))[0] for r in rows])) if rows else 0.0 for k in models}
    for label, frac in (("if 40% of the candidates are accepted", 0.4), ("if all candidates are accepted", 1.0)):
        added = frac * REVIEW_ADDITIONS["error_candidates"] + REVIEW_ADDITIONS["correct_examples"]
        # the review rows are not in the test file yet, so the mean cost of the available rows stands in for them (the first rows are the long telecom runs of test_real)
        cost = added * sum(per_row.values())
        print(f"   {label}: about {cur + added:.0f} rows ({added:.0f} new), about ${cost:.2f} for the new rows with both judges (about ${total_new + cost:.2f} including what is still to run now)")


def run_one(model_key, cfg, row):
    from llm_clients import call_openai
    res = call_openai(cfg["model"], JUDGE_SYSTEM_PROMPT, build_user_prompt(row), reasoning_effort=cfg.get("reasoning_effort"))
    return {"trajectory_id": row["trajectory_id"], "prompt_key": prompt_key(row), "domain": row["domain"], "true_label": row["label"],
            "pred_verdict": res.verdict, "pred_confidence": res.confidence, "pred_reasoning": res.reasoning, "latency_s": res.latency_s,
            "input_tokens": res.input_tokens, "output_tokens": res.output_tokens, "error": res.error}


def run(rows, models, pricing, cap, workers, use_phase0, force):
    est = 0.0
    for key in models:
        cache = read_cache(key, rows, use_phase0)
        cal = calibration(key)
        est += sum(row_cost(r, key, pricing, cal)[0] for r in rows if r["trajectory_id"] not in cache)
    if est > cap and not force:
        sys.exit(f"estimated cost ${est:.2f} is above the cap ${cap:.2f}: raise --cap-usd or pass --force")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    spent, lock, stop = 0.0, threading.Lock(), threading.Event()
    for key, cfg in models.items():
        cache = read_cache(key, rows, use_phase0)
        todo = [r for r in rows if r["trajectory_id"] not in cache]
        print(f"[{key}] {len(cache)} cached, {len(todo)} to run ({cfg['model']}, reasoning {cfg.get('reasoning_effort')})")
        out = open(CACHE_DIR / f"{key}.jsonl", "a")

        def work(row):
            if stop.is_set():
                return None
            return run_one(key, cfg, row)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(work, r) for r in todo]
            for i, f in enumerate(as_completed(futs), 1):
                rec = f.result()
                if rec is None:
                    continue
                p = pricing[key]
                rec["cost_usd"] = (rec["input_tokens"] * p["input_per_1m"] + rec["output_tokens"] * p["output_per_1m"]) / 1e6
                with lock:
                    spent += rec["cost_usd"]
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                    if spent >= cap and not stop.is_set():
                        stop.set()
                        print(f"!!! cost cap ${cap:.2f} reached (${spent:.3f} spent): stopping; rerun with a higher cap to finish (cached rows are kept)")
                if i % 25 == 0:
                    print(f"   [{key}] {i}/{len(todo)} done, ${spent:.3f} spent so far")
        out.close()
    print(f"spent ${spent:.3f} this run")


# ───────────────────────── report (same sections as the notebooks' final report) ─────────────────────────

def scores_of(key, rows, cache):
    s, unparsed = [], 0
    for r in rows:
        rec = cache.get(r["trajectory_id"])
        if rec is None or rec["pred_verdict"] is None or rec["pred_confidence"] is None:
            s.append(0.5)
            unparsed += 1
        else:
            c = float(rec["pred_confidence"])
            s.append(c if rec["pred_verdict"] == "incorrect" else 1 - c)
    return np.array(s), unparsed


def boot(y, p, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            vals.append(roc_auc_score(y[i], p[i]))
    return np.percentile(vals, [2.5, 97.5])


def recall_at_fpr(y, p, f):
    fpr, tpr, thr = roc_curve(y, p)
    k = np.flatnonzero(fpr <= f)[-1]
    return float(tpr[k])


def cell(flagged, n):
    return f"too few to read ({flagged}/{n})" if n < 10 else f"recall {flagged / n:.2f} ({flagged}/{n})"


def report(name, rows, s, unparsed, label="real test set"):
    y = np.array([r["label"] == "incorrect" for r in rows]).astype(int)
    pred = s >= 0.5
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    prec, rec = tp / max(tp + fp, 1), tp / max(int(y.sum()), 1)
    lo, hi = boot(y, s)
    print(f"\n=== {name} on the {label}: {len(y)} rows, {int(y.sum())} incorrect ({unparsed} unparsed scored 0.5) ===")
    print(f"AUROC {roc_auc_score(y, s):.3f}  95% CI {lo:.3f}-{hi:.3f}  (1,000 bootstrap resamples, seed 0)")
    print(f"verdict 'incorrect' as the flag: precision {prec:.3f}  recall {rec:.3f}  F1 {2 * prec * rec / max(prec + rec, 1e-9):.3f}  false-flag rate {fp / max(int((y == 0).sum()), 1):.3f}")
    print(f"recall at 5% false-alarm rate: {recall_at_fpr(y, s, 0.05):.3f}   at 10%: {recall_at_fpr(y, s, 0.10):.3f}")
    origins = sorted({r.get("origin", "natural") for r in rows})
    print("recall per error class by origin (a cell with fewer than 10 examples is not shown as a rate)")
    for c in range(1, 8):
        cells = []
        for o in origins:
            m = np.array([r["label"] == "incorrect" and r.get("error_class") == c and r.get("origin", "natural") == o for r in rows])
            cells.append(f"{o}: " + cell(int((pred[m]).sum()), int(m.sum())))
        print(f"  class {c}: " + " | ".join(cells))
    print("per domain")
    for d in sorted({r["domain"] for r in rows}):
        m = np.array([r["domain"] == d for r in rows])
        ni, nc = int(y[m].sum()), int((1 - y[m]).sum())
        auc = f"AUROC {roc_auc_score(y[m], s[m]):.3f}" if ni >= 10 and nc >= 10 else "AUROC too few to read"
        rec_d = f"recall {pred[m & (y == 1)].mean():.2f}" if ni else "recall n/a"
        ff_d = f"false-flag {pred[m & (y == 0)].mean():.2f}" if nc else "false-flag n/a"
        print(f"  {d:8s} n {int(m.sum()):4d} (incorrect {ni:3d}, correct {nc:3d})  {auc}  | verdict flag: {rec_d} ({int(pred[m & (y == 1)].sum())}/{ni}), {ff_d}")
    return roc_auc_score(y, s)


def save_predictions(key, name, rows, s):
    """The judge's scores with row ids, in the format of the notebooks' test_predictions.jsonl (p_incorrect = the score used for the AUROC), so they can be compared row by row."""
    with open(CACHE_DIR / f"{key}_{name}_predictions.jsonl", "w") as f:
        for r, p in zip(rows, s):
            f.write(json.dumps({"trajectory_id": r["trajectory_id"], "label": r["label"], "error_class": r.get("error_class"), "origin": r.get("origin"), "source": r.get("source"),
                                "domain": r["domain"], "split": r.get("split"), "p_incorrect": float(p)}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default=None)
    ap.add_argument("--preview", action="store_true", help="use data/derived/final_preview/ instead of data/derived/final/")
    ap.add_argument("--models", nargs="+", default=["openai-nano", "openai-mini"])
    ap.add_argument("--cap-usd", type=float, default=2.50)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--with-synth", action="store_true")
    ap.add_argument("--no-phase0-cache", action="store_true")
    ap.add_argument("--compare", default=None, help="the model's test_predictions.jsonl (same rows) to print beside the judges")
    a = ap.parse_args()

    base = DER / ("final_preview" if a.preview else "final")
    test_path = Path(a.test) if a.test else base / "test.jsonl"
    if not test_path.exists():
        sys.exit(f"{test_path} does not exist yet: the final test set is written by build_final_dataset.py --write --review-complete (use --preview to try it on the dry-run output)")
    rows = load(test_path)
    if a.with_synth:
        synth = load(test_path.parent / "test_synth.jsonl")
    else:
        synth = []
    models = {k: v for k, v in load_models().items() if k in a.models}
    pricing = load_pricing()
    use_p0 = not a.no_phase0_cache

    if not (a.run or a.report_only):
        estimate(rows + synth, models, pricing, use_p0, a.cap_usd)
        print("\nestimate only: nothing was called. Add --run to start (the cap is checked first).")
        return
    if a.run:
        try:
            from dotenv import load_dotenv
            load_dotenv(ROOT / ".env")
        except ImportError:
            pass
        if not os.environ.get("OPENAI_API_KEY"):
            sys.exit("OPENAI_API_KEY is not set (put it in .env)")
        run(rows + synth, models, pricing, a.cap_usd, a.workers, use_p0, a.force)

    ours = {}
    if a.compare:
        ours = {r["trajectory_id"]: r["p_incorrect"] for r in load(Path(a.compare))}
    aucs = {}
    for key in models:
        cache = read_cache(key, rows + synth, use_p0)
        s, un = scores_of(key, rows, cache)
        aucs[key] = report(f"{key} ({models[key]['model']})", rows, s, un)
        save_predictions(key, "test", rows, s)
        if synth:
            ss, un2 = scores_of(key, synth, cache)
            report(f"{key} (SYNTHETIC held-out Classes 2 and 6, not real errors)", synth, ss, un2, "synthetic test set")
            save_predictions(key, "test_synth", synth, ss)
        if ours:
            ids = [r["trajectory_id"] for r in rows]
            if all(i in ours for i in ids):
                y = np.array([r["label"] == "incorrect" for r in rows]).astype(int)
                print(f"   the model on the same {len(ids)} rows: AUROC {roc_auc_score(y, np.array([ours[i] for i in ids])):.3f}")
            else:
                print("   --compare file does not cover every test row; skipped")


if __name__ == "__main__":
    main()
