"""Phase 0 kill test: run cheap frontier models as multi-turn tool-call judges.

Usage:
    python eval/run_baseline.py --dry-run --limit 3          # sanity check, no API calls, no cost
    python eval/run_baseline.py --models openai-nano --limit 20   # small paid run
    python eval/run_baseline.py                                    # full run, all models, all candidates

Results are appended to results/phase0_raw/<model_key>.jsonl as they come in
(one line per candidate), so an interrupted run can be resumed: candidates
already present for a model are skipped on the next invocation. The summary
table is computed from whatever is in those files, so `--summarize-only` can
be re-run any time without spending anything.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "data"))

from dotenv import load_dotenv

from config import load_models, load_pricing
from llm_clients import CALLERS
from prompts import JUDGE_SYSTEM_PROMPT, build_user_prompt

ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "results" / "phase0_raw"


def load_candidates(path: Path, limit: int | None) -> list[dict]:
    candidates = [json.loads(line) for line in open(path)]
    if limit:
        candidates = candidates[:limit]
    return candidates


def already_done(model_key: str) -> set[str]:
    f = RAW_DIR / f"{model_key}.jsonl"
    if not f.exists():
        return set()
    ids = set()
    for line in open(f):
        try:
            ids.add(json.loads(line)["trajectory_id"])
        except (json.JSONDecodeError, KeyError):
            continue
    return ids


def run_model(model_key: str, model_cfg: dict, candidates: list[dict], dry_run: bool) -> None:
    done = already_done(model_key)
    caller = CALLERS[model_cfg["provider"]]
    model_name = model_cfg["model"]
    reasoning_effort = model_cfg.get("reasoning_effort")

    todo = [c for c in candidates if c["trajectory_id"] not in done]
    print(f"[{model_key}] {len(done)} already done, {len(todo)} to run ({model_name})")

    if dry_run:
        for i, cand in enumerate(todo):
            print(f"--- [{model_key}] candidate {i} ({cand['trajectory_id']}) ---")
            print(build_user_prompt(cand)[:500], "...\n")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"{model_key}.jsonl"
    with open(out_path, "a") as out:
        for i, cand in enumerate(todo):
            user_prompt = build_user_prompt(cand)
            result = caller(model_name, JUDGE_SYSTEM_PROMPT, user_prompt, reasoning_effort=reasoning_effort)
            record = {
                "trajectory_id": cand["trajectory_id"],
                "domain": cand["domain"],
                "true_label": cand["label"],
                "pred_verdict": result.verdict,
                "pred_confidence": result.confidence,
                "pred_reasoning": result.reasoning,
                "latency_s": result.latency_s,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "reasoning_tokens": result.extra.get("reasoning_tokens", 0),
                "error": result.error,
            }
            out.write(json.dumps(record) + "\n")
            out.flush()
            status = "OK" if result.verdict else f"UNPARSED/ERROR: {result.error}"
            print(f"  [{i + 1}/{len(todo)}] {cand['trajectory_id']} -> {result.verdict} ({status})")


def summarize(model_keys: list[str], models: dict, pricing: dict) -> None:
    print("\n" + "=" * 72)
    print("PHASE 0 SUMMARY")
    print("=" * 72)
    rows = []
    for key in model_keys:
        f = RAW_DIR / f"{key}.jsonl"
        if not f.exists():
            continue
        records = [json.loads(line) for line in open(f)]
        scored = [r for r in records if r["pred_verdict"] is not None]
        unparsed = len(records) - len(scored)
        if not scored:
            print(f"{key}: no scored predictions yet")
            continue

        correct_preds = sum(1 for r in scored if r["pred_verdict"] == r["true_label"])
        accuracy = correct_preds / len(scored)

        # The number that actually matters for a verifier: of the genuinely bad
        # calls, how many did the judge catch? And how often does it cry wolf
        # on good calls? Raw accuracy can look mediocre-but-OK while hiding a
        # judge that just rubber-stamps everything "correct".
        actual_bad = [r for r in scored if r["true_label"] == "incorrect"]
        actual_good = [r for r in scored if r["true_label"] == "correct"]
        caught_bad = sum(1 for r in actual_bad if r["pred_verdict"] == "incorrect")
        false_flagged_good = sum(1 for r in actual_good if r["pred_verdict"] == "incorrect")
        recall_bad_calls = caught_bad / len(actual_bad) if actual_bad else None
        false_flag_rate = false_flagged_good / len(actual_good) if actual_good else None
        pred_correct_rate = sum(1 for r in scored if r["pred_verdict"] == "correct") / len(scored)

        latencies = sorted(r["latency_s"] for r in records)
        p50 = statistics.median(latencies)
        p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]

        price = pricing.get(key, {"input_per_1m": 0, "output_per_1m": 0})
        total_cost = sum(
            r["input_tokens"] / 1e6 * price["input_per_1m"] + r["output_tokens"] / 1e6 * price["output_per_1m"]
            for r in records
        )
        cost_per_call = total_cost / len(records) if records else 0

        by_domain = {}
        for r in scored:
            d = by_domain.setdefault(r["domain"], {"correct": 0, "total": 0})
            d["total"] += 1
            d["correct"] += r["pred_verdict"] == r["true_label"]

        rows.append(
            {
                "model": key,
                "n": len(records),
                "unparsed": unparsed,
                "accuracy": accuracy,
                "recall_bad_calls": recall_bad_calls,
                "false_flag_rate": false_flag_rate,
                "pred_correct_rate": pred_correct_rate,
                "p50_latency_s": p50,
                "p95_latency_s": p95,
                "cost_per_call_usd": cost_per_call,
                "by_domain": {d: v["correct"] / v["total"] for d, v in by_domain.items()},
            }
        )

    header = f"{'model':<14} {'n':>5} {'unparsed':>9} {'accuracy':>9} {'p50 (s)':>8} {'p95 (s)':>8} {'$/call':>9}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['model']:<14} {r['n']:>5} {r['unparsed']:>9} {r['accuracy']:>9.1%} "
            f"{r['p50_latency_s']:>8.2f} {r['p95_latency_s']:>8.2f} {r['cost_per_call_usd']:>9.5f}"
        )
    print()
    print("The metric that matters for a verifier: does it catch bad calls, or rubber-stamp everything?")
    header2 = f"{'model':<14} {'catches bad calls':>18} {'false-flags good':>17} {'predicts correct':>17}"
    print(header2)
    print("-" * len(header2))
    for r in rows:
        print(
            f"{r['model']:<14} {r['recall_bad_calls']:>18.1%} {r['false_flag_rate']:>17.1%} {r['pred_correct_rate']:>17.1%}"
        )
    print()
    for r in rows:
        print(f"{r['model']} per-domain accuracy: {r['by_domain']}")

    if any(isinstance(p, dict) and p.get("input_per_1m") == 0 for p in pricing.values()):
        print(
            "\nNote: eval/pricing.json has placeholder $0 rates for at least one model — "
            "fill in current per-1M-token pricing before trusting $/call."
        )

    summary_path = ROOT / "results" / "phase0_baseline.json"
    summary_path.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote summary to {summary_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="data/derived/phase0_candidates.jsonl")
    ap.add_argument("--models", nargs="+", default=None, help="subset of model keys to run, e.g. openai-nano gemini")
    ap.add_argument("--limit", type=int, default=None, help="cap number of candidates (for testing / budget control)")
    ap.add_argument("--dry-run", action="store_true", help="print prompts, make no API calls")
    ap.add_argument("--summarize-only", action="store_true", help="skip API calls, just recompute the summary table")
    args = ap.parse_args()

    load_dotenv()

    models = load_models()
    pricing = load_pricing()
    model_keys = args.models or list(models.keys())
    for k in model_keys:
        if k not in models:
            sys.exit(f"Unknown model key '{k}'. Choices: {list(models.keys())}")

    candidates_path = ROOT / args.candidates
    candidates = load_candidates(candidates_path, args.limit)
    print(f"Loaded {len(candidates)} candidates from {candidates_path}")

    if not args.summarize_only:
        for key in model_keys:
            run_model(key, models[key], candidates, dry_run=args.dry_run)

    if not args.dry_run:
        summarize(model_keys, models, pricing)


if __name__ == "__main__":
    main()
