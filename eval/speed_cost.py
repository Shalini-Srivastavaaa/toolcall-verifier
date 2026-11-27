"""Latency and cost per call of every scorer on the 363 real test rows (D-030).

    <python with torch, transformers, numpy, scikit-learn> eval/speed_cost.py [--n 363] [--out results/final_qwen_v3b/speed_cost.json]

* ModernBERT v3b on this Mac's CPU (models/final_v3b/, fp32, batch of one row, torch's default threads): formatting + tokenizing + forward pass per row, measured here, all rows.
* ModernBERT and Qwen on a Kaggle T4: NOT measured per call. The only timings are the validation passes of the training runs (749 rows, length-sorted batches, fp16), taken from the
  recorded train_info.json / the Kaggle log: seconds per row amortized over a batch, which is throughput, not the latency of one request.
* GPT-5-mini and GPT-5-nano: the recorded latency of each call (Phase 0 and the judge run) and the cost recomputed from the recorded tokens at eval/pricing.json.
"""

import argparse
import json
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))


def q(xs, p):
    return float(np.percentile(xs, p))


def judges(ids):
    pricing = json.load(open(ROOT / "eval" / "pricing.json"))
    out = {}
    for key in ("openai-mini", "openai-nano"):
        recs = {}
        for f in (ROOT / "results" / "phase0_raw" / f"{key}.jsonl", ROOT / "results" / "judge_new_test" / f"{key}.jsonl"):   # the judge cache wins over Phase 0
            if f.exists():
                for l in open(f):
                    r = json.loads(l)
                    if r["trajectory_id"] in ids and r.get("pred_verdict") is not None and r.get("input_tokens"):
                        recs[r["trajectory_id"]] = r
        p = pricing[key]
        lat = [r["latency_s"] for r in recs.values()]
        cost = [(r["input_tokens"] * p["input_per_1m"] + r["output_tokens"] * p["output_per_1m"]) / 1e6 for r in recs.values()]
        out[key] = {"n_calls": len(recs), "latency_p50_s": q(lat, 50), "latency_p95_s": q(lat, 95), "latency_mean_s": float(np.mean(lat)), "cost_per_call_usd": float(np.mean(cost)),
                    "cost_total_usd": float(np.sum(cost)), "mean_input_tokens": float(np.mean([r["input_tokens"] for r in recs.values()])), "mean_output_tokens": float(np.mean([r["output_tokens"] for r in recs.values()]))}
    return out


def modernbert_cpu(n):
    import torch
    from transformers import AutoModelForSequenceClassification, PreTrainedTokenizerFast
    from diagnose_visibility import load_notebook_namespace

    tmp = Path(tempfile.mkdtemp())
    (tmp / "in" / "toolcall-verifier-data").mkdir(parents=True)
    for f in (ROOT / "data" / "kaggle_upload_v3b").glob("final_*.jsonl"):
        (tmp / "in" / "toolcall-verifier-data" / f.name).symlink_to(f.resolve())
    (tmp / "work").mkdir()
    ns = load_notebook_namespace("None", tmp / "in", tmp / "work")
    test = ns["FINAL"]["test"][:n]
    mdir = ROOT / "models" / "final_v3b"
    tok = PreTrainedTokenizerFast(tokenizer_file=str(mdir / "tokenizer.json"), pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]", unk_token="[UNK]")
    model = AutoModelForSequenceClassification.from_pretrained(str(mdir)).eval()
    saved = {json.loads(l)["trajectory_id"]: json.loads(l)["p_incorrect"] for l in open(ROOT / "results" / "final_v3b" / "test_predictions.jsonl")}
    # warm-up (first calls allocate and compile kernels): two rows, not counted
    for r in test[:2]:
        with torch.no_grad():
            model(**tok(ns["format_input_v2"](r), truncation=True, max_length=4096, return_tensors="pt"))
    rows = []
    for r in test:
        t0 = time.perf_counter()
        text = ns["format_input_v2"](r)
        t1 = time.perf_counter()
        enc = tok(text, truncation=True, max_length=4096, return_tensors="pt")
        t2 = time.perf_counter()
        with torch.no_grad():
            p = float(torch.softmax(model(**enc).logits.float(), dim=-1)[0, 1])
        t3 = time.perf_counter()
        rows.append({"tokens": int(enc["input_ids"].shape[1]), "format_s": t1 - t0, "tokenize_s": t2 - t1, "forward_s": t3 - t2, "total_s": t3 - t0, "diff_vs_saved": abs(p - saved[r["trajectory_id"]])})
    tot = [x["total_s"] for x in rows]
    fw = [x["forward_s"] for x in rows]
    toks = np.array([x["tokens"] for x in rows])
    short, long_ = toks <= np.median(toks), toks > np.median(toks)
    return {"n": len(rows), "device": f"{platform.processor() or platform.machine()} CPU, torch {torch.__version__}, {torch.get_num_threads()} threads, fp32, batch of 1",
            "total_p50_s": q(tot, 50), "total_p95_s": q(tot, 95), "total_mean_s": float(np.mean(tot)), "total_max_s": max(tot),
            "forward_p50_s": q(fw, 50), "forward_p95_s": q(fw, 95), "format_plus_tokenize_mean_s": float(np.mean([x["format_s"] + x["tokenize_s"] for x in rows])),
            "tokens_p50": float(np.median(toks)), "tokens_p95": q(toks, 95), "tokens_max": int(toks.max()),
            "total_mean_s_shorter_half": float(np.mean(np.array(tot)[short])), "total_mean_s_longer_half": float(np.mean(np.array(tot)[long_])),
            "max_diff_vs_saved_gpu_predictions": max(x["diff_vs_saved"] for x in rows), "mean_diff_vs_saved": float(np.mean([x["diff_vs_saved"] for x in rows]))}


def t4_from_records():
    mb = json.load(open(ROOT / "docs" / "final_v3b_train_info.json"))["val_metrics"]
    qwen_rt, qwen_n = 167.8, 749     # Kaggle log of the Qwen training run (not published): eval_runtime of the epoch-2 validation pass at batch 2
    return {"modernbert_t4": {"seconds_per_row_amortized": mb["eval_runtime"] / 749, "rows_per_second": mb["eval_samples_per_second"], "source": "docs/final_v3b_train_info.json val_metrics.eval_runtime (749 validation rows, fp16, length-sorted batches)"},
            "qwen_t4": {"seconds_per_row_amortized": qwen_rt / qwen_n, "rows_per_second": qwen_n / qwen_rt, "source": "Kaggle log of the Qwen training run: eval_runtime 167.8 s for the 749 validation rows at epoch 2 (4-bit, fp16, batch 2, length-sorted)"}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=363)
    ap.add_argument("--out", default=str(ROOT / "results" / "final_qwen_v3b" / "speed_cost.json"))
    a = ap.parse_args()
    ids = {json.loads(l)["trajectory_id"] for l in open(ROOT / "results" / "final_v3b" / "test_predictions.jsonl")}
    res = {"judges": judges(ids), "t4": t4_from_records(), "modernbert_cpu": modernbert_cpu(a.n)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
