"""Measure padding waste before and after length bucketing, and estimate full-run times (D-020, speed fix). No GPU, no API calls.

    python notebooks/measure_speed.py lengths          # tokenizes the training rows (about 2 min); caches token counts in results/speed_cache/
    python notebooks/measure_speed.py report           # padding waste table and the time estimates

`lengths` needs the tokenizers (ModernBERT-base and Qwen2.5-1.5B) in the Hugging Face cache, the legacy data files, and
data/derived/final_preview (build_final_dataset.py --preview) for the final-data numbers. The time estimates are a FLOP model (see
`estimates` below) divided by an effective TFLOPs rate that is derived from two earlier runs: they are estimates, not measurements,
and BENCHMARK_STEPS in the notebooks replaces them with a measurement on the real GPU.
"""

import contextlib
import io
import json
import os
import random
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks"
CACHE = ROOT / "results" / "speed_cache"
sys.path.insert(0, str(NB))
from length_bucketing_code import SAMPLER_CODE  # noqa: E402

_g = {}
exec(SAMPLER_CODE, _g)
padded_tokens, Bucket, Sorted = _g["padded_tokens"], _g["LengthBucketSampler"], _g["LengthSortedSampler"]


def run_cells(notebook, final):
    nb = json.load(open(NB / notebook))
    cells = ["\n".join(l for l in "".join(c["source"]).splitlines() if not l.lstrip().startswith(("!", "%"))) for c in nb["cells"] if c["cell_type"] == "code"]
    if final:
        cells[0] = cells[0].replace('DATASET = "legacy"', 'DATASET = "final"', 1)
        os.environ.setdefault("TCV_FINAL_DIR", str(ROOT / "data" / "derived" / "final_preview"))
    os.environ.pop("TCV_SMOKE", None)
    g = {"__name__": "__main__"}
    with contextlib.redirect_stdout(io.StringIO()):
        for i, s in enumerate(cells, 1):
            exec(compile(s, f"<cell {i}>", "exec"), g)
            if "train_rows" in g:
                break
    return g


def cmd_lengths():
    CACHE.mkdir(parents=True, exist_ok=True)
    for name, final in (("legacy", False), ("final", True)):
        g = run_cells("phase5_modernbert.ipynb", final)
        tok, fmt = g["tokenizer"], g["format_input_v2"]
        out = {k: [len(x) for x in tok([fmt(r) for r in g[k]], truncation=True, max_length=4096)["input_ids"]] for k in ("train_rows", "val_rows")}
        json.dump(out, open(CACHE / f"modernbert_v2_{name}.json", "w"))
        print(name, {k: len(v) for k, v in out.items()})
        if final:  # the same texts at budgets 4,096 and 2,048, counted with the Qwen tokenizer too (what a v2 input would cost Qwen)
            from transformers import AutoTokenizer
            qt = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
            q = {}
            for budget in (4096, 2048):
                g["TOKEN_BUDGET_V2"] = budget
                texts = [fmt(r) for r in g["train_rows"]]
                q[str(budget)] = [len(qt(t)["input_ids"]) for t in texts]
            json.dump(q, open(CACHE / "qwen_v2_final.json", "w"))
    g = run_cells("phase6_qwen.ipynb", False)  # Qwen v1 input (what Phase 6 trains on today), legacy rows
    json.dump([len(x) for x in g["tokenizer"]([g["format_input"](r) for r in g["train_rows"]])["input_ids"]], open(CACHE / "qwen_v1_legacy.json", "w"))
    print("cached in", CACHE)


def rand_batches(n, bs, seed):
    o = list(range(n))
    random.Random(seed).shuffle(o)
    return [o[i:i + bs] for i in range(0, n, bs)]


def mb_flops(n, ckpt):  # ModernBERT-base: 22 layers, d=768, ~110M non-embedding parameters; SDPA computes full n x n attention
    return (6 * 110e6 * n + 3 * 4 * 768 * 22 * n * n) * (4 / 3 if ckpt else 1)


def qw_flops(n, ckpt):  # Qwen2.5-1.5B QLoRA: 1.31e9 non-embedding parameters, 28 layers, d=1536; frozen weights: 4N per token (6N with checkpointing)
    return (6 if ckpt else 4) * 1.31e9 * n + (4 / 3 if ckpt else 1) * 3 * 4 * 1536 * 28 * n * n


def epoch_flops(L, bs, ckpt, fn, bucket=True, seeds=3):
    tot = 0
    for s in range(seeds):
        for b in (Bucket(L, bs, seed=s).batches(0) if bucket else rand_batches(len(L), bs, s)):
            tot += len(b) * fn(-(-max(L[i] for i in b) // 8) * 8, ckpt)
    return tot / seeds


def hours(s):
    return f"{s / 3600:.1f} h"


def cmd_report():
    load = lambda n: json.load(open(CACHE / n))  # noqa: E731
    for name in ("legacy", "final"):
        d = load(f"modernbert_v2_{name}.json")
        L, V = d["train_rows"], d["val_rows"]
        real = sum(L)
        print(f"\n== {name}: {len(L)} training rows, {real:,} real tokens (mean {real / len(L):.0f}, p50 {sorted(L)[len(L) // 2]}, p90 {sorted(L)[int(.9 * len(L))]}, max {max(L)})")
        for lab, bs in (("random order, batch 4 (2 GPUs x 2)", 4), ("random order, batch 2 (1 GPU)", 2)):
            p = st.mean(padded_tokens(L, rand_batches(len(L), bs, s)) for s in range(5))
            print(f"   before: {lab:36s} padding waste {1 - real / p:6.1%}")
        for bs in (2, 4, 8):
            p = st.mean(padded_tokens(L, Bucket(L, bs, seed=s).batches(0)) for s in range(5))
            print(f"   after:  length-bucketed, batch {bs:<2d}{'':19s} padding waste {1 - real / p:6.1%}")
        vr = sum(V)
        print(f"   validation: file order batch 4 waste {1 - vr / padded_tokens(V, [list(range(i, min(i + 4, len(V)))) for i in range(0, len(V), 4)]):.1%} "
              f"-> length-sorted batch 8 waste {1 - vr / padded_tokens(V, [Sorted(V).order[i:i + 8] for i in range(0, len(V), 8)]):.1%}")
    leg, fin = load("modernbert_v2_legacy.json")["train_rows"], load("modernbert_v2_final.json")["train_rows"]
    ep = 3
    kag = epoch_flops(leg, 4, True, mb_flops, bucket=False) * ep
    kag_s = 27560 - 3 * 154  # Kaggle v2 run: train_runtime 27,560 s including three 154 s validation passes
    col = epoch_flops(leg, 2, True, mb_flops, bucket=False) * ep * 6892 / 6816
    r_col, r_kag = col / 10800, kag / kag_s / 2
    print(f"\n== rates: Kaggle 2 x T4 DataParallel run: {kag / 1e15:.0f} PFLOP in {kag_s:,} s = {kag / kag_s / 1e12:.1f} TFLOPs total, {r_kag / 1e12:.1f} per GPU;"
          f" Colab 1 x T4 (about 3 h, as reported): {col / 1e15:.0f} PFLOP / 10,800 s = {r_col / 1e12:.1f} TFLOPs")
    print("== ModernBERT-base, final data (policy on, 4,096), bucketed, 3 epochs, one T4 (add about 10 min for validation and saves):")
    for lab, bs, ck in (("no checkpointing, micro-batch 4", 4, False), ("no checkpointing, micro-batch 2", 2, False), ("checkpointing, micro-batch 2", 2, True)):
        f = epoch_flops(fin, bs, ck, mb_flops) * ep
        print(f"   {lab:34s} {f / 1e15:5.1f} PFLOP: {hours(f / r_col)} at the Colab-derived rate, {hours(f / r_kag)} at the Kaggle per-GPU rate")
    q, qv1 = load("qwen_v2_final.json"), load("qwen_v1_legacy.json")
    f_obs = epoch_flops(qv1, 8, True, qw_flops, bucket=False) * ep * 6892 / 6816
    r_q = f_obs / (1293 / 0.08)  # D-020: Colab Qwen run, 1,293 steps at 0.08 it/s (about 4.5 h)
    print(f"\n== Qwen2.5-1.5B 4-bit: calibration from the Colab run (1,293 steps at 0.08 it/s): {f_obs / 1e15:.0f} PFLOP -> {r_q / 1e12:.1f} TFLOPs")
    for budget in ("4096", "2048"):
        L = q[budget]
        for ck, bs in ((True, 2), (False, 1)):
            if budget == "4096" and not ck:
                continue  # about 15 GB of activations per 4,096-token row: does not fit without checkpointing
            f = epoch_flops(L, bs, ck, qw_flops) * ep
            print(f"   final data, {budget} tokens, checkpointing {'on ' if ck else 'off'}: {sum(L) / 1e6:.2f}M tokens/epoch, {f / 1e15:5.0f} PFLOP: {hours(f / r_q)} at that rate; "
                  f"needs {f / (11 * 3600) / 1e12:.1f} TFLOPs to finish in 11 h")
    f = epoch_flops(q["4096"], 2, True, lambda n, c: (6 if c else 4) * 0.36e9 * n + (4 / 3 if c else 1) * 3 * 4 * 896 * 24 * n * n) * ep
    print(f"   Qwen2.5-0.5B, 4,096 tokens, checkpointing on (same rate assumed): {f / 1e15:.0f} PFLOP: {hours(f / r_q)}")


if __name__ == "__main__":
    {"lengths": cmd_lengths, "report": cmd_report}[sys.argv[1] if len(sys.argv) > 1 else "report"]()
