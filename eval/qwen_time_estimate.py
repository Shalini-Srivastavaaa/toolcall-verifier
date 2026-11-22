"""Time estimate for the Qwen2.5-1.5B (4-bit LoRA) run on the v3b data (D-029): a FLOP model divided by an effective rate, with the rate taken from the only Qwen timing that exists
(the Colab run: 1,293 steps at 0.08 it/s) and a cross-check of the model against the measured v3b ModernBERT run (6,780 s). An estimate, not a measurement: BENCHMARK_STEPS = 100 in the
notebook replaces it with one on the real GPU.

    .venv/bin/python3 eval/qwen_time_estimate.py          # needs results/final_v3b/qwen_train_tokens_{4096,2048}.json from eval/qwen_lengths.py and results/speed_cache/ (measure_speed.py lengths)
"""

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "notebooks"))
sys.path.insert(0, str(ROOT / "eval"))
import measure_speed as ms  # noqa: E402


def weighted(tokens, real, w):
    out = []
    for t, r in zip(tokens, real):
        out += [t] * (w if r else 1)
    return out


def modernbert_lengths(data_dir):
    import contextlib
    import io
    import os
    from diagnose_visibility import load_notebook_namespace
    tmp = Path(tempfile.mkdtemp())
    (tmp / "in" / "toolcall-verifier-data").mkdir(parents=True)
    for f in Path(data_dir).glob("final_*.jsonl"):
        (tmp / "in" / "toolcall-verifier-data" / f.name).symlink_to(f.resolve())
    (tmp / "work").mkdir()
    with contextlib.redirect_stdout(io.StringIO()):
        ns = load_notebook_namespace("None", tmp / "in", tmp / "work")
        tr = ns["FINAL"]["train"]
        L = [ns["format_input_v2"](r, meta=True)[1]["tokens"] for r in tr]
    return L, [bool(r.get("real_origin")) for r in tr]


def main():
    hours = ms.hours
    ep = 3
    # 1. cross-check of the FLOP model against the measured v3b ModernBERT run (one T4, micro-batch 2, no checkpointing, bucketed, 10,348 rows per epoch, 6,780 s)
    L, real = modernbert_lengths(ROOT / "data" / "kaggle_upload_v3b")
    Lw = weighted(L, real, 3)
    f_mb = ms.epoch_flops(Lw, 2, False, ms.mb_flops) * ep
    secs = 6780 - 3 * 28 - 120   # minus three validation passes of about 28 s and a margin for saves
    print(f"ModernBERT v3b (measured): {len(Lw)} rows per epoch, {sum(Lw) / 1e6:.1f} M tokens; FLOP model {f_mb / 1e15:.0f} PFLOP in about {secs:,} s of training = {f_mb / secs / 1e12:.1f} TFLOPs effective "
          f"(the earlier assumptions were 7.1 on Colab and 2.0 per GPU on Kaggle with DataParallel)")
    # 2. Qwen
    q_v1 = json.load(open(ms.CACHE / "qwen_v1_legacy.json"))
    f_obs = ms.epoch_flops(q_v1, 8, True, ms.qw_flops, bucket=False) * ep * 6892 / 6816
    r_q = f_obs / (1293 / 0.08)
    print(f"\nQwen2.5-1.5B 4-bit calibration (Colab run, 1,293 steps at 0.08 it/s, v1 input, random batches of 8, checkpointing): {f_obs / 1e15:.0f} PFLOP -> {r_q / 1e12:.1f} TFLOPs effective")
    print(f"{'budget':>6s} {'micro-batch':>11s} {'checkpointing':>13s} {'tokens/epoch (weighted)':>24s} {'PFLOP (3 epochs)':>16s} {'time at the calibrated rate':>28s} {'at half that rate':>18s} {'TFLOPs needed for 11 h':>23s}")
    for budget in (4096, 2048):
        d = json.load(open(ROOT / "results" / "final_v3b" / f"qwen_train_tokens_{budget}.json"))
        Lq = weighted(d["tokens"], d["real_origin"], 3)
        for bs, ck in ((2, True), (1, True), (2, False), (1, False)):
            if budget == 4096 and not ck:
                continue   # about 15 GB of activations per 4,096-token row without checkpointing: does not fit a T4
            f = ms.epoch_flops(Lq, bs, ck, ms.qw_flops) * ep
            print(f"{budget:6d} {bs:11d} {str(ck):>13s} {sum(Lq) / 1e6:21.2f} M {f / 1e15:16.0f} {hours(f / r_q):>28s} {hours(f / (r_q / 2)):>18s} {f / (11 * 3600) / 1e12:23.1f}")
    d = json.load(open(ROOT / "results" / "final_v3b" / "qwen_train_tokens_4096.json"))
    Lq = weighted(d["tokens"], d["real_origin"], 3)
    f05 = ms.epoch_flops(Lq, 2, True, lambda n, c: (6 if c else 4) * 0.36e9 * n + (4 / 3 if c else 1) * 3 * 4 * 896 * 24 * n * n) * ep
    print(f"\nQwen2.5-0.5B, 4,096 tokens, checkpointing, same rate assumed: {f05 / 1e15:.0f} PFLOP -> {hours(f05 / r_q)}")
    print("\nQwen2.5-1.5B, 4,096 tokens, micro-batch 2 with checkpointing: the changes that bring the run under 11 hours with room to spare")
    for w, e in ((3, 3), (3, 2), (2, 3), (2, 2), (1, 3)):
        Lx = weighted(d["tokens"], d["real_origin"], w)
        fx = ms.epoch_flops(Lx, 2, True, ms.qw_flops) * e
        print(f"   REAL_WEIGHT {w}, {e} epochs: {sum(Lx) / 1e6:5.2f} M tokens/epoch, {fx / 1e15:4.0f} PFLOP -> {hours(fx / r_q):>7s} at the calibrated rate (margin to 11 h: {11 * 3600 / (fx / r_q):.2f}x); needs {fx / (11 * 3600) / 1e12:4.1f} TFLOPs for 11 h")
    # validation and test scoring: forward only, 2N per token (checkpointing irrelevant), three splits
    ev_tok = sum(d["val_tokens"]) + sum(d["test_tokens"]) + sum(d["test_synth_tokens"])
    f_ev = 2 * 1.31e9 * ev_tok
    print(f"validation after each epoch (0.59 M tokens) and the final evaluation of test, test_synth and validation ({ev_tok / 1e6:.1f} M tokens in all): about {f_ev / r_q / 60:.0f} minutes at the same rate for the final evaluation, {2 * 1.31e9 * sum(d['val_tokens']) / r_q / 60:.1f} minutes per validation pass")

    # 3. D-029 (OOM fix): the default is now micro-batch 2 WITH gradient checkpointing. Does the estimate above already include it, and what does it do to the 2-epoch time?
    print("\n" + "=" * 100 + "\nCheckpointing check (D-029): REAL_WEIGHT 3, 2 epochs, 4,096 tokens, Qwen2.5-1.5B\n" + "=" * 100)
    Lw3 = weighted(d["tokens"], d["real_origin"], 3)
    print("the FLOP model of notebooks/measure_speed.py counts 4N FLOP per token for the frozen 4-bit backbone without checkpointing (forward 2N + activation gradients 2N; no weight gradients) and 6N with it")
    print("(the forward pass is recomputed: +2N). The extra cost of checkpointing is therefore +50% of the matrix FLOPs here, not the +30% of a full fine-tune, and the estimates above and in D-029 used the checkpointing figure already.")
    print(f"   {'micro-batch':>11s} {'checkpointing':>13s} {'PFLOP':>6s} {'time at 11.1 TFLOPs':>20s}")
    base = {}
    for bs, ck in ((2, False), (2, True), (1, True)):
        f = ms.epoch_flops(Lw3, bs, ck, ms.qw_flops) * 2
        base[(bs, ck)] = f
        print(f"   {bs:11d} {'on' if ck else 'off':>13s} {f / 1e15:6.0f} {hours(f / r_q):>20s}" + ("   <- does not fit the T4 at 4,096 tokens (D-029 memory estimate): shown only for the ratio" if not ck else ""))
    print(f"   checkpointing costs {base[(2, True)] / base[(2, False)] - 1:+.0%} of the FLOPs at micro-batch 2; the 2-epoch figure to use is the 'on' line")
    # per micro-batch overhead of dequantizing the 4-bit weights (read 0.65 GB, write 2.6 GB of fp16, three times per micro-batch with checkpointing: forward, recompute, backward), memory-bound at about 250 GB/s
    deq = 3 * (0.65e9 + 2.62e9) / 250e9
    for bs in (2, 1):
        n_mb = len(Lw3) / bs * 2
        print(f"   dequantization of the 4-bit weights at micro-batch {bs}: about {deq * 1000:.0f} ms per micro-batch x {n_mb:,.0f} micro-batches in 2 epochs = {n_mb * deq / 3600:.2f} h (not in the FLOP count)")
    f = base[(2, True)]
    ev = (7 + 2 * 2.3) / 60   # final evaluation plus the two validation passes, hours
    print(f"\n   sensitivity of the 2-epoch run (micro-batch 2, checkpointing on) to the effective rate; the one calibration figure is {r_q / 1e12:.1f} TFLOPs, 17% of the T4's fp16 peak of 65 TFLOPs")
    print(f"   {'rate vs calibration':>20s} {'TFLOPs':>7s} {'training':>9s} {'+ evaluation':>13s} {'margin to 11 h':>15s}")
    for fac in (1.0, 0.85, 0.7, 0.6, 0.5):
        t = f / (r_q * fac) / 3600
        print(f"   {fac:>19.0%} {r_q * fac / 1e12:7.1f} {t:8.1f}h {t + ev:12.1f}h {11 / (t + ev):14.2f}x")
    need2 = f / 3600 / (11 - ev) / 1e12          # TFLOPs needed to finish 2 epochs plus the evaluation in 11 hours
    need3 = f * 1.5 / 3600 / (11 - ev) / 1e12    # three epochs: 1.5 times the FLOPs
    print(f"   2 epochs plus the evaluation fit 11 hours as long as the real rate is at least {need2:.1f} TFLOPs ({need2 * 1e12 / r_q:.0%} of the calibration); 3 epochs need {need3:.1f} TFLOPs ({need3 * 1e12 / r_q:.0%})")

if __name__ == "__main__":
    main()
