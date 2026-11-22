"""Out-of-memory handling for the Qwen notebooks (D-029, second part): patches the Colab notebook (phase6_qwen.ipynb, in place, once) and the generated Kaggle notebook.

What went wrong on Kaggle (T4, 4,096-token inputs): the retry ladder tried batch 2 and 1 without gradient checkpointing and batch 8 with it (all out of memory, as the
memory arithmetic in D-029 predicts) and then stopped silently at batch 4 with checkpointing, a configuration that should fit (about 8 GiB of 14.7). The cause of the
silent stop is NOT known: there is no log of the GPU or host memory of that run, and the Kaggle message (if any) was not reported. Candidates: GPU memory that the failed
attempts still held (the old loop called `empty_cache()` inside the `except` block, where the exception's traceback still referenced the failed step, so that call could not
release anything, and nothing collected the Trainer's reference cycles before the next model was loaded), fragmentation after four model loads, or the host running out of RAM
(Kaggle then restarts the kernel without a Python error). A local test on the toy model did not show a live model at the start of the next attempt with the old pattern either
(smoke_test_oom.py, D-029), so the first candidate is possible but not demonstrated. The changes below remove the first candidate by construction, make the others visible in
the log, and, above all, start at a configuration that is expected to fit so that no retry is needed.

The fix:
  * `train_attempt(...)` builds the trainer AND trains inside one function. On an out-of-memory error it records the message, leaves the `except` block (which releases the
    traceback), frees the trainer's references and returns (None, message). Nothing outlives a failed attempt except what the caller keeps: a string.
  * `release_gpu()` then runs `gc.collect()` and `torch.cuda.empty_cache()`; the GPU memory is logged before each attempt and after each cleanup, and if more than
    `LEAK_LIMIT_GIB` stays allocated after the cleanup the notebook stops with the exact flags to restart with, instead of continuing into a crash.
  * Two flags in the first cell, `MICRO_BATCH` (default 2) and `GRADIENT_CHECKPOINTING` (default True); the effective batch stays 16. The ladder only goes DOWN from
    `MICRO_BATCH` (2 then 1), with the checkpointing setting fixed: it never switches checkpointing off and never tries a larger batch than the one asked for.
  * Every memory line also shows the host RAM (psutil, when installed) and `faulthandler` is switched on, so that the next silent stop leaves a trace.
  * `attn_implementation="sdpa"` is passed explicitly (an eager-attention fallback would need about 2 GiB more per 4,096-token row), the validation batch is 2, the final
    evaluation batch 4, and the training objects are released before the final evaluation.

`python notebooks/qwen_oom_code.py` patches phase6_qwen.ipynb in place when it is not patched yet; make_kaggle_notebooks.py calls `patch_colab` first (so the Kaggle notebook
is generated from the patched Colab notebook) and `apply_kaggle` last.
"""

import json
import sys
from pathlib import Path

NB = Path(__file__).resolve().parent

MARKER = "def train_attempt("

FLAG_LINES = '''\
MICRO_BATCH = 2                # per-device micro-batch the training starts with (16 must divide by it). The effective batch stays 16: accumulation = 16 / MICRO_BATCH. The retry ladder only goes DOWN from here (2 -> 1)
GRADIENT_CHECKPOINTING = True  # True (default): recompute activations in the backward pass; needed for 4,096-token Qwen inputs (about 14 GB of activations per row without it). False only for short inputs; None counts as True
'''

OLD_FLAG_LINE = "GRADIENT_CHECKPOINTING = None  # None: try without it first (faster) and switch it on after an out-of-memory error. True / False forces it\n"

LADDER_OLD_START = "# (per-device micro-batch, gradient checkpointing), fastest first;"
LADDER_OLD_END = "EVAL_BATCH = 4\n"

LADDER_NEW = '''\
# Retry ladder (D-029): starts at MICRO_BATCH with the GRADIENT_CHECKPOINTING setting of the first cell and only goes DOWN (halving the micro-batch, never switching
# checkpointing, never a larger batch); every entry gives an effective batch of 16 (accumulation = 16 / micro-batch). The longest batch comes first in every epoch
# (LengthBucketSampler), so a memory problem shows at step 1, not after hours.
_CKPT = True if GRADIENT_CHECKPOINTING is None else bool(GRADIENT_CHECKPOINTING)
_START = 2 if MICRO_BATCH is None else int(MICRO_BATCH)
assert _START in (1, 2, 4, 8, 16), f"MICRO_BATCH must be 1, 2, 4, 8 or 16 (it must divide the effective batch of 16), not {MICRO_BATCH}"
LADDER = [(b, _CKPT) for b in (16, 8, 4, 2, 1) if b <= _START]
EVAL_BATCH = 2   # validation batch (forward only, longest rows 4,096 tokens): 2 keeps the working set to about 1 GiB
LEAK_LIMIT_GIB = 0.5   # GPU memory still allocated after the cleanup of a failed attempt that makes the notebook stop instead of retrying (a real leak)
ATTEMPTS = []          # one entry per training attempt: micro-batch, checkpointing, outcome, peak memory (goes into train_info.json)
'''

ATTEMPT_CODE = '''\


# --- memory-safe training attempts (D-029) ---
def host_mem():
    try:
        import psutil
        vm = psutil.virtual_memory()
        return f"host RAM used {(vm.total - vm.available) / 2 ** 30:.1f} of {vm.total / 2 ** 30:.1f} GiB, this process {psutil.Process().memory_info().rss / 2 ** 30:.1f} GiB"
    except Exception:
        return "host RAM unknown"


def gpu_mem(tag):
    """One line with the GPU memory (what PyTorch has allocated / reserved / peaked at, what the driver says is free) and the host RAM."""
    g = 2 ** 30
    if not torch.cuda.is_available():
        print(f"[memory | {tag}] no GPU | {host_mem()}", flush=True)
        return
    free, total = torch.cuda.mem_get_info()
    print(f"[GPU memory | {tag}] allocated {torch.cuda.memory_allocated() / g:.2f} GiB | reserved {torch.cuda.memory_reserved() / g:.2f} GiB | "
          f"peak allocated {torch.cuda.max_memory_allocated() / g:.2f} GiB | free (driver) {free / g:.2f} of {total / g:.2f} GiB | {host_mem()}", flush=True)


try:
    import faulthandler
    faulthandler.enable()  # a native crash (segfault, abort) prints the Python stack instead of ending the cell silently
except Exception:
    pass


def release_gpu():
    """Free what is unreachable now. Only effective when nothing refers to the failed attempt any more (train_attempt guarantees that by returning before this runs)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def is_oom(exc):
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower())


class MemoryLog(TrainerCallback):
    """Prints the GPU memory after the first optimizer step (the longest batch, so the worst case of the whole run) and after step 10."""

    def on_step_end(self, args, state, control, **kw):
        if state.global_step in (1, 10):
            gpu_mem(f"after optimizer step {state.global_step}")


def _dispose(trainer):
    """Drop the references that keep a failed attempt's model, optimizer and gradients alive (the Trainer, its accelerator and the callbacks form reference cycles)."""
    if trainer is None:
        return
    try:
        trainer.accelerator.free_memory()  # accelerate's own release of the prepared models / optimizers
    except Exception:
        pass
    for name in ("model", "model_wrapped", "optimizer", "lr_scheduler", "train_dataset", "eval_dataset"):
        try:
            setattr(trainer, name, None)
        except Exception:
            pass


def train_attempt(make_trainer, banner, resume, pad_batch):
    """Build the trainer and train, all inside this function. Returns (trainer, None) on success and (None, message) after an out-of-memory error.
    The cleanup is NOT inside the `except` block: while the handler runs, the exception still holds its traceback and with it the frames of the failed step (activations,
    gradients, the model), which would keep all of that on the GPU. Leaving the block releases the traceback; only then are the trainer's references dropped."""
    trainer, why = None, None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        trainer = make_trainer()
        print(banner, flush=True)
        padding_report(row_lengths(train_ds), pad_batch)
        trainer.train(resume_from_checkpoint=resume)
        return trainer, None
    except Exception as exc:
        if not is_oom(exc):
            raise
        why = f"{type(exc).__name__}: {str(exc).splitlines()[0][:230] if str(exc) else ''}"
    _dispose(trainer)
    trainer = None
    return None, why


def run_ladder(make_trainer_for, resume, n_gpu):
    """Try the entries of LADDER in order. `make_trainer_for(batch_size, ckpt, accum)` returns a fresh Trainer. Returns (trainer, batch_size, ckpt, accum)."""
    progress_callbacks = [MemoryLog()]
    base = torch.cuda.memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0.0
    if base > LEAK_LIMIT_GIB:
        print(f"WARNING: {base:.2f} GiB of GPU memory are already allocated before the first attempt (objects of an earlier run of a cell in this session). "
              "For a clean start use Run > Restart session; the leak check below counts only what a failed attempt adds to this.", flush=True)
    for k, (batch_size, ckpt) in enumerate(LADDER, 1):
        assert EFFECTIVE_BATCH % (batch_size * max(n_gpu, 1)) == 0, f"micro-batch {batch_size} x {n_gpu} GPU does not divide the effective batch {EFFECTIVE_BATCH}"
        accum = EFFECTIVE_BATCH // (batch_size * max(n_gpu, 1))
        gpu_mem(f"before attempt {k}/{len(LADDER)}: micro-batch {batch_size}, checkpointing {'on' if ckpt else 'off'}")
        banner = (f"\\ntraining with {n_gpu} GPU x per-device batch {batch_size} x grad_accum {accum} = {EFFECTIVE_BATCH}, gradient checkpointing {'on' if ckpt else 'off'} "
                  f"(attempt {k}/{len(LADDER)})")
        trainer, why = train_attempt(lambda: make_trainer_for(batch_size, ckpt, accum, progress_callbacks), banner, resume, batch_size * max(n_gpu, 1))
        if trainer is not None:
            peak = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else None
            ATTEMPTS.append({"micro_batch": batch_size, "gradient_checkpointing": ckpt, "outcome": "ok", "peak_gpu_gib": peak})
            return trainer, batch_size, ckpt, accum
        peak = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else None
        ATTEMPTS.append({"micro_batch": batch_size, "gradient_checkpointing": ckpt, "outcome": "out of memory", "peak_gpu_gib": peak, "message": why})
        print(f"CUDA out of memory with micro-batch {batch_size} (checkpointing {'on' if ckpt else 'off'}); peak {peak if peak is None else format(peak, '.2f')} GiB | {why}", flush=True)
        gpu_mem("failed attempt, before cleanup")
        release_gpu()
        gpu_mem("after cleanup")
        if torch.cuda.is_available() and torch.cuda.memory_allocated() / 2 ** 30 - base > LEAK_LIMIT_GIB:
            nxt = LADDER[k][0] if k < len(LADDER) else None
            raise RuntimeError(
                f"{torch.cuda.memory_allocated() / 2 ** 30 - base:.2f} GiB of GPU memory are still allocated after the cleanup of the failed attempt (limit {LEAK_LIMIT_GIB} GiB): something still "
                "refers to it, and a retry in this process would start with less memory than the one that failed. Restart the session (Kaggle: Run > Restart session; "
                "Colab: Runtime > Restart session) and set "
                + (f"MICRO_BATCH = {nxt}" if nxt else "QWEN_MAX_LEN = 2048 (Kaggle) / a shorter input (Colab)") + " in the first cell.")
    raise RuntimeError(
        f"out of memory at every micro-batch of the ladder {[b for b, _ in LADDER]} with gradient checkpointing {'on' if LADDER[0][1] else 'off'}. See the attempts above: "
        "the peak memory of the smallest micro-batch tells how far it is from fitting. Lower QWEN_MAX_LEN (Kaggle) / the input length (Colab), or use MODEL_PATH_OVERRIDE = 'Qwen/Qwen2.5-0.5B'.")
# --- end memory-safe training attempts ---
'''

# the orchestration loops (old text, per notebook) and their replacements
COLAB_LOOP_OLD_START = "    for batch_size, ckpt in LADDER:  # (micro-batch, gradient checkpointing), fastest first; every entry gives an effective batch of 16\n"
COLAB_LOOP_OLD_END = '''        raise RuntimeError("out of memory even at batch size 2; lower MAX_LEN or TOKEN_BUDGET")\n'''
COLAB_LOOP_NEW = '''    def make_trainer_for(batch_size, ckpt, accum, callbacks):
        t = build_trainer(batch_size, accum, ckpt)
        for cb in callbacks:
            t.add_callback(cb)
        return t

    trainer, batch_size, ckpt, accum = run_ladder(make_trainer_for, resume, 1)
'''

KAGGLE_LOOP_OLD_START = "        for batch_size, ckpt in LADDER:  # (per-device micro-batch, gradient checkpointing); all give an effective batch of 16\n"
KAGGLE_LOOP_OLD_END = '''            raise RuntimeError("out of memory even at batch size 2; lower MAX_LEN or TOKEN_BUDGET")\n'''
KAGGLE_LOOP_NEW = '''        def make_trainer_for(batch_size, ckpt, accum, callbacks):
            return build_trainer(batch_size, N_GPU, [progress] + callbacks, ckpt)

        trainer, batch_size, ckpt, accum = run_ladder(make_trainer_for, resume, N_GPU)
'''


def sub(text, old, new, count=1):
    assert text.count(old) == count, f"expected {count} match(es), found {text.count(old)} for: {old[:100]!r}"
    return text.replace(old, new)


def cut(text, start, end, new):
    """Replace everything from `start` up to and including `end` by `new`."""
    i = text.index(start)
    j = text.index(end, i) + len(end)
    assert text.count(start) == 1
    return text[:i] + new + text[j:]


def _cells(nb):
    return [c for c in nb["cells"] if c["cell_type"] == "code"]


def _src(c):
    return "".join(c["source"])


def _put(c, text):
    c["source"] = text.splitlines(keepends=True)


BENCH_OLD_COLAB = "        benchmark_report(trainer, 1, batch_size, accum, ckpt)  # prints, then stops: nothing is saved\n"
BENCH_NEW_COLAB = ("        try:\n            benchmark_report(trainer, 1, batch_size, accum, ckpt)  # prints, then stops: nothing is saved\n"
                   "        finally:  # free the GPU: a later run of this cell in the same session must not start next to this attempt's model and optimizer\n"
                   "            trainer = None\n            release_gpu()\n")
BENCH_OLD_KAGGLE = "            benchmark_report(trainer, N_GPU, batch_size, accum, ckpt)  # prints, then stops: nothing is saved\n"
BENCH_NEW_KAGGLE = ("            try:\n                benchmark_report(trainer, N_GPU, batch_size, accum, ckpt)  # prints, then stops: nothing is saved\n"
                    "            finally:  # free the GPU: a later run of this cell in the same session must not start next to this attempt's model and optimizer\n"
                    "                trainer = None\n                release_gpu()\n")

MD_REPLACEMENTS = [
    ("gradient checkpointing only as the out-of-memory fallback (ladder: 2, 1 without it, then 8, 4, 2 with it)",
     "gradient checkpointing ON by default with micro-batch 2 (`GRADIENT_CHECKPOINTING`, `MICRO_BATCH` in the first cell; the retry ladder only goes down, 2 then 1, and a failed attempt is cleaned up before the next one, D-029)"),
    ("If batch 8 runs out of memory the cell retries at 4 x 4 and 2 x 8, keeping the effective batch at 16.",
     "The cell starts at micro-batch 2 (`MICRO_BATCH`) with gradient checkpointing on and retries at 1 if it runs out of memory, keeping the effective batch at 16 (D-029)."),
]


TRAIN_INFO_KEYS = ('"micro_batch": batch_size, "grad_accum": accum, "gradient_checkpointing": ckpt, "attempts": ATTEMPTS, '
                   '"peak_gpu_gib": (round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if torch.cuda.is_available() else None),')


def _patch_common(first, train):
    """The edits both notebooks share (first cell flags, ladder, helpers, attention implementation). Returns the new texts."""
    first = sub(first, OLD_FLAG_LINE, FLAG_LINES)
    train = cut(train, LADDER_OLD_START, LADDER_OLD_END, LADDER_NEW)
    # the helpers go right before the `assert torch.cuda.is_available()` line that follows the length-bucketing block
    marker = "assert torch.cuda.is_available() or SMOKE"
    i = train.index(marker)
    train = train[:i] + ATTEMPT_CODE.lstrip("\n") + "\n" + train[i:]
    train = sub(train, "        quantization_config=bnb, device_map={\"\": 0}, trust_remote_code=True,\n        torch_dtype=torch.float16)",
                "        quantization_config=bnb, device_map={\"\": 0}, trust_remote_code=True, attn_implementation=\"sdpa\",  # sdpa explicitly: an eager-attention fallback needs ~2 GiB more per 4,096-token row\n        torch_dtype=torch.float16)")
    train = sub(train, '"micro_batch": batch_size, "grad_accum": accum, "gradient_checkpointing": ckpt,', TRAIN_INFO_KEYS)
    return first, train


def patch_colab(nb):
    """Patch the Colab notebook object in place. Returns True when something changed (False: already patched)."""
    cells = _cells(nb)
    first, train = _src(cells[0]), _src(cells[_train_index(cells)])
    if MARKER in train:
        return False
    first, train = _patch_common(first, train)
    train = cut(train, COLAB_LOOP_OLD_START, COLAB_LOOP_OLD_END, COLAB_LOOP_NEW)
    train = sub(train, BENCH_OLD_COLAB, BENCH_NEW_COLAB)
    for c in nb["cells"]:
        if c["cell_type"] == "markdown":
            t = "".join(c["source"])
            for a, b in MD_REPLACEMENTS:
                if a in t:
                    t = t.replace(a, b)
            c["source"] = t.splitlines(keepends=True)
    # release the training objects before the evaluation cell loads a second model; the evaluation cell's own batch is set in the Kaggle / Colab evaluation text below
    _put(cells[0], first)
    _put(cells[_train_index(cells)], train)
    ev = cells[_train_index(cells) + 1]
    t = _src(ev)
    if "def predict_probs(texts, batch_size=8):" in t:
        t = sub(t, "def predict_probs(texts, batch_size=8):", "def predict_probs(texts, batch_size=4):  # 4 rows of up to 4,096 tokens, forward only")
        _put(ev, t)
    return True


def _train_index(cells):
    idx = [i for i, c in enumerate(cells) if "def build_trainer(" in _src(c)]
    assert len(idx) == 1, idx
    return idx[0]


def apply_kaggle(nb):
    """Patch the generated Kaggle notebook object: the config cell flags, and the orchestration loop. (The ladder, helpers and `sdpa` come from the patched Colab text.)"""
    cells = _cells(nb)
    first = _src(cells[0])
    assert OLD_FLAG_LINE in first, "Kaggle configuration cell: GRADIENT_CHECKPOINTING line not found"
    first = sub(first, OLD_FLAG_LINE, FLAG_LINES)
    _put(cells[0], first)
    ti = _train_index(cells)
    train = _src(cells[ti])
    assert MARKER in train, "the Colab notebook was not patched before the Kaggle one was generated"
    train = cut(train, KAGGLE_LOOP_OLD_START, KAGGLE_LOOP_OLD_END, KAGGLE_LOOP_NEW)
    train = sub(train, BENCH_OLD_KAGGLE, BENCH_NEW_KAGGLE)
    train = sub(train, '"micro_batch": batch_size, "grad_accum": accum, "gradient_checkpointing": ckpt,', TRAIN_INFO_KEYS)
    # free the training objects before the final evaluation loads its own model: training is complete or stopped, the adapter is saved
    train = sub(train, '            print("removed the Trainer checkpoints (training is complete)")\n',
                '            print("removed the Trainer checkpoints (training is complete)")\n'
                '        trainer = None  # the evaluation cell loads its own copy of the model: release the training model, optimizer and gradients first\n'
                '        release_gpu()\n        gpu_mem("after training, before the evaluation")\n')
    _put(cells[ti], train)
    ev = cells[ti + 1]
    t = _src(ev)
    if "def predict_probs(texts, batch_size=8):" in t:
        t = sub(t, "def predict_probs(texts, batch_size=8):", "def predict_probs(texts, batch_size=4):  # 4 rows of up to 4,096 tokens, forward only")
        _put(ev, t)


def main():
    path = NB / "phase6_qwen.ipynb"
    nb = json.load(open(path))
    if patch_colab(nb):
        json.dump(nb, open(path, "w"), indent=1, ensure_ascii=False)
        open(path, "a").write("\n")
        print("patched", path)
    else:
        print("phase6_qwen.ipynb is already patched")


if __name__ == "__main__":
    main()
