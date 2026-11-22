"""Speed fixes for the Phase 5 and Phase 6 Colab notebooks (D-020): length-bucketed batches, no gradient checkpointing unless needed,
length-sorted validation, no duplicate final validation pass, and a benchmark mode.

    python notebooks/add_speed_fixes.py             # patches the two Colab notebooks once (idempotent)
    python notebooks/make_kaggle_notebooks.py       # then regenerate the Kaggle notebooks

Why: the Kaggle v2 run (2 x T4, nn.DataParallel, random batches of 4, gradient checkpointing) took 7.75 hours; about half of the tokens it
processed were padding (50.5% on the 6,816 training rows), validation batches were in file order (43% padding) and ran four times.
Every replacement must match exactly once.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from length_bucketing_code import FIRST_CELL_SPEED_LINES, SAMPLER_CODE, TRAINER_CODE  # noqa: E402

NB = Path(__file__).resolve().parent
MARK = "# --- length bucketing (D-020) ---"


def sub(text, old, new):
    assert text.count(old) == 1, f"{text.count(old)} matches for {old[:90]!r}"
    return text.replace(old, new)


def speed_block(kind):
    if kind == "p5":
        ladder = '''# (per-device micro-batch, gradient checkpointing), fastest first; every entry gives an effective batch of 16. Gradient checkpointing recomputes the
# forward pass inside the backward pass (about 30% more compute) to save memory, so it is only used after an out-of-memory error without it. The longest
# batch comes first in every epoch (LengthBucketSampler), so a memory problem shows at step 1, not after hours.
LADDER = {None: [(4, False), (2, False), (2, True), (1, True)], True: [(2, True), (1, True)], False: [(4, False), (2, False)]}[GRADIENT_CHECKPOINTING]
EVAL_BATCH = 8  # validation is length-sorted and needs no gradients, so a larger batch costs little memory
'''
    else:
        ladder = '''# (per-device micro-batch, gradient checkpointing), fastest first; every entry gives an effective batch of 16. Gradient checkpointing recomputes the
# forward pass inside the backward pass (about 30% more compute) to save memory, so it is only used after an out-of-memory error without it. The longest
# batch comes first in every epoch (LengthBucketSampler), so a memory problem shows at step 1, not after hours.
LADDER = ([(8, False)] if SMOKE else {None: [(2, False), (1, False), (8, True), (4, True), (2, True)], True: [(8, True), (4, True), (2, True)],
                                      False: [(2, False), (1, False)]}[GRADIENT_CHECKPOINTING])
EVAL_BATCH = 4
'''
    return "\n" + MARK + "\n" + SAMPLER_CODE.split("\n", 1)[1] + "\n\n" + TRAINER_CODE + "\n" + ladder + "# --- end length bucketing ---\n"


def patch_code(text, kind):
    if MARK in text:  # already patched: refresh the inserted block only (it may have changed), the other edits stay
        i, j = text.index("\n" + MARK), text.index("# --- end length bucketing ---\n") + len("# --- end length bucketing ---\n")
        fresh = text[:i] + speed_block(kind) + text[j:]
        return fresh, fresh != text
    text = sub(text, 'TRAINER_DIR = CKPT_DIR / "trainer"\n', 'TRAINER_DIR = CKPT_DIR / "trainer"\n' + speed_block(kind))
    text = sub(text, "def build_trainer(batch_size, accum):\n", "def build_trainer(batch_size, accum, ckpt):\n")
    text = sub(text, "per_device_eval_batch_size=batch_size,", "per_device_eval_batch_size=EVAL_BATCH,")
    text = sub(text, 'eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,\n        load_best_model_at_end=True,',
               'eval_strategy=("no" if BENCHMARK_STEPS else "epoch"), save_strategy=("no" if BENCHMARK_STEPS else "epoch"), save_total_limit=2,\n'
               '        load_best_model_at_end=not BENCHMARK_STEPS,')
    if kind == "p5":
        text = sub(text, 'gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},',
                   'gradient_checkpointing=ckpt, gradient_checkpointing_kwargs={"use_reentrant": False},')
        text = sub(text, "return Trainer(model=make_model(), args=args,", "return LengthBucketTrainer(model=make_model(), args=args,")
        text = sub(text, '''    for batch_size, accum in [(2, 8), (1, 16)]:  # both give an effective batch of 16
        trainer = build_trainer(batch_size, accum)
        try:
            print(f"\\ntraining with batch_size={batch_size} x grad_accum={accum}")
''', '''    for batch_size, ckpt in LADDER:  # (micro-batch, gradient checkpointing), fastest first; every entry gives an effective batch of 16
        accum = EFFECTIVE_BATCH // batch_size
        trainer = build_trainer(batch_size, accum, ckpt)
        try:
            print(f"\\ntraining with batch_size={batch_size} x grad_accum={accum}, gradient checkpointing {'on' if ckpt else 'off'}")
            padding_report(row_lengths(train_ds), batch_size)
''')
        text = sub(text, '    trainer.save_model(str(BEST_DIR))          # the best epoch by val F1 (load_best_model_at_end)\n',
                   '    if BENCHMARK_STEPS:\n        benchmark_report(trainer, 1, batch_size, accum, ckpt)  # prints, then stops: nothing is saved\n'
                   '    trainer.save_model(str(BEST_DIR))          # the best epoch by val F1 (load_best_model_at_end)\n')
        text = sub(text, '"grad_accum": accum, "max_len": MAX_LEN,', '"grad_accum": accum, "gradient_checkpointing": ckpt, "max_len": MAX_LEN,')
    else:
        text = sub(text, "def make_peft_model():\n", "def make_peft_model(ckpt=True):\n")
        text = sub(text, "    if not SMOKE:  # casts the non-quantized weights to fp32 and enables gradient checkpointing (needed for 8 x 1536 tokens on 16 GB)\n"
                         "        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True,",
                   "    if not SMOKE:  # casts the non-quantized weights to fp32; gradient checkpointing only when `ckpt` (it is the memory fallback, D-020)\n"
                   "        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=ckpt,")
        text = sub(text, "gradient_checkpointing=not SMOKE,", "gradient_checkpointing=(ckpt and not SMOKE),")
        text = sub(text, "return Trainer(model=make_peft_model(), args=args,", "return LengthBucketTrainer(model=make_peft_model(ckpt), args=args,")
        text = sub(text, '''    for batch_size, accum in ([(8, 2)] if SMOKE else [(8, 2), (4, 4), (2, 8)]):  # all give an effective batch of 16
        trainer = build_trainer(batch_size, accum)
        try:
            print(f"\\ntraining with batch_size={batch_size} x grad_accum={accum}")
''', '''    for batch_size, ckpt in LADDER:  # (micro-batch, gradient checkpointing), fastest first; every entry gives an effective batch of 16
        accum = EFFECTIVE_BATCH // batch_size
        trainer = build_trainer(batch_size, accum, ckpt)
        try:
            print(f"\\ntraining with batch_size={batch_size} x grad_accum={accum}, gradient checkpointing {'on' if ckpt else 'off'}")
            padding_report(row_lengths(train_ds), batch_size)
''')
        text = sub(text, "    trainer.save_model(str(BEST_DIR))  # the best epoch by val F1",
                   "    if BENCHMARK_STEPS:\n        benchmark_report(trainer, 1, batch_size, accum, ckpt)  # prints, then stops: nothing is saved\n"
                   "    trainer.save_model(str(BEST_DIR))  # the best epoch by val F1")
        text = sub(text, '"micro_batch": batch_size, "grad_accum": accum, "max_len": MAX_LEN,', '"micro_batch": batch_size, "grad_accum": accum, "gradient_checkpointing": ckpt, "max_len": MAX_LEN,')
    text = sub(text, 'best = {k: v for k, v in trainer.evaluate().items() if k.startswith("eval_")}', "best = best_eval_metrics(trainer)")
    return text, True


MD_P5_OLD = ("AdamW, lr 2e-5, warmup 10%, 3 epochs, effective batch 16, `max_length=4096` with gradient checkpointing. The micro-batch is 2 with gradient "
             "accumulation 8; if the T4 runs out of memory it falls back to 1 x 16, keeping the effective batch at 16.")
MD_P5_NEW = ("AdamW, lr 2e-5, warmup 10%, 3 epochs, effective batch 16, `max_length=4096`. Speed (D-020): training batches are length-bucketed "
             "(`LengthBucketSampler`: rows of similar length share a batch and the batches are shuffled; it prints the padding waste before and after), validation "
             "batches are length-sorted, and gradient checkpointing is off unless the GPU runs out of memory: the micro-batch ladder is 4, then 2 without it, then 2 "
             "and 1 with it (`GRADIENT_CHECKPOINTING = True/False` forces one side). The effective batch is always 16. `BENCHMARK_STEPS = 100` trains 100 steps, "
             "prints seconds per step and the projected time of a full run, and stops without saving.")
MD_P6_OLD = "lr 2e-4, 3 epochs, effective batch 16 (8 x 2), 10% warmup, weight decay 0.01, fp16."
MD_P6_NEW = ("lr 2e-4, 3 epochs, effective batch 16 (micro-batch x accumulation), 10% warmup, weight decay 0.01, fp16. Speed (D-020): length-bucketed training "
             "batches, length-sorted validation, gradient checkpointing only as the out-of-memory fallback (ladder: 2, 1 without it, then 8, 4, 2 with it), and "
             "`BENCHMARK_STEPS` for a timing run that stops without saving.")


def patch_notebook(name, kind):
    path = NB / name
    nb = json.load(open(path))
    done = 0
    for c in nb["cells"]:
        s = "".join(c["source"])
        if c["cell_type"] == "code" and s.startswith("# --- final mode (D-023) ---") and "BENCHMARK_STEPS" not in s:
            s = sub(s, "\n# COLAB: pip install", "\n" + FIRST_CELL_SPEED_LINES + "# COLAB: pip install") if "INCLUDE_POLICY_OVERRIDE = None" in s else s
            c["source"] = s.splitlines(keepends=True)
            done += 1
        elif c["cell_type"] == "code" and "def build_trainer(batch_size, accum" in s:
            s2, changed = patch_code(s, kind)
            if changed:
                c["source"] = s2.splitlines(keepends=True)
                done += 1
        elif c["cell_type"] == "markdown":
            old, new = (MD_P5_OLD, MD_P5_NEW) if kind == "p5" else (MD_P6_OLD, MD_P6_NEW)
            if old in s:
                c["source"] = s.replace(old, new).splitlines(keepends=True)
                done += 1
    json.dump(nb, open(path, "w"), indent=1, ensure_ascii=False)
    open(path, "a").write("\n")
    print(f"{name}: {done} cells changed" + ("" if done else " (already patched)"))


if __name__ == "__main__":
    patch_notebook("phase5_modernbert.ipynb", "p5")
    patch_notebook("phase6_qwen.ipynb", "p6")
