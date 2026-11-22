"""Source text of the length-bucketing code that add_speed_fixes.py puts into the notebooks (D-020, speed fix).

SAMPLER_CODE is plain Python (no torch), so the same text is exec'd by the unit checks and by the padding-waste measurement. TRAINER_CODE
needs transformers and torch and is only defined inside the notebooks.
"""

SAMPLER_CODE = '''\
# --- length bucketing (D-020): batches of rows of similar length, so padding is not wasted ---
import random as _random


def padded_tokens(lengths, batches, multiple=8):
    """Tokens the model processes for these batches: every row is padded to the longest row of its batch (rounded up to `multiple`)."""
    total = 0
    for b in batches:
        longest = max(lengths[i] for i in b)
        total += len(b) * (-(-longest // multiple) * multiple)
    return total


class LengthBucketSampler:
    """transformers 5.x removed `group_by_length`, so this does it. Each epoch: a fresh random permutation is cut into chunks of
    `batch_size * mega` rows, each chunk is sorted by length and cut into batches, and then ALL batches are shuffled, so a batch holds
    rows of similar length and the order between batches is random. The batch with the longest row goes first: an out-of-memory error
    shows at step 1, and every epoch has the same worst-case batch shape. Yields row indices; every run of `batch_size` consecutive
    indices is one batch (a smaller leftover batch is last)."""

    def __init__(self, lengths, batch_size, seed=42, mega=50):
        self.lengths, self.batch_size, self.seed, self.mega, self.epoch = list(lengths), batch_size, seed, mega, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def batches(self, epoch=None):
        epoch = self.epoch if epoch is None else epoch
        rng = _random.Random(self.seed * 1000003 + epoch)
        idx = list(range(len(self.lengths)))
        rng.shuffle(idx)
        chunk, bs, out = self.batch_size * self.mega, self.batch_size, []
        for s in range(0, len(idx), chunk):
            part = sorted(idx[s:s + chunk], key=lambda i: -self.lengths[i])
            out += [part[j:j + bs] for j in range(0, len(part), bs)]
        full, rest = [b for b in out if len(b) == bs], [b for b in out if len(b) < bs]
        rng.shuffle(full)
        if full:
            k = max(range(len(full)), key=lambda j: max(self.lengths[i] for i in full[j]))
            full.insert(0, full.pop(k))
        rest = [i for b in rest for i in b]
        return full + ([rest] if rest else [])  # the leftover rows (fewer than one batch) form the last batch

    def __iter__(self):
        for b in self.batches(self.epoch):
            yield from b
        self.epoch += 1

    def __len__(self):
        return len(self.lengths)


class LengthSortedSampler:
    """Evaluation order: shortest rows first, so every eval batch has rows of about the same length (the metrics do not depend on the order)."""

    def __init__(self, lengths):
        self.order = sorted(range(len(lengths)), key=lambda i: lengths[i])

    def __iter__(self):
        return iter(self.order)

    def __len__(self):
        return len(self.order)
'''


# Lines added to the first cell of the Colab notebooks (and to the Kaggle configuration cell).
FIRST_CELL_SPEED_LINES = '''BENCHMARK_STEPS = None  # e.g. 100: train that many optimizer steps, print seconds per step and the projected time of a full run, then stop WITHOUT saving (about 15 minutes on a T4)
GRADIENT_CHECKPOINTING = None  # None: try without it first (faster) and switch it on after an out-of-memory error. True / False forces it
'''

# Shared by Phase 5 and Phase 6; needs `Trainer`, `EFFECTIVE_BATCH`, `EPOCHS`, `train_ds`, `val_ds` (set later in the cell), `math`, `torch`.
TRAINER_CODE = '''\
import statistics
import time
from transformers import TrainerCallback


def row_lengths(ds):
    """Token count of every row of a tokenized dataset (a plain dataset object with .input_ids, or a Hugging Face Dataset)."""
    ids = ds["input_ids"] if hasattr(ds, "column_names") else ds.input_ids
    return [len(x) for x in ids]


def padding_report(lengths, batch):
    """Padding waste of the training set at this loader batch size: random order (what the Trainer did before) against length-bucketed."""
    real, rnd = sum(lengths), 0
    for s in range(3):
        o = list(range(len(lengths)))
        _random.Random(s).shuffle(o)
        rnd += padded_tokens(lengths, [o[i:i + batch] for i in range(0, len(o), batch)]) / 3
    buck = padded_tokens(lengths, LengthBucketSampler(lengths, batch).batches(0))
    print(f"padding waste at loader batch {batch}: random order {1 - real / rnd:.1%} ({rnd:,.0f} tokens processed for {real:,} real) -> "
          f"length-bucketed {1 - real / buck:.1%} ({buck:,.0f}); an epoch now processes {buck / rnd:.0%} of the tokens it did before")


class BenchmarkCallback(TrainerCallback):
    """BENCHMARK_STEPS mode: times every optimizer step (data loading included) and stops training after `n` of them."""

    def __init__(self, n):
        self.n, self.times, self.t0 = n, [], None

    def on_step_begin(self, args, state, control, **kw):
        self.t0 = time.perf_counter()

    def on_step_end(self, args, state, control, **kw):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.times.append(time.perf_counter() - self.t0)
        if len(self.times) >= self.n:
            control.should_training_stop = True


class LengthBucketTrainer(Trainer):
    """Trainer with length-bucketed training batches and length-sorted evaluation batches (D-020). It overrides two hooks that exist in
    transformers 4.x and 5.x and does not use `group_by_length` (its name and place changed between versions)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bench = None
        if BENCHMARK_STEPS:
            self.bench = BenchmarkCallback(BENCHMARK_STEPS)
            self.add_callback(self.bench)

    def _get_train_sampler(self, *args, **kwargs):
        ds = args[0] if args and args[0] is not None else self.train_dataset
        return LengthBucketSampler(row_lengths(ds), self._train_batch_size, seed=self.args.seed)

    def _get_eval_sampler(self, eval_dataset, *args, **kwargs):
        return LengthSortedSampler(row_lengths(eval_dataset))


def best_eval_metrics(trainer):
    """Validation metrics of the epoch that load_best_model_at_end restored, read from the per-epoch evaluations (no extra pass)."""
    step = getattr(trainer.state, "best_global_step", None)
    for h in trainer.state.log_history:
        if "eval_f1" in h and step and h.get("step") == step:
            return {k: v for k, v in h.items() if k.startswith("eval_")}
    print("best epoch not found in the log history: evaluating once more")
    return {k: v for k, v in trainer.evaluate().items() if k.startswith("eval_")}


def benchmark_report(trainer, n_gpu, batch_size, accum, ckpt):
    """BENCHMARK_STEPS mode: seconds per step, the projected time of a full run, then stop without saving anything."""
    t = trainer.bench.times
    skip = min(2, len(t) - 1)  # step 1 holds the longest batch and the CUDA warm-up
    use = t[skip:]
    mean, se = statistics.fmean(use), (statistics.pstdev(use) / len(use) ** 0.5 if len(use) > 1 else 0.0)
    t0 = time.perf_counter()
    trainer.evaluate()
    eval_s = time.perf_counter() - t0
    steps = math.ceil(len(train_ds) / EFFECTIVE_BATCH) * EPOCHS
    saves = EPOCHS * 90  # writing a checkpoint with optimizer state takes about a minute or two
    total = mean * steps + eval_s * EPOCHS + saves
    peak = f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB peak GPU memory (one device)" if torch.cuda.is_available() else "no GPU"
    print("\\n" + "=" * 100)
    print(f"BENCHMARK: {len(t)} optimizer steps | {n_gpu} GPU(s) x micro-batch {batch_size} x accumulation {accum} = {EFFECTIVE_BATCH} | gradient checkpointing {'on' if ckpt else 'off'} | {peak}")
    print(f"  step 1 (longest batch, warm-up): {t[0]:.1f}s | steps {skip + 1}-{len(t)}: mean {mean:.2f}s +- {se:.2f}s per step, min {min(use):.2f}s, max {max(use):.2f}s")
    print(f"  {EFFECTIVE_BATCH / mean:.2f} rows/s | validation pass {eval_s:.0f}s (length-sorted)")
    print(f"  full run = {steps} steps ({len(train_ds)} rows x {EPOCHS} epochs): {steps} x {mean:.2f}s + {EPOCHS} x {eval_s:.0f}s validation + about {saves // 60} min of saves "
          f"= {total / 3600:.2f} h ({int(total):,} s)")
    budget = globals().get("TIME_BUDGET_HOURS")
    if budget:
        print(f"  that is {'inside' if total / 3600 < budget else 'OUTSIDE'} the {budget} h time budget of this notebook")
    print("=" * 100)
    raise SystemExit("BENCHMARK_STEPS mode: stopped after the benchmark, nothing was saved. Set BENCHMARK_STEPS = None for the real run.")
'''
