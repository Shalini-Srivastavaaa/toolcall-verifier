"""Generate the Kaggle versions of the Phase 5 and Phase 6 notebooks from the Colab ones.

    python notebooks/make_kaggle_notebooks.py

Reads notebooks/phase5_modernbert.ipynb and notebooks/phase6_qwen.ipynb and writes
notebooks/kaggle_phase5_modernbert.ipynb and notebooks/kaggle_phase6_qwen.ipynb. Every change is an exact
string replacement that must match once, so if a Colab notebook changes in a place this script touches, the
script stops instead of producing a Kaggle notebook that has drifted. The model, data, input format, split,
hyperparameters and metrics are the Colab code, copied; what changes is environment handling, GPU handling,
progress and time reporting, resume from an earlier run, an eval-only mode and the eval report file (D-020).
"""

import json
import sys
from pathlib import Path

NB = Path(__file__).resolve().parent
sys.path.insert(0, str(NB))
from length_bucketing_code import FIRST_CELL_SPEED_LINES  # noqa: E402


def load(name):
    return json.load(open(NB / name))


def save(nb, name):
    json.dump(nb, open(NB / name, "w"), indent=1, ensure_ascii=False)
    open(NB / name, "a").write("\n")


def src(cell):
    return "".join(cell["source"])


def put(cell, text):
    cell["source"] = text.splitlines(keepends=True)


def sub(text, old, new):
    assert text.count(old) == 1, f"expected exactly one match ({text.count(old)} found) for: {old[:90]!r}"
    return text.replace(old, new)


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


# ------------------------------------------------------------------------------------------------ shared cells

FIRST_CELL_LINE = ('DATASET = "legacy"  # "legacy": the 7,580-row dataset of D-012 with the frozen split. "final": the builder\'s output (D-022): split taken from each row, INCLUDE_POLICY on\n'
                   'INCLUDE_POLICY_OVERRIDE = None  # None: the POLICY block follows DATASET ("final": on). True / False forces it, e.g. False to score a model that was trained without the block (D-023)\n') + FIRST_CELL_SPEED_LINES

CONFIG_AND_ENV = '''\
# ================= run configuration (the only cell you normally edit) =================
DATASET = "final"          # "final": the builder's output (files final_train/val/test/test_synth.jsonl in the data dataset), split taken from the rows,
                           #          INCLUDE_POLICY on (ModernBERT). "legacy": the 7,580-row dataset of D-012 with the frozen split (needs the 7 old files).
EVAL_ONLY = False          # True: skip training and evaluate EVAL_CHECKPOINT (needs no GPU; slow on CPU)
EVAL_CHECKPOINT = None     # folder of a saved model: ".../best", or a Trainer ".../trainer/checkpoint-N" folder. None = this run's best/
TIME_BUDGET_HOURS = 11.0   # measured from the start of this notebook; training stops, saves a checkpoint and goes on to evaluation
INCLUDE_POLICY_OVERRIDE = None  # None: the POLICY block (ModernBERT) follows DATASET ("final": on). False = off, e.g. to score a model that was trained without it
USE_BOTH_GPUS = False      # False (default): train on GPU 0 only. True: nn.DataParallel over both T4s; the first Kaggle run (2 x T4, DataParallel) took 7.75 h, see D-020. Use BENCHMARK_STEPS to compare
BENCHMARK_STEPS = None     # e.g. 100: train that many optimizer steps, print seconds per step and the projected time of a full run, then stop WITHOUT saving (about 15 minutes)
GRADIENT_CHECKPOINTING = None  # None: try without it first (faster) and switch it on after an out-of-memory error. True / False forces it
MODEL_PATH_OVERRIDE = None # folder with the base model, if Internet is off and the model is attached as an input
# ========================================================================================

import os, sys, time, json, shutil, subprocess
from pathlib import Path

if not USE_BOTH_GPUS:  # hide the second GPU before CUDA starts, so nothing can wrap the model in nn.DataParallel
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]

T0 = time.time()  # start of the notebook; the 12-hour Kaggle limit counts from here


def detect_env():
    if os.environ.get("TCV_ENV"):  # lets a laptop test pretend to be Kaggle
        return os.environ["TCV_ENV"]
    if os.path.exists("/kaggle/input") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        return "kaggle"
    try:
        import google.colab  # noqa: F401
        return "colab"
    except ImportError:
        return "local"


ENV = detect_env()
KAGGLE_INPUT = Path(os.environ.get("TCV_KAGGLE_INPUT", "/kaggle/input"))
KAGGLE_WORKING = Path(os.environ.get("TCV_KAGGLE_WORKING", "/kaggle/working"))
print("environment:", ENV)


def pip_install(*packages):
    """Non-interactive pip. A failure (for example Internet switched off) is reported, not fatal: the preinstalled packages are used."""
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", *packages],
                       capture_output=True, text=True, stdin=subprocess.DEVNULL)
    print("pip install:", "ok" if r.returncode == 0 else "FAILED, continuing with the preinstalled packages\\n" + r.stderr[-500:])


def find_data_dir(name="train_pool.jsonl", max_depth=6):
    """Folder under /kaggle/input that holds `name` (Kaggle may rename the dataset folder)."""
    base_depth = len(KAGGLE_INPUT.parts)
    for dirpath, dirnames, filenames in os.walk(KAGGLE_INPUT, followlinks=True):
        if len(Path(dirpath).parts) - base_depth >= max_depth:
            dirnames[:] = []
        if name in filenames:
            return Path(dirpath)
    raise FileNotFoundError(f"{name} not found under {KAGGLE_INPUT}: attach the toolcall-verifier-data dataset to this notebook")


def fmt_td(seconds):
    s = int(max(seconds, 0))
    return f"{s // 3600}h{(s % 3600) // 60:02d}m{s % 60:02d}s"


def restore_prior_run(ckpt_dir, copy_trainers=True):
    """Kaggle: continue from the output of an earlier run that is attached as an input (Add Input > Your Work > Notebooks; its working folder shows up under
    /kaggle/input/.../<notebook>/checkpoints/<phase>/). Copies that run's Trainer checkpoints into ckpt_dir/trainer (so the Trainer resumes; not with
    copy_trainers=False, which only looks for a finished model) and returns the folder of a finished model (`best/`) if it has one."""
    if ENV != "kaggle" or not KAGGLE_INPUT.exists():
        return None
    name, finished, trainers, all_finished = ckpt_dir.name, None, {}, []
    for dirpath, dirnames, _ in os.walk(KAGGLE_INPUT, followlinks=True):
        p = Path(dirpath)
        if p.name == "best" and p.parent.name == name:
            finished = p
            all_finished.append(p)
        elif p.name.startswith("checkpoint-") and p.parent.name == "trainer" and p.parent.parent.name == name \\
                and (p / "trainer_state.json").exists():
            trainers.setdefault(p.parent, []).append(p)
    best_dir = None
    if len(all_finished) > 1:  # several earlier runs attached: the most recently written one is used, and all are listed
        finished = max(all_finished, key=lambda q: (q / "adapter_config.json").stat().st_mtime if (q / "adapter_config.json").exists() else 0)
        print("several finished models are attached as inputs:", [str(q) for q in all_finished], "| using", finished)
    if finished is not None:
        info = finished / "train_info.json"
        complete = (finished / "config.json").exists() or (finished / "adapter_config.json").exists()
        if info.exists():
            complete = complete and json.load(open(info)).get("completed", True)
        if complete:
            best_dir = finished
            print("found a finished model from an earlier run:", finished)
    if trainers and best_dir is None and copy_trainers:
        def top_step(ps):
            return max(json.load(open(c / "trainer_state.json")).get("global_step", 0) for c in ps)
        parent, ps = max(trainers.items(), key=lambda kv: top_step(kv[1]))
        (ckpt_dir / "trainer").mkdir(parents=True, exist_ok=True)
        for c in ps:  # every kept checkpoint, so the best one the Trainer remembers exists too
            if not (ckpt_dir / "trainer" / c.name).exists():
                shutil.copytree(c, ckpt_dir / "trainer" / c.name)
        print("restored", sorted(c.name for c in ps), "from", parent, "to continue training")
    return best_dir


class Tee:
    """Print to the notebook and write the same text to a file, line by line, so a crash keeps what was printed.
    A complete file-like wrapper (D-029): libraries read `sys.stdout` for more than `write`; transformers' loading report calls `sys.stdout.isatty()` (and crashed the first
    Qwen evaluation with 'Tee object has no attribute isatty'). `isatty()` is False so that no colour codes reach the report file; everything else comes from the real stdout."""

    def __init__(self, path):
        self.path, self.stdout = Path(path), sys.stdout
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.path, "w")
        sys.stdout = self

    def write(self, s):
        self.stdout.write(s)
        self.file.write(s)
        self.file.flush()
        return len(s)

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def flush(self):
        self.stdout.flush()
        if not self.file.closed:  # a logging handler that still holds the Tee after close() must not raise at interpreter exit
            self.file.flush()

    def isatty(self):
        return False

    def writable(self):
        return True

    def readable(self):
        return False

    def seekable(self):
        return False

    def fileno(self):
        return self.stdout.fileno()  # the real stdout's descriptor (raises io.UnsupportedOperation where it has none, as the real stream would)

    @property
    def encoding(self):
        return getattr(self.stdout, "encoding", None) or "utf-8"

    @property
    def errors(self):
        return getattr(self.stdout, "errors", None) or "strict"

    @property
    def closed(self):
        return self.file.closed

    def __getattr__(self, name):  # anything else (buffer, line_buffering, newlines, ...) comes from the real stdout
        if name in ("stdout", "file", "path") or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.stdout, name)

    def close(self):
        sys.stdout = self.stdout
        self.file.close()
'''

PROGRESS_CALLBACK = '''\
from transformers import TrainerCallback


class RunProgress(TrainerCallback):
    """Prints elapsed time and an estimated finish at every log line and every epoch, and stops training (after saving a
    checkpoint) once the time budget since the start of the notebook is used up."""

    def __init__(self):
        self.stopped_for_time = False

    def on_train_begin(self, args, state, control, **kw):
        self.t_start, self.t_epoch, self.step0 = time.time(), time.time(), state.global_step
        print(f"training starts at step {state.global_step}/{state.max_steps}; time budget {TIME_BUDGET_HOURS}h from notebook start")

    def _eta(self, state):
        done = state.global_step - self.step0
        if done <= 0:
            return None
        return (time.time() - self.t_start) / done * (state.max_steps - state.global_step)

    def on_log(self, args, state, control, logs=None, **kw):
        if logs and "loss" in logs:
            eta = self._eta(state)
            tail = "" if eta is None else f" | est. {fmt_td(eta)} left, finish ~{time.strftime('%H:%M', time.localtime(time.time() + eta))}"
            print(f"step {state.global_step}/{state.max_steps} loss {logs['loss']:.4f} | elapsed {fmt_td(time.time() - T0)}{tail}", flush=True)

    def on_epoch_end(self, args, state, control, **kw):
        eta = self._eta(state)
        tail = "" if eta is None else f", est. finish {time.strftime('%H:%M', time.localtime(time.time() + eta))} ({fmt_td(eta)} left)"
        print(f"=== epoch {state.epoch:.0f} finished in {fmt_td(time.time() - self.t_epoch)} | total elapsed {fmt_td(time.time() - T0)}{tail}", flush=True)
        self.t_epoch = time.time()

    def on_step_end(self, args, state, control, **kw):
        if not self.stopped_for_time and time.time() - T0 > TIME_BUDGET_HOURS * 3600:
            self.stopped_for_time = True
            control.should_save = True
            control.should_training_stop = True
            print(f"!!! time budget of {TIME_BUDGET_HOURS}h reached at step {state.global_step}/{state.max_steps}: saving a checkpoint and stopping. "
                  "Attach this run's output as an input of a new run to continue.", flush=True)
'''

TRAIN_COMMON_FN = '''\
def training_complete(model_dir):
    """A saved model counts as finished unless its train_info.json says the run stopped early (time budget)."""
    if not ((model_dir / "config.json").exists() or (model_dir / "adapter_config.json").exists()):
        return False
    info = model_dir / "train_info.json"
    return json.load(open(info)).get("completed", True) if info.exists() else True


def micro_batch_plan(per_device, n_gpu):
    """accumulation steps that keep the effective batch at EFFECTIVE_BATCH: per-device batch x GPUs x accumulation."""
    n = max(n_gpu, 1)
    assert EFFECTIVE_BATCH % (per_device * n) == 0, f"{per_device} x {n} GPUs does not divide the effective batch {EFFECTIVE_BATCH}"
    return EFFECTIVE_BATCH // (per_device * n)
'''


def kaggle_header_md(title_line, intro):
    return md(f"""# {title_line} — Kaggle version

{intro}

**Run it on Kaggle** (unattended, "Save & Run All (Commit)"):
1. Accelerator **GPU T4 x2**, **Internet on** (the base model is downloaded from Hugging Face), and add the private dataset `toolcall-verifier-data` (the 7 files of `data/derived/`) as an input. The data folder is found by searching `/kaggle/input/` for `train_pool.jsonl`.
2. Everything is written to `/kaggle/working/`: the checkpoints, `test_real_predictions.jsonl` and `eval_report.txt` (all evaluation output).
3. **Continue a stopped run:** add the earlier version's output as an input of the new run. Its Trainer checkpoints are copied in and training resumes; a finished model there makes the notebook skip straight to evaluation.
4. **Evaluate only:** set `EVAL_ONLY = True` and `EVAL_CHECKPOINT` to a saved model folder in the configuration cell (a `best/` folder or a `trainer/checkpoint-N` folder; tokenizer files are not required).
5. Training stops, saves and evaluates when `TIME_BUDGET_HOURS` (11 h) of the 12-hour session have passed.
6. `DATASET = "final"` (default here) reads `final_train/val/test/test_synth.jsonl` from the data dataset, takes the split from each row, turns the policy block on (ModernBERT) and ends with the final report (AUROC with a 95% interval, recall at 5% and 10% false alarms, per class by origin, per domain, the synthetic test set, length and source baselines); `"legacy"` is the old 7,580-row setup.

The code is generated from the Colab notebook by `notebooks/make_kaggle_notebooks.py`; the model, data, input format, split, hyperparameters and metrics are the Colab ones.

---
""")


def eval_fallback_tokenizer(var):
    return f'''\
def load_eval_tokenizer(model_dir):
    """The tokenizer saved with the checkpoint, or the base model's tokenizer when the checkpoint has none."""
    if any((model_dir / f).exists() for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json")):
        return AutoTokenizer.from_pretrained(str(model_dir))
    print(f"no tokenizer files in {{model_dir}}: using the tokenizer of {{MODEL_NAME}}")
    return {var}
'''


# ------------------------------------------------------------------------------------------------ Phase 5

def build_phase5():
    import md_cleanup  # D-032: legacy banner on the Colab notebook (once, in place) and the final-setup text of the Kaggle one
    nb = load("phase5_modernbert.ipynb")
    if md_cleanup.patch_colab(nb, "p5"):
        save(nb, "phase5_modernbert.ipynb")
        print("added the legacy banner to phase5_modernbert.ipynb")
    cells = nb["cells"]
    c = {i: src(x) for i, x in enumerate(cells)}
    assert "# COLAB: pip install" in c[1] and c[3].startswith("import gc, json, os, random, re")
    assert c[5].startswith("from transformers import AutoTokenizer") and c[12].startswith("import math") and c[14].startswith("from sklearn.metrics import confusion_matrix")

    # install: no `!pip`, never interactive, never fatal
    put(cells[1], '# pip_install is defined in the configuration cell above; a failure there is reported, not fatal\n'
                  'pip_install("transformers>=4.48.0", "accelerate>=1.0.0", "scikit-learn")\n')

    # data cell: environment-aware paths
    old = '''try:
    from google.colab import drive
    drive.mount("/content/drive")
    ROOT = Path("/content/drive/MyDrive/toolcall-verifier")
except ImportError:  # not on Colab: use a local checkout, or set TCV_ROOT
    ROOT = Path(os.environ.get("TCV_ROOT", "."))

DATA = ROOT / "data" / "derived"
CKPT_DIR = ROOT / "checkpoints" / "phase5_v2"  # v1 checkpoints stay in checkpoints/phase5/
BEST_DIR = CKPT_DIR / "best"
'''
    new = '''if ENV == "kaggle":  # data from the attached dataset, everything written to /kaggle/working, no Drive
    DATA = find_data_dir("final_train.jsonl" if DATASET == "final" else "train_pool.jsonl")
    ROOT = KAGGLE_WORKING
    CKPT_DIR = KAGGLE_WORKING / "checkpoints" / "phase5_v2"
elif ENV == "colab":
    from google.colab import drive
    drive.mount("/content/drive")
    ROOT = Path("/content/drive/MyDrive/toolcall-verifier")
    DATA = ROOT / "data" / "derived"
    CKPT_DIR = ROOT / "checkpoints" / "phase5_v2"  # v1 checkpoints stay in checkpoints/phase5/
else:  # local checkout, or set TCV_ROOT
    ROOT = Path(os.environ.get("TCV_ROOT", "."))
    DATA = ROOT / "data" / "derived"
    CKPT_DIR = ROOT / "checkpoints" / "phase5_v2"
print("data:", DATA, "| checkpoints:", CKPT_DIR)
BEST_DIR = CKPT_DIR / "best"
'''
    c[3] = sub(c[3], old, new)
    put(cells[3], c[3])

    # base-model override (offline Kaggle)
    c[5] = sub(c[5], 'MODEL_NAME = "answerdotai/ModernBERT-base"\n',
               'MODEL_NAME = "answerdotai/ModernBERT-base"\nif MODEL_PATH_OVERRIDE:\n    MODEL_NAME = str(MODEL_PATH_OVERRIDE)\n')
    put(cells[5], c[5])

    # training cell
    t = c[12]
    t = sub(t, 'TRAINER_DIR = CKPT_DIR / "trainer"\n', 'TRAINER_DIR = CKPT_DIR / "trainer"\n' + PROGRESS_CALLBACK + "\n\n" + TRAIN_COMMON_FN)
    t = sub(t, '    return AutoModelForSequenceClassification.from_pretrained(\n        MODEL_NAME, num_labels=2, classifier_pooling="cls", id2label=ID2LABEL, label2id=LABEL2ID, attn_implementation="sdpa")\n',
            '    model = AutoModelForSequenceClassification.from_pretrained(\n        MODEL_NAME, num_labels=2, classifier_pooling="cls", id2label=ID2LABEL, label2id=LABEL2ID, attn_implementation="sdpa")\n'
            '    if hasattr(model.config, "reference_compile"):  # transformers 4.5x only: no torch.compile of the encoder (safer under DataParallel, unattended)\n'
            '        model.config.reference_compile = False\n    return model\n')
    t = sub(t, '''train_ds, val_ds = VerifierDataset(train_rows), VerifierDataset(val_rows)
print("train/val examples:", len(train_ds), len(val_ds), "| longest tokenized input:", max(len(x) for x in train_ds.input_ids))
''', '''if EVAL_ONLY:
    train_ds = val_ds = None
    print("EVAL_ONLY: training data is not tokenized")
else:
    train_ds, val_ds = VerifierDataset(train_rows), VerifierDataset(val_rows)
    print("train/val examples:", len(train_ds), len(val_ds), "| longest tokenized input:", max(len(x) for x in train_ds.input_ids))
''')
    head, tail = t.split("def build_trainer(batch_size, accum, ckpt):\n")
    body_old_start = tail.index("    steps = math.ceil")
    build_old = tail[body_old_start: tail.index("\n\n\nif (BEST_DIR")]
    build_new = build_old
    build_new = sub(build_new, "steps = math.ceil(len(train_ds) / (batch_size * accum)) * EPOCHS",
                    "accum = micro_batch_plan(batch_size, n_gpu)\n    steps = math.ceil(len(train_ds) / (batch_size * max(n_gpu, 1) * accum)) * EPOCHS")
    build_new = sub(build_new, 'report_to="none", seed=SEED)',
                    'report_to="none", seed=SEED,\n        disable_tqdm=(ENV == "kaggle"))  # progress lines come from RunProgress instead of a progress bar\n'
                    '    if torch.cuda.is_available() and n_gpu == 1 and args.n_gpu > 1:\n'
                    '        args._n_gpu = 1  # several GPUs are visible but this attempt must use GPU 0 only (no nn.DataParallel)')
    build_new = sub(build_new, "compute_metrics=compute_metrics)", "compute_metrics=compute_metrics, callbacks=callbacks)")
    new_orchestration = '''
N_VISIBLE = torch.cuda.device_count()
N_GPU = N_VISIBLE if (USE_BOTH_GPUS and N_VISIBLE > 1) else min(N_VISIBLE, 1)
print(f"GPUs visible: {N_VISIBLE}; training will use {max(N_GPU, 1)} (nn.DataParallel when 2: per-device batch x GPUs x accumulation = {EFFECTIVE_BATCH})")

TRAINING_COMPLETE = False
if EVAL_ONLY:
    print("EVAL_ONLY = True: skipping training")
else:
    prior_best = restore_prior_run(CKPT_DIR)
    if prior_best is not None:
        BEST_DIR, TRAINING_COMPLETE = prior_best, True
    elif training_complete(BEST_DIR):
        print(f"Found trained model at {BEST_DIR}; skipping training.")
        TRAINING_COMPLETE = True
    else:
        resume = get_last_checkpoint(str(TRAINER_DIR)) if TRAINER_DIR.exists() else None
        print("resuming from", resume) if resume else print("starting fresh")
        progress = RunProgress()
        trained = False
        for n_gpu in ([N_GPU, 1] if N_GPU > 1 else [1]):  # two GPUs first; on any non-memory error, one GPU
            for batch_size, ckpt in LADDER:  # (per-device micro-batch, gradient checkpointing); all combinations give an effective batch of 16
                accum = micro_batch_plan(batch_size, n_gpu)
                trainer = build_trainer(batch_size, n_gpu, [progress], ckpt)
                try:
                    print(f"\\ntraining with {n_gpu} GPU(s) x per-device batch {batch_size} x grad_accum {accum} = {EFFECTIVE_BATCH}, gradient checkpointing {'on' if ckpt else 'off'}")
                    padding_report(row_lengths(train_ds), batch_size * max(n_gpu, 1))
                    trainer.train(resume_from_checkpoint=resume)
                    trained = True
                    break
                except torch.cuda.OutOfMemoryError:
                    print("CUDA out of memory; retrying with a smaller micro-batch")
                except Exception as e:
                    if n_gpu > 1:
                        print(f"!!! two-GPU training failed ({type(e).__name__}: {str(e)[:300]}); FALLING BACK TO ONE GPU")
                        break
                    raise
                del trainer
                gc.collect()
                torch.cuda.empty_cache()
            if trained:
                break
        else:
            raise RuntimeError("out of memory even at batch size 1; lower MAX_LEN and TOKEN_BUDGET_V2")
        if BENCHMARK_STEPS:
            benchmark_report(trainer, n_gpu, batch_size, accum, ckpt)  # prints, then stops: nothing is saved
        TRAINING_COMPLETE = not progress.stopped_for_time

        trainer.save_model(str(BEST_DIR))          # the best epoch by val F1 (load_best_model_at_end)
        tokenizer.save_pretrained(str(BEST_DIR))
        best = best_eval_metrics(trainer)
        (BEST_DIR / "train_info.json").write_text(json.dumps({
            "val_metrics": best, "lr": LR, "epochs": EPOCHS, "effective_batch": EFFECTIVE_BATCH, "micro_batch": batch_size,
            "grad_accum": accum, "gradient_checkpointing": ckpt, "n_gpu": n_gpu, "max_len": MAX_LEN, "token_budget": TOKEN_BUDGET_V2, "input_format": "v2", "seed": SEED,
            "n_train": len(train_ds), "n_val": len(val_ds), "completed": TRAINING_COMPLETE,
            "elapsed_seconds": round(time.time() - T0), "log_history": trainer.state.log_history}, indent=1, default=str))
        print("\\nbest-epoch validation metrics:", {k: round(v, 4) for k, v in best.items() if isinstance(v, float)})
        print("saved model + tokenizer to", BEST_DIR)
        if not TRAINING_COMPLETE:
            print("TRAINING IS NOT COMPLETE (time budget). The saved model is the best epoch so far; attach this output to a new run to continue.")
        elif ENV == "kaggle" and TRAINER_DIR.exists():
            shutil.rmtree(TRAINER_DIR)  # optimizer states are about 1.8 GB per checkpoint; the finished model is in best/
            print("removed the Trainer checkpoints (training is complete)")
'''
    c[12] = head + "def build_trainer(batch_size, n_gpu, callbacks, ckpt):\n" + build_new + "\n" + new_orchestration
    put(cells[12], c[12])

    # evaluation cell
    e = c[14]
    e = sub(e, '''model = AutoModelForSequenceClassification.from_pretrained(str(BEST_DIR))
eval_tok = AutoTokenizer.from_pretrained(str(BEST_DIR))
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()
''', eval_fallback_tokenizer("tokenizer") + '''

EVAL_DIR = Path(EVAL_CHECKPOINT) if (EVAL_ONLY and EVAL_CHECKPOINT) else BEST_DIR
assert EVAL_DIR.exists(), f"nothing to evaluate: {EVAL_DIR} does not exist"
REPORT_PATH = (KAGGLE_WORKING if ENV == "kaggle" else CKPT_DIR) / "eval_report.txt"
tee = Tee(REPORT_PATH)  # from here on everything printed is also written to eval_report.txt
print(f"evaluation report | environment {ENV} | checkpoint {EVAL_DIR} | {time.strftime('%Y-%m-%d %H:%M:%S')}")
if not training_complete(EVAL_DIR) and not EVAL_ONLY:
    print("NOTE: this model comes from an unfinished run")
device = "cuda" if torch.cuda.is_available() else "cpu"
print("evaluating on", device, "(CPU is slow for 4,096-token inputs but gives the same scores up to float precision)" if device == "cpu" else "")
model = AutoModelForSequenceClassification.from_pretrained(str(EVAL_DIR))
eval_tok = load_eval_tokenizer(EVAL_DIR)
model.to(device).eval()
''')
    e = sub(e, 'print("\\nsaved", CKPT_DIR / PRED_FILE)',
            'print("\\nsaved", CKPT_DIR / PRED_FILE)\nprint("report written to", REPORT_PATH)\ntee.close()')
    put(cells[14], e)

    # configuration cell + Kaggle header (the Colab header is kept, folded)
    intro = kaggle_header_md("Phase 5 (v2): ModernBERT-base", "ModernBERT-base on **one T4** by default (`USE_BOTH_GPUS = False`): the first run on 2 x T4 with `nn.DataParallel` took 7.75 hours (D-020). `USE_BOTH_GPUS = True` brings back DataParallel through the Trainer; the effective batch stays 16 (per-device batch x GPUs x gradient accumulation) and a non-memory failure falls back to one GPU. Set `BENCHMARK_STEPS = 100` first: it times 100 steps and prints the projected time of the full run.")
    colab_header = md("<details><summary>Legacy Colab notes (the first-round header; describes the legacy 7,580-row setup)</summary>\n\n" + src(cells[0]) + "\n\n</details>\n")
    nb["cells"] = [intro, colab_header, code(CONFIG_AND_ENV)] + cells[1:]
    import real_cv_code  # D-025: the REAL_CV experiment flag (default off), the prediction-saving fix and the validation predictions
    real_cv_code.apply(nb)
    import real_weight_code  # D-026: REAL_WEIGHT (default 1 = off) and the real-origin validation report
    real_weight_code.apply(nb)
    import normalize_agent_text_code  # D-027: NORMALIZE_AGENT_TEXT (default off)
    normalize_agent_text_code.apply(nb)
    md_cleanup.apply_kaggle(nb, "p5")
    save(nb, "kaggle_phase5_modernbert.ipynb")


# ------------------------------------------------------------------------------------------------ Phase 6

def build_phase6():
    import qwen_oom_code  # D-029: OOM-safe retry ladder, MICRO_BATCH / GRADIENT_CHECKPOINTING flags; patches the Colab notebook once, in place, before the Kaggle one is derived from it
    nb = load("phase6_qwen.ipynb")
    if qwen_oom_code.patch_colab(nb):
        save(nb, "phase6_qwen.ipynb")
        print("patched phase6_qwen.ipynb (memory-safe retry ladder)")
    import md_cleanup
    if md_cleanup.patch_colab(nb, "p6"):
        save(nb, "phase6_qwen.ipynb")
        print("added the legacy banner to phase6_qwen.ipynb")
    cells = nb["cells"]
    c = {i: src(x) for i, x in enumerate(cells)}
    assert "# COLAB: pip install" in c[2] and c[4].startswith("import gc, json, os, random, re")
    assert c[6].startswith("from transformers import AutoTokenizer") and c[8].startswith("LR, EPOCHS") and c[10].startswith("from sklearn.metrics import confusion_matrix")

    c[2] = sub(c[2], '# --- final mode (D-023) ---\n' + sub(FIRST_CELL_LINE, qwen_oom_code.OLD_FLAG_LINE, qwen_oom_code.FLAG_LINES), "")  # the Colab first cell carries the Qwen flags of D-029 (MICRO_BATCH, GRADIENT_CHECKPOINTING = True)
    c[2] = sub(c[2], '!pip install -q transformers peft accelerate bitsandbytes datasets scikit-learn\n',
               'pip_install("transformers", "peft", "accelerate", "bitsandbytes", "datasets", "scikit-learn")  # defined in the configuration cell\n')
    put(cells[2], c[2])

    old = '''try:
    from google.colab import drive
    drive.mount("/content/drive")
    ROOT = Path("/content/drive/MyDrive/toolcall-verifier")
except ImportError:  # not on Colab: use a local checkout, or set TCV_ROOT
    ROOT = Path(os.environ.get("TCV_ROOT", "."))

DATA = ROOT / "data" / "derived"
CKPT_DIR = ROOT / "checkpoints" / "phase6"
BEST_DIR = CKPT_DIR / "best"
'''
    new = '''if ENV == "kaggle":  # data from the attached dataset, everything written to /kaggle/working, no Drive
    DATA = find_data_dir("final_train.jsonl" if DATASET == "final" else "train_pool.jsonl")
    ROOT = KAGGLE_WORKING
    CKPT_DIR = KAGGLE_WORKING / "checkpoints" / "phase6"
elif ENV == "colab":
    from google.colab import drive
    drive.mount("/content/drive")
    ROOT = Path("/content/drive/MyDrive/toolcall-verifier")
    DATA = ROOT / "data" / "derived"
    CKPT_DIR = ROOT / "checkpoints" / "phase6"
else:  # local checkout, or set TCV_ROOT
    ROOT = Path(os.environ.get("TCV_ROOT", "."))
    DATA = ROOT / "data" / "derived"
    CKPT_DIR = ROOT / "checkpoints" / "phase6"
print("data:", DATA, "| checkpoints:", CKPT_DIR)
BEST_DIR = CKPT_DIR / "best"
'''
    c[4] = sub(c[4], old, new)
    put(cells[4], c[4])

    c[6] = sub(c[6], 'MODEL_NAME = "Qwen/Qwen2.5-1.5B"\n', 'MODEL_NAME = "Qwen/Qwen2.5-1.5B"\nif MODEL_PATH_OVERRIDE:\n    MODEL_NAME = str(MODEL_PATH_OVERRIDE)\n')
    put(cells[6], c[6])

    t = c[8]
    t = sub(t, 'assert torch.cuda.is_available() or SMOKE, "Select a GPU runtime (Runtime > Change runtime type > T4 GPU): 4-bit training needs CUDA"\n',
            'assert torch.cuda.is_available() or SMOKE or EVAL_ONLY, "no GPU: 4-bit training needs CUDA (switch the accelerator on, or set EVAL_ONLY = True)"\n')
    t = sub(t, 'TRAINER_DIR = CKPT_DIR / "trainer"\n', 'TRAINER_DIR = CKPT_DIR / "trainer"\n' + PROGRESS_CALLBACK + "\n\n" + TRAIN_COMMON_FN)
    # base model: 4-bit on a GPU, plain fp32 on a CPU (evaluation only)
    t = sub(t, '''    bnb = BitsAndBytesConfig(load_in_4bit=True''', '''    if not torch.cuda.is_available():  # CPU (eval only): bitsandbytes needs CUDA, so the backbone stays unquantized fp32
        print("no GPU: loading the backbone unquantized in fp32 on the CPU; scores can differ slightly from the 4-bit GPU evaluation")
        return AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID, pad_token_id=tokenizer.pad_token_id,
            trust_remote_code=True, torch_dtype=torch.float32)
    bnb = BitsAndBytesConfig(load_in_4bit=True''')
    t = sub(t, '''train_ds, val_ds = encode(train_rows), encode(val_rows)
print("train/val examples:", len(train_ds), len(val_ds), "| longest tokenized input:", max(len(x) for x in train_ds["input_ids"]), "tokens")
''', '''if EVAL_ONLY:
    train_ds = val_ds = None
    print("EVAL_ONLY: training data is not tokenized")
else:
    train_ds, val_ds = encode(train_rows), encode(val_rows)
    print("train/val examples:", len(train_ds), len(val_ds), "| longest tokenized input:", max(len(x) for x in train_ds["input_ids"]), "tokens")
''')
    head, tail = t.split("def build_trainer(batch_size, accum, ckpt):\n")
    build_old = tail[tail.index("    steps = math.ceil"): tail.index("\n\n\ndef print_history")]
    build_new = sub(build_old, "steps = math.ceil(len(train_ds) / (batch_size * accum)) * EPOCHS",
                    "accum = micro_batch_plan(batch_size, n_gpu)\n    steps = math.ceil(len(train_ds) / (batch_size * max(n_gpu, 1) * accum)) * EPOCHS")
    build_new = sub(build_new, 'report_to="none", seed=SEED)',
                    'report_to="none", seed=SEED,\n        disable_tqdm=(ENV == "kaggle"))  # progress lines come from RunProgress instead of a progress bar\n'
                    '    if torch.cuda.is_available() and n_gpu == 1 and args.n_gpu > 1:\n'
                    '        args._n_gpu = 1  # several GPUs are visible but a 4-bit model is trained on GPU 0 only (see below)')
    build_new = sub(build_new, "compute_metrics=compute_metrics)", "compute_metrics=compute_metrics, callbacks=callbacks)")
    mid_and_rest = tail[tail.index("\n\n\ndef print_history"):]
    hist_end = mid_and_rest.index('\n\n\nif (BEST_DIR / "adapter_config.json").exists():')
    print_history_fn = mid_and_rest[:hist_end]
    new_orchestration = '''
N_VISIBLE = torch.cuda.device_count()
# transformers does not wrap a 4-bit quantized model in nn.DataParallel (Trainer: "quantized models do not support DP"), and
# bitsandbytes 4-bit weights cannot be replicated across devices, so this notebook trains on GPU 0 and the second T4 stays idle.
N_GPU = 1
if N_VISIBLE > 1:
    print(f"{N_VISIBLE} GPUs visible, but 4-bit QLoRA training uses ONE GPU (GPU 0): nn.DataParallel does not support bitsandbytes 4-bit models. "
          "Effective batch stays 16 = per-device batch x 1 GPU x gradient accumulation.")

TRAINING_COMPLETE = False
if EVAL_ONLY:
    print("EVAL_ONLY = True: skipping training")
else:
    prior_best = restore_prior_run(CKPT_DIR)
    if prior_best is not None:
        BEST_DIR, TRAINING_COMPLETE = prior_best, True
    elif training_complete(BEST_DIR):
        print(f"Found a trained adapter at {BEST_DIR}; skipping training.")
        TRAINING_COMPLETE = True
        if (BEST_DIR / "train_info.json").exists():
            print_history(json.load(open(BEST_DIR / "train_info.json"))["log_history"])
    else:
        resume = get_last_checkpoint(str(TRAINER_DIR)) if TRAINER_DIR.exists() else None
        print("resuming from", resume) if resume else print("starting fresh")
        progress = RunProgress()
        for batch_size, ckpt in LADDER:  # (per-device micro-batch, gradient checkpointing); all give an effective batch of 16
            accum = micro_batch_plan(batch_size, N_GPU)
            trainer = build_trainer(batch_size, N_GPU, [progress], ckpt)
            try:
                print(f"\\ntraining with {N_GPU} GPU x per-device batch {batch_size} x grad_accum {accum} = {EFFECTIVE_BATCH}, gradient checkpointing {'on' if ckpt else 'off'}")
                padding_report(row_lengths(train_ds), batch_size * N_GPU)
                trainer.train(resume_from_checkpoint=resume)
                break
            except torch.cuda.OutOfMemoryError:
                print("CUDA out of memory; retrying with a smaller micro-batch")
                del trainer
                gc.collect()
                torch.cuda.empty_cache()
        else:
            raise RuntimeError("out of memory even at batch size 2; lower MAX_LEN or TOKEN_BUDGET")
        if BENCHMARK_STEPS:
            benchmark_report(trainer, N_GPU, batch_size, accum, ckpt)  # prints, then stops: nothing is saved
        TRAINING_COMPLETE = not progress.stopped_for_time

        trainer.save_model(str(BEST_DIR))  # the best epoch by val F1 (load_best_model_at_end): LoRA adapter + trained head
        tokenizer.save_pretrained(str(BEST_DIR))
        history = list(trainer.state.log_history)  # per-epoch history, taken before evaluate() below logs one more eval entry
        best = best_eval_metrics(trainer)
        (BEST_DIR / "train_info.json").write_text(json.dumps({
            "model": MODEL_NAME, "val_metrics": best, "lr": LR, "epochs": EPOCHS, "effective_batch": EFFECTIVE_BATCH,
            "micro_batch": batch_size, "grad_accum": accum, "gradient_checkpointing": ckpt, "n_gpu": N_GPU, "max_len": MAX_LEN, "token_budget": TOKEN_BUDGET, "seed": SEED,
            "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "targets": ["q_proj", "v_proj"]},
            "n_train": len(train_ds), "n_val": len(val_ds), "completed": TRAINING_COMPLETE,
            "elapsed_seconds": round(time.time() - T0), "log_history": history}, indent=1, default=str))
        print()
        print_history(history)
        print("\\nsaved adapter + tokenizer to", BEST_DIR)
        if not TRAINING_COMPLETE:
            print("TRAINING IS NOT COMPLETE (time budget). The saved adapter is the best epoch so far; attach this output to a new run to continue.")
        elif ENV == "kaggle" and TRAINER_DIR.exists():
            shutil.rmtree(TRAINER_DIR)  # the finished adapter is in best/
            print("removed the Trainer checkpoints (training is complete)")
'''
    c[8] = head + "def build_trainer(batch_size, n_gpu, callbacks, ckpt):\n" + build_new + print_history_fn + "\n" + new_orchestration
    put(cells[8], c[8])

    e = c[10]
    e = sub(e, '''eval_tok = AutoTokenizer.from_pretrained(str(BEST_DIR))
eval_tok.padding_side = "right"
model = PeftModel.from_pretrained(load_base_model(), str(BEST_DIR))  # 4-bit backbone + the saved adapter and head
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"  # the quantized model is already on the GPU
''', eval_fallback_tokenizer("tokenizer") + '''

if EVAL_ONLY and not EVAL_CHECKPOINT:  # auto-detect (D-029): the finished best/ of an earlier run attached as an input (Add Input > Your Work > Notebooks)
    _found = restore_prior_run(CKPT_DIR, copy_trainers=False)
    if _found is not None:
        BEST_DIR = _found
EVAL_DIR = Path(EVAL_CHECKPOINT) if (EVAL_ONLY and EVAL_CHECKPOINT) else BEST_DIR
assert EVAL_DIR.exists(), (f"nothing to evaluate: {EVAL_DIR} does not exist. EVAL_ONLY needs a finished model: attach the notebook that trained it (Add Input > Your Work > Notebooks; "
                           f"its output holds checkpoints/{CKPT_DIR.name}/best/) or set EVAL_CHECKPOINT to a folder with an adapter. Folders found under {KAGGLE_INPUT}: "
                           f"{sorted(str(q) for q in KAGGLE_INPUT.glob('*'))[:20] if KAGGLE_INPUT.exists() else 'none'}")
REPORT_PATH = (KAGGLE_WORKING if ENV == "kaggle" else CKPT_DIR) / "eval_report.txt"
tee = Tee(REPORT_PATH)  # from here on everything printed is also written to eval_report.txt
print(f"evaluation report | environment {ENV} | checkpoint {EVAL_DIR} | {time.strftime('%Y-%m-%d %H:%M:%S')}")
if not training_complete(EVAL_DIR) and not EVAL_ONLY:
    print("NOTE: this adapter comes from an unfinished run")
eval_tok = load_eval_tokenizer(EVAL_DIR)
eval_tok.padding_side = "right"
if eval_tok.pad_token is None:
    eval_tok.pad_token = eval_tok.eos_token
model = PeftModel.from_pretrained(load_base_model(), str(EVAL_DIR))  # backbone + the saved adapter and head
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"  # the quantized model is already on the GPU
''')
    e = sub(e, 'print("\\nsaved", CKPT_DIR / PRED_FILE)',
            'print("\\nsaved", CKPT_DIR / PRED_FILE)\nprint("report written to", REPORT_PATH)\ntee.close()')
    put(cells[10], e)

    intro = kaggle_header_md("Phase 6: Qwen2.5-1.5B + LoRA", "Qwen2.5-1.5B (4-bit QLoRA) on Kaggle. Both T4s are visible, but a bitsandbytes 4-bit model cannot be trained with `nn.DataParallel`, so this notebook trains on one GPU and prints a message saying so; the effective batch stays 16.")
    colab_header = md("<details><summary>Legacy Colab notes (the first-round header; describes the legacy 7,580-row setup)</summary>\n\n" + src(cells[0]) + "\n\n</details>\n")
    nb["cells"] = [intro, colab_header, code(CONFIG_AND_ENV)] + cells[1:]
    import qwen_v2_code  # D-029: the v2 input with the policy block, REAL_WEIGHT and the validation report for the Qwen notebook
    qwen_v2_code.apply(nb, NB / "kaggle_phase5_modernbert.ipynb")
    qwen_oom_code.apply_kaggle(nb)
    md_cleanup.apply_kaggle(nb, "p6")
    save(nb, "kaggle_phase6_qwen.ipynb")


if __name__ == "__main__":
    build_phase5()
    build_phase6()
    print("wrote kaggle_phase5_modernbert.ipynb and kaggle_phase6_qwen.ipynb")
