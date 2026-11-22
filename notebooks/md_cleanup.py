"""Markdown fixes for the four training notebooks (D-032, part A): the Colab originals are marked as LEGACY, and the generated Kaggle notebooks get the header and cell
descriptions of the final setup (DATASET = "final", the v3b data, one T4) instead of the first-round Colab text.

patch_colab(nb, which)     in place on phase5_modernbert.ipynb / phase6_qwen.ipynb, idempotent (a sentinel in the first cell)
apply_kaggle(nb, which)    on the generated Kaggle notebook object; `which` is "p5" (ModernBERT) or "p6" (Qwen)
kaggle_header(which)       the header cell text of the Kaggle notebooks
"""

SENTINEL = "**Legacy notebook.**"

COLAB_BANNER = (
    "> **Legacy notebook.** This is the first-round Colab version (legacy dataset: the 7,580 rows of `data/derived/`, evaluated on the 102-row `test_real.jsonl`). "
    "The final models (ModernBERT v3b, test AUROC 0.771; Qwen2.5-1.5B, 0.765) were trained with the Kaggle versions of the notebooks, `kaggle_phase5_modernbert.ipynb` and "
    "`kaggle_phase6_qwen.ipynb` (generated from this notebook by `make_kaggle_notebooks.py`), with `DATASET = \"final\"` on the v3b data (7,154 training rows, `REAL_WEIGHT = 3`, policy block on, "
    "4,096 tokens, one T4). The rest of this header describes the legacy setup. See `docs/kaggle_runbook.md`.\n\n"
)

COLAB_FIXES = {
    "p6": [
        ("# Phase 6 — Qwen2.5-1.5B + LoRA. Same data and eval protocol as Phase 5 (ModernBERT-base). AUROC 0.652 is the baseline to beat.",
         "# Phase 6 — Qwen2.5-1.5B + LoRA. Same data and eval protocol as Phase 5 (ModernBERT-base)."),
        ("- Training data has no Class 7 (dropped, see D-009); `test_real` still has 8 real Class 7 rows, so expect near-zero recall on them.",
         "- The legacy training data has no Class 7 (synthetic Class 7 examples were removed, D-009); `test_real` has 8 real Class 7 rows, so expect near-zero recall on them. The final dataset has Class 7 (state flips and real errors, D-021, D-026, D-028)."),
    ],
    "p5": [],
}


ALWAYS_COLAB_FIXES = {
    "p6": [("is evaluation-only and is not part of the 7,385;", "is evaluation-only and is not part of the 7,580 legacy training rows;")],
    "p5": [],
}


def patch_colab(nb, which):
    changed = False
    cell = nb["cells"][0]
    text = "".join(cell["source"])
    if SENTINEL not in text:
        lines = text.split("\n", 1)
        text = lines[0] + "\n\n" + COLAB_BANNER + (lines[1].lstrip("\n") if len(lines) > 1 else "")
        for old, new in COLAB_FIXES[which]:
            assert old in text, old[:60]
            text = text.replace(old, new)
        cell["source"] = text.splitlines(keepends=True)
        changed = True
    for old, new in ALWAYS_COLAB_FIXES[which]:
        for c in nb["cells"]:
            if c["cell_type"] == "markdown" and old in "".join(c["source"]):
                c["source"] = "".join(c["source"]).replace(old, new).splitlines(keepends=True)
                changed = True
    return changed


def kaggle_header(which):
    if which == "p5":
        title = "ModernBERT-base verifier — Kaggle version (final model: v3b)"
        intro = ("Final model: **ModernBERT v3b** (test AUROC 0.771, D-029). ModernBERT-base on **one T4** (`USE_BOTH_GPUS = False`; the first run on 2 x T4 with `nn.DataParallel` took 7.75 hours, D-020). "
                 "Settings of the v3b run: `DATASET = \"final\"`, `REAL_WEIGHT = 3`, `NORMALIZE_AGENT_TEXT = False`, `REAL_CV = False`, policy block on, 4,096 tokens, 3 epochs, lr 2e-5, effective batch 16; about 1.9 hours. "
                 "Set `BENCHMARK_STEPS = 100` first for a timing run.")
        outputs = "the model in `checkpoints/phase5_v2/best/`, `eval_report.txt`, `test_predictions.jsonl`, `test_synth_predictions.jsonl`, `val_predictions.jsonl` and `train_info.json` (legacy mode: `test_real_predictions.jsonl`)"
    else:
        title = "Qwen2.5-1.5B + LoRA — Kaggle version (comparison model)"
        intro = ("Comparison model: **Qwen2.5-1.5B** (4-bit QLoRA, test AUROC 0.765, D-029, D-030). A bitsandbytes 4-bit model cannot be trained with `nn.DataParallel`, so the notebook trains on one GPU. "
                 "Settings of the run: `DATASET = \"final\"`, `REAL_WEIGHT = 3`, `QWEN_MAX_LEN = 4096`, `QWEN_EPOCHS = 2`, `MICRO_BATCH = 2` with gradient checkpointing, policy block on, lr 2e-4, effective batch 16; about 6.8 hours. "
                 "Set `BENCHMARK_STEPS = 100` first for a timing run.")
        outputs = "the adapter in `checkpoints/phase6/best/`, `eval_report.txt`, `test_predictions.jsonl`, `test_synth_predictions.jsonl`, `val_predictions.jsonl` and `train_info.json`"
    return f"""# {title}

{intro}

**Run it on Kaggle** (unattended, "Save & Run All (Commit)"; the exact steps are in `docs/kaggle_runbook.md`):
1. Accelerator **GPU T4**, **Internet on** (the base model is downloaded from Hugging Face), and add the private dataset **`toolcall-verifier-data-v3b`** (`final_train.jsonl`, `final_val.jsonl`, `final_test.jsonl`, `final_test_synth.jsonl`: 7,154 / 749 / 363 / 374 rows) as the only input. The data folder is found by searching `/kaggle/input/` for the file names.
2. Everything is written to `/kaggle/working/`: {outputs}.
3. **Continue a stopped run:** add the earlier version's output as an input of the new run. Its Trainer checkpoints are copied in and training resumes; a finished model there makes the notebook skip straight to evaluation.
4. **Evaluate only:** set `EVAL_ONLY = True`; with `EVAL_CHECKPOINT = None` the finished `best/` of an attached notebook output is found automatically, or give the folder in `EVAL_CHECKPOINT`.
5. Training stops, saves and evaluates when `TIME_BUDGET_HOURS` (11 h) of the 12-hour session have passed.
6. `DATASET = "final"` reads the four files, takes the split from each row, and ends with the final report (AUROC with a 95% interval, recall at 5% and 10% false alarms, per class by origin, per domain, the synthetic test set, length and source baselines). `"legacy"` is the first-round 7,580-row setup with `test_real.jsonl`.

The code is generated from the legacy Colab notebook by `notebooks/make_kaggle_notebooks.py` and its patch modules; the model, input format and metrics are the Colab ones.

---
"""


KAGGLE_FIXES = {
    "p5": [
        ("## Cell 1: mount Drive, load the six training files", "## Cell 1: load the data (legacy: the six training files; final: `final_*.jsonl`, see the configuration cell)"),
        ("## Cell 3: train/val split\n", "## Cell 3: train/val split (legacy mode)\n"),
        ("## Cell 5: evaluate on `test_real.jsonl`", "## Cell 5: evaluate (legacy: `test_real.jsonl`; final: the 363-row test set, the 374-row synthetic test set and the validation set)"),
        ("Remember (D-010): the test set has 25 incorrect rows,", "Legacy test set (D-010): 25 incorrect rows,"),
    ],
    "p6": [
        ("## Cell 2: mount Drive and load the data", "## Cell 2: load the data (legacy: the six training files; final: `final_*.jsonl`)"),
        ("The six training files are combined exactly as in Phase 5", "In legacy mode the six training files are combined exactly as in Phase 5"),
        ("## Cell 3: `format_input` and the train/val split", "## Cell 3: `format_input` and the train/val split (legacy mode; final mode uses Cell 3b and the split stored in the rows)"),
        ("lr 2e-4, 3 epochs,", "lr 2e-4, `QWEN_EPOCHS` epochs (default 2, as in the final run),"),
        ("## Cell 5: evaluate on `test_real.jsonl`", "## Cell 5: evaluate (legacy: `test_real.jsonl`; final: the 363-row test set, the 374-row synthetic test set and the validation set)"),
        ("Remember (D-010): 25 incorrect rows,", "Legacy test set (D-010): 25 incorrect rows,"),
    ],
}


def apply_kaggle(nb, which):
    nb["cells"][0]["source"] = kaggle_header(which).splitlines(keepends=True)
    for old, new in KAGGLE_FIXES[which]:
        hit = 0
        for c in nb["cells"][1:]:
            if c["cell_type"] != "markdown":
                continue
            t = "".join(c["source"])
            if old in t:
                c["source"] = t.replace(old, new).splitlines(keepends=True)
                hit += 1
        assert hit, f"markdown text not found in the {which} notebook: {old[:60]!r}"
