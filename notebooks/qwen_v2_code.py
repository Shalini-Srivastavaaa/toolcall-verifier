"""Qwen notebook (Phase 6) brought in line with the ModernBERT notebook (D-029): the v2 input with the policy block, DATASET = "final", REAL_WEIGHT, validation predictions.

Applied to the generated kaggle_phase6_qwen.ipynb by make_kaggle_notebooks.py, after kaggle_phase5_modernbert.ipynb is written (the v2 formatting cell is copied from it, so both notebooks
format an input in exactly the same way; only the tokenizer used for the length budget differs: Qwen's). What changes against the notebook of D-020:
  * `format_input_v2` (evidence-aware input, POLICY block on in final mode, NORMALIZE_AGENT_TEXT wrapper) replaces `format_input` for tokenization and evaluation; the input budget is the
    configuration value QWEN_MAX_LEN in Qwen tokens (the notebook's ladder counts with Qwen's tokenizer, `n_tokens`);
  * REAL_WEIGHT: real-origin training rows (field `real_origin`) appear that many times per epoch;
  * the evaluation saves val_predictions.jsonl (source, origin, real_origin) and prints validation AUROC per source group and on real-origin rows, and no longer overwrites
    test_predictions.jsonl with the legacy format in final mode;
  * train_info.json records real_weight; the smoke subset keeps real-origin rows.
Unchanged because they already exist in the notebook: the length-bucketed sampler, one GPU by default, BENCHMARK_STEPS, the micro-batch / checkpointing ladder, the time budget, the final report.
"""

import json
from pathlib import Path

CONFIG_LINES = '''QWEN_MAX_LEN = 4096        # input budget in QWEN tokens for the v2 input (the ladder of format_input_v2 tightens an input until it fits); 2048 is the shorter, faster alternative (D-029)
QWEN_EPOCHS = 2            # training epochs. The FLOP estimate for 4,096 tokens, REAL_WEIGHT = 3 and 3 epochs is 9.5 hours (margin to the 11-hour budget only 1.16x), so the default is 2 epochs (6.3 hours); set 3 only if the 100-step benchmark, run with QWEN_EPOCHS = 2, projects the full run under about 5 hours (3 epochs are then under 7.5 hours; D-029, runbook (j))
REAL_WEIGHT = 1            # integer >= 1. k: the real-origin training rows (real errors and their paired correct rows, field real_origin) appear k times per epoch, the rest once (D-026). 1 = off. The planned run uses 3
NORMALIZE_AGENT_TEXT = False  # True: assistant text turns are unwrapped from JSON and cut to AGENT_TEXT_CHARS characters in every split (D-027). Shelved: leave False
AGENT_TEXT_CHARS = 200        # the cut used when NORMALIZE_AGENT_TEXT is True
'''

EPOCHS_OLD = "LR, EPOCHS, WARMUP_RATIO, EFFECTIVE_BATCH = 2e-4, (1 if SMOKE else 3), 0.1, 16"
EPOCHS_NEW = "LR, EPOCHS, WARMUP_RATIO, EFFECTIVE_BATCH = 2e-4, (1 if SMOKE else QWEN_EPOCHS), 0.1, 16"
ENCODE_OLD = "    texts = [format_input(r) for r in part]\n"
ENCODE_NEW = "    texts = [format_input_v2(r) for r in part]\n"
TRAIN_OLD = "    train_ds, val_ds = encode(train_rows), encode(val_rows)\n"
TRAIN_NEW = '''    assert int(REAL_WEIGHT) == REAL_WEIGHT and REAL_WEIGHT >= 1, "REAL_WEIGHT must be an integer >= 1"
    n_real_train = sum(bool(r.get("real_origin")) for r in train_rows)
    train_rows_w = train_rows + [r for r in train_rows if r.get("real_origin")] * (int(REAL_WEIGHT) - 1)
    print(f"REAL_WEIGHT = {REAL_WEIGHT}: {n_real_train} real-origin training rows of {len(train_rows)}; each appears {int(REAL_WEIGHT)} time(s) per epoch -> {len(train_rows_w)} rows per epoch"
          + (" (no change)" if REAL_WEIGHT == 1 else ""))
    train_ds, val_ds = encode(train_rows_w), encode(val_rows)
'''
INFO_OLD = '"n_train": len(train_ds), "n_val": len(val_ds),'
INFO_NEW = '"n_train": len(train_ds), "n_train_unique": len(train_rows), "real_weight": REAL_WEIGHT, "n_real_origin_train": n_real_train, "qwen_max_len": QWEN_MAX_LEN, "input_format": "v2", "n_val": len(val_ds),'
SMOKE_OLD = '    train_rows, val_rows = train_rows[:192], val_rows[:64]\n'
SMOKE_NEW = '''    # keep real-origin rows in the smoke subset so that REAL_WEIGHT and the real-origin report are exercised
    train_rows = [r for r in train_rows if r.get("real_origin")][:48] + [r for r in train_rows if not r.get("real_origin")][:144]
    val_rows = [r for r in val_rows if r.get("real_origin")][:16] + [r for r in val_rows if not r.get("real_origin")][:48]
'''

VAL_BLOCK = r'''# validation predictions of this model, with source, origin and real_origin, and the validation AUROC overall and per source group (D-029): the real-origin rows (real errors and
# their paired correct rows) are reported separately from the synthetic and external ones
if DATASET == "final":
    def _val_boot(y, p, n_boot=1000, seed=0):
        rng = np.random.default_rng(seed)
        vals = []
        for _ in range(n_boot):
            i = rng.integers(0, len(y), len(y))
            if y[i].min() != y[i].max():
                vals.append(roc_auc_score(y[i], p[i]))
        return np.percentile(vals, [2.5, 97.5])

    val_eval_rows = FINAL["val"]
    val_probs = predict_probs([format_input_v2(r) for r in val_eval_rows])
    with open(CKPT_DIR / "val_predictions.jsonl", "w") as f:
        for r, p in zip(val_eval_rows, val_probs):
            f.write(json.dumps({"trajectory_id": r["trajectory_id"], "label": r["label"], "error_class": r.get("error_class"), "origin": r.get("origin"),
                                "source": r.get("source"), "real_origin": bool(r.get("real_origin")), "domain": r["domain"], "split": r.get("split"), "p_incorrect": float(p)}) + "\n")
    yv = np.array([LABEL2ID[r["label"]] for r in val_eval_rows])
    print("saved", CKPT_DIR / "val_predictions.jsonl", f"(validation AUROC of this model {roc_auc_score(yv, val_probs):.3f})")

    def _val_group(r):
        if r.get("real_origin"):
            return "real-origin: " + str(r.get("origin"))
        if str(r.get("source", "")) == "critictool":
            return "CriticTool"
        if r["domain"] == "bfcl":
            return "BFCL"
        return "tau2 domains: synthetic, state flips, gold"

    print("\nvalidation AUROC by source group (errors and their correct rows of the group; the AUROC of a group is not defined without both labels):")
    groups_v = {}
    for i, r in enumerate(val_eval_rows):
        groups_v.setdefault(_val_group(r), []).append(i)
    for gname in sorted(groups_v):
        ix = np.array(groups_v[gname])
        n_inc = int(yv[ix].sum())
        if 0 < n_inc < len(ix):
            lo_v, hi_v = _val_boot(yv[ix], val_probs[ix])
            print(f"   {gname:44s} rows {len(ix):4d} (incorrect {n_inc:3d}, correct {len(ix) - n_inc:3d})  AUROC {roc_auc_score(yv[ix], val_probs[ix]):.3f}  95% CI {lo_v:.3f}-{hi_v:.3f}")
        else:
            print(f"   {gname:44s} rows {len(ix):4d} (incorrect {n_inc:3d}): AUROC not defined")
    ri = np.array([i for i, r in enumerate(val_eval_rows) if r.get("real_origin")])
    if len(ri) and 0 < yv[ri].sum() < len(ri):
        lo_r, hi_r = _val_boot(yv[ri], val_probs[ri])
        print(f"REAL-ORIGIN VALIDATION AUROC: {roc_auc_score(yv[ri], val_probs[ri]):.3f}  95% CI {lo_r:.3f}-{hi_r:.3f}  ({len(ri)} rows, {int(yv[ri].sum())} errors; few rows, so the interval is wide)")
    else:
        print("REAL-ORIGIN VALIDATION AUROC: not defined (the validation split has no real-origin errors or no real-origin correct rows)")

'''

LEGACY_OLD = '# save per-example probabilities for Phase 7 (calibration / risk-coverage)\nwith open(CKPT_DIR / PRED_FILE, "w") as f:\n    for r, p in zip(test_rows, probs):\n        f.write(json.dumps({"trajectory_id": r["trajectory_id"], "label": r["label"], "error_class": r["error_class"],\n                            "domain": r["domain"], "p_incorrect": float(p)}) + "\\n")\nprint("\\nsaved", CKPT_DIR / PRED_FILE)'
LEGACY_NEW = ('# save per-example probabilities for Phase 7 (calibration / risk-coverage). Legacy mode only: in final mode the richer file (with origin and source) was already written above\n'
              'if DATASET == "legacy":\n'
              '    with open(CKPT_DIR / PRED_FILE, "w") as f:\n'
              '        for r, p in zip(test_rows, probs):\n'
              '            f.write(json.dumps({"trajectory_id": r["trajectory_id"], "label": r["label"], "error_class": r["error_class"],\n'
              '                                "domain": r["domain"], "p_incorrect": float(p)}) + "\\n")\n'
              '    print("\\nsaved", CKPT_DIR / PRED_FILE)')


def apply(nb6, nb5_path):
    def src(c):
        return "".join(c["source"])

    cells = nb6["cells"]
    cfg = next(c for c in cells if c["cell_type"] == "code" and src(c).startswith("# ================= run configuration"))
    s = src(cfg)
    anchor = "MODEL_PATH_OVERRIDE = None"
    assert s.count(anchor) == 1
    cfg["source"] = s.replace(anchor, CONFIG_LINES + anchor, 1).splitlines(keepends=True)

    # the v2 formatting cell, copied from the ModernBERT notebook, with the Qwen budget
    nb5 = json.load(open(nb5_path))
    v2 = next(c for c in nb5["cells"] if c["cell_type"] == "code" and "def format_input_v2(example, tok=True, meta=False):" in src(c) and src(c).startswith("# --- v2: evidence-aware input"))
    v2s = src(v2)
    lines = v2s.split("\n")
    i_max = next(i for i, l in enumerate(lines) if l.startswith("MAX_LEN = 4096"))
    i_bud = next(i for i, l in enumerate(lines) if l.startswith("TOKEN_BUDGET_V2 = 4096"))
    lines[i_max] = "MAX_LEN = QWEN_MAX_LEN   # v2 tokenization / training length in QWEN tokens (the v1 functions above keep using TOKEN_BUDGET = 1500)"
    lines[i_bud] = "TOKEN_BUDGET_V2 = QWEN_MAX_LEN   # the ladder counts with n_tokens, i.e. with Qwen's tokenizer"
    split_cell = next(i for i, c in enumerate(cells) if c["cell_type"] == "code" and src(c).startswith("from transformers import AutoTokenizer"))
    new_md = {"cell_type": "markdown", "metadata": {}, "source": ["## Cell 3b: the v2 input (evidence-aware, with the policy block) for Qwen\n", "\n", "The same formatting code as the ModernBERT notebook (copied from it), counting tokens with Qwen's tokenizer and fitting `QWEN_MAX_LEN` tokens. `format_input` above is the old v1 input and is no longer used for training or evaluation.\n"]}
    new_code = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": "\n".join(lines).splitlines(keepends=True)}
    cells[split_cell + 1:split_cell + 1] = [new_md, new_code]
    # SMOKE subset in the split cell
    sc = cells[split_cell]
    t = src(sc)
    assert t.count(SMOKE_OLD) == 1
    sc["source"] = t.replace(SMOKE_OLD, SMOKE_NEW).splitlines(keepends=True)
    # training cell
    tr = next(c for c in cells if c["cell_type"] == "code" and TRAIN_OLD in src(c))
    t = src(tr)
    assert t.count(ENCODE_OLD) == 1 and t.count(TRAIN_OLD) == 1 and t.count(INFO_OLD) == 1 and t.count(EPOCHS_OLD) == 1
    tr["source"] = t.replace(EPOCHS_OLD, EPOCHS_NEW).replace(ENCODE_OLD, ENCODE_NEW).replace(TRAIN_OLD, TRAIN_NEW).replace(INFO_OLD, INFO_NEW).splitlines(keepends=True)
    # evaluation cell
    ev = next(c for c in cells if c["cell_type"] == "code" and "def load_eval_tokenizer" in src(c))
    e = src(ev)
    assert e.count("test_texts = [format_input(r) for r in test_rows]") == 1 and e.count("probs_s = predict_probs([format_input(r) for r in synth])") == 1 and e.count(LEGACY_OLD) == 1
    e = e.replace("test_texts = [format_input(r) for r in test_rows]", "test_texts = [format_input_v2(r) for r in test_rows]")
    e = e.replace("probs_s = predict_probs([format_input(r) for r in synth])", "probs_s = predict_probs([format_input_v2(r) for r in synth])")
    e = e.replace(LEGACY_OLD, VAL_BLOCK + LEGACY_NEW)
    ev["source"] = e.splitlines(keepends=True)
    return nb6
