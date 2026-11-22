"""The REAL_CV experiment of D-025 (flag in the configuration cell, default off), plus two reporting fixes to the evaluation cell of the Phase 5 Kaggle notebook.

REAL_CV: the first final run trained on 5,569 rows of which 16 are real errors (0.7%); the 363-row test set holds the only sizeable body of human-reviewed real data.
With REAL_CV = True the notebook, after the normal evaluation, fine-tunes the evaluated checkpoint on the real rows with grouped k-fold cross-validation: the
test rows are split into REAL_CV_FOLDS folds by whole (domain, task_id) groups; for each fold the checkpoint is fine-tuned for REAL_CV_EPOCHS epochs on the
rows of the other folds and scores the held-out fold. Every test row is therefore scored by a model that never saw its task group, and the pooled out-of-fold AUROC
is printed next to the plain test AUROC of the same checkpoint. It needs no new training from scratch: with EVAL_ONLY = True and EVAL_CHECKPOINT = the final_v1
best/ folder it takes about 30 minutes on one T4 (5 folds x about 290 rows x 3 epochs, plus scoring).

Reporting fixes (not training settings): (1) in the first final run the last lines of the evaluation cell overwrote the richer test_predictions.jsonl (with origin and
source) with the old legacy format; they now run in legacy mode only; (2) the model's validation predictions are saved to val_predictions.jsonl, so a validation AUROC
per source can be computed afterwards.
"""

CONFIG_LINES = '''REAL_CV = False            # True: after the evaluation, fine-tune the checkpoint on the human-reviewed real rows with grouped cross-validation and report out-of-fold scores (D-025; about 30 minutes)
REAL_CV_FOLDS = 5          # folds of whole (domain, task_id) groups over the 363 test rows
REAL_CV_EPOCHS = 3         # fine-tuning epochs per fold, on the rows of the other folds (about 290 rows)
REAL_CV_LR = 1e-5          # fine-tuning learning rate (the main run used 2e-5)
'''

CELL = r'''# ===== Experiment REAL_CV (D-025): fine-tune the checkpoint on the human-reviewed real rows, grouped cross-validation. Default off. =====
if not REAL_CV:
    print("REAL_CV = False: skipped. Set it to True (with EVAL_ONLY = True and EVAL_CHECKPOINT = the final_v1 best/ folder) to run it.")
else:
    from transformers import Trainer, TrainingArguments
    from collections import defaultdict as _dd
    t_cv = time.time()
    cv_rows = FINAL["test"][:60] if SMOKE else FINAL["test"]
    y_cv = np.array([LABEL2ID[r["label"]] for r in cv_rows])
    assert DATASET == "final", "REAL_CV needs DATASET = 'final' (the human-reviewed test rows)"

    # 1. folds: whole (domain, task_id) groups, seeded, balanced on the number of incorrect rows and rows
    by_group = _dd(list)
    for i, r in enumerate(cv_rows):
        by_group[(r["domain"], str(r["task_id"]))].append(i)
    keys = sorted(by_group)
    random.Random(SEED).shuffle(keys)
    keys.sort(key=lambda k: -sum(y_cv[i] for i in by_group[k]))
    fold_of = np.zeros(len(cv_rows), int)
    load_ = [[0, 0] for _ in range(REAL_CV_FOLDS)]
    for k in keys:
        f = min(range(REAL_CV_FOLDS), key=lambda j: (load_[j][0], load_[j][1], j))
        for i in by_group[k]:
            fold_of[i] = f
        load_[f][0] += int(sum(y_cv[i] for i in by_group[k]))
        load_[f][1] += len(by_group[k])
    print(f"REAL_CV: {len(cv_rows)} rows in {len(keys)} task groups, {REAL_CV_FOLDS} folds (incorrect, rows per fold): {[tuple(x) for x in load_]}")
    assert all(len({fold_of[i] for i in idx}) == 1 for idx in by_group.values()), "a task group straddles two folds"

    def cv_predict(m, texts, batch_size=4):
        m.eval()
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = np.zeros(len(texts))
        with torch.no_grad():
            for s in range(0, len(order), batch_size):
                idx = order[s : s + batch_size]
                enc = eval_tok([texts[i] for i in idx], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(device)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                    logits = m(**enc).logits
                out[idx] = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
        return out

    # 2. one fine-tuning per fold, always starting from the evaluated checkpoint
    oof = np.zeros(len(cv_rows))
    for f in range(REAL_CV_FOLDS):
        tr = [r for r, ff in zip(cv_rows, fold_of) if ff != f]
        te_idx = [i for i in range(len(cv_rows)) if fold_of[i] == f]
        if not te_idx:
            continue
        t_f = time.time()
        ds = VerifierDataset(tr)
        done = False
        for micro, ckpt in ((2, False), (1, True)):
            ft = AutoModelForSequenceClassification.from_pretrained(str(EVAL_DIR))
            steps = math.ceil(len(ds) / EFFECTIVE_BATCH) * REAL_CV_EPOCHS
            targs = TrainingArguments(output_dir=str(CKPT_DIR / "real_cv_tmp"), num_train_epochs=REAL_CV_EPOCHS, learning_rate=REAL_CV_LR, optim="adamw_torch",
                                      lr_scheduler_type="linear", warmup_steps=max(1, int(0.1 * steps)), weight_decay=0.01, per_device_train_batch_size=micro,
                                      gradient_accumulation_steps=max(1, EFFECTIVE_BATCH // micro), eval_strategy="no", save_strategy="no", logging_steps=10 ** 9,
                                      gradient_checkpointing=ckpt, gradient_checkpointing_kwargs={"use_reentrant": False}, fp16=torch.cuda.is_available(),
                                      report_to="none", seed=SEED, disable_tqdm=True)
            tr_ = Trainer(model=ft, args=targs, train_dataset=ds, processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8))
            try:
                tr_.train()
                done = True
                break
            except torch.cuda.OutOfMemoryError:
                print("CUDA out of memory in the fold fine-tuning; retrying with micro-batch 1 and gradient checkpointing")
                del tr_, ft
                gc.collect()
                torch.cuda.empty_cache()
        assert done, "fold fine-tuning failed even at micro-batch 1"
        ft.to(device)
        oof[te_idx] = cv_predict(ft, [format_input_v2(cv_rows[i]) for i in te_idx])
        a = roc_auc_score(y_cv[te_idx], oof[te_idx]) if len(set(y_cv[te_idx])) == 2 else float("nan")
        print(f"   fold {f}: trained on {len(tr)} rows, scored {len(te_idx)} held-out rows, fold AUROC {a:.3f}, {time.time() - t_f:.0f}s")
        del tr_, ft, ds
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 3. the report: pooled out-of-fold scores against the plain scores of the same checkpoint on the same rows
    base = probs[: len(cv_rows)] if SMOKE else probs
    lo_b, hi_b = bootstrap_auroc(y_cv, base)
    lo_o, hi_o = bootstrap_auroc(y_cv, oof)
    print("\n" + "=" * 100)
    print(f"REAL_CV RESULT ({REAL_CV_FOLDS} folds of whole task groups, fine-tuning {REAL_CV_EPOCHS} epochs at lr {REAL_CV_LR}, from {EVAL_DIR})")
    print(f"  plain checkpoint, same {len(cv_rows)} rows      : AUROC {roc_auc_score(y_cv, base):.3f}  95% CI {lo_b:.3f}-{hi_b:.3f}")
    print(f"  fine-tuned on other folds (out-of-fold): AUROC {roc_auc_score(y_cv, oof):.3f}  95% CI {lo_o:.3f}-{hi_o:.3f}")
    rng = np.random.default_rng(0)
    dd = []
    for _ in range(1000):
        i = rng.integers(0, len(y_cv), len(y_cv))
        if y_cv[i].min() != y_cv[i].max():
            dd.append(roc_auc_score(y_cv[i], oof[i]) - roc_auc_score(y_cv[i], base[i]))
    print(f"  difference (out-of-fold - plain): {roc_auc_score(y_cv, oof) - roc_auc_score(y_cv, base):+.3f}  paired bootstrap 95% interval {np.percentile(dd, 2.5):+.3f} to {np.percentile(dd, 97.5):+.3f}")
    for fr in (0.05, 0.10):
        for nm, sc in (("plain", base), ("out-of-fold", oof)):
            thr = np.quantile(sc[y_cv == 0], 1 - fr)
            print(f"  recall at {fr:.0%} false alarms, {nm:11s}: {(sc[y_cv == 1] > thr).mean():.3f} (threshold {thr:.3f})")
    # the test rows carry a length artifact (the correct examples come from before the first error), so the fine-tuned model could win by length alone: score the same
    # folds with shortcut baselines fit on the training folds only (length; length + domain)
    from sklearn.linear_model import LogisticRegression
    Lg = np.array([[math.log1p(len(json.dumps(r["turns"])) + len(json.dumps(r["candidate_tool_call"])))] for r in cv_rows])
    Dm = np.array([[float(r["domain"] == d) for d in ("airline", "retail", "telecom")] for r in cv_rows])
    for nm, X in (("length only", Lg), ("length + domain", np.hstack([Lg, Dm]))):
        sc = np.zeros(len(cv_rows))
        for f in range(REAL_CV_FOLDS):
            te_i, tr_i = fold_of == f, fold_of != f
            if te_i.any() and len(set(y_cv[tr_i])) == 2:
                sc[te_i] = LogisticRegression(max_iter=1000).fit(X[tr_i], y_cv[tr_i]).predict_proba(X[te_i])[:, 1]
        print(f"  shortcut baseline on the same folds, {nm:16s}: out-of-fold AUROC {roc_auc_score(y_cv, sc):.3f}  <- the fine-tuned model has to beat this, not just the plain checkpoint")
    print("  recall at 0.5 per class and origin, plain -> out-of-fold (cells under 10 rows are not shown as rates):")
    for c in range(1, 8):
        cells = []
        for o in sorted({r.get("origin") for r in cv_rows}):
            m = np.array([r["label"] == "incorrect" and r.get("error_class") == c and r.get("origin") == o for r in cv_rows])
            if m.sum():
                cells.append(f"{o}: " + (f"{(base[m] >= 0.5).mean():.2f} -> {(oof[m] >= 0.5).mean():.2f} (n={int(m.sum())})" if m.sum() >= 10 else f"too few to read ({int((base[m] >= 0.5).sum())} -> {int((oof[m] >= 0.5).sum())} of {int(m.sum())})"))
        if cells:
            print(f"    class {c}: " + " | ".join(cells))
    for d in sorted({r["domain"] for r in cv_rows}):
        m = np.array([r["domain"] == d for r in cv_rows])
        if len(set(y_cv[m])) == 2:
            print(f"  {d:8s} n {int(m.sum()):4d}: AUROC plain {roc_auc_score(y_cv[m], base[m]):.3f} -> out-of-fold {roc_auc_score(y_cv[m], oof[m]):.3f}")
    with open(CKPT_DIR / "real_cv_predictions.jsonl", "w") as fh:
        for r, p, b, ff in zip(cv_rows, oof, base, fold_of):
            fh.write(json.dumps({"trajectory_id": r["trajectory_id"], "label": r["label"], "error_class": r.get("error_class"), "origin": r.get("origin"), "domain": r["domain"],
                                 "fold": int(ff), "p_incorrect_oof": float(p), "p_incorrect_plain": float(b)}) + "\n")
    print(f"saved {CKPT_DIR / 'real_cv_predictions.jsonl'}; REAL_CV took {fmt_td(time.time() - t_cv)}")
    shutil.rmtree(CKPT_DIR / "real_cv_tmp", ignore_errors=True)
'''

VAL_PRED_BLOCK = r'''# validation predictions of this model, so that validation AUROC per source can be computed afterwards (D-025)
if DATASET == "final":
    val_eval_rows = FINAL["val"]
    val_probs = predict_probs([format_input_v2(r) for r in val_eval_rows])
    with open(CKPT_DIR / "val_predictions.jsonl", "w") as f:
        for r, p in zip(val_eval_rows, val_probs):
            f.write(json.dumps({"trajectory_id": r["trajectory_id"], "label": r["label"], "error_class": r.get("error_class"), "origin": r.get("origin"),
                                "source": r.get("source"), "domain": r["domain"], "split": r.get("split"), "p_incorrect": float(p)}) + "\n")
    print("saved", CKPT_DIR / "val_predictions.jsonl", f"(validation AUROC of this model {roc_auc_score([LABEL2ID[r['label']] for r in val_eval_rows], val_probs):.3f})")

'''


def apply(nb):
    """Patch the generated Phase 5 Kaggle notebook: flags in the configuration cell, the fixed prediction saving, validation predictions and the REAL_CV cell."""
    import json

    def src(c):
        return "".join(c["source"])

    cells = nb["cells"]
    cfg = next(c for c in cells if c["cell_type"] == "code" and src(c).startswith("# ================= run configuration"))
    s = src(cfg)
    anchor = "MODEL_PATH_OVERRIDE = None"
    assert s.count(anchor) == 1
    cfg["source"] = s.replace(anchor, CONFIG_LINES + anchor, 1).splitlines(keepends=True)
    ev = next(c for c in cells if c["cell_type"] == "code" and "def load_eval_tokenizer" in src(c))
    e = src(ev)
    old = '# save per-example probabilities for Phase 7 (calibration / risk-coverage)\nwith open(CKPT_DIR / PRED_FILE, "w") as f:\n    for r, p in zip(test_rows, probs):\n        f.write(json.dumps({"trajectory_id": r["trajectory_id"], "label": r["label"], "error_class": r["error_class"],\n                            "domain": r["domain"], "p_incorrect": float(p)}) + "\\n")\nprint("\\nsaved", CKPT_DIR / PRED_FILE)'
    assert e.count(old) == 1, "the legacy prediction block was not found exactly once"
    new = ('# save per-example probabilities for Phase 7 (calibration / risk-coverage). Legacy mode only: in final mode the richer file (with origin and source) was\n'
           '# already written above and this block used to overwrite it (found in D-025)\n'
           'if DATASET == "legacy":\n'
           '    with open(CKPT_DIR / PRED_FILE, "w") as f:\n'
           '        for r, p in zip(test_rows, probs):\n'
           '            f.write(json.dumps({"trajectory_id": r["trajectory_id"], "label": r["label"], "error_class": r["error_class"],\n'
           '                                "domain": r["domain"], "p_incorrect": float(p)}) + "\\n")\n'
           '    print("\\nsaved", CKPT_DIR / PRED_FILE)')
    e = e.replace(old, VAL_PRED_BLOCK + new, 1)
    ev["source"] = e.splitlines(keepends=True)
    cells.append({"cell_type": "markdown", "metadata": {}, "source": ["## Experiment REAL_CV (D-025, default off)\n", "\n", "Fine-tunes the evaluated checkpoint on the human-reviewed real rows with grouped cross-validation and prints out-of-fold scores next to the plain ones. Set `REAL_CV = True` in the configuration cell; with `EVAL_ONLY = True` and `EVAL_CHECKPOINT` pointing at an earlier run's `best/` folder it takes about 30 minutes and trains nothing from scratch.\n"]})
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": CELL.splitlines(keepends=True)})
    return nb
