"""REAL_WEIGHT (D-026): a notebook flag, default 1 (off), that samples the real-origin training rows more often, plus the validation report on real-origin rows.

Real-origin rows are the rows the builder marks `real_origin: true`: errors from real agent runs (natural failed runs and induced perturbed runs) and their paired correct
rows. With REAL_WEIGHT = k (an integer >= 1) each of them appears k times per epoch in the training set (the other rows once), which is the same as sampling them k times
more often; the validation set is not touched. The evaluation cell also reports the AUROC of the saved model on the real-origin validation rows separately, per source
group, and saves val_predictions.jsonl with source, origin and real_origin.

Applied to the generated Phase 5 Kaggle notebook by make_kaggle_notebooks.py (after real_cv_code).
"""

CONFIG_LINES = '''REAL_WEIGHT = 1            # integer >= 1. k: the real-origin training rows (real errors and their paired correct rows, field real_origin) appear k times per epoch, the rest once (D-026). 1 = off. The planned next run uses 3
'''

TRAIN_OLD = "    train_ds, val_ds = VerifierDataset(train_rows), VerifierDataset(val_rows)\n"
TRAIN_NEW = '''    assert int(REAL_WEIGHT) == REAL_WEIGHT and REAL_WEIGHT >= 1, "REAL_WEIGHT must be an integer >= 1"
    n_real_train = sum(bool(r.get("real_origin")) for r in train_rows)
    train_rows_w = train_rows + [r for r in train_rows if r.get("real_origin")] * (int(REAL_WEIGHT) - 1)
    print(f"REAL_WEIGHT = {REAL_WEIGHT}: {n_real_train} real-origin training rows of {len(train_rows)}; each appears {int(REAL_WEIGHT)} time(s) per epoch -> {len(train_rows_w)} rows per epoch"
          + (" (no change)" if REAL_WEIGHT == 1 else ""))
    train_ds, val_ds = VerifierDataset(train_rows_w), VerifierDataset(val_rows)
'''

INFO_OLD = '"n_train": len(train_ds), "n_val": len(val_ds),'
INFO_NEW = '"n_train": len(train_ds), "n_train_unique": len(train_rows), "real_weight": REAL_WEIGHT, "n_real_origin_train": n_real_train, "n_val": len(val_ds),'

SMOKE_OLD = '    train_rows, val_rows = train_rows[:192], val_rows[:64]\n'
SMOKE_NEW = '''    # keep real-origin rows in the smoke subset so that REAL_WEIGHT and the real-origin report are exercised
    train_rows = [r for r in train_rows if r.get("real_origin")][:48] + [r for r in train_rows if not r.get("real_origin")][:144]
    val_rows = [r for r in val_rows if r.get("real_origin")][:16] + [r for r in val_rows if not r.get("real_origin")][:48]
'''

VAL_OLD_START = "# validation predictions of this model, so that validation AUROC per source can be computed afterwards (D-025)\n"
VAL_NEW = r'''# validation predictions of this model (D-025), with source, origin and real_origin, and the validation AUROC overall and per source group (D-026): the real-origin
# rows (real errors and their paired correct rows) are reported separately from the synthetic and external ones
if DATASET == "final":
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
            lo_v, hi_v = bootstrap_auroc(yv[ix], val_probs[ix])
            print(f"   {gname:44s} rows {len(ix):4d} (incorrect {n_inc:3d}, correct {len(ix) - n_inc:3d})  AUROC {roc_auc_score(yv[ix], val_probs[ix]):.3f}  95% CI {lo_v:.3f}-{hi_v:.3f}")
        else:
            print(f"   {gname:44s} rows {len(ix):4d} (incorrect {n_inc:3d}): AUROC not defined")
    ri = np.array([i for i, r in enumerate(val_eval_rows) if r.get("real_origin")])
    if len(ri) and 0 < yv[ri].sum() < len(ri):
        lo_r, hi_r = bootstrap_auroc(yv[ri], val_probs[ri])
        print(f"REAL-ORIGIN VALIDATION AUROC: {roc_auc_score(yv[ri], val_probs[ri]):.3f}  95% CI {lo_r:.3f}-{hi_r:.3f}  ({len(ri)} rows, {int(yv[ri].sum())} errors; few rows, so the interval is wide)")
    else:
        print("REAL-ORIGIN VALIDATION AUROC: not defined (the validation split has no real-origin errors or no real-origin correct rows)")

'''


def apply(nb):
    def src(c):
        return "".join(c["source"])

    cells = nb["cells"]
    cfg = next(c for c in cells if c["cell_type"] == "code" and src(c).startswith("# ================= run configuration"))
    s = src(cfg)
    anchor = "REAL_CV = False"
    assert s.count(anchor) == 1
    cfg["source"] = s.replace(anchor, CONFIG_LINES + anchor, 1).splitlines(keepends=True)
    tr = next(c for c in cells if c["cell_type"] == "code" and TRAIN_OLD in src(c))
    t = src(tr)
    assert t.count(TRAIN_OLD) == 1 and t.count(INFO_OLD) == 1
    t = t.replace(TRAIN_OLD, TRAIN_NEW).replace(INFO_OLD, INFO_NEW)
    tr["source"] = t.splitlines(keepends=True)
    sm = next(c for c in cells if c["cell_type"] == "code" and SMOKE_OLD in src(c))
    m = src(sm)
    assert m.count(SMOKE_OLD) == 1
    sm["source"] = m.replace(SMOKE_OLD, SMOKE_NEW).splitlines(keepends=True)
    ev = next(c for c in cells if c["cell_type"] == "code" and VAL_OLD_START in src(c))
    e = src(ev)
    i = e.index(VAL_OLD_START)
    j = e.index("# save per-example probabilities for Phase 7")
    ev["source"] = (e[:i] + VAL_NEW + e[j:]).splitlines(keepends=True)
    return nb
