"""Add the `DATASET = "legacy" | "final"` switch and the final-mode evaluation report to the Phase 5 and Phase 6 Colab notebooks (D-023).

    python notebooks/freeze_split.py && python notebooks/add_policy_flag.py     # (templates; idempotent)
    python notebooks/add_final_mode.py                                          # patches the two Colab notebooks once
    python notebooks/make_kaggle_notebooks.py                                   # then regenerate the Kaggle notebooks

"legacy" keeps today's behavior exactly: the six training files plus test_real, 7,580 rows asserted, the frozen split.
"final" reads the builder's output (data/derived/final/{train,val,test,test_synth}.jsonl, D-022) and takes the split from each row:
no row-count assert; it prints counts by split, class, source, origin and domain, and fails only if a split is empty or a test row
is synthetic. INCLUDE_POLICY is on in final mode (Phase 5). The evaluation cell then adds the final report (AUROC with a bootstrap
interval, recall at 5% and 10% false alarms, per class by origin, per domain, test_synth, length and source baselines, predictions
with row ids). Each patch is an exact replacement that must match once; running the script twice does nothing the second time.
"""

import json
import re
from pathlib import Path

NB = Path(__file__).resolve().parent
MARK = "# --- final mode (D-023) ---"


def sub(text, old, new):
    assert text.count(old) == 1, f"{text.count(old)} matches for {old[:80]!r}"
    return text.replace(old, new)


FIRST_CELL_PREFIX = '''DATASET = "legacy"  # "legacy": the 7,580-row dataset of D-012 with the frozen split. "final": the builder's output (D-022): split taken from each row, INCLUDE_POLICY on
INCLUDE_POLICY_OVERRIDE = None  # None: the POLICY block follows DATASET ("final": on). True / False forces it, e.g. False to score a model that was trained without the block (D-023)
'''

FINAL_LOADER = '''else:
    assert DATASET == "final", f'DATASET must be "legacy" or "final", not {DATASET!r}'

    def final_path(name):  # data/derived/final/NAME.jsonl, or final_NAME.jsonl next to the other data files (the Kaggle upload); TCV_FINAL_DIR overrides
        for d in (os.environ.get("TCV_FINAL_DIR"), DATA / "final", DATA):
            for fn in (f"{name}.jsonl", f"final_{name}.jsonl"):
                if d and (Path(d) / fn).exists():
                    return Path(d) / fn
        raise FileNotFoundError(f"final dataset file {name}.jsonl not found in {DATA}: run data/package_for_kaggle.py and upload it, or set DATASET = \\"legacy\\"")

    FINAL = {s: load_jsonl(final_path(s)) for s in ("train", "val", "test", "test_synth")}
    for part in FINAL.values():
        for r in part:
            r["_file"] = "train_pool.jsonl" if r.get("source") == "tau2_gold" else str(r.get("source"))
    rows = FINAL["train"] + FINAL["val"]
    test_rows = FINAL["test"]
    # the only failures: an empty split, or a synthetic row in the test split
    assert all(FINAL.values()), f"empty split(s): {[s for s, p in FINAL.items() if not p]}"
    bad = [r["trajectory_id"] for r in FINAL["test"] if r.get("origin") == "synthetic" or str(r.get("source", "")).startswith("synthetic")]
    assert not bad, f"synthetic rows in the test split: {bad[:5]}"
    ids = [r["trajectory_id"] for p in FINAL.values() for r in p]
    off = [s for s, p in FINAL.items() for r in p if r.get("split") not in (s, None)]
    print(f"final dataset from {final_path('train').parent}: {len(ids)} rows" + ("" if len(ids) == len(set(ids)) else f" (WARNING: {len(ids) - len(set(ids))} duplicate ids)")
          + ("" if not off else f" (WARNING: {len(off)} rows whose split field differs from their file)"))

    def final_counts(part_by_split):
        def cls(r):
            return f"class {r['error_class']}" if r["label"] == "incorrect" else "correct"
        for title, key in (("class", cls), ("source", lambda r: r.get("source")), ("origin", lambda r: r.get("origin")), ("domain", lambda r: r["domain"])):
            cnt = Counter((str(key(r)), s) for s, p in part_by_split.items() for r in p)
            names = sorted({n for n, _ in cnt})
            print(f"\\nrows per {title} and split:")
            print(f"   {'':30s}" + "".join(f"{s:>11s}" for s in part_by_split) + f"{'total':>9s}")
            for n in names:
                vals = [cnt[(n, s)] for s in part_by_split]
                print(f"   {n:30s}" + "".join(f"{v:>11d}" for v in vals) + f"{sum(vals):>9d}")

    final_counts(FINAL)
'''

FINAL_EVAL = '''if DATASET == "final":
    # ================= final-mode report (D-023): real test set, per class by origin, per domain, synthetic test, baselines =================
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_curve

    def _boot(y, p, n_boot=1000, seed=0):  # 95% percentile interval over row resamples (same method as before)
        rng = np.random.default_rng(seed)
        vals = []
        for _ in range(n_boot):
            i = rng.integers(0, len(y), len(y))
            if y[i].min() != y[i].max():
                vals.append(roc_auc_score(y[i], p[i]))
        return np.percentile(vals, [2.5, 97.5])

    def _recall_at_fpr(y, p, fpr_max):
        fpr, tpr, thr = roc_curve(y, p)
        k = np.flatnonzero(fpr <= fpr_max)[-1]
        return float(tpr[k]), float(thr[k])

    def _cell(flagged, n):  # fewer than 10 examples: do not show a rate
        return f"too few to read ({flagged}/{n})" if n < 10 else f"recall {flagged / n:.2f} ({flagged}/{n})"

    def _length(r):
        return float(np.log1p(len(json.dumps(r["turns"])) + len(json.dumps(r["candidate_tool_call"]))))

    def _src(r):
        s = str(r.get("source", ""))
        return "induced" if s.startswith("induced") else "critictool" if s == "critictool" else "bfcl" if r["domain"] == "bfcl" else "tau2"

    SRC_NAMES = ["tau2", "bfcl", "critictool", "induced"]
    base_tr = np.array([r["label"] == "incorrect" for r in FINAL["train"]]).astype(int)

    def _baselines(part, yy):
        out = {}
        for name, feat in (("length only", lambda r: [_length(r)]), ("source only", lambda r: [float(_src(r) == s) for s in SRC_NAMES])):
            clf = LogisticRegression(max_iter=1000).fit(np.array([feat(r) for r in FINAL["train"]]), base_tr)
            sc = clf.predict_proba(np.array([feat(r) for r in part]))[:, 1]
            out[name] = roc_auc_score(yy, sc) if yy.min() != yy.max() else float("nan")
        return out

    def _overall(title, part, yy, pr):
        print(f"\\n=== {title}: {len(yy)} rows, {int(yy.sum())} incorrect ===")
        lo, hi = _boot(yy, pr)
        pred = pr >= THRESH
        tp, fp = int((pred & (yy == 1)).sum()), int((pred & (yy == 0)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(int(yy.sum()), 1)
        print(f"AUROC {roc_auc_score(yy, pr):.3f}  95% CI {lo:.3f}-{hi:.3f}  (1,000 bootstrap resamples, seed 0)")
        print(f"at threshold {THRESH}: precision {prec:.3f}  recall {rec:.3f}  F1 {2 * prec * rec / max(prec + rec, 1e-9):.3f}  false-flag rate {fp / max(int((yy == 0).sum()), 1):.3f}")
        for f in (0.05, 0.10):
            r_, t_ = _recall_at_fpr(yy, pr, f)
            print(f"recall at {f:.0%} false-alarm rate: {r_:.3f} (threshold {t_:.3f})")
        b = _baselines(part, yy)
        print(f"baselines on the same rows (trained on the training split): length only AUROC {b['length only']:.3f}, source only AUROC {b['source only']:.3f}")

    _overall("REAL TEST SET (human-reviewed)", test_rows, y_true, probs)
    origins = sorted({r["origin"] for r in test_rows})
    print(f"\\nrecall per error class by origin (threshold {THRESH}; a cell with fewer than 10 examples is not shown as a rate)")
    for c in range(1, 8):
        cells = []
        for o in origins:
            m = np.array([r["label"] == "incorrect" and r["error_class"] == c and r["origin"] == o for r in test_rows])
            cells.append(f"{o}: " + _cell(int((probs[m] >= THRESH).sum()), int(m.sum())))
        print(f"  class {c}: " + " | ".join(cells))
    # Class 7 = policy violation (docs/labeling_guide.md). MAIN NUMBER: overall Class 7 recall. The sub-reason split below is approximate: since 2026-11-15 the
    # reasons come from a heuristic pre-fill applied by rule (D-022), not from a human, so it is labeled that way.
    m7 = np.array([r["label"] == "incorrect" and r["error_class"] == 7 for r in test_rows])
    print(f"\\nCLASS 7 (policy violation) RECALL, threshold {THRESH}: " + _cell(int((probs[m7] >= THRESH).sum()), int(m7.sum())) + "   <- the number to quote")
    print("  Class 7 recall by sub-reason (heuristic tags, approximate; a cell with fewer than 10 examples is not shown as a rate):")
    for src_name, src_title in (("human", "reason confirmed by a human"), ("heuristic", "heuristic tags, approximate"), ("generator", "reason set by the generator")):
        shown = False
        for reason in ("no_confirmation", "ineligible_state", "missing_step", "precondition_error", "parallel_write", "other", "unlabeled"):
            m = np.array([r["label"] == "incorrect" and r["error_class"] == 7 and r.get("class7_reason") == reason and r.get("class7_reason_source") == src_name for r in test_rows])
            if m.any():
                if not shown:
                    print(f"    {src_title}:")
                    shown = True
                print(f"      {reason:20s} " + _cell(int((probs[m] >= THRESH).sum()), int(m.sum())))
    print("\\nper domain")
    for d in sorted({r["domain"] for r in test_rows}):
        m = np.array([r["domain"] == d for r in test_rows])
        ni, nc = int(y_true[m].sum()), int((1 - y_true[m]).sum())
        auc = f"AUROC {roc_auc_score(y_true[m], probs[m]):.3f}" if ni >= 10 and nc >= 10 else "AUROC too few to read"
        print(f"  {d:8s} n {int(m.sum()):4d} (incorrect {ni:3d}, correct {nc:3d})  {auc}  recall {(probs[m & (y_true == 1)] >= THRESH).mean() if ni else float('nan'):.2f}  false-flag {(probs[m & (y_true == 0)] >= THRESH).mean() if nc else float('nan'):.2f}")

    synth = FINAL["test_synth"]
    probs_s = predict_probs([{FMT}(r) for r in synth])
    ys = np.array([LABEL2ID[r["label"]] for r in synth])
    print("\\n" + "-" * 78 + "\\nSYNTHETIC TEST SET (held-out synthetic Classes 2 and 6 with their paired correct examples; NOT real errors)\\n" + "-" * 78)
    _overall("SYNTHETIC held-out test", synth, ys, probs_s)
    for c in (2, 6):
        m = np.array([r["label"] == "incorrect" and r["error_class"] == c for r in synth])
        print(f"  synthetic class {c}: " + _cell(int((probs_s[m] >= THRESH).sum()), int(m.sum())))

    for name, part, pr in (("test_predictions.jsonl", test_rows, probs), ("test_synth_predictions.jsonl", synth, probs_s)):
        with open(CKPT_DIR / name, "w") as f:
            for r, p in zip(part, pr):
                f.write(json.dumps({"trajectory_id": r["trajectory_id"], "label": r["label"], "error_class": r.get("error_class"), "origin": r.get("origin"),
                                    "source": r.get("source"), "domain": r["domain"], "split": r.get("split"), "p_incorrect": float(p)}) + "\\n")
    print("\\nsaved", CKPT_DIR / "test_predictions.jsonl", "and", CKPT_DIR / "test_synth_predictions.jsonl")

'''


def patch(path: Path, phase: int):
    nb = json.load(open(path))
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    texts = {id(c): "".join(c["source"]) for c in cells}
    if any(MARK in t for t in texts.values()):
        print(f"{path.name}: already patched")
        return
    fmt = "format_input_v2" if phase == 5 else "format_input"

    def cell_with(s):
        hits = [c for c in cells if s in texts[id(c)]]
        assert len(hits) == 1, f"{len(hits)} cells contain {s[:60]!r}"
        return hits[0]

    def put(c, t):
        texts[id(c)] = t
        c["source"] = t.splitlines(keepends=True)

    # 1. the switch, at the top of the first code cell
    first = cells[0]
    assert texts[id(first)].startswith("# COLAB: pip install")
    put(first, f"{MARK}\n" + FIRST_CELL_PREFIX + texts[id(first)])

    # 2. data cell: legacy loading stays, indented under `if DATASET == "legacy":`; final loading is the else branch
    c = cell_with("EXPECTED_ROWS = 7580")
    t = texts[id(c)]
    a = t.index("# file -> the label every row in it must have\n")
    end_line = 'assert len(rows) == EXPECTED_ROWS, f"expected {EXPECTED_ROWS} training rows, found {len(rows)}: copy the current data/derived/ to Drive"\n'
    b = t.index(end_line) + len(end_line)
    tail = 'test_rows = load_jsonl(DATA / "test_real.jsonl")\nassert len(test_rows) == 102, f"expected 102 test_real rows, found {len(test_rows)}"\nprint(f"test_real.jsonl {len(test_rows):5d} rows  (evaluation only)")\n'
    if t[b:].startswith(tail):
        b += len(tail)
    legacy = t[a:b]
    indented = "".join(("    " + ln if ln.strip() else ln) for ln in legacy.splitlines(keepends=True))
    t = t[:a] + 'if DATASET == "legacy":\n' + indented + FINAL_LOADER + t[b:]
    # TRAIN_FILES-free summary lines after the loader still work: rows, LABEL2ID and the prints use `rows`
    put(c, t)

    # 3. split cell: exact-size assert only for legacy (the frozen block itself already switches on DATASET)
    c = cell_with("# --- frozen split (begin) ---")
    t = texts[id(c)]
    t = sub(t, "assert (len(rows), len(train_rows), len(val_rows)) == (7580, 6816, 764), (len(rows), len(train_rows), len(val_rows))  # the frozen split, D-012\n"
               'print(f"split sizes: train {len(train_rows)} / val {len(val_rows)} (documented 6,816 / 764)")\n',
            'if DATASET == "legacy":\n'
            "    assert (len(rows), len(train_rows), len(val_rows)) == (7580, 6816, 764), (len(rows), len(train_rows), len(val_rows))  # the frozen split, D-012\n"
            'print(f"split sizes: train {len(train_rows)} / val {len(val_rows)}" + (" (documented 6,816 / 764)" if DATASET == "legacy" else " (taken from the split field of the rows)"))\n')
    t = sub(t, 'assert set(by_tier) == {"synthetic_class2_control", "synthetic_class4_control", "synthetic_class6_control"}, by_tier\n',
            'if DATASET == "legacy":\n    assert set(by_tier) == {"synthetic_class2_control", "synthetic_class4_control", "synthetic_class6_control"}, by_tier\n')
    put(c, t)

    # 4. evaluation cell(s)
    if phase == 5:
        c8 = cell_with("# --- v1 vs v2: token lengths and how much evidence survives")
        put(c8, sub(texts[id(c8)], 'test_rows = load_jsonl(DATA / "test_real.jsonl")\n',
                    'test_rows = load_jsonl(DATA / "test_real.jsonl") if DATASET == "legacy" else FINAL["test"]\n'))
    ce = cell_with("# save per-example probabilities for Phase 7")
    t = texts[id(ce)]
    t = sub(t, 'test_rows = load_jsonl(DATA / "test_real.jsonl")',
            'TEST_NAME = "test_real" if DATASET == "legacy" else "the final test set"\n'
            'PRED_FILE = "test_real_predictions.jsonl" if DATASET == "legacy" else "test_predictions.jsonl"\n'
            'test_rows = load_jsonl(DATA / "test_real.jsonl") if DATASET == "legacy" else FINAL["test"]')
    t = sub(t, 'print(f"=== overall on test_real (n=', 'print(f"=== overall on {TEST_NAME} (n=')
    t = sub(t, 'with open(CKPT_DIR / "test_real_predictions.jsonl", "w") as f:', 'with open(CKPT_DIR / PRED_FILE, "w") as f:')
    t = sub(t, 'print("\\nsaved", CKPT_DIR / "test_real_predictions.jsonl")', 'print("\\nsaved", CKPT_DIR / PRED_FILE)')
    t = sub(t, "# save per-example probabilities for Phase 7", FINAL_EVAL.replace("{FMT}", fmt) + "# save per-example probabilities for Phase 7")
    put(ce, t)

    json.dump(nb, open(path, "w"), indent=1, ensure_ascii=False)
    open(path, "a").write("\n")
    print(f"{path.name}: patched")


if __name__ == "__main__":
    patch(NB / "phase5_modernbert.ipynb", 5)
    patch(NB / "phase6_qwen.ipynb", 6)
