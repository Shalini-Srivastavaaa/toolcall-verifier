"""Diagnosis of the first final ModernBERT run (D-025): training composition, validation by source, the score distribution on test, old v2 versus the final model.
No API call, no model: it reads the dataset files and the saved predictions.

    .venv/bin/python3 eval/diagnose_final_v1.py [--pred results/final_v1/test_predictions.jsonl] [--v2 checkpoints/phase5_v2/test_real_predictions.jsonl]

Output is printed (and saved by the caller to results/final_v1/diagnosis.txt). The model's VALIDATION predictions were not saved by the run, so validation
AUROC per source cannot be computed from the files we have; the script reports the counts and the length-only baseline per source instead.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
FINAL = ROOT / "data" / "derived" / "final"
NAMES = {1: "wrong arg", 2: "stale state", 3: "wrong tool", 4: "repeated call", 5: "failure to propagate", 6: "ignored error", 7: "policy violation"}


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def group(r):
    """The source groups of the diagnosis."""
    s, d = str(r.get("source", "")), r["domain"]
    if s == "critictool":
        return "CriticTool"
    if d == "bfcl":
        return "BFCL"
    if s == "synthetic_stateflip":
        return "state flips (synthetic)"
    if s.startswith("induced"):
        return "induced (real agent, perturbed run)"
    if s in ("tau2_real_failed_run", "tau2_real_correct_call", "queue2_found_error", "human_labeled_phase4"):
        return "natural (real agent, unperturbed run)"
    if s == "tau2_gold":
        return "tau2 gold (correct calls)"
    return "tau2 synthetic"          # programmatic and LLM corruptions and the paired controls


def boot_auc(y, p, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            v.append(roc_auc_score(y[i], p[i]))
    return np.percentile(v, [2.5, 97.5])


def auc(y, p):
    return roc_auc_score(y, p) if len(set(y)) == 2 else float("nan")


def length(r):
    return float(np.log1p(len(json.dumps(r["turns"])) + len(json.dumps(r["candidate_tool_call"]))))


def composition(train, val):
    print("=" * 100 + "\n1. TRAINING COMPOSITION (errors per class by source group; the paired correct rows below)\n" + "=" * 100)
    for name, rows in (("train", train), ("validation", val)):
        errs = [r for r in rows if r["label"] == "incorrect"]
        cor = [r for r in rows if r["label"] == "correct"]
        groups = sorted({group(r) for r in rows})
        print(f"\n{name}: {len(rows)} rows, {len(errs)} incorrect, {len(cor)} correct")
        print(f"   {'class':26s}" + "".join(f"{g[:22]:>24s}" for g in groups if g != "tau2 gold (correct calls)") + f"{'total':>8s}")
        for k in range(1, 8):
            c = Counter(group(r) for r in errs if r["error_class"] == k)
            print(f"   {k} {NAMES[k]:24s}" + "".join(f"{c[g]:24d}" for g in groups if g != "tau2 gold (correct calls)") + f"{sum(c.values()):8d}")
        c = Counter(group(r) for r in errs)
        print(f"   {'ALL ERRORS':26s}" + "".join(f"{c[g]:24d}" for g in groups if g != "tau2 gold (correct calls)") + f"{sum(c.values()):8d}")
        c = Counter(group(r) for r in cor)
        print(f"   {'correct rows by group':26s}" + "".join(f"{g[:22]}: {c[g]}; " for g in groups))
        real = sum(1 for r in errs if group(r).startswith(("natural", "induced")))
        print(f"   -> real-origin errors (natural + induced): {real} of {len(errs)} ({real / len(errs):.1%}); synthetic or external: {len(errs) - real}")
        syn_pairs = [r for r in cor if str(r.get("source", "")).startswith("synthetic_control")]
        print(f"   -> correct rows that are synthetic controls: {len(syn_pairs)}; real correct (tau2 gold + BFCL gold + CriticTool correct): {len(cor) - len(syn_pairs)}")
    te = [r for r in train if r["label"] == "incorrect" and r["error_class"] == 7]
    print("\nClass 7 rows in train by source:", dict(Counter(group(r) for r in te)), "| by domain:", dict(Counter(r["domain"] for r in te)))
    print("tau2-domain rows (retail, airline, telecom) in train:", sum(r["domain"] in ("retail", "airline", "telecom") for r in train), "of", len(train),
          "| BFCL", sum(r["domain"] == "bfcl" for r in train), "| CriticTool", sum(r["domain"] == "critictool" for r in train))


def by_source(train, val, test):
    print("\n" + "=" * 100 + "\n2. VALIDATION BY SOURCE (counts, and the length-only baseline per source)\n" + "=" * 100)
    print("The model's validation predictions were not saved by the run, so its AUROC per source cannot be computed from the files we have (only the overall 0.895).")
    ytr = np.array([r["label"] == "incorrect" for r in train]).astype(int)
    clf = LogisticRegression(max_iter=1000).fit(np.array([[length(r)] for r in train]), ytr)
    print(f"length-only logistic regression fit on train: coefficient {clf.coef_[0][0]:+.3f} per log-token (positive = longer means more likely incorrect)")
    realdom = ("retail", "airline", "telecom")
    sets = {
        "BFCL": lambda r: r["domain"] == "bfcl",
        "CriticTool": lambda r: r["domain"] == "critictool",
        "tau2 domains, synthetic negatives": lambda r: r["domain"] in realdom and not (r["label"] == "incorrect" and group(r) in ("state flips (synthetic)", "natural (real agent, unperturbed run)", "induced (real agent, perturbed run)")),
        "state flips (negatives) vs tau2 correct": lambda r: r["domain"] in realdom and (r["label"] == "correct" or group(r) == "state flips (synthetic)"),
        "natural real errors vs tau2 correct": lambda r: r["domain"] in realdom and (r["label"] == "correct" or group(r) == "natural (real agent, unperturbed run)"),
        "induced real errors vs tau2 correct": lambda r: r["domain"] in realdom and (r["label"] == "correct" or group(r) == "induced (real agent, perturbed run)"),
    }
    print(f"\n{'validation source (incorrect vs correct rows of the same kind)':66s} {'rows':>5s} {'inc':>4s} {'cor':>4s} {'length AUROC':>13s}   {'test rows analog':>28s}")
    for name, f in sets.items():
        for sp_name, part in (("val", val),):
            rs = [r for r in part if f(r)]
            y = np.array([r["label"] == "incorrect" for r in rs]).astype(int)
            n_inc = int(y.sum())
            if len(rs) == 0 or n_inc == 0 or n_inc == len(rs):
                print(f"{name:66s} {len(rs):5d} {n_inc:4d} {len(rs) - n_inc:4d} {'not defined':>13s}")
                continue
            a = auc(y, clf.predict_proba(np.array([[length(r)] for r in rs]))[:, 1])
            print(f"{name:66s} {len(rs):5d} {n_inc:4d} {len(rs) - n_inc:4d} {a:13.3f}")
    print("\nlength-only AUROC on the other splits (same fit): ", end="")
    for sp_name, part in (("val", val), ("test", test)):
        y = np.array([r["label"] == "incorrect" for r in part]).astype(int)
        print(f"{sp_name} {auc(y, clf.predict_proba(np.array([[length(r)] for r in part]))[:, 1]):.3f}   ", end="")
    print()
    rs = [r for r in test if r["label"] == "correct" or r["origin"] == "natural"]
    for lab, keep in (("test natural errors + natural correct", lambda r: r["origin"] == "natural"), ("test induced errors + induced correct", lambda r: r["origin"] == "induced")):
        rs = [r for r in test if keep(r)]
        y = np.array([r["label"] == "incorrect" for r in rs]).astype(int)
        print(f"   {lab}: {len(rs)} rows, {int(y.sum())} incorrect, length-only AUROC {auc(y, clf.predict_proba(np.array([[length(r)] for r in rs]))[:, 1]):.3f}")
    return clf


def test_scores(test, pred, clf):
    print("\n" + "=" * 100 + "\n3. SCORE DISTRIBUTION ON TEST (model scores joined to the test rows by id)\n" + "=" * 100)
    by_id = {r["trajectory_id"]: r for r in test}
    rows = [by_id[p["trajectory_id"]] for p in pred]
    s = np.array([p["p_incorrect"] for p in pred])
    y = np.array([r["label"] == "incorrect" for r in rows]).astype(int)
    assert all(p["label"] == r["label"] for p, r in zip(pred, rows)), "labels of the predictions differ from the test file"
    print(f"{len(rows)} rows, overall AUROC {auc(y, s):.3f}; score quantiles all rows: " + ", ".join(f"p{q} {np.percentile(s, q):.2f}" for q in (5, 25, 50, 75, 95)) + f"; rows >= 0.7: {int((s >= 0.7).sum())}")
    cor = y == 0
    org = np.array([r["origin"] for r in rows])
    print(f"\ncorrect rows: n {int(cor.sum())}, median {np.median(s[cor]):.3f}, mean {s[cor].mean():.3f}, IQR {np.percentile(s[cor], 25):.2f}-{np.percentile(s[cor], 75):.2f}")
    for o in ("natural", "induced"):
        m = cor & (org == o)
        print(f"   correct, {o}: n {int(m.sum()):3d}, median {np.median(s[m]):.3f}, IQR {np.percentile(s[m], 25):.2f}-{np.percentile(s[m], 75):.2f}")
    print(f"\nincorrect rows by class and origin (AUROC = that group against ALL correct rows, 95% bootstrap interval; chance = 0.5):")
    print(f"   {'group':34s} {'n':>4s} {'median':>7s} {'IQR':>12s} {'recall@0.5':>10s} {'AUROC':>7s}  95% CI")
    cls = np.array([r.get("error_class") or 0 for r in rows])
    allm = [("ALL incorrect", y == 1)]
    for k in range(1, 8):
        allm.append((f"class {k} {NAMES[k]}", (y == 1) & (cls == k)))
    for o in ("natural", "induced"):
        allm.append((f"all incorrect, {o}", (y == 1) & (org == o)))
        for k in range(1, 8):
            m = (y == 1) & (cls == k) & (org == o)
            if m.sum():
                allm.append((f"   class {k}, {o}", m))
    for name, m in allm:
        if m.sum() == 0:
            continue
        sel = m | cor
        a = auc(m[sel].astype(int), s[sel])
        ci = boot_auc(m[sel].astype(int), s[sel]) if m.sum() >= 10 else (float("nan"), float("nan"))
        ciT = f"{ci[0]:.3f}-{ci[1]:.3f}" if m.sum() >= 10 else "n < 10"
        print(f"   {name:34s} {int(m.sum()):4d} {np.median(s[m]):7.3f} {np.percentile(s[m], 25):5.2f}-{np.percentile(s[m], 75):4.2f} {(s[m] >= 0.5).mean():10.2f} {a:7.3f}  {ciT}")
    print("\nper domain, AUROC with 95% interval (the model) and the length-only baseline on the same rows:")
    for d in ("airline", "retail", "telecom"):
        m = np.array([r["domain"] == d for r in rows])
        a = auc(y[m], s[m])
        lo, hi = boot_auc(y[m], s[m])
        lens = np.array([clf.predict_proba([[length(r)]])[0, 1] for r, k in zip(rows, m) if k])
        print(f"   {d:8s} n {int(m.sum()):4d}, {int(y[m].sum()):3d} incorrect: AUROC {a:.3f} ({lo:.3f}-{hi:.3f}); length-only {auc(y[m], lens):.3f}")
    ln = np.array([length(r) for r in rows])
    from scipy.stats import spearmanr  # noqa: PLC0415
    rho = spearmanr(ln, s)[0]
    print(f"\nSpearman correlation of the model score with the input length (log characters): {rho:+.3f} (all rows); within correct rows {spearmanr(ln[cor], s[cor])[0]:+.3f}; within incorrect {spearmanr(ln[~cor], s[~cor])[0]:+.3f}")
    # is length a property of the test construction? (correct examples come from before the first flagged call, errors from the flagged call itself)
    nt = np.array([len(r["turns"]) for r in rows])
    print("\nhistory length (number of turns before the call) in the test set, a property of how the rows were chosen:")
    for o in ("natural", "induced"):
        for lab, m in (("correct", cor & (org == o)), ("incorrect", ~cor & (org == o))):
            if m.sum():
                print(f"   {o:8s} {lab:9s} n {int(m.sum()):3d}: median {np.median(nt[m]):.0f} turns, IQR {np.percentile(nt[m], 25):.0f}-{np.percentile(nt[m], 75):.0f}")
    print(f"   AUROC of the raw number of turns (more turns = incorrect) on all test rows: {auc(y, nt):.3f}; natural {auc(y[org == 'natural'], nt[org == 'natural']):.3f}; induced {auc(y[org == 'induced'], nt[org == 'induced']):.3f}")
    # the same, at the cut the diagnosis cares about
    print("recall at the 5% and 10% false-alarm cut (thresholds 0.550 and 0.514 in the run's report):")
    for f in (0.05, 0.10):
        thr = np.quantile(s[cor], 1 - f)
        print(f"   false-alarm {f:.0%}: threshold {thr:.3f}, recall {(s[y == 1] > thr).mean():.3f}; spread of ALL scores p5-p95 = {np.percentile(s, 5):.2f}-{np.percentile(s, 95):.2f}")


def old_vs_new(test, pred, v2path):
    print("\n" + "=" * 100 + "\n5. OLD v2 VERSUS THE FINAL MODEL ON THE SAME test_real ROWS\n" + "=" * 100)
    v2 = {r["trajectory_id"]: r for r in load(v2path)}
    new = {p["trajectory_id"]: p["p_incorrect"] for p in pred}
    by_id = {r["trajectory_id"]: r for r in test}
    ids = [i for i in v2 if i in new and i in by_id and by_id[i].get("found_in") == "test_real"]
    print(f"{len(v2)} rows in the v2 file, {len(ids)} of them are test_real rows of the new test set")
    # labels of the new test file (a few test_real rows were re-judged); also the labels the v2 run used
    y = np.array([by_id[i]["label"] == "incorrect" for i in ids]).astype(int)
    y_old = np.array([v2[i]["label"] == "incorrect" for i in ids]).astype(int)
    print(f"incorrect: {int(y.sum())} with the new labels, {int(y_old.sum())} with the labels stored in the v2 file; rows where they differ: {int((y != y_old).sum())}")
    a = np.array([v2[i]["p_incorrect"] for i in ids])
    b = np.array([new[i] for i in ids])
    rng = np.random.default_rng(0)
    da, db, dd = [], [], []
    for _ in range(2000):
        k = rng.integers(0, len(ids), len(ids))
        if y[k].min() == y[k].max():
            continue
        x1, x2 = roc_auc_score(y[k], a[k]), roc_auc_score(y[k], b[k])
        da.append(x1)
        db.append(x2)
        dd.append(x2 - x1)
    for name, x, d in (("old v2 (trained on the old 7,580 rows, no policy)", a, da), ("final model (new data, policy on)", b, db)):
        print(f"   {name:52s} AUROC {roc_auc_score(y, x):.3f}  95% CI {np.percentile(d, 2.5):.3f}-{np.percentile(d, 97.5):.3f}")
    print(f"   difference final - v2: {roc_auc_score(y, b) - roc_auc_score(y, a):+.3f}  paired-bootstrap 95% CI {np.percentile(dd, 2.5):+.3f} to {np.percentile(dd, 97.5):+.3f} (2,000 resamples of the same rows)")
    for k in (1, 3, 4, 5, 7):
        m = np.array([by_id[i].get("error_class") == k for i in ids])
        if m.sum():
            print(f"   class {k} (n={int(m.sum())}): median score v2 {np.median(a[m]):.2f}, final {np.median(b[m]):.2f}; recall at 0.5 v2 {(a[m] >= 0.5).mean():.2f}, final {(b[m] >= 0.5).mean():.2f}")
    cor = y == 0
    print(f"   correct rows (n={int(cor.sum())}): median score v2 {np.median(a[cor]):.2f}, final {np.median(b[cor]):.2f}; incorrect rows (n={int((~cor).sum())}): v2 {np.median(a[~cor]):.2f}, final {np.median(b[~cor]):.2f}")
    print(f"   Spearman correlation of the two models' scores on these rows: {__import__('scipy.stats', fromlist=['spearmanr']).spearmanr(a, b)[0]:+.3f}")
    return ids


def real_cv_baselines(test, folds=5, seed=42):
    """The shortcut baselines under the folds REAL_CV uses (same algorithm as notebooks/real_cv_code.py): the bar a fine-tuned model has to beat on those folds."""
    import random
    print("\n" + "=" * 100 + "\n6. SHORTCUT BASELINES UNDER THE REAL_CV FOLDS (5 folds of whole task groups over the 363 test rows)\n" + "=" * 100)
    y = np.array([r["label"] == "incorrect" for r in test]).astype(int)
    by = defaultdict(list)
    for i, r in enumerate(test):
        by[(r["domain"], str(r["task_id"]))].append(i)
    keys = sorted(by)
    random.Random(seed).shuffle(keys)
    keys.sort(key=lambda k: -sum(y[i] for i in by[k]))
    fold = np.zeros(len(test), int)
    load_ = [[0, 0] for _ in range(folds)]
    for k in keys:
        f = min(range(folds), key=lambda j: (load_[j][0], load_[j][1], j))
        for i in by[k]:
            fold[i] = f
        load_[f][0] += int(sum(y[i] for i in by[k]))
        load_[f][1] += len(by[k])
    print(f"{len(test)} rows in {len(keys)} task groups; (incorrect, rows) per fold: {[tuple(x) for x in load_]}")
    L = np.array([[length(r)] for r in test])
    D = np.array([[float(r["domain"] == d) for d in ("airline", "retail", "telecom")] for r in test])
    for name, X in (("length only", L), ("length + domain", np.hstack([L, D]))):
        sc = np.zeros(len(test))
        for f in range(folds):
            tr, te = fold != f, fold == f
            sc[te] = LogisticRegression(max_iter=1000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        lo, hi = boot_auc(y, sc)
        print(f"   {name:16s} out-of-fold AUROC {roc_auc_score(y, sc):.3f}  95% CI {lo:.3f}-{hi:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(ROOT / "results" / "final_v1" / "test_predictions.jsonl"))
    ap.add_argument("--v2", default=str(ROOT / "checkpoints" / "phase5_v2" / "test_real_predictions.jsonl"))
    a = ap.parse_args()
    train, val, test = (load(FINAL / f"{s}.jsonl") for s in ("train", "val", "test"))
    pred = load(a.pred)
    composition(train, val)
    clf = by_source(train, val, test)
    test_scores(test, pred, clf)
    old_vs_new(test, pred, a.v2)
    real_cv_baselines(test)


if __name__ == "__main__":
    main()
