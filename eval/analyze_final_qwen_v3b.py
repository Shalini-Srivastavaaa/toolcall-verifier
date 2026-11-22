"""Analysis of the Qwen2.5-1.5B run against ModernBERT v3b, the judges, the shortcut baselines and an ensemble (D-030). No API, no model, no GPU.

    .venv/bin/python3 eval/analyze_final_qwen_v3b.py paired    > results/final_qwen_v3b/analysis_1_paired.txt     # step 1: paired bootstrap, per class, per domain
    .venv/bin/python3 eval/analyze_final_qwen_v3b.py ensemble  > results/final_qwen_v3b/analysis_2_ensemble.txt   # step 2: EXPLORATORY ensemble (decided after seeing the test results)
    .venv/bin/python3 eval/analyze_final_qwen_v3b.py gap       > results/final_qwen_v3b/analysis_3_gap.txt        # step 3: synthetic / validation / real side by side
    .venv/bin/python3 eval/analyze_final_qwen_v3b.py readme    > results/final_qwen_v3b/analysis_5_readme.txt     # step 5: the numbers of the README table

Inputs: data/derived/final_v3b/{train,test}.jsonl (the training split fits the length and tool-name baselines, as in D-029), results/final_qwen_v3b/*_predictions.jsonl, results/final_v3b/*_predictions.jsonl,
results/judge_new_test/openai-{mini,nano}_test_predictions.jsonl, results/final_qwen_v3b/speed_cost.json. Reuses the helpers of eval/analyze_final_v2.py (2,000 bootstrap resamples, seed 0, the same
resample for both scores of a paired difference; recall at a false-alarm rate from the ROC curve of the rows of the subset).
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_final_v2 import boot_ci, length, load, paired, recall_at_fa, scores  # noqa: E402
from analyze_final_v3b import NAMES, tool_scores  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"


def setup():
    train = load(ROOT / "data" / "derived" / "final_v3b" / "train.jsonl")
    test = load(ROOT / "data" / "derived" / "final_v3b" / "test.jsonl")
    sc = {
        "ModernBERT v3b": scores(R / "final_v3b" / "test_predictions.jsonl"),
        "Qwen2.5-1.5B": scores(R / "final_qwen_v3b" / "test_predictions.jsonl"),
        "GPT-5-mini": scores(R / "judge_new_test" / "openai-mini_test_predictions.jsonl"),
        "GPT-5-nano": scores(R / "judge_new_test" / "openai-nano_test_predictions.jsonl"),
    }
    clf = LogisticRegression(max_iter=1000).fit(np.array([[length(r)] for r in train]), np.array([r["label"] == "incorrect" for r in train]).astype(int))
    sc["length only"] = {r["trajectory_id"]: float(clf.predict_proba([[length(r)]])[0, 1]) for r in test}
    sc["tool name only"] = tool_scores(train, test)
    for k, d in sc.items():
        assert set(d) == {r["trajectory_id"] for r in test}, k
    return train, test, sc


def arr(d, rows):
    return np.array([d[r["trajectory_id"]] for r in rows])


def yy(rows):
    return np.array([r["label"] == "incorrect" for r in rows]).astype(int)


def subsets(test):
    return (("OVERALL", test), ("NATURAL rows", [r for r in test if r["origin"] == "natural"]), ("INDUCED rows", [r for r in test if r["origin"] == "induced"]))


def fmt_pair(label, y, a, b):
    d, lo, hi, share, p = paired(y, a, b)
    return f"   {label:34s}: {d:+.3f}  paired 95% CI {lo:+.3f} to {hi:+.3f}; first better in {share:.1%} of resamples; two-sided p about {p:.3f}" + ("   <- interval excludes 0" if lo > 0 or hi < 0 else "   (interval includes 0)")


def section_paired(train, test, sc):
    q, m = sc["Qwen2.5-1.5B"], sc["ModernBERT v3b"]
    print("=" * 100 + "\n0. THE FIGURES YOU GAVE, FROM THE PREDICTIONS\n" + "=" * 100)
    for name, rows in subsets(test):
        y, s = yy(rows), arr(q, rows)
        lo, hi = boot_ci(y, s)
        note = "; the report's 1,000-resample CI is 0.717-0.814" if name == "OVERALL" else ""
        print(f"{name:13s} n {len(rows):3d} ({int(y.sum()):3d} errors): Qwen AUROC {roc_auc_score(y, s):.3f} (95% CI {lo:.3f}-{hi:.3f}, 2,000 resamples{note}); at 0.5 recall {(s[y == 1] >= 0.5).mean():.3f}, false-flag {(s[y == 0] >= 0.5).mean():.3f}")
    print("\n" + "=" * 100 + "\n1a. PAIRED BOOTSTRAP OF AUROC DIFFERENCES: Qwen minus the other\n" + "=" * 100)
    for name, rows in subsets(test):
        y = yy(rows)
        print(f"\n{name}: {len(rows)} rows, {int(y.sum())} errors, {int((1 - y).sum())} correct")
        for k, d in sc.items():
            lo, hi = boot_ci(y, arr(d, rows))
            print(f"   {k:16s} AUROC {roc_auc_score(y, arr(d, rows)):.3f}  (95% CI {lo:.3f}-{hi:.3f})")
        for k in ("ModernBERT v3b", "GPT-5-mini", "GPT-5-nano", "length only", "tool name only"):
            print(fmt_pair(f"Qwen - {k}", y, arr(q, rows), arr(sc[k], rows)))
    print("\n" + "=" * 100 + "\n1b. OPERATING POINTS: recall at fixed false-alarm rates (thresholds from the correct rows of the subset), and at 0.5\n" + "=" * 100)
    names = ("ModernBERT v3b", "Qwen2.5-1.5B", "GPT-5-mini", "GPT-5-nano", "length only", "tool name only")
    print(f"{'subset':14s} {'FA':>4s} | " + " | ".join(f"{k:>14s}" for k in names))
    for name, rows in subsets(test)[:2]:
        y = yy(rows)
        for fa in (0.05, 0.10):
            print(f"{name:14s} {fa:4.0%} | " + " | ".join(f"{recall_at_fa(y, arr(sc[k], rows), fa)[0]:14.3f}" for k in names))
        print(f"{name:14s} {'0.5':>4s} | " + " | ".join(f"{(arr(sc[k], rows)[y == 1] >= 0.5).mean():6.3f}/{(arr(sc[k], rows)[y == 0] >= 0.5).mean():.3f}   " for k in names) + "   (recall / false-flag)")
    print("\n" + "=" * 100 + "\n1c. PER CLASS: errors of the class against ALL correct rows of the test set (AUROC; recall at the 5% and 10% false-alarm thresholds of those correct rows; recall at 0.5)\n" + "=" * 100)
    cor = [r for r in test if r["label"] == "correct"]
    cmp = ("ModernBERT v3b", "Qwen2.5-1.5B", "GPT-5-mini", "GPT-5-nano")
    print(f"   {'class':24s} {'n err':>5s} {'natural':>7s} | " + " | ".join(f"{k[:12]:>12s} AUROC  r@5  r@10  r@.5" for k in cmp))
    for k in range(1, 8):
        rows = [r for r in test if r.get("error_class") == k] + cor
        y = yy(rows)
        nat_n = sum(r["origin"] == "natural" for r in rows if r["label"] == "incorrect")
        cells = []
        for nm in cmp:
            s = arr(sc[nm], rows)
            cells.append(f"{'':12s} {roc_auc_score(y, s):.3f} {recall_at_fa(y, s, 0.05)[0]:5.2f} {recall_at_fa(y, s, 0.10)[0]:5.2f} {(s[y == 1] >= 0.5).mean():5.2f}")
        print(f"   {k} {NAMES[k]:22s} {int(y.sum()):5d} {nat_n:7d} | " + " | ".join(cells))
    print("   (classes 2 and 6: mostly induced; class 2 has 4 test errors: recall and AUROC there are not readable)")
    print("\n   paired difference of the per-class AUROC, Qwen - ModernBERT v3b (resampling the rows of the class and the correct rows together):")
    for k in range(1, 8):
        rows = [r for r in test if r.get("error_class") == k] + cor
        print(fmt_pair(f"class {k} {NAMES[k]} (n={sum(r['label'] == 'incorrect' for r in rows)})", yy(rows), arr(q, rows), arr(m, rows)))
    print("\n   natural rows only, per class (errors of the class against the natural correct rows): recall at 0.5, Qwen | ModernBERT v3b | GPT-5-mini")
    for k in range(1, 8):
        rows = [r for r in test if r["origin"] == "natural" and (r.get("error_class") == k or r["label"] == "correct")]
        y = yy(rows)
        if y.sum():
            print(f"   {k} {NAMES[k]:22s} {int(y.sum()):3d} errors | " + " | ".join(f"{(arr(sc[nm], rows)[y == 1] >= 0.5).mean():.2f} ({int((arr(sc[nm], rows)[y == 1] >= 0.5).sum())}/{int(y.sum())})" for nm in ("Qwen2.5-1.5B", "ModernBERT v3b", "GPT-5-mini")))
    print("\n" + "=" * 100 + "\n1d. PER DOMAIN: AUROC (95% CI), recall at 5% and 10% false alarms (thresholds from the correct rows of the domain), recall / false-flag at 0.5\n" + "=" * 100)
    for dom in ("airline", "retail", "telecom"):
        rows = [r for r in test if r["domain"] == dom]
        y = yy(rows)
        print(f"\n{dom}: {len(rows)} rows, {int(y.sum())} errors, {int((1 - y).sum())} correct")
        for nm in cmp:
            s = arr(sc[nm], rows)
            lo, hi = boot_ci(y, s)
            print(f"   {nm:16s} AUROC {roc_auc_score(y, s):.3f} ({lo:.3f}-{hi:.3f})  recall@5%FA {recall_at_fa(y, s, 0.05)[0]:.3f}  @10%FA {recall_at_fa(y, s, 0.10)[0]:.3f}  at 0.5: recall {(s[y == 1] >= 0.5).mean():.3f}, false-flag {(s[y == 0] >= 0.5).mean():.3f}")
        print(fmt_pair("Qwen - ModernBERT v3b", y, arr(q, rows), arr(m, rows)))
        print(fmt_pair("Qwen - GPT-5-mini", y, arr(q, rows), arr(sc["GPT-5-mini"], rows)))


def ens_scores(test, sc):
    q, m = sc["Qwen2.5-1.5B"], sc["ModernBERT v3b"]
    ids = [r["trajectory_id"] for r in test]
    qa, ma = np.array([q[i] for i in ids]), np.array([m[i] for i in ids])
    avg = (qa + ma) / 2
    rk = (rankdata(qa) + rankdata(ma)) / (2 * len(ids))   # ranks over the 363 test rows; uses no label
    return {"average": dict(zip(ids, avg)), "rank-average": dict(zip(ids, rk))}


def section_ensemble(train, test, sc):
    print("*" * 100 + "\nEXPLORATORY. The ensemble was proposed AFTER seeing the test results of both models (D-030). It has no weights and no threshold fitted on the test set\n"
          "(the plain average of the two probabilities; the average of the two rank positions over the 363 test rows, which uses no labels), but the idea itself comes from looking at the test\n"
          "numbers, two variants are reported, and every comparison below is on the same 363 rows: treat any gain as a hypothesis for a new test set, not a result.\n" + "*" * 100)
    e = ens_scores(test, sc)
    allsc = dict(sc, **{"ensemble: average": e["average"], "ensemble: rank-average": e["rank-average"]})
    q, m = sc["Qwen2.5-1.5B"], sc["ModernBERT v3b"]
    ids = [r["trajectory_id"] for r in test]
    rho = spearmanr([m[i] for i in ids], [q[i] for i in ids]).correlation
    print(f"\nSpearman correlation of the two models' scores on the 363 rows: {rho:.3f} (errors only: {spearmanr([m[r['trajectory_id']] for r in test if r['label'] == 'incorrect'], [q[r['trajectory_id']] for r in test if r['label'] == 'incorrect']).correlation:.3f}; correct rows only: {spearmanr([m[r['trajectory_id']] for r in test if r['label'] == 'correct'], [q[r['trajectory_id']] for r in test if r['label'] == 'correct']).correlation:.3f})")
    print("\n" + "=" * 100 + "\nAUROC (95% CI), recall at 5% / 10% false alarms, and the paired bootstrap against ModernBERT v3b alone and against Qwen alone\n" + "=" * 100)
    for name, rows in subsets(test)[:2]:
        y = yy(rows)
        print(f"\n{name}: {len(rows)} rows, {int(y.sum())} errors")
        for k in ("ModernBERT v3b", "Qwen2.5-1.5B", "ensemble: average", "ensemble: rank-average"):
            s = arr(allsc[k], rows)
            lo, hi = boot_ci(y, s)
            print(f"   {k:24s} AUROC {roc_auc_score(y, s):.3f} ({lo:.3f}-{hi:.3f})  recall@5%FA {recall_at_fa(y, s, 0.05)[0]:.3f}  @10%FA {recall_at_fa(y, s, 0.10)[0]:.3f}"
                  + (f"   at 0.5: recall {(s[y == 1] >= 0.5).mean():.3f}, false-flag {(s[y == 0] >= 0.5).mean():.3f}" if k != "ensemble: rank-average" else ""))
        for k in ("ensemble: average", "ensemble: rank-average"):
            print(fmt_pair(f"{k} - ModernBERT v3b", y, arr(allsc[k], rows), arr(m, rows)))
            print(fmt_pair(f"{k} - Qwen", y, arr(allsc[k], rows), arr(q, rows)))
    rows = subsets(test)[2][1]
    y = yy(rows)
    print(f"\nINDUCED rows: {len(rows)} rows, {int(y.sum())} errors (22 correct): AUROC " + ", ".join(f"{k.split(':')[-1].strip() if k.startswith('ensemble') else k} {roc_auc_score(y, arr(allsc[k], rows)):.3f}" for k in ("ModernBERT v3b", "Qwen2.5-1.5B", "ensemble: average", "ensemble: rank-average")))
    print("\n" + "=" * 100 + "\nPer class: recall at 0.5 and at the 10% false-alarm threshold (errors of the class against all correct rows), ModernBERT v3b | Qwen | average ensemble\n" + "=" * 100)
    cor = [r for r in test if r["label"] == "correct"]
    for k in range(1, 8):
        rows = [r for r in test if r.get("error_class") == k] + cor
        y = yy(rows)
        cells = []
        for nm in ("ModernBERT v3b", "Qwen2.5-1.5B", "ensemble: average"):
            s = arr(allsc[nm], rows)
            cells.append(f"AUROC {roc_auc_score(y, s):.3f} r@.5 {(s[y == 1] >= 0.5).mean():.2f} r@10%FA {recall_at_fa(y, s, 0.10)[0]:.2f}")
        print(f"   {k} {NAMES[k]:22s} n={int(y.sum()):2d} | " + " | ".join(cells))
    print("\n" + "=" * 100 + "\nDo the two models catch different errors? (errors caught at the 10%-false-alarm threshold of each model, overall rows)\n" + "=" * 100)
    y = yy(test)
    caught = {}
    for nm in ("ModernBERT v3b", "Qwen2.5-1.5B"):
        s = arr(sc[nm], test)
        thr = recall_at_fa(y, s, 0.10)[2]
        caught[nm] = (s >= thr) & (y == 1)
        print(f"   {nm:16s} threshold {thr:.3f}: catches {int(caught[nm].sum())} of {int(y.sum())} errors")
    a, b = caught["ModernBERT v3b"], caught["Qwen2.5-1.5B"]
    print(f"   caught by both {int((a & b).sum())}, only ModernBERT {int((a & ~b).sum())}, only Qwen {int((~a & b).sum())}, neither {int((~a & ~b & (y == 1)).sum())}")
    print("   by class (only-ModernBERT / only-Qwen / both): " + "; ".join(f"{k}: {int(sum(a[i] and not b[i] for i, r in enumerate(test) if r.get('error_class') == k))}/{int(sum(b[i] and not a[i] for i, r in enumerate(test) if r.get('error_class') == k))}/{int(sum(a[i] and b[i] for i, r in enumerate(test) if r.get('error_class') == k))}" for k in range(1, 8)))


def val_group(r):
    if r.get("real_origin"):
        return "real-origin: " + str(r.get("origin"))
    if str(r.get("source", "")) == "critictool":
        return "CriticTool"
    if r["domain"] == "bfcl":
        return "BFCL"
    return "tau2 domains: synthetic, state flips, gold"


def section_gap(train, test, sc):
    pred = {n: {s: load(R / d / f"{s}_predictions.jsonl") for s in ("test", "test_synth", "val")} for n, d in (("ModernBERT v3b", "final_v3b"), ("Qwen2.5-1.5B", "final_qwen_v3b"))}
    print("=" * 100 + "\nSYNTHETIC / VALIDATION / REAL, side by side (AUROC with 95% bootstrap CI, 2,000 resamples), from the prediction files\n" + "=" * 100)

    def cell(rows):
        y = yy(rows)
        s = np.array([r["p_incorrect"] for r in rows])
        if not (0 < y.sum() < len(y)):
            return "n/a"
        lo, hi = boot_ci(y, s)
        return f"{roc_auc_score(y, s):.3f} ({lo:.3f}-{hi:.3f}), n={len(rows)}"

    def sel(n, split, f=lambda r: True):
        return [r for r in pred[n][split] if f(r)]

    lines = [("REAL test (363 rows, human-reviewed)", "test", lambda r: True),
             ("   natural rows", "test", lambda r: r["origin"] == "natural"), ("   induced rows", "test", lambda r: r["origin"] == "induced"),
             ("SYNTHETIC test_synth (374 rows, Classes 2 and 6)", "test_synth", lambda r: True),
             ("VALIDATION (749 rows)", "val", lambda r: True)]
    for g in ("BFCL", "CriticTool", "tau2 domains: synthetic, state flips, gold", "real-origin: natural", "real-origin: induced"):
        lines.append((f"   validation, {g}", "val", lambda r, g=g: val_group(r) == g))
    lines.append(("   validation, real-origin rows together", "val", lambda r: bool(r.get("real_origin"))))
    print(f"{'':58s} | {'ModernBERT v3b':>34s} | {'Qwen2.5-1.5B':>34s}")
    for label, split, f in lines:
        print(f"{label:58s} | " + " | ".join(f"{cell(sel(n, split, f)):>34s}" for n in ("ModernBERT v3b", "Qwen2.5-1.5B")))
    rv = {n: roc_auc_score(yy(sel(n, "val", lambda r: bool(r.get('real_origin')))), [r["p_incorrect"] for r in sel(n, "val", lambda r: bool(r.get('real_origin')))]) for n in pred}
    rt = {n: roc_auc_score(yy(pred[n]["test"]), [r["p_incorrect"] for r in pred[n]["test"]]) for n in pred}
    st = {n: roc_auc_score(yy(pred[n]["test_synth"]), [r["p_incorrect"] for r in pred[n]["test_synth"]]) for n in pred}
    print("\nvalidation real-origin AUROC minus real test AUROC: " + ", ".join(f"{n} {rv[n] - rt[n]:+.3f}" for n in pred))
    print("test_synth AUROC minus real test AUROC:             " + ", ".join(f"{n} {st[n] - rt[n]:+.3f}" for n in pred))
    print("\nthe real-origin validation rows are NOT independent of the training rows' sources: a validation row is a different trajectory, but real-origin training and validation rows come from the same tau2 runs and tasks\n"
          "families (the split is grouped by (domain, task), so no task is in both, but the agents, domains and error styles are); the real TEST rows are the human-reviewed rows of D-022.")
    cnt = {n: (int(yy(sel(n, "val", lambda r: bool(r.get('real_origin')))).sum()), len(sel(n, "val", lambda r: bool(r.get('real_origin'))))) for n in pred}
    print(f"real-origin validation: {cnt['Qwen2.5-1.5B'][0]} errors in {cnt['Qwen2.5-1.5B'][1]} rows: the interval is wide (see above).")


def section_readme(train, test, sc):
    e = ens_scores(test, sc)
    allsc = dict(sc, **{"ensemble: average": e["average"], "ensemble: rank-average": e["rank-average"]})
    sp = json.load(open(R / "final_qwen_v3b" / "speed_cost.json"))
    nat, ind = subsets(test)[1][1], subsets(test)[2][1]
    print("TABLE (363 real rows): AUROC (CI) overall | natural | recall@5%FA | recall@10%FA | recall at 0.5 / false-flag at 0.5 | natural recall@5% / @10%")
    for nm in ("ModernBERT v3b", "Qwen2.5-1.5B", "ensemble: average", "ensemble: rank-average", "GPT-5-mini", "GPT-5-nano", "length only", "tool name only"):
        y, s = yy(test), arr(allsc[nm], test)
        lo, hi = boot_ci(y, s)
        yn, sn = yy(nat), arr(allsc[nm], nat)
        lon, hin = boot_ci(yn, sn)
        yi, si = yy(ind), arr(allsc[nm], ind)
        loi, hii = boot_ci(yi, si)
        print(f"| {nm:24s} | {roc_auc_score(y, s):.3f} ({lo:.3f}-{hi:.3f}) | {roc_auc_score(yn, sn):.3f} ({lon:.3f}-{hin:.3f}) | induced {roc_auc_score(yi, si):.3f} ({loi:.3f}-{hii:.3f}) | {recall_at_fa(y, s, 0.05)[0]:.3f} | {recall_at_fa(y, s, 0.10)[0]:.3f} | {(s[y == 1] >= 0.5).mean():.3f} / {(s[y == 0] >= 0.5).mean():.3f} | {recall_at_fa(yn, sn, 0.05)[0]:.3f} / {recall_at_fa(yn, sn, 0.10)[0]:.3f} |")
    print("\nPER CLASS recall at 0.5 (errors of the class; n total / n natural), ModernBERT | Qwen | ensemble (average) | GPT-5-mini | GPT-5-nano")
    for k in range(1, 8):
        rows = [r for r in test if r.get("error_class") == k]
        y = np.ones(len(rows))
        cells = [f"{(arr(allsc[nm], rows) >= 0.5).mean():.2f}" for nm in ("ModernBERT v3b", "Qwen2.5-1.5B", "ensemble: average", "GPT-5-mini", "GPT-5-nano")]
        print(f"| {k} {NAMES[k]:22s} | {len(rows)} / {sum(r['origin'] == 'natural' for r in rows)} | " + " | ".join(cells) + " |")
    print("\nfalse-flag rate on the 193 correct rows at 0.5:", ", ".join(f"{nm} {(arr(allsc[nm], [r for r in test if r['label'] == 'correct']) >= 0.5).mean():.2f}" for nm in ("ModernBERT v3b", "Qwen2.5-1.5B", "ensemble: average", "GPT-5-mini", "GPT-5-nano")))
    print("\nSPEED AND COST (speed_cost.json):", json.dumps({k: sp[k] for k in ("modernbert_cpu",)}, indent=0)[:400])


if __name__ == "__main__":
    train, test, sc = setup()
    {"paired": section_paired, "ensemble": section_ensemble, "gap": section_gap, "readme": section_readme}[sys.argv[1]](train, test, sc)
