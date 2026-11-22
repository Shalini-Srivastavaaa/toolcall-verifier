"""Analysis of the v3b run against the judges, final_v2 and the shortcut baselines (D-029). No API, no model.

    .venv/bin/python3 eval/analyze_final_v3b.py > results/final_v3b/analysis.txt

Inputs: data/derived/final_v3b/{train,test}.jsonl (the training split of v3b fits the length and tool-name baselines), results/final_v3b/test_predictions.jsonl, results/final_v2/test_predictions.jsonl,
results/judge_new_test/openai-{mini,nano}_test_predictions.jsonl. Reuses the helpers of eval/analyze_final_v2.py.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_final_v2 import READ_PREFIXES, boot_ci, length, load, paired, recall_at_fa, scores  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
NAMES = {1: "wrong arg", 2: "stale state", 3: "wrong tool", 4: "repeated call", 5: "failure to propagate", 6: "ignored error", 7: "policy violation"}


def tool_scores(train, rows, prior=2.0):
    E, C = Counter(), Counter()
    for r in train:
        (E if r["label"] == "incorrect" else C)[r["candidate_tool_call"]["name"]] += 1
    base = sum(E.values()) / (sum(E.values()) + sum(C.values()))
    return {r["trajectory_id"]: (E[r["candidate_tool_call"]["name"]] + prior * base) / (E[r["candidate_tool_call"]["name"]] + C[r["candidate_tool_call"]["name"]] + prior) for r in rows}


def main():
    train = load(ROOT / "data" / "derived" / "final_v3b" / "train.jsonl")
    test = load(ROOT / "data" / "derived" / "final_v3b" / "test.jsonl")
    v3 = scores(ROOT / "results" / "final_v3b" / "test_predictions.jsonl")
    v2 = scores(ROOT / "results" / "final_v2" / "test_predictions.jsonl")
    mini = scores(ROOT / "results" / "judge_new_test" / "openai-mini_test_predictions.jsonl")
    nano = scores(ROOT / "results" / "judge_new_test" / "openai-nano_test_predictions.jsonl")
    clf = LogisticRegression(max_iter=1000).fit(np.array([[length(r)] for r in train]), np.array([r["label"] == "incorrect" for r in train]).astype(int))
    ln = {r["trajectory_id"]: float(clf.predict_proba([[length(r)]])[0, 1]) for r in test}
    tn = tool_scores(train, test)
    arr = lambda d, rows: np.array([d[r["trajectory_id"]] for r in rows])
    nat = [r for r in test if r["origin"] == "natural"]
    ind = [r for r in test if r["origin"] == "induced"]
    yy = lambda rows: np.array([r["label"] == "incorrect" for r in rows]).astype(int)

    print("=" * 100 + "\nVERIFYING THE FIGURES GIVEN FOR THE RUN\n" + "=" * 100)
    for name, rows in (("all", test), ("natural", nat), ("induced", ind)):
        y, s = yy(rows), arr(v3, rows)
        lo, hi = boot_ci(y, s)
        print(f"{name:8s} n {len(rows):3d} ({int(y.sum()):3d} errors): AUROC {roc_auc_score(y, s):.3f} (95% CI {lo:.3f}-{hi:.3f}); at 0.5 recall {(s[y == 1] >= 0.5).mean():.2f}, false-flag {(s[y == 0] >= 0.5).mean():.2f}"
              f"   [final_v2: AUROC {roc_auc_score(y, arr(v2, rows)):.3f}, false-flag {(arr(v2, rows)[y == 0] >= 0.5).mean():.2f}]")

    print("\n" + "=" * 100 + "\n1. PAIRED BOOTSTRAP OF AUROC DIFFERENCES: v3b minus the other (2,000 resamples of the rows of the subset, the same resample for both scores)\n" + "=" * 100)
    for name, rows in (("OVERALL", test), ("NATURAL rows", nat), ("INDUCED rows", ind)):
        y = yy(rows)
        print(f"\n{name}: {len(rows)} rows, {int(y.sum())} errors, {int((1 - y).sum())} correct")
        sc = {"v3b": arr(v3, rows), "final_v2": arr(v2, rows), "gpt-5-mini": arr(mini, rows), "gpt-5-nano": arr(nano, rows), "length only": arr(ln, rows), "tool name only": arr(tn, rows)}
        for k, s in sc.items():
            lo, hi = boot_ci(y, s)
            print(f"   {k:16s} AUROC {roc_auc_score(y, s):.3f}  (95% CI {lo:.3f}-{hi:.3f})")
        for k in ("final_v2", "gpt-5-mini", "gpt-5-nano", "length only", "tool name only"):
            d, lo, hi, share, p = paired(y, sc["v3b"], sc[k])
            print(f"   v3b - {k:14s}: {d:+.3f}  paired 95% CI {lo:+.3f} to {hi:+.3f}; v3b better in {share:.1%} of resamples; two-sided p about {p:.3f}" + ("   <- interval excludes 0" if lo > 0 or hi < 0 else "   (interval includes 0)"))

    print("\n" + "=" * 100 + "\n2. OPERATING POINTS: recall at fixed false-alarm rates (thresholds from the correct rows of the subset)\n" + "=" * 100)
    print(f"{'subset':32s} {'FA':>5s} | {'v3b recall':>10s} (FA, thr) | {'mini recall':>11s} (FA, thr) | {'final_v2':>8s}")
    for name, rows in (("overall", test), ("natural rows", nat)):
        y = yy(rows)
        for fa in (0.05, 0.10, 0.20):
            a, fa_a, ta = recall_at_fa(y, arr(v3, rows), fa)
            b, fa_b, tb = recall_at_fa(y, arr(mini, rows), fa)
            c, _, _ = recall_at_fa(y, arr(v2, rows), fa)
            print(f"{name:32s} {fa:5.0%} | {a:10.3f} ({fa_a:.3f}, {ta:.3f}) | {b:11.3f} ({fa_b:.3f}, {tb:.3f}) | {c:8.3f}")
        s3, sm = arr(v3, rows), arr(mini, rows)
        print(f"{'':32s} {'0.5':>5s} | recall {(s3[y == 1] >= 0.5).mean():.3f}, false-flag {(s3[y == 0] >= 0.5).mean():.3f} | recall {(sm[y == 1] >= 0.5).mean():.3f}, false-flag {(sm[y == 0] >= 0.5).mean():.3f}")
    print("\nper class (errors of the class against ALL correct rows; recall at the 10% false-alarm threshold of the correct rows, at 0.5, and AUROC; v3b | gpt-5-mini | final_v2):")
    cor = [r for r in test if r["label"] == "correct"]
    print(f"   {'class':26s} {'n':>3s} | {'AUROC v3b':>9s} {'mini':>6s} {'v2':>6s} | {'rec@10%FA v3b':>13s} {'mini':>6s} {'v2':>6s} | {'rec@0.5 v3b':>11s} {'mini':>6s} {'v2':>6s}")
    for k in range(1, 8):
        rows = [r for r in test if r.get("error_class") == k] + cor
        y = yy(rows)
        out = []
        for d in (v3, mini, v2):
            s = arr(d, rows)
            out.append((roc_auc_score(y, s), recall_at_fa(y, s, 0.10)[0], (s[y == 1] >= 0.5).mean()))
        print(f"   {k} {NAMES[k]:24s} {int(y.sum()):3d} | {out[0][0]:9.3f} {out[1][0]:6.3f} {out[2][0]:6.3f} | {out[0][1]:13.3f} {out[1][1]:6.3f} {out[2][1]:6.3f} | {out[0][2]:11.3f} {out[1][2]:6.3f} {out[2][2]:6.3f}")

    print("\n" + "=" * 100 + "\n3. THE CLASS 7 DROP\n" + "=" * 100)
    for label, keep in (("all rows", lambda r: True), ("without cancel_reservation rows", lambda r: r["candidate_tool_call"]["name"] != "cancel_reservation")):
        for scope, base in (("all", test), ("natural", nat)):
            rows = [r for r in base if keep(r) and (r["label"] == "correct" or r.get("error_class") == 7)]
            y = yy(rows)
            res = []
            for nme, d in (("final_v2", v2), ("v3b", v3), ("mini", mini)):
                s = arr(d, rows)
                res.append(f"{nme} AUROC {roc_auc_score(y, s):.3f} recall@0.5 {(s[y == 1] >= 0.5).mean():.2f} (false-flag {(s[y == 0] >= 0.5).mean():.2f}) recall@10%FA {recall_at_fa(y, s, 0.10)[0]:.2f} @20% {recall_at_fa(y, s, 0.20)[0]:.2f}")
            print(f"Class 7 vs correct, {scope:8s} rows, {label:32s} ({int(y.sum())} errors, {int((1 - y).sum())} correct):\n     " + "\n     ".join(res))
    c7 = [r for r in test if r.get("error_class") == 7]
    print(f"\nClass 7 test errors by tool: {dict(Counter(r['candidate_tool_call']['name'] for r in c7).most_common(8))}")
    for nme, d in (("final_v2", v2), ("v3b", v3)):
        s = arr(d, c7)
        canc = np.array([r["candidate_tool_call"]["name"] == "cancel_reservation" for r in c7])
        print(f"   {nme}: recall at 0.5 on Class 7 errors: cancel_reservation {(s[canc] >= 0.5).mean():.2f} ({int(canc.sum())} rows), all other tools {(s[~canc] >= 0.5).mean():.2f} ({int((~canc).sum())}); median score {np.median(s[canc]):.2f} and {np.median(s[~canc]):.2f}")
        cc = [r for r in test if r["label"] == "correct" and r["candidate_tool_call"]["name"] == "cancel_reservation"]
        print(f"        correct cancel_reservation rows (14): median score {np.median(arr(d, cc)):.2f}, share at or above 0.5 {(arr(d, cc) >= 0.5).mean():.2f}, at or above 0.9 {(arr(d, cc) >= 0.9).mean():.2f}")
    print("\nClass 7 by domain (errors vs the correct rows of the domain): AUROC and recall at 0.5, final_v2 -> v3b; real Class 7 training errors in the domain, v2 data -> v3b data")
    tr2 = load(ROOT / "data" / "derived" / "final" / "train.jsonl")
    for dom in ("airline", "retail", "telecom"):
        rows = [r for r in test if r["domain"] == dom and (r["label"] == "correct" or r.get("error_class") == 7)]
        y = yy(rows)
        n2 = sum(1 for r in tr2 if r["label"] == "incorrect" and r.get("error_class") == 7 and r.get("real_origin") and r["domain"] == dom)
        n3 = sum(1 for r in train if r["label"] == "incorrect" and r.get("error_class") == 7 and r.get("real_origin") and r["domain"] == dom)
        a, b = arr(v2, rows), arr(v3, rows)
        print(f"   {dom:8s} {int(y.sum()):3d} errors: AUROC {roc_auc_score(y, a):.3f} -> {roc_auc_score(y, b):.3f}; recall@0.5 {(a[y == 1] >= 0.5).mean():.2f} -> {(b[y == 1] >= 0.5).mean():.2f}; real Class 7 training errors {n2} -> {n3}")
    print("\nthe real Class 7 labels that were added (v3b data, not in v2 data): " + str(Counter((r["domain"], r["class7_reason"]) for r in train if r["label"] == "incorrect" and r.get("error_class") == 7 and r.get("batch") == "train_more").most_common(6)))

    print("\n" + "=" * 100 + "\n4. CORRECT rows scored at or above 0.9 by v3b\n" + "=" * 100)
    hi_rows = [r for r in test if r["label"] == "correct" and v3[r["trajectory_id"]] >= 0.9]
    allc = [r for r in test if r["label"] == "correct"]
    print(f"{len(hi_rows)} of {len(allc)} correct rows (final_v2: {sum(v2[r['trajectory_id']] >= 0.9 for r in allc)}); at or above 0.7: {sum(v3[r['trajectory_id']] >= 0.7 for r in allc)} (final_v2 {sum(v2[r['trajectory_id']] >= 0.7 for r in allc)})")
    for title, key in (("origin", lambda r: r["origin"]), ("domain", lambda r: r["domain"]), ("source", lambda r: r["source"]), ("found_in", lambda r: r["found_in"]),
                       ("candidate tool", lambda r: r["candidate_tool_call"]["name"]), ("read-only call", lambda r: r["candidate_tool_call"]["name"].startswith(READ_PREFIXES)),
                       ("JSON-style agent text", lambda r: any(t.get("role") == "assistant" and t.get("content", "").lstrip().startswith("{") for t in r["turns"]))):
        a, b = Counter(key(r) for r in hi_rows), Counter(key(r) for r in allc)
        print(f"   by {title:22s}: " + "; ".join(f"{k}: {v} of {b[k]}" for k, v in a.most_common(6)))
    print(f"   turns of history: median {np.median([len(r['turns']) for r in hi_rows]):.0f} (those rows) against {np.median([len(r['turns']) for r in allc]):.0f} (all correct rows); gpt-5-mini median score on them {np.median([mini[r['trajectory_id']] for r in hi_rows]):.2f}")
    cc = [r for r in test if r["candidate_tool_call"]["name"] == "cancel_reservation"]
    for lab in ("correct", "incorrect"):
        s = arr(v3, [r for r in cc if r["label"] == lab])
        print(f"   cancel_reservation {lab:9s} ({len(s)} rows): v3b median {np.median(s):.2f}, share >= 0.9 {(s >= 0.9).mean():.2f}")

    print("\n" + "=" * 100 + "\nTABLE FOR THE README: AUROC (95% bootstrap CI, 2,000 resamples) overall / natural / induced, recall at 5% and 10% false alarms (overall rows), recall and false-flag at 0.5\n" + "=" * 100)
    for name, d in (("ModernBERT v3b", v3), ("ModernBERT final_v2", v2), ("gpt-5-mini", mini), ("gpt-5-nano", nano), ("length only", ln), ("tool name only", tn)):
        cells = []
        for rows in (test, nat, ind):
            y, s = yy(rows), arr(d, rows)
            lo, hi = boot_ci(y, s)
            cells.append(f"{roc_auc_score(y, s):.3f} ({lo:.3f}-{hi:.3f})")
        y, s = yy(test), arr(d, test)
        print(f"| {name:20s} | " + " | ".join(cells) + f" | {recall_at_fa(y, s, 0.05)[0]:.3f} | {recall_at_fa(y, s, 0.10)[0]:.3f} | {(s[y == 1] >= 0.5).mean():.3f} / {(s[y == 0] >= 0.5).mean():.3f} |")
    yn, sn = yy(nat), None
    print("natural rows only, recall at 5% / 10% false alarms: " + "; ".join(f"{name} {recall_at_fa(yn, arr(d, nat), 0.05)[0]:.3f} / {recall_at_fa(yn, arr(d, nat), 0.10)[0]:.3f}" for name, d in (("v3b", v3), ("final_v2", v2), ("gpt-5-mini", mini), ("gpt-5-nano", nano))))
    ic = [r for r in ind if r["label"] == "correct"]
    n6, n2 = sum(v3[r["trajectory_id"]] >= 0.5 for r in ic), sum(v2[r["trajectory_id"]] >= 0.5 for r in ic)
    b, c = sum(v2[r["trajectory_id"]] >= 0.5 and v3[r["trajectory_id"]] < 0.5 for r in ic), sum(v2[r["trajectory_id"]] < 0.5 and v3[r["trajectory_id"]] >= 0.5 for r in ic)
    print(f"induced correct rows flagged at 0.5: final_v2 {n2}/22, v3b {n6}/22 (v2 flags and v3b does not: {b}; the reverse: {c})")
    cs = [r for r in test if r["candidate_tool_call"]["name"] == "cancel_reservation"]
    y = yy(cs)
    print(f"cancel_reservation rows (14 correct, 6 errors): AUROC within the tool: final_v2 {roc_auc_score(y, arr(v2, cs)):.3f}, v3b {roc_auc_score(y, arr(v3, cs)):.3f}, gpt-5-mini {roc_auc_score(y, arr(mini, cs)):.3f}")


if __name__ == "__main__":
    main()
