"""Analysis of the real-weighted run final_v2 (D-027): paired comparison with the judges, operating points, high-scoring correct rows, weak spots. No API, no model.

    .venv/bin/python3 eval/analyze_final_v2.py > results/final_v2/analysis.txt

Inputs: data/derived/final/*.jsonl, results/final_v2/test_predictions.jsonl, results/judge_new_test/openai-{mini,nano}_test_predictions.jsonl.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parent.parent
F = ROOT / "data" / "derived" / "final"
READ_PREFIXES = ("get_", "search_", "find_", "list_", "check_", "can_")


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def scores(path):
    return {r["trajectory_id"]: r["p_incorrect"] for r in load(path)}


def length(r):
    return float(np.log1p(len(json.dumps(r["turns"])) + len(json.dumps(r["candidate_tool_call"]))))


def recall_at_fa(y, s, fa):
    fpr, tpr, thr = roc_curve(y, s)
    k = np.flatnonzero(fpr <= fa)[-1]
    return float(tpr[k]), float(fpr[k]), float(thr[k])


def paired(y, a, b, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            d.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    d = np.array(d)
    share = (d > 0).mean()
    return roc_auc_score(y, a) - roc_auc_score(y, b), np.percentile(d, 2.5), np.percentile(d, 97.5), share, 2 * min(share, 1 - share)


def boot_ci(y, s, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    v = [roc_auc_score(y[i], s[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(B)) if y[i].min() != y[i].max()]
    return np.percentile(v, [2.5, 97.5])


def main():
    train, val, test = (load(F / f"{s}.jsonl") for s in ("train", "val", "test"))
    model = scores(ROOT / "results" / "final_v2" / "test_predictions.jsonl")
    mini = scores(ROOT / "results" / "judge_new_test" / "openai-mini_test_predictions.jsonl")
    nano = scores(ROOT / "results" / "judge_new_test" / "openai-nano_test_predictions.jsonl")
    clf = LogisticRegression(max_iter=1000).fit(np.array([[length(r)] for r in train]), np.array([r["label"] == "incorrect" for r in train]).astype(int))
    ln = {r["trajectory_id"]: float(clf.predict_proba([[length(r)]])[0, 1]) for r in test}
    ids = [r["trajectory_id"] for r in test]
    ids_s = {i: r for i, r in zip(ids, test)}

    def arr(d, rows):
        return np.array([d[r["trajectory_id"]] for r in rows])

    print("=" * 100 + "\nVERIFYING THE FIGURES GIVEN FOR THE RUN\n" + "=" * 100)
    for name, rows in (("all", test), ("natural", [r for r in test if r["origin"] == "natural"]), ("induced", [r for r in test if r["origin"] == "induced"])):
        y = np.array([r["label"] == "incorrect" for r in rows]).astype(int)
        s = arr(model, rows)
        lo, hi = boot_ci(y, s)
        print(f"{name:8s} n {len(rows):3d} ({int(y.sum()):3d} errors): AUROC {roc_auc_score(y, s):.3f} (95% CI {lo:.3f}-{hi:.3f}); at 0.5 recall {(s[y == 1] >= 0.5).mean():.2f}, false-flag {(s[y == 0] >= 0.5).mean():.2f}")
    for d in ("airline", "retail", "telecom"):
        rows = [r for r in test if r["domain"] == d]
        y = np.array([r["label"] == "incorrect" for r in rows]).astype(int)
        s = arr(model, rows)
        print(f"{d:8s} n {len(rows):3d}: AUROC {roc_auc_score(y, s):.3f}, recall at 0.5 {(s[y == 1] >= 0.5).mean():.2f}, false-flag {(s[y == 0] >= 0.5).mean():.2f}")

    print("\n" + "=" * 100 + "\n1. PAIRED BOOTSTRAP OF AUROC DIFFERENCES (2,000 resamples of the test rows of the subset; same resample for both scores)\n" + "=" * 100)
    for name, rows in (("NATURAL rows (headline)", [r for r in test if r["origin"] == "natural"]), ("overall", test), ("INDUCED rows", [r for r in test if r["origin"] == "induced"])):
        y = np.array([r["label"] == "incorrect" for r in rows]).astype(int)
        print(f"\n{name}: {len(rows)} rows, {int(y.sum())} errors, {int((1 - y).sum())} correct")
        sc = {"model (final_v2)": arr(model, rows), "gpt-5-mini": arr(mini, rows), "gpt-5-nano": arr(nano, rows), "length only": arr(ln, rows)}
        for k, s in sc.items():
            lo, hi = boot_ci(y, s)
            print(f"   {k:18s} AUROC {roc_auc_score(y, s):.3f}  (95% CI {lo:.3f}-{hi:.3f})")
        for k in ("gpt-5-mini", "gpt-5-nano", "length only"):
            d, lo, hi, share, p = paired(y, sc["model (final_v2)"], sc[k])
            print(f"   model - {k:12s}: {d:+.3f}  paired 95% CI {lo:+.3f} to {hi:+.3f}; model better in {share:.1%} of resamples; two-sided p about {p:.3f}" + ("   <- interval excludes 0" if lo > 0 or hi < 0 else "   (interval includes 0)"))
    nat = [r for r in test if r["origin"] == "natural"]
    for k in (7,):
        rows = [r for r in nat if r["label"] == "correct" or r.get("error_class") == k]
        y = np.array([r["label"] == "incorrect" for r in rows]).astype(int)
        d, lo, hi, share, p = paired(y, arr(model, rows), arr(mini, rows))
        print(f"\nnatural Class {k} against natural correct rows ({int(y.sum())} errors): model {roc_auc_score(y, arr(model, rows)):.3f}, mini {roc_auc_score(y, arr(mini, rows)):.3f}, difference {d:+.3f} (95% CI {lo:+.3f} to {hi:+.3f})")

    print("\n" + "=" * 100 + "\n3. OPERATING POINTS: recall at fixed false-alarm rates (threshold from the correct rows of the subset; best recall with false-alarm rate <= the target)\n" + "=" * 100)
    subsets = [("natural rows", nat, None), ("all rows", test, None), ("Class 7 errors, all correct rows", [r for r in test if r["label"] == "correct" or r.get("error_class") == 7], None),
               ("natural Class 7 errors, natural correct rows", [r for r in nat if r["label"] == "correct" or r.get("error_class") == 7], None)]
    print(f"{'subset':46s} {'FA target':>9s} | {'model recall':>12s} (FA, thr) | {'mini recall':>11s} (FA, thr)")
    for name, rows, _ in subsets:
        y = np.array([r["label"] == "incorrect" for r in rows]).astype(int)
        for fa in (0.05, 0.10, 0.20):
            rm, fm, tm = recall_at_fa(y, arr(model, rows), fa)
            rj, fj, tj = recall_at_fa(y, arr(mini, rows), fa)
            print(f"{name:46s} {fa:9.0%} | {rm:12.3f} ({fm:.3f}, {tm:.3f}) | {rj:11.3f} ({fj:.3f}, {tj:.3f})")
        print(f"{'':46s} {'at 0.5':>9s} | model recall {(arr(model, rows)[y == 1] >= 0.5).mean():.3f}, false-flag {(arr(model, rows)[y == 0] >= 0.5).mean():.3f} | mini recall {(arr(mini, rows)[y == 1] >= 0.5).mean():.3f}, false-flag {(arr(mini, rows)[y == 0] >= 0.5).mean():.3f}")
    hi_rows = [r for r in test if r["label"] == "correct" and model[r["trajectory_id"]] >= 0.9]
    print(f"\nCORRECT rows the model scores at or above 0.9: {len(hi_rows)} of {sum(r['label'] == 'correct' for r in test)} correct rows")
    allc = [r for r in test if r["label"] == "correct"]
    for title, key in (("origin", lambda r: r["origin"]), ("domain", lambda r: r["domain"]), ("source", lambda r: r["source"]), ("found_in", lambda r: r["found_in"]),
                       ("candidate tool", lambda r: r["candidate_tool_call"]["name"]), ("read-only call", lambda r: r["candidate_tool_call"]["name"].startswith(READ_PREFIXES)),
                       ("JSON-style agent text", lambda r: any(t.get("role") == "assistant" and t.get("content", "").lstrip().startswith("{") for t in r["turns"]))):
        a, b = Counter(key(r) for r in hi_rows), Counter(key(r) for r in allc)
        print(f"   by {title:22s}: " + "; ".join(f"{k}: {v} of {b[k]}" for k, v in a.most_common(6)))
    ln_all = np.array([len(r["turns"]) for r in allc]); ln_hi = np.array([len(r["turns"]) for r in hi_rows]) if hi_rows else np.array([0])
    print(f"   turns of history: median {np.median(ln_hi):.0f} (those rows) against {np.median(ln_all):.0f} (all correct rows)")
    tags = Counter(t for r in hi_rows for t in r.get("tags", []))
    print("   tags:", dict(tags), "| mini's score on these rows: median", f"{np.median([mini[r['trajectory_id']] for r in hi_rows]):.2f}" if hi_rows else "n/a")

    print("\n" + "=" * 100 + "\n4. WEAK SPOTS: natural Class 4 and telecom\n" + "=" * 100)
    c4 = [r for r in nat if r.get("error_class") == 4]
    print(f"natural Class 4 test errors: {len(c4)}; recall at 0.5 {np.mean([model[r['trajectory_id']] >= 0.5 for r in c4]):.2f}; median score {np.median([model[r['trajectory_id']] for r in c4]):.2f}; mini median {np.median([mini[r['trajectory_id']] for r in c4]):.2f}")
    print("   by domain:", dict(Counter(r["domain"] for r in c4)), "| read-only calls:", sum(r["candidate_tool_call"]["name"].startswith(READ_PREFIXES) for r in c4), "of", len(c4), "| tools:", dict(Counter(r["candidate_tool_call"]["name"] for r in c4).most_common(6)))
    print("   found_in:", dict(Counter(r["found_in"] for r in c4)), "| the call was already made earlier with the same arguments:", sum(any(t.get("role") == "assistant_tool_call" and t["name"] == r["candidate_tool_call"]["name"] and t.get("arguments") == r["candidate_tool_call"]["arguments"] for t in r["turns"]) for r in c4))
    for lab, rows in (("train natural Class 4", [r for r in train if r["label"] == "incorrect" and r.get("error_class") == 4 and r["origin"] == "natural"]),
                      ("train synthetic Class 4", [r for r in train if r["label"] == "incorrect" and r.get("error_class") == 4 and r["origin"] == "synthetic"])):
        print(f"   {lab}: {len(rows)} rows; domains {dict(Counter(r['domain'] for r in rows))}; read-only calls {sum(r['candidate_tool_call']['name'].startswith(READ_PREFIXES) for r in rows)}; top tools {dict(Counter(r['candidate_tool_call']['name'] for r in rows).most_common(4))}")
    syn4 = [r for r in train if r["label"] == "incorrect" and r.get("error_class") == 4 and r["origin"] == "synthetic"]
    print(f"   synthetic Class 4: the injected duplicate result {{'status': 'success'}} appears in {sum(any(t.get('role') == 'tool' and 'success' in str(t.get('content', '')) for t in r['turns'][-4:]) for r in syn4)} of {len(syn4)} rows' last turns")
    tel = [r for r in test if r["domain"] == "telecom"]
    te_err = [r for r in tel if r["label"] == "incorrect"]
    print(f"\ntelecom test: {len(tel)} rows, {len(te_err)} errors; recall at 0.5 {np.mean([model[r['trajectory_id']] >= 0.5 for r in te_err]):.2f}; correct rows false-flag {np.mean([model[r['trajectory_id']] >= 0.5 for r in tel if r['label'] == 'correct']):.2f}")
    print(f"   telecom errors by class (n / recall at 0.5): " + ", ".join(f"{k}: {sum(r['error_class'] == k for r in te_err)} / {np.mean([model[r['trajectory_id']] >= 0.5 for r in te_err if r['error_class'] == k]):.2f}" for k in range(1, 8) if any(r['error_class'] == k for r in te_err)))
    print(f"   telecom errors by origin: {dict(Counter(r['origin'] for r in te_err))}; by read-only call: {sum(r['candidate_tool_call']['name'].startswith(READ_PREFIXES) for r in te_err)} of {len(te_err)}")
    tt = [r for r in train if r["domain"] == "telecom"]
    print(f"   telecom TRAIN rows: {len(tt)} ({sum(r['label'] == 'incorrect' for r in tt)} errors); real-origin {sum(bool(r.get('real_origin')) for r in tt)}; errors by class and origin: " +
          ", ".join(f"{k}: {Counter(r['origin'] for r in tt if r['label'] == 'incorrect' and r['error_class'] == k)}" for k in range(1, 8) if any(r['label'] == 'incorrect' and r['error_class'] == k for r in tt)))
    print(f"   telecom real errors in train by class: {dict(Counter(r['error_class'] for r in tt if r['label'] == 'incorrect' and r.get('real_origin')))}; real correct rows {sum(1 for r in tt if r['label'] == 'correct' and r.get('real_origin'))}")
    by_dom = defaultdict(Counter)
    for r in train:
        if r.get("real_origin") and r["label"] == "incorrect":
            by_dom[r["domain"]][r["error_class"]] += 1
    print("   real errors in train by domain and class:", {d: dict(c) for d, c in by_dom.items()})


if __name__ == "__main__":
    main()
