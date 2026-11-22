"""Shortcut baselines: how well can the label be predicted WITHOUT reading the conversation? (D-019)

Two baselines, printed side by side on the validation fold of the notebook's grouped split (StratifiedGroupKFold,
10 folds, seed 42, groups = domain::task_id, first fold = validation):

  length   a logistic regression on log(characters of the turns + candidate): is "long" a giveaway?
  source   a logistic regression on the data source alone: tau2, bfcl, critictool or induced (one-hot)

A trained model that does not clearly beat these on validation may be reading the data source or the length,
not the error. Mixing in outside data (CriticTool, induced runs) makes the source baseline the one to watch,
because those sources have their own label balance. Nothing is merged into the training files: --external only
adds rows to this script's in-memory pool so you can see the AUROC before deciding to merge.

    python eval/shortcut_baselines.py                      # the legacy 7,580 training rows (first-round dataset)
    python eval/shortcut_baselines.py --external data/derived/external_critictool.jsonl [more files]
    python eval/shortcut_baselines.py --induced            # artifact check of the induced rows + paired-positive counts (needs data/raw/induced/induced_runs.jsonl)
    python eval/shortcut_baselines.py --stateflip          # Class 7 state-flip pairs: do they differ only in the flipped value, and do length or source give them away? (D-021)
    python eval/shortcut_baselines.py --final [DIR]        # the final dataset (default data/derived/final_preview): source and length baselines with / without CriticTool and the state flips (D-023)
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent
DER = ROOT / "data" / "derived"
TRAIN_FILES = ["train_pool", "bfcl_pool", "positives_class2_controls", "positives_class4_controls", "positives_class6_controls", "negatives_pool_final"]
SOURCES = ["tau2", "bfcl", "critictool", "induced"]


def load_jsonl(p: Path) -> list[dict]:
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def source_of(r: dict) -> str:
    if str(r.get("source", "")).startswith("induced"):
        return "induced"
    if r.get("source") == "critictool" or r["domain"] == "critictool":
        return "critictool"
    return "bfcl" if r["domain"] == "bfcl" else "tau2"


def length_of(r: dict) -> float:
    return float(np.log1p(len(json.dumps(r["turns"])) + len(json.dumps(r["candidate_tool_call"]))))


def fit_auc(Xtr, ytr, Xva, yva) -> float:
    if len(set(yva)) < 2 or len(set(ytr)) < 2:
        return float("nan")
    clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    return float(roc_auc_score(yva, clf.predict_proba(Xva)[:, 1]))


def evaluate(rows: list[dict], label: str) -> None:
    y = np.array([r["label"] == "incorrect" for r in rows]).astype(int)
    groups = [f"{r['domain']}::{r['task_id']}" for r in rows]
    tr, va = next(iter(StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42).split(rows, y, groups)))
    src = np.array([[source_of(r) == s for s in SOURCES] for r in rows], dtype=float)
    ln = np.array([[length_of(r)] for r in rows])
    out = {"length": fit_auc(ln[tr], y[tr], ln[va], y[va]), "source": fit_auc(src[tr], y[tr], src[va], y[va]),
           "source+length": fit_auc(np.hstack([src, ln])[tr], y[tr], np.hstack([src, ln])[va], y[va])}
    print(f"\n{label}: {len(rows)} rows, train {len(tr)} / val {len(va)}, {y.mean():.1%} incorrect")
    for s in SOURCES:
        m = np.array([source_of(r) == s for r in rows])
        if m.any():
            print(f"   source {s:11s} rows {int(m.sum()):5d}  incorrect {y[m].mean():5.1%}  (val rows {int(m[va].sum())})")
    print(f"   validation AUROC   length {out['length']:.3f}   source {out['source']:.3f}   source+length {out['source+length']:.3f}")


# ───────────────────────── induced rows: artifact check (D-019 follow-up) ─────────────────────────
RAW_INDUCED = ROOT / "data" / "raw" / "induced" / "induced_runs.jsonl"
ISRC = ["induced_correction", "induced_fault", "induced_fault_soft", "induced_propagation"]


def _call_msgs(messages: list[dict]) -> list[int]:
    return [i for i, m in enumerate(messages) if m.get("role") == "assistant" for _ in (m.get("tool_calls") or [])]


def _ptype(plan: dict) -> str:
    if plan["source"] == "induced_correction":
        return plan["correction"]["type"]
    if plan["source"] == "induced_propagation":
        return plan["prop"]["type"]
    return f"{plan['fault'].get('mode', 'hard')}_{plan['fault']['kind']}"


def induced_rows() -> list[dict]:
    """One row per induced error candidate (incorrect) and per paired correct handling (correct), with artifact features computed
    from the run's history BEFORE the call (so a feature can only be something the model would see)."""
    runs = {}
    for line in open(RAW_INDUCED):
        r = json.loads(line)
        runs[r["run_id"]] = r
    out = []
    for fname, idkey, lab in (("induced_candidates_review.jsonl", "candidate_id", "incorrect"), ("induced_positive_candidates.jsonl", "positive_id", "correct")):
        for r in load_jsonl(DER / fname):
            run = runs[r["run_key"]]
            k = r["flagged_call"]["call_index"] if lab == "incorrect" else r["call_index"]
            msgs = run["messages"]
            hist = msgs[: _call_msgs(msgs)[k]]
            plan = run["plan"]
            tool_text = " ".join(str(m.get("content") or "") for m in hist if m.get("role") == "tool")
            user_text = " ".join(str(m.get("content") or "") for m in hist if m.get("role") == "user").lower()
            if plan["source"] == "induced_propagation":
                old = [plan["prop"]["old"].lower()]
            elif plan["source"] == "induced_correction":
                old = [x.lower() for x in plan["correction"]["old_markers"]]
            else:
                old = []
            call = r["flagged_call"] if lab == "incorrect" else {"name": r["name"], "arguments": r["arguments"]}
            out.append({"label": lab, "source": plan["source"], "ptype": _ptype(plan), "domain": plan["domain"], "split": plan["split"], "group": f"{plan['domain']}::{plan['task_id']}",
                        "f_previous_field": float('"previous_' in tool_text), "f_old_value_in_user_text": float(any(o in user_text for o in old)),
                        "f_not_found_in_history": float("not found" in tool_text.lower()),
                        "length": float(np.log1p(len(json.dumps(hist, default=str)) + len(json.dumps(call["arguments"], default=str))))})
    return out


def induced_checks() -> None:
    if not RAW_INDUCED.exists():
        print(f"{RAW_INDUCED} is missing (the run dump is gitignored); copy it back to run this check.")
        return
    rows = induced_rows()
    y = np.array([r["label"] == "incorrect" for r in rows]).astype(int)
    print(f"\n=== induced rows alone: {len(rows)} rows, {y.sum()} error candidates + {len(rows) - y.sum()} paired correct handlings ===")
    # ---- paired positives: counts by source, perturbation type, domain and split
    print("\npairing: error candidates (E) against correct handlings (C), by split")
    cnt = Counter((r["source"], r["ptype"], r["domain"], r["split"], r["label"]) for r in rows)
    keys = sorted({k[:4] for k in cnt})
    print(f"   {'source':20s} {'type':22s} {'domain':8s} {'split':5s}    E    C   C per E")
    for k in keys:
        e, c = cnt[k + ("incorrect",)], cnt[k + ("correct",)]
        flag = "  <-- NO correct handling" if e and not c else ("  <-- far fewer correct" if e and c / e < 0.5 else "")
        print(f"   {k[0]:20s} {k[1]:22s} {k[2]:8s} {k[3]:5s} {e:4d} {c:4d}   {c / e if e else float('nan'):5.1f}{flag}")
    for sp in ("test", "train"):
        e = sum(v for k, v in cnt.items() if k[3] == sp and k[4] == "incorrect")
        c = sum(v for k, v in cnt.items() if k[3] == sp and k[4] == "correct")
        print(f"   total {sp}: E {e}, C {c}")
    # ---- single artifact features
    feats = ["f_previous_field", "f_old_value_in_user_text", "f_not_found_in_history"]
    print("\nsingle artifact feature as a score (AUROC; 0.5 = no signal; under 0.5 means the feature points the other way)")
    def auc(x, yy):
        return float(roc_auc_score(yy, x)) if len(set(yy)) > 1 and len(set(x)) > 1 else 0.5
    over = []
    for name in feats + ["length"]:
        x = np.array([r[name] for r in rows])
        a = auc(x, y)
        print(f"   {name:28s} pooled {a:.3f}  (present in {int((x > 0).sum())} rows" + (")" if name != "length" else ")"))
        if name != "length" and max(a, 1 - a) > 0.65:
            over.append((name, "pooled", a))
        for src in ISRC:
            m = np.array([r["source"] == src for r in rows])
            if m.sum() and len(set(y[m])) > 1:
                a2 = auc(x[m], y[m])
                print(f"      within {src:20s} AUROC {a2:.3f}  (n {int(m.sum())}, E {int(y[m].sum())})")
                if name != "length" and max(a2, 1 - a2) > 0.65:
                    over.append((name, src, a2))
    srcm = np.array([[r["source"] == s_ for s_ in ISRC] for r in rows], dtype=float)
    # ---- classifiers, grouped by task, out-of-fold
    groups = [r["group"] for r in rows]
    def oof(X):
        pred = np.zeros(len(rows))
        for tr, va in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(X, y, groups):
            pred[va] = LogisticRegression(max_iter=1000).fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
        return float(roc_auc_score(y, pred))
    A = np.array([[r[f] for f in feats] for r in rows])
    L = np.array([[r["length"]] for r in rows])
    print("\nclassifiers on the induced rows alone (logistic regression, 5-fold grouped by task, out-of-fold AUROC)")
    print(f"   source label only                         {oof(srcm):.3f}")
    print(f"   the 3 artifact features only               {oof(A):.3f}")
    print(f"   3 artifact features + source label         {oof(np.hstack([A, srcm])):.3f}")
    print(f"   length only                                {oof(L):.3f}")
    print(f"   artifact + source + length                 {oof(np.hstack([A, srcm, L])):.3f}")
    for sp in ("test", "train"):
        m = np.array([r["split"] == sp for r in rows])
        print(f"   (rows in split {sp}: {int(m.sum())}, {int(y[m].sum())} errors)")
    print("\nfeatures above 0.65 (either direction):", over or "none")
    if over:
        # proposed fix, tested: (1) drop paired positives whose history already holds a 'not found' reply in the propagation runs (they come AFTER an
        # earlier stale attempt, the candidate is the first one), (2) match positives to candidates 1:1 inside every (source, type, domain, split) cell
        # that has candidates, so the source label and the perturbation type carry no class balance, (3) inside a cell match on 'old value in the user's text'.
        import random
        rng = random.Random(20261114)
        bal = [r for r in rows if r["label"] == "incorrect"]
        for k in keys:
            pool = [r for r in rows if r["label"] == "correct" and (r["source"], r["ptype"], r["domain"], r["split"]) == k
                    and not (r["source"] == "induced_propagation" and r["f_not_found_in_history"])]
            errs = [r for r in rows if r["label"] == "incorrect" and (r["source"], r["ptype"], r["domain"], r["split"]) == k]
            rng.shuffle(pool)
            for v in (0.0, 1.0):  # (3) also match on "the old value is in the user's own text": as many positives as candidates with the same value
                need = sum(r["f_old_value_in_user_text"] == v for r in errs)
                take = [r for r in pool if r["f_old_value_in_user_text"] == v][:need]
                bal += take
        yb = np.array([r["label"] == "incorrect" for r in bal]).astype(int)
        print(f"\nAFTER the proposed fix: {len(bal)} rows, {int(yb.sum())} errors, {len(bal) - int(yb.sum())} correct (1:1 per cell where enough positives exist)")
        for name in feats + ["length"]:
            x = np.array([r[name] for r in bal])
            print(f"   {name:28s} pooled AUROC {auc(x, yb):.3f}" + "".join(f"   {src.replace('induced_', '')} {auc(x[m], yb[m]):.3f}" for src in ISRC
                                                                       for m in [np.array([r['source'] == src for r in bal])] if m.sum() and len(set(yb[m])) > 1))
        g2 = [r["group"] for r in bal]
        S2 = np.array([[r["source"] == s_ for s_ in ISRC] for r in bal], dtype=float)
        A2 = np.array([[r[f] for f in feats] for r in bal])
        def oof2(X):
            pred = np.zeros(len(bal))
            for tr, va in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(X, yb, g2):
                pred[va] = LogisticRegression(max_iter=1000).fit(X[tr], yb[tr]).predict_proba(X[va])[:, 1]
            return float(roc_auc_score(yb, pred))
        print(f"   classifiers after the fix: source only {oof2(S2):.3f} | 3 artifact features {oof2(A2):.3f} | artifact + source {oof2(np.hstack([A2, S2])):.3f}")


# ───────────────────────── Class 7 state-flip pairs (D-021) ─────────────────────────

def json_diff(a, b, path="") -> list[str]:
    """Paths at which two parsed JSON values differ (a list of different length counts as one difference at its path)."""
    if type(a) is not type(b):
        return [path or "."]
    if isinstance(a, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}.{k}")
            else:
                out += json_diff(a[k], b[k], f"{path}.{k}")
        return out
    if isinstance(a, list):
        if len(a) != len(b):
            return [path or "."]
        return [d for x, y in zip(a, b) for d in json_diff(x, y, path)]
    return [] if a == b else [path or "."]


STATUS_WORDS = ("pending", "processed", "delivered", "cancelled")


def status_mismatch(r: dict) -> float:
    """1 if a user/agent message names a status for the order that differs from the last order lookup (retail only): the
    inconsistency an edited status can leave behind."""
    ts = r["turns"]
    last = None
    for t in ts:
        if t["role"] == "tool":
            m = re.search(r'"status": "(pending|processed|delivered|cancelled)"', str(t.get("content", "")))
            if m:
                last = m.group(1)
    if last is None:
        return 0.0
    for t in ts:
        if t["role"] in ("user", "assistant"):
            text = str(t.get("content", "")).lower()
            if any(re.search(rf"\b(is|was|currently|status)\W+(\w+\W+){{0,2}}{w}\b", text) for w in STATUS_WORDS if w != last):
                return 1.0
    return 0.0


def stateflip_checks() -> None:
    flips = load_jsonl(DER / "class7_stateflip.jsonl")
    pool = {r["trajectory_id"]: r for r in load_jsonl(DER / "train_pool.jsonl")}
    pairs = [(pool[f["paired_correct_id"]], f) for f in flips]
    print(f"=== Class 7 state-flip pairs: {len(flips)} negatives, each with its unedited positive in train_pool ===")

    # 1. the pairs differ only in the flipped value
    other, paths, bad = Counter(), Counter(), 0
    for o, f in pairs:
        for k in o:
            if k not in ("turns", "trajectory_id", "label", "label_tier", "split", "error_class", "error_class_name", "source", "origin") and o[k] != f[k]:
                other[k] += 1
        edited = [i for i, (a, b) in enumerate(zip(o["turns"], f["turns"])) if a != b]
        if len(o["turns"]) != len(f["turns"]) or edited != f["flip"]["edited_turns"]:
            bad += 1
            continue
        for i in edited:
            d = json_diff(json.loads(o["turns"][i]["content"]), json.loads(f["turns"][i]["content"]))
            paths[(f["policy_rule"]["id"], d[0].split(".")[1] if d and d[0].count(".") else "?", len(d))] += 1
    print(f"\nfields other than turns that differ within a pair: {dict(other) or 'none'}")
    print(f"pairs whose turns differ somewhere other than the recorded edited lookups: {bad}")
    print("edited lookups: (rule, top-level field, number of differing JSON values in that result) -> count")
    for k, n in sorted(paths.items()):
        print(f"   {k}: {n}")

    # 2. length
    d = np.array([len(json.dumps(f["turns"])) - len(json.dumps(o["turns"])) for o, f in pairs])
    print(f"\ncharacter length, negative minus its positive: mean {d.mean():+.1f}, min {d.min():+d}, max {d.max():+d}; "
          f"negative longer in {(d > 0).mean():.0%}, equal in {(d == 0).mean():.0%}, shorter in {(d < 0).mean():.0%}")
    by_rule = defaultdict(list)
    for (o, f), x in zip(pairs, d):
        by_rule[f["policy_rule"]["id"]].append(x)
    for k, v in sorted(by_rule.items()):
        print(f"   {k:36s} n {len(v):3d}  mean {np.mean(v):+6.1f}  range {min(v):+d}..{max(v):+d}")

    # 3. baselines: the pairs alone, then added to the current training rows
    rows = [r for o, f in pairs for r in (o, f)]
    evaluate(rows, "state-flip pairs alone (positives are the unedited originals)")
    train = [r for fn in TRAIN_FILES for r in load_jsonl(DER / f"{fn}.jsonl")]
    evaluate(train + flips, "current training rows + the state-flip negatives (in memory)")

    # 4. consistency artifacts (retail): an order whose status does not fit its fulfillments, and messages that name another status
    def inconsistent_order(r):
        for t in r["turns"]:
            if t["role"] == "tool":
                try:
                    o = json.loads(t.get("content", ""))
                except Exception:
                    continue
                if isinstance(o, dict) and o.get("status") in ("pending", "processed", "delivered", "cancelled") and "fulfillments" in o:
                    if (o["status"] == "pending") != (not o["fulfillments"]):
                        return 1.0
        return 0.0
    ret0 = [(o, f) for o, f in pairs if f["domain"] == "retail"]
    y0 = np.array([0] * len(ret0) + [1] * len(ret0))
    x0 = np.array([inconsistent_order(o) for o, _ in ret0] + [inconsistent_order(f) for _, f in ret0])
    print(f"\nretail pairs ({len(ret0)}): an order lookup whose status does not fit its fulfillments (pending iff none): "
          f"positives {x0[:len(ret0)].mean():.1%}, negatives {x0[len(ret0):].mean():.1%}; AUROC of that feature {roc_auc_score(y0, x0):.3f}")
    ret = [(o, f) for o, f in pairs if f["domain"] == "retail"]
    y = np.array([0] * len(ret) + [1] * len(ret))
    x = np.array([status_mismatch(o) for o, _ in ret] + [status_mismatch(f) for _, f in ret])
    print(f"\nretail pairs ({len(ret)}): a message names an order status different from the last lookup: "
          f"positives {x[:len(ret)].mean():.1%}, negatives {x[len(ret):].mean():.1%}; AUROC of that feature {roc_auc_score(y, x):.3f}")


# ───────────────────────── the final dataset: does the data source give the label away? (D-023) ─────────────────────────

FINE = ["tau2_gold", "bfcl_gold", "critictool", "synthetic_stateflip", "tau2_real_failed_run", "tau2_real_correct_call"]


def final_checks(directory: Path) -> None:
    D = {s: load_jsonl(directory / f"{s}.jsonl") for s in ("train", "val", "test", "test_synth")}
    print(f"final dataset in {directory}: " + ", ".join(f"{s} {len(v)}" for s, v in D.items()))
    configs = [("everything", lambda r: True), ("without CriticTool", lambda r: source_of(r) != "critictool"),
               ("without the state flips", lambda r: r.get("source") != "synthetic_stateflip"),
               ("without CriticTool and without the state flips", lambda r: source_of(r) != "critictool" and r.get("source") != "synthetic_stateflip")]
    fine_names = sorted({r["source"] for v in D.values() for r in v})
    feats = {"length": lambda r: [length_of(r)],
             "source (tau2 / bfcl / critictool / induced)": lambda r: [float(source_of(r) == x) for x in SOURCES],
             "source + length": lambda r: [float(source_of(r) == x) for x in SOURCES] + [length_of(r)],
             "fine source (the `source` column, informational)": lambda r: [float(r["source"] == x) for x in fine_names]}
    print("\nAUROC of a logistic regression fitted on the training split of each configuration (above 0.65 is flagged)")
    print(f"   {'configuration':50s} {'feature':50s} {'val':>6s} {'test':>6s} {'test_synth':>11s}")
    worst = []
    for cname, keep in configs:
        tr = [r for r in D["train"] if keep(r)]
        va = [r for r in D["val"] if keep(r)]
        ytr = np.array([r["label"] == "incorrect" for r in tr]).astype(int)
        inc = ytr.mean()
        for fname, f in feats.items():
            clf = LogisticRegression(max_iter=2000).fit(np.array([f(r) for r in tr]), ytr)
            out = []
            for part in (va, D["test"], D["test_synth"]):
                y = np.array([r["label"] == "incorrect" for r in part]).astype(int)
                out.append(roc_auc_score(y, clf.predict_proba(np.array([f(r) for r in part]))[:, 1]) if y.min() != y.max() else float("nan"))
            flag = "  <-- above 0.65" if out[0] > 0.65 and fname.startswith(("source (", "source +")) else ""
            print(f"   {cname:50s} {fname:50s} {out[0]:6.3f} {out[1]:6.3f} {out[2]:11.3f}{flag}")
            if fname.startswith("source ("):
                worst.append((cname, out[0]))
        print(f"   {'':50s} (train rows {len(tr)}, incorrect {inc:.1%}; val rows {len(va)})")
    print("\nsource-only (coarse) validation AUROC by configuration:", {c: round(a, 3) for c, a in worst})
    tr = D["train"]
    print("\nincorrect share by source in the training split:")
    for x in sorted({source_of(r) for r in tr}):
        m = [r for r in tr if source_of(r) == x]
        print(f"   {x:11s} rows {len(m):5d}  incorrect {np.mean([r['label'] == 'incorrect' for r in m]):.1%}")
    print("\ncorrect rows per source (train), the proposed balance target:")
    print("   ", dict(Counter(source_of(r) for r in tr if r["label"] == "correct")), "| incorrect:", dict(Counter(source_of(r) for r in tr if r["label"] == "incorrect")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--external", nargs="*", default=[], help="extra jsonl files to add to the pool (in memory only)")
    ap.add_argument("--induced", action="store_true", help="artifact check of the induced rows + paired-positive counts")
    ap.add_argument("--final", nargs="?", const=str(DER / "final_preview"), default=None, help="source and length baselines on the final dataset (default: the dry-run preview)")
    ap.add_argument("--stateflip", action="store_true", help="Class 7 state-flip pairs: differ only in the flipped value? length and source baselines")
    args = ap.parse_args()
    if args.induced:
        return induced_checks()
    if args.stateflip:
        return stateflip_checks()
    if args.final:
        return final_checks(Path(args.final))
    rows = [r for f in TRAIN_FILES for r in load_jsonl(DER / f"{f}.jsonl")]
    evaluate(rows, "current training rows")
    test = load_jsonl(DER / "test_real.jsonl")
    ty = np.array([r["label"] == "incorrect" for r in test]).astype(int)
    tl = np.array([[length_of(r)] for r in test])
    tr_len = np.array([[length_of(r)] for r in rows])
    ytr = np.array([r["label"] == "incorrect" for r in rows]).astype(int)
    print(f"test_real ({len(test)} rows, all tau2): source baseline is a constant, AUROC 0.500; "
          f"length baseline trained on all training rows scores AUROC {fit_auc(tr_len, ytr, tl, ty):.3f}")
    if args.external:
        extra = [r for f in args.external for r in load_jsonl(Path(f))]
        print("\nadding (in memory only):", dict(Counter(source_of(r) for r in extra)))
        evaluate(rows + extra, "training rows + external")


if __name__ == "__main__":
    main()
