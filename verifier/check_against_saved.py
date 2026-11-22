"""Check `verifier/predict.py` against the training notebook and the saved predictions (D-031).

    python -m verifier.check_against_saved [--model-dir models/final_v3b] [--n-score 12]

1. TEXT: for all 363 test rows the input text built by `verifier.predict.format_input` must equal, character for character, what the notebook's `format_input_v2` builds
   (the notebook cells are run through eval/diagnose_visibility.load_notebook_namespace on data/kaggle_upload_v3b; needs numpy, scikit-learn and transformers).
2. SCORES: a few test rows (the shortest ones and some long ones, across classes) are scored with `score()` on this machine's CPU and compared with the probabilities saved by the
   Kaggle run (GPU, fp16). Differences of a few thousandths are expected from fp16 against fp32 (D-029: at most 0.006 over all 363 rows).
"""

import argparse
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(ROOT / "models" / "final_v3b"))
    ap.add_argument("--n-score", type=int, default=12)
    a = ap.parse_args()
    from diagnose_visibility import load_notebook_namespace
    from verifier import predict

    tmp = Path(tempfile.mkdtemp())
    (tmp / "in" / "toolcall-verifier-data").mkdir(parents=True)
    for f in (ROOT / "data" / "kaggle_upload_v3b").glob("final_*.jsonl"):
        (tmp / "in" / "toolcall-verifier-data" / f.name).symlink_to(f.resolve())
    (tmp / "work").mkdir()
    with contextlib.redirect_stdout(io.StringIO()):
        ns = load_notebook_namespace("None", tmp / "in", tmp / "work")
    test = ns["FINAL"]["test"]
    saved = {json.loads(l)["trajectory_id"]: json.loads(l)["p_incorrect"] for l in open(ROOT / "results" / "final_v3b" / "test_predictions.jsonl")}
    v = predict.Verifier(a.model_dir)

    bad = 0
    for r in test:
        mine = predict.format_input(r, v.n_tokens)
        theirs = ns["format_input_v2"](r)
        if mine != theirs:
            bad += 1
    print(f"1. TEXT: {len(test) - bad} of {len(test)} test rows give exactly the notebook's input text ({bad} differ)")
    assert bad == 0

    order = sorted(range(len(test)), key=lambda i: v.n_tokens(predict.format_input(test[i], v.n_tokens)))
    short, long_ = order[: a.n_score // 2], order[-(a.n_score // 2):]
    diffs = []
    print("2. SCORES: row, class, tokens, saved GPU probability, probability here, difference")
    for i in short + long_:
        r = test[i]
        d = v.score(r["turns"], r["candidate_tool_call"], r["tool_schemas_text"], r["domain"], "fa10")
        diffs.append(abs(d.probability - saved[r["trajectory_id"]]))
        print(f"   {r['domain']:8s} class {str(r.get('error_class')):4s} {d.tokens:5d} tokens  {saved[r['trajectory_id']]:.4f}  {d.probability:.4f}  {diffs[-1]:.4f}")
    # the same through the public interface with the bundled schema text and the domain only
    r = test[short[0]]
    d2 = predict.score(r["turns"], r["candidate_tool_call"], None, r["domain"], verifier=v) if r["domain"] in predict.TAU2_SCHEMAS else None
    print(f"   max difference over {len(diffs)} rows: {max(diffs):.4f} (mean {sum(diffs) / len(diffs):.4f})")
    assert max(diffs) < 0.01
    if d2:
        print(f"   with tool_schemas=None (bundled text): probability {d2.probability:.4f}, flagged {d2.flagged} at fa10 (threshold {d2.threshold:.3f})")
    print("OK")


if __name__ == "__main__":
    main()
