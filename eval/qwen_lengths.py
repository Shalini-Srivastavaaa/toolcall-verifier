"""Token lengths of the v2 input under Qwen's tokenizer, for the Qwen notebook's data cells on a packaged dataset (D-029): fits the budget QWEN_MAX_LEN, how often the ladder tightens,
tokens per epoch (with REAL_WEIGHT) for the time estimate. No GPU, no model; needs transformers (the tokenizer is downloaded), numpy, scikit-learn.

    .venv/bin/python3 eval/qwen_lengths.py --data data/kaggle_upload_v3b --budget 4096 --budget 2048
"""

import argparse
import contextlib
import io
import json
import os
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def run(data: Path, budget: int):
    tmp = Path(tempfile.mkdtemp())
    (tmp / "in" / "toolcall-verifier-data").mkdir(parents=True)
    for f in data.glob("final_*.jsonl"):
        (tmp / "in" / "toolcall-verifier-data" / f.name).symlink_to(f.resolve())
    (tmp / "work").mkdir()
    os.environ.update(TCV_ENV="kaggle", TCV_KAGGLE_INPUT=str(tmp / "in"), TCV_KAGGLE_WORKING=str(tmp / "work"), TOKENIZERS_PARALLELISM="false")
    nb = json.load(open(ROOT / "notebooks" / "kaggle_phase6_qwen.ipynb"))
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    ns = {"__name__": "__main__"}
    with contextlib.redirect_stdout(io.StringIO()):
        for idx in (0, 2, 3, 4):
            s = "".join(cells[idx]["source"])
            if idx == 0:
                assert "QWEN_MAX_LEN = 4096" in s
                s = s.replace("QWEN_MAX_LEN = 4096", f"QWEN_MAX_LEN = {budget}", 1)
            s = "\n".join(("# " + l if l.lstrip().startswith(("!", "%")) else l) for l in s.split("\n"))
            exec(compile(s, f"cell{idx}", "exec"), ns)
    return ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "kaggle_upload_v3b"))
    ap.add_argument("--budget", type=int, action="append", default=[])
    ap.add_argument("--real-weight", type=int, default=3)
    a = ap.parse_args()
    for budget in a.budget or [4096, 2048]:
        ns = run(Path(a.data), budget)
        F = ns["FINAL"]
        print("=" * 100 + f"\nQwen tokenizer, budget {budget} tokens (v2 input, policy block {'on' if ns['INCLUDE_POLICY'] else 'off'})\n" + "=" * 100)
        tot = {}
        for sp in ("train", "val", "test", "test_synth"):
            ms = [ns["format_input_v2"](r, meta=True)[1] for r in F[sp]]
            nat = np.array([m["natural_tokens"] for m in ms])
            tok = np.array([m["tokens"] for m in ms])
            sh = [m["shown"] / m["total"] for m in ms if m["total"]]
            tot[sp] = (tok, nat, ms)
            print(f"{sp:10s} rows {len(ms):5d} | tokens p50 {int(np.percentile(tok, 50)):5d} p90 {int(np.percentile(tok, 90)):5d} p99 {int(np.percentile(tok, 99)):5d} max {int(tok.max()):5d} mean {tok.mean():6.0f} | "
                  f"natural length over {budget}: {int((nat > budget).sum()):4d} ({(nat > budget).mean():.1%}) | still over after the ladder {sum(m['over_budget'] for m in ms)} | tool-result chars visible (pooled) {sum(m['shown'] for m in ms) / max(sum(m['total'] for m in ms), 1):.1%}")
        tr = F["train"]
        w = np.array([a.real_weight if r.get("real_origin") else 1 for r in tr])
        tok = tot["train"][0]
        print(f"training tokens per epoch: {tok.sum() / 1e6:.2f} M for {len(tr)} rows; with REAL_WEIGHT = {a.real_weight}: {(tok * w).sum() / 1e6:.2f} M for {int(w.sum())} rows; "
              f"validation {tot['val'][0].sum() / 1e6:.2f} M, test {tot['test'][0].sum() / 1e6:.2f} M, test_synth {tot['test_synth'][0].sum() / 1e6:.2f} M tokens")
        out = ROOT / "results" / "final_v3b" / f"qwen_train_tokens_{budget}.json"
        out.write_text(json.dumps({"budget": budget, "tokens": [int(x) for x in tok], "real_origin": [bool(r.get("real_origin")) for r in tr],
                                   "val_tokens": [int(x) for x in tot["val"][0]], "test_tokens": [int(x) for x in tot["test"][0]], "test_synth_tokens": [int(x) for x in tot["test_synth"][0]]}))
        print("saved", out)


if __name__ == "__main__":
    main()
