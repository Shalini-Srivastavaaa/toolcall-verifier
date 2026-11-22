"""Check that a downloaded model (models/final_v2/) loads with transformers on CPU and reproduces its saved test predictions on a few rows (D-027).

    <python with torch and transformers> eval/verify_model_locally.py [--model models/final_v2] [--pred results/final_v2/test_predictions.jsonl] [--n 6]

The inputs are built with the notebook's own formatting cells (kaggle_phase5_modernbert.ipynb, policy block on, as in the run), the tokenizer is the one saved with the model
(tokenizer.json) and, for comparison, the base model's tokenizer; the rows are the shortest test inputs so that CPU inference is quick. The saved predictions come from a GPU run
with fp16 autocast, so a difference of about 0.01 is expected.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_visibility import load_notebook_namespace  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "final_v2"))
    ap.add_argument("--pred", default=str(ROOT / "results" / "final_v2" / "test_predictions.jsonl"))
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--offset", type=int, default=0, help="start this many rows into the length-sorted list (0 = the shortest inputs)")
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizerFast

    tmp = Path(tempfile.mkdtemp())
    (tmp / "in" / "toolcall-verifier-data").mkdir(parents=True)
    for f in (ROOT / "data" / "kaggle_upload").glob("final_*.jsonl"):
        (tmp / "in" / "toolcall-verifier-data" / f.name).symlink_to(f)
    (tmp / "work").mkdir()
    ns = load_notebook_namespace("None", tmp / "in", tmp / "work")
    test = ns["FINAL"]["test"]
    saved = {json.loads(l)["trajectory_id"]: json.loads(l)["p_incorrect"] for l in open(a.pred)}
    texts = [ns["format_input_v2"](r) for r in test]
    order = sorted(range(len(test)), key=lambda i: len(texts[i]))
    pick = [i for i in order if test[i]["trajectory_id"] in saved][a.offset : a.offset + a.n]
    mdir = Path(a.model)
    print(f"model folder {mdir}: {sorted(p.name for p in mdir.iterdir())}")
    tok_saved = PreTrainedTokenizerFast(tokenizer_file=str(mdir / "tokenizer.json"), pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]", unk_token="[UNK]")
    tok_base = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    same = all(tok_saved(texts[i], truncation=True, max_length=4096)["input_ids"] == tok_base(texts[i], truncation=True, max_length=4096)["input_ids"] for i in pick)
    print(f"saved tokenizer.json gives the same ids as the base tokenizer on the {len(pick)} rows: {same}")
    model = AutoModelForSequenceClassification.from_pretrained(str(mdir)).eval()
    print("loaded", type(model).__name__, f"{sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters, labels {model.config.id2label}")
    diffs = []
    for i in pick:
        enc = tok_saved(texts[i], truncation=True, max_length=4096, return_tensors="pt")
        with torch.no_grad():
            p = float(torch.softmax(model(**enc).logits.float(), dim=-1)[0, 1])
        s = saved[test[i]["trajectory_id"]]
        diffs.append(abs(p - s))
        print(f"   {test[i]['label']:9s} class {test[i].get('error_class')} tokens {enc['input_ids'].shape[1]:5d}: local {p:.4f}  saved {s:.4f}  diff {abs(p - s):.4f}")
    print(f"max difference {max(diffs):.4f}, mean {np.mean(diffs):.4f}")


if __name__ == "__main__":
    main()
