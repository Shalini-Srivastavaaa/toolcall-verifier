"""Ablation of the 'induced run' signature on the final_v2 model (D-027), on CPU with the downloaded checkpoint.

The induced runs were made with gpt-5-nano as the agent, and its text messages are JSON objects (`{ "response": "..." }`); the tau2 agents of the natural runs write plain text.
That style is visible in the model input (the AGENT lines). Two tests, scored with the real model and the notebook's own input formatting:
  A  induced test rows with each JSON agent message replaced by its plain-text value   -> do the scores of induced rows fall?
  B  natural test rows (correct, and incorrect) with their plain agent messages wrapped as {"response": "..."}   -> do their scores rise?
  A2 induced test rows with every agent message unwrapped AND cut to 200 characters (the D-027 flag NORMALIZE_AGENT_TEXT, applied to a model trained without it)
  C  natural test rows with every agent message cut to 200 characters (the control for A2)

    <python with torch and transformers> eval/ablate_induced_signature.py [--max-tokens 2600] [--n-natural 40]

Rows longer than --max-tokens are skipped (CPU time); the saved GPU predictions are the baseline. Output: results/final_v2/ablation_induced_signature.jsonl and a summary.
"""

import argparse
import copy
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_visibility import load_notebook_namespace  # noqa: E402


def unwrap(text: str) -> str:
    try:
        obj = json.loads(text)
    except ValueError:
        return text.strip().lstrip("{").rstrip("}").strip()
    if isinstance(obj, dict):
        for k in ("response", "text", "message"):
            if isinstance(obj.get(k), str):
                return obj[k]
        vals = [v for v in obj.values() if isinstance(v, str)]
        if vals:
            return max(vals, key=len)
    return text


def wrap(text: str) -> str:
    return json.dumps({"response": text}, indent=2, ensure_ascii=False)


def cap200(text: str) -> str:
    s = unwrap(text)
    return s if len(s) <= 200 else s[:199] + "…"


def transform(row, fn, only_json):
    """only_json True: only JSON-object agent messages; False: only plain ones; None: every agent message."""
    r = copy.deepcopy(row)
    n = 0
    for t in r["turns"]:
        if t.get("role") == "assistant" and t.get("content") and (only_json is None or t["content"].lstrip().startswith("{") == only_json):
            t["content"] = fn(t["content"])
            n += 1
    return r, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "final_v2"))
    ap.add_argument("--max-tokens", type=int, default=2600)
    ap.add_argument("--n-natural", type=int, default=40)
    ap.add_argument("--normalize-agent-text", action="store_true", help="score with NORMALIZE_AGENT_TEXT = True, for a model trained with that flag (D-027)")
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForSequenceClassification, PreTrainedTokenizerFast

    tmp = Path(tempfile.mkdtemp())
    (tmp / "in" / "toolcall-verifier-data").mkdir(parents=True)
    for f in (ROOT / "data" / "kaggle_upload").glob("final_*.jsonl"):
        (tmp / "in" / "toolcall-verifier-data" / f.name).symlink_to(f)
    (tmp / "work").mkdir()
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        ns = load_notebook_namespace("None", tmp / "in", tmp / "work", {"NORMALIZE_AGENT_TEXT = False": "NORMALIZE_AGENT_TEXT = True"} if a.normalize_agent_text else None)
    test = ns["FINAL"]["test"]
    saved = {json.loads(l)["trajectory_id"]: json.loads(l)["p_incorrect"] for l in open(ROOT / "results" / "final_v2" / "test_predictions.jsonl")}
    tok = PreTrainedTokenizerFast(tokenizer_file=str(Path(a.model) / "tokenizer.json"), pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]", unk_token="[UNK]")
    model = AutoModelForSequenceClassification.from_pretrained(a.model).eval()

    def score(row):
        enc = tok(ns["format_input_v2"](row), truncation=True, max_length=4096, return_tensors="pt")
        with torch.no_grad():
            return float(torch.softmax(model(**enc).logits.float(), dim=-1)[0, 1]), int(enc["input_ids"].shape[1])

    jobs = []
    for r in test:
        if r["origin"] == "induced":
            r2, n = transform(r, unwrap, True)
            if n:
                jobs.append(("A_induced_unwrapped", r, r2, n))
    nat = [r for r in test if r["origin"] == "natural" and r["domain"] != "telecom"]
    nat.sort(key=lambda r: len(json.dumps(r["turns"])))
    short = [r for r in nat if r["label"] == "correct"][: a.n_natural // 2] + [r for r in nat if r["label"] == "incorrect"][: a.n_natural // 2]
    for r in short:
        r2, n = transform(r, wrap, False)
        if n:
            jobs.append(("B_natural_wrapped", r, r2, n))
    for r in test:
        if r["origin"] == "induced":
            r2, n = transform(r, cap200, None)
            if n:
                jobs.append(("A2_induced_unwrapped_capped", r, r2, n))
    import random
    nat_all = [r for r in test if r["origin"] == "natural" and len(json.dumps(r["turns"])) < 9000]
    random.Random(0).shuffle(nat_all)
    for r in nat_all[:60]:
        r2, n = transform(r, cap200, None)
        if n:
            jobs.append(("C_natural_capped", r, r2, n))
    print(f"{len(jobs)} rows to score", flush=True)
    out = open(ROOT / "results" / "final_v2" / "ablation_induced_signature.jsonl", "w")
    t0 = time.time()
    for i, (kind, r, r2, n) in enumerate(jobs):
        p, ntok = score(r2)
        if ntok >= a.max_tokens:
            continue
        rec = {"test": kind, "trajectory_id": r["trajectory_id"], "label": r["label"], "error_class": r.get("error_class"), "origin": r["origin"], "domain": r["domain"],
               "n_messages_changed": n, "tokens": ntok, "saved_score": saved[r["trajectory_id"]], "ablated_score": p}
        out.write(json.dumps(rec) + "\n")
        out.flush()
        if i % 10 == 0:
            print(f"   {i}/{len(jobs)} done, {time.time() - t0:.0f}s", flush=True)
    out.close()
    rows = [json.loads(l) for l in open(ROOT / "results" / "final_v2" / "ablation_induced_signature.jsonl")]
    for kind in ("A_induced_unwrapped", "A2_induced_unwrapped_capped", "B_natural_wrapped", "C_natural_capped"):
        for lab in ("correct", "incorrect"):
            rs = [r for r in rows if r["test"] == kind and r["label"] == lab]
            if rs:
                s0, s1 = np.array([r["saved_score"] for r in rs]), np.array([r["ablated_score"] for r in rs])
                print(f"{kind:28s} {lab:9s} n {len(rs):3d}: median score {np.median(s0):.2f} -> {np.median(s1):.2f}; share >= 0.5: {(s0 >= 0.5).mean():.2f} -> {(s1 >= 0.5).mean():.2f}; mean change {np.mean(s1 - s0):+.3f}")


if __name__ == "__main__":
    main()
