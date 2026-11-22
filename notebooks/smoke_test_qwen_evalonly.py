"""Local test of the Qwen notebook's evaluate-only path with a saved adapter (D-029): a tiny random Qwen2 model stands in for Qwen2.5-1.5B (CPU, the notebook's CPU evaluation path;
the 4-bit GPU path cannot be tested here).

    .venv/bin/python notebooks/smoke_test_qwen_evalonly.py [notebook.ipynb]        # needs torch, transformers, peft, datasets, scikit-learn, the Qwen tokenizer (downloaded), data/kaggle_upload_v3b/

What it does, each step in its own process (as Kaggle runs a notebook):
 A. TRAIN: the notebook's cells 0-6 with the SMOKE toy model (random weights, a few rows) train an adapter and evaluate it; the toy BASE (without the `score` head) and the
    tokenizer are saved as a local "base model" folder, and the adapter folder `best/` is copied to where Kaggle puts an attached notebook output:
    <input>/notebooks/<user>/<notebook>/checkpoints/phase6/best/ (its train_info.json says completed).
 B. EVAL ONLY, auto-detect: EVAL_ONLY = True, EVAL_CHECKPOINT = None, MODEL_PATH_OVERRIDE = the tiny base, TCV_SMOKE off: the notebook must find best/ in the attached input, load the
    base with AutoModelForSequenceClassification.from_pretrained (a checkpoint without the `score` head: transformers prints the LOAD REPORT with a MISSING key, the call that crashed
    through the Tee), put the adapter and its trained head on top with PeftModel.from_pretrained, and write predictions. Checks: the validation, test and test_synth probabilities equal
    those of step A (same adapter, same base) to 1e-5, so the head really came from the adapter (the fresh head of the loaded base is random and differs); the report file exists.
 C. EVAL ONLY with an explicit EVAL_CHECKPOINT: same result.
 D. TRAINING NOT SKIPPED WITHOUT AN ATTACHED RUN / SKIPPED WITH ONE: EVAL_ONLY = False with the toy model: with the attached notebook output restore_prior_run finds best/ and no training
    starts ("found a finished model from an earlier run", no "training with"); with nothing attached it trains.
 F. CONTROL: the same adapter with its `score` head replaced gives different predictions (peft raises a KeyError for an adapter without it) (the check above can fail).
 E. EVAL ONLY with nothing attached: stops with the explanatory assertion, not a crash later.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve()


def child(nb_path, cells, overrides, post):
    """Run in a child process: execute the notebook's code cells, then the post step."""
    nb = json.load(open(nb_path))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    ns = {"__name__": "__main__"}
    for idx in cells:
        src = "".join(code[idx]["source"])
        if idx == 0:
            for old, new in overrides:
                assert old in src, f"override not found in the configuration cell: {old!r}"
                src = src.replace(old, new.replace("\\n", "\n"), 1)
        src = "\n".join(("# " + l if l.lstrip().startswith(("!", "%")) else l) for l in src.split("\n"))
        print(f"######## code cell {idx}", flush=True)
        exec(compile(src, f"cell{idx}", "exec"), ns)
    if post == "export_base":  # the toy base of the SMOKE run (same seed, same weights) without the score head, plus the tokenizer
        base = Path(os.environ["TINY_BASE"])
        m = ns["load_base_model"]()
        m.model.save_pretrained(base)
        ns["tokenizer"].save_pretrained(base)
        print("exported the tiny base model to", base)


def run(label, nb_path, cells, overrides, env, expect_ok=True):
    root = Path(env["R"])
    e = dict(os.environ, TCV_ENV="kaggle", TCV_KAGGLE_INPUT=str(root / "in"), **{k: v for k, v in env.items() if k != "R"})
    e.pop("TCV_SMOKE", None) if env.get("TCV_SMOKE") == "" else None
    cmd = [sys.executable, str(HERE), "--child", str(nb_path), ",".join(map(str, cells)), json.dumps(overrides), env.get("POST", "")]
    r = subprocess.run(cmd, env=e, capture_output=True, text=True)
    out = r.stdout + r.stderr
    (root / f"log_{label}.txt").write_text(out)
    print(f"[{label}] exit {r.returncode}", flush=True)
    if expect_ok:
        assert r.returncode == 0, out[-3000:]
    return r.returncode, out


def main():
    nb_path = Path(next((a for a in sys.argv[1:] if a.endswith(".ipynb")), ROOT / "notebooks" / "kaggle_phase6_qwen.ipynb"))
    R = Path(tempfile.mkdtemp(prefix="qwen_evalonly_"))
    data = R / "in" / "toolcall-verifier-data-v3b"
    data.mkdir(parents=True)
    for f in (ROOT / "data" / "kaggle_upload_v3b").glob("final_*.jsonl"):
        (data / f.name).symlink_to(f.resolve())
    nopip = ["def pip_install(*packages):=>def pip_install(*packages):\n    return"]
    common = {"R": str(R), "TINY_BASE": str(R / "tinybase")}
    # A. train + evaluate with the toy model, export the base, publish best/ as an attached notebook output
    A = dict(common, TCV_SMOKE="1", TCV_KAGGLE_WORKING=str(R / "work_a"), POST="export_base")
    (R / "work_a").mkdir()
    run("A_train", nb_path, [0, 1, 2, 3, 4, 5, 6], [o.split("=>") for o in nopip], A)
    best_a = R / "work_a" / "checkpoints" / "phase6" / "best"
    assert (best_a / "adapter_config.json").exists() and json.load(open(best_a / "train_info.json"))["completed"] is True
    from safetensors import safe_open
    keys = list(safe_open(str(best_a / "adapter_model.safetensors"), "pt").keys())
    assert any("score" in k for k in keys), f"the adapter does not carry the classification head: {keys[:6]}"
    print("adapter keys include the head:", [k for k in keys if "score" in k][:2], "| LoRA keys:", len(keys) - len([k for k in keys if "score" in k]))
    attached = R / "in" / "notebooks" / "someuser" / "qwen-v3b" / "checkpoints" / "phase6" / "best"
    shutil.copytree(best_a, attached)
    ref = {n: [json.loads(l) for l in open(R / "work_a" / "checkpoints" / "phase6" / n)] for n in ("val_predictions.jsonl", "test_predictions.jsonl", "test_synth_predictions.jsonl")}

    def prob(row):
        return row["p_incorrect"]

    def check_equal(workdir, label):
        for n, rows_a in ref.items():
            rows_b = [json.loads(l) for l in open(workdir / "checkpoints" / "phase6" / n)]
            assert len(rows_a) == len(rows_b), (n, len(rows_a), len(rows_b))
            assert [a["trajectory_id"] for a in rows_a] == [b["trajectory_id"] for b in rows_b]
            d = max(abs(prob(a) - prob(b)) for a, b in zip(rows_a, rows_b))
            print(f"   {label}: {n}: {len(rows_b)} rows, max |probability difference| to the training run's evaluation = {d:.2e}")
            assert d < 1e-5, f"{label} {n}: predictions differ ({d})"

    # B. eval only, auto-detect
    base_flags = [o.split("=>") for o in nopip] + [["EVAL_ONLY = False", "EVAL_ONLY = True"], ["MODEL_PATH_OVERRIDE = None", f"MODEL_PATH_OVERRIDE = '{R / 'tinybase'}'"]]
    B = dict(common, TCV_SMOKE="", TCV_KAGGLE_WORKING=str(R / "work_b"))
    (R / "work_b").mkdir()
    rc, out = run("B_eval_auto", nb_path, [0, 1, 2, 3, 4, 5, 6], base_flags, B)
    assert "found a finished model from an earlier run" in out and str(attached) in out, "auto-detect did not find the attached best/"
    assert "LOAD REPORT" in out and "MISSING" in out, "the loading report with the missing head was not exercised"
    assert "Traceback" not in out
    assert "EVAL_ONLY: training data is not tokenized" in out and "training with" not in out
    assert (R / "work_b" / "eval_report.txt").exists() and "evaluation report | environment kaggle | checkpoint " + str(attached) in (R / "work_b" / "eval_report.txt").read_text()
    check_equal(R / "work_b", "B (auto-detect)")
    # C. eval only, explicit EVAL_CHECKPOINT
    C = dict(common, TCV_SMOKE="", TCV_KAGGLE_WORKING=str(R / "work_c"))
    (R / "work_c").mkdir()
    run("C_eval_explicit", nb_path, [0, 1, 2, 3, 4, 5, 6], base_flags + [["EVAL_CHECKPOINT = None", f"EVAL_CHECKPOINT = '{attached}'"]], C)
    check_equal(R / "work_c", "C (explicit EVAL_CHECKPOINT)")
    # F. negative control: the same adapter with a DIFFERENT head must give different predictions (so that the equality above really shows that the head comes from the adapter)
    from safetensors.torch import load_file, save_file
    ctrl = R / "ctrl_best"
    shutil.copytree(attached, ctrl)
    import torch
    tensors = load_file(str(ctrl / "adapter_model.safetensors"))
    g = torch.Generator().manual_seed(1)
    for k in tensors:
        if "score" in k:
            tensors[k] = torch.randn(tensors[k].shape, generator=g, dtype=tensors[k].dtype)  # a different head (peft refuses an adapter without it: KeyError, checked separately)
    save_file(tensors, str(ctrl / "adapter_model.safetensors"))
    F = dict(common, TCV_SMOKE="", TCV_KAGGLE_WORKING=str(R / "work_f"))
    (R / "work_f").mkdir()
    run("F_control_no_head", nb_path, [0, 1, 2, 3, 4, 5, 6], base_flags + [["EVAL_CHECKPOINT = None", f"EVAL_CHECKPOINT = '{ctrl}'"]], F)
    rows_b = [json.loads(l) for l in open(R / "work_f" / "checkpoints" / "phase6" / "val_predictions.jsonl")]
    d = max(abs(a["p_incorrect"] - b["p_incorrect"]) for a, b in zip(ref["val_predictions.jsonl"], rows_b))
    print(f"   F (control, adapter with another head): max |probability difference| = {d:.3f} (must be large)")
    assert d > 1e-3, "the control did not change the predictions: the equality check cannot tell whether the head is loaded"
    # D. training is skipped when a finished run is attached, and happens when none is
    D = dict(common, TCV_SMOKE="1", TCV_KAGGLE_WORKING=str(R / "work_d"))
    (R / "work_d").mkdir()
    rc, out = run("D_skip_training", nb_path, [0, 1, 2, 3, 4, 5], [o.split("=>") for o in nopip], D)
    assert "found a finished model from an earlier run" in out and "training with" not in out, "training was not skipped although a finished run is attached"
    assert not (R / "work_d" / "checkpoints" / "phase6" / "trainer").exists()
    print("   D: with the notebook output attached, training is skipped (no 'training with' line, no trainer folder)")
    # E. eval only, nothing attached
    shutil.rmtree(R / "in" / "notebooks")
    E = dict(common, TCV_SMOKE="", TCV_KAGGLE_WORKING=str(R / "work_e"))
    (R / "work_e").mkdir()
    rc, out = run("E_nothing_attached", nb_path, [0, 1, 2, 3, 4, 5, 6], base_flags, E, expect_ok=False)
    assert rc != 0 and "nothing to evaluate" in out and "attach the notebook that trained it" in out, out[-1500:]
    print("   E: nothing attached -> stops with the explanatory message")
    print("\nALL CHECKS PASSED; logs in", R)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(sys.argv[2], [int(x) for x in sys.argv[3].split(",")], json.loads(sys.argv[4]), sys.argv[5] if len(sys.argv) > 5 else "")
    else:
        main()
