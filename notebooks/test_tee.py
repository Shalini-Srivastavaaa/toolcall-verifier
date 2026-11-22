"""Regression test for the `Tee` of the Kaggle notebooks (D-029): the first Qwen evaluation crashed with "'Tee' object has no attribute 'isatty'" because transformers' loading
report calls `sys.stdout.isatty()` while `sys.stdout` was a Tee. The test takes the Tee class out of each notebook, makes it `sys.stdout` and runs the code path that crashed:
loading a Qwen2ForSequenceClassification from a checkpoint without the `score` head (transformers prints a LOAD REPORT with a MISSING key, which calls `_style()` ->
`sys.stdout.isatty()`), plus `_style()` directly and the other file-like methods.

    .venv/bin/python notebooks/test_tee.py [notebook.ipynb ...]      # default: both Kaggle notebooks; needs torch and transformers (a tiny random model is created, nothing is downloaded)
    .venv/bin/python notebooks/test_tee.py --old-from-git            # negative control: the Tee of the last commit before the fix must FAIL this test
"""

import io
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def tee_source(nb_json):
    for c in nb_json["cells"]:
        s = "".join(c["source"])
        if c["cell_type"] == "code" and "class Tee:" in s:
            i = s.index("class Tee:")
            j = s.index("\n\n\n", i) if "\n\n\n" in s[i:] else len(s)
            return s[i:j]
    return None


def run_checks(Tee, label):
    import torch  # noqa: F401
    from transformers import Qwen2Config, Qwen2ForSequenceClassification, Qwen2Model
    from transformers.utils import loading_report

    tmp = Path(tempfile.mkdtemp())
    real = sys.stdout
    tee = Tee(tmp / "report.txt")
    try:
        assert sys.stdout is tee
        # 1. the code path that crashed: a Qwen2 checkpoint WITHOUT the score head loaded as a sequence classifier
        cfg = Qwen2Config(vocab_size=100, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, pad_token_id=0)
        base = tmp / "base"
        Qwen2Model(cfg).save_pretrained(base)
        model = Qwen2ForSequenceClassification.from_pretrained(base, num_labels=2, pad_token_id=0)
        assert model.score.weight.shape == (2, 16)
        # 2. file-like surface
        assert tee.isatty() is False
        assert tee.writable() is True and tee.readable() is False and tee.seekable() is False
        assert isinstance(tee.encoding, str) and isinstance(tee.errors, str)
        try:
            fd = tee.fileno()
            assert isinstance(fd, int)
        except io.UnsupportedOperation:
            pass  # the real stdout has no descriptor either (captured output): the same behaviour
        print("print() through the tee", file=sys.stdout)
        tee.writelines(["a\n", "b\n"])
        h = logging.StreamHandler(sys.stdout)
        logging.getLogger("tee_test").addHandler(h)
        logging.getLogger("tee_test").warning("logging through a StreamHandler on the tee")
        _ = tee.line_buffering if hasattr(real, "line_buffering") else None  # delegated through __getattr__
        try:
            tee.no_such_attribute
            raise SystemExit("an unknown attribute must raise AttributeError")
        except AttributeError:
            pass
        # 3. the function that crashed
        if hasattr(loading_report, "_style"):
            assert loading_report._style("x", "red") == "x"
        tee.flush()
    finally:
        logging.getLogger("tee_test").removeHandler(h)
        tee.close()
    assert sys.stdout is real
    text = (tmp / "report.txt").read_text()
    assert "print() through the tee" in text and "logging through a StreamHandler" in text
    assert "\x1b[" not in text, "colour codes reached the report file"
    print(f"{label}: OK ({len(text)} characters in the report file)")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--old-from-git" in sys.argv:
        out = subprocess.run(["git", "show", "HEAD:notebooks/kaggle_phase6_qwen.ipynb"], capture_output=True, text=True, cwd=ROOT, check=True).stdout
        sources = {"Tee of the last commit (before the fix)": tee_source(json.loads(out))}
    else:
        paths = args or [str(ROOT / "notebooks" / "kaggle_phase5_modernbert.ipynb"), str(ROOT / "notebooks" / "kaggle_phase6_qwen.ipynb")]
        sources = {Path(p).name: tee_source(json.load(open(p))) for p in paths}
    for label, src in sources.items():
        assert src, f"no Tee class in {label}"
        ns = {"sys": sys, "Path": Path, "__name__": "tee_test"}
        exec(compile(src, f"Tee of {label}", "exec"), ns)
        run_checks(ns["Tee"], label)
    colab = [p for p in ("phase5_modernbert.ipynb", "phase6_qwen.ipynb") if tee_source(json.load(open(ROOT / "notebooks" / p)))]
    assert not colab, f"the Colab notebooks do not use a Tee; found one in {colab}"


if __name__ == "__main__":
    main()
