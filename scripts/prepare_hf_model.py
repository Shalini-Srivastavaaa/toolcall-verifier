"""Prepare the Hugging Face model repository folder `release/hf_model/` (D-032). Nothing is uploaded.

    python3 scripts/prepare_hf_model.py [--repo-url https://github.com/OWNER/REPO] [--hf-user HF_USERNAME] [--model-dir models/final_v3b]

Copies the v3b weights and tokenizer (config.json, model.safetensors, tokenizer.json, tokenizer_config.json, train_info.json), the model card as README.md (relative links turned
into links to the code repository, the usage example changed to the Hub layout), `predict.py` with its `data/` folder (policy text, tool schemas, thresholds), LICENSE and NOTICE.
The weights are not tracked by git (release/hf_model/model.safetensors is in .gitignore and in the release deny list); everything else in the folder is.
"""

import argparse
import hashlib
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "release" / "hf_model"
EXPECTED_SHA = "617accea7753c55ff139c71abc14b283b7bd96ff3a9ba9042159f2e10d004c24"   # D-029 (models/final_v3b/model.safetensors)

HF_USAGE = '''```python
from huggingface_hub import snapshot_download
import sys
path = snapshot_download("HF_USERNAME/toolcall-verifier-modernbert-v3b")   # downloads this repository, including predict.py and data/
sys.path.insert(0, path)
from predict import load, score
load(path)
v = score(conversation, {"name": "cancel_pending_order", "arguments": {...}}, tool_schemas=None, domain="retail", operating_point="fa10")
v.probability, v.flagged, v.input_text      # P(incorrect), above the threshold?, the exact text the model read
```

(or clone this repository and `import predict` from it; it needs `torch` and `transformers`)'''


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-url", default="https://github.com/OWNER/REPO", help="the public code repository, used for the links in the card")
    ap.add_argument("--hf-user", default="HF_USERNAME", help="your Hugging Face user name, used in the usage example of the card")
    ap.add_argument("--model-dir", default=str(ROOT / "models" / "final_v3b"))
    a = ap.parse_args()
    src = Path(a.model_dir)
    assert sha256(src / "model.safetensors") == EXPECTED_SHA, "models/final_v3b/model.safetensors is not the v3b file recorded in D-029"
    OUT.mkdir(parents=True, exist_ok=True)
    for f in ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json", "train_info.json"):
        shutil.copy2(src / f, OUT / f)
    shutil.copy2(ROOT / "verifier" / "predict.py", OUT / "predict.py")
    shutil.rmtree(OUT / "data", ignore_errors=True)
    shutil.copytree(ROOT / "verifier" / "data", OUT / "data")
    for f in ("LICENSE", "NOTICE"):
        shutil.copy2(ROOT / f, OUT / f)
    card = (ROOT / "docs" / "model_card.md").read_text()
    repo = a.repo_url.rstrip("/")
    card = re.sub(r"\]\(\.\./([^)]+)\)", lambda m: f"]({repo}/blob/main/{m.group(1)})", card)
    card = re.sub(r"\]\((writeup|licenses|labeling_guide|decision_log)\.md\)", lambda m: f"]({repo}/blob/main/docs/{m.group(1)}.md)", card)
    i = card.index("```python\nfrom verifier.predict import load, score")
    j = card.index("```", i + 5) + 3
    card = card[:i] + HF_USAGE.replace("HF_USERNAME", a.hf_user) + card[j:]
    card = card.replace("Use `verifier/predict.py`, which contains", "Use `predict.py` of this repository (`verifier/predict.py` in the code repository), which contains")
    card = card.replace("`NOTICE` and `THIRD_PARTY_NOTICES.md`", f"`NOTICE` here and [`THIRD_PARTY_NOTICES.md`]({repo}/blob/main/THIRD_PARTY_NOTICES.md) in the code repository")
    card = card.replace("(`DATA_LICENSE.md`)", f"([`DATA_LICENSE.md`]({repo}/blob/main/DATA_LICENSE.md))")
    files = "\n## Files in this repository\n\n`model.safetensors` (598 MB, SHA-256 `%s`), `config.json`, `tokenizer.json`, `tokenizer_config.json`, `train_info.json` (training settings and the validation history), `predict.py` and `data/` (policy text of the 18 write tools, tool-schema text of the three τ²-bench domains, operating-point thresholds), `LICENSE`, `NOTICE`.\n" % EXPECTED_SHA
    k = card.index("## Intended use")
    card = card[:k] + files.lstrip("\n") + "\n" + card[k:]
    (OUT / "README.md").write_text(card)
    left = [m for m in re.findall(r"\]\((?!http)([^)#]+)\)", card) if not m.startswith("mailto")]
    print("written", OUT, "| relative links left in the card:", left)
    for f in sorted(OUT.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(OUT)}  {f.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
