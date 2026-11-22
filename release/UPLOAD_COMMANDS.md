# Upload commands (run these yourself; nothing here has been run)

The `hf` command is the current Hugging Face CLI (the one in `huggingface_hub` 2.4.0, checked with `hf --help`); the older `huggingface-cli` command was replaced by it, and if you have an old version installed the equivalents are `huggingface-cli login`, `huggingface-cli repo create NAME --type model` and `huggingface-cli upload REPO_ID LOCAL_PATH PATH_IN_REPO --repo-type model`.

## 0. Before anything: check the folders

The full sequence (GitHub first, then Hugging Face) is in [`docs/release_steps.md`](../docs/release_steps.md); this file has the Hugging Face commands only. The names (`ShaliniSrivastava`, `Shalini-Srivastavaaa`) are filled in.

```bash
cd /home/user/Projects/toolcall-verifier
pip install -U huggingface_hub
ls -la release/hf_model release/test_set
shasum -a 256 release/hf_model/model.safetensors      # must be 617accea7753c55ff139c71abc14b283b7bd96ff3a9ba9042159f2e10d004c24
```

## 1. Log in

```bash
.venv/bin/hf auth login            # opens a browser or asks for a token with write access from https://huggingface.co/settings/tokens
.venv/bin/hf auth whoami
```

## 2. The model repository (weights, tokenizer, card, `predict.py`)

```bash
.venv/bin/hf repos create ShaliniSrivastava/toolcall-verifier-modernbert-v3b --type model --public
.venv/bin/hf upload ShaliniSrivastava/toolcall-verifier-modernbert-v3b release/hf_model . --repo-type model --commit-message "ModernBERT v3b tool-call verifier (Apache-2.0)"
```

Add `--private` to the first command (and drop `--public`) if you want to look at the page before it is public; you can switch visibility in the repository settings.

## 3. The test set as a dataset repository (optional; the files are also in the code repository)

```bash
.venv/bin/hf repos create ShaliniSrivastava/toolcall-verifier-test-set --type dataset --public
.venv/bin/hf upload ShaliniSrivastava/toolcall-verifier-test-set release/test_set . --repo-type dataset --commit-message "Human-reviewed tool-call test set (CC BY 4.0 labels, MIT tau2-bench content)"
```

`release/test_set/` is self-contained (README with the card header, `DATA_LICENSE.md`, `licenses/`, `pii_check.txt`, the two JSONL files).

## 4. The Space (NOT DONE: needs HF PRO, 402 on create; demo; upload it after the model repository, because the app downloads the model from the Hub)

```bash
.venv/bin/hf repos create ShaliniSrivastava/toolcall-verifier-demo --type space --sdk gradio --public
.venv/bin/hf spaces variables add ShaliniSrivastava/toolcall-verifier-demo -e MODEL_REPO=ShaliniSrivastava/toolcall-verifier-modernbert-v3b
.venv/bin/hf upload ShaliniSrivastava/toolcall-verifier-demo release/space . --repo-type space --commit-message "Gradio demo of the tool-call verifier"
.venv/bin/hf spaces logs ShaliniSrivastava/toolcall-verifier-demo        # build and run logs; the first start downloads the 600 MB model
```

The free CPU hardware is the default for a new Space (`.venv/bin/hf spaces hardware` lists the options). The first request after a sleep pays the model load. If the model repository is private, the Space also needs a read token as a secret (`.venv/bin/hf spaces secrets ...`) and `snapshot_download(..., token=...)`; with a public model repository nothing more is needed. To test the app on your machine first: `pip install gradio torch transformers huggingface_hub` and `MODEL_DIR=models/final_v3b python release/space/app.py`.

## 5. After uploading

Open the model page, the dataset page and the Space and read each once as a stranger would; check that the links in the model card point to your code repository (they were generated from `--repo-url`), that the Space's limitations panel is visible, and that `NOTICE` has the copyright holder filled in. Nothing in this folder pushes the code repository: that is `scripts/make_release.sh`, then `git remote add` and `git push` from `../toolcall-verifier-release/`, which you do yourself.
