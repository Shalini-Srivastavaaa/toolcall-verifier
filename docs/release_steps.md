# Release steps (run these yourself)

Nothing in this repository pushes or uploads anything. Do the steps in this order: build the clean release copy, push it to GitHub, upload the model to Hugging Face, create the Space. Each step ends with what to check. Replace `Shalini-Srivastavaaa` (your GitHub user or organization), `ShaliniSrivastava` (your Hugging Face user or organization) and the name in `--holder`.

## 0. Placeholders (done)

The copyright holder (`NOTICE`: Shalini Srivastava), the GitHub owner (`Shalini-Srivastavaaa`), the Hugging Face user (`ShaliniSrivastava`), the developer and the citation are filled in everywhere (D-032). If you ever change a name, `git grep` the old one and edit it by hand (the fill script only replaces the original placeholders), then regenerate the Hugging Face card:

```bash
cd /home/user/Projects/toolcall-verifier
python3 scripts/prepare_hf_model.py --repo-url https://github.com/Shalini-Srivastavaaa/toolcall-verifier --hf-user ShaliniSrivastava
git grep -n -E 'OWNER/REPO|COPYRIGHT HOLDER: fill|TODO' -- . ':!docs/decision_log.md' ':!docs/archive/' ':!docs/doc_audit.md' ':!docs/release_steps.md' ':!scripts/' ':!*.jsonl' ':!release/hf_model/tokenizer.json'    # prints nothing
```

## 1. Build the clean release copy and scan it

`scripts/make_release.sh` exports the last commit to `../toolcall-verifier-release/` with a **new git history (one commit)**, removes the training data, raw data, weights, run dumps, review files, raw judge outputs and anything over 5 MB, replaces your identifiers from the local file `.release-scrub`, and runs gitleaks.

```bash
cd /home/user/Projects/toolcall-verifier
# gitleaks: install it (for example from https://github.com/gitleaks/gitleaks/releases) and point GITLEAKS at the binary, or skip the scan and run your own
RELEASE_AUTHOR_NAME="YOUR NAME" RELEASE_AUTHOR_EMAIL="YOUR PUBLIC OR NOREPLY EMAIL" GITLEAKS=/path/to/gitleaks scripts/make_release.sh --force
```

Without the two `RELEASE_AUTHOR_*` variables the single commit gets the author `toolcall-verifier <noreply@example.invalid>`. For a GitHub noreply address use the one from GitHub > Settings > Emails.

**Check, in `../toolcall-verifier-release/`:**
- `git log --oneline` shows exactly one commit.
- `git ls-files | grep -E 'safetensors|derived|data/raw|kaggle_upload|\.env$|models/'` prints nothing; no file over 5 MB (`find . -path ./.git -prune -o -type f -size +5M -print`).
- `grep -rIl -i 'YOUR_LOCAL_USERNAME' . --exclude-dir=.git` prints nothing (your local and Kaggle user names; edit `.release-scrub` if you want more patterns removed).
- gitleaks reports 0 findings (the allowlist in `.gitleaks.toml` covers the BFCL placeholder tokens in `release/test_set/test_synth.jsonl`).
- Open `README.md`, `LICENSE`, `NOTICE`, `DATA_LICENSE.md`, `release/test_set/README.md` once, as a stranger would.
- Optional: in a clean virtual environment, `pip install torch transformers`, copy `models/final_v3b` next to the release copy and run `python -m verifier.check_against_saved` (it needs the training data folders, which are absent: use the README quick-start example instead, expected probability about 0.02).

## 2. GitHub: create the repository and push

1. On github.com: **New repository**, owner `Shalini-Srivastavaaa`, name `toolcall-verifier`, **public**, and **do not** add a README, license or .gitignore (the release copy has them).
2. Push the release copy:

   ```bash
   cd /home/user/Projects/toolcall-verifier-release
   git remote add origin git@github.com:Shalini-Srivastavaaa/toolcall-verifier.git      # or https://github.com/Shalini-Srivastavaaa/toolcall-verifier.git
   git push -u origin main
   ```

**Check:**
- The repository page shows the README with the results table; the license badge in the sidebar says **Apache-2.0** (GitHub detects `LICENSE`).
- `release/test_set/` has `test.jsonl`, `test_synth.jsonl`, `README.md`, `DATA_LICENSE.md`; the JSONL files display or download (each under 5 MB).
- **Settings > Code security**: secret scanning shows no alerts (enable "secret scanning" and "push protection" if they are off).
- Clone it fresh into a temporary folder and follow the README quick start once the model is on Hugging Face (step 3).
- Add the repository description and topics (`tool-use`, `agents`, `verification`, `modernbert`).

## 3. Hugging Face: upload the model

Prerequisite: `pip install -U huggingface_hub` and `.venv/bin/hf auth login` (a token with write access). The CLI is `hf` (the old `huggingface-cli` no longer exists in current versions). The same commands are in `release/UPLOAD_COMMANDS.md`.

```bash
cd /home/user/Projects/toolcall-verifier
shasum -a 256 release/hf_model/model.safetensors        # must print 617accea7753c55ff139c71abc14b283b7bd96ff3a9ba9042159f2e10d004c24
.venv/bin/hf repos create ShaliniSrivastava/toolcall-verifier-modernbert-v3b --type model --public
.venv/bin/hf upload ShaliniSrivastava/toolcall-verifier-modernbert-v3b release/hf_model . --repo-type model --commit-message "ModernBERT v3b tool-call verifier (Apache-2.0)"
```

(Use `--private` first if you want to look at the page before it is public.) Optional, the test set as a dataset:

```bash
.venv/bin/hf repos create ShaliniSrivastava/toolcall-verifier-test-set --type dataset --public
.venv/bin/hf upload ShaliniSrivastava/toolcall-verifier-test-set release/test_set . --repo-type dataset --commit-message "Human-reviewed tool-call test set (CC BY 4.0 labels, MIT tau2-bench content)"
```

**Check:**
- The model page shows the card, **License: apache-2.0**, the files (`model.safetensors` 598 MB, `predict.py`, `data/`), and the links in the card open your GitHub repository.
- Download and score from a clean folder:

  ```bash
  python3 -m venv /tmp/hfcheck && /tmp/hfcheck/bin/pip install -q torch transformers huggingface_hub
  /tmp/hfcheck/bin/python - <<'PY'
  import sys
  from huggingface_hub import snapshot_download
  p = snapshot_download("ShaliniSrivastava/toolcall-verifier-modernbert-v3b"); sys.path.insert(0, p)
  from predict import load, score
  load(p)
  t = [{"role": "user", "content": "Hi, I want to cancel order #W2378156, I ordered it by mistake."}, {"role": "assistant_tool_call", "name": "get_order_details", "arguments": {"order_id": "#W2378156"}}, {"role": "tool", "content": "{\"order_id\": \"#W2378156\", \"status\": \"pending\", \"items\": []}"}, {"role": "assistant", "content": "The order is pending, so I can cancel it. Please confirm the order id and the reason (ordered by mistake)."}, {"role": "user", "content": "Yes, #W2378156, ordered by mistake. Please cancel it."}]
  print(score(t, {"name": "cancel_pending_order", "arguments": {"order_id": "#W2378156", "reason": "ordered by mistake"}}, domain="retail").probability)   # about 0.02
  PY
  ```

- Compare `shasum -a 256` of the downloaded `model.safetensors` with the value above.

## 4. Hugging Face: create the Space (NOT DONE: free Gradio hosting needs a Hugging Face PRO plan, the create call returns 402; run the demo locally instead, see the README)

The Space downloads the model from the Hub, so do it after step 3.

```bash
cd /home/user/Projects/toolcall-verifier
.venv/bin/hf repos create ShaliniSrivastava/toolcall-verifier-demo --type space --sdk gradio --public
.venv/bin/hf spaces variables add ShaliniSrivastava/toolcall-verifier-demo -e MODEL_REPO=ShaliniSrivastava/toolcall-verifier-modernbert-v3b
.venv/bin/hf upload ShaliniSrivastava/toolcall-verifier-demo release/space . --repo-type space --commit-message "Gradio demo of the tool-call verifier"
.venv/bin/hf spaces logs ShaliniSrivastava/toolcall-verifier-demo
```

**Check:**
- The build finishes ("Running" in the Space header); the first start downloads the 600 MB model and can take a few minutes. If it fails, read the logs: the usual causes are the `requirements.txt` (CPU torch from the PyTorch index) and the `sdk_version` (6.32.0) in the Space `README.md`.
- The page opens with the **"Known limitations" panel visible**; pick an example and press **Score**: it should show FLAG or ALLOW, the policy text and the human label. Example 1 (telecom `refuel_data`) should show a score near 0.025 and the label "incorrect, Class 7", which is a miss.
- **Time a click on the free CPU hardware** (I measured 0.4 s median on a 10-core Mac only); if it takes more than about 10 s, say so on the Space page.
- Edit one conversation line and score again: the page must say the edited input has no human label.
- If the model repository is private, the Space needs a read token as a secret (`.venv/bin/hf spaces secrets`); with a public model nothing more is needed.

## 5. Link everything and tag

1. In the GitHub README add the links to the model page, the dataset page and the Space (one line near the top).
2. In the Hugging Face model card (`docs/model_card.md`, regenerate with `scripts/prepare_hf_model.py`) the GitHub links are already set; add the Space link if you want it.
3. Tag the release in the release copy: `git tag v1.0 && git push origin v1.0`, and on GitHub create a release from the tag.
4. Re-read `docs/licenses.md` section 3 (the terms for model-generated text are still not reviewed) and decide whether the README should say more about it than it does.

## If something goes wrong

- **A secret or personal data turns up after publishing:** make the repository private, remove it, rebuild with `scripts/make_release.sh --force` and force-push to a new repository; rotating a key is the first step if it was a key.
- **A Hugging Face upload is interrupted:** re-run the same `.venv/bin/hf upload` command; it resumes.
- **The Space does not start:** `.venv/bin/hf spaces logs`, then fix the file locally in `release/space/` and upload again.
