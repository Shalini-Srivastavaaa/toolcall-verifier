---
title: Tool-call verifier demo
emoji: 🛡️
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.32.0
python_version: "3.12"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Would this agent tool call be wrong? A ModernBERT verifier.
---

# Tool-call verifier demo

**A ~150M-parameter model that runs on a laptop CPU in about 0.7 s (GPT-5-mini: 3.9 s), at no API cost per call and with conversations that never leave your machine, matches or exceeds GPT-5-mini at catching real agent mistakes (AUROC 0.771 vs 0.719) and significantly beats GPT-5-nano (0.619) and the length shortcut (0.675).** It scores a call in about 0.7 s on a laptop CPU against 3.9 s for GPT-5-mini, with no API cost and conversations that never leave your machine.

Scores whether a customer-service agent's proposed tool call is incorrect, before it runs, with the ModernBERT v3b verifier (149M parameters, Apache-2.0). It shows twelve real, human-labeled test calls (tau2-bench retail, airline, telecom), the score, the decision at the 10% false-alarm threshold, the policy text the model saw, and the human label; the conversation and the call can be edited.

**It is a research demo with real limits, listed at the top of the app:** the examples are curated (6 catches, 3 misses, 3 correct calls), the model catches about one error in three at this threshold, the threshold is set on the same 363 test rows that chose the model, the test set is small, and scores react to phrasing.

- Model: the Hub repository named in `MODEL_REPO` (default `ShaliniSrivastava/toolcall-verifier-modernbert-v3b`; set the variable in the Space settings or edit the default in `app.py`).
- Files: `app.py`, `predict.py` and `data/` (the inference code and the policy text, identical to the code repository), `examples.json` (the 12 rows, built by `scripts/build_space.py` from the public test set).
- License: Apache-2.0 for the code; the example conversations are from tau2-bench (MIT, Sierra Research) with our labels (CC BY 4.0); see `DATA_LICENSE.md`, `LICENSE` and `NOTICE`.
- Run locally: `pip install -r requirements.txt gradio==6.32.0` then `python app.py` (set `MODEL_DIR=/path/to/model` to use local weights).
