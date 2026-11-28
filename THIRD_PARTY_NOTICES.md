# Third-party notices

What this repository uses from other projects, the license of each and how it is used. The full license texts are in [`licenses/`](licenses/). A summary of the terms, including points to check before reuse, is in [`docs/licenses.md`](docs/licenses.md).

## tau2-bench (MIT)

- **Source:** https://github.com/sierra-research/tau2-bench, "Copyright (c) 2025 Sierra Research". License text: [`licenses/tau2-bench-MIT.txt`](licenses/tau2-bench-MIT.txt).
- **What is used:** the retail, airline and telecom domain definitions (tool schemas, policy documents, task definitions) and the recorded simulations. The 363-row test set in `release/test_set/` contains conversation text from τ²-bench simulations, and `verifier/data/` contains the policy text and tool-schema text of its domains. The copyright notice and the permission notice of the MIT License are kept with those files (see `DATA_LICENSE.md`).
- **Note:** the conversations in the recorded simulations were produced by language models (agent and simulated user) run by the benchmark's authors; see `docs/licenses.md` section 3.2.

## Berkeley Function-Calling Leaderboard / Gorilla (Apache-2.0)

- **Source:** https://github.com/ShishirPatil/gorilla (`berkeley-function-call-leaderboard`), data at https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard. License text: [`licenses/BFCL-Gorilla-Apache-2.0.txt`](licenses/BFCL-Gorilla-Apache-2.0.txt) (the repository's `LICENSE`, whose appendix leaves the copyright line as a template).
- **What is used:** multi-turn function-calling tasks (v4) as extra training data and as the source of part of the synthetic test set (`test_synth`: 198 of its 374 rows, domain `bfcl`). These rows are modified for this project (synthetic corruptions and paired controls). The training data is not published; the `test_synth` rows with BFCL content are, under Apache-2.0 for that content.

## ModernBERT-base (Apache-2.0)

- **Source:** https://huggingface.co/answerdotai/ModernBERT-base; "ModernBERT is a collaboration between Answer.AI, LightOn, and friends" (model card). The model card declares `license: apache-2.0`; it does not state a copyright line. License text: [`licenses/Apache-2.0.txt`](licenses/Apache-2.0.txt).
- **What is used:** the base weights, fine-tuned into `toolcall-verifier-modernbert-v3b`. **This is a modified version of the base model.**
- **Citation requested by the authors:** Warner et al., "Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference", arXiv:2412.13663.

## Qwen2.5-1.5B (Apache-2.0)

- **Source:** https://huggingface.co/Qwen/Qwen2.5-1.5B; the repository's LICENSE: "Copyright 2024 Alibaba Cloud", Apache License 2.0. License text: [`licenses/Qwen2.5-1.5B-Apache-2.0.txt`](licenses/Qwen2.5-1.5B-Apache-2.0.txt).
- **What is used:** a comparison model fine-tuned with LoRA (results in `docs/decision_log.md` D-029 and D-030). **No Qwen weights or adapters are published.** Listed because the writeup and the notebooks refer to it.

## Training data used and not published

The training data (not published) also contains rows from CriticTool (`chocckaka/CriticTool-Dataset`, a Hugging Face card with `license: apache-2.0`; see `docs/licenses.md` 3.1), and text written by language models (gpt-5-nano, gpt-5-mini, gpt-4.1-mini and the agents of the benchmark runs). The published weights were trained on it. This is recorded for transparency; nothing from CriticTool is redistributed.

## Creative Commons BY 4.0

The text of CC BY 4.0, which applies to the human labels (`DATA_LICENSE.md`), is in [`licenses/CC-BY-4.0.txt`](licenses/CC-BY-4.0.txt).
