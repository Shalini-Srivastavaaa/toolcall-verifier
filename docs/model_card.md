---
# Hugging Face model card. License decided in D-032: Apache-2.0 for the weights (base model: Apache-2.0).
license: apache-2.0
language:
- en
library_name: transformers
base_model: answerdotai/ModernBERT-base
pipeline_tag: text-classification
tags:
- tool-use
- agents
- verification
- modernbert
- text-classification
model-index:
- name: toolcall-verifier-modernbert-v3b
  results:
  - task:
      type: text-classification
      name: Pre-execution verification of a tool call (incorrect vs correct)
    dataset:
      name: Human-reviewed real test set (363 rows, tau2-bench retail, airline, telecom)
      type: custom
    metrics:
    - type: roc_auc
      value: 0.771
      name: AUROC (95% bootstrap CI 0.720-0.818)
---

# Tool-call verifier: ModernBERT v3b

**Links:** [code and results on GitHub](https://github.com/Shalini-Srivastavaaa/toolcall-verifier) · [human-labeled test set (dataset)](https://huggingface.co/datasets/ShaliniSrivastava/toolcall-verifier-test-set)

A 149M-parameter classifier that reads a conversation between a user and a customer-service agent plus **one proposed tool call** and returns the probability that the call is **incorrect**, before the call is executed. It is `answerdotai/ModernBERT-base` fine-tuned on the data described below ([decision log](decision_log.md) D-022 to D-029; project writeup in [`docs/writeup.md`](writeup.md)).

**Summary.** A ~150M-parameter model that runs on a laptop CPU in about 0.7 s (GPT-5-mini: 3.9 s), at no API cost per call and with conversations that never leave your machine, matches or exceeds GPT-5-mini at catching real agent mistakes (AUROC 0.771 vs 0.719) and significantly beats GPT-5-nano (0.619) and the length shortcut (0.675). On the 363 human-reviewed real test calls the model has AUROC 0.771 (0.720-0.818); at the setting that flags 10% of correct calls it catches 33% of the errors, at the setting that flags 5% it catches 23%. It is a triage signal, not a gate that stops most errors.

## Model details

| | |
|---|---|
| Base model | `answerdotai/ModernBERT-base` (Apache-2.0 per its model card, verified 2026-11-22) |
| Architecture | `ModernBertForSequenceClassification`, 22 layers, hidden size 768, `[CLS]` pooling, 2 labels (`0` correct, `1` incorrect) |
| Context used | up to 4,096 tokens (the model supports 8,192) |
| Files | `model.safetensors` (598,439,784 bytes, SHA-256 `617accea7753c55ff139c71abc14b283b7bd96ff3a9ba9042159f2e10d004c24`), `config.json`, `tokenizer.json`, `tokenizer_config.json` (checksums in D-029) |
| Training run | Kaggle, one T4, fp16, 3 epochs, learning rate 2e-5, effective batch 16 (micro-batch 2, 8 accumulation steps), seed 42, 1.9 hours (6,780 s); the checkpoint is the best epoch by validation F1 |
| License | **Apache-2.0** (the weights are a modified version of ModernBERT-base, also Apache-2.0; see `NOTICE` and `THIRD_PARTY_NOTICES.md`). The human labels of the published test set are CC BY 4.0 (`DATA_LICENSE.md`). |
| Developed by | Shalini Srivastava |

## Intended use

**Intended:** a pre-execution check on tool calls proposed by an LLM agent in **customer-service workflows of the kind in τ²-bench (retail orders, airline reservations, telecom lines)**: score the call, and route high scores to a human reviewer, a second model or a retry with the model's reasons in mind. It detects seven kinds of mistakes: wrong argument value, stale state (a value the user has since changed), wrong tool, repeated call, failure to propagate a tool result, ignored tool error, and policy violation (D-024, [`docs/labeling_guide.md`](labeling_guide.md)).

**Out of scope:**
- Automatically blocking or approving calls with real consequences without a human in the loop. At the strict setting it misses two errors in three (recall 0.33 at 10% false alarms).
- Other domains, other tools or other languages. All training and test data are English τ²-bench customer-service conversations plus BFCL function-calling data; behaviour elsewhere is untested.
- Explaining *why* a call is wrong. The model outputs a score, not a reason; the sub-reason tags of Class 7 are labeling aids, not outputs.
- Security use (detecting malicious or injected calls); it was not trained or tested for that.
- Treating the score as a calibrated probability. Scores are not calibrated; the thresholds below are rank-based settings.

## Input format and how to call it

The model was trained and evaluated on a text built by `format_input_v2` with the policy block (D-017, D-021). The same text must be built at inference time. Use `verifier/predict.py`, which contains that code and has been checked to give **character-for-character the training input for all 363 test rows** (D-031):

```python
from verifier.predict import load, score
load("models/final_v3b")
v = score(conversation, {"name": "cancel_pending_order", "arguments": {...}}, tool_schemas=None, domain="retail", operating_point="fa10")
v.probability, v.flagged, v.input_text      # P(incorrect), above the threshold?, the exact text the model read
```

`conversation` is a list of turns `{"role": "user" | "assistant" | "assistant_tool_call" | "tool", ...}` (an `assistant_tool_call` has `name` and `arguments`); `turns_from_openai` converts OpenAI-style messages. The text has this shape (one block per line):

```
TOOLS: <tools called in the last 5 turns plus the candidate's tool, with parameter names only>   (a "# Requires:" prerequisite line for the candidate's tool in the tau2 domains)
POLICY: <the policy text for the candidate's tool; only for the 18 write tools of the three tau2 domains, otherwise absent>
EARLIER_CALLS: <calls that fall before the window, most recent last>
HISTORY:
USER: ... / AGENT: ... / AGENT_CALL: tool(arg=value, ...) / TOOL_RESULT: ...      (up to the last 30 turns)
CANDIDATE: tool(arg=value, ...)          <- always last
```

Rules that matter: a tool result is shown in full if it contains any string value (3+ characters) of the candidate's arguments, and the two most recent results are always shown in full; other results are cut to 200 characters and messages to 500; if the text exceeds 4,096 tokens it is tightened step by step (older results, then messages, then fewer turns), and the candidate is never cut. Operating points: `fa05` (threshold 0.895, flags 4.7% of correct test calls, recall 0.229), `fa10` (0.731, 9.8%, recall 0.329) and `default` (0.5, 19.7%, recall 0.494). **They were set on the test set**, so on other traffic the share of correct calls flagged will differ. A probe of the model on hand-made examples showed that the score reacts strongly to phrasing and to the format of IDs (see Failure modes).

## Training data

Built by `data/build_final_dataset.py` (D-022, D-026, D-028): **7,154 training rows** (2,570 errors, 4,584 correct) and 749 validation rows, grouped by task so no task is in both; the real-origin rows (real errors and their paired correct calls, 1,597 rows) are shown three times per epoch (`REAL_WEIGHT = 3`, 10,348 rows per epoch). Composition of the errors: **220 real errors** from τ²-bench failed runs (the first incorrect call, proposed by gpt-5-mini under rules, no per-row human decision) and synthetic or external negatives for the rest: τ²-bench corruptions made by an LLM or by code with **paired controls** for Classes 2, 4 and 6, BFCL multi-turn data, CriticTool tool-error examples (Class 6) and policy "state flips" for Class 7. Correct rows: gold-matched calls, earlier calls of failed runs (screened by the labeling rules), controls, and added correct calls of the tools that were over-represented as errors (tool balancing, D-028). **No row of the test set or of its task groups is in training** (checked by ID, task and conversation text).

**The training data is not published.** Sources and their licenses are listed in [`docs/licenses.md`](licenses.md) and `THIRD_PARTY_NOTICES.md` (the training data includes CriticTool rows, Apache-2.0 per its Hugging Face card; none is redistributed). Some of the data (τ²-bench transcripts, induced runs, LLM-written negatives and labels) is text generated by commercial models. The author reviewed the providers' terms (outputs are owned by the user under the OpenAI and Anthropic terms; no use to build competing models); this is the author's reading, not legal advice, and anyone with a strict compliance need should read that part of `docs/licenses.md`.

## Evaluation

All on the **363-row human-reviewed test set** (170 errors, 193 correct; 98 of the errors natural, 72 induced; airline 116, retail 166, telecom 81), never used for training. Intervals are 95% bootstrap intervals over rows (2,000 resamples; the Kaggle evaluation report quotes 0.720-0.820 for the overall AUROC from 1,000 resamples). Details: D-029, D-030, `results/final_v3b/analysis.txt`, `results/final_qwen_v3b/analysis_*.txt`.

| | ModernBERT v3b | GPT-5-mini | GPT-5-nano | length only | tool name only |
|---|---|---|---|---|---|
| AUROC, all 363 rows | **0.771** (0.720-0.818) | 0.719 (0.666-0.770) | 0.619 (0.562-0.677) | 0.675 (0.620-0.730) | 0.716 (0.661-0.767) |
| AUROC, 269 natural rows | 0.753 (0.699-0.807) | 0.695 (0.620-0.760) | 0.623 (0.549-0.691) | 0.626 (0.556-0.693) | 0.705 (0.643-0.765) |
| AUROC, 94 induced rows | 0.724 (0.606-0.832) | 0.698 (0.567-0.820) | 0.586 (0.449-0.728) | 0.559 (0.424-0.700) | 0.579 (0.440-0.718) |
| recall at 5% / 10% false alarms | 0.229 / 0.329 | 0.100 / 0.471 | 0.112 / 0.229 | 0.212 / 0.329 | 0.076 / 0.135 |
| at 0.5 (recall / false-flag) | 0.494 / 0.197 | 0.547 / 0.145 | 0.435 / 0.218 | 0.094 / 0.021 | 0.335 / 0.155 |

Paired differences (ModernBERT minus the other): GPT-5-mini +0.052 (-0.020 to +0.119); GPT-5-nano +0.151 (+0.065 to +0.232); length only +0.096 (+0.038 to +0.151); tool name only +0.055 (+0.001 to +0.108). Other test sets: held-out synthetic Classes 2 and 6 (test_synth, 374 rows, not real errors) AUROC 0.756 (0.710-0.803); validation (749 rows, mostly synthetic) 0.927. On this 363-row test set, the difference from GPT-5-mini is within the confidence interval (paired 95% CI includes zero); a larger test set would be needed to confirm the gap.

Per class, recall at 0.5 (n errors): Class 1 0.43 (30), Class 2 1.00 (4), Class 3 0.44 (18), Class 4 0.63 (19), Class 5 0.40 (20), Class 6 0.50 (16), Class 7 0.49 (63). **Cells with few rows are not readable**: Class 2 has 4 rows (all induced), Classes 5 and 6 have 1 natural error each. Per domain AUROC: airline 0.783 (0.699-0.861), retail 0.763 (0.690-0.834), telecom 0.771 (0.654-0.872). Speed: 0.68 s median (95th percentile 2.3 s) per call on one Mac CPU, 4 threads, fp32, one row at a time; 0.04 s per row batched on a T4 (D-030).

## Known failure modes and caveats

- **It misses most errors at strict settings** (recall 0.23 at 5% and 0.33 at 10% false alarms), and a GPT-5-mini judge does better at 10% false alarms and on Class 1 (wrong argument value: recall 0.77 against 0.43) and Class 3.
- **Shortcuts exist in the test set that the model may use.** The test errors have longer histories than the correct calls (length alone gives AUROC 0.675), and the tool name alone gives 0.716; the model is only about 0.05 above the tool-name baseline. For `cancel_reservation` in airline, 5 of 14 correct test calls still score above 0.9 (D-029).
- **Induced rows behave differently from natural ones.** They come from runs with another agent and user simulator; the earlier model flagged 73% of their correct calls and v3b still flags 27%. Class 7 in telecom was weaker after the v3b data (AUROC 0.708, D-029).
- **Phrasing and format sensitivity (anecdotal; two or three probes, D-031).** A correct cancellation of a pending order scored 0.02 with an explicit confirmation of order ID and reason, 0.88 with a shorter "Yes.", and 0.9999 with a 3-character order ID; a clear policy violation (cancelling a delivered order) scored 0.015 and was not flagged. Hand-made examples are outside the training distribution.
- **Agent reply style is a cue.** Wrapping natural agent messages in the JSON style of the induced runs raised scores by about 0.17 in an ablation of the previous model (D-027); the normalization that would remove it was built and shelved.
- **Evidence can be cut.** Rows over 4,096 tokens are tightened by a fixed ladder; 26 of the 363 test rows needed it with the policy block on (D-025). Policy text exists only for 18 write tools in the three τ²-bench domains.
- **Labels.** Test rows were reviewed by a human under a written guide with two non-obvious rules (read-only calls are never policy violations; a first lookup of a customer's telecom line is correct unless the target was knowable). These rules were then applied to some labels without a second look: 34 calls first marked wrong were changed to correct, and 26 accepted errors were removed. Training labels for real errors were proposed by an LLM under rules. Other reviewers might label differently, and the model inherits the guide's choices.
- **Small test set, reused.** 363 rows; the test set was used to choose among three ModernBERT runs and to decide to try an ensemble, so the numbers are optimistic by an unknown amount; the thresholds above come from the same rows. On this 363-row test set, the difference from GPT-5-mini is within the confidence interval (paired 95% CI includes zero); a larger test set would be needed to confirm the gap.
- **Not tested:** other domains, other agents, adversarial inputs, fairness across user groups (the data has no user attributes, and user names in τ²-bench are simulated).

## Citation and contact

Shalini Srivastava, https://github.com/Shalini-Srivastavaaa/toolcall-verifier (questions and corrections: open an issue there).

```bibtex
@misc{srivastava2026toolcallverifier,
  author = {Srivastava, Shalini},
  title  = {toolcall-verifier: a small local verifier for AI-agent tool calls},
  year   = {2026},
  url    = {https://github.com/Shalini-Srivastavaaa/toolcall-verifier}
}
```

Base model: ModernBERT (Answer.AI and LightOn). Data: τ²-bench (Sierra Research), BFCL (UC Berkeley Gorilla), CriticTool; see `docs/licenses.md` for what must be cited and under which terms.
