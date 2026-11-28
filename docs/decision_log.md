# Decision log

The main decisions of the project, in order: what was decided, why, and what happened. Numbers refer to the files in this repository. The final model is **ModernBERT v3b**. Decision numbers (D-0xx) are cited by the writeup, the model card and the code; there is no D-016.

## D-001 — ModernBERT-base as the encoder baseline
Chosen for its long context (8,192 tokens), low training cost and the ability to run on a CPU.

## D-002 — Qwen2.5-1.5B as the comparison decoder
A model about ten times larger, fine-tuned with LoRA, to test whether model size is the limit. Planned as the Instruct variant; the trained model is the base `Qwen/Qwen2.5-1.5B` (D-013).

## D-003 — State compression
Long conversations do not fit the input. Left open here and settled by D-011 and then D-017.

## D-004 — Two sources of synthetic negatives
An LLM (gpt-5-nano) and rule-based corruptions, so that the model does not learn one generator's habits.

## D-005 — A kill test before building
The project would stop if cheap LLM judges were already strong (above 85% accuracy with little variation across error classes).

## D-006 — Baseline judges, and a correction
On 210 calls GPT-5-nano and GPT-5-mini scored 47-49% accuracy against heuristic labels. Those labels turned out to be only 24.5% precise (D-010). Re-scored on 102 hand labels, the judges reach AUROC 0.709 (nano) and 0.667 (mini) with 32% and 48% recall. The headroom was real but smaller than first reported, mainly in repeated calls and policy violations.

## D-007 — Taxonomy check on real data
Of 102 hand-reviewed calls marked wrong by the benchmark heuristic, 25 were real errors. Stale state and ignored errors did not appear. Duplicate telecom conversations were found by content hash and removed.

## D-008 — Correct-only positive pool
Benchmark "incorrect" labels are too noisy to use. Correct examples come from gold-matched calls; negatives come only from controlled corruption. Splits are by task, never by row.

## D-009 — Corruption strategy and paired controls
Each synthetic generator leaves a fingerprint. For example, a fake success result appeared in all 1,000 repeated-call negatives and in no positive. Paired controls keep the artifact and remove the error. Synthetic data for Class 7 made negatives shorter than positives, a model scored below chance (AUROC 0.276), and that data was dropped.

## D-010 — First real test set
102 hand-labeled calls (25 errors), never used for training, with zero overlap by id, task and conversation text.

## D-011 — First encoder run and input format
Last 10 turns, tool results cut to 200 characters, 1,536 tokens, with the split grouped by task.

## D-012 — Controls for repeated calls and ignored errors
Paired controls added for Classes 4 and 6. The validation split was frozen after it was found to change between library versions.

## D-013 — Qwen with LoRA
Rank 16, alpha 32 on the query and value projections, 4-bit weights, same data and evaluation as the encoder.

## D-014 — First results
On the 102-row test set ModernBERT scored AUROC 0.652 and Qwen 0.671. A rule-based checker for Class 7 caught none of the 8 real cases, because they were policy violations and not missing lookups.

## D-015 — More wrong-argument negatives
195 negatives added in which a value is swapped for another value of the same kind from the conversation.

## D-017 — Evidence-aware input
The first format showed a median of 11% of tool-result text. The new one keeps 30 turns, shows a result in full when it contains a value of the proposed call, and uses 4,096 tokens: median visibility 93%. AUROC moved only from 0.655 to 0.686 (interval 0.535-0.834, 102 rows).

## D-018 — Mining real errors
τ²-bench provides 10,832 recorded runs, 3,651 of them failed. gpt-5-mini proposes the first wrong call in a failed run. A second labeler agreed on call and class only 19% of the time, so proposals are never treated as ground truth. Test tasks were fixed by a seeded split.

## D-019 — Inducing rare errors
Stale values and ignored errors rarely occur naturally, so tool faults, user corrections and ID changes were injected and a real agent (gpt-5-nano, with a gpt-4.1-mini user simulator) reacted. The agent's calls were never edited. CriticTool tool-error examples were added to training. A length-only baseline was introduced.

## D-020 — Training on Kaggle
The first run took 7.75 hours: half of the processed tokens were padding and two GPUs were slower than one. Length-bucketed batches (padding about 2%) and one GPU brought later runs to about 1.2-1.9 hours.

## D-021 — Policy-state flips and policy text as input
Class 7 negatives were rebuilt by editing one lookup result so that the call breaks a policy rule, keeping records consistent so the edit cannot be spotted on its own. The policy text of the proposed tool was added to the input for 18 write tools.

## D-022 — Final dataset and human review
Test rows come only from human review queues: 149 proposed errors (79 accepted, 31 reclassified, 22 rejected, 17 skipped), 150 assumed-correct calls (148 decided: 82 correct, 66 wrong) and 54 rare-class candidates (29 accepted, 2 reclassified, 11 rejected, 12 skipped).

Written rules then adjusted some labels **without a second human review**:
- A further queue of 100 earlier decisions, mostly Class 7, was applied by rule (only two items had been re-decided by hand): 34 read-only calls first marked wrong became correct, one more read-only error candidate was removed from the errors, three kept their class and 62 decisions on write calls were kept.
- A rule for telecom line lookups removed 26 accepted errors (the test errors went from 197 to 170).

Final test set: 363 rows, 170 errors and 193 correct. Natural Class 7 training labels were kept although their measured precision (68%) was slightly under the 70% threshold.

## D-023 — Preparing the final run
A seeded, task-grouped validation split; a final report with shortcut baselines and per-class counts; more policy-state flips; and a judge script with a spending cap.

## D-024 — Labeling guide
Class 7 redefined as "policy violation" with sub-reasons. A read-only call is never Class 7 unless policy forbids that lookup, and parallel read-only lookups are correct. See [`labeling_guide.md`](labeling_guide.md).

## D-025 — Diagnosis of the first final run
Test AUROC 0.667 against 0.895 on validation. Only 16 of 2,366 training errors were real; validation errors were mostly synthetic.

## D-026 — Real data in training
The admission rules for real errors were relaxed to match the test labels (94 real errors), and real-origin rows can be repeated within an epoch.

## D-027 — Analysis of the weighted run
AUROC 0.732. Two shortcuts were found: all 20 test `cancel_reservation` calls scored a median of 0.99, and correct calls from induced runs were flagged 73% of the time. A text-normalization fix was built and then shelved.

## D-028 — Tool balancing and more real errors
Correct calls were added for over-represented tools, and 520 more failed runs were localized (about $1.27): 220 real errors. Final training set 7,154 rows; validation 749; test 363; synthetic test 374.

SHA-256 of the final data files:

| file | SHA-256 |
|---|---|
| `final_train.jsonl` | `d55797e89b262fd5c1272b6a178f8f63d5626b106e365db8238884bb1e574dab` |
| `final_val.jsonl` | `1ebfb5d92aef67bd6e69a14e5eab8414906242966a2537e385559f1b558fbb56` |
| `final_test.jsonl` | `f1c20d8e6656ce44be3746489abc98ed102abc46e04b2d1ee2de6ee0a68fdee6` |
| `final_test_synth.jsonl` | `fb2b869b2175ebafc190c35c9a1263652f0bb4ca17873ff07af78b6a51ced779` |

## D-029 — Final model: ModernBERT v3b
Test AUROC 0.771 (95% interval 0.720-0.818), natural rows 0.753, induced rows 0.724; training took 1.9 hours on one T4. It is not significantly better than the GPT-5-mini judge, which also catches more errors at 10% and 20% false alarms. Further tuning was stopped to avoid selecting on the test set. For Qwen, running out of GPU memory was fixed with micro-batch 2 and gradient checkpointing, and an evaluation crash was fixed with an evaluate-only path.

SHA-256 of the model files:

| file | SHA-256 |
|---|---|
| `model.safetensors` | `617accea7753c55ff139c71abc14b283b7bd96ff3a9ba9042159f2e10d004c24` |
| `config.json` | `538d50966465c05bc290ec4cd4d0b2c5d6d18fc9a6055afb63c7b491517b558e` |
| `tokenizer.json` | `0fc88ec706f80b2eb15174d04fa7377948ca3f07065985deec1e28d3fb1cda9d` |
| `tokenizer_config.json` | `fb54f027372062b2ca52282efb04d178a8b57167a00cd8f4e816515823a2c016` |
| `train_info.json` | `aca17fca87dc84a2582ad48475db522b4a95676358b08edcf4f1f0456358931c` |

## D-030 — Qwen, ensemble, synthetic-real gap, speed and cost
Qwen scores 0.765, tied with ModernBERT overall (paired difference -0.006, interval -0.052 to +0.037), ahead on natural rows and behind on induced rows. It fits the synthetic test perfectly (1.000) but not real errors, which is consistent with real labeled data being the limit; this was not tested in isolation. Averaging the two models showed no demonstrated gain. ModernBERT takes 0.68 s median per call on a laptop CPU, against 3.9 s for the GPT-5-mini API call.

## D-031 — Packaging and probes
The inference code reproduces the training input for all 363 test rows. The shipped thresholds (0.895 and 0.731) were set on the test set. A few hand-made probes showed sensitivity to phrasing: the same cancellation scored 0.02 with an explicit confirmation, 0.88 with a bare "Yes." and 0.9999 with a three-character order ID; cancelling a delivered order scored 0.015 and was not flagged.

## D-032 — Publishing and licenses
Code and weights under Apache-2.0, human labels under CC BY 4.0, the test set with the τ²-bench MIT notice. The training data is not published. The release was made from a fresh git history.

## D-033 — Release
GitHub repository, Hugging Face model and test set published. A hosted demo is pending because free Gradio hosting requires a paid Hugging Face plan; the demo runs locally.

## D-034 — Data provenance
The dataset card states that the conversations are model-generated and the labels come from human review. The model providers' terms were reviewed (see [`licenses.md`](licenses.md)).

## D-035 — Wording corrections
Statements that every test label was a human decision were corrected to describe the written rules of D-022. No data, label or number changed.

## D-036 — Repository cleanup
Internal working notes and superseded drafts were removed from the public repository, and this log was condensed from the full working notes. No code results, data or model files changed.
