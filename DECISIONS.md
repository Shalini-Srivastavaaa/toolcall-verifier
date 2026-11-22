# Decisions that shaped the project

A short list of the decisions that determined the final project: what was decided, why and the key number, with a link to the full entry. The complete log, with every experiment, caveat and correction, is [`docs/decision_log.md`](docs/decision_log.md) (D-001 to D-032).

## 1. Seven-class taxonomy, defined by what the call does wrong

Wrong argument value, stale state, wrong tool, repeated call, failure to propagate, ignored error, policy violation. The first 102 hand-labeled trajectories showed ~75% of the benchmark's heuristic "incorrect" rows were artifacts and Classes 2 and 6 absent from real data. Full entry: [D-007](docs/decision_log.md#d-007--phase-1-taxonomy-validation-results).

## 2. Phase 0 judge result corrected

GPT-5-nano/mini looked like rubber stamps (recall 18-23%) only because they were scored against heuristic labels with 24.5% precision. Re-scored on hand labels they reach AUROC 0.709 / 0.667 with 32-48% recall: the headroom is smaller and lies in Class 4 and 7. Full entry: [D-005](docs/decision_log.md#d-005--phase-0-as-hard-project-kill-gate), [D-006](docs/decision_log.md#d-006--phase-0-baseline-results), [D-010](docs/decision_log.md#d-010--phase-4-real-held-out-test-set-test_realjsonl).

## 3. Correct-only positives, synthetic negatives

No negative comes from the benchmarks' own incorrect labels (75% mislabeled). Positives are gold-matched or verified calls; negatives are corruptions, split by `(domain, task_id)` so twins never straddle train and test. Full entry: [D-008](docs/decision_log.md#d-008--correct-only-training-pool-strategy), [D-009](docs/decision_log.md#d-009--phase-3-corruption-strategy).

## 4. Paired controls for every artifact

Each generator leaves a fingerprint (a fake `{"status": "success"}` result in all 1,000 Class 4 negatives and none of 3,517 positives). Each gets a control that keeps the artifact without the error, so the artifact alone cannot predict the label. Full entry: [D-009](docs/decision_log.md#d-009--phase-3-corruption-strategy), [D-012](docs/decision_log.md#d-012--control-positives-for-classes-4-and-6).

## 5. Class 7: dropped, then redefined as "policy violation" and trained again

Synthetic Class 7 (deleted lookups) taught a length shortcut and was dropped. It returned as edited-lookup "state flips" (D-021), real errors and the policy text as input; in D-024 it was redefined as "policy violation" with sub-reasons after 50 of 66 wrong calls in a review were Class 7 (37 read-only). Full entry: [D-009](docs/decision_log.md#d-009--phase-3-corruption-strategy), [D-021](docs/decision_log.md#d-021--class-7-state-flip-negatives-and-the-policy-input), [D-024](docs/decision_log.md#d-024--labeling-guide-class-7-renamed-policy-violation-queue-5-and-queue-2-errors-in-the-test-set-2026-11-15).

## 6. Evidence-aware input (`format_input_v2`)

The first input showed a median 11% of tool-result text. v2 keeps 30 turns, shows a result in full when it contains a value of the candidate call, and uses a 4,096-token budget (median visibility 93%). AUROC moved 0.655 to 0.686 on 102 rows: not enough alone. Full entry: [D-017](docs/decision_log.md#d-017--evidence-aware-input-for-modernbert-format_input_v2).

## 7. Training moves to Kaggle, one T4

Colab's quota ran out mid-run. On Kaggle two T4s with DataParallel took 7.75 h; length bucketing and one GPU cut ModernBERT to about 1.2-1.9 h. Qwen needs a GPU, micro-batch 2 with gradient checkpointing (D-029). Full entry: [D-020](docs/decision_log.md#d-020--training-moves-to-kaggle-notebooks-kaggle_phase5_modernbertipynb-kaggle_phase6_qwenipynb), [D-029](docs/decision_log.md#d-029--analysis-of-the-v3b-run-220-real-training-errors-real_weight--3-the-recommendation-and-the-qwen-preparation-2026-11-21).

## 8. Real errors mined from failed runs

Of 3,651 failed τ²-bench runs, gpt-5-mini localizes the first incorrect call (420 runs for $1.16), a second labeler and humans check it. Agreement between the two cheap labelers was only 19%, so proposals are never ground truth. Full entry: [D-018](docs/decision_log.md#d-018--mining-real-errors-from-tau2-bench-failed-runs-stage-1-review-file-only).

## 9. Induced errors for Classes 2, 5 and 6

These almost never occur naturally, so we inject a tool fault, a user correction or an ID change and let a real agent react. They stay reported separately: they carry a signature (another agent and user simulator) the model can learn. Full entry: [D-019](docs/decision_log.md#d-019--inducing-class-2-and-class-6-situations-and-outside-sources-for-them), [D-027](docs/decision_log.md#d-027--analysis-of-the-real-weighted-run-final_v2-real_weight--3-94-real-training-errors-2026-11-21).

## 10. Human review, a labeling guide and two rules

Every test label is a human decision (five queues). Rules written during review: a read-only call is never Class 7 and parallel lookups are correct; a first telecom line lookup is correct unless the target was knowable (removed 26 errors; test went from 197 to 170 errors). Full entry: [D-022](docs/decision_log.md#d-022--final-dataset-builder-databuild_final_datasetpy-dry-run-only), [D-024](docs/decision_log.md#d-024--labeling-guide-class-7-renamed-policy-violation-queue-5-and-queue-2-errors-in-the-test-set-2026-11-15).

## 11. Final dataset builder and training gates

Test rows come only from human decisions; no test task is in training. Real-origin training rows are admitted per class by measured first-labeler precision (Classes 1, 4, 7 on the first labeler's word, with the guide's rules applied). Full entry: [D-022](docs/decision_log.md#d-022--final-dataset-builder-databuild_final_datasetpy-dry-run-only), [D-026](docs/decision_log.md#d-026--real-data-in-training-the-data-fix-after-the-first-final-run-2026-11-21).

## 12. Diagnosis: 16 real errors in training

The first run on the final data scored 0.667, equal to the length baseline: of 2,366 training errors 16 (0.7%) were real. The cause was the admission gates, not the model. Full entry: [D-025](docs/decision_log.md#d-025--diagnosis-of-the-first-final-modernbert-run-final_v1-and-the-next-experiment-2026-11-21).

## 13. `REAL_WEIGHT = 3`

With 94 real errors and 580 real correct rows admitted, real-origin rows are shown three times per epoch: AUROC 0.667 to 0.732, but also exposed the `cancel_reservation` tool-name shortcut and the induced signature. Full entry: [D-026](docs/decision_log.md#d-026--real-data-in-training-the-data-fix-after-the-first-final-run-2026-11-21), [D-027](docs/decision_log.md#d-027--analysis-of-the-real-weighted-run-final_v2-real_weight--3-94-real-training-errors-2026-11-21).

## 14. Tool balancing and 126 more real errors

Correct calls of tools over-represented as errors were added (143 rows), and 520 more failed runs localized ($1.27): 220 real errors. AUROC 0.771. The text-normalization fix for the induced signature was built and shelved. Full entry: [D-028](docs/decision_log.md#d-028--two-dataset-versions-v3a-tool-balancing-and-v3b-more-real-errors-2026-11-21).

## 15. Final model: ModernBERT v3b

AUROC 0.771 (0.720-0.818), above GPT-5-mini (0.719), GPT-5-nano (0.619) and the length and tool-name baselines (0.675, 0.716). Kept because it is the most accurate scorer measured and runs in 0.68 s on a CPU at no per-call cost. Full entry: [D-029](docs/decision_log.md#d-029--analysis-of-the-v3b-run-220-real-training-errors-real_weight--3-the-recommendation-and-the-qwen-preparation-2026-11-21), [D-030](docs/decision_log.md#d-030--analysis-of-the-qwen25-15b-result-the-ensemble-the-synthetic-real-gap-speed-and-cost-and-the-final-readme-results-2026-11-22).

## 16. Qwen2.5-1.5B as a comparison, not the final model

AUROC 0.765 (tie overall; ahead on natural rows, behind on induced rows), 6 times slower per row on the same GPU. It fits synthetic errors perfectly (test_synth 1.000) but not real ones, pointing to real labeled data as the limit. An ensemble showed no demonstrated gain. Full entry: [D-030](docs/decision_log.md#d-030--analysis-of-the-qwen25-15b-result-the-ensemble-the-synthetic-real-gap-speed-and-cost-and-the-final-readme-results-2026-11-22).

## 17. Operating points are in-sample

The shipped thresholds (5% and 10% false-alarm rate: 0.895 and 0.731, recall 0.23 and 0.33) were set on the same 363 test rows that chose the model, and the demo and model card say so. Full entry: [D-031](docs/decision_log.md#d-031--portfolio-packaging-license-audit-inference-package-writeup-model-card-demo-plan-2026-11-22).

## 18. Publishing: Apache-2.0, CC BY 4.0 labels, test set only

Code and ModernBERT v3b weights are Apache-2.0; our human labels CC BY 4.0; the test set ships with τ²-bench's MIT notice; the training data (and with it CriticTool rows) is not published; release from a fresh git history. The terms for model-generated text are unreviewed. Full entry: [D-031](docs/decision_log.md#d-031--portfolio-packaging-license-audit-inference-package-writeup-model-card-demo-plan-2026-11-22), [D-032](docs/decision_log.md#d-032--publishing-licenses-public-test-set-clean-release-repo-hugging-face-prep-demo-v1-2026-11-22).
