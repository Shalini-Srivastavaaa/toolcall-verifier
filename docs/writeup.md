# Checking an agent's tool call before it runs: what worked, what didn't, and what the numbers can and cannot say

*Project writeup. Every number below comes from a file committed in this repository; the references in brackets (D-0xx) are entries of [`docs/decision_log.md`](decision_log.md), the project's full decision log, which also records the caveats this text compresses. The final model is ModernBERT v3b.*

**In one sentence.** A ~150M-parameter model that runs on a laptop CPU in about 0.7 s (GPT-5-mini: 3.9 s), at no API cost per call and with conversations that never leave your machine, matches or exceeds GPT-5-mini at catching real agent mistakes (AUROC 0.771 vs 0.719) and significantly beats GPT-5-nano (0.619) and the length shortcut (0.675).

## 1. The problem

An LLM agent that books flights, cancels orders or changes a phone plan does its work through tool calls. A call can be perfectly well formed and still be wrong: the agent uses a delivery address the customer corrected two turns ago, issues the same refund twice, carries on after a tool returned an error, or acts on an order that policy says it may not touch. A schema validator accepts all of these. The call executes, the database changes, and the damage is done before anyone looks.

The idea of this project is to put a small, fast model in front of the executor. It reads the conversation so far and the proposed call and returns a score for "this call is wrong". If the score is high, the call is held for a human or sent back to the agent. The useful question is not "can a model do this at all" (a frontier model asked to judge a call will give an answer) but "can a small specialist, trained on this kind of failure, beat a general judge on real agent mistakes, cheaply enough to run on every call?". Phase 0 of the project was a kill gate on that question (D-005), and the answer it eventually got is more modest than the first run suggested (section 4).

The testbed is τ²-bench, a benchmark of customer-service agents in three domains (retail, airline, telecom) whose tasks, policies and simulated users make multi-turn tool use realistic, plus BFCL multi-turn function-calling data for extra variety.

## 2. The error taxonomy

Seven classes, defined in [`data/taxonomy.md`](../data/taxonomy.md) and [`docs/labeling_guide.md`](labeling_guide.md):

| # | Class | Example |
|---|---|---|
| 1 | Wrong argument value | right tool, a mis-copied ID or date |
| 2 | Stale state | uses a value the user has since changed |
| 3 | Wrong tool | calls a different tool than the situation needs |
| 4 | Repeated call | re-issues a call that already went through |
| 5 | Failure to propagate | ignores or mis-copies a value an earlier tool returned |
| 6 | Ignored error | proceeds as if a failed call had succeeded |
| 7 | Policy violation | performs an action the domain policy does not allow at that point |

**Class 7 was redefined, and the redefinition mattered.** It started as "missing precondition / wrong order". When a human checked 150 calls that had been assumed correct, 66 of the 148 decided were judged wrong and 50 of those 66 had been put in Class 7; 37 of the 66 were read-only lookups, many of them issued in parallel (D-024). The old definition had been stretched to cover "several lookups at once" and "acted without a yes", which are different things. The labeling guide now says: Class 7 is an action the policy does not allow at that point; **a read-only call is never Class 7 unless the policy forbids that lookup; parallel read-only lookups are correct**, even though the benchmark's own policy asks for one call at a time, because that rule protects writes and independent lookups cannot hurt. Each Class 7 row also carries a sub-reason (`no_confirmation`, `ineligible_state`, `missing_step`, `precondition_error`, `parallel_write`, `other`); those tags are heuristic and approximate, and rates for sub-reasons with fewer than 10 rows are not shown.

## 3. The data

**Sources.** τ²-bench provides 10,832 recorded simulations, 3,651 of them failed (D-018); BFCL v4 multi-turn provides gold calls; CriticTool provides tool-error examples for Class 6 (D-019). Correct examples are calls that match the gold action, calls a human confirmed, and (for the real-origin training rows) the earlier calls of a failed run that precede the call flagged as the first mistake, which are assumed correct and screened by the labeling rules.

**Synthetic corruptions with paired controls.** Real errors are rare, so most training negatives are corruptions of correct calls: an LLM swaps a value, picks a different tool or injects a failing predecessor; programmatic rules insert a repeated call or a user correction (D-009). Every such generator leaves a fingerprint a model could learn instead of the error. The Class 4 generator, for example, inserts a fake `{"status": "success"}` result and that artifact was in all 1,000 Class 4 negatives and in none of the 3,517 positives. So each artifact gets a **paired control**: the same artifact with the error removed, labeled correct (D-012). Synthetic Class 7 data was dropped entirely: deleting a lookup made negatives shorter than positives, a first model scored below chance on real rows, and a rule-based checker built for Class 7 found 0 of 8 real errors. Later, Class 7 negatives were rebuilt as "state flips", where one lookup result is edited so the call breaks a policy rule, with the edit made consistent so the flip is not detectable from an inconsistent record (D-021).

**Induced real errors.** Class 2 and Class 6 almost never occur in the benchmark's own failed runs, because simulated users rarely change their minds and tools rarely fail. So we created the situation and let a real agent react, changing the environment or the user's instructions and never the agent's calls (D-019): a **fault** wrapper makes one call return an error (or, in the "soft" variant, a normal-looking response); a **mind-change user** gets one added instruction to correct an earlier value (payment method, address, bag count, date); an **induced value change** returns an order or reservation under a new ID. These runs used gpt-5-nano as the agent and gpt-4.1-mini as the user simulator. They are real model behavior, but they are less typical than runs of the benchmark itself, and section 4 shows how that bit us.

**LLM-assisted localization with human review.** The benchmark's checker says which expected action was missing, never which call was wrong. gpt-5-mini was asked to name the first incorrect call in a failed run (420 runs for $1.16; 273 got a flagged call, D-018). A second, independently prompted labeler (gpt-5-nano) agreed on the same call and class for only 19% of 202 training-split candidates (39 of 202), so these labels are proposals, not ground truth. Humans reviewed the proposals in queues: queue 1, 149 candidate errors from test-split tasks (79 accepted, 31 reclassified, 22 rejected, 17 skipped; induced Class 2 and Class 6 candidates were confirmed only 3 of 16 and 5 of 13 times, which led to stricter automatic checks, D-019); queue 2, 150 random "assumed correct" calls from the same runs; queue 4, 54 more rare-class candidates; queue 5, 100 earlier decisions (mostly Class 7) to be re-checked under the new definition, which was applied by written rules instead of a second human review: 34 read-only calls first marked wrong became correct, one more read-only error candidate was removed from the errors, and the decisions on write calls were kept (D-022).

**The test set.** 363 rows, all reviewed by a human (written rules then adjusted some labels), from tasks that never appear in training (the split is by task, with zero overlap checked by ID, by task and by identical conversation text, D-010, D-022): **170 errors and 193 correct calls**, 98 of the errors natural and 72 induced; by class, 30 / 4 / 18 / 19 / 20 / 16 / 63 for Classes 1 to 7; by domain 116 airline, 166 retail, 81 telecom. Two rules were applied to the labels: the read-only rule above, and a **line-probe rule** (a first lookup of one of a customer's several telecom lines, to find out which one is affected, is correct unless the target was already knowable; never Class 3). The second rule was written after the same kind of call had been decided in opposite ways during review; applying it removed 26 accepted errors (17 Class 3, 9 Class 1) and took the test set from 197 incorrect rows to 170 (README, D-022).

**The final training data** (the "v3b" dataset, D-028): 7,154 training rows (2,570 errors, 4,584 correct) and 749 validation rows; 220 of the errors are real (212 natural, 8 induced) and 1,377 correct rows are real, and these 1,597 real-origin rows are shown three times per epoch (`REAL_WEIGHT = 3`). All seven classes are present; Class 7 has 120 real and 163 state-flip errors. Test, test_synth and validation files are the same in v2 and v3b. The model input includes the policy text of the candidate's tool (D-021) and is at most 4,096 tokens. The final ModernBERT run is 3 epochs on one Kaggle T4 (1.9 hours); the Qwen comparison model is 2 epochs (6.8 hours).

A separate **test_synth** set of 374 held-out synthetic rows (Classes 2 and 6 and their controls) is reported on its own and is never mixed into the real numbers.

Two honest points about the labels. First, the same LLM-assisted pipeline proposed the real-origin *training* labels (with rules, no human decision per row), while every test row was reviewed by a human, with written rules adjusting some labels (section 3); the model can only be as good as its training labels, and section 6 comes back to that. Second, Class 2 has 4 test errors, all induced, and Classes 5 and 6 have one natural error each: for those classes the test set shows detection of an injected situation at best.

## 4. What went wrong, and how it was found

**The Phase 0 result was built on bad labels.** The first experiment asked GPT-5-nano and GPT-5-mini to judge 210 calls and reported accuracy of 47% and 49% and recall on bad calls of 18% to 23%: "the judges rubber-stamp calls", hence a big gap for a specialist (D-006). The labels were a heuristic (the call differs from the gold action, or the task failed). When 102 of those rows were checked by hand, only 25 were real errors, a precision of 24.5% (D-010). Re-scored against the hand labels the judges had AUROC 0.709 and 0.667, recall 32% and 48%, false-flag rate 13% and 16%: not rubber stamps, and statistically indistinguishable from the first trained models (D-006 correction). The thesis survived (the judges are weak on repeated calls and policy violations) but the headroom was smaller and different than first claimed.

**Shortcuts, two of them.** *Length:* a logistic regression on the length of the conversation alone scores AUROC 0.654 on the first test set and 0.675 on the final one (D-019, D-025): test errors have longer histories than correct calls, and an early synthetic Class 7 generator made the opposite artifact in training. A model that matches this baseline has not shown understanding. *Tool name:* the final_v2 model scored all 20 test calls to `cancel_reservation` (14 correct, 6 errors) with a median of 0.99, because in training that tool was mostly an error; those 14 correct calls alone set the 5% false-alarm threshold (D-027). The fix was to balance tool names in training with correct calls from successful runs (D-028); after it the median on the correct cancels is 0.17, though 5 of the 14 still score above 0.9 (D-029). A tool-name-only baseline scores AUROC 0.716 overall, which is why all results below are read against it.

**The model could see 11% of the evidence.** The first input format clipped each tool result to 200 characters and kept the last 10 turns. Measured on the first test set, the median share of tool-result text the model saw was 11% (pooled 7.6%); the correct order status or ID was usually cut off. The second format (`format_input_v2`) keeps results in full when they contain a value from the candidate call, always keeps the two latest results, and shrinks everything else only if the 4,096-token budget requires; visibility rose to a median of 93% (pooled 56%) (D-017). It helped less than hoped: AUROC went from 0.655 to 0.686 with an interval of 0.535 to 0.834 on 102 rows, and the first run on the final data later showed that seeing the evidence is not the same as having learned from real examples.

**Only 16 of 2,366 training errors were real.** The first run on the new, human-reviewed dataset scored 0.667 on the real test set, no better than the older model or than the length baseline (0.675), while validation AUROC was 0.895, because validation errors were synthetic (D-025). The diagnosis: the data builder had held back almost all real errors, because its admission gates were strict (a second labeler had to agree, induced errors needed paired correct calls, three classes were excluded after poor review precision). Only 16 of the 2,366 training errors (0.7%) came from real runs. Relaxing the gates with rules that matched the test labels and sampling real-origin rows three times per epoch (94 real errors) gave 0.732 (D-026, D-027). A further $1.27 of localization on 520 more failed training runs raised it to 220 real errors and 0.771 (D-028, D-029). The trend is clear; it is not a controlled dose-response, since each step also changed other things.

**The induced-run signature.** The final_v2 model gave induced *correct* rows a median score of 0.63 against 0.03 for natural correct rows, and flagged 73% of them at the 0.5 threshold (D-027). Induced runs differ from natural ones in ways the model can see: different agent, different user simulator, and agent replies wrapped in a JSON style. A direct test confirmed that the JSON style raises a natural row's score by about 0.17, but removing it from induced rows moved their scores only 0.04 to 0.07, so the signature is only partly explained. Tool balancing and more natural training data brought the induced false-flag rate down to 27% (D-029). A text-normalization fix was built and then shelved, since its own ablation said it would move induced scores only a little (D-027, D-028).

## 5. Results

All numbers are on the 363 real test rows; intervals are 95% bootstrap intervals over rows (2,000 resamples; the Kaggle evaluation report of ModernBERT quotes 0.720 to 0.820 for the overall AUROC from 1,000 resamples). "Recall at 5% false alarms" means the threshold is set so 5% of the correct calls are flagged. Details and the paired tests are in D-029 and D-030; the table is the one in the README.

| scorer | AUROC overall | natural rows | induced rows | recall at 5% FA | at 10% FA | latency / cost per call |
|---|---|---|---|---|---|---|
| **ModernBERT v3b** (149M) | **0.771** (0.720-0.818) | 0.753 (0.699-0.807) | 0.724 (0.606-0.832) | 0.229 | 0.329 | 0.68 s median on a laptop CPU; 0.04 s/row batched on a T4; no per-call fee |
| Qwen2.5-1.5B + LoRA | 0.765 (0.714-0.814) | 0.803 (0.753-0.853) | 0.533 (0.370-0.691) | 0.106 | 0.218 | 0.22 s/row batched on a T4 |
| *average of the two (exploratory)* | 0.780 (0.730-0.825) | 0.785 (0.733-0.838) | 0.676 (0.533-0.806) | 0.124 | 0.312 | both |
| GPT-5-mini judge | 0.719 (0.666-0.770) | 0.695 (0.620-0.760) | 0.698 (0.567-0.820) | 0.100 | **0.471** | 3.9 s median, $0.0012 |
| GPT-5-nano judge | 0.619 (0.562-0.677) | 0.623 (0.549-0.691) | 0.586 (0.449-0.728) | 0.112 | 0.229 | 2.9 s median, $0.00026 |
| length only | 0.675 (0.620-0.730) | 0.626 (0.556-0.693) | 0.559 (0.424-0.700) | 0.212 | 0.329 | negligible |
| tool name only | 0.716 (0.661-0.767) | 0.705 (0.643-0.765) | 0.579 (0.440-0.718) | 0.076 | 0.135 | negligible |

**Comparisons.** The paired bootstrap of AUROC differences is the right tool, because all scorers see the same rows. ModernBERT v3b is ahead of GPT-5-mini by 0.052 overall (0.771 vs 0.719), and Qwen by 0.046; On this 363-row test set, the difference from GPT-5-mini is within the confidence interval (paired 95% CI includes zero); a larger test set would be needed to confirm the gap. Both trained models are significantly better than GPT-5-nano and than the length shortcut. Against the tool name alone the gap is +0.055 for ModernBERT (interval +0.001 to +0.108) and +0.049 for Qwen (-0.007 to +0.100). At a strict operating point the judge catches more: at 10% false alarms GPT-5-mini catches 47% of errors against 33% for ModernBERT. Qwen and ModernBERT are the same overall (difference -0.006, -0.052 to +0.037), but differ by subset: Qwen is ahead on natural rows by 0.049 (interval 0.005 to 0.095) and behind on induced rows by 0.191 (-0.367 to -0.013). Those are two borderline intervals out of many comparisons, and the induced rows contain only 22 correct calls, so they are hints, not findings. The ensemble gains +0.009 overall over ModernBERT (interval -0.018 to +0.034): no demonstrated gain, and it was proposed after looking at the test results.

**Per class** (recall at the 0.5 threshold; n errors, of which natural):

| class | n (natural) | ModernBERT v3b | Qwen | GPT-5-mini |
|---|---|---|---|---|
| 1 wrong argument | 30 (21) | 0.43 | 0.27 | **0.77** |
| 2 stale state | 4 (0) | 1.00 | 0.75 | 1.00 |
| 3 wrong tool | 18 (14) | 0.44 | 0.39 | **0.67** |
| 4 repeated call | 19 (16) | 0.63 | **0.79** | 0.63 |
| 5 failure to propagate | 20 (1) | 0.40 | 0.50 | 0.55 |
| 6 ignored error | 16 (1) | 0.50 | 0.31 | 0.50 |
| 7 policy violation | 63 (45) | 0.49 | 0.40 | 0.37 |

The pattern is consistent with what the judges were already known to do (D-006): they are good at wrong arguments and wrong tools, where reading the facts of the conversation is enough, and weak at policy violations; the trained models are better at repeated calls and policy violations. The false-flag rate at 0.5 is 0.20 for ModernBERT and 0.15 for the mini judge. Classes 2, 5 and 6 are almost entirely induced and Class 2 has four rows, so those cells say little. On the 98 natural errors only Classes 1, 3, 4 and 7 have enough rows to read.

**Speed and cost.** ModernBERT v3b scores a call in 0.68 s at the median (95th percentile 2.3 s) on one laptop CPU, measured over all 363 rows, with no per-call fee; on a Kaggle T4, batched, 0.04 s per row. Qwen needs a GPU and takes six times as long per row on the same T4. The judges take 3 to 4 seconds per call and cost $0.0012 (mini) and $0.00026 (nano); scoring the 363 test rows with the mini costs $0.42. The T4 figures are batch throughput, not single-request latency, and the judges' latency is wall-clock time at the time of the calls (D-030). For the same overall AUROC as Qwen, ModernBERT is about six times faster per row on a GPU and runs without one, which is why it is the final model.

## 6. The gap between synthetic and real

On the held-out synthetic Classes 2 and 6 (test_synth) Qwen scores AUROC 1.000 and ModernBERT 0.756. On the validation split, whose errors are mostly synthetic, 0.982 and 0.927. On the real test set, 0.765 and 0.771. On the validation rows that come from real runs, 0.978 and 0.962, which is about 0.2 above the real test, for both models (D-030).

What this supports: **the larger model fitted the synthetic errors perfectly and did not do better on the real ones**, and the drop from "validation of real-origin rows" to "human-reviewed test rows" is the same for the model with ten times the parameters as for the small one. That is consistent with the real labeled data, not the capacity of the model, being the limit. The project's own history points the same way: ModernBERT's real-test AUROC went 0.667 (16 real training errors), 0.732 (94), 0.771 (220, with tool balancing), and a 1.5B model trained on the 220 reaches 0.765.

What it does not show. This is one pair of models, one recipe each (Qwen trained for two epochs, ModernBERT for three; different pretraining) and one 363-row test set with intervals of about +-0.05. "Real labeled data" means two different things here: *how many* real errors there are and *how their labels are made*. The training and validation labels come from an LLM-assisted pipeline with rules, and the test labels from human review; a model can learn the habits of the pipeline, which would make validation easy and test hard for any number of examples, and this experiment cannot separate the two explanations. The real-origin validation set has 92 rows and 20 errors. The test rows come from other agents than most training rows, and a distribution difference, not label noise, could produce the same gap. The sentence we would stand behind: making the model ten times larger fitted the synthetic errors perfectly and did not improve the real-test AUROC overall; the evidence points to the amount and the labeling of real data as the main limit, and that has not been tested in isolation.

## 7. Limitations and next steps

- **The test set is small and has been reused.** 363 rows with 170 errors (98 natural); intervals of about +-0.05; the test set was used to choose among three ModernBERT runs and to decide to try an ensemble, so the numbers are optimistic by an unknown amount. The operating-point thresholds shipped with the model were set on the same rows. On this 363-row test set, the difference from GPT-5-mini is within the confidence interval (paired 95% CI includes zero); a larger test set would be needed to confirm the gap.
- **Weak classes.** Class 1 recall is 0.43 for the final model against 0.77 for GPT-5-mini; recall at strict operating points is low for every trained model (0.33 at 10% false alarms). The model catches a third of the errors it is asked about at a 10% false-alarm cost, which makes it a triage signal, not a gate that stops most errors.
- **Brittleness.** Hand-made examples are outside the training distribution: in probes the same call scored 0.02 or 0.88 depending on how the confirmation was phrased, and a clear policy violation (cancelling a delivered order) was not flagged (D-031). This is anecdotal and not a study.
- **One kind of agent.** All data is customer-service dialogue in three τ²-bench domains plus BFCL; nothing here says how the model behaves on other tools, other domains or a different agent.
- **Licensing.** The test set and parts of the training data contain model-generated conversation text, The project owner reviewed those providers' terms (outputs owned by the user; no use to build competing models); that is the owner's reading, not legal advice (see [`docs/licenses.md`](licenses.md)). The code, the model and the test set are published; the training data is not.

**Next steps, in order of value.** (1) A larger, fresh human-reviewed set of *natural* errors, enough to separate effects of +0.05, kept untouched until the end. (2) A controlled test of the labeling-process question: train on the same number of errors labeled by the pipeline and labeled by humans. (3) Choose thresholds on a validation set of real rows, not on the test set. (4) Try Qwen again on the natural errors, where it was nominally ahead, and fix the induced-row behavior before any ensemble. (5) A demo that replays a conversation and blocks the flagged call (plan in [`docs/demo_plan.md`](demo_plan.md)).

## Where the numbers come from

| Topic | Source |
|---|---|
| Phase 0 judges, label precision, re-scoring | D-006 (correction), D-010; `results/phase0_baseline.json`, `results/phase0_rescored.json` |
| Visibility 11% to 93% | D-017 |
| Taxonomy rules, queue results, read-only and line-probe rules | D-018, D-019, D-022, D-024; `docs/labeling_guide.md` |
| Data sizes and real-error counts per run | D-025, D-026, D-028; `results/final_v1/train_composition_before_after.txt` |
| Test set composition | README "Final dataset, test set and results" |
| Shortcut baselines, `cancel_reservation`, induced signature | D-019, D-025, D-027, D-029; `results/final_v3b/analysis.txt` |
| Main table, paired tests, per class, per domain | D-029, D-030; `results/final_qwen_v3b/analysis_*.txt`, `results/final_v3b/analysis.txt` |
| Ensemble | D-030 step 2; `results/final_qwen_v3b/analysis_2_ensemble.txt` |
| Synthetic-real gap | D-030 step 3; `results/final_qwen_v3b/analysis_3_gap.txt` |
| Speed and cost | D-030 step 4; `results/final_qwen_v3b/speed_cost.json` |
