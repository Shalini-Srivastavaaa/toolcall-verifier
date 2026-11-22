---
license: other
license_name: cc-by-4.0-labels-and-mit-tau2-content
license_link: https://huggingface.co/datasets/ShaliniSrivastava/toolcall-verifier-test-set/blob/main/DATA_LICENSE.md
language:
- en
task_categories:
- text-classification
configs:
- config_name: real
  default: true
  data_files:
  - split: test
    path: test.jsonl
- config_name: synthetic
  data_files:
  - split: test
    path: test_synth.jsonl
pretty_name: Tool-call verifier test set (human-reviewed real tool calls from tau2-bench)
size_categories:
- n<1K
---

# Test set: is this tool call correct?

**Links:** [code and results on GitHub](https://github.com/Shalini-Srivastavaaa/toolcall-verifier) · [the verifier model](https://huggingface.co/ShaliniSrivastava/toolcall-verifier-modernbert-v3b)

363 **real** tool calls from customer-service agents (τ²-bench retail, airline, telecom), each labeled by a human as `correct` or `incorrect` (with one of seven error classes), and 374 **synthetic** calls (`test_synth`) that are reported separately. Built for the ModernBERT v3b verifier of the `toolcall-verifier` project; see the writeup `docs/writeup.md` and decision log entries D-010, D-022, D-024, D-030 (in `docs/decision_log.md` of the code repository).

**License: our labels CC BY 4.0, the τ²-bench content MIT (Sierra Research), the BFCL rows Apache-2.0.** Read [`DATA_LICENSE.md`](DATA_LICENSE.md) before reuse; it contains the notices you must keep.

## Files

| file | rows | content |
|---|---|---|
| `test.jsonl` | 363 | the real test set: 170 `incorrect`, 193 `correct` |
| `test_synth.jsonl` | 374 | held-out **synthetic** rows for Classes 2 and 6 and their paired controls (187 incorrect, 187 correct); not real errors; 176 from τ²-bench conversations, 198 from BFCL |
| `pii_check.txt` | | the personal-data check (below) |

## Fields

| field | meaning |
|---|---|
| `id` | row id (a τ²-bench run id and call, or a generated id); the same ids are in the prediction files of the repository |
| `domain` | `retail`, `airline`, `telecom` (τ²-bench) or `bfcl` |
| `conversation` | the turns before the call: `{"role": "user" \| "assistant" \| "assistant_tool_call" \| "tool", "content": ...}`; an `assistant_tool_call` has `name` and `arguments` |
| `candidate_call` | the proposed tool call being judged: `{"name", "arguments"}` |
| `tool_schemas_text` | the text of the tool list available to the agent (needed to build the model input) |
| `label` | `correct` or `incorrect` (the positive class) |
| `error_class`, `error_class_name` | for incorrect rows: 1 wrong argument value, 2 stale state, 3 wrong tool, 4 repeated call, 5 failure to propagate, 6 ignored error, 7 policy violation; null for correct rows |
| `origin` | `natural` (a real run of the benchmark), `induced` (a real run in which we injected a fault, a user correction or an ID change, and a real agent reacted) or `synthetic` (`test_synth` only) |
| `class7_reason` | for Class 7: `no_confirmation`, `ineligible_state`, `missing_step`, `precondition_error`, `parallel_write`, `other` or `unlabeled`; `class7_reason_source` says `human`, `heuristic` or `generator`. **62 of the 63 Class 7 test rows have a heuristic tag, one is human: treat the sub-reasons as approximate** |
| `synthetic` | true for `test_synth` |

## Composition of `test.jsonl`

Errors by class (natural / induced): 1 wrong argument 30 (21 / 9); 2 stale state 4 (0 / 4); 3 wrong tool 18 (14 / 4); 4 repeated call 19 (16 / 3); 5 failure to propagate 20 (1 / 19); 6 ignored error 16 (1 / 15); 7 policy violation 63 (45 / 18). In all 98 natural and 72 induced errors; 171 natural and 22 induced correct calls. By domain: airline 116 rows (53 errors), retail 166 (72), telecom 81 (45). **Classes 2, 5 and 6 are almost all induced and Class 2 has four rows: per-class results for them are not readable.** Natural errors in the test set are limited to Classes 1, 3, 4 and 7 in any number.

## How it was built and reviewed

1. **Sources.** Calls from τ²-bench's recorded simulations (10,832 runs, 3,651 failed) and from our own induced runs (an agent and a user simulator run on τ²-bench tasks with one injected perturbation). Test tasks were fixed by a seeded split of the benchmark's tasks (every task of the first 102 hand-labeled rows, then 30% of the rest of each domain); no task of the test set is in the training data (checked by id, by task and by identical conversation text).
2. **Proposals.** A language model (gpt-5-mini) proposed, for each failed run, the first incorrect agent call and its class; rule-based heuristics proposed Class 2 and 6 candidates; correct calls were the earlier calls of the same runs, sampled at random. These are only proposals.
3. **Human review.** Every test row is a human decision: queue 1 (every error candidate of the test tasks), queue 2 (a random sample of calls assumed correct; any judged wrong became an error), queue 4 (further rare-class candidates), queue 5 (Class 7 decisions re-made under the final definition). 102 of the rows come from the earlier hand labeling (Phase 4); its Class 7 labels were re-decided under the final definition in queue 5. Reviewers saw the conversation, the policy of the domain and the call, and chose correct, a class, or skip (skipped items are not in the set).
4. **Two rules written during the review** (in `docs/labeling_guide.md`): a read-only call (`get_*`, `search_*`, `find_*`, `list_*`, `calculate`, and telecom `check_*`/`can_*`) is **never** a policy violation unless the policy forbids that lookup, and parallel read-only lookups are correct even though the benchmark's policy asks for one call at a time; and a first lookup of one of a customer's several telecom lines, to find out which one is affected, is correct unless the target was knowable (never Class 3). The second rule removed 26 accepted errors (17 Class 3, 9 Class 1) when it was applied, taking the test set from 197 incorrect rows to 170.
5. **Known limits of the labels.** One reviewer under a written guide, not several: other reviewers might decide some rows differently (the Class 7 and Class 1 borderline cases most). The labels are not adjudicated. Test rows from `induced` runs share a signature (another agent and user simulator) that a model can learn; report natural and induced rows separately. The 116 correct calls from the random sample and 77 from the first hand labeling were each confirmed by a human; the 34 that a rule turned from "wrong" into "correct" are tagged in D-022 of the repository's decision log.

## Personal data check

Run by `scripts/build_test_set_release.py`; full output in `pii_check.txt` (values masked). It searched every conversation, tool result and call for e-mail addresses, phone numbers, card-like numbers (Luhn-valid), SSN-like numbers, IBANs, street addresses, dates of birth and capitalized word pairs that look like names, and compared each hit with τ²-bench's own database, task and policy files.

**Result, real test rows (363) and the τ²-bench rows of `test_synth` (176):** no phone number, card number, SSN or IBAN pattern outside τ²-bench's own records. Of 584 e-mail occurrences (55 distinct) all but one are in the benchmark's records; the one (`youremail@example.com`) is a placeholder in the example field of a tool result. Of 1,036 street-address occurrences all but one are in the records; the one is "123 Maple Street, Springfield", a generic address written by an LLM agent in an induced run. Dates of birth (searched near the words birth, DOB, born): all but four distinct dates are in the records. The four are values that someone typed during a run: `1970-01-01` for "John Smith" (a placeholder an agent used in a telecom lookup), `1959-04-15` (given by the simulated user in an induced telecom run), and two airline dates of birth (2012 and 2015) that the simulated user gave for two children in one o4-mini run; none is in the benchmark's database, and all four were produced by the language-model simulators, not by real people (as far as the run logs show). Name-like pairs not explained by the benchmark's vocabulary are, on inspection, phrases ("Preferred Network", "Galaxy Tab"), the poet "Robert Frost" (a user quoting a poem), "John Doe" and "John Smith". **BFCL rows (198):** no e-mail, phone, SSN, IBAN or street address; one Luhn-valid 16-digit string, `2345-6789-1234-5678` (a sequential placeholder from BFCL's own data), and a few dates and names from BFCL's fictional scenarios (for example "Maxwell Edison", "Lara Croft").

**What this does not prove.** The check is pattern-based; it would not catch a real name written as a single word, a free-text address in another format, or a real person's details that happen to equal a string in the benchmark. It shows that the data has the properties of benchmark-simulated customers, which is what τ²-bench documents; it is not a guarantee. If you find personal data in these files, open an issue and we will remove the rows.

## Using it

```python
import json
rows = [json.loads(l) for l in open("test.jsonl")]
y = [r["label"] == "incorrect" for r in rows]       # the positive class; score with the probability that a call is incorrect
```

The verifier in this repository (`verifier/predict.py`) takes `conversation`, `candidate_call`, `tool_schemas_text` and `domain` straight from these rows. Report AUROC on all rows, on `origin == "natural"` and on `origin == "induced"` separately, with a bootstrap interval; the set has 170 errors (98 natural), so differences below about 0.05 AUROC are not resolvable.

## Citation

Shalini Srivastava, *toolcall-verifier*, 2026, https://github.com/Shalini-Srivastavaaa/toolcall-verifier. If you use the τ²-bench content please also cite τ²-bench; for the BFCL rows, the Berkeley Function-Calling Leaderboard.
