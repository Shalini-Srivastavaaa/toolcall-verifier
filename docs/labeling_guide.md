# Labeling guide

How a human decides whether a proposed tool call is correct and, if not, which class of error it is. It is the reference for review queues 1 to 5
(`data/review_queues.py`) and for the sub-reasons stored on Class 7 rows. Decisions are the reviewer's; heuristics only pre-fill.

## The question

Judge **one call**, the one marked `>>>`, against what the agent knew at that point: the conversation, the tool results so far and the domain
policy (`data/raw/tau2-bench/data/tau2/domains/<domain>/policy.md`; telecom is `main_policy.md`). Use the policy of the domain on screen, not
another domain's. A call is not wrong because the run fails later, and not right because the run succeeds later.

Correct means: right tool, right arguments, nothing ignored, nothing stale, and the policy allows it at this point.

## Classes

| # | Class | The call ... |
|---|---|---|
| 1 | Wrong argument value | uses the right tool with a wrong value, although the right one was available |
| 2 | Stale state | uses a value the user had corrected or that the system had since changed |
| 3 | Wrong tool | uses a different tool from the one the situation calls for |
| 4 | Repeated call | repeats an action that already succeeded (a duplicate side effect) |
| 5 | Failure to propagate | ignores or mis-copies a value an earlier tool result gave |
| 6 | Ignored tool error | proceeds as if an earlier failed call had succeeded |
| 7 | **Policy violation** | performs or starts an action the domain policy does not allow at that point |

Class 7 was called "missing precondition / wrong order" until 2026-11-15. The numeric id is unchanged, so old files, labels and results keep
meaning the same class. When several classes apply, prefer the more specific one: 6, 4, 3, 2, 5, 1, and Class 7 when the call is wrong because
of the policy itself.

## Class 7: policy violation

**Definition.** The call performs or starts an action the domain policy does not allow at that point.

Every Class 7 row carries a `class7_reason`, one of:

| `class7_reason` | Meaning | Example |
|---|---|---|
| `no_confirmation` | a write without explicit user confirmation | the agent calls `cancel_reservation` right after the user asked a question, before listing the details and getting a "yes" (airline and retail policy both require it) |
| `ineligible_state` | a lookup already shows the action is not allowed for the current state | `modify_pending_order_items` on an order whose `get_order_details` says `delivered`; `update_reservation_flights` on a basic economy reservation |
| `missing_step` | a required check or step was skipped | `send_certificate` for a delay without a flight-status check; `resume_line` before the SIM PIN was unlocked |
| `precondition_error` | acting despite a domain precondition that is not met, though the information was in the conversation | exchanging items of a different product type; paying with more than the allowed number of cards |
| `parallel_write` | a write issued while an earlier call's result is still outstanding | one assistant message carries `get_user_details` and `update_reservation_baggages`; the write cannot depend on the lookup it sits beside |
| `other` | none of the above; always write a note | |

**Which reason when several fit.** Take the first that applies: `parallel_write`, then `no_confirmation`, then `ineligible_state`, then
`missing_step`, then `precondition_error`, then `other`. One reason per row, so recall by reason is well defined.

**`ineligible_state` or `precondition_error`.** If an earlier tool result in the conversation shows that the state forbids the action, it is
`ineligible_state`. If the forbidding fact comes from the user, the policy text or the arguments themselves, it is `precondition_error`.

## Two rules about read-only calls

1. **A read-only call is never Class 7.** Read-only means the tool name starts with `get_`, `search_`, `find_`, `list_` or is `calculate`.
   The exception is a policy that forbids that specific lookup (for example looking up an account before the user is authenticated, where the
   policy requires authentication first). Write the exception in the note. Telecom diagnostics (`check_*`, `can_*`, `run_speed_test`) read
   the device and change nothing; they are treated as read-only by the same rule. A call that is wrong for another reason (a duplicate
   lookup is Class 4, a lookup of the wrong id is Class 1) keeps that class.
2. **Parallel read-only lookups are not errors.** Two or more read-only calls in one assistant message are labeled **correct**, with the tag
   `parallel_lookup`. This **departs on purpose from the τ²-bench policies**, which tell the agent to make at most one tool call at a time
   (airline, retail and telecom policy files all say it). The reason: that rule exists because writes can depend on one another and on what a
   lookup returns; independent lookups cannot. Penalizing them would teach the verifier to flag harmless behavior and would mix a style rule with
   the errors the verifier is meant to catch. A write issued beside an outstanding call stays an error (`parallel_write`).

## Line lookups (telecom `get_details_by_id` on a line id)

Added 2026-11-16, after the review showed the same call, a first lookup of `L1001` while the user's line was one of several, decided in opposite ways.
A customer record lists line ids (`L1001`, `L1002`, `L1003`) with no phone numbers, so which line belongs to the stated number is learned by looking lines up.

- **Knowable.** The target line was identifiable before the call: the user named a line id, the customer has a single line, or an earlier tool result tied the
  stated phone number (or the reported problem) to one line. If the agent looked up a **different** line it is **Class 1**; if it looked up **that** line it is
  **correct**.
- **Not knowable.** A read-only lookup to find out which line is affected is **correct**, not an error.
- **Never Class 3.** No other tool should have been used for a line lookup.
- A lookup of a line already looked up earlier in the conversation is a repeated call and is outside this rule (Class 4 territory).

The check is structural (`data/line_probe_rule.py`): phone numbers and line ids in the user's messages, line records in earlier results (`line_id` with
`phone_number`), and the length of the customer's `line_ids`. It is applied as rule decisions (`data/apply_line_probe_rule.py`, file
`data/derived/review_queue_lineprobe_decisions.jsonl`, `source: "rule"`, reversible with `undo`); when it cannot decide confidently the item goes to queue 6
for a human, who sees the rule, the evidence and the earlier decision.

## Correct examples

A correct example is a call a human confirmed as correct (queue 2 or queue 5). A call that merely came before the first flagged call in a
failed run is not assumed correct: 66 of the 148 such calls the human decided on were judged wrong (D-022). Only human-confirmed correct
calls are used in the test set.

## Pre-fill and what is stored

Queue 5 shows the original decision, a rule summary, and a heuristic pre-fill (`data/class7_reasons.py`). The pre-fill only reads structure:
calls sharing an assistant message (parallel), the absence of an affirmative user message before a write (confirmation), the status in the
latest lookup of the same object (eligibility), and whether the object was ever looked up (missing step). It does not read the policy, so it is
a suggestion that the reviewer confirms or changes, and it is blank when nothing fires.

A queue-5 decision supersedes the earlier one for that item; the earlier record stays in the decisions file.
Queue 5 was applied by rule on 2026-11-15 instead of by hand (`data/apply_queue5_rules.py`, records in `data/derived/review_queue_rule_decisions.jsonl` with `source: "rule"`; a human queue-5 decision overrides a rule record, and deleting the file undoes the rules), so `class7_reason_source` is `heuristic` for most Class 7 rows and the sub-reason recall is approximate. Stored on the decision and on the
dataset row: `class7_reason` and `class7_reason_source` (`human` when the reviewer confirmed it in queue 5, `generator` for the synthetic state-flip rows,
`heuristic` otherwise), and `tags` (`["parallel_lookup"]` or absent).
