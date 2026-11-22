# Multi-Turn Tool-Call Verifier — Labeling Guide (phase-1 version)

> **Superseded for labeling by [`docs/labeling_guide.md`](../docs/labeling_guide.md)**, which fixes the final definitions (Class 7 = policy violation with sub-reasons, the read-only and line-lookup rules, the priority order). This page is the original guide the first 102 test rows were labeled with; the class table and examples below remain valid, the Class 7 text and the priority list do not.

Use this guide when assigning error classes to failing trajectories.
Each trajectory that ends in an incorrect tool call gets exactly one error class — the **primary** failure cause.

> **Labeling philosophy:** Gold actions are *not* ground truth. τ²-bench gold actions were constructed before model runs and can contain DOB mismatches, card numbers not in the user profile, or other benchmark artifacts. Judge the agent's tool call against the **conversation** — not against the gold action. Only label a genuine agent reasoning error.

---

## Ground truth definition

A tool call is **correct** if and only if:
1. The tool selected matches the user's current intent
2. Every argument value matches the current conversation state (not a stale or overridden value)
3. The call has not already been successfully made in this trajectory
4. All prerequisite tool calls have been completed and succeeded
5. The arguments are derived from the correct sources (user input, tool outputs, schema constraints)

If any of the above fails, the call is **incorrect**. Assign the class that describes the **first** thing that fails in the list above.

---

## Class 1 — Wrong Argument Value

**When to use:** The tool is correct and the intent is right, but at least one argument has the wrong value.

**Key signal:** The value exists somewhere in the conversation — it's just the wrong one. Look for: numbers that appear elsewhere in the context, names from earlier turns, IDs from prior messages.

**Real example from τ²-bench (airline task 33):**
```
User: "Let's keep both flights in economy and just add 2 checked bags to my reservation."
Agent: update_reservation_baggages(reservation_id="HXDUBJ", total_baggages=2, nonfree_baggages=1, ...)
                                                                               ^^^^^^^^^^^^^^^^
Gold:  update_reservation_baggages(reservation_id="HXDUBJ", total_baggages=2, nonfree_baggages=2, ...)
```
The user explicitly requested 2 bags. The user is a gold member with 1 free bag, so `nonfree_baggages` should be 2−1=1 *only if* the agent computed the free-bag entitlement correctly. Here the conversation shows the agent computed it incorrectly (gold says `nonfree_baggages=2`, meaning the agent over-counted free bags). Class 1: wrong value derivable from conversation context.

**Includes:**
- Wrong numerical value (amount, quantity, count, bag count)
- Wrong name or string value
- Wrong ID (when the correct ID was derivable from context)
- Wrong date or time

**Do NOT use if:** The value was correct at some earlier point but the user later changed it (→ Class 2). The tool itself is wrong (→ Class 3).

---

## Class 2 — Stale State / Ignored Correction

**When to use:** The tool and intent are correct, but the agent sourced an argument value from an earlier turn that was subsequently overridden or corrected.

**Key signal:** The value was *valid at some earlier point* but the user explicitly changed it. The agent used the old value.

**Example (template — not observed in Phase 1 τ²-bench data):**
```
Turn 2: User: "Transfer $500 to Rahul."
Turn 4: User: "Actually, make it $300."
Turn 5: Agent: transfer_money(amount=500)
→ Class 2: $500 was valid at Turn 2 but overridden at Turn 4
```

**Distinguishing from Class 1:** Class 1 is a value that was *never* correct in this conversation. Class 2 is a value that *was* correct earlier but is now stale.

**Includes:**
- Ignoring user corrections ("actually", "wait", "change that to", "I meant")
- Using a value from before an explicit override
- Acting on a cancelled or modified instruction

> **Note:** Not observed in τ²-bench Phase 1 data. More likely in longer agentic tasks with explicit user corrections mid-flow.

---

## Class 3 — Wrong Tool Selection

**When to use:** The agent chose the wrong tool for the user's intent, regardless of arguments.

**Key signal:** There is a clearly better-matching tool in the schema for what the user asked.

**Real example from τ²-bench (retail task 71):**
```
User: "Change my payment method to PayPal. And only modify the backpack item."
Agent: modify_pending_order_payment(order_id="#W5782623", payment_method_id="paypal_7729105")
Expected: modify_pending_order_items(order_id="#W5782623", ...)
```
The user's primary request was to change the *item* in the order (backpack variant), with PayPal for the price difference. The agent called `modify_pending_order_payment` — a tool for changing the payment method only, not item contents. That's the wrong tool.

**Real example (retail task 111):**
```
User: "Modify the laptop to i5/256GB/space grey + exchange the watch for the black dial version."
Agent: exchange_delivered_order_items(order_id="#W9810810", ...)
# The laptop is in a *different* order than the watch; the agent mixed up which action belongs where
```

**Priority:** Check this before Class 1. If the tool is wrong, don't assign Class 1 even if arguments are also wrong.

---

## Class 4 — Repeated Call / Duplicate Side Effect

**When to use:** The tool call is a duplicate of one that already succeeded earlier in the trajectory.

**Key signal:** Find the identical (or functionally identical) call earlier in the trajectory with a success result.

**Real example from τ²-bench (telecom, 199+ turn trajectories):**
```
Turn 12: get_details_by_id(id="D1002") → {device info for D1002}
...
Turn 198: get_details_by_id(id="D1002")   ← same call, same ID, already returned
```
The agent loops through get_details_by_id calls in very long telecom debugging trajectories. By turn 198, it has already fetched the same device/line/plan details multiple times. The call at turn 198 is a pure duplicate.

**Also observed (retail task 103):**
```
Turn N: get_order_details(order_id="#W1154986")  ← after already fetching this order
```

**Note:** The duplicate doesn't have to be *identical*. If the first call returned success and the agent calls the same action on the same entity again, it's Class 4. This class is very common in long telecom trajectories where the agent re-fetches data it already has.

---

## Class 5 — Failure to Propagate Tool Output

**When to use:** A prior tool call returned a value that should be used in the current call, but the agent used a different value or ignored the returned value to make a downstream decision.

**Key signal:** Find the tool result in the trajectory that contains the value/signal that *should have determined* the next action, and confirm the agent did not use it correctly.

**Real example from τ²-bench (telecom, 58-turn trajectory):**
```
Turn 52: send_payment_request(bill_id="B1002") → "Payment request sent"
Turn 54: [simulated tool] Payment of $150 made for bill B1002
Turn 55: User: "I've completed the payment. What should I do next?"
Turn 56: get_bill_details(bill_id="B1002") → bill object returned (payment_status not verified by agent)
Turn 57 (candidate): resume_line(customer_id="C1001", line_id="L1002")
```
The agent called `get_bill_details` at turn 56 but then jumped to `resume_line` without propagating/checking the `payment_status` field from the returned bill object. The correct flow requires confirming the bill shows as paid before resuming the line.

**Distinguish from Class 1:** Class 1 is a wrong value from the *conversation* itself. Class 5 is a wrong action or value when the correct signal came from a *prior tool result* that the agent fetched but didn't use.

---

## Class 6 — Ignoring a Tool Error

**When to use:** A prior tool call in the trajectory returned an error or failure, but the agent proceeds with a call that depends on the failed operation succeeding.

**Example (template — not observed in Phase 1 τ²-bench data):**
```
Turn 3: update_payment_method(...) → {"error": "payment method not found"}
Turn 4: confirm_payment(payment_method_id=...)
→ Class 6: Turn 4 assumes Turn 3 succeeded; it didn't
```

**Priority:** Check the preceding tool results. If any of them is an error that the current call depends on, this is Class 6.

> **Note:** Not observed in τ²-bench Phase 1 data. τ²-bench environments typically return graceful results rather than hard errors.

---

## Class 7 — Policy Violation

> Renamed on 2026-11-15 from "Missing Precondition / Wrong Order"; the numeric id and the stored slug `missing_precondition_wrong_order` are unchanged. Current definition and sub-reasons: [`docs/labeling_guide.md`](../docs/labeling_guide.md).

**When to use (phase-1 wording; the current definition is "performs or starts an action the domain policy does not allow at that point", see the guide):** The agent skipped a required step in the workflow before making this call (sub-reason `missing_step`). The *current* call is not necessarily wrong in isolation — the sequence is wrong.

**Key signal:** There is a required preceding call that hasn't happened (or hasn't succeeded) in this trajectory.

**Real example from τ²-bench (telecom service_issue, task: lock_sim_card_pin | overdue_bill_suspension):**
```
Correct workflow:  ... pay_overdue_bill → unlock_sim_pin → resume_line
Agent:             ... pay_overdue_bill → resume_line   ← skipped unlock_sim_pin
Candidate:         resume_line(customer_id="C1001", line_id="L1002")
```
The telecom task had *two* suspension causes: overdue bill AND a locked SIM PIN. The agent paid the bill (correctly) but then immediately called `resume_line` without first calling `unlock_sim_pin`. The line cannot be resumed while the SIM PIN is still locked — wrong order.

**Real example (airline task 14 — Claude Sonnet):**
```
User has basic economy tickets; wants to upgrade to business + use travel certificate.
Agent: update_reservation_flights(flights=[...], cabin="business")
← Should have called calculate() or check_upgrade_fee() before modifying
```

**Real example (airline task 5 — send_certificate before status check):**
```
Agent: send_certificate(user_id="mei_brown_7075", amount=200)
← Flight status check (required to verify delay) was not completed with a success result before sending
```

**Note:** The required workflow may be stated in the system prompt, implied by domain logic, or established by earlier turns.

---

## Priority order (when multiple classes could apply)

If a trajectory could plausibly be labeled multiple classes, use this order:

(phase-1 order; the final order in the guide is 6, 4, 3, 2, 5, 1, and Class 7 when the call is wrong because of the policy itself)

1. **Class 6** first — if a prior tool failed, that's the root cause; downstream errors follow from it
2. **Class 7** — if a prerequisite step is missing, that's the structural failure
3. **Class 4** — if it's a duplicate of a successful call
4. **Class 3** — if the tool selection is wrong
5. **Class 2** — if a stale/overridden value was used
6. **Class 5** — if a tool output wasn't propagated
7. **Class 1** — if a wrong value was used (catch-all for value errors)

---

## When to skip (label `no_error`)

Label a trajectory `no_error` and skip it when:

1. **Benchmark artifact:** The gold action differs from the candidate because of τ²-bench construction issues — DOB in the transcript doesn't match the gold, a card number in the gold isn't in the user's profile from conversation context, or the gold action index is inconsistent with what was asked. The agent may be *correct* given what the conversation says.
2. **Ambiguous policy:** The agent's choice (e.g., `nonfree_baggages=0` for a gold elite member) is plausible under domain policy that isn't explicitly stated in the transcript.
3. **Incomplete transcript:** The conversation is truncated and you cannot confirm whether the prerequisite step succeeded or the referenced value is wrong.

> **Phase 1 finding:** ~75% of heuristic-labeled "incorrect" examples (those flagged by the exact-match comparison against gold) were benchmark artifacts. Only ~25% were genuine agent errors.

---

## Label format (for `labels_manual.jsonl`)

```json
{
  "trajectory_id": "...",
  "domain": "airline",
  "task_id": "33",
  "source_model": "gpt-4.1-2025-04-14",
  "taxonomy_class": "wrong_argument_value",
  "label": "wrong_argument_value",
  "raw_input": "wrong_argument_value",
  "labeled_at": "2026-10-03T..."
}
```

**Error class names (snake_case):**
- `wrong_arg_value` / `wrong_argument_value`
- `stale_state`
- `wrong_tool_selection`
- `repeated_call_duplicate_side_effect`
- `failure_to_propagate_tool_output`
- `ignored_error`
- `missing_precondition_wrong_order`
- `no_error`
