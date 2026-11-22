# Multi-Turn Tool-Call Verifier — Failure Taxonomy

> **Original taxonomy draft (project phase 1).** The definitions and rules used to label the final dataset are in [`labeling_guide.md`](labeling_guide.md); where this draft differs (Class 7, read-only calls, the priority order) the guide wins.

## Core scope

The verifier evaluates whether an agent's proposed tool call is correct given the current multi-turn interaction state, previous tool calls and results, user intent, and required workflow.

The project focuses on seven primary failure categories.

---

## 1. Wrong Argument Value

The agent selects the correct tool but supplies an incorrect argument value.

Mis-extraction of information is treated as a subtype of this failure.

### Example

User:
> "Transfer ₹3,000 to Rahul."

Agent:
```python
transfer_money(
    contact_id=8271,
    amount=5000
)
```

### Failure

The tool is correct, but the argument value is wrong.

### Includes

- Incorrectly extracted values
- Incorrect numbers
- Incorrect names
- Incorrect IDs
- Incorrect dates
- Incorrect quantities
- Incorrect parameters derived from the conversation

---

## 2. Stale State / Ignored Correction / Wrong Turn

These are treated as the same underlying failure: **the agent sources truth from the wrong turn in the conversation.**

The agent uses information that was valid earlier but is no longer the correct state.

### Example

User:
> "Transfer ₹5,000 to Rahul."

Later:
> "Actually, transfer ₹3,000."

Agent:
```python
transfer_money(
    contact_id=8271,
    amount=5000
)
```

### Failure

The agent used information from an earlier turn instead of the current state.

### Includes

- Stale information
- Ignoring user corrections
- Using an old value after it was changed
- Using information from the wrong conversation turn
- Failing to update the current state after a user modification

---

## 3. Wrong Tool Selection

The agent selects the wrong tool for the user's current intent.

### Example

User:
> "What's the status of my order?"

Agent:
```python
cancel_order(order_id=123)
```

Expected:
```python
get_order_status(order_id=123)
```

### Failure

The agent's selected tool does not correspond to the intended action.

---

## 4. Repeated Call / Duplicate Side Effect

The agent repeats a tool call that has already been successfully performed.

A duplicate side effect is the particularly important case where repeating the call can cause the external action to happen twice.

### Example

First call:
```python
create_order(order_id=123)
```

Tool result:
```text
success
```

Agent calls:
```python
create_order(order_id=123)
```

again.

### Failure

The agent failed to recognize that the action had already been completed.

### Includes

- Repeating an already successful call
- Duplicate transactions
- Duplicate messages
- Duplicate orders
- Duplicate updates
- Any repeated call that can produce an unintended second side effect

---

## 5. Failure to Propagate Tool Output

The agent receives useful information from a previous tool call but fails to use that information correctly in a subsequent tool call.

### Example

Previous tool call:
```text
search_customer("Rahul")
```

Tool result:
```text
customer_id = 8271
```

Agent then calls:
```python
update_customer(customer_id=9124)
```

### Failure

The agent failed to propagate the relevant information returned by the previous tool.

### Includes

- Ignoring returned IDs
- Ignoring returned values
- Failing to pass a required result to the next tool
- Using an unrelated value instead of the tool's output

---

## 6. Ignoring a Tool Error

A previous tool call fails, but the agent behaves as if it succeeded or proceeds with an action that depends on the failed operation.

### Example

Previous call:
```python
update_payment_method(...)
```

Tool result:
```text
ERROR: payment method not found
```

Agent then calls:
```python
confirm_payment(...)
```

### Failure

The agent failed to account for the previous tool error.

### Includes

- Treating an error as success
- Continuing a dependent workflow after failure
- Using data that was never successfully produced
- Taking an action that assumes a failed operation succeeded

---

## 7. Policy Violation

> Renamed on 2026-11-15 from "Missing Precondition / Wrong Order"; the numeric id is unchanged. The definition and sub-reasons used for labeling are in [`labeling_guide.md`](labeling_guide.md): the call performs or starts an action the domain policy does not allow at that point.

The call performs or starts an action the domain policy does not allow at that point. Every Class 7 row carries a sub-reason: `no_confirmation`, `ineligible_state`, `missing_step`, `precondition_error`, `parallel_write` or `other`. A read-only call is never Class 7 unless the policy forbids that lookup (see the guide). Skipping a required verification step is the sub-reason `missing_step`; the draft example below is of that kind.

### Example (sub-reason `missing_step`)

Required workflow:
```text
1. Search customer
2. Verify customer
3. Update customer
```

Agent:
```text
update_customer(...)
```

without first completing the required verification.

### Failure

The agent performed an action before its required precondition was satisfied.

### Includes

- Skipping required verification
- Calling dependent tools too early
- Incorrect tool-call sequence
- Performing an action before required information has been obtained
- Violating a required workflow order

---

# Summary

| # | Failure Category | Core Problem |
|---|---|---|
| 1 | Wrong Argument Value | Correct tool, incorrect argument |
| 2 | Stale State / Ignored Correction / Wrong Turn | Truth sourced from the wrong turn |
| 3 | Wrong Tool Selection | Incorrect tool chosen |
| 4 | Repeated Call / Duplicate Side Effect | Already-completed action repeated |
| 5 | Failure to Propagate Tool Output | Previous tool result not carried forward |
| 6 | Ignoring a Tool Error | Agent proceeds as if a failed call succeeded |
| 7 | Policy Violation | An action the domain policy does not allow at that point (missing step, no confirmation, ineligible state, ...) |

## Unifying Principle

All seven categories can be viewed as failures where the proposed tool call is inconsistent with the information or state available to the agent at that point in the trajectory.

```text
Current interaction state
        +
Previous tool calls/results
        +
User intent
        +
Workflow constraints
        +
Candidate tool call
        ↓
     VERIFIER
        ↓
   Correct / Incorrect
```

The project does not attempt to detect every possible agent failure. It specifically focuses on failures that make an **agent's tool invocation incorrect in the context of the multi-turn trajectory**.
