> **Historical document (the original plan, 2026-10).** The project changed after it was written: training moved from Colab to Kaggle, Class 7 was redefined as policy violation, and the final model, data and results are described in [`../README.md`](../README.md) and [`writeup.md`](writeup.md). Kept for reference.

# Multi-Turn Tool-Call Verifier — Project Plan

**One-line pitch:** A small trained model that scores whether an agent's tool call is correct *before it executes*, catching the state-tracking failures that single-turn schema validators miss.

**Duration:** 10–12 weeks, part-time
**Compute:** Colab (T4 / L4 / occasional A100)

---

## 1. Goals

### Primary goal
Ship a verifier that beats a cheap frontier judge (GPT-5-nano / mini) on **multi-turn** tool-call verification, at meaningfully lower latency and cost per call.

### What "done" looks like
Four artifacts, in priority order:

1. **A latency/cost/accuracy table** comparing your verifier against nano-as-judge and mini-as-judge on multi-turn trajectories. This is the headline.
2. **A risk-coverage curve** — at what confidence threshold do you catch X% of bad calls while only flagging Y% of good ones. This is what makes it a production gate rather than a leaderboard entry.
3. **A held-out real-failure test set** with hand-assigned error classes, and your score on it alongside your score on synthetic data. Publishing both numbers, even when the gap is ugly, is the single most credible thing in the writeup.
4. **A demo** where the gate visibly fires and escalates before a bad call runs.

### Non-goals (write these down, they protect you)
- Not beating BFCL single-turn SOTA. That's solved and nobody cares.
- Not reproducing ToolRM. Their taxonomy is single-turn; yours isn't.
- Not building an agent framework. You already have that signal.
- Not a throughput/cost-savings pitch. Inference deflation killed those.

---

## 2. Stack

**Models**
- `answerdotai/ModernBERT-base` (150M) — the encoder floor. 8192 context.
- `Qwen/Qwen2.5-1.5B-Instruct` — the decoder, LoRA-tuned. Strong tool/code priors.
- Optional stretch: `Qwen2.5-7B-Instruct` with QLoRA if you get A100 time.

**Training**
- `transformers`, `peft` (LoRA), `trl` (reward modeling / Bradley-Terry if you go pairwise), `bitsandbytes` (4-bit for 7B), `accelerate`
- `wandb` free tier for run tracking — matters more than you'd think when Colab disconnects

**Data sources**
- τ²-bench (Sierra) — multi-turn agent trajectories, retail/airline/telecom domains
- BFCL v3/v4 multi-turn (Berkeley) — the standard leaderboard
- FC-RewardBench (IBM, on HF) — 1,500 examples, use as a single-turn sanity check only

**Eval**
- `scikit-learn` for calibration (ECE, Brier, AUROC), custom risk-coverage plotting
- Your own harness — you will end up writing this, plan for it

**Demo / serving**
- FastAPI + a Gradio or Streamlit front-end. Keep it small.

**Publishing**
- GitHub repo, HF Hub model card, one long-form writeup post

---

## 3. Phase plan

### Phase 0 — The kill test (Week 1)
**Do this before anything else.**

Run GPT-5-nano and GPT-5-mini as verification judges over a slice of multi-turn trajectories. Measure accuracy, p50/p95 latency, and cost per verification.

- If nano is already strong → stop, and we pick a different project. You lost a week, not three months.
- If nano is mediocre (likely — the hard part is state tracking across turns, not schema matching) → you now own the headroom number the whole project rests on.

Output: one table. It becomes slide one of the writeup either way.

### Phase 1 — Error taxonomy (Week 2)
Write down the multi-turn failure classes before touching a GPU. Starting candidates, to be cut and extended against what you actually see:

- acting on stale state (value changed earlier in the conversation)
- repeating a call already made
- skipping a required precondition
- tool was correct three turns ago, isn't now
- failing to re-read a user-corrected value
- correct tool, arguments drawn from the wrong turn

The taxonomy drives your data pipeline and your labeling guide. Getting it wrong here is expensive later.

### Phase 2 — Harness + failure collection (Weeks 3–4)
Run an agent (small open model is fine — you *want* failures) against τ²-bench and BFCL multi-turn. Collect and store full trajectories, tagged pass/fail.

You need two things out of this: a pool of real failures to label, and the plumbing to replay trajectories during eval.

### Phase 3 — Synthetic negatives (Week 4–5)
Take passing trajectories, corrupt exactly one step in a way that maps to a named taxonomy class. You get a labeled pair with a known error *location*, not just a failed rollout.

**Critical:** vary the corrupting model across several open models of different sizes. If one generator produces all your negatives, your classifier learns its fingerprint instead of learning what a wrong call looks like.

### Phase 4 — Hand-labeled test set (Week 5, one weekend)
A few hundred real failures from Phase 2, error class assigned by hand. **Never train on these.** This is your honesty check.

Paste batches into a chat and we work through them together — I propose a class and location, you correct me. Your corrections set the standard; I just speed up throughput.

### Phase 5 — Encoder baseline (Weeks 6–7)
ModernBERT classification head. ~1 hour on an L4. Gives you a floor fast.

The interesting constraint: a multi-turn trajectory plus full tool schemas will blow past 8192 tokens. How you compress state — schema summarization, sliding window over turns, structured state extraction — is a real modeling decision. Write up what you tried and what broke.

### Phase 6 — Decoder model (Weeks 8–9)
Qwen2.5-1.5B + LoRA. A few hours on T4/L4 per epoch.

The comparison against the encoder is where your modeling-depth signal lives: at what trajectory length do the decoder's pretrained tool priors start paying for their latency?

### Phase 7 — Calibration and abstention (Week 10)
Temperature scaling or similar, then the risk-coverage curve. This is the step that converts a classifier into a gate someone would actually deploy.

### Phase 8 — Demo and writeup (Weeks 11–12)
Demo, latency/cost table, both accuracy numbers (synthetic and real), model card, post.

---

## 4. Things that will go wrong

**Colab disconnects mid-training.** Checkpoint to Drive every N steps from day one. Not week eight.

**Synthetic-vs-real gap.** If you score 92% on synthetic and 61% on real, you learned the perturbation artifact. That's what Phase 4 exists to catch — and finding it in week six is a good outcome, not a bad one.

**Context length.** Trajectories are long. Budget real time for the state-compression problem; it is not a footnote.

**Scope creep toward single-turn.** Single-turn data is clean and abundant, which makes it seductive. It is also the part nobody will be impressed by. Notice if you drift.

**Benchmark harness mismatch.** τ²-bench numbers vary wildly across trackers because the harnesses differ. Always report which harness you ran.

---

## 5. Immediate next steps

1. Set up the repo and a Colab notebook that writes checkpoints to Drive.
2. Get API access sorted for the nano/mini baseline.
3. Pull τ²-bench and BFCL, confirm the data format and licensing.
4. Run Phase 0.

Nothing else matters until Phase 0 comes back.
