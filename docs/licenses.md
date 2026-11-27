# Licenses and data terms

This page summarizes what the project is built on and the terms it is published under. It is not legal advice. Full license texts are in [`licenses/`](../licenses/); per-project notices are in [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

## 1. What the project uses

| Component | License | Notes |
|---|---|---|
| τ²-bench (Sierra Research): domains, policies, task definitions, recorded simulations | MIT | keep the copyright notice and license text |
| BFCL / Gorilla (Berkeley Function-Calling Leaderboard v4 multi-turn data) | Apache-2.0 | keep the license and notices; state changes |
| CriticTool (`chocckaka/CriticTool-Dataset`, tool-error examples) | Apache-2.0 according to its Hugging Face dataset card | training and validation only; not redistributed (see 3.1) |
| ModernBERT-base (`answerdotai/ModernBERT-base`) | Apache-2.0 | base of the published model |
| Qwen2.5-1.5B (`Qwen/Qwen2.5-1.5B`) | Apache-2.0 | comparison model; no Qwen weights are published |

## 2. What this project publishes

- **Code and the fine-tuned ModernBERT v3b weights:** Apache-2.0 ([`LICENSE`](../LICENSE), [`NOTICE`](../NOTICE)).
- **Human labels of the test set:** CC BY 4.0.
- **Conversation text in the test set:** from τ²-bench under MIT, and BFCL rows under Apache-2.0 ([`DATA_LICENSE.md`](../DATA_LICENSE.md)).
- **Training data:** not published.

## 3. Points to check before reuse

**3.1 CriticTool.** The only license statement found for the dataset is the Hugging Face card. Its GitHub repository shows no license. The published weights were trained on part of this data (878 converted rows, 439 of them errors); none of it is redistributed.

**3.2 Model-generated text.** The conversations in the test set, the induced runs, the synthetic negatives and the proposed labels include output of commercial language models. The author reviewed the OpenAI and Anthropic terms (2026-11-22) and concluded that the output belongs to the user and that this project does not use it to build competing models. That is the author's own reading. Anyone reusing the text should check those terms for their own use. The τ²-bench authors' position on reuse of their recorded transcripts was not separately checked.

**3.3 BFCL provenance.** Only the multi-turn categories of BFCL v4 are used. The origin of each category was not audited.

**3.4 Personal data.** The conversations come from benchmark-simulated customers. The test set was checked with a pattern-based scan ([`release/test_set/pii_check.txt`](../release/test_set/pii_check.txt)), which cannot prove the absence of all personal data.

## 4. Attribution

- Model and code: "Fine-tuned from `answerdotai/ModernBERT-base` (Apache-2.0)", with the Apache-2.0 text and a statement of changes.
- Data: "Contains conversation text from τ²-bench (Sierra Research, MIT) and function-calling data from BFCL (Gorilla, Apache-2.0)", with both license texts.
