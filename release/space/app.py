"""Gradio demo of the ModernBERT v3b tool-call verifier (D-032). Runs on a CPU Space.

Local run:   MODEL_DIR=models/final_v3b python release/space/app.py          (the model folder of the repository)
On a Space:  the model is downloaded from the Hub repository named in MODEL_REPO (set it in the Space settings, or edit the default below).
"""

import json
import os
import re
import sys
from pathlib import Path

import gradio as gr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import predict  # noqa: E402

MODEL_REPO = os.environ.get("MODEL_REPO", "ShaliniSrivastava/toolcall-verifier-modernbert-v3b")
MODEL_DIR = os.environ.get("MODEL_DIR")
BLOB = json.load(open(HERE / "examples.json"))
EXAMPLES = BLOB["examples"]
TITLES = [e["title"] for e in EXAMPLES]
OPERATING = {"fa10 (10% of correct test calls flagged; the demo default)": "fa10", "fa05 (5%: stricter, flags less)": "fa05", "0.5": "default"}

LIMITATIONS = """
**Read this before drawing conclusions from any single score.**

- **The twelve examples are curated, not typical.** They are real test rows: 6 errors the model catches, 3 it misses and 3 correct calls (one of them wrongly flagged). **At the setting used here the model catches about one error in three** (recall 0.33 at a 10% false-alarm rate on the 363-row test set), so in real use most errors would pass.
- **The threshold is in-sample.** The 10% false-alarm threshold (0.731) was set on the same 363 test rows that were also used to choose the model. On other traffic the share of correct calls flagged will differ.
- **Small test set.** 363 rows, 170 errors (98 natural). AUROC 0.771 (95% interval 0.720-0.818). On this 363-row test set, the difference from GPT-5-mini is within the confidence interval (paired 95% CI includes zero); a larger test set would be needed to confirm the gap. At a 10% false-alarm rate the GPT-5-mini judge finds 47% of the errors against 33% for this model.
- **Brittleness (hand-made probes, anecdotal).** The same correct cancellation scored 0.02 with an explicit confirmation of order ID and reason, 0.88 with a bare "Yes.", and 0.9999 with a 3-character order ID; cancelling a *delivered* order (a clear policy violation) scored 0.015 and was not flagged.
- **Known shortcut.** `cancel_reservation` calls in the airline domain are often scored high even when correct (example with the false alarm; 5 of 14 correct test calls still score above 0.9).
- **Edited inputs may behave unpredictably.** The model was trained on benchmark conversations in a fixed text format; edited or hand-written text is outside that distribution.
- **Scope.** Customer-service conversations in the style of tau2-bench (retail, airline, telecom). The score is not a calibrated probability, and the model is not a security tool.
"""

_verifier = None


def verifier():
    global _verifier
    if _verifier is None:
        path = MODEL_DIR
        if not path:
            from huggingface_hub import snapshot_download
            path = snapshot_download(MODEL_REPO)
        _verifier = predict.Verifier(path)
    return _verifier


# --------------------------------------------------------------------------------------------- text <-> turns

PREFIX = re.compile(r"^(USER|AGENT|CALL|RESULT): ?(.*)$")


def turns_to_text(turns):
    out = []
    for t in turns:
        r = t.get("role")
        if r == "user":
            out.append("USER: " + str(t.get("content", "")))
        elif r == "assistant":
            out.append("AGENT: " + str(t.get("content", "")))
        elif r == "assistant_tool_call":
            out.append("CALL: " + json.dumps({"name": t["name"], "arguments": t.get("arguments", {})}, ensure_ascii=False))
        elif r == "tool":
            out.append("RESULT: " + str(t.get("content", "")))
    return "\n".join(out)


def text_to_turns(text):
    """Lines starting with USER:, AGENT:, CALL: or RESULT: begin a turn; other lines continue the previous turn. A CALL line holds JSON {"name": ..., "arguments": {...}}."""
    turns = []
    for line in text.split("\n"):
        m = PREFIX.match(line)
        if m:
            turns.append([m.group(1), m.group(2)])
        elif turns:
            turns[-1][1] += "\n" + line
        elif line.strip():
            raise gr.Error("The first line must start with USER:, AGENT:, CALL: or RESULT:")
    out = []
    for kind, body in turns:
        if kind == "USER":
            out.append({"role": "user", "content": body})
        elif kind == "AGENT":
            out.append({"role": "assistant", "content": body})
        elif kind == "RESULT":
            out.append({"role": "tool", "content": body})
        else:
            try:
                c = json.loads(body)
                out.append({"role": "assistant_tool_call", "name": c["name"], "arguments": c.get("arguments", {})})
            except Exception:
                raise gr.Error('A CALL line must be JSON like {"name": "get_order_details", "arguments": {"order_id": "#W123"}}: ' + body[:80])
    return out


def call_to_text(c):
    return json.dumps(c, ensure_ascii=False, indent=1)


# --------------------------------------------------------------------------------------------- the app logic

def load_example(title):
    e = EXAMPLES[TITLES.index(title)]
    return e["domain"], turns_to_text(e["conversation"]), call_to_text(e["candidate_call"]), None, "", "", "", ""


def score(title, domain, conv_text, call_text, op_label):
    if not conv_text.strip():
        raise gr.Error("The conversation is empty.")
    try:
        call = json.loads(call_text)
        assert isinstance(call, dict) and "name" in call
    except Exception:
        raise gr.Error('The candidate call must be JSON like {"name": "cancel_pending_order", "arguments": {"order_id": "#W123"}}')
    ex = EXAMPLES[TITLES.index(title)] if title in TITLES else None
    unedited = bool(ex) and domain == ex["domain"] and conv_text == turns_to_text(ex["conversation"]) and call == ex["candidate_call"]
    if unedited:
        turns, schemas = ex["conversation"], ex["tool_schemas_text"]      # exactly the stored row, so that the score equals the evaluation run's
    else:
        turns, schemas = text_to_turns(conv_text), None                    # the bundled tool-schema text of the domain
    op = OPERATING[op_label]
    v = verifier().score(turns, call, schemas, domain, op)
    flagged = v.flagged
    head = (f"## {'🔴 FLAG' if flagged else '🟢 ALLOW'}\n\n**Score (probability the call is incorrect): {v.probability:.3f}**  ·  threshold {v.threshold:.3f} ({v.operating_point})  ·  {v.tokens} tokens read"
            + ("  ·  ⚠️ close to the 4,096-token limit: older evidence may have been shortened" if v.tokens > 3900 else ""))
    policy = [l[len("POLICY: "):] for l in v.input_text.split("\n") if l.startswith("POLICY: ")]
    policy_text = policy[0] if policy else "No policy text for this tool: the model is given policy text only for the 18 write tools of the retail, airline and telecom domains."
    if unedited:
        h = ex["human"]
        what = "**correct**" if h["label"] == "correct" else f"**incorrect**, Class {h['error_class']} ({h['error_class_name']})"
        extra = f", sub-reason `{h['class7_reason']}` ({h['class7_reason_source']} tag, approximate)" if h.get("class7_reason") else ""
        outcome = {"caught": "an error that this threshold **catches**", "missed": "an error that this threshold **misses**", "allowed": "a correct call that is **allowed**", "false alarm": "a correct call that is **wrongly flagged**"}[ex["outcome_at_fa10"]]
        human = (f"**Human label:** {what}{extra}; origin: {h['origin']}. At the 10% false-alarm threshold this row is {outcome}. "
                 f"Probability on the evaluation run (Kaggle GPU, fp16): {ex['saved_probability']:.3f}; here (CPU, fp32): {v.probability:.3f}.")
    else:
        human = "**You edited the example, so no human label applies.** Edited inputs may behave unpredictably: the model was trained on benchmark conversations in a fixed format (see the limitations above)."
    return {"incorrect": v.probability, "correct": 1 - v.probability}, head, policy_text, human, v.input_text


with gr.Blocks(title="Tool-call verifier demo") as demo:
    gr.Markdown("# Tool-call verifier: would this call be wrong?\nA 149M-parameter ModernBERT classifier reads a customer-service conversation and **one proposed tool call** and scores how likely the call is incorrect, before it runs.\n\n**A ~150M-parameter model that runs on a laptop CPU in about 0.7 s (GPT-5-mini: 3.9 s), at no API cost per call and with conversations that never leave your machine, matches or exceeds GPT-5-mini at catching real agent mistakes (AUROC 0.771 vs 0.719) and significantly beats GPT-5-nano (0.619) and the length shortcut (0.675).** It scores a call in about 0.7 s on a laptop CPU against 3.9 s for GPT-5-mini, with no API cost and conversations that never leave your machine. "
                "Research demo on benchmark data (tau2-bench retail, airline, telecom). Not for production decisions. Code, model card and writeup: see the repository linked from the model page.")
    with gr.Accordion("⚠️ Known limitations (read first)", open=True):
        gr.Markdown(LIMITATIONS)
    with gr.Row():
        example = gr.Dropdown(TITLES, value=TITLES[0], label="Real test example (human-labeled; the label is shown after scoring)")
        domain = gr.Dropdown(["retail", "airline", "telecom"], value=EXAMPLES[0]["domain"], label="Domain")
        op = gr.Radio(list(OPERATING), value=list(OPERATING)[0], label="Decision threshold")
    with gr.Row():
        with gr.Column(scale=3):
            conv = gr.Textbox(value=turns_to_text(EXAMPLES[0]["conversation"]), lines=18, max_lines=40, label="Conversation so far (editable). Lines start with USER:, AGENT:, CALL: {json} or RESULT:",
                              info="Editing is allowed, but edited inputs may behave unpredictably.")
            call = gr.Textbox(value=call_to_text(EXAMPLES[0]["candidate_call"]), lines=5, label='Candidate call (editable JSON: {"name": ..., "arguments": {...}})')
            with gr.Row():
                go = gr.Button("Score", variant="primary")
                reset = gr.Button("Reset to the example")
        with gr.Column(scale=2):
            verdict = gr.Markdown("*Press Score.*")
            label = gr.Label(label="P(incorrect) / P(correct)", num_top_classes=2)
            human = gr.Markdown()
            policy = gr.Textbox(label="Policy text the model saw for this tool", lines=6, interactive=False)
    with gr.Accordion("The exact text the model read", open=False):
        seen = gr.Textbox(lines=20, interactive=False, show_label=False)
    example.change(load_example, example, [domain, conv, call, label, verdict, human, policy, seen])
    reset.click(load_example, example, [domain, conv, call, label, verdict, human, policy, seen])
    go.click(score, [example, domain, conv, call, op], [label, verdict, policy, human, seen])

if __name__ == "__main__":
    demo.queue(max_size=8).launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)), theme=gr.themes.Soft())
