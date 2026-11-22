"""NORMALIZE_AGENT_TEXT (D-027): a notebook flag, default False (off), that removes visible signatures of the induced runs from the model input.

The induced runs (data/induce_errors.py) used gpt-5-nano as the agent. Its text messages differ from those of the tau2 agents of the natural runs in two ways that the model can
read in the AGENT lines of its input: they are JSON objects (`{ "response": "..." }`; 82 of the 94 induced test rows have one, against 2 of 269 natural rows) and they are long
(median 844 characters of agent text per message against 261). In the final_v2 model, wrapping a natural row's agent messages in that JSON style raises its score by 0.16 (correct
rows) and 0.17 (errors) and the score of natural rows follows the length of the agent's messages (Spearman +0.55 within correct rows), and induced correct rows are scored like errors (D-027).
With NORMALIZE_AGENT_TEXT = True every assistant TEXT turn (not tool calls, not tool results, not user turns) is replaced by its plain-text value (a JSON object is unwrapped) cut to
AGENT_TEXT_CHARS characters, before the input is built, for training, validation, test and REAL_CV alike (the wrapper is around format_input_v2, which every part of the notebook calls).

Applied to the generated Phase 5 Kaggle notebook by make_kaggle_notebooks.py (after real_weight_code).
"""

CONFIG_LINES = '''NORMALIZE_AGENT_TEXT = False  # True: every assistant text turn is unwrapped from JSON ({ "response": "..." }, the style of the gpt-5-nano agent of the induced runs) and cut to AGENT_TEXT_CHARS characters in EVERY split before the input is built, removing visible signatures of induced rows (D-027). False = as in final_v1 / final_v2
AGENT_TEXT_CHARS = 200        # the cut used when NORMALIZE_AGENT_TEXT is True
'''

WRAPPER = r'''

# --- NORMALIZE_AGENT_TEXT (D-027): the wrapper around format_input_v2 (every tokenization and every evaluation calls it) ---
_format_input_v2_raw = format_input_v2


def _plain_agent_text(text):
    """The plain text of an agent message: a JSON object is unwrapped to its text value (keys response, text, message, else the longest string value), then the text is cut."""
    s = text.lstrip()
    if s.startswith("{"):
        try:
            obj = json.loads(s)
        except ValueError:
            obj = None
        if isinstance(obj, dict):
            for k in ("response", "text", "message"):
                if isinstance(obj.get(k), str):
                    text = obj[k]
                    break
            else:
                vals = [v for v in obj.values() if isinstance(v, str)]
                if vals:
                    text = max(vals, key=len)
    return text if len(text) <= AGENT_TEXT_CHARS else text[: AGENT_TEXT_CHARS - 1] + "…"


def _normalized_example(example):
    turns, changed = [], False
    for t in example["turns"]:
        if t.get("role") == "assistant" and isinstance(t.get("content"), str) and t["content"]:
            new = _plain_agent_text(t["content"])
            if new != t["content"]:
                t, changed = {**t, "content": new}, True
        turns.append(t)
    return {**example, "turns": turns} if changed else example


def format_input_v2(example, tok=True, meta=False):
    return _format_input_v2_raw(_normalized_example(example) if NORMALIZE_AGENT_TEXT else example, tok=tok, meta=meta)


print(f"NORMALIZE_AGENT_TEXT = {NORMALIZE_AGENT_TEXT}" + (f": assistant text turns are unwrapped from JSON and cut to {AGENT_TEXT_CHARS} characters in every split" if NORMALIZE_AGENT_TEXT else ""))
'''


def apply(nb):
    def src(c):
        return "".join(c["source"])

    cells = nb["cells"]
    cfg = next(c for c in cells if c["cell_type"] == "code" and src(c).startswith("# ================= run configuration"))
    s = src(cfg)
    anchor = "REAL_WEIGHT = 1"
    assert s.count(anchor) == 1
    cfg["source"] = s.replace(anchor, CONFIG_LINES + anchor, 1).splitlines(keepends=True)
    fmt = next(c for c in cells if c["cell_type"] == "code" and "def format_input_v2(example, tok=True, meta=False):" in src(c))
    fmt["source"] = (src(fmt).rstrip("\n") + WRAPPER).splitlines(keepends=True)
    return nb
