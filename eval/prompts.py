"""Judge prompt for Phase 0: can a cheap frontier model verify a multi-turn tool call?"""

JUDGE_SYSTEM_PROMPT = """You are a strict verifier for an AI customer-service agent. \
You will be shown the available tools, the full conversation so far (including \
the agent's prior tool calls and their results), and a CANDIDATE tool call the \
agent is about to make next.

Your job: decide whether the candidate tool call is correct given everything \
that happened in the conversation. A call can be well-formed (right tool, \
valid-looking arguments) and still be WRONG — for example: it uses a value \
from an earlier turn that the user later corrected, it repeats an action \
already completed, it ignores a previous tool error, it skips a required \
verification step, or it uses the wrong tool for what the user is asking for now.

Respond with ONLY a JSON object, no other text:
{"verdict": "correct" or "incorrect", "confidence": <float 0-1>, "reasoning": "<one sentence>"}
"""


def render_turns(turns: list[dict]) -> str:
    lines = []
    for t in turns:
        role = t["role"]
        if role == "assistant_tool_call":
            lines.append(f"[agent called tool] {t['name']}({t['arguments']})")
        elif role == "tool":
            lines.append(f"[tool result] {t['content']}")
        elif role == "user":
            lines.append(f"[user] {t['content']}")
        elif role == "assistant":
            lines.append(f"[agent] {t['content']}")
        else:
            lines.append(f"[{role}] {t.get('content', '')}")
    return "\n".join(lines)


def build_user_prompt(candidate: dict) -> str:
    turns_text = render_turns(candidate["turns"])
    call = candidate["candidate_tool_call"]
    return f"""AVAILABLE TOOLS ({candidate["domain"]} domain):
{candidate["tool_schemas_text"]}

CONVERSATION SO FAR:
{turns_text}

CANDIDATE TOOL CALL (about to be executed):
{call['name']}({call['arguments']})

Is this candidate tool call correct? Respond with the JSON object only."""
