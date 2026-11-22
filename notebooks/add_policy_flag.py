"""Add the optional POLICY: block (flag INCLUDE_POLICY, default False) to format_input_v2 in the Phase 5 Colab notebook (D-021).

    python notebooks/add_policy_flag.py        # idempotent; then run notebooks/make_kaggle_notebooks.py

The snippets come from data/policy_snippets.py (verbatim policy text per write tool). The block is placed after TOOLS and
before the history, so CANDIDATE stays last; it counts toward TOKEN_BUDGET_V2 and is never clipped (the ladder shortens
the other parts first). With the flag off the input is byte-identical to before.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data"))
import policy_snippets  # noqa: E402

BEGIN, END = "# --- policy snippets (begin) ---", "# --- policy snippets (end) ---"
NB = ROOT / "notebooks" / "phase5_modernbert.ipynb"


def sub(text, old, new):
    assert text.count(old) == 1, f"{text.count(old)} matches for {old[:70]!r}"
    return text.replace(old, new)


def block_text() -> str:
    sn = policy_snippets.build()
    listing = ",\n".join(f"    {json.dumps(k)}: {json.dumps(v, ensure_ascii=False)}" for k, v in sn.items())
    return (f"{BEGIN}\n"
            "# INCLUDE_POLICY adds a `POLICY:` block with the policy text of the candidate's tool (D-021). On in DATASET = \"final\" mode (D-023), off in legacy.\n"
            "# The text depends on the tool only (never on the row), copied from the benchmark's policy documents by data/policy_snippets.py.\n"
            "INCLUDE_POLICY = (DATASET == \"final\") if INCLUDE_POLICY_OVERRIDE is None else bool(INCLUDE_POLICY_OVERRIDE)\n"
            "print(\"INCLUDE_POLICY =\", INCLUDE_POLICY)\n"
            f"POLICY_SNIPPETS = {{\n{listing},\n}}\n"
            f"{END}\n")


def main():
    nb = json.load(open(NB))
    cell = next(c for c in nb["cells"] if c["cell_type"] == "code" and "def format_input_v2" in "".join(c["source"]))
    t = "".join(cell["source"])
    if BEGIN in t:
        t = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END) + r"\n", lambda m: block_text(), t, flags=re.S)
    else:
        t = sub(t, "TOKEN_BUDGET_V2 = 4096\n", "TOKEN_BUDGET_V2 = 4096\n" + block_text())
        t = sub(t, '''    # EARLIER_CALLS: calls that fall before the window (one line, most recent last), as in v1
''', '''    # POLICY: the policy text of the candidate's tool (D-021), only when INCLUDE_POLICY is on; never clipped
    policy = POLICY_SNIPPETS.get(f"{ex.get('domain')}::{cand['name']}") if INCLUDE_POLICY else None
    if policy:
        lines.append("POLICY: " + policy)

    # EARLIER_CALLS: calls that fall before the window (one line, most recent last), as in v1
''')
    cell["source"] = t.splitlines(keepends=True)
    json.dump(nb, open(NB, "w"), indent=1, ensure_ascii=False)
    open(NB, "a").write("\n")
    print("patched", NB.name)


main()
