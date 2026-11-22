"""Add the Class 7 (policy violation) recall-by-sub-reason block to the final-mode report of the two Colab notebooks (labeling guide, 2026-11-15).

    python notebooks/add_class7_report.py            # patches phase5_modernbert.ipynb and phase6_qwen.ipynb once (idempotent)
    python notebooks/make_kaggle_notebooks.py        # then regenerate the Kaggle notebooks

The block is the one now in FINAL_EVAL of add_final_mode.py (fresh notebooks get it from there; this script brings the already patched ones up to date).
"""

import json
from pathlib import Path

NB = Path(__file__).resolve().parent
MARK = "<- the number to quote"
OLD_START = "    # Class 7 = policy violation: recall by sub-reason"
ANCHOR = '    print("\\nper domain")\n'
BLOCK = '''    # Class 7 = policy violation (docs/labeling_guide.md). MAIN NUMBER: overall Class 7 recall. The sub-reason split below is approximate: since 2026-11-15 the
    # reasons come from a heuristic pre-fill applied by rule (D-022), not from a human, so it is labeled that way.
    m7 = np.array([r["label"] == "incorrect" and r["error_class"] == 7 for r in test_rows])
    print(f"\\nCLASS 7 (policy violation) RECALL, threshold {THRESH}: " + _cell(int((probs[m7] >= THRESH).sum()), int(m7.sum())) + "   <- the number to quote")
    print("  Class 7 recall by sub-reason (heuristic tags, approximate; a cell with fewer than 10 examples is not shown as a rate):")
    for src_name, src_title in (("human", "reason confirmed by a human"), ("heuristic", "heuristic tags, approximate"), ("generator", "reason set by the generator")):
        shown = False
        for reason in ("no_confirmation", "ineligible_state", "missing_step", "precondition_error", "parallel_write", "other", "unlabeled"):
            m = np.array([r["label"] == "incorrect" and r["error_class"] == 7 and r.get("class7_reason") == reason and r.get("class7_reason_source") == src_name for r in test_rows])
            if m.any():
                if not shown:
                    print(f"    {src_title}:")
                    shown = True
                print(f"      {reason:20s} " + _cell(int((probs[m] >= THRESH).sum()), int(m.sum())))
'''


def patch(path: Path):
    nb = json.load(open(path))
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    if any(MARK in "".join(c["source"]) for c in cells):
        print(f"{path.name}: already patched")
        return
    hit = [c for c in cells if ANCHOR in "".join(c["source"])]
    assert len(hit) == 1, f"{path.name}: {len(hit)} cells contain the anchor"
    src = "".join(hit[0]["source"])
    if OLD_START in src:                       # the first version of the block (human-first sub-reason table): replace it
        i = src.index(OLD_START)
        src = src[:i] + BLOCK + src[src.index(ANCHOR):]
    else:
        src = src.replace(ANCHOR, BLOCK + ANCHOR)
    hit[0]["source"] = src.splitlines(keepends=True)
    json.dump(nb, open(path, "w"), indent=1, ensure_ascii=False)
    print(f"{path.name}: patched")


if __name__ == "__main__":
    patch(NB / "phase5_modernbert.ipynb")
    patch(NB / "phase6_qwen.ipynb")
