"""Visibility check for the first final run (D-025): with the POLICY block on and a 4,096-token budget, how often does the tightening ladder cut evidence (tool results)
in the model input, compared with the same rows without the policy block (what the earlier v2 run saw)? It runs the notebook's own formatting cells
(kaggle_phase5_modernbert.ipynb, code cells 0, 2, 3 and 4) twice on data/kaggle_upload/ and measures the same rows. Needs scikit-learn, numpy and transformers (the base
tokenizer is downloaded); no torch, no GPU, no API.

    .venv/bin/python3 eval/diagnose_visibility.py
"""

import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def load_notebook_namespace(override: str, kin: Path, kwork: Path, extra: dict | None = None) -> dict:
    """Run the notebook's data and formatting cells. `extra` maps a configuration line to its replacement, e.g. {"NORMALIZE_AGENT_TEXT = False": "NORMALIZE_AGENT_TEXT = True"}."""
    os.environ.update(TCV_ENV="kaggle", TCV_KAGGLE_INPUT=str(kin), TCV_KAGGLE_WORKING=str(kwork), TOKENIZERS_PARALLELISM="false")
    nb = json.load(open(ROOT / "notebooks" / "kaggle_phase5_modernbert.ipynb"))
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    ns = {"__name__": "__main__"}
    for idx in (0, 2, 3, 4):
        src = "".join(cells[idx]["source"])
        if idx == 0:
            assert "INCLUDE_POLICY_OVERRIDE = None" in src
            src = src.replace("INCLUDE_POLICY_OVERRIDE = None", f"INCLUDE_POLICY_OVERRIDE = {override}")
            for old, new in (extra or {}).items():
                assert old in src, f"configuration line not found: {old!r}"
                src = src.replace(old, new, 1)
        src = "\n".join(("# " + l if l.lstrip().startswith(("!", "%")) else l) for l in src.split("\n"))
        exec(compile(src, f"cell{idx}", "exec"), ns)
    return ns


def measure(ns, rows):
    out = []
    for r in rows:
        _, m = ns["format_input_v2"](r, meta=True)
        out.append(m)
    return out


def summarize(name, ms, rows):
    nat = np.array([m["natural_tokens"] for m in ms])
    sh = np.array([m["shown"] / m["total"] if m["total"] else np.nan for m in ms])
    has = ~np.isnan(sh)
    print(f"   {name:34s} rows {len(ms):5d} | natural tokens p50 {int(np.percentile(nat, 50)):5d} p90 {int(np.percentile(nat, 90)):5d} | ladder fired (natural > 4096) {int((nat > 4096).sum()):4d} "
          f"({(nat > 4096).mean():.1%}) | levels {dict(sorted(Counter(m['level'] for m in ms).items()))} | still over {sum(m['over_budget'] for m in ms)} | "
          f"tool-result chars visible: median {np.nanmedian(sh):.0%}, mean {np.nanmean(sh):.1%}, pooled {sum(m['shown'] for m in ms) / max(sum(m['total'] for m in ms), 1):.1%}; rows with a cut {int((sh[has] < 0.999).sum())}/{int(has.sum())}")


def main():
    tmp = Path(tempfile.mkdtemp())
    (tmp / "in" / "toolcall-verifier-data").mkdir(parents=True)
    for f in (ROOT / "data" / "kaggle_upload").glob("final_*.jsonl"):
        (tmp / "in" / "toolcall-verifier-data" / f.name).symlink_to(f)
    (tmp / "work").mkdir()
    res = {}
    for label, override in (("policy ON (the final run)", "True"), ("policy OFF (what the earlier v2 run saw)", "False")):
        ns = load_notebook_namespace(override, tmp / "in", tmp / "work")
        assert ns["INCLUDE_POLICY"] is (override == "True"), "the policy override did not take effect"
        final = ns["FINAL"]
        res[label] = {"test": (final["test"], measure(ns, final["test"])), "train": (final["train"], measure(ns, final["train"]))}
    on, off = list(res)
    print("=" * 100 + "\n4. VISIBILITY: the 4,096-token budget with and without the POLICY block (same rows)\n" + "=" * 100)
    for part in ("test", "train"):
        print(f"\n{part} split:")
        for label in res:
            rows, ms = res[label][part]
            summarize(label, ms, rows)
    # direct comparison on the same test rows
    rows, m_on = res[on]["test"]
    _, m_off = res[off]["test"]
    d_tok = np.array([a["natural_tokens"] - b["natural_tokens"] for a, b in zip(m_on, m_off)])
    has_policy = d_tok > 0
    print(f"\ntest rows that carry a policy block: {int(has_policy.sum())} of {len(rows)} ({has_policy.mean():.1%}); tokens it adds: median {int(np.median(d_tok[has_policy])) if has_policy.any() else 0}, max {int(d_tok.max())}")
    lv = np.array([a["level"] - b["level"] for a, b in zip(m_on, m_off)])
    print(f"rows where the policy pushed the ladder one or more levels further: {int((lv > 0).sum())} ({(lv > 0).mean():.1%}); rows where it fired only with the policy: "
          f"{int(sum(a['natural_tokens'] > 4096 and b['natural_tokens'] <= 4096 for a, b in zip(m_on, m_off)))}")
    sh_on = np.array([a["shown"] / a["total"] if a["total"] else np.nan for a in m_on])
    sh_off = np.array([b["shown"] / b["total"] if b["total"] else np.nan for b in m_off])
    lost = (sh_off - sh_on)
    lost = lost[~np.isnan(lost)]
    print(f"rows whose visible share of tool-result characters is lower with the policy: {int((lost > 0.001).sum())}; mean loss on those {lost[lost > 0.001].mean() if (lost > 0.001).any() else 0:.1%}; largest loss {lost.max():.1%}")
    for name, sel in (("incorrect rows", [r["label"] == "incorrect" for r in rows]), ("test_real rows (the earlier v2 run's 102 inputs)", [r.get("found_in") == "test_real" for r in rows])):
        sel = np.array(sel)
        print(f"   {name}: ladder fired with policy {int(sum(m['natural_tokens'] > 4096 for m, k in zip(m_on, sel) if k))} / without {int(sum(m['natural_tokens'] > 4096 for m, k in zip(m_off, sel) if k))} of {int(sel.sum())}; "
              f"mean visible share {np.nanmean(sh_on[sel]):.1%} with / {np.nanmean(sh_off[sel]):.1%} without")
    print("\nper class (incorrect rows), visible share of tool-result characters, policy on / off, and rows with a cut:")
    for k in range(1, 8):
        sel = np.array([r["label"] == "incorrect" and r.get("error_class") == k for r in rows])
        if sel.any():
            print(f"   class {k} (n={int(sel.sum())}): on {np.nanmean(sh_on[sel]):.1%} / off {np.nanmean(sh_off[sel]):.1%}; ladder fired on {int(sum(m['natural_tokens'] > 4096 for m, s in zip(m_on, sel) if s))} / {int(sum(m['natural_tokens'] > 4096 for m, s in zip(m_off, sel) if s))}")


if __name__ == "__main__":
    main()
