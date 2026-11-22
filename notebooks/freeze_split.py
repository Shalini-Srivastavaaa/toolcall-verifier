"""Freeze the train/validation split in the Phase 5 and Phase 6 Colab notebooks.

StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42), first fold, grouped by (domain, task_id),
gives a different fold in scikit-learn <= 1.7.2 (6,892 train / 688 val on the 7,580 rows, what Colab runs)
than in scikit-learn >= 1.8.0 (6,816 / 764, the split D-012 documents). To make every environment
(Colab, Kaggle, local) train on the documented split, this script computes the validation groups with
scikit-learn >= 1.8 and writes them into the split cell of each notebook; the notebooks then no longer
call StratifiedGroupKFold.

    python notebooks/freeze_split.py        # rewrites the split cell of both notebooks (idempotent)
"""

import json
import re
import sys
from pathlib import Path

import sklearn
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent
FILES = ["train_pool", "bfcl_pool", "positives_class2_controls", "positives_class4_controls",
         "positives_class6_controls", "negatives_pool_final"]  # the notebooks' TRAIN_FILES order
BEGIN, END = "# --- frozen split (begin) ---", "# --- frozen split (end) ---"
EXPECTED = (7580, 6816, 764)

OLD_SPLIT = (
    "sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED)\n"
    "train_idx, val_idx = next(iter(sgkf.split(rows, y, groups)))\n"
)
OLD_ASSERT = re.compile(
    r"(# the grouped fold can differ slightly between scikit-learn versions, so the sizes are checked within a tolerance\n)?"
    r"assert 6500 <= len\(train_rows\).*\n"
)


def val_groups() -> list[str]:
    assert tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8), (
        f"scikit-learn {sklearn.__version__}: the documented split needs >= 1.8 (the fold differs in 1.7 and earlier)")
    rows = []
    for f in FILES:
        rows += [json.loads(line) for line in open(ROOT / "data" / "derived" / f"{f}.jsonl")]
    groups = [f"{r['domain']}::{r['task_id']}" for r in rows]
    y = [int(r["label"] == "incorrect") for r in rows]
    tr, va = next(iter(StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42).split(rows, y, groups)))
    assert (len(rows), len(tr), len(va)) == EXPECTED, (len(rows), len(tr), len(va))
    return sorted({groups[i] for i in va})


def split_block(vg: list[str]) -> str:
    listing = ",\n".join("    " + json.dumps(g, ensure_ascii=False) for g in vg)
    return (
        f"{BEGIN}\n"
        "# The grouped fold StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42) returns depends on the\n"
        "# scikit-learn version: 6,816 train / 764 val from 1.8 on (the split D-012 documents), 6,892 / 688 up to 1.7.\n"
        "# So the validation groups are frozen here (computed with scikit-learn 1.9.1 by notebooks/freeze_split.py) and\n"
        "# every environment trains on the same split. Every row of a (domain, task_id) group goes to the same side.\n"
        f"VAL_GROUPS = {{\n{listing},\n}}\n"
        "if DATASET == \"final\":  # the final dataset carries its split in every row (D-022)\n"
        "    train_idx = np.array([i for i, r in enumerate(rows) if r[\"split\"] == \"train\"])\n"
        "    val_idx = np.array([i for i, r in enumerate(rows) if r[\"split\"] == \"val\"])\n"
        "else:\n"
        "    assert VAL_GROUPS <= set(groups), \"frozen validation groups that are not in the data: the data changed, rerun freeze_split.py\"\n"
        "    val_idx = np.array([i for i, g in enumerate(groups) if g in VAL_GROUPS])\n"
        "    train_idx = np.array([i for i, g in enumerate(groups) if g not in VAL_GROUPS])\n"
        f"{END}\n"
    )


def patch(path: Path, block: str) -> None:
    nb = json.load(open(path))
    hits = 0
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if BEGIN in src:
            src = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END) + r"\n", lambda m: block, src, flags=re.S)
        elif OLD_SPLIT in src:
            src = src.replace(OLD_SPLIT, block)
        else:
            continue
        src, n = OLD_ASSERT.subn(
            "assert (len(rows), len(train_rows), len(val_rows)) == (7580, 6816, 764), (len(rows), len(train_rows), len(val_rows))  # the frozen split, D-012\n"
            "print(f\"split sizes: train {len(train_rows)} / val {len(val_rows)} (documented 6,816 / 764)\")\n",
            src)
        c["source"] = src.splitlines(keepends=True)
        hits += 1
        # drop the old "split sizes" print that followed the tolerance assert (Phase 6 style)
        src2 = "".join(c["source"])
        src2 = re.sub(r"\nprint\(f\"\\nsplit sizes: train \{len\(train_rows\)\} \(expected ~6,816\), val \{len\(val_rows\)\} \(expected ~764\)\"\)\n", "\n", src2)
        src2 = re.sub(r"print\(f\"split sizes: train \{len\(train_rows\)\} \(expected ~6,816\), val \{len\(val_rows\)\} \(expected ~764\)\"\)\n", "", src2)
        c["source"] = src2.splitlines(keepends=True)
    assert hits == 1, f"{path.name}: found {hits} split cells"
    json.dump(nb, open(path, "w"), indent=1, ensure_ascii=False)
    open(path, "a").write("\n")


if __name__ == "__main__":
    vg = val_groups()
    print(f"{len(vg)} validation groups (scikit-learn {sklearn.__version__})")
    for name in ("phase5_modernbert.ipynb", "phase6_qwen.ipynb"):
        patch(ROOT / "notebooks" / name, split_block(vg))
        print("patched", name)
