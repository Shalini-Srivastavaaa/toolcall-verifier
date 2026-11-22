"""Run the data cells of a Kaggle notebook locally on the packaged files, to check that the notebook loads, formats and splits exactly what was uploaded.

    mkdir -p /tmp/kin/toolcall-verifier-data /tmp/kwork && ln -s "$PWD"/data/kaggle_upload/*.jsonl /tmp/kin/toolcall-verifier-data/
    TCV_ENV=kaggle TCV_KAGGLE_INPUT=/tmp/kin TCV_KAGGLE_WORKING=/tmp/kwork .venv/bin/python notebooks/smoke_test_cells.py 0,2,3,4,5,6

Add --set='OLD=>NEW' (repeatable) to edit the configuration cell, for example --set='REAL_CV = False=>REAL_CV = True'. With TCV_SMOKE=1 the notebook
trains a random toy model on a few rows (the CPU smoke path of the notebooks), so the whole notebook runs on a laptop in a few minutes; it then needs torch.
The numbers are indexes into the notebook's CODE cells (0 configuration, 1 pip install: skip, 2 load, 3 tokenizer and v1 format, 4 policy flag, 5 v2 format and
lengths, 6 split); cells 7 and later need torch and a GPU. The notebook's own Kaggle-path overrides (TCV_ENV, TCV_KAGGLE_INPUT, TCV_KAGGLE_WORKING) point it at the
folder. Needs scikit-learn and transformers (tokenizer download, Internet).
"""

import json
import sys
import time

nb = json.load(open(sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--set") else "notebooks/kaggle_phase5_modernbert.ipynb"))
overrides = [a[len("--set="):].split("=>") for a in sys.argv if a.startswith("--set=")]   # e.g. --set='REAL_CV = False=>REAL_CV = True' (applied to the configuration cell)
cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
ns = {"__name__": "__main__"}
for idx in [int(x) for x in sys.argv[1].split(",")]:
    src = "".join(cells[idx]["source"])
    if idx == 0:
        for old, new in overrides:
            assert old in src, f"override not found in the configuration cell: {old!r}"
            src = src.replace(old, new, 1)
    src = "\n".join(("# " + l if l.lstrip().startswith(("!", "%")) else l) for l in src.split("\n"))
    print(f"\n######## code cell {idx} ########", flush=True)
    t = time.time()
    exec(compile(src, f"cell{idx}", "exec"), ns)
    print(f"######## cell {idx} ok in {time.time() - t:.1f}s", flush=True)
