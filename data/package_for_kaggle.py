"""Copy exactly the files DATASET = "final" needs into data/kaggle_upload/ (gitignored), to upload as a new version of the Kaggle dataset
`toolcall-verifier-data` (D-023).

    python data/package_for_kaggle.py                 # from data/derived/final/ (refuses if the final dataset has not been written)
    python data/package_for_kaggle.py --preview       # from the dry-run preview data/derived/final_preview/ (marked, for testing only)
    python data/package_for_kaggle.py --with-legacy   # also copy the 7 files DATASET = "legacy" needs
    python data/package_for_kaggle.py --variant v3a   # from data/derived/final_v3a/ into data/kaggle_upload_v3a/ (D-028; v3b likewise)

The notebooks look for final_train.jsonl, final_val.jsonl, final_test.jsonl and final_test_synth.jsonl anywhere under /kaggle/input/
(they search for final_train.jsonl), so the files are flattened and renamed final_<split>.jsonl. Nothing else is needed: the notebooks
take the split from each row and compute the baselines from the same files. The folder is emptied first, so it holds exactly this set.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

DER = Path(__file__).resolve().parent / "derived"
OUT = Path(__file__).resolve().parent / "kaggle_upload"
SPLITS = ("train", "val", "test", "test_synth")
LEGACY = ["train_pool", "bfcl_pool", "positives_class2_controls", "positives_class4_controls", "positives_class6_controls", "negatives_pool_final", "test_real"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--with-legacy", action="store_true")
    ap.add_argument("--variant", choices=["v2", "v3a", "v3b"], default="v2", help="dataset version (D-028): v2 = data/derived/final -> data/kaggle_upload/; v3a, v3b -> data/kaggle_upload_<variant>/")
    a = ap.parse_args()
    global OUT
    if a.variant != "v2":
        OUT = Path(__file__).resolve().parent / f"kaggle_upload_{a.variant}"
    src = DER / ("final_preview" if a.preview else ("final" if a.variant == "v2" else f"final_{a.variant}"))
    missing = [s for s in SPLITS if not (src / f"{s}.jsonl").exists()]
    if missing:
        sys.exit(f"{src} has no {missing}: the final dataset is written by `build_final_dataset.py --write --review-complete` once the review is done"
                 + ("" if a.preview else " (use --preview to package the dry-run output for a test)"))
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    rows = {}
    for s in SPLITS:
        shutil.copy2(src / f"{s}.jsonl", OUT / f"final_{s}.jsonl")
        rows[s] = [json.loads(line) for line in open(OUT / f"final_{s}.jsonl") if line.strip()]
    # the same two conditions the notebooks fail on
    empty = [s for s, r in rows.items() if not r]
    synth_test = [r["trajectory_id"] for r in rows["test"] if r.get("origin") == "synthetic" or str(r.get("source", "")).startswith("synthetic")]
    if empty or synth_test:
        shutil.rmtree(OUT)
        sys.exit(f"not packaged: empty splits {empty}, synthetic test rows {synth_test[:3]}")
    if a.with_legacy:
        for n in LEGACY:
            shutil.copy2(DER / f"{n}.jsonl", OUT / f"{n}.jsonl")
    if a.preview:
        (OUT / "PREVIEW_NOT_FINAL.txt").write_text("Packaged from the dry-run preview: the human review was not complete. Do not use for the final run.\n")
    print(f"packaged from {src} into {OUT}" + ("  [PREVIEW: not the final dataset]" if a.preview else ""))
    total = 0
    for p in sorted(OUT.iterdir()):
        size = p.stat().st_size
        total += size
        n = len(rows[p.stem[len('final_'):]]) if p.name.startswith("final_") else sum(1 for _ in open(p))
        print(f"   {p.name:36s} {size / 1e6:8.2f} MB {n:7d} rows" if p.suffix == ".jsonl" else f"   {p.name}")
    print(f"   {'total':36s} {total / 1e6:8.2f} MB")
    print("\nUpload: Kaggle > Datasets > toolcall-verifier-data > New version > add these files. Keep the old files in the new version if you still want DATASET = \"legacy\".")


if __name__ == "__main__":
    main()
