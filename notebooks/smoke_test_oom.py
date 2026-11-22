"""Local test of the memory-safe retry ladder of the Qwen notebook (D-029), with the toy model of the notebook's SMOKE path (CPU, a few rows).

    TCV_SMOKE=1 TCV_ENV=kaggle TCV_KAGGLE_INPUT=<dir with toolcall-verifier-data/final_*.jsonl> TCV_KAGGLE_WORKING=<empty dir> \
        python notebooks/smoke_test_oom.py [--fake-oom-above=1] [--set='OLD=>NEW' ...] [notebook.ipynb]

Runs the notebook's code cells 0-5 (configuration, imports, data, tokenizer, v2 input, split, training). With --fake-oom-above=N, `Trainer.training_step` raises
torch.cuda.OutOfMemoryError for every attempt whose micro-batch is larger than N, so the error comes from inside the training loop with the model and the batch on the
traceback, as a real one does. The test checks the property that the old loop violated: when an attempt starts, no model of an earlier attempt is still alive. It counts
live PeftModel objects (gc.get_objects(), which also lists cyclic garbage that has not been collected yet) right before each `get_peft_model` call. It tests the flag paths and the cleanup logic, not GPU memory. Needs torch, transformers, peft, datasets,
scikit-learn; the base tokenizer is downloaded.
"""

import gc
import json
import sys

import torch
import transformers
from peft import PeftModel

args = [a for a in sys.argv[1:] if not a.startswith("--")]
nb_path = args[0] if args else "notebooks/kaggle_phase6_qwen.ipynb"
overrides = [a[len("--set="):].split("=>") for a in sys.argv if a.startswith("--set=")]
fake = [int(a.split("=")[1]) for a in sys.argv if a.startswith("--fake-oom-above=")]

if fake:
    _orig = transformers.Trainer.training_step

    def _training_step(self, model, inputs, *a, **k):
        if self.args.per_device_train_batch_size > fake[0]:
            raise torch.cuda.OutOfMemoryError(f"simulated out of memory at micro-batch {self.args.per_device_train_batch_size} (test)")
        return _orig(self, model, inputs, *a, **k)

    transformers.Trainer.training_step = _training_step

nb = json.load(open(nb_path))
cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
ns = {"__name__": "__main__"}
live_before = []
for idx in range(0, 6):
    src = "".join(cells[idx]["source"])
    if idx == 0:
        for old, new in overrides:
            assert old in src, f"override not found in the configuration cell: {old!r}"
            src = src.replace(old, new.replace("\\n", "\n"), 1)
    src = "\n".join(("# " + l if l.lstrip().startswith(("!", "%")) else l) for l in src.split("\n"))
    print(f"\n######## code cell {idx} ########", flush=True)
    if idx == 5:  # probe: how many models are alive when a new one is built
        _gp = ns["get_peft_model"]

        def _probe(*a, **k):  # no gc.collect() here: whatever the notebook's own cleanup left behind must show up
            live_before.append(sum(isinstance(o, PeftModel) for o in gc.get_objects()))
            return _gp(*a, **k)

        ns["get_peft_model"] = _probe
    exec(compile(src, f"cell{idx}", "exec"), ns)

print("\nATTEMPTS:", json.dumps(ns.get("ATTEMPTS")))
print("live PeftModel objects right before each model build:", live_before)
assert live_before and all(n == 0 for n in live_before), "a model of an earlier attempt was still alive when the next one was built"
gc.collect()
left = sum(isinstance(o, PeftModel) for o in gc.get_objects())
print("live PeftModel objects after the training cell:", left)
assert left == 0
if fake:
    assert ns["ATTEMPTS"][0]["outcome"] == "out of memory" and ns["ATTEMPTS"][-1]["outcome"] == "ok", ns["ATTEMPTS"]
print("SMOKE TEST OK")
