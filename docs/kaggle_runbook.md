# Reproducing the training runs on Kaggle

How the final ModernBERT v3b model (test AUROC 0.771) and the Qwen2.5-1.5B comparison model (0.765) were trained and evaluated. Both use the notebooks `notebooks/kaggle_phase5_modernbert.ipynb` and `notebooks/kaggle_phase6_qwen.ipynb` with `DATASET = "final"`.

The training data is not published. Rebuilding it needs the original benchmarks and the review decision files, which are not part of this repository. The test set alone is published in `release/test_set/`.

## 1. Build and check the data

```bash
.venv/bin/python3 data/build_final_dataset.py --variant v3b --review-complete --write
.venv/bin/python3 data/check_final_dataset.py --dir data/derived/final_v3b    # must end with "ALL CHECKS PASSED"
.venv/bin/python3 data/package_for_kaggle.py --variant v3b
shasum -a 256 data/kaggle_upload_v3b/*.jsonl                                  # compare with decision log D-028
```

Upload the four files as a private Kaggle dataset named `toolcall-verifier-data-v3b` and attach only that dataset to a notebook. Rows: train 7,154, validation 749, test 363, synthetic test 374.

## 2. ModernBERT v3b

Accelerator: one T4. Internet: on. First code cell:

```python
DATASET = "final"
EVAL_ONLY = False
REAL_WEIGHT = 3
NORMALIZE_AGENT_TEXT = False
USE_BOTH_GPUS = False
BENCHMARK_STEPS = None            # set to 100 for a short timing run first
INCLUDE_POLICY_OVERRIDE = None    # policy text on
REAL_CV = False
```

Epochs 3, learning rate 2e-5, effective batch 16, 4,096 tokens. With `REAL_WEIGHT = 3` the notebook should report 10,348 rows per epoch.

Result of the final run (`docs/final_v3b_eval_report.txt`): about 1.9 hours; test AUROC 0.771, validation 0.927, synthetic test 0.756; recall 0.229 and 0.329 at 5% and 10% false alarms. A rerun will not match to the last digit because GPU training is not deterministic.

## 3. Qwen2.5-1.5B with LoRA

Same dataset and accelerator. First code cell:

```python
DATASET = "final"
EVAL_ONLY = False
REAL_WEIGHT = 3
QWEN_MAX_LEN = 4096
QWEN_EPOCHS = 2
MICRO_BATCH = 2                   # effective batch 16
GRADIENT_CHECKPOINTING = True     # required at 4,096 tokens
NORMALIZE_AGENT_TEXT = False
USE_BOTH_GPUS = False             # 4-bit models run on one GPU
BENCHMARK_STEPS = None
INCLUDE_POLICY_OVERRIDE = None
MODEL_PATH_OVERRIDE = None
```

The final run took about 6.8 hours. The notebook stops at an 11-hour time budget, saves the best adapter and can resume from its own output.

## 4. Evaluate a saved Qwen adapter

Attach the output of the training notebook and the same dataset, then set `EVAL_ONLY = True` and `EVAL_CHECKPOINT = None` (the finished model is found automatically), keeping `QWEN_MAX_LEN`, the policy text and `NORMALIZE_AGENT_TEXT` as in training. This takes about 15 minutes. A `score.weight | MISSING` load message is expected: the trained head is loaded with the adapter.

## 5. LLM judge baseline

Needs an OpenAI API key in `.env`.

```bash
set -a; . ./.env; set +a
.venv/bin/python3 eval/judge_new_test.py --cap-usd 0.80                   # cost estimate only
.venv/bin/python3 eval/judge_new_test.py --run --cap-usd 0.80             # judge the 363 test rows
.venv/bin/python3 eval/judge_new_test.py --report-only --with-synth       # report
```
