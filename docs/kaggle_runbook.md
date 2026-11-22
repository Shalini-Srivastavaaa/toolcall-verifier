# Kaggle runbook (final setup)

Steps to reproduce the final runs on Kaggle: the ModernBERT v3b model (test AUROC 0.771), the Qwen2.5-1.5B comparison model (0.765) and their evaluation. Both use the notebooks `notebooks/kaggle_phase5_modernbert.ipynb` and
`notebooks/kaggle_phase6_qwen.ipynb` with `DATASET = "final"` on the **v3b dataset** (train 7,154 / validation 749 / test 363 / test_synth 374 rows). The training data is not published, so rebuilding it needs the raw benchmarks and the review
files of the project (section 0); the test set alone is published in `release/test_set/`. Settings of the final runs: one T4, 4,096 tokens, `REAL_WEIGHT = 3`, policy block on, ModernBERT 3 epochs (lr 2e-5), Qwen 2 epochs (lr 2e-4).
The steps of the earlier, intermediate runs are archived in [`archive/kaggle_runbook_history.md`](archive/kaggle_runbook_history.md).

## 0. Rebuild and verify the data (only if you have the raw data and the review files)

```bash
.venv/bin/python3 data/build_final_dataset.py --variant v3b --review-complete --write   # data/derived/final_v3b/*.jsonl (seeded; a rebuild is byte-identical)
.venv/bin/python3 data/check_final_dataset.py --dir data/derived/final_v3b              # must end with "ALL CHECKS PASSED" (leakage, duplicates, baselines)
.venv/bin/python3 data/package_for_kaggle.py --variant v3b                              # data/kaggle_upload_v3b/final_{train,val,test,test_synth}.jsonl
shasum -a 256 data/kaggle_upload_v3b/*.jsonl                                            # compare with the checksums in section 2 and in docs/decision_log.md D-028
```

Upload the four files as a **private Kaggle dataset** `toolcall-verifier-data-v3b` (Kaggle > Datasets > New Dataset); sizes 70,893,819 / 6,601,566 / 4,468,166 / 3,904,332 bytes (train, val, test, test_synth). Attach exactly one such dataset to a notebook (the file names are the same in every version).

## 1. 100-step benchmark in an interactive session (about 15 minutes; optional)

Purpose: measure seconds per step and the projected time before spending a Save & Run All session (D-020).

1. Kaggle > **Code** > **New Notebook** > **File > Import Notebook** > `notebooks/kaggle_phase5_modernbert.ipynb`; accelerator **GPU T4**, **Internet: On**, **Persistence: off**; **Add Input** > `toolcall-verifier-data-v3b`.
2. First code cell: `DATASET = "final"`, `EVAL_ONLY = False`, `REAL_WEIGHT = 3`, `BENCHMARK_STEPS = 100`, `USE_BOTH_GPUS = False`.
3. **Run All** (not Save & Run All). It trains 100 optimizer steps, runs one validation pass, prints the block between the two `====` lines that starts with `BENCHMARK:` (micro-batch, accumulation, gradient checkpointing, peak GPU memory, seconds per step, `full run = ... h`)
   and stops with a `SystemExit` message ("stopped after the benchmark, nothing was saved"); that message is expected.
4. If `full run` is under about 8 hours, continue with section 2. If it is longer, do not start the long run.


## 2. The final ModernBERT run on the v3b data (D-028, D-029; about 1.9 hours measured)

The v3b training file has **7,154 rows: 220 real errors and 1,377 real correct rows** (1,597 real-origin rows in all; telecom real errors 138, real Class 4 errors 73) plus synthetic and external rows; validation 749, test 363 and test_synth 374 rows.
With `REAL_WEIGHT = 3` an epoch has 10,348 rows. SHA-256 (D-028): train `d55797e8...4dab`, val `1ebfb5d9...bb56`, test `f1c20d8e...fdee6`, test_synth `fb2b869b...d779`.

1. Import `notebooks/kaggle_phase5_modernbert.ipynb`; accelerator **GPU T4** (one GPU), **Internet on**, **only** the v3b dataset attached.
2. First code cell, set exactly:

   ```python
   DATASET = "final"
   EVAL_ONLY = False
   REAL_WEIGHT = 3
   NORMALIZE_AGENT_TEXT = False     # shelved (D-028)
   USE_BOTH_GPUS = False            # one GPU
   BENCHMARK_STEPS = None
   INCLUDE_POLICY_OVERRIDE = None   # policy block on
   REAL_CV = False
   ```

   Epochs 3 (the default), learning rate 2e-5, effective batch 16, 4,096 tokens.
3. Check at the start: `final dataset ...: 8640 rows`, `split sizes: train 7154 / val 749`, `REAL_WEIGHT = 3: 1597 real-origin training rows of 7154; each appears 3 time(s) per epoch -> 10348 rows per epoch`. A different number means the wrong dataset is attached: stop.
4. **Save & Run All.** The measured time of the final run was 6,780 s (1.9 hours). **Send back** `eval_report.txt`, `train_info.json`, `test_predictions.jsonl`, `test_synth_predictions.jsonl`, `val_predictions.jsonl`.
5. What the final run gave (`docs/final_v3b_eval_report.txt`): test AUROC 0.771 (0.720-0.820 in the report, 1,000 resamples), validation AUROC 0.927, test_synth AUROC 0.756, recall at 5% and 10% false alarms 0.229 and 0.329. A rerun will not match to the last digit (GPU nondeterminism).

## 3. Qwen2.5-1.5B on the v3b data (D-029; the run took 6.8 hours for 2 epochs)

What changed in `notebooks/kaggle_phase6_qwen.ipynb`: the same v2 input as ModernBERT (with the policy block, counted in Qwen tokens, budget `QWEN_MAX_LEN = 4096`), `REAL_WEIGHT`, a new `QWEN_EPOCHS`, and the same final report (test, test_synth and validation predictions,
real-origin validation AUROC). **No new upload: attach the existing dataset `toolcall-verifier-data-v3b` (and only that one).** The time estimate is a FLOP model on one noisy calibration figure (D-029): the benchmark below replaces it with a measurement.

1. Import `notebooks/kaggle_phase6_qwen.ipynb` (File > Import Notebook). Right panel: **Accelerator: GPU T4** (one GPU is enough; the second one stays unused), **Internet: On** (the base model is downloaded), **Persistence: off**,
   **Add Input** > `toolcall-verifier-data-v3b` only.
2. First code cell, set exactly (this is also the setting for the benchmark, apart from the `BENCHMARK_STEPS` line):

   ```python
   DATASET = "final"
   EVAL_ONLY = False
   REAL_WEIGHT = 3
   QWEN_MAX_LEN = 4096
   QWEN_EPOCHS = 2                  # 3 only if the benchmark projects the 2-epoch run under about 5 hours (see below)
   MICRO_BATCH = 2                  # D-029: start here (accumulation 8, effective batch 16); the ladder only goes down: 2, then 1
   GRADIENT_CHECKPOINTING = True    # D-029: ON. Never False at 4,096 tokens (about 14 GiB of activations per row: out of memory even at batch 1)
   NORMALIZE_AGENT_TEXT = False     # shelved
   USE_BOTH_GPUS = False            # a 4-bit model cannot be wrapped in nn.DataParallel; the notebook hides the second GPU
   BENCHMARK_STEPS = 100            # the benchmark first; None for the real run
   INCLUDE_POLICY_OVERRIDE = None   # None = policy block on
   MODEL_PATH_OVERRIDE = None
   ```

3. **Benchmark first, in a FRESH interactive session (if a cell of this notebook already ran in the session, use Run > Restart session first; Run All, not Save & Run All; about 15-25 minutes including the 4-bit download).** It trains 100 optimizer steps, runs one validation pass, prints the block between the two `====` lines that starts
   with `BENCHMARK:` and stops with a `SystemExit` message ("stopped after the benchmark, nothing was saved"); that message is expected. **Send back:** (i) the data lines `final dataset from ...: 8640 rows` and `split sizes: train 7154 / val 749`, and
   the `REAL_WEIGHT = 3: 1597 real-origin training rows of 7154; each appears 3 time(s) per epoch -> 10348 rows per epoch` line (a different number means the wrong dataset is attached: stop); (ii) the whole `BENCHMARK:` block (micro-batch, accumulation,
   gradient checkpointing on or off, peak GPU memory, seconds per step, rows per second, validation pass seconds, `full run = ... h`); (iii) every `[GPU memory | ...]` line (they show GPU and host RAM before each attempt, after optimizer step 1 and 10, and after a failed attempt and its cleanup) and every `CUDA out of memory with micro-batch ...` line.
   **What to expect (D-029, arithmetic, not measured):** one attempt only (`attempt 1/2`: micro-batch 2, checkpointing on), peak GPU memory after step 1 about 5.5 GiB (a bad case 8-9 GiB) of 14.6; `n_train`, steps and the `BENCHMARK:` block as above.
   If peak memory after step 1 is over 11 GiB, or the first attempt runs out of memory, **stop and send the lines**: do not start the long run. If the cell ends **without any message** (no traceback, no `BENCHMARK:` block), look at the top of the notebook page and at the session
   panel for a message (for example "Your notebook tried to allocate more memory than is available. It has restarted."), copy it, restart the session and send it with the last `[GPU memory | ...]` line; with `MICRO_BATCH = 1` as the next try.
4. **Decision rule** (the `full run` figure is for the epochs set in `QWEN_EPOCHS`; the budget is 11 hours of which about 15 minutes go to the final evaluation):

   | projected `full run` with `QWEN_EPOCHS = 2` | what to do |
   |---|---|
   | under 5 hours | set `QWEN_EPOCHS = 3` (about 1.5 times as long, under 7.5 hours) |
   | 5 to 8 hours | keep 2 epochs |
   | 8 to 10 hours | keep 2 epochs but expect the time budget to bind: or set `REAL_WEIGHT = 2` (11.2 M instead of 13.0 M tokens per epoch) |
   | over 10 hours | do not start; send the block (alternatives: `REAL_WEIGHT = 2`, or `MODEL_PATH_OVERRIDE = "Qwen/Qwen2.5-0.5B"`, which changes the model) |

   The expected projection with 2 epochs is 6.3 hours (3 epochs 9.5 hours, which is why 3 is not the default).
5. **The real run:** set `BENCHMARK_STEPS = None` (and `QWEN_EPOCHS` as decided), **Save & Run All**. The notebook stops by itself at the 11-hour budget, saves the best adapter and can be continued from its own output (`restore_prior_run`), so a time-out loses nothing.
6. **Send back** `eval_report.txt`, `train_info.json` (it must say `real_weight: 3`, `n_train: 10348`, `n_real_origin_train: 1597`, `qwen_max_len: 4096`, `input_format: v2`, `gradient_checkpointing: true`, `micro_batch` (2, or 1 after a retry), `attempts` with each attempt's peak GPU memory, and `completed: true`), `test_predictions.jsonl`, `test_synth_predictions.jsonl`, `val_predictions.jsonl`, and the adapter folder (`best/`).
7. How to read it: against v3b ModernBERT on the same 363 test rows (AUROC 0.771, natural 0.753, induced 0.724; recall at 5% / 10% false alarms 0.229 / 0.329), with the paired bootstrap of `eval/analyze_final_v3b.py` adapted to the Qwen predictions. Remember the two models differ in weights, length of training and (here) epochs: a difference is not a clean comparison of the architectures.

## 4. Qwen2.5-1.5B: evaluate only, from a saved adapter (D-029; about 15 minutes)

Situation: Version 3 of the Kaggle notebook `qwen_v3b` trained for 2 epochs (6.85 hours, validation AUROC 0.971 then 0.982) and saved the adapter to `checkpoints/phase6/best/`, then crashed at the start of the final evaluation
(`'Tee' object has no attribute 'isatty'`, fixed in the notebook; log in `docs/qwen_v3b_run_log.txt`). Nothing is trained again: the evaluation loads the 4-bit base model, puts the saved adapter and its trained `score` head on top
of it, scores the test, test_synth and validation rows and writes the report. **The adapter exists only in the output of that version: do not delete it, and download `best/` from the Output tab of Version 3 as a copy (about 20 MB with the tokenizer; keep it in `models/final_qwen_v3b/`, gitignored).**

1. **A new notebook, not the old one.** Kaggle > **Code** > **New Notebook** > **File > Import Notebook** > `notebooks/kaggle_phase6_qwen.ipynb` (the regenerated one of commit `D-029 eval-only for Qwen`); name it `qwen_v3b_eval`.
   Reason: the notebook that holds the output (`qwen_v3b`) should stay untouched, and I cannot verify that Kaggle lets a notebook take its own earlier version as an input.
2. Right panel: **Accelerator: GPU T4** (one is enough; a second one stays unused), **Internet: On** (the base model `Qwen/Qwen2.5-1.5B` is downloaded, about 3 GB, and `pip` runs), **Persistence: off**.
3. **Add Input** (exactly these two): (a) **Your Work > Notebooks > `qwen_v3b` > Version 3** (its output holds `checkpoints/phase6/best/`); (b) **Datasets > Your datasets > `toolcall-verifier-data-v3b`** (the same data as the training run: 8,640 rows).
   If Version 3 is not listed under Your Work, use the search box with the notebook's name and pick "Version 3" explicitly (Kaggle attaches the latest version by default; the latest may be a later failed one without `best/`).
4. First code cell, set exactly (everything that shapes the input must equal the training run: `DATASET`, `QWEN_MAX_LEN`, the policy block, `NORMALIZE_AGENT_TEXT`):

   ```python
   DATASET = "final"
   EVAL_ONLY = True                 # no training, no tokenizing of the training rows
   EVAL_CHECKPOINT = None           # None = auto-detect: the finished best/ of the attached notebook output (restore_prior_run); or the exact path of the best folder
   QWEN_MAX_LEN = 4096              # as in the training run
   NORMALIZE_AGENT_TEXT = False     # as in the training run
   INCLUDE_POLICY_OVERRIDE = None   # None = policy block on, as in the training run
   REAL_WEIGHT = 3                  # not used without training (kept equal to the run)
   USE_BOTH_GPUS = False
   BENCHMARK_STEPS = None
   MODEL_PATH_OVERRIDE = None
   ```

   (`MICRO_BATCH`, `GRADIENT_CHECKPOINTING` and `QWEN_EPOCHS` are not used in this mode.)
5. **Save & Run All** (Version > Save Version > Save & Run All (Commit)). **Check in the log within the first few minutes:**
   `final dataset ...: 8640 rows` and `split sizes: train 7154 / val 749`; `EVAL_ONLY: training data is not tokenized`; `EVAL_ONLY = True: skipping training`;
   **`found a finished model from an earlier run: /kaggle/input/.../checkpoints/phase6/best`** (the path of the attached notebook output) and `evaluation report | ... | checkpoint <the same path>`.
   Expected and harmless: a `LOAD REPORT` with `score.weight | MISSING` (the new base model has no head; the saved adapter brings the trained one right after; this was the line that crashed through the Tee).
   If it stops with **`nothing to evaluate ... attach the notebook that trained it`**: the notebook output is not attached (the message lists what is under `/kaggle/input`); attach it, or set `EVAL_CHECKPOINT = "<path of the best folder>"` (in the right panel under Input, hover the `best` folder and copy its path).
   If it says `several finished models are attached`, it uses the newest and lists all: attach only Version 3.
6. **Expected time: about 15 minutes** (12 to 25): notebook start, pip and data about 1.5 minutes (the 110 seconds of tokenizing the training rows are skipped); base model download and 4-bit load 1 to 3 minutes; scoring 1.8 M tokens (test 363, test_synth 374, validation 749 rows) about 8.5 minutes
   (from the measured validation pass of the run: 168 seconds for 0.59 M tokens); bootstraps and report 1 to 2 minutes. Saving the version adds a few minutes. Far more than 30 minutes means something is wrong: stop and send the log.
7. **Send back:** `eval_report.txt` (in the output root), `checkpoints/phase6/test_predictions.jsonl`, `test_synth_predictions.jsonl`, `val_predictions.jsonl`, and the full log (the Logs tab, download); from Version 3's output also `checkpoints/phase6/best/train_info.json` (training time, log history, attempts, peak GPU memory) if it is not in the repository yet.
   Then the same comparison as for ModernBERT v3b (paired bootstrap against v3b, gpt-5-mini and the baselines on the same 363 test rows; `eval/analyze_final_v3b.py` adapted), D-029 part C.

## 5. Judge baseline on the same test set (needs an OpenAI key)

```bash
set -a; . ./.env; set +a                                                   # OPENAI_API_KEY
.venv/bin/python3 eval/judge_new_test.py --cap-usd 0.80                    # estimate only, no API call
.venv/bin/python3 eval/judge_new_test.py --run --cap-usd 0.80              # the 363 test rows, nano and mini
.venv/bin/python3 eval/judge_new_test.py --report-only --with-synth        # the report (add --run --with-synth to pay for test_synth)
```

The 102 `test_real` rows reuse the Phase 0 answers (same prompt); only the other rows are paid for. After (c), add `--compare test_predictions.jsonl` to print the model next to
the judges on the same rows. Results and costs are in D-023.
