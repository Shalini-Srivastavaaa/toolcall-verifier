# Kaggle runbook: history of the earlier runs (archived)

> **Archived.** These are the steps of the intermediate runs (final_v1, REAL_CV, final_v2, normalization experiment, v3a) as they were written at the time, with the dataset sizes and checksums of those versions. The current steps are in [`../kaggle_runbook.md`](../kaggle_runbook.md). Kept for reference; the numbers in it are not the final ones.


The final dataset was rebuilt on 2026-11-21 with the real-data fix (D-026): `data/kaggle_upload/` holds `final_train.jsonl` (6,231 rows, 674 of them real-origin),
`final_val.jsonl` (749, 92 real-origin), `final_test.jsonl` (363) and `final_test_synth.jsonl` (374), 79.4 MB. **The test and test_synth files are byte-identical to the ones of
the first final run** (same SHA-256), so a new run is directly comparable to final_v1; only the training and validation files changed. Nothing in this runbook has been run on Kaggle yet; each step says what to look
for so that a surprise shows up early.

## 0. Rebuild and verify the data (only if you changed a decision or the code)

```bash
.venv/bin/python3 data/build_final_dataset.py --review-complete --write   # data/derived/final/*.jsonl (not in git; seeded, a rebuild is byte-identical)
.venv/bin/python3 data/check_final_dataset.py                             # must end with "ALL CHECKS PASSED" (leakage, duplicates, baselines)
.venv/bin/python3 data/package_for_kaggle.py                              # data/kaggle_upload/final_{train,val,test,test_synth}.jsonl
shasum -a 256 data/kaggle_upload/*.jsonl                                  # compare with the table in DECISIONS.md D-022
```

The SHA-256 of the four files today (D-026): train `391bd349...c39f`, val `1ebfb5d9...bb56`, test `f1c20d8e...fdee6` (unchanged), test_synth `fb2b869b...d779` (unchanged)
(full values in D-022). If they differ, the inputs changed: find out why before uploading.

## (a) Upload `data/kaggle_upload/` as a new version of `toolcall-verifier-data`

1. Open kaggle.com > **Datasets** > your **toolcall-verifier-data** dataset > **New Version**.
2. Drag in the four files from `data/kaggle_upload/` (select all four in Finder; they are `final_train.jsonl`, `final_val.jsonl`, `final_test.jsonl`,
   `final_test_synth.jsonl`; 79.4 MB together). Keep the seven older files in the dataset if you still want `DATASET = "legacy"` (the eval-only run in (d) uses
   `final`, so it does not need them); remove them only if you are sure.
3. Version notes: `final dataset 2026-11-21 (D-026, real-data fix): 6,231 / 749 / 363 / 374 rows; test files unchanged; SHA-256 in DECISIONS.md D-026`. Click **Create**; wait until the version shows
   "Ready" (a minute or two).
4. Check the upload: open the version's **Data** tab and compare the four file sizes with `ls -l data/kaggle_upload/` (64,406,684 / 6,601,566 / 4,468,166 /
   3,904,332 bytes). A different size means a partial upload: upload again.
5. Command-line alternative (not tested here): put a `dataset-metadata.json` with the dataset's id in `data/kaggle_upload/` and run
   `kaggle datasets version -p data/kaggle_upload -m "final dataset 2026-11-21 (D-026)"`. The web upload above is the tested-by-use path.

## (b) 100-step benchmark in an interactive session (about 15 minutes)

Purpose: measure seconds per step and the projected time of the full run before spending a Save & Run All session (D-020).

1. Kaggle > **Code** > **New Notebook** > **File > Import Notebook** > `notebooks/kaggle_phase5_modernbert.ipynb`.
2. Right panel: **Accelerator: GPU T4 x2**, **Internet: On**, **Persistence: off**; **Add Input** > your dataset `toolcall-verifier-data` (the new version).
3. First code cell, set exactly:

   ```python
   DATASET = "final"
   EVAL_ONLY = False
   BENCHMARK_STEPS = 100
   USE_BOTH_GPUS = False            # one T4; set True in a second benchmark only to compare
   GRADIENT_CHECKPOINTING = None    # the notebook tries without it first and switches it on after an out-of-memory error
   INCLUDE_POLICY_OVERRIDE = None   # None = POLICY block on in final mode
   ```

4. **Run All** (not Save & Run All). It trains 100 optimizer steps, runs one validation pass, prints the benchmark block and stops with a `SystemExit` message
   ("stopped after the benchmark, nothing was saved"); that message is expected.
5. **Send back** (copy as text, or take screenshots): (i) the lines of the data cells that say `final dataset from ...: 6985 rows` and `split sizes: train 6231 / val 749`;
   (ii) the `padding waste at loader batch ...` line; (iii) the whole block between the two `====` lines that starts with `BENCHMARK:` (micro-batch, accumulation, gradient
   checkpointing on or off, peak GPU memory, seconds per step, rows per second, validation pass seconds and `full run = ... h`); (iv) any `CUDA out of memory; retrying`
   line. For the comparison, repeat with `USE_BOTH_GPUS = True` and send the second block.
6. Decision rule: if `full run` is under about 8 hours, go to (c). If it is longer, send the block and stop; do not start the long run.

## (c) Final ModernBERT run with Save & Run All

1. In the same notebook set `BENCHMARK_STEPS = None` and keep `DATASET = "final"`, `EVAL_ONLY = False`, `USE_BOTH_GPUS = False`, `INCLUDE_POLICY_OVERRIDE = None`
   (`INCLUDE_POLICY` is then on in final mode; the data cell prints `INCLUDE_POLICY = True`). Keep `TIME_BUDGET_HOURS = 11.0`.
2. Settings as in (b): GPU T4 (one GPU is used), Internet on, the new dataset version attached.
3. **Save Version > Save & Run All (Commit)**. Training stops by itself at the time budget, saves a checkpoint and goes on to evaluation.
4. Output files (in the version's **Output** tab): `eval_report.txt`, `test_predictions.jsonl`, `test_synth_predictions.jsonl`, `train_info.json`,
   `checkpoints/phase5_v2/best`. **Send back** `eval_report.txt`, `train_info.json` and both predictions files.
5. Read the report with these in mind: the bar is the judge baseline of section 4 and the length-only baseline (AUROC 0.675 on the new test set); the headline
   Class 7 number is overall Class 7 recall (sub-reasons are heuristic tags, approximate); per-class cells with fewer than 10 rows print "too few to read".

## (d) Eval-only run of the earlier v2 checkpoint on the new test set

Purpose: isolate what the new data adds. Same 363 test rows, old model (trained on the old 7,580 rows, without the POLICY block).

1. **Attach the checkpoint:** Add Input > Notebook Output Files > the notebook version that trained v2 (its files appear as
   `/kaggle/input/<notebook-slug>/checkpoints/phase5_v2/best/`; open the Input panel to copy the exact path). If v2 was trained on Colab instead, upload the Drive
   folder `checkpoints/phase5_v2/best` (config.json and model.safetensors are enough) as a dataset and attach that.
2. New notebook from `kaggle_phase5_modernbert.ipynb`; inputs: the new data version and the checkpoint above; accelerator GPU T4 (CPU works but is slow for
   4,096-token inputs); Internet on (the base tokenizer is downloaded).
3. First code cell:

   ```python
   DATASET = "final"
   EVAL_ONLY = True
   EVAL_CHECKPOINT = "/kaggle/input/<notebook-slug>/checkpoints/phase5_v2/best"
   INCLUDE_POLICY_OVERRIDE = False   # v2 was trained without the POLICY block; with it on the comparison would measure the format change, not the data
   BENCHMARK_STEPS = None
   ```

4. Save & Run All. Keep `eval_report.txt` and `test_predictions.jsonl`.
5. **Use the CLEAN line only.** The old training data contained 1,309 rows of 37 tasks that are now test tasks, so the old model has seen those tasks. Locally, with the
   data built (`data/derived/final/test.jsonl` present):

   ```bash
   .venv/bin/python3 eval/score_old_model_on_new_test.py test_predictions.jsonl
   ```

   The new test set splits into **CLEAN rows: 242 (104 incorrect, 138 correct)** and OVERLAPPING rows: 121 (66 incorrect, 55 correct). Quote the CLEAN line (AUROC with a
   95% interval); ignore the overall and overlapping lines for the old model. The clean rows have, per class, 13 / 4 / 10 / 12 / 15 / 14 / 36 errors (classes 1 to 7), so most
   per-class cells are small. `test_synth` is not a held-out result for the old model at all (its task groups were in the old training data), so ignore that part too.
6. Compare with the new-data run of (c) on the same CLEAN rows (the same ids in `test_predictions.jsonl`).

## (e) REAL_CV: fine-tune the final_v1 checkpoint on the real rows with cross-validation (D-025; about 30 minutes)

Purpose: test whether real examples are what the model lacks (final_v1 trained on 16 real errors). Nothing is trained from scratch.

1. Attach as inputs the new data version and the **output of the final_v1 notebook version** (its `checkpoints/phase5_v2/best/` folder; copy the exact path from the Input panel). Accelerator GPU T4, Internet on.
2. Import the regenerated `notebooks/kaggle_phase5_modernbert.ipynb` and set the first code cell:

   ```python
   DATASET = "final"
   EVAL_ONLY = True
   EVAL_CHECKPOINT = "/kaggle/input/<notebook-slug>/checkpoints/phase5_v2/best"
   REAL_CV = True            # REAL_CV_FOLDS = 5, REAL_CV_EPOCHS = 3, REAL_CV_LR = 1e-5 are the defaults
   BENCHMARK_STEPS = None
   ```

3. Save & Run All. The evaluation cells run first (plain scores of the same checkpoint, now also `val_predictions.jsonl`), then the last cell prints one line per fold and the block that starts with `REAL_CV RESULT`.
4. **Send back:** `eval_report.txt` (it ends with the REAL_CV block), `real_cv_predictions.jsonl`, `val_predictions.jsonl`, `test_predictions.jsonl`.
5. How to read it: compare the **out-of-fold AUROC** with the plain AUROC (0.667) and with the **shortcut baselines on the same folds** printed below it (length only 0.667, length + domain 0.674 from `eval/diagnose_final_v1.py`). The 363 rows are no longer a held-out set in this run: they are a cross-validated estimate.

## (f) The next training run (final_v2): the real-data fix with REAL_WEIGHT = 3 (D-026; about 1.6 hours)

What changed against final_v1: the training file has 94 real errors (final_v1: 16) and 580 real correct rows, and the notebook can sample the real-origin rows more often. The test files
are unchanged. Run it after the upload in (a); it needs no benchmark if (b) was already run on the old data (the rows are about 12% more, the epoch with `REAL_WEIGHT = 3` is
about 36% longer).

1. Import the regenerated `notebooks/kaggle_phase5_modernbert.ipynb`; accelerator GPU T4 (one GPU is used), Internet on, the new dataset version attached.
2. First code cell, set exactly:

   ```python
   DATASET = "final"
   EVAL_ONLY = False
   REAL_WEIGHT = 3                  # the 674 real-origin training rows appear 3 times per epoch: 7,579 rows per epoch, 3 epochs
   USE_BOTH_GPUS = False            # one GPU
   BENCHMARK_STEPS = None
   INCLUDE_POLICY_OVERRIDE = None   # policy block on, as in final_v1
   REAL_CV = False                  # do not run REAL_CV in this run
   ```

   Epochs stay at the notebook's default of 3 (the `EPOCHS` constant of the training cell, not a flag); learning rate 2e-5 and the effective batch 16 are unchanged.
3. Check at the start of the run: the data cell prints `final dataset ...: 7717 rows` and `split sizes: train 6231 / val 749`; the training cell prints
   `REAL_WEIGHT = 3: 674 real-origin training rows of 6231; each appears 3 time(s) per epoch -> 7579 rows per epoch`. If either differs, stop.
4. Save & Run All. Expected time: final_v1 took 4,338 s for 16,707 samples (3.85 samples per second); 22,737 samples is about 5,900 s, about 1.7 hours with evaluation.
5. **Send back:** `eval_report.txt`, `train_info.json`, `test_predictions.jsonl`, `test_synth_predictions.jsonl` and **`val_predictions.jsonl`** (it now has source, origin and
   real_origin). The evaluation cell also prints the validation AUROC per source group and, separately, the **real-origin validation AUROC** (92 rows, 20 errors: the interval is wide).
6. How to read it: the test AUROC against final_v1 (0.667), the length-only baseline (0.675) and gpt-5-mini (0.719) on the same 363 rows; the real-origin validation AUROC against the
   synthetic groups. If `REAL_WEIGHT = 3` helps, a run with `REAL_WEIGHT = 1` on the same data separates the effect of the extra rows from the effect of the weighting.

## (g) The next experiment: NORMALIZE_AGENT_TEXT (final_v3; D-027; about 1.7 hours)

One change against final_v2 (the real-weighted run, test AUROC 0.732): the agent's text messages are unwrapped from JSON and cut to 200 characters in every split, so the input no longer shows the
JSON style and the verbosity of the gpt-5-nano agent of the induced runs. The data files are the ones of final_v2: **no new upload** (the dataset version of section (f) stays attached).

1. Import the regenerated `notebooks/kaggle_phase5_modernbert.ipynb`; accelerator GPU T4 (one GPU), Internet on, the dataset version of (f) attached.
2. First code cell, set exactly:

   ```python
   DATASET = "final"
   EVAL_ONLY = False
   REAL_WEIGHT = 3                  # as in final_v2
   NORMALIZE_AGENT_TEXT = True      # the one change
   AGENT_TEXT_CHARS = 200
   USE_BOTH_GPUS = False            # one GPU
   BENCHMARK_STEPS = None
   INCLUDE_POLICY_OVERRIDE = None   # policy block on, as in final_v2
   REAL_CV = False
   ```

   Epochs stay at the default 3; learning rate 2e-5, effective batch 16 as before.
3. Check at the start: the formatting cell prints `NORMALIZE_AGENT_TEXT = True: ...`; the data cell prints `final dataset ...: 7717 rows` and `split sizes: train 6231 / val 749`; the training cell prints `REAL_WEIGHT = 3: 674 real-origin training rows of 6231 ...
   -> 7579 rows per epoch`. The inputs are shorter than in final_v2 (the visibility check of the length cell shows the ladder firing on fewer rows), so the run should not take longer than final_v2's 1.7 hours.
4. Save & Run All. **Send back** `eval_report.txt`, `train_info.json`, `test_predictions.jsonl`, `test_synth_predictions.jsonl`, `val_predictions.jsonl`, and the model files if it is better (`model.safetensors`, `config.json`, `tokenizer.json`, `tokenizer_config.json`).
5. How to read it (compare with final_v2 on the same 363 rows, `eval/analyze_final_v2.py` gives the paired comparisons): (i) **induced false-flag at 0.5** (final_v2: 0.73) and the median score of induced
   correct rows (0.63); (ii) **natural-row AUROC** (final_v2: 0.715; gpt-5-mini 0.695) and the paired difference with final_v2; (iii) Class 7 recall and the telecom recall must not fall; (iv) the data cell prints `NORMALIZE_AGENT_TEXT = True: assistant text turns are unwrapped from JSON and cut to 200 characters in every split`; if it prints False the flag was not set.
   Expected (a judgment, not a measurement): the induced shift shrinks only a little (removing the cues from final_v2's inputs lowered induced scores by 0.04 to 0.07), natural-row AUROC stays within about 0.02.
   If it does, the shift is not about agent style and the next step is human-reviewed induced correct handlings (D-027, fix ii).

## (h) Dataset v3a: tool balancing (D-028; about 1.8 hours)

What changed against final_v2: 143 correct training rows were added, taken from successful (reward 1) tau2-bench simulations of train-split tasks, for the tools that had more training
errors than correct rows (`cancel_reservation` 6 to 23 correct against 21 errors, `exchange_delivered_order_items` 321 to 406, `modify_pending_order_items`, `modify_pending_order_payment` 0 to 6, ...).
They are real-origin rows (`origin: natural`, `source: tau2_success`). **Validation, test and test_synth are byte-identical to final_v2** (same SHA-256), so a v3a run is directly comparable to it.
The training file has 6,374 rows (817 real-origin); with `REAL_WEIGHT = 3` an epoch has 8,008 rows (final_v2: 7,579).

1. **Upload `data/kaggle_upload_v3a/` as its own dataset** (Kaggle > Datasets > New Dataset, title `toolcall-verifier-data-v3a`, the four files `final_train.jsonl`, `final_val.jsonl`, `final_test.jsonl`,
   `final_test_synth.jsonl`; 81.4 MB). A separate dataset, not a new version of `toolcall-verifier-data`, because the file names are the same as in the final_v2 data and the notebook takes the first
   `final_train.jsonl` it finds under `/kaggle/input`: **attach exactly one of the datasets to a notebook.** Check the sizes after the upload: 66,432,971 / 6,601,566 / 4,468,166 / 3,904,332 bytes (train, val, test, test_synth).
   SHA-256 (D-028): train `ed9efc2a...e6cc`, val `1ebfb5d9...bb56` (= final_v2), test `f1c20d8e...fdee6` (= final_v2), test_synth `fb2b869b...d779` (= final_v2).
2. Import the regenerated `notebooks/kaggle_phase5_modernbert.ipynb`; accelerator GPU T4 (one GPU), Internet on, **only** the v3a dataset attached.
3. First code cell, set exactly:

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

   Epochs stay at the default 3 (the `EPOCHS` constant of the training cell), learning rate 2e-5, effective batch 16.
4. Check at the start: `final dataset ...: 7860 rows`, `split sizes: train 6374 / val 749`, `REAL_WEIGHT = 3: 817 real-origin training rows of 6374; each appears 3 time(s) per epoch -> 8008 rows per epoch`. If a number differs the wrong dataset is attached: stop.
5. Save & Run All. Expected about 1.8 hours (final_v2: 22,737 samples; this run 24,024). **Send back** `eval_report.txt`, `train_info.json`, `test_predictions.jsonl`, `test_synth_predictions.jsonl`, `val_predictions.jsonl`.
6. How to read it (against final_v2 on the same 363 test rows and the same 749 validation rows; `eval/analyze_final_v2.py` gives the paired comparisons): natural-row AUROC (0.715) and its paired difference;
   recall at 5% and 10% false alarms on natural rows (0.05 and 0.15); the scores of the 14 correct `cancel_reservation` test rows (all at least 0.9 in final_v2); Class 7 recall (0.79); induced false-flag (0.73).

## (i) Dataset v3b: v3a plus 126 real errors from 520 newly localized train runs (D-028; about 2.3 hours)

What changed against v3a: 520 train-split failed runs (438 telecom, 82 retail) were localized with gpt-5-mini ($1.27), and their first-labeler proposals went through the D-026 gate (natural Classes 1, 4, 7;
read-only and line-probe rules applied; their earlier calls as correct rows). The training file has **7,154 rows: 220 real errors (v3a: 94; telecom 138, was 31; real Class 4 73, was 22) and 1,377 real correct rows**
(1,597 real-origin rows in all). **Validation, test and test_synth are byte-identical to final_v2 and v3a.** With `REAL_WEIGHT = 3` an epoch has 10,348 rows (v3a 8,008, final_v2 7,579): the run takes longer.

1. **Upload `data/kaggle_upload_v3b/` as its own dataset** (title `toolcall-verifier-data-v3b`; the four files; 85.9 MB). Check the sizes: 70,893,819 / 6,601,566 / 4,468,166 / 3,904,332 bytes (train, val, test, test_synth).
   SHA-256 (D-028): train `d55797e8...4dab`, val `1ebfb5d9...bb56`, test `f1c20d8e...fdee6`, test_synth `fb2b869b...d779` (the last three equal to final_v2). **Attach exactly one dataset to the notebook** (the file names are the same in all versions).
2. Import the regenerated `notebooks/kaggle_phase5_modernbert.ipynb`; accelerator GPU T4 (one GPU), Internet on, only the v3b dataset attached.
3. First code cell, set exactly as in (h):

   ```python
   DATASET = "final"
   EVAL_ONLY = False
   REAL_WEIGHT = 3
   NORMALIZE_AGENT_TEXT = False     # shelved (D-028)
   USE_BOTH_GPUS = False            # one GPU
   BENCHMARK_STEPS = None
   INCLUDE_POLICY_OVERRIDE = None
   REAL_CV = False
   ```

   Epochs 3 (the default), learning rate 2e-5, effective batch 16. If the time is a concern, `REAL_WEIGHT = 2` gives 8,751 rows per epoch (about 1.9 hours), but then the run is no longer comparable to (h) in its weighting.
4. Check at the start: `final dataset ...: 8640 rows`, `split sizes: train 7154 / val 749`, `REAL_WEIGHT = 3: 1597 real-origin training rows of 7154; each appears 3 time(s) per epoch -> 10348 rows per epoch`. A different number means the wrong dataset is attached: stop.
5. Save & Run All. Expected about 2.3 hours (31,044 samples at the 3.85 samples per second of final_v1; final_v2's own time was not recorded). **Send back** `eval_report.txt`, `train_info.json`, `test_predictions.jsonl`, `test_synth_predictions.jsonl`, `val_predictions.jsonl`.
6. How to read it (against final_v2 and the v3a run on the same 363 test rows and 749 validation rows): natural-row AUROC (0.715), **natural Class 4 recall** (2 of 16 at 0.5), **telecom recall at 0.5** (0.31) and telecom AUROC (0.727),
   Class 7 recall (0.79) and its false-flag, the induced false-flag (0.73). Run v3a and v3b with the same settings and compare them with each other: the difference is the 126 real errors and 654 real correct rows.

## (j) Qwen2.5-1.5B on the v3b data (D-029; about 6.3 hours for 2 epochs, plus about 15 minutes of evaluation)

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

## (k) Qwen2.5-1.5B: evaluate only, from the saved adapter of `qwen_v3b` Version 3 (D-029; about 15 minutes)

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

## 4. Judge baseline on the same test set

```bash
set -a; . ./.env; set +a                                                   # OPENAI_API_KEY
.venv/bin/python3 eval/judge_new_test.py --cap-usd 0.80                    # estimate only, no API call
.venv/bin/python3 eval/judge_new_test.py --run --cap-usd 0.80              # the 363 test rows, nano and mini
.venv/bin/python3 eval/judge_new_test.py --report-only --with-synth        # the report (add --run --with-synth to pay for test_synth)
```

The 102 `test_real` rows reuse the Phase 0 answers (same prompt); only the other rows are paid for. After (c), add `--compare test_predictions.jsonl` to print the model next to
the judges on the same rows. Results and costs are in D-023.
