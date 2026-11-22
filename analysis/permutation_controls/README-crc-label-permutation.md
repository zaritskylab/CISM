# CRC patient-label permutation experiment

From the repository root, submit 100 shuffles (10 tasks, 10 shuffles per task),
then schedule aggregation only when every task succeeds:



The files are prepared for submission; no jobs are submitted by generating them.
Task 0 also runs the unshuffled baseline. Run this task when resubmitting a subset
if the original baseline has not yet completed.

For 1,000 shuffles, use `--array=0-99` on the first command and
`--export=ALL,EXPECTED_SHUFFLES=1000` on the aggregation command.
Keep `SEEDS_PER_JOB=10`. A task runs its seeds sequentially and parallelizes
LOOCV folds, with each forest using one CPU. No FANMOD runs are repeated.

## Statistical procedure

- Reuse `/home/omriavit/research/cism_CRC_4.pkl` (4 nodes, 1,000 FANMOD iterations).
  Load labels from `/groups/assafza_group/assafza/Datasets/CRC/CRC_patient_class.csv`.
- Permute one label per patient with `RandomState(seed)`, preserving the
  17 CLR / 18 DII split. All FOVs for a patient retain that patient's label.
- For each shuffle, perform leave-one-patient-out classification at all 12
  thresholds used in the original notebook:
  `0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.70 0.75 0.80`.
  "Optimal" means the largest LOOCV AUC on this grid; ties use the lowest
  stringency. No held-out label is used to select that fold's motifs.
- Match CISM's hard-discriminative selection: significant motif rows (`p<0.05`),
  IDs significant exclusively in one training class, strict patient prevalence
  `> stringency`, rank by row `Freq`, take 30 rows per class, then deduplicate IDs.
  This row cap preserves the notebook's implementation; it is not a cap of 30
  unique motif IDs.
- Compute feature values from significant rows as `sum(Count) / sum(Count/Freq)`
  over distinct FOVs, filling missing patient/motif values with zero.
  Random forest: 100 trees, sklearn defaults. Each original/shuffled sweep starts
  with `RandomState(0)` and advances it across fits in threshold-major,
  patient-major order, matching the historical notebook. Forest RNG states are
  reserved serially before parallel fitting, so worker scheduling does not alter
  results. Identical feature sets are refit at each threshold, as in the notebook.
  No SHAP is calculated.
- A fold with no selected motifs contributes probability 0.5 and remains in AUC
  computation; it is never silently dropped. Counts of such folds are recorded.
- Also evaluate each shuffle at the notebook's original fixed optimum, **0.55**.
  This is evaluated even if a custom search grid excludes 0.55, without adding
  that threshold to the optimization search.
- Recompute the unshuffled original using the historical RNG progression.
  The target notebook result is AUC 0.9199346 at 0.55. Histograms use the actual
  reproduced baseline, never a manually substituted AUC. The fixed test uses
  that same sweep's score at 0.55. The original and every shuffle use the same
  RNG policy; permutations from the older per-fit-seed run cannot be mixed in.
- Compute separate one-sided empirical AUC p-values for optimized and fixed
  evaluations: `(1 + number of shuffled AUCs >= original AUC) / (N + 1)`.
  Optimize the original and every permutation over the same grid. This tests
  the complete selection/search procedure; the optimized AUC is not a nested-CV
  estimate of prediction performance on new patients.

## Outputs and restarting

Default root: `analysis/outputs/crc_label_permutation_historical/`.
Each `shuffled/seed_N/` contains permuted patient labels, configuration,
all threshold AUCs, all held-out predictions and feature counts, and a summary.
`original/` contains the matched unshuffled run.
Aggregation creates three histograms in PNG/PDF/SVG:

1. Optimized AUC (blue).
2. Optimal stringency (green).
3. AUC at fixed 0.55 (darker blue).

Original values are vertical red lines. CSVs include every shuffle's optimized
AUC, optimal stringency, fixed AUC, no-feature-fold counts, and empirical p-values.
Stringency is descriptive; no directional p-value is assigned to it.

Completed shuffles are skipped only if their configuration fingerprint matches.
The fingerprint includes source hashes, code hash, search grid, model settings,
and library versions. Use a new `OUTPUT_ROOT` if these change. A summary is the
completion marker, written atomically after predictions and sweeps. Incomplete
seeds rerun; a failed seed does not prevent attempts at the rest of its block.
Aggregation rejects missing seeds and mismatched configurations. Submit the same
array again to resume, or `--array=<failed task IDs>` to repeat selected blocks.
Keep the code and source files unchanged while the array is running.

Quick local execution (these perform real analysis):



The former per-fit-seed experiment remains in `analysis/outputs/crc_label_permutation/` for traceability. The updated scripts default to a separate historical-RNG output root; rerun all permutations before interpreting the updated figure. `--rng-mode per-fit` is available only to reproduce the former evaluator.


Run `python -m analysis.permutation_controls.crc_label_permutation --help` from the repository root for the available arguments. Scheduler and shell launchers are not distributed.
