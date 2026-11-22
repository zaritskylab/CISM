# TNBC reassessment with the updated permutation evaluator

The code and Slurm shell scripts are prepared for manual submission. Nothing is
submitted by creating or inspecting these files. From the repository root:



The default array runs 100 shuffles (10 tasks, 10 seeds per task). Task 0 also
recomputes the original. Aggregation starts only after every array task succeeds.
Scripts use the existing Anaconda `CISM` environment. Paths and resources can be
overridden with `REPO_ROOT`, `PYTHON_BIN`, `ARTIFACT`, `LABEL_FILE`, `OUTPUT_ROOT`,
`N_JOBS`, `BASE_SEED`, and `SEEDS_PER_JOB`. Aggregation must receive the same
`OUTPUT_ROOT` and `BASE_SEED`. For 1,000 permutations, use `--array=0-99` on the
first submission and `--export=ALL,EXPECTED_SHUFFLES=1000` on the second.
Resubmit incomplete task blocks to resume; include task 0 if the baseline is
missing. Keep code and inputs unchanged until all tasks finish.

Equivalent local commands, for reference only (these execute the analysis;
numpy, pandas, sklearn, joblib, and matplotlib are required):



Outputs default to `analysis/outputs/tnbc_label_permutation_updated/`. Completed
seeds resume only when the configuration fingerprint matches. Source files must
stay unchanged while a run is in progress. `--seed-start` and `--count` allow
separate processes to run disjoint seed blocks; `--n-jobs` controls fold workers.

## Inputs and cohort

- Saved artifact: `tutorials/runtime/serialized/TNBC_cism_ready.pkl`, 3-node
  motifs and 1,000 FANMOD randomizations. Reuse motif outputs; do not rerun FANMOD.
- Survival labels: `pet_data/TNBC/patient_class.csv`, checked against the saved
  artifact. Short-term survival means fewer than 1,000 days; long-term means at
  least 1,000 days. There are 37 patients: 7 short-term and 30 long-term.
- Raw cell data and graph files are available in `pet_data/`; they are not needed
  to rerun this patient-label test. Two FOVs belong to TNBC15 and receive one
  shared patient label.

## Evaluation

The adapter directly calls the updated `crc_label_permutation.evaluate` and
`prepare_data` functions. For each leave-one-patient-out fold, identify significant
motifs exclusive to one training class, require strict prevalence above the
candidate threshold, rank rows by frequency, take at most 30 rows per class,
and deduplicate motif IDs. Calculate motif frequencies across FOVs using the
common evaluator. A fold with no features receives probability 0.5 and remains
in the AUC calculation.

Reevaluate the three thresholds searched in the saved TNBC tutorial, in its
original trial order:

1. 0.4323220255890987
2. 0.4572784049558629
3. 0.4404145064107466

These were the first three random-startup Optuna proposals from seed 0. Every
shuffle searches these same candidates. The original fixed optimum is
0.4404145064107466. Do not substitute the CRC grid or fixed 0.55 into TNBC.

Use the updated evaluator's advancing forest RNG across candidates and folds,
100 trees, seed 0, and no SHAP. The saved tutorial reset the RNG for each Optuna
trial and used process workers; exact individual trial scores can therefore
differ. The original and every permutation here use the same updated RNG policy.
The optimized AUC and AUC at the saved fixed threshold are compared with their
respective recomputed original values.

For compatibility with the existing evaluator, encode long-term as its internal
`CLR` category and short-term as `DII`. The adapter restores survival labels in
saved predictions and writes `p_long_term_survival`. Reversing both the positive
label and probability gives the same binary AUC as the tutorial's short-term
positive convention. No CRC patients or CRC outcomes enter this analysis.

Permute one label per patient with `RandomState(seed)`, preserving the 7/30 split.
For both optimized and fixed AUC calculate the one-sided empirical p-value as
`(1 + number of shuffled AUCs >= recomputed original AUC) / (N + 1)`. Keep the
optimized and fixed tests separate. At N=100 the minimum possible p is 1/101.
The maximum over these thresholds is a search-adjusted permutation statistic;
it is not a nested-CV estimate of performance on new patients. This is a binary
survival-group analysis using the supplied labels, not a censoring-aware survival
model.

Each seed stores labels, all threshold AUCs, every held-out prediction, feature
counts, configuration hashes, and a summary written last. Aggregation checks
all expected seeds and their fingerprints and produces CSVs and PNG/PDF/SVG
histograms.

Validation:



The interrupted local execution from this editing session was stopped and its
partial files were moved to
`analysis/outputs/tnbc_label_permutation_updated_cancelled_local_run/`.
The default output directory is separate, so a manually submitted run will start
clean. No aggregate conclusions were generated from that interrupted run.


Run `python -m analysis.permutation_controls.tnbc_label_permutation --help` from the repository root for the available arguments. Scheduler and shell launchers are not distributed.
