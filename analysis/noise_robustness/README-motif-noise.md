# Motif-induced distribution robustness to label noise

Run from any directory:



The runner uses `.venv-noise/bin/python` when present. Set `CISM_NOISE_PYTHON`
to use another Python with the packages in `requirements-motif-noise.txt`.
The historical exploratory script `motif_noise_summary.py` is preserved, but
the runner now uses `cism_noise_robustness.py` and its full-cache RF analysis.

All new results are under `motif_noise_summary_outputs/final_analysis/`.
The report is `motif_noise_final_report.pdf` (also available as Markdown).
Each figure is exported as PNG and vector PDF. The earlier report and its
approximate AUC calculations are not overwritten and should not be used as
the final RF results.

## Analysis

1. Discover every `output*/Melanoma/Motifs` condition and its
   `parse_cache/run_*` folders. The `.df_cache` files are CSV tables.
2. Read the full FANMOD tables, keep non-missing `Mean_Freq` and `p_value < .05`,
   and compare the same NN/NP patient cohort across runs.
3. For each threshold and each held-out patient, select motifs in the training
   cohort only. Match the historical CISM hard-discriminative selection:
   one-class significance, strict prevalence `> threshold`, and top 30 FOV
   rows per class sorted by frequency before motif-ID deduplication.
4. Use CISM's patient motif frequency, `sum(Count) / sum(Count/Freq)`, as the
   Random Forest feature. Fit 100 trees with seed 0. A threshold is eligible
   for RF AUC and optimum selection only if every fold has at least one feature.
   Neutral fallback predictions are diagnostic only and do not enter tuning.
5. Pool held-out predictions to obtain AUC at each threshold. Choose the
   maximum; ties select the lowest tested threshold. The chosen AUC is a
   tuning statistic, not a nested-CV generalization estimate.
6. For fixed .46 and the run's optimal threshold, take the union of motifs
   selected in the training folds. Weight node/undirected-edge multiplicities
   by motif occurrence counts within NN and NP, then normalize each vector.
7. Compare each distribution with raw run 0 under the same policy. Also
   compare both policies with the common raw/.46 baseline. Report total
   variation, `0.5 * sum(abs(P-Q))`, separately for NN and NP.
8. Plot trial distances and all within-condition trial-pair distances.
   Diamonds show distances of condition-mean distributions. Pairwise trial
   observations are dependent. Missing/empty distributions remain missing.
9. Report paired noise-minus-raw-retrial mean distances with bootstrap
   intervals conditional on this cohort and these trial IDs.

The raw reference is a declared original run, not the average of retrials.
The primary raw retrial box excludes the reference itself.
Motif-induced distributions count overlapping occurrences repeatedly; they
are not unique-cell/unique-edge tissue composition distributions.

## Faceted mean distributions

The report also includes one facet per condition, separately for fixed .46
and run-specific optimal stringency:

- `faceted_cell_distributions_<mode>.png`: NN/NP horizontal bars for all 15
  cell types, with the same fraction axis across conditions and policies.
- `faceted_pair_distributions_<mode>_<NN|NP>.png`: all 120 undirected pairs
  in the upper triangle, including same-type pairs. All conditions, classes,
  and policies share one probability scale; the lower triangle is masked.

Each mean is an equal-weight arithmetic mean of normalized run distributions,
not a pooled count distribution. The number of defined runs is printed in
each facet; undefined distributions are excluded rather than replaced by zero.
PNG and vector PDF versions are produced by `--report-only` from saved run
checkpoints, without repeating model fitting.

## Controls



By default the threshold grid matches/extends the earlier report within
.2-.8 and includes .45 and .46. Use `--thresholds` followed by space-separated
values to change it; .46 is always added for the fixed analysis. Every run's
settings, source signatures, fold selections, and held-out probabilities are
saved in `runs/<condition>/run_NNN.json`. Checkpoint reuse requires matching
settings and source size/mtime fingerprints. Changing the reference requires
only re-reporting, while changing the grid/forest settings requires refitting.

## Main tables

- `completion_inventory.csv`: requested and completed trials per condition.
- `run_metrics.csv`: selected threshold, AUC, motif count, featureless folds.
- `stringency_sweep.csv`: every tested threshold in every run.
- `auc_by_stringency_summary.csv`: mean/SD and eligible run count at every threshold.
- `average_distributions_by_condition.csv`: component means and run SDs.
- `condition_mean_distances_to_original.csv`: distances of mean distributions.
- `distances_to_original.csv`: trial distances under both reference policies.
- `within_condition_distances.csv`: every distinct within-condition trial pair.
- `distance_summary.csv`: mean/SD, median, and quartiles of reference distances.
- `noise_excess_distance_vs_raw_retrials.csv`: matched-trial effect estimates.
- `export_validation_mismatches.json`: .46 motif-ID discrepancies, if any.
- `software_versions.json`, `analysis_settings.json`: reproducibility metadata.
- `tied_optimal_thresholds_by_run.csv`: every threshold tied for each run's best AUC.
- `observed_graph_noise_rates.csv`: graph-node flip rates in run 0 where saved graphs exist.
- `distribution_availability_summary.csv`: fraction of trials with undefined class distributions.
- `undefined_distributions.csv`: exact class/trial/policy entries with zero motif-induced mass.
- `source_data_exclusions.json`: unusable trials with malformed/truncated source caches.

Malformed caches are not silently repaired, even when the final malformed row
has a nonsignificant p-value: the file may also lack later significant rows.
An exclusion checkpoint is reused only while its source fingerprint matches.
Restoring the source file changes the fingerprint and allows that trial to be
recomputed automatically. Completion inventories distinguish valid, excluded,
and pending trials.

Folder names are used as condition labels. The historical source overwrites
a matrix-noise object with prior-based noise; the actual historical mechanism
cannot be established from folder names alone. Keep the matrix-labelled
condition separate and interpret its mechanism cautiously.


Run `python -m analysis.noise_robustness.cism_noise_robustness --help` from the repository root for the available arguments. Scheduler and shell launchers are not distributed.
