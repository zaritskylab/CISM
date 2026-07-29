# CISM Tutorials

This folder contains the notebook workflow for running CISM from raw spatial single-cell data through motif selection, analysis, and export.

The tutorials are designed to be run in order. Tutorial 01 prepares a CISM/FANMOD-compatible dataset. Tutorial 02 runs FANMOD+, initializes CISM, tunes the discrimination stringency, and saves an analysis-ready artifact. Tutorial 03 and 03b both start from that saved artifact: tutorial 03 performs the main downstream analysis and visualization export, while tutorial 03b runs the multi-objective soft motif selection workflow.

## Tutorial Workflow

1. `01_data_preparation.ipynb`

   Prepare the input dataset. Use the section that matches your starting data: annotated centroid tables, annotated edge tables, or prebuilt `networkx` graphs. The notebook writes one graph text file per patient/FOV and creates or validates `patient_class.csv`.

2. `02_fanmod_and_cism_initialization.ipynb`

   Start from the prepared dataset folder. Resolve the FANMOD+ binary, initialize `CISM`, load the dataset, define the discriminator metadata, tune the stringency settings with Random Forest ROC AUC, and save a reusable pickle artifact.

3. `03_analysis_from_serialized_cism.ipynb`

   Load the artifact from tutorial 02. Rebuild the selected discriminative motif result, inspect motif recurrence and abundance, draw the top motifs, compare motif-induced pairwise structure to general pairwise structure, and export motif annotation and hit-table files.

4. `03b_soft_motif_selection.ipynb`

   Load the same tutorial 02 artifact and rank motifs with the multi-objective soft motif selection framework. This notebook optimizes additive motif-scoring weights, performs leakage-safe leave-one-patient-out validation, summarizes feature importance, visualizes top motifs, and bridges the selected motif IDs back into CISM analysis.

## Execution Guidelines

These guidelines are part of the intended workflow. Follow them before treating a tutorial run as complete.

### Prerequisites

- Use Python 3.9.
- Install the package requirements from the repository root:

  ```bash
  python -m pip install -r requirements.txt
  python -m pip install -e .
  ```

- Make sure FANMOD+ is available. The tutorials use the FANMOD+ binaries bundled under `cism/FANMOD_binaries/` when possible. If you compile FANMOD+ yourself, use a working C/C++ toolchain with CMake and Boost.

### Runtime Paths

Keep runtime paths explicit so results are portable across machines and reruns.

- `CISM_INPUT`: root folder for raw inputs, such as CSV files.
- `CISM_OUTPUT`: root folder for generated graph files, FANMOD outputs, caches, plots, and exported motif files.
- `FANMOD_PATH`: folder containing the FANMOD+ binary.

Use separate output and cache subfolders per run, especially when changing motif size, stringency, cell-type exclusions, or preprocessing rules.

### Input Data Contract

CISM ultimately consumes a dataset folder with graph files and a patient-class table:

```text
data/
  <dataset_name>/
    Patient_<patient_id>_FOV<fov>.txt
    Patient_<patient_id>_FOV<fov>.txt
    ...
    patient_class.csv
```

Each graph file must be a colored edge list:

```text
<src_id> <dst_id> <src_type_id> <dst_type_id>
```

The `patient_class.csv` file maps patient keys to class labels or continuous values. Patient keys should match the internal CISM convention:

```text
<dataset_name><patient_id>,<class_or_value>
```

For example:

```text
CRC1,POSITIVE
CRC2,NEGATIVE
```

or:

```text
CRC1,2612
CRC2,3822
```

### Core Parameters

- `motif_size`: number of nodes per enumerated motif. Use `4` as a balanced default for richer analysis, `3` for faster lightweight baseline runs, and `5` only when the richer pattern space is worth the higher runtime and sparser statistics.
- `shared_percentage`: main discrimination stringency parameter. Start around `0.3`, then increase it if motifs are too rare or noisy, or decrease it if too few motifs remain.
- `max_distance`: spatial adjacency radius. Set it in the same units as the input coordinates. For example, if coordinates are pixels at `0.5 um/pixel`, then a `50 um` radius corresponds to `max_distance = 100`.
- `iterations`: FANMOD+ randomization budget. Use `1000` for stable analysis runs and fewer only for quick sanity checks.
- `n_jobs`: parallelism for FANMOD/CISM work. Match this to available CPU cores.
- `force_run_fanmod` and `force_parse`: keep these `False` for normal cached reruns; switch them on only when upstream inputs or parsing assumptions changed.

### Bias And Exclusion Checks

Use `exclude_cell_type` only when there is a concrete biological or technical reason. Do not remove a cell type simply because it is predictive.

Before excluding a cell type:

- Define the biological compartment in advance. Exclude a population only if it is outside the intended biological question, such as removing tumor cells when studying the surrounding immune microenvironment.
- Check for group-exclusive or strongly imbalanced cell types. A cell type present mostly in one class can create redundant discriminative motifs driven by composition rather than higher-order spatial organization.
- Report cell-type composition before motif analysis. Compare cell-type frequencies and simple pairwise interactions across groups to identify likely confounders.
- Assess density after preprocessing. Compare local cell-density distributions between groups to check whether motif differences may reflect unequal sampling density.
- Document every exclusion explicitly. Report the excluded population, the biological rationale, the exact exclusion rule, and whether surrounding cells were also removed by spatial masking.

### Outputs To Track

The tutorials produce several classes of outputs:

- Prepared graph files: one colored edge-list text file per patient/FOV.
- FANMOD+ outputs: motif enumeration and statistics written to the configured output folder.
- Cache files: parsed FANMOD outputs and dataset-level motif tables that allow reruns without repeating FANMOD+.
- Serialized artifact: the reusable CISM/discriminator/config pickle saved by tutorial 02.
- Analysis exports: motif plots, pairwise comparison outputs, top-motif annotation files, and exact-hit CSV files.

Generated outputs, caches, and run-local artifacts should remain outside source control unless they are deliberately curated examples.

## Completion Checklist

Before moving from tutorial 01 to tutorial 02:

- The dataset folder contains one `Patient_<patient_id>_FOV<fov>.txt` file per FOV.
- Every graph row has exactly four values: `src_id dst_id src_type_id dst_type_id`.
- `patient_class.csv` exists and uses the same patient-key convention expected by CISM.
- Dataset validation passes without errors.

Before moving from tutorial 02 to downstream analysis:

- FANMOD+ was resolved correctly.
- `CISM(...)` was initialized with the intended motif size, paths, randomization budget, and cache/output folders.
- `cism.add_dataset(...)` completed successfully.
- The discriminator metadata matches the dataset labels and cell-type mapping.
- Stringency was selected using a documented criterion, such as Random Forest ROC AUC.
- The analysis-ready pickle artifact was saved.

Before reporting final biological conclusions:

- Composition and density checks have been reviewed.
- Any cell-type exclusion is documented.
- The selected motif set is stable enough for the intended claim.
- Exported motifs include interpretable node labels, edge lists, and per-patient counts.
