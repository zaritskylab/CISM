# CISM parameter guidelines

This guide is for anyone running the CISM pipeline on their own data. It explains the parameters you set, what
values to start from, and how to adjust them for your dataset and biological question. It expands the "Parameter
selection and optimization guidelines" section of the CISM paper's Methods with the exact function arguments, code
examples, bias checks and computational settings.

For a step-by-step walkthrough of the pipeline itself, see the [tutorial README](tutorial/README_tutorial.md) and the
notebooks in [tutorial/](tutorial/).

**Contents**

1. [Where each parameter is set](#where-each-parameter-is-set)
2. [Quick reference](#quick-reference)
3. [A minimal run](#a-minimal-run)
4. [Motif size](#motif-size)
5. [Discriminative-motif stringency](#discriminative-motif-stringency)
6. [Maximum spatial distance](#maximum-spatial-distance)
7. [Cell-type exclusion](#cell-type-exclusion)
8. [Cell-type annotations and consolidation](#cell-type-annotations-and-consolidation)
9. [Bias assessment before motif analysis](#bias-assessment-before-motif-analysis)
10. [Other implementation parameters](#other-implementation-parameters)
11. [Caching and reruns](#caching-and-reruns)
12. [Computational configuration](#computational-configuration)
13. [What to report](#what-to-report)

## Where each parameter is set

CISM runs in three stages, and each main parameter belongs to one of them:

| Stage | Function | Main parameters |
|---|---|---|
| 1. Build cell graphs | `prepare_from_centroids(...)` | `max_distance`, `cell_type_mapper`, `exclude_cell_type`, `removed_cluster_alpha`, `removed_cluster_buffer` |
| 2. Enumerate motifs with FANMOD+ | `CISM(...)`, `cism.add_dataset(...)` | `motif_size`, `iterations`, `p_value`, `n_jobs` |
| 3. Select discriminative motifs and classify | `HardDiscriminativeFC(...)`, `analyze_motifs(...)`, `tune_stringency(...)` | `shared_percentage` (stringency), `max_class_features`, `labels` |

Changing a stage 1 parameter means building new graphs and rerunning FANMOD+, which can take hours (see
[Caching and reruns](#caching-and-reruns)). Stage 3 parameters reuse the motifs already enumerated, so they are
cheap to change.

## What parameters we will dive into?

| Parameter | Start with | Adjust when |
|---|---|---|
| `motif_size` | `4` | Use `3` for faster baseline runs, `5` for richer patterns when data and compute allow |
| `shared_percentage` (stringency) | `0.3` | Raise it if many motifs pass, lower it if few or none pass |
| `max_class_features` | `30` | Caps how many motifs per class become classifier features |
| `max_distance` | Your chosen physical distance, in coordinate units | Convert from microns when coordinates are in pixels |
| `exclude_cell_type` | `None` | Only with a biological or technical reason defined in advance |
| `cell_type_mapper` | `None` (keep your annotations) | Consolidate related cell types when motifs are too sparse |
| `iterations` | `1000` | Lower only for quick test runs |
| `p_value` | `0.05` | Usually left unchanged |
| `n_jobs` | Number of free CPU cores | Lower it if memory runs out, especially for 5-node motifs |

## A minimal run

The example below goes from a cell table to a cross-validated classifier, with every main parameter visible. It
assumes you run from the repository root, that `cells.csv` has one row per cell, and that the patient classes are in a
headerless `patient_class.csv` (format described in the [main README](README.md#patient_classcsv-format)).

```python
from pathlib import Path

import pandas as pd

from cism import (
    CISM,
    DiscriminativeFeatureKey,
    HardDiscriminativeFC,
    TissueStateDiscriminativeMotifs,
    prepare_from_centroids,
)

DATA_ROOT = Path("data")                 # holds one folder per dataset
DATASET = "MyCohort"                     # dataset folder, and the prefix of every patient key
LABELS = ["responder", "non_responder"]  # the two classes to compare, without spaces

# Stage 1: one graph file per patient and FOV
cells = pd.read_csv("cells.csv")
prep = prepare_from_centroids(
    dataframe=cells,
    path_to_output_dir=str(DATA_ROOT / DATASET),
    column_mapping={              # CISM column name -> your column name
        "patient_id": "patient",
        "fov": "fov",
        "cell_type": "cell_type",
        "centroid-0": "x",
        "centroid-1": "y",
    },
    cell_type_mapper=None,        # or {raw label: consolidated label}
    max_distance=100,             # in the units of x and y
    exclude_cell_type=None,
)
common_cells_type = {idx: name for name, idx in prep.cell_type_to_id.items()}
# Put patient_class.csv in DATA_ROOT / DATASET, with rows such as "MyCohort7,responder".

# Stage 2: motif enumeration with FANMOD+
cism = CISM(
    fanmod_path="cism/FANMOD_binaries",
    fanmod_exe="LocalFANMOD",
    network_dataset_root_path=str(DATA_ROOT),
    fanmod_output_root_path="runtime/fanmod_output",
    fanmod_cache_root_path="runtime/fanmod_cache",
    motif_size=4,
    iterations=1000,
)
cism.add_dataset(
    dataset_folder=DATASET,
    dataset_type="Disease",
    dataset_name=DATASET,         # must match the key prefix in patient_class.csv
    n_jobs=4,
    require_patient_class=True,
)

# Stage 3: discriminative motifs and leave-one-patient-out classification
discriminator = TissueStateDiscriminativeMotifs(
    cism=cism,
    tissue_state_csv_path=str(DATA_ROOT / DATASET / "patient_class.csv"),
    tissue_state_to_string={label: label for label in LABELS},  # must cover every class in the csv
    common_cells_type=common_cells_type,
)
feature_conf = HardDiscriminativeFC(
    labels=LABELS,
    extract_by=DiscriminativeFeatureKey.STRUCTURE_AND_CELL_IDENTITIES,
    use_cells_type_composition=False,
    use_motifs=True,
    shared_percentage=0.3,        # stringency
    max_class_features=30,
)
result = discriminator.analyze_motifs(
    feature_conf=feature_conf, exclude_patients=[], n_jobs=1, prefer="threads"
)
print(f"Leave-one-patient-out ROC AUC: {result.get_roc_auc_score():.3f}")
```

The sections below explain each of these choices. The later code snippets reuse the names defined here (`cells`,
`prep`, `cism`, `discriminator`, `feature_conf`, `DATA_ROOT`, `DATASET`, `LABELS`).

## Motif size

`motif_size` (in `CISM(...)`) is the number of cells in each motif that FANMOD+ enumerates. The scripts in
[tutorial/](tutorial/) accept 3, 4 or 5.

We recommend a motif size of **4** as a default. It balances structural complexity, motif abundance and computational
cost. Use **3** for faster baseline analyses. Consider **5** when richer spatial patterns are needed and you have
enough patients, FOVs and computing resources.

Why the size matters:

- **Structure.** There are 2 connected shapes with 3 cells (a path and a triangle), 6 with 4 cells and 20 planar ones
  with 5 cells. Size 3 motifs mostly describe which cell types sit together; size 4 and 5 motifs also describe how they
  are arranged.
- **Sparsity.** Each shape can be labeled with cell types in many ways. With 16 cell types there are about 3,000
  possible 3-node motifs, 112,000 possible 4-node motifs and 6.8 million possible 5-node motifs. The larger the motif,
  the fewer patients share any given one, so fewer motifs pass the stringency threshold.
- **Cost.** FANMOD+ runtime and memory grow steeply with motif size (see
  [Computational configuration](#computational-configuration)).

FANMOD+ results are stored per motif size, so you can run several sizes on the same graphs. Choose the stringency
separately for each size, because the number of discriminative motifs changes with size.

## Discriminative-motif stringency

The stringency parameter, `shared_percentage` in `HardDiscriminativeFC`, controls which motifs count as
discriminative. A motif is selected for a disease state when:

1. FANMOD+ finds it significant (p < 0.05) in at least one FOV of a patient in that state,
2. it is not significant in any patient of the other state, and
3. the fraction of that state's patients in which it is significant is **greater than** `shared_percentage`.

The motifs that pass are capped by `max_class_features` (below) and become the features of a random forest
classifier. `analyze_motifs` repeats this selection inside every leave-one-patient-out fold using only the training
patients, so the held-out patient never influences which motifs are chosen.

**Start with 0.3**, then adjust it according to the number of discriminative motifs it produces:

- If one class gets no motifs, or `analyze_motifs` raises `No discriminative features were selected for the current
  stringency`, lower the stringency.
- If hundreds of motifs pass and most are carried by only a few patients, raise it.

Think of the threshold in patients, not only as a fraction. A motif must be carried by more than
`shared_percentage × n` patients of a class with `n` patients, so the effective threshold moves in steps of `1/n`. In
a class of 7 patients, 0.3 requires at least 3 patients; in a class of 30 it requires at least 10. When the classes
differ in size, the same stringency corresponds to different patient counts in each class.

### Count discriminative motifs across stringencies

`discover` lists every class-exclusive motif in the full cohort together with its patient fraction. Use it to see how
many motifs each class keeps at each stringency before you commit to a value:

```python
discover_result = discriminator.discover(
    extract_by=DiscriminativeFeatureKey.STRUCTURE_AND_CELL_IDENTITIES,
    classes=LABELS,
)
discover_result.plot_number_of_motifs_versus_discrimination_stringency_parameter(
    class_to_color={LABELS[0]: "tab:blue", LABELS[1]: "tab:orange"}
)

motifs = (
    discover_result.get_discriminative_motifs()
    .groupby("ID", observed=True)
    .agg(patient_class=("patient_class", "first"),
         patient_fraction=("patient_percentage", "max"))
)
for stringency in (0.1, 0.2, 0.3, 0.4, 0.5):
    kept = motifs[motifs["patient_fraction"] > stringency]
    print(stringency, kept["patient_class"].value_counts().to_dict())
```

This view uses all patients at once. It is meant for choosing a reasonable range, not for estimating performance.

### `max_class_features`

`max_class_features` caps the number of motifs per class passed to the classifier. CISM ranks each class's passing
motifs by their frequency in individual FOVs and keeps the motifs found in the top `max_class_features` entries. A
class therefore gets at most `max_class_features` motifs, and fewer when one motif ranks high in several FOVs. The
analyses in this repository fix it at 30, so stringency is the only selection parameter that changes. The cap only
matters when many motifs pass.

### Optional: tune stringency by cross-validated AUC

`tune_stringency` searches the stringency with Optuna and scores each trial by the leave-one-patient-out ROC AUC of
`analyze_motifs`. These are the settings used by [tutorial/cism_tutorial.py](tutorial/cism_tutorial.py):

```python
tuning_result = discriminator.tune_stringency(
    feature_conf=feature_conf,           # template; only the two ranges below are searched
    n_trials=20,
    metric="roc_auc",
    random_state=0,
    n_jobs=1,
    prefer="threads",
    shared_percentage_range=(0.2, 0.6),
    max_class_features_range=(30, 30),   # keep the cap fixed
)
print(tuning_result.best_params, tuning_result.best_score)
```

The best AUC is chosen among many trials on the same patients, so it is optimistic. Compare it with the AUC obtained
after shuffling the patient labels at the chosen stringency:

```python
import numpy as np

chosen_conf = HardDiscriminativeFC(
    labels=LABELS,
    extract_by=DiscriminativeFeatureKey.STRUCTURE_AND_CELL_IDENTITIES,
    use_cells_type_composition=False,
    use_motifs=True,
    shared_percentage=tuning_result.best_params["shared_percentage"],
    max_class_features=tuning_result.best_params["max_class_features"],
)
null_aucs = []
for seed in range(10):
    try:
        shuffled = discriminator.analyze_motifs(
            feature_conf=chosen_conf, exclude_patients=[], n_jobs=1, prefer="threads",
            rand_patient_class=True, random_state=np.random.RandomState(seed),
        )
        null_aucs.append(shuffled.get_roc_auc_score())
    except ValueError:                   # no discriminative motifs under this shuffle
        null_aucs.append(np.nan)
print(np.nanmean(null_aucs), np.nanmax(null_aucs))
```

This shuffles at a fixed stringency. For a stricter control that reruns the whole stringency search on each shuffled
labeling, see [figures/scripts/permutation_control.py](figures/scripts/permutation_control.py).

## Maximum spatial distance

`max_distance` (in `prepare_from_centroids`) defines spatial adjacency. For each FOV, CISM connects cells by a
Delaunay triangulation of their centroids and then removes every edge longer than `max_distance`. It is an upper bound
on edge length, not a search radius: two cells are linked only if they are Delaunay neighbors and lie within
`max_distance` of each other.

The parameter is in **the same units as the input coordinates** (`centroid-0`, `centroid-1`). When coordinates are in
pixels, choose a physical distance first and convert it with the pixel size of your images:

```text
max_distance (pixels) = distance (µm) / pixel size (µm per pixel)
```

For example, the MIBI-TOF images in the TNBC tutorial cover 800 µm with 2048 pixels (about 0.39 µm per pixel), so the
tutorial's `max_distance=100` pixels corresponds to about 39 µm. When coordinates are already in microns, as in many
spatial transcriptomics tables, give the distance in microns directly.

Guidelines:

- **Always set it.** With `max_distance=None`, every Delaunay edge is kept, including long edges that cross empty
  regions, tissue tears and FOV borders.
- **Look at the edge lengths** in your data before choosing. The snippet below shows what fraction of Delaunay edges
  each candidate cutoff removes:

  ```python
  import numpy as np
  from scipy.spatial import Delaunay

  from cism import rename_columns_copy

  column_mapping = {"patient_id": "patient", "fov": "fov", "cell_type": "cell_type",
                    "centroid-0": "x", "centroid-1": "y"}
  canonical = rename_columns_copy(cells, column_mapping)   # CISM column names

  def delaunay_edge_lengths(xy):
      indptr, neighbours = Delaunay(xy).vertex_neighbor_vertices
      return np.array([np.linalg.norm(xy[i] - xy[j])
                       for i in range(len(xy))
                       for j in neighbours[indptr[i]:indptr[i + 1]] if i < j])

  lengths = np.concatenate([
      delaunay_edge_lengths(fov[["centroid-0", "centroid-1"]].to_numpy())
      for _, fov in canonical.groupby(["patient_id", "fov"]) if len(fov) >= 3
  ])
  for cutoff in (50, 100, 150):
      print(f"max_distance={cutoff}: removes {(lengths > cutoff).mean():.1%} of Delaunay edges")
  ```

- **Use the same value everywhere.** The motif-instance export in tutorial 03
  (`export_top_motif_visualization_inputs(max_distance=...)`) must use the cutoff the graphs were built with.
- **A new value means new graphs.** Write them to a new dataset folder, or FANMOD+ will reuse results from the old
  graphs (see [Caching and reruns](#caching-and-reruns)).

## Cell-type exclusion

Apply cell-type exclusion only when a biological or technical reason supports it, defined before you look at results.
Do not exclude a cell type because doing so improves predictive performance.

Before excluding anything, compare cell-type composition, pairwise interactions and local cell density across disease
states (see [Bias assessment](#bias-assessment-before-motif-analysis)) to find compositional or sampling biases.

Exclusion may be appropriate when a cell type:

- is restricted to one clinical group,
- forms a distinct, spatially contiguous compartment whose presence alone would give a trivial group-specific signal,
  and
- lies outside the biological question, which concerns the remaining tissue microenvironment.

In such cases, exclude the spatial compartment as a whole rather than removing individual cells here and there. CISM
does this with `exclude_cell_type` in `prepare_from_centroids`:

1. All cells with that label are removed.
2. The removed cells are grouped into spatially connected clusters (connected in the distance-limited Delaunay graph).
3. An alpha shape is drawn around each cluster of 3 or more cells, then optionally grown or shrunk by a buffer.
4. Every remaining cell inside a shape is removed too.

The triangulation is computed before removal, so no new edges are created across the removed region.

```python
prep = prepare_from_centroids(
    dataframe=cells,
    path_to_output_dir=str(DATA_ROOT / "MyCohort_no_tumor"),  # a new folder for new graphs
    column_mapping=column_mapping,
    max_distance=100,
    exclude_cell_type="Tumor",       # one label, after cell_type_mapper is applied
    removed_cluster_alpha=0.01,      # 0 = convex hull; larger values follow the cluster outline more tightly
    removed_cluster_buffer=0,        # in coordinate units; > 0 grows each shape, < 0 shrinks it
)
```

These are the settings used for the melanoma cohort in [tutorial/run_melanoma_pipeline.py](tutorial/run_melanoma_pipeline.py).

Notes:

- `exclude_cell_type` accepts a single label, matched after `cell_type_mapper` is applied.
- The alpha value depends on the scale of your coordinates. Plot a few FOVs with the removed cells highlighted to
  check that the shapes cover the compartment and not the surrounding tissue.
- The excluded label stays in `prep.cell_type_to_id`, so it also appears in `common_cells_type`; it just no longer
  occurs in any graph.
- Record the excluded label, the reason, the alpha and buffer values, and how many cells were removed in each class.

## Cell-type annotations and consolidation

CISM assumes that cell-type annotations are reliable and consistent across samples. Assess and correct batch effects
or systematic differences in cell-type classification upstream, before building graphs.

Decide cell-type definitions and how far to consolidate them according to your biological question and the dataset's
annotation scheme. The trade-off:

- **Fine-grained annotations** give the resolution to detect specific multicellular patterns. However, they multiply
  the possible motifs (about 7,600 possible 4-node motifs with 8 cell types and 112,000 with 16), which makes each
  motif rarer across patients. They also make results more sensitive to cell-type classification errors.
- **Consolidated annotations** group related cell types into broader, biologically meaningful categories, so each
  motif is represented in more patients.

Consolidate with `cell_type_mapper`, a dictionary from every raw label to its consolidated label:

```python
cell_type_mapper = {
    "CD4 T cell": "T cell",
    "CD8 T cell": "T cell",
    "Treg": "T cell",
    "M1 macrophage": "Macrophage",
    "M2 macrophage": "Macrophage",
    "Tumor": "Tumor",
    "Endothelial": "Endothelial",
}
prep = prepare_from_centroids(
    dataframe=cells,
    path_to_output_dir=str(DATA_ROOT / "MyCohort_broad_types"),
    column_mapping=column_mapping,
    cell_type_mapper=cell_type_mapper,
    max_distance=100,
)
common_cells_type = {idx: name for name, idx in prep.cell_type_to_id.items()}
```

- The mapper must cover every raw label. A missing label raises a `KeyError` that lists it.
- The consolidated labels are numbered in sorted order. `prep.cell_type_to_id` holds that numbering, and its inverse
  is the `common_cells_type` that `TissueStateDiscriminativeMotifs` needs.
- Fix the annotation scheme from the biological question before running CISM, not by comparing the AUCs that
  different schemes produce.

## Bias assessment before motif analysis

Run these checks before interpreting motifs, and before any cell-type exclusion. They use the cell table in CISM's
column names (`canonical`, from the [maximum distance](#maximum-spatial-distance) snippet) joined with the patient
classes:

```python
patient_class = pd.read_csv(DATA_ROOT / DATASET / "patient_class.csv",
                            header=None, index_col=0).iloc[:, 0]
canonical["patient_uid"] = DATASET + canonical["patient_id"].astype(str)
canonical["patient_class"] = canonical["patient_uid"].map(patient_class)
canonical = canonical[canonical["patient_class"].isin(LABELS)]
group = canonical.groupby("patient_uid")["patient_class"].first()
```

**Cell-type composition.** Compare each cell type's share of a patient's cells between the two classes. A cell type
that is nearly absent from one class can produce motifs that reflect composition rather than spatial organization.

```python
from scipy.stats import false_discovery_control, mannwhitneyu

composition = pd.crosstab(canonical["patient_uid"], canonical["cell_type"], normalize="index")
in_class = {label: group.reindex(composition.index) == label for label in LABELS}
tests = pd.DataFrame({
    f"mean_{LABELS[0]}": composition[in_class[LABELS[0]]].mean(),
    f"mean_{LABELS[1]}": composition[in_class[LABELS[1]]].mean(),
    "p": [mannwhitneyu(composition.loc[in_class[LABELS[0]], c],
                       composition.loc[in_class[LABELS[1]], c]).pvalue
          for c in composition.columns],
})
tests["q_bh"] = false_discovery_control(tests["p"])
print(tests.sort_values("p"))
```

**Pairwise interactions.** Compare how often each pair of cell types is connected in the graphs. Interactions that
appear in only one class point to compartments that motifs will pick up.

```python
from baselines.pairwise import GraphReader, PairwiseAnalysis

graphs = GraphReader.get_graphs(
    full_graph_df=pd.DataFrame(),
    raw_data_folder=str(DATA_ROOT),
    raw_data_folder_type=f"/{DATASET}",
    disease=DATASET,                     # the patient-key prefix
    cells_type=common_cells_type,
)
pairwise = PairwiseAnalysis(
    tissue_state_csv_path=str(DATA_ROOT / DATASET / "patient_class.csv"),
    tissue_state_to_string={label: label for label in LABELS},
)
pairwise.plot_pairwise_difference_heatmap(
    full_graph_df=graphs, patient_class_a=LABELS[0], patient_class_b=LABELS[1],
    cells_type=common_cells_type,
)
```

**Local cell density.** Denser tissue has shorter cell-to-cell distances, so more edges survive `max_distance` and
more motif instances form. Compare a per-FOV density measure, such as the mean distance to the 5 nearest cells, between
classes:

```python
from scipy.spatial import cKDTree

def mean_knn_distance(fov_cells, k=5):
    xy = fov_cells.to_numpy()
    distances, _ = cKDTree(xy).query(xy, k=k + 1)  # column 0 is the cell itself
    return distances[:, 1:].mean()

fovs = canonical.groupby(["patient_uid", "fov"]).filter(lambda fov: len(fov) > 5)
density = (fovs.groupby(["patient_uid", "fov"])[["centroid-0", "centroid-1"]]
           .apply(mean_knn_distance)
           .groupby(level="patient_uid").median())
density_class = group.reindex(density.index)
print(density.groupby(density_class).describe())
print(mannwhitneyu(density[density_class == LABELS[0]], density[density_class == LABELS[1]]))
```

If any of these checks shows a strong difference between classes, keep it in mind when interpreting motifs. Change the
preprocessing only when the difference has a biological or technical explanation defined independently of the
results.

## Other implementation parameters

| Parameter | Where | Default | Guidance |
|---|---|---|---|
| `iterations` | `CISM(...)` | required (`1000` in all repository analyses) | Number of randomized networks FANMOD+ generates to test each motif's significance. P-values come from these networks, so fewer iterations give coarser p-values. Use 1000; lower it only for quick test runs. |
| `p_value` | `add_dataset(...)` | `0.05` | Motifs with a FOV-level p-value at or above this cutoff are dropped at parsing time. The later selection steps also apply p < 0.05, so values above 0.05 have no effect, and lower values make selection stricter. |
| `quantile_threshold` | `add_dataset(...)` | `1` (off) | Drops, within each FOV, motifs whose frequency is above this quantile. Leave it at 1. |
| `labels` | `HardDiscriminativeFC(...)` | required | Exactly two class names. Other classes in `patient_class.csv` are ignored. |
| `tissue_state_to_string` | `TissueStateDiscriminativeMotifs(...)` | required unless `tissue_state_func` is given | Maps every raw class in `patient_class.csv` to a name, including classes you do not compare. Spaces are removed from the names, so use labels without spaces. |
| `tissue_state_func` | `TissueStateDiscriminativeMotifs(...)` | `None` | For a continuous outcome, a function that turns the value into a class name (example below). Pass `tissue_state_to_string=None`. |
| `extract_by` | `HardDiscriminativeFC(...)`, `discover(...)` | required | `STRUCTURE_AND_CELL_IDENTITIES` treats a motif as its shape plus cell types, which is the CISM definition. `CELL_IDENTITIES` ignores the shape and keeps only the cell-type composition. |
| `use_motifs`, `use_cells_type_composition` | `HardDiscriminativeFC(...)` | required | `True` and `False` give the standard CISM features (motif frequencies only). |
| `random_state` | `analyze_motifs(...)` | unseeded | Pass `np.random.RandomState(seed)` to make the random forest results reproducible. |
| `dataset_name` | `add_dataset(...)` | required | Prefix of every patient key (`Patient_uId = dataset_name + patient_id`). It must match the keys in `patient_class.csv`. |

A continuous outcome, such as survival days, needs a cutoff function. The TNBC analysis in
[tutorial/run_tnbc_pipeline.py](tutorial/run_tnbc_pipeline.py) splits survival at 1000 days:

```python
def survival_class(survival_days):
    return "short_term" if survival_days < 1000 else "long_term"

discriminator = TissueStateDiscriminativeMotifs(
    cism=cism,
    tissue_state_csv_path=str(DATA_ROOT / DATASET / "patient_class.csv"),  # rows like "MyCohort7,2612"
    tissue_state_to_string=None,
    tissue_state_func=survival_class,
    common_cells_type=common_cells_type,
)
```

Motifs that tie in frequency at the `max_class_features` cut are ordered by an unstable sort, so the selected motif set
can differ slightly between machines. Fix `random_state` and record the hardware when you need exact reproducibility.

## Caching and reruns

`add_dataset` stores FANMOD+ results and parsed tables per dataset folder and motif size:

```text
<fanmod_output_root_path>/<dataset_folder>/<motif_size>/<patient>_<fov>.csv   # raw FANMOD+ output
<fanmod_cache_root_path>/<dataset_folder>/<motif_size>/<patient>_<fov>.df_cache  # parsed table
```

On a rerun, existing results are reused and only missing FOVs, or outputs smaller than 2 KB (interrupted runs), are
computed again. An interrupted run can therefore be resumed by calling `add_dataset` again.

The cache is keyed by folder, motif size and file name, not by graph content. If you rebuild graphs with a different
`max_distance`, `cell_type_mapper` or `exclude_cell_type`, do one of the following:

- **Recommended:** write the new graphs to a new dataset folder. `dataset_name` sets the patient-key prefix, so you can
  keep it unchanged while `dataset_folder` changes.
- Rebuild in place and pass `force_run_fanmod=True, force_parse=True` to `add_dataset` once.

Use separate output folders for every configuration you compare. It also helps to save a small file next to the graphs
that records the settings used to build them, as [tutorial/run_seaad_pipeline.py](tutorial/run_seaad_pipeline.py)
does with `graph_max_distance.json`. Avoid `.txt` for that file, because every `.txt` file in the dataset folder is
read as a graph.

## Computational configuration

**FANMOD+ (stage 2)** is the slowest step. `add_dataset` runs one FANMOD+ process per FOV, `n_jobs` at a time, and
the runtime of each process grows roughly linearly with the number of lines in the FOV's graph file. On our cluster's
CPU nodes, with 1000 iterations, one worker processed about:

| Motif size | Graph-file lines per minute per worker |
|---|---|
| 3 | ~4,000 |
| 4 | ~920 |
| 5 | Far slower: hours per FOV for FOVs with tens of thousands of lines |

Count the lines with `wc -l data/<dataset>/*.txt` to estimate your runtime. Memory also grows with motif size: in the
5-node TNBC run, 12 FANMOD+ processes at once needed more than 60 GB. For 5-node motifs, start with fewer parallel
jobs than cores.

**Motif selection and classification (stage 3)**: every parallel worker of `analyze_motifs`, `get_features` and
`tune_stringency` holds its own copy of the motif table, which reaches millions of rows for 5-node motifs. Start with
`n_jobs=1, prefer="threads"`, as the scripts in [tutorial/](tutorial/) do, and raise it only if memory allows. Each
tuning trial is a full leave-one-patient-out run. Expect minutes per trial for 3-node motifs on about 40 patients and
tens of minutes for 5-node motifs.

**Cluster runs**: [tutorial/](tutorial/) has SLURM examples (`*_runner.sbatch`) that run one motif size per job
through the `run_*_pipeline.py` scripts.

## What to report

To make a CISM analysis reproducible and interpretable, report:

- the coordinate units, pixel size (if applicable) and `max_distance`, in both coordinate units and microns,
- the cell-type annotation scheme and any `cell_type_mapper` consolidation,
- any excluded cell type, the reason for excluding it, the alpha and buffer values, and how many cells were removed in
  each class,
- `motif_size`, FANMOD+ `iterations` and the p-value cutoff,
- the stringency (`shared_percentage`) and `max_class_features`, and how they were chosen (number of discriminative
  motifs, or the tuning range and number of trials),
- the number of discriminative motifs per class at the chosen stringency,
- the composition, pairwise-interaction and density checks, and
- for AUC-based tuning, the label-permutation comparison.
