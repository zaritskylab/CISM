# CRC dataset provenance and analysis version

## Published source

- Schürch et al., *Coordinated Cellular Neighborhoods Orchestrate Antitumoral
  Immunity at the Colorectal Cancer Invasive Front*, Cell (2020),
  DOI: <https://doi.org/10.1016/j.cell.2020.07.005>.
- Published single-cell table, Mendeley Data version 1, DOI:
  <https://doi.org/10.17632/mpjzbtfgfr.1>.
- Published file name: `CRC_clusters_neighborhoods_markers.csv`.
- Processed primary images: TCIA collection
  `CRC_FFPE-CODEX_CellNeighs`, DOI:
  <https://doi.org/10.7937/TCIA.2020.FQN0-0326>.

## CISM analysis copy

- File: `/groups/assafza_group/assafza/Datasets/CRC/CRC_cells.csv`
- SHA-256: `37eb7d0c3f7611addf43f78a6307cf662e0c480c7d781658eff7656bfb6e78a8`
- Size: 86,684,359 bytes
- Rows: 94,389 cells
- Regions: 35 FOVs
- Patients: 35, with exactly one region per patient
- Region convention: one `A` region per patient (`P01_A_FOV1` through
  `P35_A_FOV69`, with nonconsecutive FOV identifiers)
- Duplicate `(fov, cellID)` keys: 0
- Missing values in `fov`, `cellID`, `centroid_x`, `centroid_y`, or `class`: 0
- Cell labels: 17 values: `Bcell`, `CD3T`, `CD4T`, `CD8T`, `DC`,
  `Endothelial`, `Fibroblast`, `Lymphatic`, `Macrophage`, `Neuron`,
  `Neutrophil`, `Plasma`, `SMV`, `Stroma`, `Treg`, `Tumor`, and
  `Unidentified`. These comprise 16 annotated major types plus the
  `Unidentified` category.

The analysis file is therefore a processed, patient-balanced subset rather
than a byte-identical copy of the complete published table. The paper reports
140 tissue regions from the same 35 patients; this copy retains one region per
patient.

## Clinical labels

- File: `/groups/assafza_group/assafza/Datasets/CRC/CRC_patient_class.csv`
- SHA-256: `c1f7770ef640f3c6909b220d146e26d1559be1633a8fb54ff6dfca5bb40dcabe`
- Among patients 1–35: 17 CLR (`Group=1`) and 18 DII (`Group=2`), matching the
  cohort sizes reported by the publication.

## Defensible description in a manuscript

> We analyzed a patient-balanced derivative of the published Schürch et al.
> CRC CODEX dataset (Mendeley Data, version 1, DOI:
> 10.17632/mpjzbtfgfr.1). The analysis copy retained one A-region from each of
> the 35 patients (94,389 cells; 17 CLR and 18 DII patients) and used 16 major
> annotated cell classes plus an Unidentified category. We treated patients,
> rather than tissue regions, as the independent units of analysis.

The phrase “patient-balanced” is justified by the observed file structure.
The stronger claim that the retained region was selected as representative,
randomly, or according to a predeclared quality criterion requires the missing
preprocessing script or contemporaneous documentation and should not be made
without that evidence.

## Remaining provenance gap

The repository currently does not contain the script that converts
`CRC_clusters_neighborhoods_markers.csv` into `CRC_cells.csv`. Consequently,
the exact cell filtering, region-selection rationale, coordinate conversion,
marker transformation, and fine-to-major cell-type mapping cannot yet be
reconstructed from code. For full reproducibility, archive that transformation
script (or recreate and validate it against the published table), retain the
downloaded source file with its checksum, and add a row-level audit showing
which published cells were retained.
