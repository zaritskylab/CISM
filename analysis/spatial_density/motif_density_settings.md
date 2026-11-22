# Motif Appearance Versus Tissue Density Analysis Settings

This analysis tests whether the 19 hard discriminative melanoma motifs from notebooks 2/3 appear in locally dense or sparse tissue regions.

## Inputs

- Motif source: `analysis/outputs/notebook03_selected_discriminative_motifs.csv`
- Motif IDs: reconstructed from notebook 03 via `discriminator.get_features(best_feature_conf)` and `GetMotifsResult.get_motifs_by_class()`
- Motif associations: the source table `direction` column (`NN` or `NP`)
- Excluded source family: notebook 03b/04b soft-selection, LASSO, and refined-panel outputs are not used.
- Serialized CISM artifact: `melanoma_tutorials/melanoma_tutorials/runtime/serialized/Melanoma_cism_ready.pkl`
- Default raw centroid table: `melanoma_tutorials/pet_data/Melanoma/raw_cells.csv`
- Known external centroid table used by the melanoma preparation notebook: `/groups/assafza_group/assafza/Datasets/Melanoma/mapped_cell_types.csv`

The raw centroid table must contain:

- `patient number`
- `fov`
- `cell_id`
- `centroid-0`
- `centroid-1`
- `pred_mapped`

## Density Settings

- Euclidean radius thresholds: `25`, `50`, `100`, `150`, `200` pixels
- Pixel-to-micron conversion: `1 pixel = 0.5 micron`; the `200` pixel radius is therefore `100 microns`.
- Euclidean density units: `cells / micron^2`, using circular area `pi * (radius_pixels * 0.5)^2`.
- Euclidean kNN mean distances are reported in microns, and kNN densities use micron-scaled areas.
- Euclidean kNN values: `5`, `10`, `20`
- Graph kNN values: `5`, `10`, `20`
- Graph hop radii: `1`, `2`, `3`
- Patch size for FOV density overlays: `100 x 100` pixels (`50 x 50` microns)
- Random seed: `0`
- Null samples per observed motif instance: `100`

## Null Models

- Null A: random points sampled uniformly from the same FOV coordinate bounding box.
- Null B: random connected cell sets from the same FOV graph, with the same motif size as the observed motif and no cell-type matching.
- Radius density counts available FOV cells within the circle but divides by the full circle area, including where the circle extends beyond the FOV. No boundary-area correction or edge exclusion is applied to observed or null centers. The cell-coordinate bounding box used for point sampling is not a tissue mask.

## Outputs

Corrected report outputs are written under `melanoma_tutorials/melanoma_tutorials/runtime/motif_density_analysis_report_notebook03_discriminative/`:

- `motif_density_instance_metrics.csv`
- `motif_density_summary.csv`
- `motif_density_null_samples.csv`
- `plots/observed_vs_null_*.png`
- `plots/enrichment_*.png`
- `plots/boxplot_observed_radius_200_density_by_motif.png`
- `plots/boxplot_observed_radius_200_density_by_association.png`
- `plots/boxplot_null_radius_200_density_by_motif.png`
- `plots/boxplot_null_radius_200_density_by_association.png`
- `plots/patch_density_overlay_*.png`
