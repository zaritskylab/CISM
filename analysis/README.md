# Paper revision analyses

This directory groups the revision analysis code by scientific topic. Only source code, cleared notebooks, requirements, and usage documentation are included. Input datasets, serialized CISM objects, caches, intermediate tables, generated plots, and reports remain external. Shell scripts and scheduler launchers are excluded.

Run Python modules from the repository root, for example:

```bash
python -m analysis.permutation_controls.crc_label_permutation --help
python -m analysis.smore_comparison.evaluate_smore_matched --help
python -m analysis.noise_robustness.cism_noise_robustness --help
```

Direct script execution is also supported. Existing output locations under `analysis/outputs/`, `analysis/reports/`, and tutorial runtime directories are preserved. Supply the input paths required by each command; some scripts retain the original cluster paths and environment defaults. Plotting/report builders consume external saved results and are included as source code only. Scripts that parse historical notebook outputs require a separate executed notebook; for example, pass it using `plot_unidentified_motif_cluster_enrichment.py --notebook PATH`.

## Topics

| Topic | Code |
| --- | --- |
| Absent motifs | [absent_motifs/](absent_motifs/) |
| CISM robustness revision | [cism_robustness_revision/](cism_robustness_revision/) |
| Cohort imbalance | [cohort_imbalance/](cohort_imbalance/) |
| CosMx preparation | [cosmx_preparation/](cosmx_preparation/) |
| CRC characterization | [crc_characterization/](crc_characterization/) |
| Marker ownership | [marker_ownership/](marker_ownership/) |
| Melanoma characterization | [melanoma_characterization/](melanoma_characterization/) |
| Motif clustering | [motif_clustering/](motif_clustering/) |
| Noise robustness | [noise_robustness/](noise_robustness/) |
| Patient level validation | [patient_level_validation/](patient_level_validation/) |
| Permutation controls | [permutation_controls/](permutation_controls/) |
| SMORE comparison | [smore_comparison/](smore_comparison/) |
| Spatial density | [spatial_density/](spatial_density/) |
| Spatial scale | [spatial_scale/](spatial_scale/) |

## Related analysis code

- [Multiclass melanoma classification](../multi_class/): one-vs-one and one-vs-all confidence stacking, motif panels, evaluation, and report builders.
- [Melanoma tutorial notebooks](../melanoma_tutorials/): preparation, FANMOD, selection, motif refinement, and motif frequencies.
- [General tutorial notebooks](../tutorials/): data preparation, initialization, and hard/soft motif selection.
- [GCN comparison](../GCN/): graph neural network preparation and classification.
- [CRC analysis notebook](../CISM_on_CRC.ipynb), [motif cell-type distribution](../in_motif_cell_type_distribution.ipynb), [melanoma classification](../melanoma%20classification.ipynb), and [unidentified-cell post-processing](../unidentified_post_process.ipynb).
- [Regression tests](../tests/) and [shared CISM library](../cism/).

[source_layout.json](source_layout.json) maps every included workspace source path to its location in this branch. Existing topic packages for parameter robustness and patient-level validation retain their layouts.

## Validation

Install the project dependencies from `requirements.txt`, plus `pytest` for regression tests, then run `python -m pytest tests`. Additional noise-analysis dependencies are listed in [noise_robustness/requirements-motif-noise.txt](noise_robustness/requirements-motif-noise.txt). Raw-data and historical-artifact checks require the separately stored source data.
