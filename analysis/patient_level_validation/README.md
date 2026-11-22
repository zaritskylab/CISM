# Patient-level composition and interaction validation

Run the Python programs from the repository root with the external source data mounted, in this order:

1. `python analysis/patient_level_validation/composition.py`
2. `python analysis/patient_level_validation/interactions.py`
3. `python analysis/patient_level_validation/validate_independently.py`
4. `python analysis/patient_level_validation/build_report.py`
5. `python analysis/patient_level_validation/build_pdf_report.py`

Requires numpy, pandas, scipy, matplotlib, and numba. Source paths are configurable in the scripts. Generated tables, null archives, figures, reports, and environment/provenance outputs are not committed. Restore saved results or rerun the analyses before invoking report builders.
