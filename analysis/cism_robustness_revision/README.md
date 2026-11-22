# Parameter robustness and uncertainty

Source code for parameter sensitivity, repeated-execution variability, motif recovery, patient-bootstrap uncertainty, and CRC/TNBC extensions.

- `analyze.py`: shared cohort preparation, native evaluation, and patient bootstrap.
- `native_classifier.py` and `native_sweep.py`: original-selector execution and parameter sweeps.
- `selection_audit.py`, `reference_mapping.py`, and `verify_outputs.py`: selection, motif identity, and output audits.
- `additional_uncertainty.py`, `crc_extension.py`, and `tnbc_extension.py`: additional cohorts and uncertainty analyses.
- `report.py`, `additional_report.py`, and `compile_report.py`: report generation from separately stored results.
- `test_analysis.py`: scientific regression checks; run `python -m unittest discover -s analysis/cism_robustness_revision -p test_analysis.py`.

Run scripts from the repository root. External cohort caches, serialized objects, repeated-run results, and generated artifacts are not included. Scripts retain the original external source defaults; inspect the constants before running on another machine.
