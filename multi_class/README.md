# Multiclass melanoma classification

One-vs-one and one-vs-all classification with confidence stacking and motif-panel comparisons.

- `config.py` and `data_artifacts.py`: cohort variants and artifact preparation.
- `motif_panels.py`: motif-panel construction and provenance.
- `confidence_stacking.py` and `evaluate.py`: classification and evaluation.
- `run_analysis.py`: analysis entry point; run `python -m multi_class.run_analysis --help`.
- `results.ipynb`: inspect external saved results; notebook outputs are cleared.
- `reporting/`: Python report and figure builders.

Input data, serialized objects, runtime outputs, and generated reports are external. See `tests/test_multi_class.py` for self-contained regression checks.
