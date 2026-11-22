from pathlib import Path

import pandas as pd

from analysis.smore_comparison.compact_smore_results import load_labeled_smore


def test_headerless_patient_class_keeps_first_patient(tmp_path: Path) -> None:
    (tmp_path / "patient_class.csv").write_text("Melanoma1,NP\nMelanoma2,NN\n")
    pd.DataFrame({
        "patient number": [1, 2], "fov": ["FOV2", "FOV6"]
    }).to_csv(tmp_path / "patient_fov.csv", index=False)
    results = tmp_path / "smore.csv"
    pd.DataFrame({
        "SID": ["FOV2", "FOV6"], "seed": ["1-2-3-4", "4-3-2-1"],
        "frequency": [0.1, 0.2]
    }).to_csv(results, index=False)

    labeled = load_labeled_smore(results, tmp_path)

    assert labeled[["patient number", "class"]].drop_duplicates().to_records(index=False).tolist() == [
        (1, "NP"), (2, "NN")
    ]
