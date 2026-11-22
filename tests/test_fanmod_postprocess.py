import math

import pandas as pd

from cism.fanmod_postprocess import (
    add_normal_significance_columns,
    build_patient_motif_matrix_from_fanmod,
)


def test_normal_significance_marks_absence_and_proliferation():
    motifs = pd.DataFrame(
        [
            {"ID": 1, "Freq": 0.10, "Mean_Freq": 0.15, "STD": 0.01, "z_score": -5.0},
            {"ID": 2, "Freq": 0.20, "Mean_Freq": 0.10, "STD": 0.02, "z_score": 5.0},
        ]
    )

    scored = add_normal_significance_columns(motifs)

    assert scored.loc[0, "p_value_direction"] == "absence"
    assert scored.loc[1, "p_value_direction"] == "proliferation"
    assert math.isclose(scored.loc[0, "p_value_normal_two_tailed"], scored.loc[1, "p_value_normal_two_tailed"])
    assert scored.loc[0, "depletion"] > 0
    assert scored.loc[1, "depletion"] == 0


def test_underrepresented_matrix_uses_expected_minus_observed_frequency():
    motifs = pd.DataFrame(
        [
            {
                "Patient_uId": "P1",
                "FOV": "F1",
                "ID": 1,
                "Freq": 0.10,
                "Count": 10,
                "Mean_Freq": 0.15,
            },
            {
                "Patient_uId": "P1",
                "FOV": "F2",
                "ID": 1,
                "Freq": 0.20,
                "Count": 10,
                "Mean_Freq": 0.30,
            },
        ]
    )

    matrix = build_patient_motif_matrix_from_fanmod(motifs, patient_ids=["P1"], value="depletion")

    assert math.isclose(matrix.loc["P1", 1], 1 / 15)
