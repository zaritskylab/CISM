import math

import numpy as np
import pandas as pd

from cism import (
    MotifSelectionWeights,
    SoftMotifSelectionConfig,
    StabilityGateConfig,
    build_patient_motif_matrix,
    compute_loocv_selection_stability,
    score_soft_motifs,
)


def _patient_classes():
    return pd.DataFrame(
        {"patient_class": ["A", "A", "B", "B"]},
        index=["P0", "P1", "P2", "P3"],
    )


def _motif_rows():
    rows = []

    def add(patient, motif_id, freq, count=10, fov="F0", p_value=0.01):
        rows.append(
            {
                "Patient_uId": patient,
                "Patient": patient,
                "FOV": fov,
                "ID": motif_id,
                "Freq": freq,
                "Count": count,
                "p_value": p_value,
            }
        )

    for patient, freq in {"P0": 0.0, "P1": 0.1, "P2": 0.8, "P3": 1.0}.items():
        add(patient, 1, freq)
    for patient, freq in {"P0": 1.0, "P1": 0.9, "P2": 0.1, "P3": 0.0}.items():
        add(patient, 2, freq)
    for patient, freq in {"P2": 0.7, "P3": 0.6}.items():
        add(patient, 3, freq)

    return pd.DataFrame(rows)


def test_patient_motif_matrix_uses_cism_frequency_formula_and_zero_fill():
    motifs = pd.DataFrame(
        [
            {"Patient_uId": "P0", "FOV": "F0", "ID": 1, "Count": 2, "Freq": 0.2},
            {"Patient_uId": "P0", "FOV": "F1", "ID": 1, "Count": 2, "Freq": 0.1},
            {"Patient_uId": "P1", "FOV": "F0", "ID": 2, "Count": 5, "Freq": 0.5},
        ]
    )

    matrix = build_patient_motif_matrix(motifs, patient_ids=["P0", "P1", "P2"])

    assert math.isclose(matrix.loc["P0", 1], 4 / 30)
    assert matrix.loc["P1", 1] == 0
    assert matrix.loc["P2", 1] == 0
    assert matrix.loc["P2", 2] == 0


def test_direction_free_auc_scores_both_enrichment_directions():
    config = SoftMotifSelectionConfig(labels=["A", "B"], top_k=2)

    result = score_soft_motifs(_motif_rows(), _patient_classes(), config)
    scores = result.scores.set_index("ID")

    assert scores.loc[1, "auc"] == 1.0
    assert scores.loc[1, "effect_score"] == 1.0
    assert scores.loc[2, "auc"] == 0.0
    assert scores.loc[2, "effect_score"] == 1.0


def test_exclusive_motifs_are_reported_without_specificity_penalty():
    config = SoftMotifSelectionConfig(labels=["A", "B"], top_k=3)

    result = score_soft_motifs(_motif_rows(), _patient_classes(), config)
    scores = result.scores.set_index("ID")

    assert scores.loc[3, "overall_prevalence"] == 0.5
    assert scores.loc[3, "prevalence_group_0"] == 0.0
    assert scores.loc[3, "prevalence_group_1"] == 1.0
    assert scores.loc[3, "specificity"] == 1.0
    assert scores.loc[3, "prevalence_score"] == 0.5


def test_normalized_components_are_bounded():
    config = SoftMotifSelectionConfig(labels=["A", "B"], top_k=3)

    result = score_soft_motifs(_motif_rows(), _patient_classes(), config)
    scores = result.scores

    for column in [
        "abundance_score",
        "prevalence_score",
        "confidence_score",
        "dispersion_desirability_score",
    ]:
        assert ((scores[column] >= 0) & (scores[column] <= 1)).all()


def test_stability_gate_examples_match_plan():
    gate = StabilityGateConfig(tau=0.6, gamma=2.0)

    assert gate.apply(0.60) == 1.0
    assert round(gate.apply(0.45), 2) == 0.56
    assert round(gate.apply(0.30), 2) == 0.25
    assert round(gate.apply(0.15), 2) == 0.06


def test_final_score_is_gated_normalized_additive_score():
    weights = MotifSelectionWeights(effect=2, abundance=1, prevalence=1, confidence=1, dispersion=1)
    gate = StabilityGateConfig(tau=0.5, gamma=1.0)
    config = SoftMotifSelectionConfig(labels=["A", "B"], top_k=2, weights=weights, gate=gate)

    result = score_soft_motifs(
        _motif_rows(),
        _patient_classes(),
        config,
        stability={1: 0.25, 2: 0.5, 3: 1.0},
    )
    scores = result.scores.set_index("ID")

    assert np.allclose(scores["final_score"], scores["stability_gate"] * scores["additive_score"])
    assert scores.loc[1, "stability_gate"] == 0.5
    assert scores.loc[2, "stability_gate"] == 1.0


def test_loocv_stability_uses_training_folds():
    config = SoftMotifSelectionConfig(labels=["A", "B"], top_k=1)

    stability = compute_loocv_selection_stability(_motif_rows(), _patient_classes(), config)

    assert stability.between(0, 1).all()
    assert set(stability.index).issubset({1, 2, 3})


def test_public_api_imports_from_cism():
    from cism import score_soft_motifs as exported_score_soft_motifs

    assert exported_score_soft_motifs is score_soft_motifs
