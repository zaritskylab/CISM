import pandas as pd

from multi_class.config import VARIANTS
from multi_class.confidence_stacking import make_confidence_scoring_folds
from multi_class.motif_panels import make_comparisons, merge_selected_motifs


def test_comparison_generation_all_classes():
    labels = VARIANTS["all_classes_keep_tumor"].labels

    one_vs_all = make_comparisons(labels, "one_vs_all_union")
    one_vs_one = make_comparisons(labels, "one_vs_one_union")

    assert len(one_vs_all) == 4
    assert {comparison.name for comparison in one_vs_all} == {
        "NN_vs_all",
        "NP_vs_all",
        "PN_vs_all",
        "PP_vs_all",
    }
    assert len(one_vs_one) == 6


def test_comparison_generation_without_pp():
    labels = VARIANTS["without_pp_remove_tumor"].labels

    assert len(make_comparisons(labels, "one_vs_all_union")) == 3
    assert len(make_comparisons(labels, "one_vs_one_union")) == 3


def test_tumor_removal_routing_variants():
    assert VARIANTS["all_classes_keep_tumor"].tumor_removed_classes == ()
    assert VARIANTS["all_classes_remove_tumor"].tumor_removed_classes == ("PN", "PP")
    assert VARIANTS["without_pp_remove_tumor"].tumor_removed_classes == ("PN",)


def test_motif_panel_union_deduplicates_and_preserves_provenance():
    table_a = pd.DataFrame(
        {
            "ID": [1, 2],
            "comparison": ["NN_vs_all", "NN_vs_all"],
            "comparison_label": ["NN", "OTHER"],
            "patient_percentage": [0.7, 0.6],
            "mean_freq": [0.3, 0.2],
            "total_count": [10, 5],
        }
    )
    table_b = pd.DataFrame(
        {
            "ID": [1, 3],
            "comparison": ["NN_vs_NP", "NN_vs_NP"],
            "comparison_label": ["NN", "NP"],
            "patient_percentage": [0.8, 0.5],
            "mean_freq": [0.4, 0.1],
            "total_count": [8, 4],
        }
    )

    merged = merge_selected_motifs([table_a, table_b]).set_index("ID")

    assert set(merged.index) == {1, 2, 3}
    assert merged.loc[1, "comparison_provenance"] == "NN_vs_NP;NN_vs_all"
    assert merged.loc[1, "comparison_label_provenance"] == "NN"
    assert merged.loc[1, "total_count_across_selected_rows"] == 18


def test_empty_motif_panel_union_returns_empty_table():
    merged = merge_selected_motifs([pd.DataFrame()])

    assert merged.empty
    assert "ID" in merged.columns


def test_one_vs_all_confidence_folds_score_every_patient_once():
    classes = pd.DataFrame(
        {"patient_class": ["NN", "NN", "NP", "PN"]},
        index=["n1", "n2", "p1", "p2"],
    )
    comparison = make_comparisons(("NN", "NP", "PN"), "one_vs_all_union")[0]

    folds = make_confidence_scoring_folds(classes, comparison)

    assert {fold.fold_type for fold in folds} == {"binary_loocv"}
    assert [patient for fold in folds for patient in fold.score_patients] == list(classes.index)


def test_one_vs_one_confidence_adds_full_pair_fold_for_third_class():
    classes = pd.DataFrame(
        {"patient_class": ["NN", "NN", "NP", "PN", "PN"]},
        index=["n1", "n2", "p1", "x1", "x2"],
    )
    comparison = make_comparisons(("NN", "NP", "PN"), "one_vs_one_union")[0]

    folds = make_confidence_scoring_folds(classes, comparison)
    extra_fold = [fold for fold in folds if fold.fold_type == "full_pair_to_excluded_class"]

    assert len(extra_fold) == 1
    assert set(extra_fold[0].train_patients) == {"n1", "n2", "p1"}
    assert set(extra_fold[0].score_patients) == {"x1", "x2"}
    scored = [patient for fold in folds for patient in fold.score_patients]
    assert sorted(scored) == sorted(classes.index)
