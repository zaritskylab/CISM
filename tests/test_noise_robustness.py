"""Scientific checks for fold selection and motif-induced distributions."""

import ast
import base64
from collections import Counter
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.noise_robustness.cism_noise_robustness import (
    CELL_ORDER, PAIRS, GRAPH_TYPE_NAMES, build_tables, make_distributions,
    prepare_data, select_fold, topology_vectors, total_variation, load_significant_rows, IncompleteCacheError,
    best_admissible_curve,
)
from analysis.noise_robustness.motif_noise_summary import class_weighted_distributions


def sample_rows():
    rows = []
    for motif, patients in [(10, ["a", "b"]), (11, ["a", "b", "c"]),
                             (12, ["a", "b", "c", "d"]), (13, ["d", "e", "f"])]:
        for patient in patients:
            rows.append(dict(ID=motif, patient=patient, Count=1, Freq=float(motif),
                             nunique_colors=2, motif="unused", FOV=patient))
    return pd.DataFrame(rows)


def test_exclusivity_recomputed_after_holding_out_patient():
    classes = dict(zip("abcdef", ["NN"] * 3 + ["NP"] * 3))
    data = prepare_data(sample_rows(), classes, [.46])
    heldout_d = data["patients"].index("d")
    selected = set(data["ids"][list(select_fold(data, heldout_d, .46, 30))])
    assert 12 in selected  # Globally nonexclusive; exclusive after d is held out.
    heldout_e = data["patients"].index("e")
    selected = set(data["ids"][list(select_fold(data, heldout_e, .46, 30))])
    assert 12 not in selected


def test_strict_threshold_and_fov_row_cap():
    classes = dict(zip("abcdef", ["NN"] * 3 + ["NP"] * 3))
    data = prepare_data(sample_rows(), classes, [.46, 2 / 3])
    heldout = data["patients"].index("f")
    selected = set(data["ids"][list(select_fold(data, heldout, 2 / 3, 30))])
    assert 10 not in selected  # Prevalence equals threshold, not greater.
    selected = set(data["ids"][list(select_fold(data, heldout, .46, 2))])
    assert 11 in selected
    assert 10 not in selected  # Top two FOV rows both belong to motif 11.


def test_undirected_edges_and_node_multiplicity():
    g = nx.DiGraph()
    g.add_nodes_from([(1, {"type": "4"}), (2, {"type": "4"}), (3, {"type": "8"})])
    g.add_edges_from([(1, 2), (2, 1), (3, 1), (1, 3)])
    encoded = base64.b64encode(pickle.dumps(g)).decode()
    cells, pairs = topology_vectors(encoded)
    assert cells[CELL_ORDER.index("CD8 T cell")] == 2
    assert cells.sum() == 3
    assert pairs.sum() == 2  # Reverse directed edges collapse to one undirected edge.
    assert pairs[PAIRS.index(("CD8 T cell", "MAC"))] == 1


def test_tv_and_empty_distribution():
    assert total_variation([.2, .8], [.2, .8]) == 0
    assert total_variation([1, 0], [0, 1]) == 1
    assert total_variation([.1, .9], [.3, .7]) == pytest.approx(.2)
    assert np.isnan(total_variation(None, [.3, .7]))
    data = dict(labels=np.array(["NN", "NP"]), ids=np.array([]), counts=np.empty((0, 2)))
    empty = make_distributions(data, set(), {})
    assert empty["NN"]["cell"] is None


def test_original_reference_excluded_from_raw_retrial_boxes():
    def run(run_id, condition="raw"):
        selections = {mode: dict(auc=.8, stringency=.46, n_motifs=1, motif_ids=[1],
                                distributions={lab: {"cell": [.5, .5], "pair": [.2, .8]} for lab in ["NN", "NP"]})
                      for mode in ["fixed", "optimal"]}
        return dict(condition=condition, noise_rate=0, run_id=run_id, curves=[], selections=selections, export_validation={})
    _, _, _, distances, within = build_tables([run(0), run(1), run(2)], 0)
    assert set(distances[distances.reference == "raw_same_mode"].run_id) == {1, 2}
    assert set(distances[distances.reference == "raw_fixed_0.46"].run_id) == {1, 2}
    assert len(within) == 3 * 2 * 2 * 2


def test_selection_matches_exporter_methods():
    """Execute only the two original methods, avoiding CISM's heavyweight imports."""
    source = Path("/groups/assafza_group/assafza/Omri/golan_output/cism/cism.py")
    if not source.exists():
        pytest.skip("Original exported implementation is unavailable")
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TissueStateDiscriminativeMotifs")
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ["_extract_discriminative", "_extract_features"]]
    for method in methods:
        method.decorator_list = []
    module = ast.Module(body=methods, type_ignores=[])
    namespace = dict(pd=pd, np=np, typing=__import__("typing"), FeatureConfiguration=object,
                     SoftDiscriminativeFC=type("Soft", (), {}),
                     DiscriminativeMotifs=SimpleNamespace(PATIENT_CLASS="patient_class", PATIENT_COUNT_KEY="patient_count", PATIENT_PERCENTAGE_KEY="patient_percentage"))
    exec(compile(module, str(source), "exec"), namespace)
    classes = dict(zip("abcdef", ["NN"] * 3 + ["NP"] * 3))
    frame = sample_rows()
    # Avoid exact frequency ties in this parity test; the new implementation uses stable tie order.
    frame["Freq"] += np.arange(len(frame)) * .001
    data = prepare_data(frame, classes, [.2, .46, .5, .8])
    frame["patient_class"] = frame.patient.map(classes)
    frame["Patient_uId"] = frame.patient
    frame["p_value"] = 0.
    conf = SimpleNamespace(get_shared_percentage=lambda: .46, max_class_features=30,
                           use_cells_type_composition=False, use_motifs=True, cells_type_composition_vec=[],
                           switch_motif_with_cell_type_composition=[], switch_cell_type_composition_with_motif=[],
                           switch_cell_type_composition_with_motif_map={}, cells_type_composition_hash_remove=[], motifs_ids_remove=[])
    for heldout, patient in enumerate(data["patients"]):
        train = frame[frame.patient != patient].copy()
        summary, _ = namespace["_extract_discriminative"](None, train, "patient_class", "ID", [], 1, 1, .05)
        for threshold in [.2, .46, .5, .8]:
            conf.get_shared_percentage = lambda: threshold
            _, original, _ = namespace["_extract_features"](summary, conf)
            actual = data["ids"][list(select_fold(data, heldout, threshold, 30))].tolist()
            assert actual == original


def test_reconstructed_distributions_match_original_export():
    root = Path(__file__).resolve().parents[1]
    checkpoint = root / "motif_noise_summary_outputs/final_analysis/runs/raw/run_000.json"
    source = Path("/groups/assafza_group/assafza/Omri/golan_output")
    if not checkpoint.exists() or not source.exists():
        pytest.skip("Raw reference checkpoint/source unavailable")
    import json
    result = json.loads(checkpoint.read_text())
    export = pd.read_csv(source / "output/Melanoma/Motifs/discriminative_motifs0.csv")
    classes = result["settings"]["patient_classes"]
    cell, pair = class_weighted_distributions(export, classes, ["NN", "NP"])
    cell_index = {name: ix for ix, name in enumerate(CELL_ORDER)}
    pair_index = {pair: ix for ix, pair in enumerate(PAIRS)}
    for label in ["NN", "NP"]:
        expected_cells = np.zeros(len(CELL_ORDER))
        expected_pairs = np.zeros(len(PAIRS))
        for row in cell[cell["class"] == label].itertuples():
            expected_cells[cell_index[row.cell_type]] += row.fraction
        for row in pair[pair["class"] == label].itertuples():
            key = tuple(sorted([row.source, row.target], key=cell_index.__getitem__))
            expected_pairs[pair_index[key]] += row.probability
        np.testing.assert_allclose(expected_cells, result["selections"]["fixed"]["distributions"][label]["cell"])
        np.testing.assert_allclose(expected_pairs, result["selections"]["fixed"]["distributions"][label]["pair"])


def test_truncated_cache_is_rejected_even_if_last_row_is_nonsignificant(tmp_path):
    path = tmp_path / "truncated.df_cache"
    header = ",Patient,FOV,ID,Freq,Count,Mean_Freq,STD,z_score,p_value,hash,motif,nunique_colors\n"
    valid = "0,1,FOV1,1,.01,1,.001,.01,2,0," + "a" * 32 + ",encoded,2\n"
    bad_prefix = "1,1,FOV1,2,.001,1,.001,.01,0,"
    path.write_text(header + valid + bad_prefix + ".999," + "b" * 32 + ",truncated\n")
    with pytest.raises(IncompleteCacheError) as error:
        load_significant_rows([path], .05)
    assert error.value.issues[0]["recorded_p_value"] == ".999"
    path.write_text(header + valid + bad_prefix + ".001," + "b" * 32 + ",truncated\n")
    with pytest.raises(IncompleteCacheError):
        load_significant_rows([path], .05)
    path.write_text(header + valid + bad_prefix + "unknown," + "b" * 32 + ",truncated\n")
    with pytest.raises(IncompleteCacheError):
        load_significant_rows([path], .05)


def test_pdf_text_pagination_keeps_lines_inside_page():
    import io
    from matplotlib.backends.backend_pdf import PdfPages
    pypdf = pytest.importorskip("pypdf")
    from analysis.noise_robustness.report_cism_noise_robustness import text_pages
    output = io.BytesIO()
    with PdfPages(output) as pdf:
        text_pages(pdf, "Methods", [f"Paragraph {i}: retained experimental detail." for i in range(80)])
    reader = pypdf.PdfReader(output)
    assert len(reader.pages) > 1
    for page in reader.pages:
        coordinates = []
        def visit(text, cm, tm, _font, _size):
            if text.strip():
                coordinates.append(cm[5] + tm[5])
        page.extract_text(visitor_text=visit)
        assert min(coordinates) > 30


def test_optimum_requires_a_fittable_forest_in_every_fold():
    curves = [dict(stringency=.2, auc=.8, empty_folds=0),
              dict(stringency=.3, auc=.8, empty_folds=0),
              dict(stringency=.5, auc=None, neutral_fallback_auc=.95, empty_folds=5)]
    assert best_admissible_curve(curves)["stringency"] == .2
    with pytest.raises(ValueError, match="every LOOCV fold"):
        best_admissible_curve([curves[-1]])


def test_pair_heatmap_preserves_undirected_probability_mass():
    from analysis.noise_robustness.report_cism_noise_robustness import pair_probability_matrix
    values = np.zeros(len(PAIRS))
    values[PAIRS.index(("B cell", "B cell"))] = .25
    values[PAIRS.index(("B cell", "CD8 T cell"))] = .75
    matrix = pair_probability_matrix(values)
    assert matrix[0, 0] == .25
    assert matrix[0, CELL_ORDER.index("CD8 T cell")] == .75
    assert np.isnan(matrix[CELL_ORDER.index("CD8 T cell"), 0])
    assert np.nansum(matrix) == 1
    assert np.isfinite(matrix).sum() == 120
    with pytest.raises(ValueError, match="120"):
        pair_probability_matrix(values[:-1])


def test_mean_composition_requires_complete_categories_and_consistent_run_counts():
    from analysis.noise_robustness.report_cism_noise_robustness import mean_composition
    summary = pd.DataFrame([dict(condition="raw", mode="fixed", label="NN", kind="cell",
                                 component=name, mean_probability=1 / len(CELL_ORDER), n_valid_runs=99)
                            for name in reversed(CELL_ORDER)])
    values, count = mean_composition(summary, "raw", "fixed", "NN", "cell")
    np.testing.assert_allclose(values, np.full(15, 1 / 15))
    assert count == 99
    with pytest.raises(ValueError, match="Incomplete"):
        mean_composition(summary.iloc[1:], "raw", "fixed", "NN", "cell")
    summary.loc[0, "n_valid_runs"] = 100
    with pytest.raises(ValueError, match="Incomplete"):
        mean_composition(summary, "raw", "fixed", "NN", "cell")
