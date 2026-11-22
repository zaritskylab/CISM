#!/usr/bin/env python3
"""Recompute fixed/optimal-stringency motif robustness from complete FANMOD caches.

Selection follows the exported CISM implementation: significant motifs exclusive
to one training class, strict patient-prevalence threshold, then the top 30 FOV
rows per class by frequency (which can contain repeated motif IDs). Feature
selection is repeated inside every leave-one-patient-out fold.
"""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import argparse
import base64
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import pickle
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cism-motif-noise")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.csv as pacsv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from analysis.noise_robustness.motif_noise_summary import CELL_ORDER, condition_paths, normalize_cell


ROOT = _REPO_ROOT
INPUT_ROOT = Path("/groups/assafza_group/assafza/Omri/golan_output")
ALGORITHM_VERSION = 1
LABELS = ("NN", "NP")
# GraphBuilder enumerates sorted consolidated cell names; tumor is absent here.
GRAPH_TYPE_NAMES = sorted([
    "b cell", "blood vessels", "cd4 apc", "cd4 t cell", "cd8 t cell",
    "dc", "gc", "hevs", "mac", "memory%cd4 t cell", "neutrophil",
    "nk cell", "stroma", "treg", "unidentified",
])
PAIRS = [(a, b) for i, a in enumerate(CELL_ORDER) for b in CELL_ORDER[i:]]


class IncompleteCacheError(ValueError):
    def __init__(self, issues):
        self.issues = issues
        super().__init__("Malformed/truncated FANMOD cache; full row coverage cannot be established")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "motif_noise_summary_outputs" / "final_analysis")
    parser.add_argument("--patient-class", type=Path)
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--run-ids", nargs="+", type=int)
    parser.add_argument("--reference-run", type=int, default=0)
    parser.add_argument("--fixed-stringency", type=float, choices=[0.46], default=0.46)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[.2, .3, .4, .42, .44, .45, .46, .48, .5, .6, .7, .72, .74, .75, .76, .8])
    parser.add_argument("--max-class-features", type=int, default=30)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--p-value", type=float, default=.05)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-new-runs", type=int,
                        help="Compute at most this many uncached runs, then refresh the partial report; repeat to resume.")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--compute-only", action="store_true",
                        help="Write run checkpoints without rebuilding report figures (useful for batched jobs).")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run_sources(input_root):
    rows = []
    seen = set()
    for condition, noise_rate, motif_dir in condition_paths(input_root):
        output_root = motif_dir.parents[1]
        for cache_dir in sorted((output_root / "parse_cache").glob("run_*")):
            if not cache_dir.name[4:].isdigit():
                continue
            paths = sorted(p for p in cache_dir.rglob("*.df_cache") if p.parent.name == "4")
            if not paths:
                continue
            run_id = int(cache_dir.name[4:])
            if (condition, run_id) in seen:
                raise ValueError(f"Duplicate condition/run key across output folders: {condition}, {run_id}")
            seen.add((condition, run_id))
            rows.append(dict(condition=condition, noise_rate=noise_rate, run_id=run_id,
                             paths=[str(p) for p in paths],
                             export=str(motif_dir / f"discriminative_motifs{run_id}.csv"),
                             bytes=sum(p.stat().st_size for p in paths)))
    return rows


def load_significant_rows(paths, p_value):
    """The .df_cache files are CSVs, not pickle files. Filter each FOV early."""
    frames = []
    for path in paths:
        malformed_rows = []
        def invalid_row_handler(row):
            # A truncated nonsignificant row can still indicate missing later rows.
            # Inspect malformed rows, but reject the entire run before analysis.
            fields = next(csv.reader([row.text]))
            malformed_rows.append(dict(path=str(path), row_number=row.number,
                                       recorded_p_value=fields[9] if len(fields) > 9 else None,
                                       actual_columns=row.actual_columns, expected_columns=row.expected_columns))
            return "skip"
        table = pacsv.read_csv(path, read_options=pacsv.ReadOptions(use_threads=False),
                              parse_options=pacsv.ParseOptions(invalid_row_handler=invalid_row_handler),
                              convert_options=pacsv.ConvertOptions(include_columns=[
                                  "ID", "Patient", "FOV", "Freq", "Count", "Mean_Freq",
                                  "p_value", "nunique_colors", "motif"]))
        if malformed_rows:
            raise IncompleteCacheError(malformed_rows)
        mask = pc.and_(pc.less(table["p_value"], p_value), pc.is_valid(table["Mean_Freq"]))
        table = table.filter(mask)
        if table.num_rows:
            frames.append(table.to_pandas())
    if not frames:
        raise ValueError("No significant FANMOD rows found")
    frame = pd.concat(frames, ignore_index=True)
    frame["patient"] = "Melanoma" + frame["Patient"].astype(int).astype(str)
    # CISM casts this to float32 before the LOOCV filtering step.
    return frame


def audit_saved_graphs(input_root, output_dir):
    """Check observed label-flip rates in run 0 where graph files were retained."""
    specs = [(name, rate, directory.parents[1] / "Melanoma" / "Graphs" / "0")
             for name, rate, directory in condition_paths(input_root)]
    manifest = [(name, rate, [(str(p), p.stat().st_size, p.stat().st_mtime_ns)
                             for p in sorted(directory.glob("*.txt"))]) for name, rate, directory in specs]
    signature = hashlib.sha256(json.dumps(manifest).encode()).hexdigest()
    stamp = output_dir / "graph_audit_fingerprint.json"
    target = output_dir / "observed_graph_noise_rates.csv"
    if stamp.exists() and target.exists() and json.loads(stamp.read_text()).get("signature") == signature:
        return

    def nodes(path):
        values = {}
        with path.open() as handle:
            for line in handle:
                a, b, type_a, type_b = line.split()
                for node, color in [(a, type_a), (b, type_b)]:
                    if node in values and values[node] != color:
                        raise ValueError(f"Inconsistent node color in {path}: {node}")
                    values[node] = color
        return values

    raw_dir = input_root / "output" / "Melanoma" / "Graphs" / "0"
    raw = {p.name: nodes(p) for p in sorted(raw_dir.glob("*.txt"))}
    rows = []
    for condition, rate, directory in specs:
        total = changed = compared = missing = 0
        for name, original in raw.items():
            path = directory / name
            if not path.exists():
                missing += 1
                continue
            current = original if condition == "raw" else nodes(path)
            if set(current) != set(original):
                raise ValueError(f"Graph node IDs differ from raw in {path}")
            compared += 1
            total += len(original)
            changed += sum(current[node] != color for node, color in original.items())
        rows.append(dict(condition=condition, nominal_rate=rate, run_id=0, compared_fovs=compared,
                         missing_fovs=missing, compared_nodes=total, changed_nodes=changed,
                         observed_flip_rate=changed / total if total else None))
    pd.DataFrame(rows).to_csv(target, index=False)
    atomic_json(stamp, dict(signature=signature))


def prepare_data(frame, patient_classes, thresholds):
    patients = sorted(set(patient_classes) & set(frame["patient"]))
    if set(patients) != set(patient_classes):
        raise ValueError(f"Missing patients: {sorted(set(patient_classes) - set(patients))}")
    labels = np.array([patient_classes[p] for p in patients])
    if set(labels) != set(LABELS):
        raise ValueError("Both NN and NP patients are required")
    frame = frame[frame["patient"].isin(patients)].copy()
    motif_patients = frame[["ID", "patient"]].drop_duplicates()
    motif_patients["label"] = motif_patients["patient"].map(patient_classes)
    presence_counts = motif_patients.groupby(["ID", "label"]).size().unstack(fill_value=0)
    for label in LABELS:
        if label not in presence_counts:
            presence_counts[label] = 0
    # A fold-exclusive motif can occur in at most one patient of the other class.
    # This filter cannot exclude any motif eligible at a tested threshold.
    denominators = np.array([(labels == lab).sum() - 1 for lab in LABELS])
    potential = (presence_counts[list(LABELS)].min(axis=1) <= 1)
    potential &= (presence_counts[list(LABELS)].to_numpy() / denominators).max(axis=1) > min(thresholds)
    candidate_ids = presence_counts.index[potential]
    candidate = frame[frame["ID"].isin(candidate_ids)].copy()
    ids = np.array(sorted(candidate["ID"].unique()), dtype=np.int64)
    motif_index = {int(mid): ix for ix, mid in enumerate(ids)}
    patient_index = {p: ix for ix, p in enumerate(patients)}
    row_m = candidate["ID"].map(motif_index).to_numpy(dtype=int)
    row_p = candidate["patient"].map(patient_index).to_numpy(dtype=int)
    counts = np.zeros((len(ids), len(patients)))
    denominators_by_motif = np.zeros_like(counts)
    presence = np.zeros_like(counts, dtype=bool)
    np.add.at(counts, (row_m, row_p), candidate["Count"].to_numpy(dtype=float))
    frequency = candidate["Freq"].to_numpy(dtype=float)
    np.add.at(denominators_by_motif, (row_m, row_p),
              np.divide(candidate["Count"], frequency, out=np.zeros(len(candidate)), where=frequency > 0))
    presence[row_m, row_p] = True
    features = np.divide(counts, denominators_by_motif, out=np.zeros_like(counts), where=denominators_by_motif > 0)
    ranked = np.argsort(-frequency, kind="stable")
    graph_strings = dict(zip(candidate["ID"].astype(int), candidate["motif"]))
    return dict(ids=ids, patients=patients, labels=labels, counts=counts, features=features,
                presence=presence, row_m=row_m[ranked], row_p=row_p[ranked],
                row_colors=candidate["nunique_colors"].to_numpy()[ranked], graphs=graph_strings,
                significant_rows=len(frame), candidate_count=len(ids))


def select_fold(data, heldout, threshold, max_features):
    labels, presence = data["labels"], data["presence"]
    train = np.arange(len(labels)) != heldout
    eligible = np.zeros(presence.shape[0], dtype=bool)
    for label, other in [("NN", "NP"), ("NP", "NN")]:
        local = train & (labels == label)
        opposite = train & (labels == other)
        prevalence = presence[:, local].sum(axis=1) / local.sum()
        eligible |= (prevalence > threshold) & ~presence[:, opposite].any(axis=1)
    row_valid = eligible[data["row_m"]] & (data["row_p"] != heldout) & (data["row_colors"] >= 1)
    selected_rows = []
    for label in LABELS:
        hits = np.flatnonzero(row_valid & (labels[data["row_p"]] == label))
        selected_rows.extend(hits[:max_features])
    selected_rows.sort()
    return tuple(dict.fromkeys(data["row_m"][selected_rows].tolist()))


def evaluate_run(data, thresholds, max_features, trees, seed):
    records, unions, fold_rows = [], {}, []
    probabilities = {th: [] for th in thresholds}
    fold_features = {th: [] for th in thresholds}
    empty_folds = {th: 0 for th in thresholds}
    unions = {th: set() for th in thresholds}
    fit_count = 0
    labels = data["labels"]
    for heldout, patient in enumerate(data["patients"]):
        train = np.arange(len(labels)) != heldout
        predictions_cache = {}
        for threshold in thresholds:
            selected = select_fold(data, heldout, threshold, max_features)
            fold_features[threshold].append(len(selected))
            unions[threshold].update(selected)
            for ix in selected:
                fold_rows.append(dict(stringency=threshold, held_out_patient=patient, motif_id=int(data["ids"][ix])))
            if not selected:
                probability = .5
                empty_folds[threshold] += 1
            elif selected in predictions_cache:
                probability = predictions_cache[selected]
            else:
                model = RandomForestClassifier(n_estimators=trees, random_state=seed, n_jobs=1)
                x = data["features"][list(selected)].T
                model.fit(x[train], labels[train])
                probability = float(model.predict_proba(x[[heldout]])[0, list(model.classes_).index("NN")])
                predictions_cache[selected] = probability
                fit_count += 1
            probabilities[threshold].append(probability)
    for threshold in thresholds:
        score = float(roc_auc_score(labels == "NN", probabilities[threshold]))
        records.append(dict(stringency=threshold,
                            auc=score if empty_folds[threshold] == 0 else None,
                            neutral_fallback_auc=score if empty_folds[threshold] else None,
                            rf_admissible=empty_folds[threshold] == 0,
                            n_motifs=len(unions[threshold]),
                            mean_fold_motifs=float(np.mean(fold_features[threshold])),
                            empty_folds=empty_folds[threshold],
                            n_patients=len(labels)))
    return records, unions, fold_rows, probabilities, fit_count


def topology_vectors(encoded, type_names=GRAPH_TYPE_NAMES):
    graph = pickle.loads(base64.b64decode(encoded))
    nodes = [normalize_cell(type_names[int(graph.nodes[node]["type"])]) for node in graph.nodes]
    name_by_node = dict(zip(graph.nodes, nodes))
    node_counts = np.array([nodes.count(cell) for cell in CELL_ORDER], dtype=float)
    index = {cell: i for i, cell in enumerate(CELL_ORDER)}
    pair_index = {pair: i for i, pair in enumerate(PAIRS)}
    edge_counts = np.zeros(len(PAIRS))
    for a, b in graph.to_undirected().edges():
        names = sorted((name_by_node[a], name_by_node[b]), key=index.__getitem__)
        edge_counts[pair_index[tuple(names)]] += 1
    return node_counts, edge_counts


def make_distributions(data, selected, topology_cache):
    by_label = {}
    for label in LABELS:
        node_counts, edge_counts = np.zeros(len(CELL_ORDER)), np.zeros(len(PAIRS))
        class_mask = data["labels"] == label
        for ix in sorted(selected):
            motif_id = int(data["ids"][ix])
            if motif_id not in topology_cache:
                topology_cache[motif_id] = topology_vectors(data["graphs"][motif_id])
            nodes, edges = topology_cache[motif_id]
            weight = data["counts"][ix, class_mask].sum()
            node_counts += nodes * weight
            edge_counts += edges * weight
        by_label[label] = {
            "cell": (node_counts / node_counts.sum()).tolist() if node_counts.sum() else None,
            "pair": (edge_counts / edge_counts.sum()).tolist() if edge_counts.sum() else None,
        }
    return by_label


def validate_export(data, selected, path):
    if not Path(path).exists():
        return {"available": False}
    expected = set(pd.read_csv(path)["motif_id"].astype(int))
    actual = set(data["ids"][list(selected)].astype(int))
    return dict(available=True, exported_count=len(expected), reconstructed_count=len(actual),
                matched=expected == actual, missing=sorted(map(int, expected - actual)),
                extra=sorted(map(int, actual - expected)))


def result_fingerprint(spec, settings):
    sources = [(p, Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in spec["paths"]]
    payload = dict(version=ALGORITHM_VERSION, settings=settings, sources=sources)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def record_exclusion(spec, signature, output_dir, issues):
    path = Path(output_dir) / "excluded_runs" / spec["condition"] / f"run_{spec['run_id']:03d}.json"
    record = dict(condition=spec["condition"], noise_rate=spec["noise_rate"], run_id=spec["run_id"],
                  fingerprint=signature, reason="Truncated/malformed parse cache; completeness cannot be established",
                  issues=issues)
    atomic_json(path, record)
    return record


def normalize_rf_admissibility(result):
    for row in result["curves"] + list(result["selections"].values()):
        row["rf_admissible"] = row["empty_folds"] == 0
        if not row["rf_admissible"] and row.get("auc") is not None:
            row["neutral_fallback_auc"] = row["auc"]
            row["auc"] = None
    return result


def best_admissible_curve(curves):
    valid = [row for row in curves if row["empty_folds"] == 0 and row["auc"] is not None]
    if not valid:
        raise ValueError("No tested threshold permits Random Forest fitting in every LOOCV fold")
    return sorted(valid, key=lambda row: (-row["auc"], row["stringency"]))[0]


def checkpoint_status(spec, settings, output_dir):
    signature = result_fingerprint(spec, settings)
    path = Path(output_dir) / "runs" / spec["condition"] / f"run_{spec['run_id']:03d}.json"
    if path.exists():
        result = json.loads(path.read_text())
        if result.get("fingerprint") == signature:
            if result.get("ignored_nonsignificant_malformed_rows"):
                exclusion = record_exclusion(spec, signature, output_dir, result["ignored_nonsignificant_malformed_rows"])
                destination = Path(output_dir) / "excluded_runs" / spec["condition"] / f"run_{spec['run_id']:03d}.quarantined.json"
                path.replace(destination)
                return "excluded", exclusion
            if not result.get("strict_rf_optimum"):
                result = normalize_rf_admissibility(result)
                if result["selections"]["optimal"]["empty_folds"]:
                    return "retune", result
                result["strict_rf_optimum"] = True
                atomic_json(path, result)
            return "complete", result
    exclusion_path = Path(output_dir) / "excluded_runs" / spec["condition"] / f"run_{spec['run_id']:03d}.json"
    if exclusion_path.exists():
        result = json.loads(exclusion_path.read_text())
        if result.get("fingerprint") == signature:
            return "excluded", result
    return "pending", None


def process_run(spec, settings, output_dir, force=False):
    started = time.monotonic()
    path = Path(output_dir) / "runs" / spec["condition"] / f"run_{spec['run_id']:03d}.json"
    signature = result_fingerprint(spec, settings)
    if not force:
        status, saved = checkpoint_status(spec, settings, output_dir)
        if status == "retune":
            best = best_admissible_curve(saved["curves"])
            frame = load_significant_rows(spec["paths"], settings["p_value"])
            data = prepare_data(frame, settings["patient_classes"], settings["thresholds"])
            selected_ids = {row["motif_id"] for row in saved["fold_motifs"] if row["stringency"] == best["stringency"]}
            selected = {ix for ix, mid in enumerate(data["ids"]) if int(mid) in selected_ids}
            if len(selected) != len(selected_ids):
                raise ValueError("Cached fold-union IDs cannot be reconstructed from source data")
            saved["legacy_optimum_before_rf_admissibility_filter"] = {
                k: v for k, v in saved["selections"]["optimal"].items() if k not in ["distributions", "motif_ids"]}
            saved["selections"]["optimal"] = dict(**best, motif_ids=sorted(selected_ids),
                distributions=make_distributions(data, selected, {}))
            saved["strict_rf_optimum"] = True
            atomic_json(path, saved)
            return dict(condition=spec["condition"], run_id=spec["run_id"], status="retuned_without_refitting",
                        best_auc=best["auc"], best_stringency=best["stringency"], seconds=round(time.monotonic()-started, 1))
        if status != "pending":
            return dict(condition=spec["condition"], run_id=spec["run_id"], status=status, seconds=0)
    try:
        frame = load_significant_rows(spec["paths"], settings["p_value"])
    except IncompleteCacheError as error:
        record_exclusion(spec, signature, output_dir, error.issues)
        return dict(condition=spec["condition"], run_id=spec["run_id"], status="excluded_source_data", seconds=0)
    data = prepare_data(frame, settings["patient_classes"], settings["thresholds"])
    del frame
    curves, unions, fold_rows, probabilities, fits = evaluate_run(
        data, settings["thresholds"], settings["max_features"], settings["trees"], settings["seed"])
    best = best_admissible_curve(curves)
    fixed = next(row for row in curves if row["stringency"] == settings["fixed"])
    topology_cache = {}
    selections = {}
    for mode, row in [("fixed", fixed), ("optimal", best)]:
        selected = unions[row["stringency"]]
        selections[mode] = dict(**row, motif_ids=sorted(int(data["ids"][ix]) for ix in selected),
                                distributions=make_distributions(data, selected, topology_cache))
    result = dict(fingerprint=signature, condition=spec["condition"], noise_rate=spec["noise_rate"],
                  run_id=spec["run_id"], n_fovs=len(spec["paths"]),
                  significant_rows=data["significant_rows"], candidate_count=data["candidate_count"],
                  fits=fits, settings=settings, patients=data["patients"], curves=curves,
                  probabilities={str(k): v for k, v in probabilities.items()},
                  selections=selections, fold_motifs=fold_rows,
                  export_validation=validate_export(data, unions[settings["fixed"]], spec["export"]),
                  strict_rf_optimum=True, seconds=time.monotonic() - started)
    atomic_json(path, result)
    return dict(condition=spec["condition"], run_id=spec["run_id"], status="computed",
                seconds=round(result["seconds"], 1), best_auc=best["auc"], best_stringency=best["stringency"],
                fixed_motifs=fixed["n_motifs"], fits=fits, export_match=result["export_validation"].get("matched"))


def total_variation(a, b):
    if a is None or b is None:
        return float("nan")
    return float(.5 * np.abs(np.asarray(a) - np.asarray(b)).sum())


def build_tables(results, reference_run):
    reference = next((r for r in results if r["condition"] == "raw" and r["run_id"] == reference_run), None)
    if reference is None:
        raise ValueError(f"Raw reference run {reference_run} has not been evaluated")
    metrics, curves, distributions, distances, within = [], [], [], [], []
    for run in results:
        info = {key: run[key] for key in ["condition", "noise_rate", "run_id"]}
        curves.extend([{**info, **row} for row in run["curves"]])
        for mode, selection in run["selections"].items():
            metrics.append({**info, "mode": mode, **{k: v for k, v in selection.items() if k not in ["motif_ids", "distributions"]},
                            "export_match": run["export_validation"].get("matched")})
            for label in LABELS:
                for kind in ["cell", "pair"]:
                    vector = selection["distributions"][label][kind]
                    distributions.append({**info, "mode": mode, "label": label, "kind": kind, "vector": vector})
                    # Same-mode reference isolates perturbation from the choice of analysis policy.
                    ref = reference["selections"][mode]["distributions"][label][kind]
                    if not (run["condition"] == "raw" and run["run_id"] == reference_run):
                        distances.append({**info, "mode": mode, "label": label, "kind": kind,
                                          "reference": "raw_same_mode", "reference_run": reference_run,
                                          "tv": total_variation(vector, ref)})
                    # Also provide a common fixed-stringency baseline across both policies.
                    common = reference["selections"]["fixed"]["distributions"][label][kind]
                    if not (run["condition"] == "raw" and run["run_id"] == reference_run):
                        distances.append({**info, "mode": mode, "label": label, "kind": kind,
                                          "reference": "raw_fixed_0.46", "reference_run": reference_run,
                                          "tv": total_variation(vector, common)})
    for condition in sorted({r["condition"] for r in results}):
        local = [r for r in results if r["condition"] == condition]
        for i, run in enumerate(local):
            for other in local[i + 1:]:
                for mode in ["fixed", "optimal"]:
                    for label in LABELS:
                        for kind in ["cell", "pair"]:
                            within.append(dict(condition=condition, noise_rate=run["noise_rate"],
                                               run_id=run["run_id"], other_run_id=other["run_id"],
                                               mode=mode, label=label, kind=kind,
                                               tv=total_variation(run["selections"][mode]["distributions"][label][kind],
                                                                  other["selections"][mode]["distributions"][label][kind])))
    return pd.DataFrame(metrics), pd.DataFrame(curves), pd.DataFrame(distributions), pd.DataFrame(distances), pd.DataFrame(within)


def main():
    args = parse_args()
    if args.workers < 1 or args.trees < 1 or args.max_class_features < 1:
        raise ValueError("Workers, trees, and the per-class feature cap must be positive")
    if args.max_new_runs is not None and args.max_new_runs < 1:
        raise ValueError("--max-new-runs must be positive")
    if any(not np.isfinite(value) or not 0 <= value <= 1 for value in args.thresholds):
        raise ValueError("Stringency thresholds must be finite fractions between 0 and 1")
    if args.force and args.max_new_runs:
        raise ValueError("--force and --max-new-runs cannot be combined")
    if args.report_only and args.compute_only:
        raise ValueError("--report-only and --compute-only cannot be combined")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    patient_file = args.patient_class or args.input_root / "Melanoma" / "patient_class.csv"
    patient_df = pd.read_csv(patient_file, names=["patient", "label"])
    classes = dict(patient_df[patient_df["label"].isin(LABELS)].itertuples(index=False, name=None))
    settings = dict(thresholds=sorted(set(args.thresholds + [args.fixed_stringency])), fixed=args.fixed_stringency,
                    max_features=args.max_class_features, trees=args.trees, seed=args.seed,
                    p_value=args.p_value, patient_classes=classes)
    specs = run_sources(args.input_root)
    if args.conditions:
        specs = [s for s in specs if s["condition"] in args.conditions]
    if args.run_ids is not None:
        specs = [s for s in specs if s["run_id"] in args.run_ids]
    if not specs:
        raise ValueError("No complete FANMOD parse-cache runs were discovered")
    inventory = pd.DataFrame([{k: v for k, v in spec.items() if k not in ["paths", "export"]} | {"n_fovs": len(spec["paths"])} for spec in specs])
    inventory.to_csv(args.output_dir / "source_inventory.csv", index=False)
    atomic_json(args.output_dir / "analysis_settings.json", dict(settings=settings, reference_run=args.reference_run,
                input_root=str(args.input_root), algorithm_version=ALGORITHM_VERSION,
                rf_auc_admissibility="At least one selected motif in every LOOCV training fold"))
    audit_saved_graphs(args.input_root, args.output_dir)
    if not args.report_only:
        work_specs = specs
        if args.max_new_runs:
            work_specs = []
            for spec in specs:
                status, _ = checkpoint_status(spec, settings, args.output_dir)
                if status in ("pending", "retune"):
                    work_specs.append(spec)
                if len(work_specs) == args.max_new_runs:
                    break
        print(f"Evaluating {len(work_specs)} runs (of {len(specs)} total), {len(settings['thresholds'])} thresholds, {args.trees} RF trees, {args.workers} workers", flush=True)
        failures = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_run, spec, settings, str(args.output_dir), args.force): spec for spec in work_specs}
            for finished, future in enumerate(as_completed(futures), 1):
                spec = futures[future]
                try:
                    result = future.result()
                    print(f"[{finished}/{len(work_specs)}] {json.dumps(result)}", flush=True)
                except Exception as error:
                    failure = dict(condition=spec["condition"], run_id=spec["run_id"], error=repr(error))
                    failures.append(failure)
                    print(f"[{finished}/{len(work_specs)}] FAILED {failure}", flush=True)
        atomic_json(args.output_dir / "failures.json", failures)
    results, exclusions = [], []
    for spec in specs:
        status, result = checkpoint_status(spec, settings, args.output_dir)
        if status == "complete":
            results.append(result)
        elif status == "excluded":
            exclusions.append(result)
    atomic_json(args.output_dir / "source_data_exclusions.json", exclusions)
    if not results:
        raise RuntimeError("No completed runs; see failures.json")
    if args.compute_only:
        print(f"Checkpointed {len(results)} valid runs; {len(exclusions)} excluded; {len(specs) - len(results) - len(exclusions)} pending. Use --report-only to rebuild the final report.", flush=True)
        if failures:
            raise SystemExit("Some runs failed; see failures.json and rerun to retry them")
        return
    from analysis.noise_robustness.report_cism_noise_robustness import generate_report
    metrics, curves, distributions, distances, within = build_tables(results, args.reference_run)
    for name, table in [("run_metrics", metrics), ("stringency_sweep", curves),
                        ("distances_to_original", distances), ("within_condition_distances", within)]:
        table.to_csv(args.output_dir / f"{name}.csv", index=False)
    generate_report(args.output_dir, results, inventory, metrics, curves, distributions, distances, within, args.reference_run)
    print(f"Report completed with {len(results)} valid / {len(specs)} requested runs, {len(exclusions)} source exclusions: {args.output_dir}", flush=True)
    if len(results) + len(exclusions) != len(specs):
        if args.max_new_runs:
            print("Batch finished; repeat this command to resume the remaining trials. The report is explicitly partial.", flush=True)
        else:
            raise SystemExit("Incomplete analysis: failures or missing runs are explicitly listed in the report")


if __name__ == "__main__":
    main()
