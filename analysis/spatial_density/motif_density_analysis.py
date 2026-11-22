#!/usr/bin/env python3
"""Analyze motif appearance versus local tissue density for melanoma motifs."""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import argparse
import math
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

os.makedirs("/tmp/codex_matplotlib", exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_matplotlib")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import networkx as nx
from networkx.algorithms import isomorphism as nx_iso
import numpy as np
import pandas as pd
from scipy.stats import entropy, mannwhitneyu, ttest_ind
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

SCRIPT_DIR = ANALYSIS_DIR
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cism import helpers  # noqa: E402


DEFAULT_ARTIFACT_PATH = (
    REPO_ROOT
    / "melanoma_tutorials"
    / "melanoma_tutorials"
    / "runtime"
    / "serialized"
    / "Melanoma_cism_ready.pkl"
)
DEFAULT_MOTIF_SOURCE_PATH = (
    REPO_ROOT
    / "analysis"
    / "outputs"
    / "notebook03_selected_discriminative_motifs.csv"
)
DEFAULT_RAW_CELLS_CSV = REPO_ROOT / "melanoma_tutorials" / "pet_data" / "Melanoma" / "raw_cells.csv"
DEFAULT_GRAPH_DIR = REPO_ROOT / "melanoma_tutorials" / "pet_data" / "Melanoma"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "melanoma_tutorials" / "melanoma_tutorials" / "runtime" / "motif_density_analysis"
)
TISSUE_STATE_COLORS = {
    "NN": "#5B7DB1",
    "NP": "#D88452",
    "PN": "#5FA36D",
    "PP": "#9B79B6",
    "unknown": "#B8B8B8",
}
TISSUE_STATE_LIGHT_COLORS = {
    "NN": "#9DB6D8",
    "NP": "#E8A06F",
    "PN": "#8CC493",
    "PP": "#C2A5CF",
    "unknown": "#D8D8D8",
}


@dataclass(frozen=True)
class FovKey:
    patient: str
    fov: str


@dataclass
class FovData:
    key: FovKey
    graph: nx.DiGraph
    undirected_graph: nx.Graph
    cells: pd.DataFrame | None = None
    coords: np.ndarray | None = None
    cell_ids: np.ndarray | None = None
    kd_tree: cKDTree | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-path", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--motif-source-path", type=Path, default=DEFAULT_MOTIF_SOURCE_PATH)
    parser.add_argument("--motif-ids", nargs="*", type=int, default=None, help="Explicit motif IDs to analyze.")
    parser.add_argument("--raw-cells-csv", type=Path, default=DEFAULT_RAW_CELLS_CSV)
    parser.add_argument("--graph-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--patient-col", default="patient number")
    parser.add_argument("--fov-col", default="fov")
    parser.add_argument("--cell-id-col", default="cell_id")
    parser.add_argument("--x-col", default="centroid-0")
    parser.add_argument("--y-col", default="centroid-1")
    parser.add_argument("--cell-type-col", default="pred_mapped")
    parser.add_argument("--tissue-state-labels", nargs="+", default=["NN", "NP"])
    parser.add_argument("--radii", nargs="+", type=float, default=[25, 50, 100, 150, 200])
    parser.add_argument("--knn-k", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--graph-hops", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--pixel-size-microns", type=float, default=0.5)
    parser.add_argument("--patch-size", type=float, default=100)
    parser.add_argument("--boxplot-metric", default="radius_200_density")
    parser.add_argument("--boxplot-metrics", nargs="*", default=None)
    parser.add_argument("--null-samples", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--exclude-motifs", nargs="*", type=int, default=[], help="Motif IDs to skip.")
    parser.add_argument("--max-motifs", type=int, default=None, help="Limit motifs for smoke tests.")
    parser.add_argument("--max-fovs-per-motif", type=int, default=None, help="Limit FOVs per motif for smoke tests.")
    parser.add_argument("--max-hits-per-fov", type=int, default=None, help="Cap motif instances per FOV for runtime-bounded reports.")
    parser.add_argument(
        "--fov-sort",
        choices=["descending-count", "ascending-count"],
        default="descending-count",
        help="Order motif-positive FOVs before applying --max-fovs-per-motif.",
    )
    parser.add_argument("--max-patch-plots", type=int, default=6)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def normalize_patient(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    try:
        numeric = float(text)
    except ValueError:
        return text
    if numeric.is_integer():
        return str(int(numeric))
    return text


def normalize_fov(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    while text.upper().startswith("FOV"):
        text = text[3:]
    try:
        numeric = float(text)
    except ValueError:
        return text
    if numeric.is_integer():
        return str(int(numeric))
    return text


def load_selected_motif_ids(path: Path, max_motifs: int | None = None) -> list[int]:
    source = pd.read_csv(path)
    if "ID" in source.columns:
        motif_ids = source["ID"].dropna().astype(int).tolist()
    else:
        motif_ids = [
            int(column.replace("original_motif_", ""))
            for column in source.columns
            if column.startswith("original_motif_")
        ]
    motif_ids = list(dict.fromkeys(motif_ids))
    if max_motifs is not None:
        motif_ids = motif_ids[:max_motifs]
    return motif_ids


def load_source_motif_associations(path: Path, motif_ids: Iterable[int]) -> dict[int, str]:
    source = pd.read_csv(path)
    if "ID" not in source.columns:
        return {}
    association_column = None
    for candidate in ["direction", "dominant_label", "dominant_group", "patient_class", "motif_association"]:
        if candidate in source.columns:
            association_column = candidate
            break
    if association_column is None:
        return {}
    wanted = set(int(motif_id) for motif_id in motif_ids)
    associations: dict[int, str] = {}
    for row in source[source["ID"].isin(wanted)].itertuples(index=False):
        motif_id = int(getattr(row, "ID"))
        value = str(getattr(row, association_column)).strip()
        if value in TISSUE_STATE_COLORS:
            associations[motif_id] = value
    return associations


def load_artifact(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Serialized CISM artifact not found: {path}")
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_motif_graphs(motifs_df: pd.DataFrame, motif_ids: Iterable[int]) -> dict[int, nx.DiGraph]:
    graphs: dict[int, nx.DiGraph] = {}
    for motif_id in motif_ids:
        rows = motifs_df[motifs_df["ID"] == motif_id]
        if rows.empty:
            raise ValueError(f"Motif ID {motif_id} is absent from the artifact motif dataset.")
        motif = rows.iloc[0]["motif"]
        graph = motif if isinstance(motif, nx.Graph) else helpers.string_base64_pickle(motif)
        directed = nx.DiGraph(graph)
        for _, attrs in directed.nodes(data=True):
            attrs["type"] = str(attrs.get("type"))
        graphs[int(motif_id)] = directed
    return graphs


def derive_motif_associations(motifs_df: pd.DataFrame, motif_ids: Iterable[int]) -> dict[int, str]:
    relevant = motifs_df[motifs_df["ID"].isin(motif_ids) & motifs_df["patient_class"].isin(["NN", "NP"])].copy()
    if relevant.empty:
        return {int(motif_id): "unknown" for motif_id in motif_ids}
    counts = (
        relevant.groupby(["ID", "patient_class"], observed=True)["Patient_uId"]
        .nunique()
        .unstack(fill_value=0)
    )
    associations: dict[int, str] = {}
    for motif_id in motif_ids:
        row = counts.loc[motif_id] if motif_id in counts.index else pd.Series(dtype=float)
        nn_count = float(row.get("NN", 0))
        np_count = float(row.get("NP", 0))
        if nn_count == np_count == 0:
            associations[int(motif_id)] = "unknown"
        else:
            associations[int(motif_id)] = "NN" if nn_count >= np_count else "NP"
    return associations


def graph_path_for_fov(graph_dir: Path, key: FovKey) -> Path:
    return graph_dir / f"Patient_{key.patient}_FOVFOV{key.fov}.txt"


def read_fov_graph(path: Path) -> nx.DiGraph:
    if not path.exists():
        raise FileNotFoundError(f"FOV graph file not found: {path}")
    graph = nx.DiGraph()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            source, target, source_type, target_type = parts[:4]
            source_id = int(source)
            target_id = int(target)
            graph.add_node(source_id, type=str(source_type))
            graph.add_node(target_id, type=str(target_type))
            graph.add_edge(source_id, target_id)
    return graph


def load_raw_cells(args: argparse.Namespace) -> pd.DataFrame:
    if not args.raw_cells_csv.exists():
        raise FileNotFoundError(
            "Raw melanoma centroid table not found. Pass --raw-cells-csv, for example "
            "/groups/assafza_group/assafza/Datasets/Melanoma/mapped_cell_types.csv"
        )
    cells = pd.read_csv(args.raw_cells_csv)
    required = [args.patient_col, args.fov_col, args.cell_id_col, args.x_col, args.y_col, args.cell_type_col]
    missing = [column for column in required if column not in cells.columns]
    if missing:
        raise ValueError(f"Raw centroid table is missing required columns {missing}. Available: {list(cells.columns)}")
    cells = cells[required].copy()
    cells["patient_norm"] = cells[args.patient_col].map(normalize_patient)
    cells["fov_norm"] = cells[args.fov_col].map(normalize_fov)
    cells["raw_cell_id_norm"] = cells[args.cell_id_col].astype(int)
    cells["graph_node_id"] = cells.groupby(["patient_norm", "fov_norm"], sort=False).cumcount().astype(int)
    cells[args.x_col] = pd.to_numeric(cells[args.x_col], errors="coerce")
    cells[args.y_col] = pd.to_numeric(cells[args.y_col], errors="coerce")
    cells = cells.dropna(subset=[args.x_col, args.y_col])
    return cells


def build_fov_data(
    key: FovKey,
    graph_dir: Path,
    raw_cells: pd.DataFrame,
    args: argparse.Namespace,
) -> FovData:
    graph = read_fov_graph(graph_path_for_fov(graph_dir, key))
    cells = raw_cells[(raw_cells["patient_norm"] == key.patient) & (raw_cells["fov_norm"] == key.fov)].copy()
    if cells.empty:
        raise ValueError(f"No centroid rows found for patient {key.patient}, FOV {key.fov}.")
    cells = cells.drop_duplicates("graph_node_id", keep="first").set_index("graph_node_id", drop=False)
    graph = graph.subgraph(set(cells.index.astype(int))).copy()
    coords = cells[[args.x_col, args.y_col]].to_numpy(dtype=float)
    return FovData(
        key=key,
        graph=graph,
        undirected_graph=nx.Graph(graph),
        cells=cells,
        coords=coords,
        cell_ids=cells["graph_node_id"].to_numpy(dtype=int),
        kd_tree=cKDTree(coords),
    )


def motif_candidate_fovs(
    motifs_df: pd.DataFrame,
    motif_id: int,
    max_fovs: int | None,
    fov_sort: str,
    tissue_state_labels: list[str],
) -> list[FovKey]:
    rows = motifs_df[motifs_df["ID"] == motif_id].copy()
    if tissue_state_labels:
        rows = rows[rows["patient_class"].isin(tissue_state_labels)].copy()
    rows = rows.sort_values("Count", ascending=fov_sort == "ascending-count")
    rows = rows.drop_duplicates(["Patient", "FOV"])
    keys = [FovKey(normalize_patient(row.Patient), normalize_fov(row.FOV)) for row in rows.itertuples()]
    if max_fovs is not None:
        keys = keys[:max_fovs]
    return keys


def find_motif_hits(graph: nx.DiGraph, motif: nx.DiGraph, max_hits: int | None = None) -> list[tuple[int, ...]]:
    motif_nodes = list(motif.nodes())
    node_match = lambda graph_attrs, motif_attrs: str(graph_attrs.get("type")) == str(motif_attrs.get("type"))
    matcher = nx_iso.DiGraphMatcher(graph, motif, node_match=node_match)
    dedup: dict[tuple[int, ...], tuple[int, ...]] = {}
    for graph_to_motif in matcher.subgraph_monomorphisms_iter():
        motif_to_graph = {motif_node: int(graph_node) for graph_node, motif_node in graph_to_motif.items()}
        hit = tuple(motif_to_graph[motif_node] for motif_node in motif_nodes)
        dedup.setdefault(tuple(sorted(hit)), hit)
        if max_hits is not None and len(dedup) >= max_hits:
            break
    return list(dedup.values())


def radius_density(
    coords: np.ndarray,
    point: np.ndarray,
    radii: list[float],
    pixel_size_microns: float,
) -> dict[str, float]:
    if len(coords) == 0:
        return {f"radius_{int(radius)}_density": np.nan for radius in radii}
    distances = np.linalg.norm(coords - point, axis=1)
    return {
        f"radius_{int(radius)}_density": float(
            (distances <= radius).sum() / (math.pi * (radius * pixel_size_microns) ** 2)
        )
        for radius in radii
    }


def participant_knn_density(
    fov_data: FovData,
    participant_ids: set[int],
    k_values: list[int],
    args: argparse.Namespace,
) -> dict[str, float]:
    cells = fov_data.cells
    assert cells is not None
    values: dict[str, float] = {}
    available_ids = set(cells.index.astype(int))
    valid_participants = [cell_id for cell_id in participant_ids if cell_id in available_ids]
    background = cells.loc[~cells.index.isin(valid_participants)]
    if background.empty or not valid_participants:
        for k in k_values:
            values[f"knn_{k}_mean_distance"] = np.nan
            values[f"knn_{k}_density"] = np.nan
        return values
    background_coords = background[[args.x_col, args.y_col]].to_numpy(dtype=float)
    tree = cKDTree(background_coords)
    participant_coords = cells.loc[valid_participants, [args.x_col, args.y_col]].to_numpy(dtype=float)
    for k in k_values:
        if len(background_coords) < k:
            values[f"knn_{k}_mean_distance"] = np.nan
            values[f"knn_{k}_density"] = np.nan
            continue
        distances, _ = tree.query(participant_coords, k=k)
        kth_distances = distances if k == 1 else distances[:, -1]
        mean_distance = float(np.mean(kth_distances) * args.pixel_size_microns)
        values[f"knn_{k}_mean_distance"] = mean_distance
        values[f"knn_{k}_density"] = float(k / (math.pi * mean_distance * mean_distance)) if mean_distance > 0 else np.inf
    return values


def graph_hop_counts(graph: nx.Graph, participant_ids: set[int], hops: list[int]) -> dict[str, float]:
    values: dict[str, float] = {}
    valid_sources = [cell_id for cell_id in participant_ids if graph.has_node(cell_id)]
    for hop in hops:
        seen: set[int] = set()
        for source in valid_sources:
            lengths = nx.single_source_shortest_path_length(graph, source, cutoff=hop)
            seen.update(lengths.keys())
        seen.difference_update(participant_ids)
        values[f"graph_hop_{hop}_count"] = float(len(seen))
    return values


def graph_knn_density(graph: nx.Graph, participant_ids: set[int], k_values: list[int]) -> dict[str, float]:
    values: dict[str, float] = {}
    valid_sources = [cell_id for cell_id in participant_ids if graph.has_node(cell_id)]
    max_k = max(k_values) if k_values else 0

    def nearest_graph_distances(source: int) -> list[float]:
        distances: list[float] = []
        seen = {source}
        queue: list[tuple[int, int]] = [(source, 0)]
        queue_index = 0
        while queue_index < len(queue) and len(distances) < max_k:
            node, distance = queue[queue_index]
            queue_index += 1
            for neighbor in graph.neighbors(node):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                next_distance = distance + 1
                if neighbor not in participant_ids:
                    distances.append(float(next_distance))
                    if len(distances) >= max_k:
                        break
                queue.append((neighbor, next_distance))
        return distances

    source_distances = {source: nearest_graph_distances(source) for source in valid_sources}
    for k in k_values:
        kth_distances: list[float] = []
        for source in valid_sources:
            neighbor_distances = source_distances[source]
            if len(neighbor_distances) >= k:
                kth_distances.append(float(neighbor_distances[k - 1]))
        if kth_distances:
            mean_distance = float(np.mean(kth_distances))
            values[f"graph_knn_{k}_mean_distance"] = mean_distance
            values[f"graph_knn_{k}_density"] = float(k / mean_distance) if mean_distance > 0 else np.inf
        else:
            values[f"graph_knn_{k}_mean_distance"] = np.nan
            values[f"graph_knn_{k}_density"] = np.nan
    return values


def compute_cellset_metrics(
    fov_data: FovData,
    participant_ids: Iterable[int],
    radii: list[float],
    k_values: list[int],
    hops: list[int],
    args: argparse.Namespace,
) -> dict[str, float]:
    participant_set = {int(cell_id) for cell_id in participant_ids}
    cells = fov_data.cells
    coords = fov_data.coords
    assert cells is not None and coords is not None
    missing = sorted(participant_set - set(cells.index.astype(int)))
    if missing:
        raise ValueError(f"Cannot merge motif-hit cells into centroids for {fov_data.key}: {missing[:10]}")
    participant_coords = cells.loc[list(participant_set), [args.x_col, args.y_col]].to_numpy(dtype=float)
    centroid = participant_coords.mean(axis=0)
    values = {
        "motif_centroid_x": float(centroid[0]),
        "motif_centroid_y": float(centroid[1]),
        "missing_centroid_cell_count": 0,
    }
    values.update(radius_density(coords, centroid, radii, args.pixel_size_microns))
    values.update(participant_knn_density(fov_data, participant_set, k_values, args))
    values.update(graph_hop_counts(fov_data.undirected_graph, participant_set, hops))
    values.update(graph_knn_density(fov_data.undirected_graph, participant_set, k_values))
    return values


def random_connected_cellset(graph: nx.Graph, size: int, rng: np.random.Generator, max_attempts: int = 500) -> tuple[int, ...] | None:
    nodes = list(graph.nodes())
    if len(nodes) < size:
        return None
    for _ in range(max_attempts):
        start = int(rng.choice(nodes))
        selected = {start}
        frontier = set(graph.neighbors(start))
        while frontier and len(selected) < size:
            node = int(rng.choice(list(frontier)))
            selected.add(node)
            frontier.update(graph.neighbors(node))
            frontier.difference_update(selected)
        if len(selected) == size and nx.is_connected(graph.subgraph(selected)):
            return tuple(sorted(selected))
    return None


def compute_null_rows(
    instance_row: dict[str, Any],
    fov_data: FovData,
    motif_size: int,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    coords = fov_data.coords
    assert coords is not None
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    for sample_index in range(args.null_samples):
        point = rng.uniform(mins, maxs)
        random_point_values = radius_density(coords, point, args.radii, args.pixel_size_microns)
        rows.append(
            {
                "motif_id": instance_row["motif_id"],
                "motif_association": instance_row.get("motif_association", "unknown"),
                "patient": instance_row["patient"],
                "FOV": instance_row["FOV"],
                "tissue_state_filter": instance_row.get("tissue_state_filter", ""),
                "instance_id": instance_row["instance_id"],
                "null_type": "random_point",
                "sample_index": sample_index,
                "motif_centroid_x": float(point[0]),
                "motif_centroid_y": float(point[1]),
                **random_point_values,
            }
        )

        connected = random_connected_cellset(fov_data.undirected_graph, motif_size, rng)
        if connected is None:
            continue
        connected_values = compute_cellset_metrics(
            fov_data,
            connected,
            args.radii,
            args.knn_k,
            args.graph_hops,
            args,
        )
        rows.append(
            {
                "motif_id": instance_row["motif_id"],
                "motif_association": instance_row.get("motif_association", "unknown"),
                "patient": instance_row["patient"],
                "FOV": instance_row["FOV"],
                "tissue_state_filter": instance_row.get("tissue_state_filter", ""),
                "instance_id": instance_row["instance_id"],
                "null_type": "random_connected_cellset",
                "sample_index": sample_index,
                "cell_ids": "|".join(map(str, connected)),
                **connected_values,
            }
        )
    return rows


def summarize_metric(
    observed: pd.DataFrame,
    nulls: pd.DataFrame,
    metric: str,
    null_type: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    relevant_nulls = nulls[nulls["null_type"] == null_type]
    for motif_id, observed_group in observed.groupby("motif_id", observed=True):
        null_group = relevant_nulls[relevant_nulls["motif_id"] == motif_id]
        observed_values = observed_group[metric].dropna().to_numpy(dtype=float)
        null_values = null_group[metric].dropna().to_numpy(dtype=float) if metric in null_group.columns else np.array([])
        if len(observed_values) == 0 or len(null_values) == 0:
            continue
        observed_mean = float(np.mean(observed_values))
        null_mean = float(np.mean(null_values))
        delta = observed_mean - null_mean
        centered_null = np.abs(null_values - null_mean)
        p_value = float((1 + np.sum(centered_null >= abs(delta))) / (len(null_values) + 1))
        rows.append(
            {
                "motif_id": motif_id,
                "metric": metric,
                "null_type": null_type,
                "n_observed": int(len(observed_values)),
                "n_null": int(len(null_values)),
                "observed_mean": observed_mean,
                "null_mean": null_mean,
                "enrichment_delta": delta,
                "enrichment_ratio": float(observed_mean / null_mean) if null_mean != 0 else np.inf,
                "p_value": p_value,
            }
        )
    return rows


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    p = p_values.to_numpy(dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    n = len(p)
    adjusted = np.empty(n, dtype=float)
    running = 1.0
    for index in range(n - 1, -1, -1):
        running = min(running, ranked[index] * n / (index + 1))
        adjusted[order[index]] = running
    return pd.Series(np.clip(adjusted, 0, 1), index=p_values.index)


def build_summary(observed: pd.DataFrame, nulls: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        column
        for column in observed.columns
        if column.endswith("_density") or column.endswith("_count") or column.endswith("_mean_distance")
    ]
    rows: list[dict[str, Any]] = []
    for metric in metric_columns:
        if metric.startswith("radius_"):
            rows.extend(summarize_metric(observed, nulls, metric, "random_point"))
        rows.extend(summarize_metric(observed, nulls, metric, "random_connected_cellset"))
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["q_value"] = benjamini_hochberg(summary["p_value"])
        summary = summary.sort_values(["metric", "null_type", "q_value", "p_value"]).reset_index(drop=True)
    return summary


def motif_average_values(
    observed: pd.DataFrame,
    nulls: pd.DataFrame,
    metric: str,
) -> dict[str, pd.DataFrame]:
    values: dict[str, pd.DataFrame] = {}
    if metric in observed.columns and "motif_association" in observed.columns:
        values["observed"] = (
            observed.dropna(subset=[metric])
            .groupby(["motif_id", "motif_association"], as_index=False, observed=True)
            .agg(value=(metric, "mean"), n_values=(metric, "count"))
        )
    if metric in nulls.columns and "motif_association" in nulls.columns and "null_type" in nulls.columns:
        for null_type, null_group in nulls.dropna(subset=[metric]).groupby("null_type", observed=True):
            values[str(null_type)] = (
                null_group.groupby(["motif_id", "motif_association"], as_index=False, observed=True)
                .agg(value=(metric, "mean"), n_values=(metric, "count"))
            )
    return values


def smoothed_kl(values_a: np.ndarray, values_b: np.ndarray, bins: int = 8) -> tuple[float, float]:
    combined = np.concatenate([values_a, values_b])
    if len(combined) == 0 or np.nanmin(combined) == np.nanmax(combined):
        return np.nan, np.nan
    hist_a, edges = np.histogram(values_a, bins=bins, range=(float(np.nanmin(combined)), float(np.nanmax(combined))))
    hist_b, _ = np.histogram(values_b, bins=edges)
    hist_a = hist_a.astype(float) + 0.5
    hist_b = hist_b.astype(float) + 0.5
    prob_a = hist_a / hist_a.sum()
    prob_b = hist_b / hist_b.sum()
    return float(entropy(prob_a, prob_b)), float(entropy(prob_b, prob_a))


def build_motif_average_tests(observed: pd.DataFrame, nulls: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        for distribution, motif_average_df in motif_average_values(observed, nulls, metric).items():
            motif_average_df = motif_average_df[motif_average_df["motif_association"].isin(["NN", "NP"])].copy()
            nn_values = motif_average_df.loc[motif_average_df["motif_association"] == "NN", "value"].to_numpy(dtype=float)
            np_values = motif_average_df.loc[motif_average_df["motif_association"] == "NP", "value"].to_numpy(dtype=float)
            if len(nn_values) == 0 or len(np_values) == 0:
                continue
            t_result = ttest_ind(nn_values, np_values, equal_var=False, nan_policy="omit")
            try:
                mw_result = mannwhitneyu(nn_values, np_values, alternative="two-sided")
                mw_statistic = float(mw_result.statistic)
                mw_p_value = float(mw_result.pvalue)
            except ValueError:
                mw_statistic = np.nan
                mw_p_value = np.nan
            kl_nn_np, kl_np_nn = smoothed_kl(nn_values, np_values)
            rows.append(
                {
                    "metric": metric,
                    "distribution": distribution,
                    "unit": "motif_average",
                    "n_nn_motifs": int(len(nn_values)),
                    "n_np_motifs": int(len(np_values)),
                    "nn_mean": float(np.mean(nn_values)),
                    "np_mean": float(np.mean(np_values)),
                    "mean_difference_nn_minus_np": float(np.mean(nn_values) - np.mean(np_values)),
                    "welch_t_statistic": float(t_result.statistic) if pd.notna(t_result.statistic) else np.nan,
                    "welch_t_p_value": float(t_result.pvalue) if pd.notna(t_result.pvalue) else np.nan,
                    "mannwhitney_u_statistic": mw_statistic,
                    "mannwhitney_p_value": mw_p_value,
                    "kl_nn_to_np": kl_nn_np,
                    "kl_np_to_nn": kl_np_nn,
                }
            )
    tests = pd.DataFrame(rows)
    if not tests.empty:
        for column in ["welch_t_p_value", "mannwhitney_p_value"]:
            tests[f"{column.replace('_p_value', '')}_q_value"] = benjamini_hochberg(tests[column].fillna(1.0))
    return tests


def plot_outputs(observed: pd.DataFrame, nulls: pd.DataFrame, summary: pd.DataFrame, raw_cells: pd.DataFrame, args: argparse.Namespace) -> None:
    plot_dir = args.output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for stale_plot in plot_dir.glob("*.png"):
        stale_plot.unlink()
    if summary.empty:
        return

    top_rows = summary.sort_values("q_value").head(12)
    for row in top_rows.itertuples(index=False):
        metric = row.metric
        null_type = row.null_type
        motif_id = row.motif_id
        observed_values = observed.loc[observed["motif_id"] == motif_id, metric].dropna()
        null_values = nulls.loc[(nulls["motif_id"] == motif_id) & (nulls["null_type"] == null_type), metric].dropna()
        if observed_values.empty or null_values.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(null_values, bins=35, alpha=0.65, label=f"{null_type} null")
        ax.axvline(observed_values.mean(), color="black", linewidth=2, label="observed mean")
        ax.set_title(f"Motif {motif_id}: {metric}")
        ax.set_xlabel(metric)
        ax.set_ylabel("Count")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / f"observed_vs_null_{motif_id}_{metric}_{null_type}.png", dpi=160)
        plt.close(fig)

    motif_association_by_id = (
        observed.drop_duplicates("motif_id")
        .set_index("motif_id")["motif_association"]
        .to_dict()
        if "motif_association" in observed.columns
        else {}
    )
    for metric, metric_summary in summary.groupby("metric", observed=True):
        rank_data = metric_summary[metric_summary["null_type"] == "random_connected_cellset"].copy()
        if rank_data.empty:
            rank_data = metric_summary.copy()
        rank_data = rank_data.sort_values("enrichment_delta")
        bar_colors = [
            TISSUE_STATE_COLORS.get(
                motif_association_by_id.get(motif_id, "unknown"),
                TISSUE_STATE_COLORS["unknown"],
            )
            for motif_id in rank_data["motif_id"]
        ]
        fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(rank_data))))
        ax.barh(rank_data["motif_id"].astype(str), rank_data["enrichment_delta"], color=bar_colors)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_title(f"Density enrichment: {metric}")
        ax.set_xlabel("Observed mean - null mean")
        ax.set_ylabel("Motif ID")
        present_associations = [
            association
            for association in ["NN", "NP", "PN"]
            if association in set(motif_association_by_id.values())
        ]
        if present_associations:
            ax.legend(
                handles=[
                    Patch(facecolor=TISSUE_STATE_COLORS[association], edgecolor="none", label=association)
                    for association in present_associations
                ],
                frameon=False,
                loc="lower right",
                title="Motif association",
            )
        fig.tight_layout()
        fig.savefig(plot_dir / f"enrichment_{metric}.png", dpi=160)
        plt.close(fig)

    plot_patch_overlays(observed, raw_cells, args, plot_dir)
    plot_density_boxplots(observed, nulls, args, plot_dir)


def plot_density_boxplots(observed: pd.DataFrame, nulls: pd.DataFrame, args: argparse.Namespace, plot_dir: Path) -> None:
    metrics = args.boxplot_metrics if getattr(args, "boxplot_metrics", None) else [args.boxplot_metric]
    for metric in metrics:
        plot_density_boxplots_for_metric(observed, nulls, metric, plot_dir)
        plot_motif_average_boxplots_for_metric(observed, nulls, metric, plot_dir)


def plot_density_boxplots_for_metric(observed: pd.DataFrame, nulls: pd.DataFrame, metric: str, plot_dir: Path) -> None:
    if metric not in observed.columns:
        return

    observed_plot = observed.dropna(subset=[metric]).copy()
    if not observed_plot.empty:
        motif_order = [
            str(motif_id)
            for motif_id in (
            observed_plot.groupby("motif_id", observed=True)[metric]
            .median()
            .sort_values()
            .index.astype(str)
            .tolist()
            )
        ]
        observed_plot["motif_id_label"] = observed_plot["motif_id"].astype(str)
        motif_groups = [
            observed_plot.loc[observed_plot["motif_id_label"] == motif_label, metric].to_numpy(dtype=float)
            for motif_label in motif_order
        ]
        colors = [
            TISSUE_STATE_COLORS.get(
                observed_plot.loc[observed_plot["motif_id_label"] == motif_label, "motif_association"].iloc[0],
                TISSUE_STATE_COLORS["unknown"],
            )
            for motif_label in motif_order
        ]
        fig, ax = plt.subplots(figsize=(12, 5))
        draw_styled_boxplot(ax, motif_groups, motif_order, colors)
        add_jittered_points(ax, motif_groups)
        ax.set_title(f"Observed local density by motif ({metric})")
        ax.set_xlabel("Motif ID")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=90)
        fig.tight_layout()
        fig.savefig(plot_dir / f"boxplot_observed_{metric}_by_motif.png", dpi=180)
        plt.close(fig)

        if "motif_association" in observed_plot.columns:
            assoc_plot = observed_plot[observed_plot["motif_association"].isin(["NN", "NP"])].copy()
            if not assoc_plot.empty:
                labels = [label for label in ["NN", "NP"] if (assoc_plot["motif_association"] == label).any()]
                groups = [
                    assoc_plot.loc[assoc_plot["motif_association"] == label, metric].to_numpy(dtype=float)
                    for label in labels
                ]
                colors = [TISSUE_STATE_COLORS[label] for label in labels]
                fig, ax = plt.subplots(figsize=(5, 4))
                draw_styled_boxplot(ax, groups, labels, colors)
                add_jittered_points(ax, groups)
                ax.set_title(f"Observed local density by association\n({metric})")
                ax.set_xlabel("Motif association")
                ax.set_ylabel(metric)
                fig.tight_layout()
                fig.savefig(plot_dir / f"boxplot_observed_{metric}_by_association.png", dpi=180)
                plt.close(fig)

    if metric in nulls.columns and "motif_association" in nulls.columns:
        null_plot = nulls.dropna(subset=[metric]).copy()
        if not null_plot.empty:
            null_plot["motif_id_label"] = null_plot["motif_id"].astype(str)
            motif_order = (
                null_plot.groupby("motif_id_label", observed=True)[metric]
                .median()
                .sort_values()
                .index
                .tolist()
            )
            motif_groups = [
                null_plot.loc[null_plot["motif_id_label"] == motif_label, metric].to_numpy(dtype=float)
                for motif_label in motif_order
            ]
            if motif_groups:
                colors = [
                    TISSUE_STATE_LIGHT_COLORS.get(
                        null_plot.loc[null_plot["motif_id_label"] == motif_label, "motif_association"].iloc[0],
                        TISSUE_STATE_LIGHT_COLORS["unknown"],
                    )
                    for motif_label in motif_order
                ]
                fig, ax = plt.subplots(figsize=(12, 5))
                draw_styled_boxplot(ax, motif_groups, motif_order, colors)
                ax.set_title(f"Null local density by motif\n({metric})")
                ax.set_xlabel("Motif ID")
                ax.set_ylabel(metric)
                ax.tick_params(axis="x", rotation=90)
                fig.tight_layout()
                fig.savefig(plot_dir / f"boxplot_null_{metric}_by_motif.png", dpi=180)
                plt.close(fig)

        null_plot = null_plot[null_plot["motif_association"].isin(["NN", "NP"])]
        if not null_plot.empty:
            groups = []
            labels = []
            colors = []
            for association in ["NN", "NP"]:
                for null_type in ["random_point", "random_connected_cellset"]:
                    values = null_plot[
                        (null_plot["motif_association"] == association)
                        & (null_plot["null_type"] == null_type)
                    ][metric].dropna()
                    if values.empty:
                        continue
                    groups.append(values.to_numpy(dtype=float))
                    labels.append(f"{association}\n{null_type.replace('random_', '').replace('_', ' ')}")
                    color_map = TISSUE_STATE_LIGHT_COLORS if null_type == "random_point" else TISSUE_STATE_COLORS
                    colors.append(color_map[association])
            if groups:
                fig, ax = plt.subplots(figsize=(8, 4.5))
                draw_styled_boxplot(ax, groups, labels, colors)
                ax.set_title(f"Null local density by association\n({metric})")
                ax.set_xlabel("Motif association / null model")
                ax.set_ylabel(metric)
                fig.tight_layout()
                fig.savefig(plot_dir / f"boxplot_null_{metric}_by_association.png", dpi=180)
                plt.close(fig)

        plot_observed_and_null_by_association(
            observed_plot,
            null_plot,
            metric,
            "random_connected_cellset",
            "connected null",
            "connected_null",
            plot_dir,
        )
        plot_observed_and_null_by_association(
            observed_plot,
            null_plot,
            metric,
            "random_point",
            "point null",
            "point_null",
            plot_dir,
        )


def plot_observed_and_null_by_association(
    observed_plot: pd.DataFrame,
    null_plot: pd.DataFrame,
    metric: str,
    null_type: str,
    null_label: str,
    filename_label: str,
    plot_dir: Path,
) -> None:
    if observed_plot.empty or null_plot.empty:
        return
    if "motif_association" not in observed_plot.columns or "motif_association" not in null_plot.columns:
        return

    selected_null = null_plot[null_plot["null_type"] == null_type].copy()
    groups: list[np.ndarray] = []
    labels: list[str] = []
    colors: list[str] = []
    positions: list[float] = []

    for position, association in zip([1.0, 2.0], ["NN", "NP"]):
        values = observed_plot.loc[observed_plot["motif_association"] == association, metric].dropna()
        if values.empty:
            continue
        groups.append(values.to_numpy(dtype=float))
        labels.append(f"{association}\nobserved")
        colors.append(TISSUE_STATE_COLORS[association])
        positions.append(position)

    for position, association in zip([3.4, 4.4], ["NN", "NP"]):
        values = selected_null.loc[selected_null["motif_association"] == association, metric].dropna()
        if values.empty:
            continue
        groups.append(values.to_numpy(dtype=float))
        labels.append(f"{association}\n{null_label}")
        colors.append(TISSUE_STATE_LIGHT_COLORS[association])
        positions.append(position)

    if len(groups) < 2:
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    draw_styled_boxplot(ax, groups, labels, colors, positions=positions)
    add_jittered_points(ax, groups, positions=positions)
    ax.axvline(2.7, linestyle="--", color="#A8A8A8", linewidth=1.0)
    ax.set_title(f"Observed and {null_label} local density by association\n({metric})")
    ax.set_xlabel("Motif association / distribution")
    ax.set_ylabel(metric)
    fig.tight_layout()
    fig.savefig(plot_dir / f"boxplot_observed_{filename_label}_{metric}_by_association.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    draw_styled_boxplot(ax, groups, labels, colors, positions=positions)
    ax.axvline(2.7, linestyle="--", color="#A8A8A8", linewidth=1.0)
    ax.set_title(f"Observed and {null_label} local density by association\n({metric})")
    ax.set_xlabel("Motif association / distribution")
    ax.set_ylabel(metric)
    fig.tight_layout()
    fig.savefig(plot_dir / f"boxplot_observed_{filename_label}_{metric}_by_association_no_scatter.png", dpi=180)
    plt.close(fig)


def plot_motif_average_boxplots_for_metric(observed: pd.DataFrame, nulls: pd.DataFrame, metric: str, plot_dir: Path) -> None:
    distributions = motif_average_values(observed, nulls, metric)
    observed_average = distributions.get("observed", pd.DataFrame())
    if not observed_average.empty:
        plot_motif_average_association_boxplot(
            observed_average,
            metric,
            f"Motif-average observed local density by association\n({metric})",
            plot_dir / f"boxplot_motif_average_observed_{metric}_by_association.png",
        )

    for null_type, null_label, filename_label in [
        ("random_connected_cellset", "connected null", "connected_null"),
        ("random_point", "point null", "point_null"),
    ]:
        null_average = distributions.get(null_type, pd.DataFrame())
        if observed_average.empty or null_average.empty:
            continue
        plot_motif_average_observed_and_null(
            observed_average,
            null_average,
            metric,
            null_label,
            filename_label,
            plot_dir,
            include_points=True,
        )
        plot_motif_average_observed_and_null(
            observed_average,
            null_average,
            metric,
            null_label,
            filename_label,
            plot_dir,
            include_points=False,
        )


def motif_average_groups(plot_df: pd.DataFrame) -> tuple[list[np.ndarray], list[str], list[str]]:
    groups: list[np.ndarray] = []
    labels: list[str] = []
    colors: list[str] = []
    for association in ["NN", "NP"]:
        values = plot_df.loc[plot_df["motif_association"] == association, "value"].dropna()
        if values.empty:
            continue
        groups.append(values.to_numpy(dtype=float))
        labels.append(association)
        colors.append(TISSUE_STATE_COLORS[association])
    return groups, labels, colors


def plot_motif_average_association_boxplot(plot_df: pd.DataFrame, metric: str, title: str, output_path: Path) -> None:
    plot_df = plot_df[plot_df["motif_association"].isin(["NN", "NP"])].copy()
    groups, labels, colors = motif_average_groups(plot_df)
    if len(groups) < 2:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    draw_styled_boxplot(ax, groups, labels, colors)
    add_jittered_points(ax, groups)
    ax.set_title(title)
    ax.set_xlabel("Motif association")
    ax.set_ylabel(f"{metric}\n(mean per motif)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_motif_average_observed_and_null(
    observed_average: pd.DataFrame,
    null_average: pd.DataFrame,
    metric: str,
    null_label: str,
    filename_label: str,
    plot_dir: Path,
    include_points: bool,
) -> None:
    groups: list[np.ndarray] = []
    labels: list[str] = []
    colors: list[str] = []
    positions: list[float] = []
    for position, association in zip([1.0, 2.0], ["NN", "NP"]):
        values = observed_average.loc[observed_average["motif_association"] == association, "value"].dropna()
        if values.empty:
            continue
        groups.append(values.to_numpy(dtype=float))
        labels.append(f"{association}\nobserved")
        colors.append(TISSUE_STATE_COLORS[association])
        positions.append(position)
    for position, association in zip([3.4, 4.4], ["NN", "NP"]):
        values = null_average.loc[null_average["motif_association"] == association, "value"].dropna()
        if values.empty:
            continue
        groups.append(values.to_numpy(dtype=float))
        labels.append(f"{association}\n{null_label}")
        colors.append(TISSUE_STATE_LIGHT_COLORS[association])
        positions.append(position)
    if len(groups) < 2:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    draw_styled_boxplot(ax, groups, labels, colors, positions=positions)
    if include_points:
        add_jittered_points(ax, groups, positions=positions)
    ax.axvline(2.7, linestyle="--", color="#A8A8A8", linewidth=1.0)
    ax.set_title(f"Motif-average observed and {null_label} local density\n({metric})")
    ax.set_xlabel("Motif association / distribution")
    ax.set_ylabel(f"{metric}\n(mean per motif)")
    fig.tight_layout()
    suffix = "" if include_points else "_no_scatter"
    fig.savefig(plot_dir / f"boxplot_motif_average_observed_{filename_label}_{metric}_by_association{suffix}.png", dpi=180)
    plt.close(fig)


def draw_styled_boxplot(
    ax: plt.Axes,
    groups: list[np.ndarray],
    labels: list[str],
    colors: list[str],
    positions: list[float] | None = None,
) -> None:
    boxplot = ax.boxplot(
        groups,
        positions=positions,
        tick_labels=labels,
        patch_artist=True,
        showfliers=False,
        widths=0.55,
        medianprops={"color": "#5A5A5A", "linewidth": 1.2},
        whiskerprops={"color": "#A8A8A8", "linewidth": 1.0},
        capprops={"color": "#A8A8A8", "linewidth": 1.0},
        boxprops={"edgecolor": "#666666", "linewidth": 1.0},
    )
    for patch, color in zip(boxplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.92)
    ax.grid(axis="y", color="#D0D0D0", linewidth=0.6, alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_jittered_points(ax: plt.Axes, groups: list[np.ndarray], positions: list[float] | None = None) -> None:
    rng = np.random.default_rng(0)
    if positions is None:
        positions = list(range(1, len(groups) + 1))
    for position, values in zip(positions, groups):
        if len(values) == 0:
            continue
        jitter = rng.normal(0, 0.035, size=len(values))
        ax.scatter(
            np.full(len(values), position) + jitter,
            values,
            s=10,
            c="#303030",
            alpha=0.75,
            linewidths=0,
            zorder=3,
        )


def plot_patch_overlays(observed: pd.DataFrame, raw_cells: pd.DataFrame, args: argparse.Namespace, plot_dir: Path) -> None:
    rich_fovs = (
        observed.groupby(["patient", "FOV"], observed=True)
        .size()
        .sort_values(ascending=False)
        .head(args.max_patch_plots)
    )
    for (patient, fov), _ in rich_fovs.items():
        cells = raw_cells[(raw_cells["patient_norm"] == str(patient)) & (raw_cells["fov_norm"] == str(fov))]
        instances = observed[(observed["patient"] == str(patient)) & (observed["FOV"] == str(fov))]
        if cells.empty or instances.empty:
            continue
        x = cells[args.x_col].to_numpy(dtype=float)
        y = cells[args.y_col].to_numpy(dtype=float)
        x_edges = np.arange(x.min(), x.max() + args.patch_size, args.patch_size)
        y_edges = np.arange(y.min(), y.max() + args.patch_size, args.patch_size)
        density, _, _ = np.histogram2d(y, x, bins=[y_edges, x_edges])
        fig, ax = plt.subplots(figsize=(7, 6))
        mesh = ax.pcolormesh(x_edges, y_edges, density, cmap="viridis", shading="auto")
        fig.colorbar(mesh, ax=ax, label="Cells per 100 x 100 patch")
        patch_counts = (
            instances.assign(
                patch_x=((instances["motif_centroid_x"] - x.min()) // args.patch_size).astype(int),
                patch_y=((instances["motif_centroid_y"] - y.min()) // args.patch_size).astype(int),
            )
            .groupby(["patch_x", "patch_y"], observed=True)
            .size()
            .reset_index(name="motif_count")
        )
        dot_x = x.min() + (patch_counts["patch_x"] + 0.5) * args.patch_size
        dot_y = y.min() + (patch_counts["patch_y"] + 0.5) * args.patch_size
        ax.scatter(dot_x, dot_y, s=20 + 16 * patch_counts["motif_count"], c="white", edgecolors="black", linewidths=0.8)
        ax.set_title(f"Patient {patient}, FOV {fov}: patch density and motif locations")
        ax.set_xlabel(args.x_col)
        ax.set_ylabel(args.y_col)
        ax.set_aspect("equal", adjustable="box")
        fig.tight_layout()
        fig.savefig(plot_dir / f"patch_density_overlay_patient_{patient}_fov_{fov}.png", dpi=180)
        plt.close(fig)


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    motif_ids = list(dict.fromkeys(args.motif_ids)) if args.motif_ids else load_selected_motif_ids(args.motif_source_path, args.max_motifs)
    if args.max_motifs is not None and args.motif_ids:
        motif_ids = motif_ids[: args.max_motifs]
    if args.exclude_motifs:
        motif_ids = [motif_id for motif_id in motif_ids if motif_id not in set(args.exclude_motifs)]
    artifact = load_artifact(args.artifact_path)
    motifs_df = artifact["cism"].motifs_dataset.copy()
    motif_graphs = load_motif_graphs(motifs_df, motif_ids)
    motif_associations = derive_motif_associations(motifs_df, motif_ids)
    motif_associations.update(load_source_motif_associations(args.motif_source_path, motif_ids))
    raw_cells = load_raw_cells(args)
    rng = np.random.default_rng(args.random_state)

    fov_cache: dict[FovKey, FovData] = {}
    instance_rows: list[dict[str, Any]] = []
    null_rows: list[dict[str, Any]] = []
    iterator = tqdm(motif_ids, desc="Motifs", disable=args.no_progress)
    for motif_id in iterator:
        motif_graph = motif_graphs[motif_id]
        fov_keys = motif_candidate_fovs(
            motifs_df,
            motif_id,
            args.max_fovs_per_motif,
            args.fov_sort,
            args.tissue_state_labels,
        )
        for key in tqdm(fov_keys, desc=f"Motif {motif_id} FOVs", leave=False, disable=args.no_progress):
            fov_data = fov_cache.get(key)
            if fov_data is None:
                fov_data = build_fov_data(key, args.graph_dir, raw_cells, args)
                fov_cache[key] = fov_data
            hits = find_motif_hits(fov_data.graph, motif_graph, max_hits=args.max_hits_per_fov)
            for hit_index, hit in enumerate(hits):
                metrics = compute_cellset_metrics(fov_data, hit, args.radii, args.knn_k, args.graph_hops, args)
                cell_types = [
                    str(fov_data.cells.loc[int(cell_id), args.cell_type_col])
                    for cell_id in hit
                    if fov_data.cells is not None and int(cell_id) in fov_data.cells.index
                ]
                raw_cell_ids = [
                    int(fov_data.cells.loc[int(cell_id), "raw_cell_id_norm"])
                    for cell_id in hit
                    if fov_data.cells is not None and int(cell_id) in fov_data.cells.index
                ]
                instance_id = f"{motif_id}:{key.patient}:{key.fov}:{hit_index}"
                row = {
                    "motif_id": motif_id,
                    "motif_association": motif_associations.get(motif_id, "unknown"),
                    "patient": key.patient,
                    "FOV": key.fov,
                    "tissue_state_filter": "|".join(args.tissue_state_labels),
                    "instance_id": instance_id,
                    "motif_size": len(hit),
                    "graph_node_ids": "|".join(map(str, hit)),
                    "raw_cell_ids": "|".join(map(str, raw_cell_ids)),
                    "cell_ids": "|".join(map(str, hit)),
                    "cell_type_composition": "|".join(cell_types),
                    **metrics,
                }
                instance_rows.append(row)
                null_rows.extend(compute_null_rows(row, fov_data, len(hit), args, rng))

    instances = pd.DataFrame(instance_rows)
    nulls = pd.DataFrame(null_rows)
    if instances.empty:
        raise ValueError("No motif instances were recovered. Check motif IDs, graph files, and motif matching.")
    summary = build_summary(instances, nulls)
    test_metrics = args.boxplot_metrics if getattr(args, "boxplot_metrics", None) else [args.boxplot_metric]
    motif_average_tests = build_motif_average_tests(instances, nulls, test_metrics)
    instances.to_csv(args.output_dir / "motif_density_instance_metrics.csv", index=False)
    nulls.to_csv(args.output_dir / "motif_density_null_samples.csv", index=False)
    summary.to_csv(args.output_dir / "motif_density_summary.csv", index=False)
    motif_average_tests.to_csv(args.output_dir / "motif_density_motif_average_tests.csv", index=False)
    if not args.no_plots:
        plot_outputs(instances, nulls, summary, raw_cells, args)
    return instances, nulls, summary


def main() -> None:
    args = parse_args()
    instances, nulls, summary = run(args)
    print(f"Saved {len(instances):,} motif instance rows to {args.output_dir / 'motif_density_instance_metrics.csv'}")
    print(f"Saved {len(nulls):,} null rows to {args.output_dir / 'motif_density_null_samples.csv'}")
    print(f"Saved {len(summary):,} summary rows to {args.output_dir / 'motif_density_summary.csv'}")


if __name__ == "__main__":
    main()
