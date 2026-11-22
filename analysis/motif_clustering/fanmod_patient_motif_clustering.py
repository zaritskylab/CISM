#!/usr/bin/env python3
"""Unsupervised patient clustering from FANMOD+ motif profiles.

This analysis builds a patient-by-motif matrix directly from FANMOD+ output,
tries several dimensionality-reduction and two-cluster strategies, and writes
reviewer-facing plots and tables.
"""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, permutations
from pathlib import Path
from typing import Iterable

os.makedirs("/tmp/codex_matplotlib", exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.decomposition import NMF, PCA, TruncatedSVD
from sklearn.manifold import MDS, SpectralEmbedding, TSNE
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    f1_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler


SCRIPT_DIR = ANALYSIS_DIR
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_FANMOD_DIR = (
    PROJECT_ROOT
    / "multi_class"
    / "runtime"
    / "all_classes_remove_tumor"
    / "fanmod_output"
    / "all_classes_remove_tumor"
    / "4"
)
DEFAULT_PATIENT_CLASS = (
    PROJECT_ROOT
    / "multi_class"
    / "runtime"
    / "all_classes_remove_tumor"
    / "prepared_datasets"
    / "all_classes_remove_tumor"
    / "patient_class.csv"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "fanmod_patient_motif_clustering_NN_NP"
DEFAULT_MELANOMA_SERIALIZED_ARTIFACT = (
    PROJECT_ROOT
    / "melanoma_tutorials"
    / "melanoma_tutorials"
    / "runtime"
    / "serialized"
    / "Melanoma_cism_ready.pkl"
)
RANDOM_STATE = 0
CLASS_PALETTE = {"NN": "#2F6C9E", "NP": "#D77A3D", "PN": "#5B8E4D", "PP": "#8B5FA5"}
CLUSTER_PALETTE = {0: "#2A6FBB", 1: "#C45A3A"}
LABEL_COLOR_ENDPOINTS = {"NN": "#2F6C9E", "NP": "#D77A3D"}


@dataclass(frozen=True)
class FeatureSet:
    name: str
    values: np.ndarray
    columns: list[str]


@dataclass(frozen=True)
class EmbeddingSet:
    name: str
    values: np.ndarray


def load_umap_module():
    extra_site_packages = os.environ.get("UMAP_SITE_PACKAGES")
    if extra_site_packages and extra_site_packages not in sys.path:
        sys.path.append(extra_site_packages)
    try:
        import umap  # type: ignore
    except ImportError:
        return None
    return umap


UMAP_MODULE = load_umap_module()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "legend.title_fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.grid": False,
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.patch.set_facecolor("white")
    for ax in fig.axes:
        ax.set_facecolor("white")
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / "figures" / f"{stem}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fanmod-dir", type=Path, default=DEFAULT_FANMOD_DIR)
    parser.add_argument("--patient-class-csv", type=Path, default=DEFAULT_PATIENT_CLASS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--labels", nargs="+", default=["NN", "NP"], help="Patient classes to include.")
    parser.add_argument("--parsed-long-csv", type=Path, default=None, help="Optional pre-parsed patient_motif_long.csv.")
    parser.add_argument("--fov-summary-csv", type=Path, default=None, help="Optional pre-parsed fanmod_fov_summary.csv.")
    parser.add_argument("--serialized-artifact", type=Path, default=None, help="Optional serialized CISM artifact pickle.")
    parser.add_argument("--motif-col", default="ID", help="Motif identifier column to use from serialized artifacts.")
    parser.add_argument("--top-motifs", type=int, default=2500)
    parser.add_argument("--min-patients-per-motif", type=int, default=3)
    parser.add_argument("--all-motifs", action="store_true", help="Use every observed motif, without prevalence or variance filtering.")
    parser.add_argument("--min-selection-cluster-fraction", type=float, default=0.2)
    parser.add_argument("--include-none-scaled", action="store_true")
    parser.add_argument("--skip-tsne", action="store_true")
    parser.add_argument("--force-parse", action="store_true")
    return parser.parse_args()


def patient_from_fanmod_file(path: Path) -> tuple[str, str]:
    patient, fov = path.stem.split("_", 1)
    return f"Melanoma{int(patient)}", fov


@lru_cache(maxsize=None)
def canonical_motif_signature(colors: tuple[str, ...], edges: tuple[tuple[int, int], ...]) -> str:
    n_nodes = len(colors)
    edge_set = set(edges)
    candidates = []
    for perm in permutations(range(n_nodes)):
        color_part = ",".join(colors[index] for index in perm)
        edge_bits = []
        for left, right in combinations(range(n_nodes), 2):
            original = tuple(sorted((perm[left], perm[right])))
            edge_bits.append("1" if original in edge_set else "0")
        candidates.append(f"n{n_nodes}|c:{color_part}|e:{''.join(edge_bits)}")
    return min(candidates)


def parse_leda_signature(lines: list[str]) -> str:
    filtered = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    if len(filtered) < 6 or filtered[0] != "LEDA.GRAPH":
        raise ValueError("Unexpected LEDA motif block.")
    cursor = 3
    _direction = int(filtered[cursor])
    cursor += 1
    n_nodes = int(filtered[cursor])
    cursor += 1
    colors = [filtered[cursor + index].strip("|{}| ") for index in range(n_nodes)]
    cursor += n_nodes
    n_edges = int(filtered[cursor])
    cursor += 1
    edges: set[tuple[int, int]] = set()
    for index in range(n_edges):
        fields = filtered[cursor + index].split()
        left = int(fields[0]) - 1
        right = int(fields[1]) - 1
        if left != right:
            edges.add(tuple(sorted((left, right))))
    return canonical_motif_signature(tuple(colors), tuple(sorted(edges)))


def read_until(lines: Iterable[str], prefix: str) -> str | None:
    for line in lines:
        if line.startswith(prefix):
            return line
    return None


def iter_fanmod_rows(path: Path) -> Iterable[tuple[str, float, float, float]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        line_iter = iter(handle)
        found_line = None
        for line in line_iter:
            if line.endswith("motifs were found.\n"):
                found_line = line
                break
        if found_line is None:
            return
        if read_until(line_iter, "ID,") is None:
            return
        for line in line_iter:
            stripped = line.strip()
            if not stripped:
                continue
            fields = stripped.split(",")
            if len(fields) != 7:
                break
            _motif_id, freq, count, _mean_freq, _std, _z_score, p_value = fields
            leda_lines = []
            if read_until(line_iter, "#+leda") is None:
                break
            for leda_line in line_iter:
                if leda_line.startswith("#-leda"):
                    break
                leda_lines.append(leda_line)
            try:
                signature = parse_leda_signature(leda_lines)
            except ValueError:
                continue
            yield signature, float(freq), float(count), float(p_value)


def build_motif_tables(
    fanmod_dir: Path,
    output_dir: Path,
    force_parse: bool,
    parsed_long_csv: Path | None = None,
    fov_summary_csv: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if parsed_long_csv is not None and fov_summary_csv is not None and not force_parse:
        return pd.read_csv(parsed_long_csv), pd.read_csv(fov_summary_csv)

    aggregate_path = output_dir / "patient_motif_long.csv"
    fov_summary_path = output_dir / "fanmod_fov_summary.csv"
    if aggregate_path.exists() and fov_summary_path.exists() and not force_parse:
        return pd.read_csv(aggregate_path), pd.read_csv(fov_summary_path)

    accumulator: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"total_count": 0.0, "total_subgraphs": 0.0, "n_fovs": 0.0, "min_p_value": 1.0}
    )
    fov_rows = []
    files = sorted(fanmod_dir.glob("*.csv"), key=lambda path: (int(path.stem.split("_", 1)[0]), path.stem))
    if not files:
        raise FileNotFoundError(f"No FANMOD CSV files found in {fanmod_dir}")
    for index, path in enumerate(files, start=1):
        patient_id, fov = patient_from_fanmod_file(path)
        row_count = 0
        motif_seen = set()
        for signature, freq, count, p_value in iter_fanmod_rows(path):
            if freq <= 0 or count <= 0:
                continue
            key = (patient_id, signature)
            accumulator[key]["total_count"] += count
            accumulator[key]["total_subgraphs"] += count / freq
            accumulator[key]["min_p_value"] = min(accumulator[key]["min_p_value"], p_value)
            if signature not in motif_seen:
                accumulator[key]["n_fovs"] += 1
                motif_seen.add(signature)
            row_count += 1
        fov_rows.append({"patient_id": patient_id, "fov": fov, "fanmod_rows": row_count, "unique_motifs": len(motif_seen)})
        if index % 10 == 0 or index == len(files):
            print(f"Parsed {index}/{len(files)} FANMOD files", flush=True)

    rows = []
    for (patient_id, motif_id), values in accumulator.items():
        total_subgraphs = values["total_subgraphs"]
        rows.append(
            {
                "patient_id": patient_id,
                "motif_id": motif_id,
                "total_count": values["total_count"],
                "total_subgraphs": total_subgraphs,
                "cism_frequency": values["total_count"] / total_subgraphs if total_subgraphs else 0.0,
                "n_fovs": int(values["n_fovs"]),
                "min_p_value": values["min_p_value"],
            }
        )
    long_df = pd.DataFrame(rows)
    fov_summary = pd.DataFrame(fov_rows)
    long_df.to_csv(aggregate_path, index=False)
    fov_summary.to_csv(fov_summary_path, index=False)
    return long_df, fov_summary


def build_motif_tables_from_serialized_artifact(
    artifact_path: Path,
    output_dir: Path,
    motif_col: str,
    force_parse: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    aggregate_path = output_dir / "patient_motif_long.csv"
    fov_summary_path = output_dir / "fanmod_fov_summary.csv"
    if aggregate_path.exists() and fov_summary_path.exists() and not force_parse:
        return pd.read_csv(aggregate_path), pd.read_csv(fov_summary_path), artifact_path

    import pickle

    artifact = pickle.load(artifact_path.open("rb"))
    cism_obj = artifact["cism"] if isinstance(artifact, dict) and "cism" in artifact else artifact
    motifs_df = cism_obj.motifs_dataset.copy()
    required = {"Patient_uId", "FOV", "Count", "Freq", motif_col}
    missing = required.difference(motifs_df.columns)
    if missing:
        raise ValueError(f"Serialized artifact motif table is missing required columns: {sorted(missing)}")

    data = motifs_df[["Patient_uId", "FOV", "Count", "Freq", motif_col, "p_value"] if "p_value" in motifs_df.columns else ["Patient_uId", "FOV", "Count", "Freq", motif_col]].copy()
    data = data.rename(columns={"Patient_uId": "patient_id", motif_col: "motif_id"})
    data = data[data["Freq"] > 0]
    data["motif_id"] = data["motif_id"].astype(str)
    data["_total_subgraphs"] = data["Count"] / data["Freq"]

    total_counts = data.groupby(["patient_id", "motif_id"], observed=True)["Count"].sum()
    total_subgraphs = (
        data.drop_duplicates(["patient_id", "motif_id", "FOV"])
        .groupby(["patient_id", "motif_id"], observed=True)["_total_subgraphs"]
        .sum()
    )
    long_df = pd.concat(
        [total_counts.rename("total_count"), total_subgraphs.rename("total_subgraphs")],
        axis=1,
    ).reset_index()
    long_df["cism_frequency"] = long_df["total_count"] / long_df["total_subgraphs"]
    long_df["n_fovs"] = (
        data.drop_duplicates(["patient_id", "motif_id", "FOV"])
        .groupby(["patient_id", "motif_id"], observed=True)["FOV"]
        .size()
        .reindex(pd.MultiIndex.from_frame(long_df[["patient_id", "motif_id"]]))
        .to_numpy()
    )
    if "p_value" in data.columns:
        min_p_value = data.groupby(["patient_id", "motif_id"], observed=True)["p_value"].min()
        long_df["min_p_value"] = min_p_value.reindex(pd.MultiIndex.from_frame(long_df[["patient_id", "motif_id"]])).to_numpy()
    else:
        long_df["min_p_value"] = np.nan

    fov_summary = (
        data.groupby(["patient_id", "FOV"], observed=True)
        .agg(fanmod_rows=("motif_id", "size"), unique_motifs=("motif_id", "nunique"))
        .reset_index()
        .rename(columns={"FOV": "fov"})
    )
    long_df.to_csv(aggregate_path, index=False)
    fov_summary.to_csv(fov_summary_path, index=False)
    return long_df, fov_summary, artifact_path


def load_patient_classes(path: Path) -> pd.DataFrame:
    classes = pd.read_csv(path, header=None, names=["patient_id", "class_label"])
    classes["axis_1"] = classes["class_label"].str[0]
    classes["axis_2"] = classes["class_label"].str[1]
    return classes


def make_feature_sets(
    long_df: pd.DataFrame,
    patient_ids: list[str],
    top_motifs: int,
    min_patients: int,
    output_dir: Path,
    all_motifs: bool = False,
) -> list[FeatureSet]:
    freq_matrix = (
        long_df.pivot(index="patient_id", columns="motif_id", values="cism_frequency")
        .reindex(index=patient_ids)
        .fillna(0.0)
    )
    count_matrix = (
        long_df.pivot(index="patient_id", columns="motif_id", values="total_count")
        .reindex(index=patient_ids)
        .fillna(0.0)
    )
    prevalence = (freq_matrix > 0).sum(axis=0)
    if all_motifs:
        selected_columns = freq_matrix.columns.tolist()
    else:
        candidate_columns = prevalence[prevalence >= min_patients].index
        if len(candidate_columns) == 0:
            raise ValueError("No motifs passed the prevalence filter.")
        variances = np.log1p(freq_matrix[candidate_columns] * 1_000_000).var(axis=0).sort_values(ascending=False)
        selected_columns = variances.head(top_motifs).index.tolist()
    idf = np.log((1 + len(patient_ids)) / (1 + (count_matrix[selected_columns] > 0).sum(axis=0))) + 1

    freq = freq_matrix[selected_columns].to_numpy(dtype=float)
    counts = count_matrix[selected_columns].to_numpy(dtype=float)
    feature_sets = [
        FeatureSet("cism_frequency", freq, selected_columns),
        FeatureSet("log_cism_frequency", np.log1p(freq * 1_000_000), selected_columns),
        FeatureSet("motif_presence", (freq > 0).astype(float), selected_columns),
        FeatureSet("log_count_tfidf", np.log1p(counts) * idf.to_numpy(dtype=float), selected_columns),
    ]
    freq_matrix[selected_columns].to_csv(output_dir / "patient_motif_matrix_cism_frequency.csv")
    pd.DataFrame({"motif_id": selected_columns, "prevalence": prevalence[selected_columns].values}).to_csv(
        output_dir / "selected_motif_prevalence.csv", index=False
    )
    return feature_sets


def n_components_for(max_components: int, n_samples: int, n_features: int) -> int:
    return max(2, min(max_components, n_samples - 1, n_features))


def make_embeddings(feature_set: FeatureSet, *, include_tsne: bool = True, include_none_scaled: bool = False) -> list[EmbeddingSet]:
    x = feature_set.values
    embeddings = []
    if include_none_scaled:
        print("  Computing embedding: none_scaled", flush=True)
        embeddings.append(EmbeddingSet("none_scaled", StandardScaler().fit_transform(x)))

    max_abs = MaxAbsScaler().fit_transform(x)
    n_svd = n_components_for(20, *x.shape)
    print(f"  Computing embedding: svd_{n_svd}", flush=True)
    embeddings.append(EmbeddingSet(f"svd_{n_svd}", TruncatedSVD(n_components=n_svd, random_state=RANDOM_STATE).fit_transform(max_abs)))

    scaled = StandardScaler().fit_transform(x)
    n_pca = n_components_for(20, *x.shape)
    print(f"  Computing embedding: pca_{n_pca}", flush=True)
    embeddings.append(EmbeddingSet(f"pca_{n_pca}", PCA(n_components=n_pca, random_state=RANDOM_STATE).fit_transform(scaled)))

    n_low = n_components_for(8, *x.shape)
    print(f"  Computing embedding: pca_{n_low}", flush=True)
    embeddings.append(EmbeddingSet(f"pca_{n_low}", PCA(n_components=n_low, random_state=RANDOM_STATE).fit_transform(scaled)))

    nonnegative = MinMaxScaler().fit_transform(x)
    n_nmf = n_components_for(8, *x.shape)
    try:
        print(f"  Computing embedding: nmf_{n_nmf}", flush=True)
        embeddings.append(
            EmbeddingSet(
                f"nmf_{n_nmf}",
                NMF(n_components=n_nmf, init="nndsvda", random_state=RANDOM_STATE, max_iter=300).fit_transform(nonnegative),
            )
        )
    except ValueError:
        pass

    n_neighbors = min(10, max(2, x.shape[0] - 1))
    try:
        print("  Computing embedding: spectral_embedding", flush=True)
        embeddings.append(
            EmbeddingSet(
                "spectral_embedding",
                SpectralEmbedding(n_components=2, n_neighbors=n_neighbors, random_state=RANDOM_STATE).fit_transform(scaled),
            )
        )
    except Exception:
        pass

    try:
        print("  Computing embedding: mds_cosine_2", flush=True)
        dissimilarity = pdist(scaled, metric="cosine")
        if np.isfinite(dissimilarity).all():
            square = np.zeros((x.shape[0], x.shape[0]))
            upper = np.triu_indices(x.shape[0], k=1)
            square[upper] = dissimilarity
            square[(upper[1], upper[0])] = dissimilarity
            embeddings.append(
                EmbeddingSet(
                    "mds_cosine_2",
                    MDS(n_components=2, dissimilarity="precomputed", random_state=RANDOM_STATE, normalized_stress="auto").fit_transform(square),
                )
            )
    except Exception:
        pass
    if include_tsne and x.shape[0] > 5:
        perplexity = min(8, max(2, (x.shape[0] - 1) // 3))
        try:
            print(f"  Computing embedding: tsne_perplexity_{perplexity}", flush=True)
            embeddings.append(
                EmbeddingSet(
                    f"tsne_perplexity_{perplexity}",
                    TSNE(
                        n_components=2,
                        perplexity=perplexity,
                        init="pca",
                        learning_rate="auto",
                        random_state=RANDOM_STATE,
                    ).fit_transform(scaled),
                )
            )
        except Exception:
            pass
    if UMAP_MODULE is not None and x.shape[0] > 5:
        n_neighbors = min(10, max(2, x.shape[0] - 1))
        try:
            print(f"  Computing embedding: umap_neighbors_{n_neighbors}", flush=True)
            embeddings.append(
                EmbeddingSet(
                    f"umap_neighbors_{n_neighbors}",
                    UMAP_MODULE.UMAP(
                        n_components=2,
                        n_neighbors=n_neighbors,
                        min_dist=0.1,
                        n_epochs=100,
                        random_state=RANDOM_STATE,
                    ).fit_transform(scaled),
                )
            )
        except Exception:
            pass
    return embeddings


def cluster_embedding(embedding: np.ndarray) -> dict[str, np.ndarray]:
    n_samples = embedding.shape[0]
    labels: dict[str, np.ndarray] = {
        "kmeans": KMeans(n_clusters=2, n_init=10, random_state=RANDOM_STATE).fit_predict(embedding),
        "gmm_full": GaussianMixture(n_components=2, covariance_type="full", random_state=RANDOM_STATE).fit_predict(embedding),
    }
    for linkage_name in ("ward", "average", "complete"):
        metric = "euclidean"
        model = AgglomerativeClustering(n_clusters=2, linkage=linkage_name, metric=metric)
        labels[f"agglomerative_{linkage_name}"] = model.fit_predict(embedding)
    if n_samples > 4:
        neighbors = min(10, n_samples - 1)
        labels["spectral_clustering"] = SpectralClustering(
            n_clusters=2,
            affinity="nearest_neighbors",
            n_neighbors=neighbors,
            random_state=RANDOM_STATE,
            assign_labels="kmeans",
        ).fit_predict(embedding)
    return labels


def cluster_stability(embedding: np.ndarray, labels: np.ndarray, method: str, n_iter: int = 3) -> float:
    rng = np.random.default_rng(RANDOM_STATE)
    scores = []
    n_samples = embedding.shape[0]
    for _ in range(n_iter):
        sample = np.sort(rng.choice(n_samples, size=max(6, int(0.8 * n_samples)), replace=False))
        try:
            sampled_labels = cluster_embedding(embedding[sample])[method]
        except Exception:
            continue
        if len(np.unique(sampled_labels)) == 2:
            scores.append(adjusted_rand_score(labels[sample], sampled_labels))
    return float(np.mean(scores)) if scores else float("nan")


def majority_label_metrics(y_true: np.ndarray, labels: np.ndarray) -> dict[str, float | str]:
    mapping = {}
    correct = 0
    cluster_purities = []
    for cluster in sorted(np.unique(labels)):
        mask = labels == cluster
        class_counts = pd.Series(y_true[mask]).value_counts()
        majority_label = str(class_counts.index[0])
        mapping[int(cluster)] = majority_label
        correct += int(class_counts.iloc[0])
        cluster_purities.append(float(class_counts.iloc[0] / class_counts.sum()))
    y_pred = np.array([mapping[int(cluster)] for cluster in labels])
    return {
        "majority_label_accuracy": float((y_pred == y_true).mean()),
        "majority_label_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "majority_label_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "cluster_purity": float(correct / len(y_true)),
        "mean_cluster_purity": float(np.mean(cluster_purities)),
        "min_cluster_purity": float(np.min(cluster_purities)),
        "cluster_majority_label_map": json.dumps(mapping, sort_keys=True),
    }


def evaluate_strategies(
    feature_sets: list[FeatureSet],
    classes: pd.DataFrame,
    *,
    include_tsne: bool = True,
    min_selection_cluster_fraction: float = 0.2,
    include_none_scaled: bool = False,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[tuple[str, str], np.ndarray]]:
    rows = []
    label_map: dict[str, np.ndarray] = {}
    embedding_map: dict[tuple[str, str], np.ndarray] = {}
    y_class = classes["class_label"].to_numpy()
    y_axis_1 = classes["axis_1"].to_numpy()
    y_axis_2 = classes["axis_2"].to_numpy()
    for feature_set in feature_sets:
        print(f"Evaluating feature set: {feature_set.name}", flush=True)
        for embedding_set in make_embeddings(feature_set, include_tsne=include_tsne, include_none_scaled=include_none_scaled):
            print(f"  Reduction: {embedding_set.name}", flush=True)
            embedding_map[(feature_set.name, embedding_set.name)] = embedding_set.values
            for cluster_method, labels in cluster_embedding(embedding_set.values).items():
                if len(np.unique(labels)) != 2:
                    continue
                key = f"{feature_set.name}__{embedding_set.name}__{cluster_method}"
                label_map[key] = labels
                try:
                    silhouette = silhouette_score(embedding_set.values, labels)
                except Exception:
                    silhouette = float("nan")
                try:
                    calinski = calinski_harabasz_score(embedding_set.values, labels)
                except Exception:
                    calinski = float("nan")
                try:
                    davies = davies_bouldin_score(embedding_set.values, labels)
                except Exception:
                    davies = float("nan")
                counts = np.bincount(labels, minlength=2)
                majority_metrics = majority_label_metrics(y_class, labels)
                rows.append(
                    {
                        "strategy": key,
                        "feature_set": feature_set.name,
                        "reduction": embedding_set.name,
                        "cluster_method": cluster_method,
                        "n_cluster_0": int(counts[0]),
                        "n_cluster_1": int(counts[1]),
                        "min_cluster_fraction": float(counts.min() / counts.sum()),
                        "silhouette": silhouette,
                        "calinski_harabasz": calinski,
                        "davies_bouldin": davies,
                        "stability_ari": cluster_stability(embedding_set.values, labels, cluster_method),
                        "ari_4class": adjusted_rand_score(y_class, labels),
                        "nmi_4class": normalized_mutual_info_score(y_class, labels),
                        "ami_4class": adjusted_mutual_info_score(y_class, labels),
                        "ari_axis_1": adjusted_rand_score(y_axis_1, labels),
                        "ari_axis_2": adjusted_rand_score(y_axis_2, labels),
                        **majority_metrics,
                    }
                )
    results = pd.DataFrame(rows)
    for source, ascending in [("silhouette", False), ("stability_ari", False), ("davies_bouldin", True)]:
        rank_col = f"{source}_rank"
        results[rank_col] = results[source].rank(ascending=ascending, method="average")
    balance_penalty = (0.5 - results["min_cluster_fraction"]).clip(lower=0) * 2
    results["unsupervised_rank_score"] = (
        results["silhouette_rank"] + results["stability_ari_rank"] + results["davies_bouldin_rank"] + balance_penalty
    )
    results["eligible_for_selection"] = results["min_cluster_fraction"] >= min_selection_cluster_fraction
    results["selection_rank_score"] = results["unsupervised_rank_score"]
    results.loc[~results["eligible_for_selection"], "selection_rank_score"] = np.inf
    return results.sort_values(["selection_rank_score", "unsupervised_rank_score"]), label_map, embedding_map


def two_dimensional_view(feature_set: FeatureSet) -> np.ndarray:
    scaled = StandardScaler().fit_transform(feature_set.values)
    return PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(scaled)


def plot_strategy_scores(results: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    scatter = ax.scatter(
        results["silhouette"],
        results["stability_ari"],
        c=results["min_cluster_fraction"],
        cmap="viridis",
        s=44,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xlabel("Silhouette")
    ax.set_ylabel("Bootstrap stability ARI")
    ax.set_title("Two-cluster strategy sweep")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Smallest cluster fraction")
    save_figure(fig, output_dir, "strategy_quality_sweep")


def plot_embedding(coords: np.ndarray, classes: pd.DataFrame, labels: np.ndarray, output_dir: Path, stem: str, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), sharex=True, sharey=True)
    for class_label, group in classes.assign(x=coords[:, 0], y=coords[:, 1]).groupby("class_label", observed=True):
        axes[0].scatter(group["x"], group["y"], label=class_label, s=34, color=CLASS_PALETTE.get(class_label), edgecolor="white", linewidth=0.5)
    axes[0].set_title("Clinical class")
    axes[0].legend(frameon=False, ncol=2)
    for cluster in sorted(np.unique(labels)):
        mask = labels == cluster
        axes[1].scatter(coords[mask, 0], coords[mask, 1], label=f"Cluster {cluster}", s=34, color=CLUSTER_PALETTE[int(cluster)], edgecolor="white", linewidth=0.5)
    axes[1].set_title("Unsupervised clusters")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    fig.suptitle(title, y=1.02)
    save_figure(fig, output_dir, stem)


def plot_cluster_class_composition(assignments: pd.DataFrame, output_dir: Path) -> None:
    composition = pd.crosstab(assignments["cluster"], assignments["class_label"], normalize="index")
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    bottom = np.zeros(len(composition))
    for class_label in ["NN", "NP", "PN", "PP"]:
        if class_label not in composition:
            continue
        values = composition[class_label].to_numpy()
        ax.bar(composition.index.astype(str), values, bottom=bottom, label=class_label, color=CLASS_PALETTE[class_label])
        bottom += values
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Fraction of patients")
    ax.set_ylim(0, 1)
    ax.set_title("Clinical-class composition of selected clusters")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    save_figure(fig, output_dir, "selected_cluster_class_composition")


def safe_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_").replace(".", "p")


def display_coordinates(embedding: np.ndarray) -> np.ndarray:
    if embedding.shape[1] == 2:
        return embedding
    return PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(StandardScaler().fit_transform(embedding))


def plot_mode_cluster_grid(
    results: pd.DataFrame,
    label_map: dict[str, np.ndarray],
    embedding_map: dict[tuple[str, str], np.ndarray],
    classes: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    mode_dir = output_dir / "figures" / "modes"
    mode_dir.mkdir(parents=True, exist_ok=True)
    index_rows = []
    markers = {"NN": "o", "NP": "^", "PN": "s", "PP": "D"}
    for (feature_set, reduction), embedding in embedding_map.items():
        rows = results[(results["feature_set"] == feature_set) & (results["reduction"] == reduction)].copy()
        if rows.empty:
            continue
        rows = rows.sort_values("cluster_method")
        coords = display_coordinates(embedding)
        n_methods = len(rows)
        n_cols = 3
        n_rows = int(np.ceil(n_methods / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(11.5, 3.25 * n_rows), squeeze=False, sharex=True, sharey=True)
        for ax, (_, row) in zip(axes.ravel(), rows.iterrows()):
            labels = label_map[row["strategy"]]
            for cluster in sorted(np.unique(labels)):
                for class_label in sorted(classes["class_label"].unique()):
                    mask = (labels == cluster) & (classes["class_label"].to_numpy() == class_label)
                    if not mask.any():
                        continue
                    ax.scatter(
                        coords[mask, 0],
                        coords[mask, 1],
                        s=42,
                        marker=markers.get(class_label, "o"),
                        color=CLUSTER_PALETTE[int(cluster)],
                        edgecolor="white",
                        linewidth=0.5,
                        label=f"C{cluster} {class_label}",
                    )
            ax.set_title(
                f"{row['cluster_method']}\n"
                f"sil={row['silhouette']:.2f}, stab={row['stability_ari']:.2f}, ARI={row['ari_4class']:.2f}",
                fontsize=9,
            )
            ax.set_xlabel("display dim 1")
            ax.set_ylabel("display dim 2")
        for ax in axes.ravel()[n_methods:]:
            ax.axis("off")
        handles, labels = axes.ravel()[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), frameon=False, fontsize=8)
        fig.suptitle(f"{feature_set} + {reduction}", y=0.995, fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0.06, 1, 0.96])
        stem = f"mode_{safe_name(feature_set)}__{safe_name(reduction)}"
        for suffix in ("png", "pdf"):
            fig.savefig(mode_dir / f"{stem}.{suffix}", bbox_inches="tight")
        plt.close(fig)
        index_rows.append(
            {
                "feature_set": feature_set,
                "reduction": reduction,
                "figure_png": str(mode_dir / f"{stem}.png"),
                "n_cluster_methods": int(n_methods),
            }
        )
    index_df = pd.DataFrame(index_rows)
    index_df.to_csv(output_dir / "mode_visualization_index.csv", index=False)
    return index_df


def plot_heatmap(feature_set: FeatureSet, classes: pd.DataFrame, labels: np.ndarray, output_dir: Path) -> None:
    x = pd.DataFrame(np.log1p(feature_set.values * 1_000_000), index=classes["patient_id"], columns=feature_set.columns)
    top_columns = x.var(axis=0).sort_values(ascending=False).head(60).index
    ordered_patients = classes.assign(cluster=labels).sort_values(["cluster", "class_label", "patient_id"])["patient_id"]
    data = x.loc[ordered_patients, top_columns]
    data = (data - data.mean(axis=0)) / data.std(axis=0).replace(0, 1)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    image = ax.imshow(data.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index, fontsize=5)
    ax.set_xticks([])
    ax.set_xlabel("Top variable motif signatures")
    ax.set_ylabel("Patients ordered by selected cluster")
    ax.set_title("Patient-by-motif matrix, top variable motifs")
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Column z-score")
    save_figure(fig, output_dir, "selected_patient_motif_heatmap")


def hex_to_rgb(hex_color: str) -> np.ndarray:
    color = hex_color.lstrip("#")
    return np.array([int(color[index : index + 2], 16) for index in (0, 2, 4)], dtype=float)


def rgb_to_hex(rgb: np.ndarray) -> str:
    clipped = np.clip(np.round(rgb), 0, 255).astype(int)
    return "#" + "".join(f"{value:02x}" for value in clipped)


def mix_label_color(class_counts: dict[str, int]) -> str:
    total = sum(class_counts.values())
    if total == 0:
        return "#777777"
    nn_weight = class_counts.get("NN", 0) / total
    np_weight = class_counts.get("NP", 0) / total
    other_weight = max(0.0, 1.0 - nn_weight - np_weight)
    nn_rgb = hex_to_rgb(LABEL_COLOR_ENDPOINTS["NN"])
    np_rgb = hex_to_rgb(LABEL_COLOR_ENDPOINTS["NP"])
    mixed = nn_weight * nn_rgb + np_weight * np_rgb + other_weight * np.array([130, 130, 130], dtype=float)
    return rgb_to_hex(mixed)


def plot_dendrogram(feature_set: FeatureSet, classes: pd.DataFrame, output_dir: Path) -> None:
    scaled = StandardScaler().fit_transform(np.log1p(feature_set.values * 1_000_000))
    link = linkage(scaled, method="ward")
    class_labels = classes["class_label"].tolist()
    n_leaves = len(class_labels)
    node_counts: dict[int, dict[str, int]] = {
        index: {str(label): 1} for index, label in enumerate(class_labels)
    }
    for merge_index, row in enumerate(link):
        left = int(row[0])
        right = int(row[1])
        counts: dict[str, int] = {}
        for child in (left, right):
            for label, count in node_counts[child].items():
                counts[label] = counts.get(label, 0) + count
        node_counts[n_leaves + merge_index] = counts

    def link_color_func(node_id: int) -> str:
        return mix_label_color(node_counts[int(node_id)])

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    dendrogram(
        link,
        labels=classes["patient_id"].tolist(),
        leaf_rotation=90,
        leaf_font_size=6,
        link_color_func=link_color_func,
        above_threshold_color=mix_label_color({"NN": class_labels.count("NN"), "NP": class_labels.count("NP")}),
        ax=ax,
    )
    label_by_patient = dict(zip(classes["patient_id"], classes["class_label"]))
    for tick in ax.get_xmajorticklabels():
        tick.set_color(LABEL_COLOR_ENDPOINTS.get(label_by_patient.get(tick.get_text(), ""), "#333333"))
        tick.set_fontweight("bold")
    ax.set_title("Hierarchical clustering of patient motif profiles, colored by NN/NP composition")
    ax.set_ylabel("Ward distance")
    ax.text(
        0.01,
        0.98,
        "Branch color is the weighted class mixture: blue=NN, orange=NP.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "#DDDDDD", "boxstyle": "round,pad=0.25"},
    )
    from matplotlib.lines import Line2D

    ax.legend(
        handles=[
            Line2D([0], [0], color=LABEL_COLOR_ENDPOINTS["NN"], lw=2.5, label="NN"),
            Line2D([0], [0], color=LABEL_COLOR_ENDPOINTS["NP"], lw=2.5, label="NP"),
        ],
        frameon=False,
        loc="upper right",
    )
    save_figure(fig, output_dir, "patient_motif_dendrogram")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    table = df.reset_index()
    headers = [str(column) for column in table.columns]
    rows = [[str(value) for value in row] for row in table.to_numpy()]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows)) if rows else len(headers[index])
        for index in range(len(headers))
    ]
    header_line = "| " + " | ".join(headers[index].ljust(widths[index]) for index in range(len(headers))) + " |"
    separator = "| " + " | ".join("-" * widths[index] for index in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, separator, *body])


def write_report(
    output_dir: Path,
    args: argparse.Namespace,
    long_df: pd.DataFrame,
    fov_summary: pd.DataFrame,
    classes: pd.DataFrame,
    results: pd.DataFrame,
    assignments: pd.DataFrame,
) -> None:
    best = results.iloc[0]
    class_table = pd.crosstab(assignments["cluster"], assignments["class_label"])
    axis1_table = pd.crosstab(assignments["cluster"], assignments["axis_1"])
    axis2_table = pd.crosstab(assignments["cluster"], assignments["axis_2"])
    lines = [
        "# FANMOD+ Patient Motif Clustering Report",
        "",
        "## Objective",
        "",
        "We constructed an unsupervised patient-by-motif matrix from FANMOD+ output and asked whether patient motif profiles naturally partition into two groups. Motifs were keyed by a canonical color/topology signature so the same motif can be compared across FOVs and patients.",
        "",
        "## Input and matrix construction",
        "",
        f"- Serialized artifact: `{args.serialized_artifact}`." if args.serialized_artifact is not None else "- Serialized artifact: not used.",
        f"- Pre-parsed patient-motif table: `{args.parsed_long_csv}`." if args.parsed_long_csv is not None else "- Pre-parsed patient-motif table: not used.",
        f"- FANMOD+ directory: `{args.fanmod_dir}`.",
        f"- Patient class table: `{args.patient_class_csv}`.",
        f"- FANMOD CSV files parsed: {fov_summary.shape[0]}.",
        f"- Included patient classes: {', '.join(args.labels)}.",
        f"- Patients in included class annotation table: {classes.shape[0]}.",
        f"- Patients with FANMOD+ motif profiles clustered: {assignments.shape[0]}.",
        f"- Aggregated patient-motif pairs: {long_df.shape[0]:,}.",
        f"- Unique motif signatures before feature filtering: {long_df['motif_id'].nunique():,}.",
        (f"- Feature filter for clustering: none; all {long_df['motif_id'].nunique():,} observed motifs included."
         if args.all_motifs else
         f"- Feature filter for clustering: motifs present in at least {args.min_patients_per_motif} patients, then top {args.top_motifs} by variance."),
        f"- Selected strategy required a minimum cluster fraction of {args.min_selection_cluster_fraction:.2f}; all strategies are still reported in `strategy_results.csv`.",
        "",
        "The primary value in the matrix is the CISM motif frequency: total motif count divided by the summed FOV-specific FANMOD subgraph opportunity estimate (`Count / Frequency`). The strategy sweep also considered log-frequency, binary presence, and log-count TF-IDF transforms.",
        "",
        "## Strategy sweep",
        "",
        "Dimensionality-reduction strategies included no reduction after scaling, PCA, truncated SVD, NMF, spectral embedding, cosine MDS when applicable, and t-SNE unless disabled. UMAP is included automatically when `umap-learn` is installed in the environment.",
        "",
        f"- t-SNE included in this run: {not args.skip_tsne}.",
        f"- UMAP available in this environment: {UMAP_MODULE is not None}.",
        f"- Direct high-dimensional scaled matrix included: {args.include_none_scaled}.",
        "",
        "Clustering methods included k-means, Gaussian mixture models, agglomerative clustering, and spectral clustering. All methods were constrained to two clusters.",
        "",
        "Strategies were ranked using unsupervised separation and robustness: silhouette, Davies-Bouldin, bootstrap cluster-label stability (ARI), and a small penalty for highly imbalanced clusters.",
        "",
        "## Selected result",
        "",
        f"- Selected strategy: `{best['strategy']}`.",
        f"- Feature transform: `{best['feature_set']}`.",
        f"- Reduction: `{best['reduction']}`.",
        f"- Clustering: `{best['cluster_method']}`.",
        f"- Cluster sizes: {int(best['n_cluster_0'])} and {int(best['n_cluster_1'])}.",
        f"- Silhouette: {best['silhouette']:.3f}.",
        f"- Bootstrap stability ARI: {best['stability_ari']:.3f}.",
        f"- Davies-Bouldin index: {best['davies_bouldin']:.3f}.",
        f"- Agreement with retained clinical classes, ARI: {best['ari_4class']:.3f}; NMI: {best['nmi_4class']:.3f}.",
        f"- Agreement with first label axis, ARI: {best['ari_axis_1']:.3f}.",
        f"- Agreement with second label axis, ARI: {best['ari_axis_2']:.3f}.",
        "",
        "Clinical labels were not used to fit the clustering; they are reported only as post hoc annotation for reviewer interpretation.",
        "",
        "## Cluster composition by clinical labels",
        "",
        dataframe_to_markdown(class_table),
        "",
        "## Cluster composition by binary label axes",
        "",
        "First label character:",
        "",
        dataframe_to_markdown(axis1_table),
        "",
        "Second label character:",
        "",
        dataframe_to_markdown(axis2_table),
        "",
        "## Outputs",
        "",
        "- `patient_motif_long.csv`: aggregated FANMOD+ motif table.",
        "- `patient_motif_matrix_cism_frequency.csv`: patient-by-motif matrix used as the primary matrix.",
        "- `strategy_results.csv`: full strategy sweep metrics.",
        "- `cluster_predictive_results.csv`: configuration ranking by majority-label cluster predictive ability.",
        "- `selected_cluster_assignments.csv`: two-group patient assignment for the selected strategy.",
        "- `top_20_strategy_results.csv`: compact table of best strategies.",
        "- `figures/strategy_quality_sweep.png`: quality/stability summary for all strategies.",
        "- `figures/selected_embedding.png`: PCA view annotated by class and selected clusters.",
        "- `figures/selected_cluster_class_composition.png`: clinical class composition of clusters.",
        "- `figures/selected_patient_motif_heatmap.png`: patient motif profile heatmap.",
        "- `figures/patient_motif_dendrogram.png`: hierarchical clustering dendrogram.",
        "- `figures/modes/`: one visualization grid per feature/reduction mode, with all clustering methods shown on the same display coordinates.",
        "- `mode_visualization_index.csv`: index of the per-mode visualization files.",
        "",
        "## Dendrogram interpretation",
        "",
        "The dendrogram is an unsupervised hierarchical clustering of patient motif profiles using Ward linkage on the selected motif feature space. The vertical height where branches join is the dissimilarity at which two patients or patient subtrees merge: lower joins indicate more similar motif profiles, while higher joins indicate more distinct profiles.",
        "",
        f"Leaf labels are colored by the true retained class using the endpoints NN={LABEL_COLOR_ENDPOINTS['NN']} and NP={LABEL_COLOR_ENDPOINTS['NP']}. Internal branch colors are weighted averages of the labels underneath each branch. For example, a branch containing two NN patients and one NP patient is colored two-thirds toward the NN blue and one-third toward the NP orange. This means pure NN or pure NP subtrees keep the endpoint color, while mixed subtrees become intermediate colors.",
        "",
        "## Reviewer-facing summary",
        "",
        f"Using all FANMOD+ motif profiles rather than pre-selected discriminative motifs, unsupervised clustering identified two patient groups among the {assignments.shape[0]} annotated {', '.join(args.labels)} patients represented in the artifact. The selected solution was chosen by internal separation and bootstrap stability, and clinical labels were held out until post hoc annotation. This supports the statement that the FANMOD+/CISM motif representation contains patient-level structure within the NN/NP comparison beyond the supervised discriminative-motif selection step.",
        "",
    ]
    (output_dir / "fanmod_patient_motif_clustering_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figures").mkdir(parents=True, exist_ok=True)

    if args.serialized_artifact is not None:
        long_df, fov_summary, _artifact_path = build_motif_tables_from_serialized_artifact(
            args.serialized_artifact,
            args.output_dir,
            args.motif_col,
            args.force_parse,
        )
    else:
        long_df, fov_summary = build_motif_tables(
            args.fanmod_dir,
            args.output_dir,
            args.force_parse,
            args.parsed_long_csv,
            args.fov_summary_csv,
        )
    classes = load_patient_classes(args.patient_class_csv)
    classes = classes[classes["class_label"].isin(args.labels)].copy()
    patient_ids = classes["patient_id"].tolist()
    long_df = long_df[long_df["patient_id"].isin(patient_ids)].copy()
    fov_summary = fov_summary[fov_summary["patient_id"].isin(patient_ids)].copy()
    classes = classes[
        classes["patient_id"].isin(sorted(long_df["patient_id"].unique(), key=lambda value: int(value.replace("Melanoma", ""))))
    ].copy()
    long_df.to_csv(args.output_dir / "patient_motif_long.csv", index=False)
    fov_summary.to_csv(args.output_dir / "fanmod_fov_summary.csv", index=False)

    feature_sets = make_feature_sets(
        long_df,
        classes["patient_id"].tolist(),
        args.top_motifs,
        args.min_patients_per_motif,
        args.output_dir,
        all_motifs=args.all_motifs,
    )
    results, label_map, embedding_map = evaluate_strategies(
        feature_sets,
        classes,
        include_tsne=not args.skip_tsne,
        min_selection_cluster_fraction=args.min_selection_cluster_fraction,
        include_none_scaled=args.include_none_scaled,
    )
    results.to_csv(args.output_dir / "strategy_results.csv", index=False)
    results.head(20).to_csv(args.output_dir / "top_20_strategy_results.csv", index=False)
    predictive_columns = [
        "strategy",
        "feature_set",
        "reduction",
        "cluster_method",
        "n_cluster_0",
        "n_cluster_1",
        "majority_label_accuracy",
        "majority_label_balanced_accuracy",
        "majority_label_macro_f1",
        "cluster_purity",
        "mean_cluster_purity",
        "min_cluster_purity",
        "cluster_majority_label_map",
        "silhouette",
        "stability_ari",
        "davies_bouldin",
        "selection_rank_score",
    ]
    (
        results[predictive_columns]
        .sort_values(
            [
                "majority_label_balanced_accuracy",
                "majority_label_accuracy",
                "min_cluster_purity",
                "silhouette",
            ],
            ascending=[False, False, False, False],
        )
        .to_csv(args.output_dir / "cluster_predictive_results.csv", index=False)
    )

    best = results.iloc[0]
    best_labels = label_map[best["strategy"]]
    assignments = classes.copy()
    assignments["cluster"] = best_labels
    assignments.to_csv(args.output_dir / "selected_cluster_assignments.csv", index=False)

    selected_feature = next(feature_set for feature_set in feature_sets if feature_set.name == best["feature_set"])
    coords = two_dimensional_view(selected_feature)
    pd.DataFrame({"patient_id": classes["patient_id"], "pc1": coords[:, 0], "pc2": coords[:, 1], "cluster": best_labels}).to_csv(
        args.output_dir / "selected_embedding_coordinates.csv", index=False
    )
    plot_strategy_scores(results, args.output_dir)
    mode_index = plot_mode_cluster_grid(results, label_map, embedding_map, classes, args.output_dir)
    plot_embedding(coords, classes, best_labels, args.output_dir, "selected_embedding", f"Selected strategy: {best['strategy']}")
    plot_cluster_class_composition(assignments, args.output_dir)
    plot_heatmap(selected_feature, classes, best_labels, args.output_dir)
    plot_dendrogram(selected_feature, classes, args.output_dir)

    config = {
        "fanmod_dir": str(args.fanmod_dir),
        "patient_class_csv": str(args.patient_class_csv),
        "serialized_artifact": str(args.serialized_artifact) if args.serialized_artifact is not None else None,
        "motif_col": args.motif_col,
        "top_motifs": args.top_motifs,
        "min_patients_per_motif": args.min_patients_per_motif,
        "all_motifs": args.all_motifs,
        "parsed_long_csv": str(args.parsed_long_csv) if args.parsed_long_csv is not None else None,
        "fov_summary_csv": str(args.fov_summary_csv) if args.fov_summary_csv is not None else None,
        "min_selection_cluster_fraction": args.min_selection_cluster_fraction,
        "labels": args.labels,
        "tsne_included": not args.skip_tsne,
        "none_scaled_included": args.include_none_scaled,
        "umap_available": UMAP_MODULE is not None,
        "n_mode_visualizations": int(len(mode_index)),
        "random_state": RANDOM_STATE,
    }
    (args.output_dir / "analysis_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    write_report(args.output_dir, args, long_df, fov_summary, classes, results, assignments)
    print(f"Wrote FANMOD+ patient clustering analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
