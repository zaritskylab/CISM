#!/usr/bin/env python3
"""Create compact CSV summaries from SMORE motif results."""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import argparse
import ast
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SMORE_RESULTS = Path("/home/omriavit/research/smore_results.csv")
DEFAULT_METADATA_DIR = Path("/groups/assafza_group/assafza/Omri/processed_data_23_10_2025")
DEFAULT_NOTEBOOK = Path("/home/omriavit/research/smore_analysis.ipynb")
DEFAULT_OUTPUT_DIR = Path("analysis/outputs/smore_compact")
CELL_TYPE_LABELS = {
    "0": "B cell",
    "1": "CD4 APC",
    "2": "CD4 T cell",
    "3": "CD4 Treg",
    "4": "CD8 T cell",
    "5": "DCs",
    "6": "Germinal Center B cell",
    "7": "Hevs",
    "8": "MAC",
    "9": "Memory CD4 T Cell",
    "10": "NK cell",
    "11": "Neutrophil",
    "12": "Stroma",
    "13": "Tumor",
    "14": "Unidentified",
    "15": "Vessels",
}


def load_labeled_smore(smore_results: Path, metadata_dir: Path) -> pd.DataFrame:
    results = pd.read_csv(smore_results).rename(columns={"SID": "fov"})

    patient_class = pd.read_csv(
        metadata_dir / "patient_class.csv", header=None, names=["patient number", "class"]
    )
    patient_class["patient number"] = (
        patient_class["patient number"].str.replace("Melanoma", "", regex=False).astype(int)
    )

    patient_fov = pd.read_csv(metadata_dir / "patient_fov.csv")[["patient number", "fov"]]
    fov_classes = pd.merge(patient_fov, patient_class, on="patient number", how="inner")

    labeled = pd.merge(results, fov_classes, on="fov", how="left").dropna(subset=["class"])
    labeled["patient number"] = labeled["patient number"].astype(int)
    labeled = labeled[labeled["class"].isin(["NN", "NP"])].copy()
    return labeled


def add_motif_types(df: pd.DataFrame) -> pd.DataFrame:
    if "motif_types" in df.columns:
        return df
    df = df.copy()
    df["motif_types"] = df["seed"].str.replace("-", ",", regex=False)
    return df


def motif_stringency(labeled: pd.DataFrame) -> pd.DataFrame:
    class_sizes = (
        labeled[["patient number", "class"]]
        .drop_duplicates()
        .groupby("class", observed=True)["patient number"]
        .nunique()
        .to_dict()
    )
    counts = (
        labeled.groupby(["seed", "class"], observed=True)["patient number"]
        .nunique()
        .reset_index(name="n_patients")
        .pivot(index="seed", columns="class", values="n_patients")
        .fillna(0)
        .astype(int)
    )
    for cls in ["NN", "NP"]:
        if cls not in counts:
            counts[cls] = 0
    counts = counts[["NN", "NP"]]
    exclusive = counts[(counts["NN"] == 0) | (counts["NP"] == 0)].copy()
    exclusive["exclusive_to"] = np.where(exclusive["NN"] == 0, "NP", "NN")
    exclusive["stringency"] = [
        row[cls] / class_sizes[cls]
        for cls, (_, row) in zip(exclusive["exclusive_to"], exclusive.iterrows())
    ]
    exclusive["total_patients"] = exclusive["NN"] + exclusive["NP"]

    motif_lookup = (
        add_motif_types(labeled)[["seed", "motif_types"]]
        .drop_duplicates("seed")
        .set_index("seed")["motif_types"]
    )
    exclusive["motif_types"] = exclusive.index.map(motif_lookup)
    return exclusive.reset_index().rename(columns={"seed": "motif"})


def count_vs_stringency(exclusive: pd.DataFrame) -> pd.DataFrame:
    thresholds = sorted(exclusive["stringency"].dropna().unique())
    rows = []
    for threshold in thresholds:
        kept = exclusive[exclusive["stringency"] >= threshold]
        counts = kept["exclusive_to"].value_counts()
        rows.append(
            {
                "stringency": threshold,
                "NN_motif_count": int(counts.get("NN", 0)),
                "NP_motif_count": int(counts.get("NP", 0)),
                "total_motif_count": int(len(kept)),
            }
        )
    return pd.DataFrame(rows)


def auc_vs_stringency(notebook: Path) -> pd.DataFrame:
    if not notebook.exists():
        raise FileNotFoundError(f"AUC notebook not found: {notebook}")

    nb = json.loads(notebook.read_text())
    for cell in nb.get("cells", []):
        source = "".join(cell.get("source", []))
        if source.strip() != "scores":
            continue
        for output in cell.get("outputs", []):
            text = output.get("data", {}).get("text/plain")
            if text is None:
                continue
            if isinstance(text, list):
                text = "".join(text)
            scores = ast.literal_eval(text)
            out = pd.DataFrame(scores)
            out["source"] = "smore_analysis.ipynb CISM sweep"
            return out[["stringency", "auc", "source"]]
    raise ValueError(f"Could not find a saved `scores` output in {notebook}")


def pairwise_interactions(exclusive: pd.DataFrame, top_n: int) -> pd.DataFrame:
    counter: Counter[tuple[str, str, str]] = Counter()
    weighted: Counter[tuple[str, str, str]] = Counter()
    for row in exclusive.itertuples(index=False):
        types = [str(x) for x in str(row.motif_types).replace("-", ",").split(",") if str(x) != "nan"]
        for a, b in combinations(types, 2):
            pair = tuple(sorted((a, b)))
            key = (row.exclusive_to, pair[0], pair[1])
            counter[key] += 1
            weighted[key] += row.stringency

    records = [
        {
            "exclusive_to": cls,
            "cell_type_a": a,
            "cell_type_a_label": CELL_TYPE_LABELS.get(a, a),
            "cell_type_b": b,
            "cell_type_b_label": CELL_TYPE_LABELS.get(b, b),
            "motif_count": count,
            "stringency_weighted_count": weighted[(cls, a, b)],
        }
        for (cls, a, b), count in counter.items()
    ]
    out = pd.DataFrame(records)
    if out.empty:
        return out
    return (
        out.sort_values(
            ["exclusive_to", "motif_count", "stringency_weighted_count"],
            ascending=[True, False, False],
        )
        .groupby("exclusive_to", observed=True)
        .head(top_n)
        .reset_index(drop=True)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smore-results", type=Path, default=DEFAULT_SMORE_RESULTS)
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-pairs", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    labeled = load_labeled_smore(args.smore_results, args.metadata_dir)
    exclusive = motif_stringency(labeled)
    counts = count_vs_stringency(exclusive)
    auc = auc_vs_stringency(args.notebook)
    pairs = pairwise_interactions(exclusive, args.top_pairs)

    exclusive.to_csv(args.output_dir / "smore_exclusive_motifs_by_stringency.csv", index=False)
    counts.to_csv(args.output_dir / "smore_motif_count_vs_stringency.csv", index=False)
    auc.to_csv(args.output_dir / "smore_auc_vs_stringency.csv", index=False)
    pairs.to_csv(args.output_dir / "smore_top_pairwise_interactions.csv", index=False)

    print(f"Wrote {len(exclusive)} exclusive motifs")
    print(f"Wrote {len(counts)} motif-count thresholds")
    print(f"Wrote {len(auc)} AUC thresholds")
    print(f"Wrote {len(pairs)} pairwise interaction rows")


if __name__ == "__main__":
    main()
