#!/usr/bin/env python3
"""Reconstruct HNSCC CosMx RNA cell subtypes with marker signatures.

This is a transparent fallback when the authors' processed
CosMX_RNA_merged.rda object is unavailable. It reads the raw CosMx RNA
expression and metadata CSVs, attaches TMA/FOV/patient metadata, scores
the 16 author-facing subtype names with marker signatures, and writes a
cell metadata table suitable for CISM graph preparation.
"""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


RUNS = {
    "TMA_A": {
        "expr": "Run5607_TMAA_exprMat_file.csv",
        "meta": "Run5607_TMAA_metadata_file.csv",
    },
    "TMA_B": {
        "expr": "Run5545_TMAB_exprMat_file.csv",
        "meta": "Run5545_TMAB_metadata_file.csv",
    },
    "TMA_C": {
        "expr": "Run5668_TMAC_exprMat_file.csv",
        "meta": "Run5668_TMAC_metadata_file.csv",
    },
}


AUTHOR_MARKERS = [
    "MS4A1", "CD79A", "IGHG2", "HLA-DRB1", "HLA-DRA", "LAMP3", "CLEC9A",
    "CD1C", "PECAM1", "CD34", "VIM", "DCN", "COL1A2", "ACTA2", "VEGFA",
    "COL1A1", "C1QC", "C1QA", "CD68", "CD14", "FCGR3A", "CD38", "JCHAIN",
    "CD3E", "CD4", "CD8A", "FOXP3", "CTLA4", "HIF1A", "KRT17", "KRT14",
    "KRT16", "KRT19", "KRT15", "KRT5", "KRT4",
]


SIGNATURES = {
    "Endothelials": ["PECAM1", "CD34", "ACKR1", "VWF", "ENG"],
    "Bcells": ["MS4A1", "CD79A", "CD79B", "CD19", "BANK1"],
    "Plasma": ["JCHAIN", "IGHG2", "IGHG1", "MZB1", "XBP1", "CD38"],
    "Fibroblast_myCAFs": ["COL1A1", "COL1A2", "DCN", "ACTA2", "TAGLN", "MYL9"],
    "Fibroblast_apCAFs": ["HLA-DRA", "HLA-DRB1", "CD74", "DCN", "COL1A1"],
    "Fibroblast_VEGFA_inflammatorylike_CAFs": ["VEGFA", "CXCL12", "IL6", "CXCL14", "COL1A1", "DCN"],
    "Monocytes": ["CD14", "FCGR3A", "LYZ", "S100A8", "S100A9", "LST1"],
    "Macrophages": ["C1QA", "C1QB", "C1QC", "CD68", "CD163", "MRC1"],
    "DCs": ["LAMP3", "HLA-DRA", "HLA-DRB1", "CCR7", "CD1C", "CLEC9A", "ITGAX"],
    "Tcells_CD4+": ["CD3E", "CD4", "IL7R", "CCR7", "LTB"],
    "TcellsCD8+": ["CD3E", "CD8A", "CD8B", "GZMB", "NKG7"],
    "Tregs": ["FOXP3", "CTLA4", "IL2RA", "TIGIT", "CD4"],
    "Tumors_HIF1A+": ["EPCAM", "KRT5", "KRT14", "KRT17", "HIF1A", "CA9", "VEGFA"],
    "Tumors_KRT17+": ["EPCAM", "KRT17", "KRT14", "KRT16", "KRT5"],
    "Tumors_KRT19+": ["EPCAM", "KRT19", "KRT15", "KRT8", "KRT18"],
    "Tumors_Others": ["EPCAM", "KRT4", "KRT5", "KRT14", "KRT15", "KRT16", "KRT17", "KRT19"],
}


COMPARTMENT_GATES = {
    "tumor": ["EPCAM", "KRT4", "KRT5", "KRT8", "KRT14", "KRT15", "KRT16", "KRT17", "KRT18", "KRT19"],
    "immune": ["PTPRC", "CD45", "CD3E", "CD4", "CD8A", "MS4A1", "CD79A", "CD14", "CD68"],
    "stromal": ["COL1A1", "COL1A2", "DCN", "VIM", "ACTA2", "PECAM1", "CD34"],
}


def existing_genes(genes: Iterable[str], columns: Iterable[str]) -> list[str]:
    available = set(columns)
    return [gene for gene in genes if gene in available]


def score_signature(norm: pd.DataFrame, genes: list[str]) -> pd.Series:
    if not genes:
        return pd.Series(np.nan, index=norm.index)
    return norm[genes].mean(axis=1)


def assign_subtypes(scores: pd.DataFrame) -> pd.Series:
    broad = pd.DataFrame(
        {
            "tumor": scores[[c for c in scores if c.startswith("Tumors_")]].max(axis=1),
            "immune": scores[
                [
                    "Bcells",
                    "Plasma",
                    "Monocytes",
                    "Macrophages",
                    "DCs",
                    "Tcells_CD4+",
                    "TcellsCD8+",
                    "Tregs",
                ]
            ].max(axis=1),
            "stromal": scores[
                [
                    "Endothelials",
                    "Fibroblast_myCAFs",
                    "Fibroblast_apCAFs",
                    "Fibroblast_VEGFA_inflammatorylike_CAFs",
                ]
            ].max(axis=1),
        }
    )
    compartment = broad.idxmax(axis=1)

    labels = scores.idxmax(axis=1)
    tumor_cols = [c for c in scores if c.startswith("Tumors_")]
    immune_cols = [
        "Bcells",
        "Plasma",
        "Monocytes",
        "Macrophages",
        "DCs",
        "Tcells_CD4+",
        "TcellsCD8+",
        "Tregs",
    ]
    stromal_cols = [
        "Endothelials",
        "Fibroblast_myCAFs",
        "Fibroblast_apCAFs",
        "Fibroblast_VEGFA_inflammatorylike_CAFs",
    ]
    labels[compartment == "tumor"] = scores.loc[compartment == "tumor", tumor_cols].idxmax(axis=1)
    labels[compartment == "immune"] = scores.loc[compartment == "immune", immune_cols].idxmax(axis=1)
    labels[compartment == "stromal"] = scores.loc[compartment == "stromal", stromal_cols].idxmax(axis=1)
    return labels


def read_tma_map(path: Path) -> pd.DataFrame:
    tma = pd.read_csv(path, sep=r"\s+", engine="python")
    tma.columns = ["tma_fov_id", "PatientID"]
    tma["tma"] = tma["tma_fov_id"].str.extract(r"TMA_([ABC])_FOV")[0].map(lambda x: f"TMA_{x}")
    tma["fov"] = tma["tma_fov_id"].str.extract(r"FOV(\d+)")[0].astype(int)
    return tma


def process_run(expr_path: Path, meta_path: Path, tma: str, tma_map: pd.DataFrame, chunksize: int) -> pd.DataFrame:
    expr_header = pd.read_csv(expr_path, nrows=0).columns.tolist()
    marker_genes = sorted(set(sum(SIGNATURES.values(), []) + sum(COMPARTMENT_GATES.values(), []) + AUTHOR_MARKERS))
    genes = existing_genes(marker_genes, expr_header)
    usecols = ["fov", "cell_ID", *genes]

    meta_cols = [
        "fov",
        "cell_ID",
        "Area",
        "CenterX_global_px",
        "CenterY_global_px",
        "CenterX_local_px",
        "CenterY_local_px",
    ]
    meta = pd.read_csv(meta_path, usecols=lambda c: c in meta_cols)
    meta["tma"] = tma
    meta = meta.merge(tma_map[tma_map["tma"] == tma], on=["tma", "fov"], how="left")

    pieces = []
    for chunk in pd.read_csv(expr_path, usecols=usecols, chunksize=chunksize):
        ids = chunk[["fov", "cell_ID"]].copy()
        counts = chunk[genes].astype(float)
        totals = counts.sum(axis=1).replace(0, np.nan)
        norm = np.log1p(counts.div(totals, axis=0) * 10000.0).fillna(0)

        scores = pd.DataFrame(index=chunk.index)
        for label, sig_genes in SIGNATURES.items():
            scores[label] = score_signature(norm, existing_genes(sig_genes, norm.columns))

        ids["predicted_annot_cluster2"] = assign_subtypes(scores)
        ids["subtype_score"] = scores.max(axis=1)
        ids["subtype_score_margin"] = scores.apply(lambda row: row.nlargest(2).iloc[0] - row.nlargest(2).iloc[1], axis=1)
        pieces.append(ids)

    calls = pd.concat(pieces, ignore_index=True)
    out = meta.merge(calls, on=["fov", "cell_ID"], how="inner")
    return out


def write_marker_report(expr_path: Path, out_path: Path) -> None:
    cols = pd.read_csv(expr_path, nrows=0).columns.tolist()
    rows = []
    for label, genes in SIGNATURES.items():
        present = existing_genes(genes, cols)
        rows.append(
            {
                "signature": label,
                "n_markers": len(genes),
                "n_present": len(present),
                "present_markers": ";".join(present),
                "missing_markers": ";".join([g for g in genes if g not in present]),
            }
        )
    pd.DataFrame(rows).to_csv(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="/groups/assafza_group/assafza/Datasets/HNSCC_CosMX")
    parser.add_argument("--outdir", default="analysis/hnscc_cosmx/reconstructed_rna_subtypes")
    parser.add_argument("--chunksize", type=int, default=25000)
    parser.add_argument("--dry-run", action="store_true", help="Only write marker availability report.")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    rna_dir = dataset_root / "RNA" / "CosMX_RNA_TMA_Data_Expr"
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tma_map = read_tma_map(dataset_root / "CosMX_TMA_RNA_TMA.txt")
    write_marker_report(rna_dir / RUNS["TMA_A"]["expr"], outdir / "rna_subtype_marker_availability.csv")
    if args.dry_run:
        return

    all_runs = []
    for tma, files in RUNS.items():
        run = process_run(
            expr_path=rna_dir / files["expr"],
            meta_path=rna_dir / files["meta"],
            tma=tma,
            tma_map=tma_map,
            chunksize=args.chunksize,
        )
        run.to_csv(outdir / f"{tma}_reconstructed_cell_metadata.csv", index=False)
        all_runs.append(run)

    cells = pd.concat(all_runs, ignore_index=True)
    cells.to_csv(outdir / "hnscc_cosmx_rna_reconstructed_cell_metadata.csv", index=False)

    summary = (
        cells.groupby(["PatientID", "tma_fov_id", "predicted_annot_cluster2"])
        .size()
        .rename("n_cells")
        .reset_index()
    )
    summary.to_csv(outdir / "hnscc_cosmx_rna_reconstructed_subtype_counts.csv", index=False)


if __name__ == "__main__":
    main()
