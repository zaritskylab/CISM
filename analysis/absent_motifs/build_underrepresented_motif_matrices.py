from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = _REPO_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cism.fanmod_postprocess import (
    build_patient_motif_matrix_from_fanmod,
    filter_by_normal_significance,
    load_fanmod_output_tree,
    summarize_directional_motifs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build patient-by-motif matrices from existing FANMOD+ CSV artifacts, "
            "including normally significant under-represented motifs."
        )
    )
    parser.add_argument(
        "--fanmod-output-root",
        type=Path,
        required=True,
        help="Directory containing FANMOD+ output CSV files, e.g. runtime/fanmod_output/DATASET.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/outputs/underrepresented_motifs"),
        help="Directory for generated long tables, matrices, and reviewer summary.",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Optional prefix used to reconstruct CISM-style Patient_uId values.",
    )
    parser.add_argument(
        "--motif-size",
        type=int,
        default=None,
        help="Optional motif-size subdirectory filter. If omitted, all CSVs under the root are scanned.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Two-tailed normal p-value threshold computed from FANMOD Z-Score.",
    )
    parser.add_argument(
        "--matrix-value",
        choices=["observed_frequency", "signed_delta", "depletion"],
        default="depletion",
        help=(
            "Additional matrix value to write for all three direction groups. "
            "The script always writes standardized signed-delta and absence-depletion matrices."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    motifs = load_fanmod_output_tree(
        args.fanmod_output_root,
        dataset_name=args.dataset_name,
        motif_size=args.motif_size,
    )
    if motifs.empty:
        raise SystemExit(f"No FANMOD motif records found under {args.fanmod_output_root}")

    all_significant = filter_by_normal_significance(motifs, alpha=args.alpha, direction="all")
    proliferated = filter_by_normal_significance(motifs, alpha=args.alpha, direction="proliferation")
    underrepresented = filter_by_normal_significance(motifs, alpha=args.alpha, direction="absence")

    patient_ids = pd.Index(motifs["Patient_uId"].dropna().unique(), name="Patient_uId")
    standard_matrices = {
        "all_significant_signed_delta": build_patient_motif_matrix_from_fanmod(
            all_significant,
            patient_ids=patient_ids,
            value="signed_delta",
        ),
        "proliferation_observed_frequency": build_patient_motif_matrix_from_fanmod(
            proliferated,
            patient_ids=patient_ids,
            value="observed_frequency",
        ),
        "absence_depletion": build_patient_motif_matrix_from_fanmod(
            underrepresented,
            patient_ids=patient_ids,
            value="depletion",
        ),
        "absence_observed_frequency": build_patient_motif_matrix_from_fanmod(
            underrepresented,
            patient_ids=patient_ids,
            value="observed_frequency",
        ),
    }
    extra_matrices = {
        f"all_significant_{args.matrix_value}": build_patient_motif_matrix_from_fanmod(
            all_significant,
            patient_ids=patient_ids,
            value=args.matrix_value,
        ),
        f"proliferation_{args.matrix_value}": build_patient_motif_matrix_from_fanmod(
            proliferated,
            patient_ids=patient_ids,
            value=args.matrix_value,
        ),
        f"absence_{args.matrix_value}": build_patient_motif_matrix_from_fanmod(
            underrepresented,
            patient_ids=patient_ids,
            value=args.matrix_value,
        ),
    }
    matrices = {**extra_matrices, **standard_matrices}

    motifs.to_csv(args.output_dir / "fanmod_motifs_with_normal_two_tailed_p_values.csv", index=False)
    all_significant.to_csv(args.output_dir / "significant_motifs_all_directions.csv", index=False)
    proliferated.to_csv(args.output_dir / "significant_motifs_proliferation.csv", index=False)
    underrepresented.to_csv(args.output_dir / "significant_motifs_absence.csv", index=False)

    for name, matrix in matrices.items():
        matrix.to_csv(args.output_dir / f"patient_motif_matrix_{name}.csv")

    summary = summarize_directional_motifs(motifs, alpha=args.alpha)
    summary.to_csv(args.output_dir / "directional_significance_summary.csv", index=False)

    config = {
        "fanmod_output_root": str(args.fanmod_output_root),
        "dataset_name": args.dataset_name,
        "motif_size": args.motif_size,
        "alpha": args.alpha,
        "matrix_value": args.matrix_value,
        "n_fanmod_rows": int(motifs.shape[0]),
        "n_significant_rows": int(all_significant.shape[0]),
        "n_underrepresented_rows": int(underrepresented.shape[0]),
        "n_underrepresented_motifs": int(underrepresented["ID"].nunique()),
        "absence_matrix_shape": list(standard_matrices["absence_depletion"].shape),
    }
    (args.output_dir / "analysis_config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.output_dir / "reviewer_underrepresented_motif_summary.md").write_text(
        build_reviewer_summary(config, summary),
    )

    print(f"Wrote under-represented motif analysis to {args.output_dir}")
    print(summary.to_string(index=False))


def build_reviewer_summary(config: dict, summary: pd.DataFrame) -> str:
    if summary.empty:
        summary_text = "No motifs passed the selected two-tailed normal threshold."
    else:
        summary_text = dataframe_to_markdown(summary)

    return f"""# Under-represented FANMOD Motif Analysis

This analysis addresses the reviewer concern that underrepresented motifs may reflect inhibitory, exclusive, competitive, or repulsive cell-cell interactions.

Using the existing FANMOD+ artifacts, we recomputed a normal two-tailed p-value from the stored Z-score:

`p_two_tailed = erfc(abs(Z-Score) / sqrt(2))`

Motifs with negative Z-scores are labeled `absence`; motifs with positive Z-scores are labeled `proliferation`.

## Outputs

- `significant_motifs_absence.csv`: significantly under-represented motif rows.
- `patient_motif_matrix_absence_depletion.csv`: patient-by-motif matrix retaining only under-represented motifs, scored as expected frequency minus observed frequency.
- `patient_motif_matrix_absence_observed_frequency.csv`: CISM-style observed-frequency matrix for only under-represented motifs.
- `significant_motifs_proliferation.csv`: significantly enriched motif rows.
- `patient_motif_matrix_all_significant_signed_delta.csv`: combined two-tailed significant matrix, where positive values indicate enrichment and negative values indicate under-representation.

## Summary

{summary_text}

## Interpretation Note

The current FANMOD+ CSV artifacts contain motifs observed in the original network. Therefore, this postprocessing captures significantly depleted observed configurations. Motifs that are completely absent from the original network but expected under the null cannot be recovered from these artifacts unless FANMOD+ is modified to emit zero-count motifs as well.
"""


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    rows = [[_format_markdown_value(value) for value in row] for row in df.to_numpy()]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def _format_markdown_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


if __name__ == "__main__":
    main()
