from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = _REPO_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "melanoma_tutorials"
    / "melanoma_tutorials"
    / "runtime"
    / "absent_discriminative_cism_nn_np"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot absent-motif stringency tradeoffs for melanoma NN/NP CISM."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--selected-motifs", type=Path, default=None)
    parser.add_argument("--optuna-trials", type=Path, default=None)
    parser.add_argument("--loocv-results", type=Path, default=None)
    parser.add_argument("--stringency", type=float, default=0.3642694428844888)
    args = parser.parse_args()

    selected_path = args.selected_motifs or args.output_dir / "absent_discriminative_motifs.csv"
    trials_path = args.optuna_trials or args.output_dir / "absent_stringency_optuna_trials.csv"
    loocv_path = args.loocv_results or args.output_dir / "absent_discriminative_loocv_results.csv"

    selected = pd.read_csv(selected_path)
    trials = pd.read_csv(trials_path)
    loocv = pd.read_csv(loocv_path)

    rows = []
    for _, trial in trials.sort_values("params_shared_percentage").iterrows():
        stringency = float(trial["params_shared_percentage"])
        passing = selected[selected["patient_percentage"] > stringency].copy()
        capped = (
            passing.sort_values("Freq", ascending=False)
            .groupby("patient_class", observed=True)
            .head(int(trial["params_max_class_features"]))
        )
        rows.append(
            {
                "trial": int(trial["number"]),
                "stringency": stringency,
                "roc_auc": float(trial["value"]),
                "candidate_rows": int(len(passing)),
                "candidate_unique_subgraphs": int(passing["ID"].nunique()),
                "capped_selected_rows": int(len(capped)),
                "capped_selected_unique_subgraphs": int(capped["ID"].nunique()),
                "max_class_features": int(trial["params_max_class_features"]),
                "state": trial["state"],
            }
        )

    tradeoff = pd.DataFrame(rows)
    tradeoff.to_csv(args.output_dir / "absent_stringency_tradeoff.csv", index=False)

    target = selected[selected["patient_percentage"] > args.stringency]
    capped_target = (
        target.sort_values("Freq", ascending=False)
        .groupby("patient_class", observed=True)
        .head(30)
    )
    summary = {
        "stringency": args.stringency,
        "candidate_rows": int(len(target)),
        "candidate_unique_subgraphs": int(target["ID"].nunique()),
        "capped_selected_rows": int(len(capped_target)),
        "capped_selected_unique_subgraphs": int(capped_target["ID"].nunique()),
        "loocv_mean_cfeatures": float(loocv["cFeatures"].mean()),
        "loocv_median_cfeatures": float(loocv["cFeatures"].median()),
        "loocv_min_cfeatures": int(loocv["cFeatures"].min()),
        "loocv_max_cfeatures": int(loocv["cFeatures"].max()),
    }
    pd.Series(summary).to_json(args.output_dir / "absent_stringency_0364_subgraph_count.json", indent=2)

    plot_tradeoffs(tradeoff, args.output_dir)
    print(pd.Series(summary).to_string())
    print(f"Wrote {args.output_dir / 'absent_stringency_tradeoff.csv'}")


def plot_tradeoffs(tradeoff: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    tradeoff = tradeoff.sort_values("stringency")
    display_tradeoff = thin_adjacent_stringencies(tradeoff)
    auc_tradeoff = display_tradeoff[display_tradeoff["stringency"] <= 0.4].copy()
    best = tradeoff.sort_values("roc_auc", ascending=False).iloc[0]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=180)
    ax.plot(
        display_tradeoff["stringency"],
        display_tradeoff["candidate_unique_subgraphs"].clip(lower=1),
        marker="o",
        linewidth=2,
        color="#176B87",
    )
    ax.scatter(
        [best["stringency"]],
        [best["candidate_unique_subgraphs"]],
        color="#B42318",
        zorder=5,
        label=f"best AUC stringency = {best['stringency']:.3f}",
    )
    ax.set_xlabel("Stringency: shared patient fraction")
    ax.set_ylabel("Unique underrepresented subgraphs")
    ax.set_yscale("log")
    ax.set_title("Absent motif count vs stringency")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "absent_subgraphs_vs_stringency.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=180)
    ax.plot(
        auc_tradeoff["stringency"],
        auc_tradeoff["roc_auc"],
        marker="o",
        linewidth=2,
        color="#5B5F97",
    )
    ax.scatter(
        [best["stringency"]],
        [best["roc_auc"]],
        color="#B42318",
        zorder=5,
        label=f"best AUC = {best['roc_auc']:.3f}",
    )
    ax.set_xlabel("Stringency: shared patient fraction")
    ax.set_ylabel("LOOCV ROC AUC")
    ax.set_xlim(left=0, right=0.4)
    ax.set_ylim(0, 1)
    ax.set_title("AUC vs stringency")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "absent_auc_vs_stringency.png")
    plt.close(fig)


def thin_adjacent_stringencies(tradeoff: pd.DataFrame, min_spacing: float = 0.02) -> pd.DataFrame:
    """Keep one representative point from clusters of adjacent stringencies."""
    rows = []
    current_group = []
    for _, row in tradeoff.sort_values("stringency").iterrows():
        if not current_group:
            current_group.append(row)
            continue
        if float(row["stringency"]) - float(current_group[-1]["stringency"]) <= min_spacing:
            current_group.append(row)
        else:
            rows.append(select_group_representative(current_group))
            current_group = [row]
    if current_group:
        rows.append(select_group_representative(current_group))
    return pd.DataFrame(rows)


def select_group_representative(rows: list[pd.Series]) -> pd.Series:
    group = pd.DataFrame(rows)
    return group.sort_values(
        ["roc_auc", "candidate_unique_subgraphs", "stringency"],
        ascending=[False, False, True],
    ).iloc[0]


if __name__ == "__main__":
    main()
