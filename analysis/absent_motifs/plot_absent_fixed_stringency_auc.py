from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = _REPO_ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cism.optimization import _clone_feature_conf


RUNTIME = ROOT / "melanoma_tutorials/melanoma_tutorials/runtime"
ARTIFACTS = {
    "two_tailed": RUNTIME / "absent_discriminative_cism_nn_np/Melanoma_NN_NP_absent_cism_ready.pkl",
    "lower_tailed": RUNTIME / "absent_discriminative_cism_nn_np_onetailed/Melanoma_NN_NP_absent_cism_ready.pkl",
}
STRINGENCIES = (0.20, 0.30, 0.35, 0.40)
COLORS = {"two_tailed": "#68449c", "lower_tailed": "#27856b"}
LABELS = {
    "two_tailed": "Two-tailed normal Z, p < 0.05",
    "lower_tailed": "Lower-tail normal Z, p < 0.05",
}


def evaluate(cache_path: Path, n_jobs: int) -> pd.DataFrame:
    if cache_path.exists():
        rows = pd.read_csv(cache_path).to_dict("records")
    else:
        rows = []
    completed = {(row["test"], round(float(row["stringency"]), 8)) for row in rows}
    for test, artifact_path in ARTIFACTS.items():
        pending = [x for x in STRINGENCIES if (test, round(x, 8)) not in completed]
        if not pending:
            continue
        with artifact_path.open("rb") as handle:
            artifact = pickle.load(handle)
        discriminator = artifact["discriminator"]
        base_conf = artifact["feature_conf"]
        if list(base_conf.labels) != ["NN", "NP"]:
            raise ValueError(f"Unexpected labels in {artifact_path}: {base_conf.labels}")
        if discriminator.cism.motif_size != 4:
            raise ValueError(f"Expected 4-node motifs in {artifact_path}")
        for stringency in pending:
            conf = _clone_feature_conf(
                base_conf, shared_percentage=stringency, max_class_features=30,
            )
            print(f"Evaluating {test} at stringency={stringency:.2f}", flush=True)
            try:
                result = discriminator.analyze_motifs(
                    feature_conf=conf,
                    exclude_patients=[],
                    n_jobs=n_jobs,
                    prefer="threads",
                    random_state=np.random.RandomState(0),
                )
                auc = float(result.get_roc_auc_score())
                status = "ok"
            except ValueError as exc:
                if "No discriminative features were selected" not in str(exc):
                    raise
                auc = np.nan
                status = "no_features"
            rows.append({
                "test": test,
                "stringency": stringency,
                "roc_auc": auc,
                "status": status,
                "max_class_features": 30,
            })
            pd.DataFrame(rows).sort_values(["test", "stringency"]).to_csv(cache_path, index=False)
            print(f"  AUC={auc:.6f} ({status})", flush=True)
    return pd.DataFrame(rows).sort_values(["test", "stringency"])


def plot_auc(data: pd.DataFrame, tests: tuple[str, ...], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for test in tests:
        series = data.loc[data.test.eq(test)].sort_values("stringency")
        if series.empty:
            raise ValueError(f"No evaluations for {test}")
        ax.plot(
            series.stringency, series.roc_auc,
            color=COLORS[test], marker="o", markersize=5.5,
            linewidth=2.3,
            label="4-node anti-motifs" if len(tests) == 1 else LABELS[test],
        )
    ax.set(
        xlim=(0.2, 0.5), ylim=(0.2, 1.0),
        xlabel="Discrimination stringency", ylabel="NN-NP LOOCV ROC AUC",
    )
    ax.set_xticks([0.2, 0.3, 0.35, 0.4, 0.5])
    ax.set_yticks(np.arange(0.2, 1.01, 0.1))
    ax.tick_params(axis="both", labelsize=13)
    ax.xaxis.label.set_size(15)
    ax.yaxis.label.set_size(15)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right", fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=240, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate 4-node melanoma anti-motif AUC at fixed stringencies.")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument(
        "--output-dir", type=Path,
        default=RUNTIME / "absent_discriminative_cism_nn_np/figures",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = evaluate(args.output_dir / "auc_fixed_stringencies_two_vs_lower_tail.csv", args.n_jobs)
    for tests, name in (
        (("two_tailed",), "auc_fixed_stringencies_4node_two_tailed"),
        (("lower_tailed",), "auc_fixed_stringencies_4node_lower_tailed"),
        (("two_tailed", "lower_tailed"), "auc_fixed_stringencies_4node_comparison"),
    ):
        output = args.output_dir / f"{name}.png"
        plot_auc(data, tests, output)
        print(output, flush=True)


if __name__ == "__main__":
    main()
