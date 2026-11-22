#!/usr/bin/env python3
"""Run the pinned official SMORE implementation on matched melanoma FOVs."""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import argparse
import subprocess
from pathlib import Path

import pandas as pd


ROOT = _REPO_ROOT
OUTPUT = ROOT / "analysis/outputs/smore_matched"
MATLAB = Path("/storage/modules/packages/matlab/R2023b/bin/matlab")
SMORE_SHA = "580338e5ffd473a011d9f1a658df138a42824417"


def matlab_quote(value: Path) -> str:
    return str(value.resolve()).replace("'", "''")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "external/SMORE")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--fov", action="append", help="Restrict to one or more FOV IDs")
    parser.add_argument("--task-id", type=int, help="Zero-based FOV row for a SLURM array task")
    parser.add_argument("--limit", type=int, help="Maximum number of FOVs this invocation")
    parser.add_argument("--all-fovs", action="store_true", help="Explicitly permit a full-cohort run")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout", type=int, default=80000)
    parser.add_argument("--path-probability", type=float, default=1)
    parser.add_argument("--graph-mode", choices=("delaunay", "epsilon"), default="delaunay")
    parser.add_argument("--max-motifs", type=int, default=5)
    parser.add_argument("--n-train", type=int, default=50)
    parser.add_argument("--n-score", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=1240)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--no-enrich", action="store_true")
    args = parser.parse_args()
    if not (args.fov or args.task_id is not None or args.limit is not None or args.all_fovs):
        parser.error("Choose --fov, --task-id, --limit, or --all-fovs explicitly")
    if not 0 < args.path_probability <= 1:
        parser.error("--path-probability must be in (0, 1]")
    if args.random_seed < 0:
        parser.error("--random-seed must be nonnegative")

    manifest = pd.read_csv(args.output_dir / "cohort_fovs.csv")
    if (manifest.patient.nunique(), manifest.fov.nunique()) != (38, 76):
        raise ValueError("The SMORE manifest does not contain the complete 38-patient cohort")
    if manifest.groupby("class").patient.nunique().to_dict() != {"NN": 20, "NP": 18}:
        raise ValueError("The SMORE manifest has unexpected class counts")
    sha = subprocess.check_output(["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip()
    if sha != SMORE_SHA:
        raise ValueError(f"SMORE source commit {sha} does not match pinned {SMORE_SHA}")
    for mex in ("getZNICIM.mexa64", "getBinomLogPvalues.mexa64"):
        if not (args.source / "func" / mex).exists():
            raise FileNotFoundError(f"Compile {mex} in the SMORE func folder before running")

    tag = (f"g{args.graph_mode}_p{args.path_probability:g}_n{args.max_motifs}_"
           f"train{args.n_train}_score{args.n_score}")
    if args.random_seed != 1240:
        tag += f"_seed{args.random_seed}"
    if args.backend != "cpu":
        tag += f"_{args.backend}"
    if args.no_enrich:
        tag += "_noenrich"
    run_dir = args.output_dir / "runs" / tag
    log_dir = args.output_dir / "logs" / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    selected = manifest.sort_values("fov")
    if args.task_id is not None:
        if args.fov or args.limit is not None:
            raise ValueError("--task-id cannot be combined with --fov or --limit")
        if args.task_id < 0 or args.task_id >= len(selected):
            raise ValueError(f"Task ID must be in 0..{len(selected) - 1}")
        selected = selected.iloc[[args.task_id]]
    if args.fov:
        unknown = set(args.fov) - set(manifest.fov)
        if unknown:
            raise ValueError(f"Unknown FOVs: {sorted(unknown)}")
        selected = selected[selected.fov.isin(args.fov)]
    if args.limit is not None:
        selected = selected.head(args.limit)

    for row in selected.itertuples(index=False):
        catalog = run_dir / f"{row.fov}_motifs.csv"
        mat = run_dir / f"{row.fov}.mat"
        if catalog.exists() and mat.exists():
            print(f"{row.fov}: complete, skipping", flush=True)
            continue
        source_csv = Path(row.smore_input)
        if not source_csv.is_file():
            raise FileNotFoundError(source_csv)
        statement = (
            f"addpath('{matlab_quote(ROOT / 'analysis/smore_comparison')}'); "
            f"run_smore_matched_fov('{matlab_quote(source_csv)}',"
            f"'{matlab_quote(run_dir)}','{matlab_quote(args.source)}',"
            f"{args.path_probability},{args.max_motifs},{args.n_train},{args.n_score},"
            f"'{args.graph_mode}',{args.random_seed},'{args.backend}',"
            f"{str(not args.no_enrich).lower()})"
        )
        command = [str(MATLAB), "-batch", statement]
        print(f"{row.fov}: running {row.graph_vertices} cells", flush=True)
        if args.dry_run:
            continue
        with (log_dir / f"{row.fov}.log").open("w") as log:
            try:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                               timeout=args.timeout, check=True)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                raise RuntimeError(f"SMORE failed on {row.fov}; inspect {log.name}") from exc
        if not (catalog.exists() and mat.exists()):
            raise RuntimeError(f"SMORE returned without both output files for {row.fov}")


if __name__ == "__main__":
    main()
