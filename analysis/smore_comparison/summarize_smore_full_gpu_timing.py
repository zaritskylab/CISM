#!/usr/bin/env python3
"""Summarize elapsed GPU task time for a full-sampling SMORE run."""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import argparse
import re
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = _REPO_ROOT
OUTPUT = ROOT / "analysis/outputs/smore_matched"
RUN_TAG = "gdelaunay_p1_n0_train10_score2_gpu_noenrich"
START = re.compile(r"STARTED_UTC=(\S+)")
FINISH = re.compile(r"FINISHED_UTC=(\S+) ELAPSED_SECONDS=(\d+)")


def read_timing(path: Path) -> dict:
    log = path.read_text(errors="replace")
    start = START.search(log)
    finish = FINISH.search(log)
    if not (start and finish):
        raise ValueError(f"Missing complete timing markers in {path}")
    if "Traceback" in log or "RuntimeError" in log:
        raise ValueError(f"Task failed: {path}")
    return {
        "started_utc": datetime.fromisoformat(start.group(1).replace("Z", "+00:00")),
        "finished_utc": datetime.fromisoformat(finish.group(1).replace("Z", "+00:00")),
        "elapsed_seconds": int(finish.group(2)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-job-id", required=True)
    parser.add_argument("--cohort-job-id", required=True)
    args = parser.parse_args()

    manifest = pd.read_csv(OUTPUT / "cohort_fovs.csv").sort_values("fov").reset_index(drop=True)
    log_dir = OUTPUT / "slurm"
    pilot_fovs = {0: "FOV152", 1: "FOV420"}
    pilot = {
        fov: read_timing(log_dir / f"gpu_full_pilot_{args.pilot_job_id}_{task_id}.out")
        for task_id, fov in pilot_fovs.items()
    }
    cohort = {
        row.fov: read_timing(log_dir / f"gpu_full_cohort_{args.cohort_job_id}_{task_id}.out")
        for task_id, row in manifest.iterrows()
    }
    rows = []
    for row in manifest.itertuples(index=False):
        source = "pilot" if row.fov in pilot else "cohort"
        timing = pilot[row.fov] if source == "pilot" else cohort[row.fov]
        rows.append({"patient": row.patient, "class": row[1], "fov": row.fov,
                     "source": source, **timing})
    per_fov = pd.DataFrame(rows)
    run_dir = OUTPUT / "runs" / RUN_TAG
    if len(per_fov) != 76 or not all((run_dir / f"{fov}_motifs.csv").is_file()
                                     for fov in per_fov.fov):
        raise ValueError("Full-sampling run does not have all 76 motif catalogs")
    per_fov.to_csv(run_dir / "smore_full_gpu_timing_per_fov.csv", index=False)

    all_starts = [item["started_utc"] for item in (*pilot.values(), *cohort.values())]
    all_finishes = [item["finished_utc"] for item in (*pilot.values(), *cohort.values())]
    summary = pd.DataFrame([{
        "fovs": len(per_fov),
        "pilot_wall_seconds": (max(item["finished_utc"] for item in pilot.values())
                               - min(item["started_utc"] for item in pilot.values())).total_seconds(),
        "cohort_wall_seconds": (max(item["finished_utc"] for item in cohort.values())
                                - min(item["started_utc"] for item in cohort.values())).total_seconds(),
        "end_to_end_wall_seconds": (max(all_finishes) - min(all_starts)).total_seconds(),
        "sum_fov_gpu_seconds": int(per_fov.elapsed_seconds.sum()),
        "median_fov_seconds": float(per_fov.elapsed_seconds.median()),
        "max_fov_seconds": int(per_fov.elapsed_seconds.max()),
        "pilot_job_id": args.pilot_job_id,
        "cohort_job_id": args.cohort_job_id,
    }])
    summary.to_csv(run_dir / "smore_full_gpu_timing_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
