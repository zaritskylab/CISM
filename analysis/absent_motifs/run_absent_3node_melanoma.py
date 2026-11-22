from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


import argparse
import csv
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

from joblib import Parallel, delayed

REPO_ROOT = _REPO_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cism.data_preparation import parse_network_filename  # noqa: E402
from cism.helpers import run_fanmod  # noqa: E402
from multi_class.config import (  # noqa: E402
    EXISTING_ARTIFACT,
    FANMOD_BINARY_DIR,
    FANMOD_ITERATIONS,
    ORIGINAL_DATASET_DIR,
    TUTORIAL_RUNTIME_DIR,
    fanmod_executable_name,
)

MOTIF_SIZE = 3
LABELS = ("NN", "NP")


def get_network_files(network_dir: Path, patient_class_csv: Path) -> list[tuple[Path, str]]:
    with patient_class_csv.open(newline="") as handle:
        patient_labels = dict(csv.reader(handle))
    selected = {
        patient_uid.removeprefix("Melanoma")
        for patient_uid, label in patient_labels.items()
        if label in LABELS
    }
    if not selected:
        raise ValueError(f"No NN/NP patients in {patient_class_csv}")
    files = []
    seen_patients = set()
    for path in sorted(network_dir.glob("*.txt")):
        patient_num, fov = parse_network_filename(path.name)
        if patient_num in selected:
            files.append((path, f"{patient_num}_{fov}"))
            seen_patients.add(patient_num)
    missing = selected - seen_patients
    if missing:
        raise ValueError(f"Missing network files for NN/NP patients: {sorted(missing)}")
    return files


def fanmod_output_complete(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    content = path.read_text(errors="replace")
    starts = content.count("#+leda format")
    return (
        "Subgraph size: 3" in content
        and " motifs were found." in content
        and "ID,Frequency,Count,Mean-Freq,Standard-Dev,Z-Score,p-Value" in content
        and starts > 0
        and starts == content.count("#-leda format")
        and content.rstrip().endswith("#-leda format")
    )


def run_one_fov(path: Path, name: str, binary: Path, output_dir: Path, network_dir: Path) -> bool:
    # FANMOD writes directly to its output path. Promote only a successful, complete file.
    started = time.monotonic()
    print(f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] FANMOD start {name}", flush=True)
    with tempfile.TemporaryDirectory(prefix=f".{name}.", dir=output_dir) as temporary:
        generated = Path(temporary) / f"{name}.csv"
        try:
            succeeded = run_fanmod(
                FANMOD_exe=binary.name,
                FANMOD_path=str(binary.parent),
                iterations=FANMOD_ITERATIONS,
                motif_size=MOTIF_SIZE,
                output_dir=temporary,
                raw_data_folder=str(network_dir),
                file=path.name,
                patient_num=name,
            )
        except Exception as exc:
            print(f"FANMOD error {name} after {time.monotonic() - started:.1f}s: {exc}", flush=True)
            raise
        if not succeeded or not fanmod_output_complete(generated):
            print(f"FANMOD incomplete {name} after {time.monotonic() - started:.1f}s", flush=True)
            return False
        os.replace(generated, output_dir / generated.name)
    print(f"FANMOD complete {name} in {time.monotonic() - started:.1f}s", flush=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 3-node FANMOD output and run two-tailed absent-motif CISM on melanoma NN/NP."
    )
    parser.add_argument("--network-dir", type=Path, default=ORIGINAL_DATASET_DIR)
    parser.add_argument("--patient-class-csv", type=Path, default=ORIGINAL_DATASET_DIR / "patient_class.csv")
    parser.add_argument("--serialized-artifact", type=Path, default=EXISTING_ARTIFACT)
    parser.add_argument("--fanmod-output-root", type=Path, default=TUTORIAL_RUNTIME_DIR / "fanmod_output/Melanoma")
    parser.add_argument("--output-dir", type=Path, default=TUTORIAL_RUNTIME_DIR / "absent_discriminative_cism_nn_np_3node")
    parser.add_argument("--fanmod-binary-dir", type=Path, default=FANMOD_BINARY_DIR)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--shared-percentage-max", type=float, default=0.95)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--skip-fanmod", action="store_true", help="Use existing complete 3-node FANMOD CSVs.")
    parser.add_argument("--force-fanmod", action="store_true", help="Regenerate existing 3-node FANMOD CSVs.")
    args = parser.parse_args()

    if args.n_jobs < 1 or args.n_trials < 1:
        parser.error("--n-jobs and --n-trials must be positive")
    if not 0 < args.alpha < 1 or not 0 < args.shared_percentage_max <= 1:
        parser.error("--alpha and --shared-percentage-max must be in (0, 1]")
    if args.skip_fanmod and args.force_fanmod:
        parser.error("--skip-fanmod and --force-fanmod are mutually exclusive")

    network_files = get_network_files(args.network_dir, args.patient_class_csv)
    output_3node = args.fanmod_output_root / str(MOTIF_SIZE)
    binary = args.fanmod_binary_dir / fanmod_executable_name()
    if not args.skip_fanmod:
        if not binary.is_file():
            raise FileNotFoundError(f"FANMOD executable not found: {binary}")
        output_3node.mkdir(parents=True, exist_ok=True)
        pending = [
            (path, name)
            for path, name in network_files
            if args.force_fanmod
            or not fanmod_output_complete(output_3node / f"{name}.csv")
        ]
        print(f"Running 3-node FANMOD on {len(pending)} of {len(network_files)} NN/NP FOVs", flush=True)
        results = Parallel(n_jobs=args.n_jobs, prefer="threads")(
            delayed(run_one_fov)(path, name, binary, output_3node, args.network_dir)
            for path, name in pending
        )
        if not all(results):
            raise RuntimeError(f"FANMOD failed for {sum(not result for result in results)} FOVs")

    incomplete = [
        name
        for _, name in network_files
        if not fanmod_output_complete(output_3node / f"{name}.csv")
    ]
    if incomplete:
        raise RuntimeError(f"Missing or incomplete 3-node FANMOD outputs for {len(incomplete)} FOVs: {incomplete[:5]}")

    command = [
        sys.executable,
        "-u",
        str(REPO_ROOT / "analysis/absent_motifs/run_absent_discriminative_cism.py"),
        "--serialized-artifact", str(args.serialized_artifact),
        "--fanmod-output-root", str(args.fanmod_output_root),
        "--output-dir", str(args.output_dir),
        "--motif-size", str(MOTIF_SIZE),
        "--fanmod-source-only",
        "--significance-mode", "normal_two_tailed",
        "--labels", *LABELS,
        "--alpha", str(args.alpha),
        "--max-class-features", "30",
        "--tune-stringency",
        "--n-trials", str(args.n_trials),
        "--shared-percentage-max", str(args.shared_percentage_max),
        "--n-jobs", str(args.n_jobs),
        "--random-seed", str(args.random_seed),
    ]
    print("Running absence CISM from 3-node FANMOD CSVs", flush=True)
    subprocess.run(command, check=True, cwd=REPO_ROOT)


if __name__ == "__main__":
    main()
