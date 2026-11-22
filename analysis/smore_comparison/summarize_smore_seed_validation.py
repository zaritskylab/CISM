#!/usr/bin/env python3
"""Compare significant FOV420 SMORE motifs across 10% URPEN sampling seeds."""

from __future__ import annotations

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


from itertools import combinations
from pathlib import Path

import pandas as pd


ROOT = _REPO_ROOT
RUNS = ROOT / "analysis/outputs/smore_matched/runs"
OUTPUT = ROOT / "analysis/outputs/smore_matched/seed_validation"
SEEDS = (1240, 1241, 1242, 1243)


def run_dir(seed: int) -> Path:
    suffix = "" if seed == 1240 else f"_seed{seed}"
    return RUNS / f"gdelaunay_p0.1_n5_train50_score10{suffix}"


def main() -> None:
    rows = []
    summary = []
    motif_sets = {}
    for seed in SEEDS:
        folder = run_dir(seed)
        motifs = pd.read_csv(folder / "FOV420_motifs.csv")
        metadata = pd.read_csv(folder / "FOV420_metadata.csv")
        if len(metadata) != 1 or metadata.path_probability.iloc[0] != 0.1:
            raise ValueError(f"Invalid 10% metadata for seed {seed}")
        if seed != 1240 and int(metadata.random_seed.iloc[0]) != seed:
            raise ValueError(f"Random seed mismatch for seed {seed}")
        if len(motifs) and not ((motifs.path_probability == 0.1) &
                                (motifs.n_train == 50) & (motifs.n_score == 10)).all():
            raise ValueError(f"Parameter mismatch for seed {seed}")
        significant = motifs[motifs.p_value < 0.05]
        motif_sets[seed] = set(significant.consensus_cell_types)
        summary.append({"random_seed": seed,
                        "sampled_radial_paths": int(metadata.sampled_radial_paths.iloc[0]),
                        "output_motifs": len(motifs),
                        "significant_motifs": len(significant),
                        "unique_significant_consensus": len(motif_sets[seed])})
        for motif in motifs.itertuples(index=False):
            rows.append({"random_seed": seed, "motif_number": motif.motif_number,
                         "consensus_cell_types": motif.consensus_cell_types,
                         "p_value": motif.p_value,
                         "significant": motif.p_value < 0.05,
                         "sampled_frequency": motif.sampled_frequency})
    comparisons = []
    for a, b in combinations(SEEDS, 2):
        intersection = motif_sets[a] & motif_sets[b]
        union = motif_sets[a] | motif_sets[b]
        comparisons.append({"seed_a": a, "seed_b": b,
                            "shared_significant_consensus": len(intersection),
                            "union_significant_consensus": len(union),
                            "jaccard": len(intersection) / len(union) if union else 1.0})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT / "motifs_by_seed.csv", index=False)
    pd.DataFrame(summary).to_csv(OUTPUT / "run_summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(OUTPUT / "pairwise_consensus_overlap.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False))
    print(pd.DataFrame(comparisons).to_string(index=False))


if __name__ == "__main__":
    main()
