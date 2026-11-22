# SMORE default-parameter pilot

Run `python -m analysis.smore_comparison.run_smore_matched_batch --help` from the repository root for the Python runner. Scheduler launchers are not distributed.
The two array tasks run official SMORE on the same small FOV (FOV420, 2,059 cells):

- Task 0: all eligible four-cell radial paths (`path_probability=1`).
- Task 1: uniform URPEN sampling of 10% of those paths (`path_probability=0.1`), matching the paper's large-3D example.

Override the sampled rate with `SMORE_SAMPLE_PROBABILITY` if needed.
The 10% arm may hit the pilot time limit.

Both use Delaunay graph construction, global within-FOV cell-label shuffling,
`nMotifs=5`, `nTrain=50`, `nScore=10`, `nEval=25`, `nRefine=4`, and
`nRefineIter=20`. These are the documented SMORE discovery defaults except
that the sampled arm uses the paper's large-3D example rate. The paper uses
full paths in some 2D examples; the MATLAB runner uses CPU-only compatibility
shims.

Each task has a 40-minute inner timeout and a 45-minute scheduler wall limit.
A timeout is a failed feasibility pilot, not a completed motif result. Outputs
are isolated in `analysis/outputs/smore_matched/runs/gdelaunay_p1_n5_train50_score10/`
and `runs/gdelaunay_p0.1_n5_train50_score10/`. Inspect their corresponding
logs under `analysis/outputs/smore_matched/logs/` and scheduler logs under
`analysis/outputs/smore_matched/slurm/` before considering a full-cohort run.

Full enumeration scales with the number of eligible paths, not just cell count;
the 45-minute pilot bound does not make a 76-FOV full-enumeration run fast.
Do not submit a cohort-wide full enumeration without a representative large-FOV
benchmark and an explicit runtime budget.
