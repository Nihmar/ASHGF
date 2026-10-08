#!/usr/bin/env bash
# Full-catalog replication grid launcher (background-safe, resume-friendly).
#
# Stages (all written into one output directory, existing run files skipped):
#   A: entire canonical function catalog x dims {10, 100} x both sides (new + frozen);
#   B: thesis-starred functions x dim 1000 x new side (frozen side is out of scope at
#      dim 1000 by design); non-starred dim-1000 cells are added later via
#      `--phase runs --side new --dims 1000 --all-functions` once the ASEBO
#      eigenstep optimization lands (its full-spectrum eigh makes dim-1000 cells
#      ~1s/iter today);
#   C: profiles + report over everything available.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="comparisons/$(date +%F)-full-grid"
mkdir -p "$OUT/logs"

# Cap per-thread BLAS/OpenMP threads: 3 worker threads x 2 BLAS threads ~= 6 busy
# cores, leaving headroom on this shared 12-core box.
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
       VECLIB_MAXIMUM_THREADS=2 NUMEXPR_MAX_THREADS=2

echo "=== [stage A] full catalog, dims 10+100, both sides ==="
uv run python tools/run_thesis_replication.py \
    --phase runs --side both --dims 10 100 --old-dims 10 100 \
    --all-functions --workers 3 --out "$OUT" >"$OUT/logs/stageA_runs.log" 2>&1

echo "=== [stage B] starred functions, dim 1000, new side ==="
uv run python tools/run_thesis_replication.py \
    --phase runs --side new --dims 1000 \
    --workers 3 --out "$OUT" >"$OUT/logs/stageB_runs.log" 2>&1

echo "=== [profiles] ==="
uv run python tools/run_thesis_replication.py \
    --phase profiles --side both --dims 10 100 1000 --old-dims 10 100 \
    --all-functions --out "$OUT" >"$OUT/logs/profiles.log" 2>&1

echo "=== [report] ==="
uv run python tools/run_thesis_replication.py \
    --phase report --side both --dims 10 100 1000 --old-dims 10 100 \
    --all-functions --out "$OUT" >"$OUT/logs/report.log" 2>&1

echo "done; output in $OUT"
