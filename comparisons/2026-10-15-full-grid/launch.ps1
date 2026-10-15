# Full-catalog thesis-replication grid launcher for Windows: background-safe,
# resume-friendly mirror of tools/launch_full_grid.sh.
#
# Stages (all written into one output directory; existing run files are skipped,
# so re-running this script resumes where it stopped):
#   A: entire canonical function catalog x dims {10, 100} x both sides (new + frozen);
#   B: entire canonical function catalog x dim 1000 x new side (frozen side is out of
#      scope at dim 1000 by design). This merges the originally planned starred-only
#      stage with the previously deferred non-starred cells, now feasible thanks to the
#      ASEBO reduced-spectrum eigenstep (PR #10);
#   C: profiles, D: report over everything available.
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Resolve-Path (Join-Path $PSScriptRoot '..\..'))

$OUT = 'comparisons/2026-10-15-full-grid'
New-Item -ItemType Directory -Force -Path "$OUT/logs" | Out-Null

# Cap per-thread BLAS/OpenMP threads: 3 workers x 2 BLAS threads ~= 6 busy cores,
# leaving headroom on this 12-core machine.
$env:OMP_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:VECLIB_MAXIMUM_THREADS = '2'
$env:NUMEXPR_MAX_THREADS = '2'

function Invoke-Stage {
    param([string]$Name, [string[]]$StageArgs)
    $log = Join-Path $OUT "logs/$Name.log"
    $stamp = Get-Date -Format s
    Write-Output "=== [$Name] start $stamp ==="
    # Append (never truncate): resumes must preserve the history of prior invocations.
    Add-Content -LiteralPath $log -Value "`n=== [$Name] start $stamp | args: $($StageArgs -join ' ') ==="
    & uv run python tools/run_thesis_replication.py @StageArgs *>> $log
    if ($LASTEXITCODE -ne 0) {
        Write-Output "=== [$Name] FAILED (exit $LASTEXITCODE) at $(Get-Date -Format s); re-run this script to resume ==="
        exit $LASTEXITCODE
    }
    Write-Output "=== [$Name] done $(Get-Date -Format s) ==="
}

Invoke-Stage 'stageA_runs' @('--phase', 'runs', '--side', 'both', '--dims', '10', '100',
    '--old-dims', '10', '100', '--all-functions', '--workers', '3', '--out', $OUT)

Invoke-Stage 'stageB_runs' @('--phase', 'runs', '--side', 'new', '--dims', '1000',
    '--all-functions', '--workers', '3', '--out', $OUT)

Invoke-Stage 'profiles' @('--phase', 'profiles', '--side', 'both', '--dims', '10', '100', '1000',
    '--old-dims', '10', '100', '--all-functions', '--out', $OUT)

Invoke-Stage 'report' @('--phase', 'report', '--side', 'both', '--dims', '10', '100', '1000',
    '--old-dims', '10', '100', '--all-functions', '--out', $OUT)

Write-Output "done; output in $OUT"
