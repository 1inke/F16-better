param(
    [switch]$DryRun,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$Python = "C:\coding_toobox\envs\astra2\python.exe"
$Series = $PSScriptRoot
$Runner = Join-Path $Series "run_v2_coverage_propagation.py"
$Analyzer = Join-Path $Series "analyze_v2_coverage_propagation.py"
$RunArgs = @()
if ($Force) {
    $RunArgs += "--force"
}
if ($DryRun) {
    $RunArgs += "--dry-run"
}

Write-Host "[1/3] Auditing H3 sources, existing C00, config identity, and validation IDs..."
& $Python -u $Runner --stage preflight

Write-Host "[2/3] Training one new V2 condition: H3 coverage=0 -> V2 coverage=.20..."
& $Python -u $Runner --stage v2_off @RunArgs

if ($DryRun) {
    Write-Host "[DRY RUN] Training command validated; analysis was not started."
    return
}

Write-Host "[3/3] Evaluating H3 and V2 propagation at best and matched final checkpoints..."
& $Python -u $Analyzer `
    --device cuda:0 `
    --batch-size 128 `
    --bootstrap-repetitions 10000 `
    --bootstrap-seed 20260922

Write-Host "[DONE] Review:"
Write-Host (Join-Path $Series "results\analysis\V2_COVERAGE_PROPAGATION_REPORT.md")
