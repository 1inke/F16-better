param([switch]$Force)

$ErrorActionPreference = "Stop"
$Python = "C:\coding_toobox\envs\astra2\python.exe"
$Series = $PSScriptRoot
$Runner = Join-Path $Series "run_h3_objective_attribution.py"
$Analyzer = Join-Path $Series "analyze_h3_objective_attribution.py"
$ForceArg = @()
if ($Force) {
    $ForceArg = @("--force")
}

Write-Host "[1/3] Preflight and provenance audit..."
& $Python -u $Runner --condition preflight

Write-Host "[2/3] Training the two missing H3 cells (ETH, seed 1, validation only)..."
& $Python -u $Runner --condition all @ForceArg

Write-Host "[3/3] Evaluating best checkpoints and matched epoch-30 models..."
& $Python -u $Analyzer `
    --device cuda:0 `
    --batch-size 128 `
    --bootstrap-repetitions 10000 `
    --bootstrap-seed 20260922

Write-Host "[DONE] Review:"
Write-Host (Join-Path $Series "results\analysis\H3_OBJECTIVE_ATTRIBUTION_REPORT.md")
