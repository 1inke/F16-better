param(
    [ValidateSet("H3", "V2")]
    [string]$Phase = "H3",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$Python = "C:\coding_toobox\envs\astra2\python.exe"
$Series = $PSScriptRoot
$Runner = Join-Path $Series "run_stage_attribution.py"
$Analyzer = Join-Path $Series "analyze_stage_attribution.py"
$ForceArg = @()
if ($Force) {
    $ForceArg = @("--force")
}

if ($Phase -eq "H3") {
    Write-Host "[0/3] Auditing configs, baseline checkpoints, and validation provenance..."
    & $Python -u $Runner --stage preflight

    Write-Host "[1/3] Training H3-NoDiv (ETH, seed 1, validation only)..."
    & $Python -u $Runner --stage h3 @ForceArg

    Write-Host "[2/3] Evaluating H3 baseline versus H3-NoDiv..."
    & $Python -u $Analyzer --phase h3 --device cuda:0 --batch-size 128

    Write-Host "[3/3] Stop point reached. Review:"
    Write-Host (Join-Path $Series "results\h3_interim_analysis\H3_INTERIM_REPORT.md")
    Write-Host "Run again with -Phase V2 only after this interim report is acceptable."
    return
}

Write-Host "[1/3] Training C10: H3=0 -> V2=0.05..."
& $Python -u $Runner --stage c10 @ForceArg

Write-Host "[2/3] Training C11: H3=0 -> V2=0..."
& $Python -u $Runner --stage c11 @ForceArg

Write-Host "[3/3] Running the final validation-only paired analysis..."
& $Python -u $Analyzer `
    --phase final `
    --device cuda:0 `
    --batch-size 128 `
    --bootstrap-repetitions 10000 `
    --bootstrap-seed 20260922

Write-Host "[DONE] Review before any test access:"
Write-Host (Join-Path $Series "results\stage_attribution_analysis\STAGE_ATTRIBUTION_REPORT.md")
