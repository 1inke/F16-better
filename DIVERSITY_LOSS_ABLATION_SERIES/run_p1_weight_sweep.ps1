$ErrorActionPreference = "Stop"

# Run from the repository root after copying the DIVERSITY_LOSS_ABLATION_SERIES folder there.
$Python = "C:\coding_toobox\envs\astra2\python.exe"
$Series = Join-Path $PSScriptRoot "DIVERSITY_LOSS_ABLATION_SERIES"
if (-not (Test-Path $Series)) {
    # If the .ps1 itself already lives inside DIVERSITY_LOSS_ABLATION_SERIES.
    $Series = $PSScriptRoot
}

Write-Host "[P1] Training diversity-weight sweep..."
& $Python -u (Join-Path $Series "run_diversity_ablation.py") `
    --stage weight `
    --weights "0.000,0.010,0.025,0.050" `
    --fixed-margin 0.400 `
    --seeds "1" `
    --epochs 20 `
    --batch-size 64 `
    --device cuda:0

Write-Host "[P1] Analyzing validation candidates..."
& $Python -u (Join-Path $Series "analyze_diversity_ablation.py") `
    --sweep-root (Join-Path $Series "results\weight_sweep") `
    --device cuda:0 `
    --batch-size 128

Write-Host "[DONE] Open:"
Write-Host (Join-Path $Series "results\weight_sweep\DIVERSITY_ABLATION_REPORT.md")
Write-Host (Join-Path $Series "results\weight_sweep\condition_summary.csv")
Write-Host (Join-Path $Series "results\weight_sweep\paired_vs_baseline.csv")
