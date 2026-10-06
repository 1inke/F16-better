param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("Preflight", "Train", "Analyze", "All", "Status")]
    [string]$Phase,
    [string]$Device = "cuda:0",
    [switch]$DryRun,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$Python = "C:\coding_toobox\envs\astra2\python.exe"
$Series = "C:\Users\linke\Documents\New project\RGDM_SUPPLEMENT_SERIES"
$Runner = Join-Path $Series "run_rgdm_factorial.py"
$Analyzer = Join-Path $Series "analyze_rgdm_factorial.py"

if ($Phase -eq "Status") {
    & $Python -u $Runner --stage status
    exit $LASTEXITCODE
}

if ($Phase -eq "Preflight") {
    & $Python -u $Runner --stage preflight
    exit $LASTEXITCODE
}

if (($Phase -eq "Train") -or ($Phase -eq "All")) {
    $Arguments = @("-u", $Runner, "--stage", "train", "--device", $Device)
    if ($DryRun) { $Arguments += "--dry-run" }
    if ($Force) { $Arguments += "--force" }
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "RGDM training stage failed: $LASTEXITCODE" }
    if ($DryRun) {
        Write-Host "[DRY RUN] Commands validated. No training or analysis was started."
        exit 0
    }
}

if (($Phase -eq "Analyze") -or ($Phase -eq "All")) {
    & $Python -u $Analyzer --device $Device --bootstrap-repetitions 10000
    if ($LASTEXITCODE -ne 0) { throw "RGDM analysis stage failed: $LASTEXITCODE" }
}

Write-Host "[DONE] Review:"
Write-Host (Join-Path $Series "results\analysis\RGDM_FACTORIAL_REPORT.md")

