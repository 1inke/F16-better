param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("Preflight", "All", "Status")]
    [string]$Phase,
    [string]$Device = "cuda:0"
)

$ErrorActionPreference = "Stop"
$Python = "C:\coding_toobox\envs\astra2\python.exe"
$Series = "C:\Users\linke\Documents\New project\REGIME_CHANGE_DETECTOR_SERIES"
$Runner = Join-Path $Series "run_detector_experiment.py"

if ($Phase -eq "Preflight") {
    & $Python -u $Runner --stage preflight --device $Device
    exit $LASTEXITCODE
}

if ($Phase -eq "Status") {
    & $Python -u $Runner --stage status --device $Device
    exit $LASTEXITCODE
}

& $Python -u $Runner --stage preflight --device $Device
if ($LASTEXITCODE -ne 0) { throw "Detector preflight failed: $LASTEXITCODE" }
& $Python -u $Runner --stage run --device $Device
if ($LASTEXITCODE -ne 0) { throw "Detector experiment failed: $LASTEXITCODE" }

