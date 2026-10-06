param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("Preflight", "PrepareTrain", "Train", "PrepareTest", "Evaluate", "Analyze", "All", "Status", "Smoke")]
    [string]$Phase,
    [string]$Device = "cuda:0",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$Python = "C:\coding_toobox\envs\astra2\python.exe"
$Runner = "C:\Users\linke\Documents\New project\REGIME_ADAPTIVE_SELECTION_FORMAL_SERIES\run_formal_regime_selection.py"
$stage = $Phase.ToLowerInvariant()
$arguments = @("-u", $Runner, "--stage", $stage, "--device", $Device)
if ($Force) { $arguments += "--force" }
& $Python @arguments
if ($LASTEXITCODE -ne 0) { throw "Formal regime-selection experiment failed in phase $Phase with code $LASTEXITCODE" }
