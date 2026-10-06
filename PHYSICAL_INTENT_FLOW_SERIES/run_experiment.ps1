param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("Audit", "Pilot", "Analyze")]
    [string]$Phase,
    [string]$Device = "cuda:0",
    [switch]$DryRun,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$Python = "C:\coding_toobox\envs\astra2\python.exe"
$Runner = "C:\Users\linke\Documents\New project\PHYSICAL_INTENT_FLOW_SERIES\run_pif1.py"
$Stage = $Phase.ToLowerInvariant()
$Arguments = @("-u", $Runner, "--stage", $Stage, "--device", $Device)
if ($DryRun) { $Arguments += "--dry-run" }
if ($Force) { $Arguments += "--force" }
& $Python @Arguments
if ($LASTEXITCODE -ne 0) { throw "PIF1 phase failed with exit code $LASTEXITCODE" }

