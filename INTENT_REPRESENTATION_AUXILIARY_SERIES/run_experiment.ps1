param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("Pilot", "Analyze", "Status")]
    [string]$Phase,
    [string]$Device = "cuda:0",
    [switch]$DryRun,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$Python = "C:\coding_toobox\envs\astra2\python.exe"
$Series = "C:\Users\linke\Documents\New project\INTENT_REPRESENTATION_AUXILIARY_SERIES"
$Runner = Join-Path $Series "run_ira1.py"

if ($Phase -eq "Status") {
    $Paths = @(
        (Join-Path $Series "results\ira0_detached\f16_temporal_social_cross\seed_1\val_scene_-1\metrics.json"),
        (Join-Path $Series "results\ira1_joint\f16_temporal_social_cross\seed_1\val_scene_-1\metrics.json"),
        (Join-Path $Series "results\analysis\verdict.json")
    )
    foreach ($Path in $Paths) {
        if (Test-Path -LiteralPath $Path) {
            $Item = Get-Item -LiteralPath $Path
            Write-Host "[ready] $($Item.FullName)  $($Item.LastWriteTime)"
        } else {
            Write-Host "[pending] $Path"
        }
    }
    exit 0
}

$Stage = $Phase.ToLowerInvariant()
$Arguments = @("-u", $Runner, "--stage", $Stage, "--device", $Device)
if ($DryRun) { $Arguments += "--dry-run" }
if ($Force) { $Arguments += "--force" }
& $Python @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "IRA1 phase failed with exit code $LASTEXITCODE"
}

