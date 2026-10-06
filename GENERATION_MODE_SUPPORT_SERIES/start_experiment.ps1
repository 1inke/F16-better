$ErrorActionPreference = 'Stop'
$experimentRoot = $PSScriptRoot
$pythonExe = 'C:\coding_toobox\envs\astra2\python.exe'
$entry = Join-Path $experimentRoot 'run_mode_support.py'
$stdout = Join-Path $experimentRoot 'stdout.log'
$stderr = Join-Path $experimentRoot 'stderr.log'
if ((Test-Path -LiteralPath $stdout) -or (Test-Path -LiteralPath $stderr)) {
    throw 'Existing launch logs found. Inspect the prior run; do not overwrite.'
}
& $pythonExe $entry --action preflight --device cuda:0
if ($LASTEXITCODE -ne 0) { throw 'Preflight failed' }
$argumentLine = '"' + $entry + '" --action run --device cuda:0'
$trainingProcess = Start-Process -FilePath $pythonExe -ArgumentList $argumentLine -WorkingDirectory $experimentRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
$launch = [ordered]@{ pid = $trainingProcess.Id; launched_at = (Get-Date).ToString('o'); stdout = $stdout; stderr = $stderr; script = $entry }
$launch | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $experimentRoot 'launch.json') -Encoding UTF8
$launch | ConvertTo-Json
