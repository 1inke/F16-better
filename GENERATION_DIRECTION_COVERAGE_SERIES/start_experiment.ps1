$ErrorActionPreference = 'Stop'
$experimentRoot = $PSScriptRoot
$pythonExe = 'C:\coding_toobox\envs\astra2\python.exe'
$entry = Join-Path $experimentRoot 'run_direction_coverage.py'
$stdout = Join-Path $experimentRoot 'stdout.log'
$stderr = Join-Path $experimentRoot 'stderr.log'
if ((Test-Path -LiteralPath $stdout) -or (Test-Path -LiteralPath $stderr)) {
    throw 'Existing run logs found. Inspect before restarting; do not overwrite.'
}
& $pythonExe $entry --action preflight --device cuda:0
if ($LASTEXITCODE -ne 0) { throw 'Preflight failed' }
$argumentLine = '"' + $entry + '" --action run --device cuda:0'
$trainingProcess = Start-Process -FilePath $pythonExe -ArgumentList $argumentLine -WorkingDirectory $experimentRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
[ordered]@{pid=$trainingProcess.Id;launched_at=(Get-Date).ToString('o');stdout=$stdout;stderr=$stderr} | ConvertTo-Json
