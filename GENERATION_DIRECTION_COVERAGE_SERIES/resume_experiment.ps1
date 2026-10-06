$ErrorActionPreference = 'Stop'
$experimentRoot = $PSScriptRoot
$pythonExe = 'C:\coding_toobox\envs\astra2\python.exe'
$entry = Join-Path $experimentRoot 'resume_experiment.py'
$runStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout = Join-Path $experimentRoot ('resume_' + $runStamp + '_stdout.log')
$stderr = Join-Path $experimentRoot ('resume_' + $runStamp + '_stderr.log')
if ((Test-Path -LiteralPath $stdout) -or (Test-Path -LiteralPath $stderr)) { throw 'Resume logs already exist' }
$activeRuns = Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' -and $_.CommandLine -match 'GENERATION_DIRECTION_COVERAGE_SERIES' }
if ($activeRuns) { throw 'Coverage experiment already running' }
& $pythonExe $entry --action preflight --device cuda:0
if ($LASTEXITCODE -ne 0) { throw 'Resume preflight failed' }
$argumentLine = '"' + $entry + '" --action run --device cuda:0'
$trainingProcess = Start-Process -FilePath $pythonExe -ArgumentList $argumentLine -WorkingDirectory $experimentRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
[ordered]@{pid=$trainingProcess.Id;launched_at=(Get-Date).ToString('o');stdout=$stdout;stderr=$stderr} | ConvertTo-Json
