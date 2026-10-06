$ErrorActionPreference='Stop'
$experimentRoot=$PSScriptRoot
$pythonExe='C:\coding_toobox\envs\astra2\python.exe'
$entry=Join-Path $experimentRoot 'run_basis.py'
$stdout=Join-Path $experimentRoot 'stdout.log'
$stderr=Join-Path $experimentRoot 'stderr.log'
if ((Test-Path -LiteralPath $stdout) -or (Test-Path -LiteralPath $stderr)) { throw 'Existing launch logs' }
$activeRuns=Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' -and $_.CommandLine -match 'SERIES' }
if ($activeRuns) { throw 'Another experiment is active' }
if (-not (Test-Path -LiteralPath (Join-Path $experimentRoot 'resume_check.json'))) { throw 'Resume smoke required' }
& $pythonExe $entry --action preflight --device cuda:0
if ($LASTEXITCODE -ne 0) { throw 'Preflight failed' }
$trainingProcess=Start-Process -FilePath $pythonExe -ArgumentList ('"'+$entry+'" --action run --device cuda:0') -WorkingDirectory $experimentRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
[ordered]@{pid=$trainingProcess.Id;launched_at=(Get-Date).ToString('o');stdout=$stdout;stderr=$stderr} | ConvertTo-Json
