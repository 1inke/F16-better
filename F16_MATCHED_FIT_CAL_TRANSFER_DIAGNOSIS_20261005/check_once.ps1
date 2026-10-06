$ErrorActionPreference='Stop'
$root=$PSScriptRoot
$state=Get-Content -LiteralPath (Join-Path $root 'worker_state.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$process=Get-CimInstance Win32_Process -Filter ('ProcessId='+$state.pid)
if($process -and ($process.Name -notmatch '^python' -or $process.CommandLine -notlike ('*'+(Join-Path $root 'independent_worker.py')+'*'))){throw 'PID identity mismatch'}
$task=Get-ScheduledTask -TaskName 'Trajectory-MatchedFitCalTransfer-20261005'
$triggers=([xml](Export-ScheduledTask -TaskName 'Trajectory-MatchedFitCalTransfer-20261005')).Task.Triggers.ChildNodes.Count
$progressPath=Join-Path $root 'progress.json'
$progress=if(Test-Path -LiteralPath $progressPath){Get-Content -LiteralPath $progressPath -Raw -Encoding UTF8|ConvertFrom-Json}else{$null}
$exitPath=Join-Path $root 'exit_status.json'
$exitState=if(Test-Path -LiteralPath $exitPath){Get-Content -LiteralPath $exitPath -Raw -Encoding UTF8|ConvertFrom-Json}else{$null}
$errorFile=Get-Item -LiteralPath (Join-Path $root 'worker_stderr.log') -ErrorAction SilentlyContinue
[ordered]@{checked_at=(Get-Date).ToString('o');worker=$state;actual_process=$process|Select-Object ProcessId,ParentProcessId,Name,CommandLine;task_state=[string]$task.State;triggers=$triggers;progress=$progress;exit_state=$exitState;stderr_bytes=$errorFile.Length;done_jobs=@(Get-ChildItem -LiteralPath (Join-Path $root 'jobs') -Filter '*.done.json' -ErrorAction SilentlyContinue).Count}|ConvertTo-Json -Depth 8

