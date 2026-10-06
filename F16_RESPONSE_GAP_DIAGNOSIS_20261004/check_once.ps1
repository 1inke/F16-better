$ErrorActionPreference='Stop'
$experimentRoot=$PSScriptRoot
$state=Get-Content -LiteralPath (Join-Path $experimentRoot 'independent_worker_state.json') -Raw | ConvertFrom-Json
$process=Get-CimInstance Win32_Process -Filter ('ProcessId='+$state.pid)
$taskName='Trajectory-ResponseGap-Diagnosis-20261004'
$task=Get-ScheduledTask -TaskName $taskName
$triggers=([xml](Export-ScheduledTask -TaskName $taskName)).Task.Triggers.ChildNodes.Count
$progressPath=Join-Path $experimentRoot 'progress.json'
$progress=if(Test-Path -LiteralPath $progressPath){Get-Content -LiteralPath $progressPath -Raw | ConvertFrom-Json}else{$null}
$exitPath=Join-Path $experimentRoot 'exit_status.json'
$exitState=if(Test-Path -LiteralPath $exitPath){Get-Content -LiteralPath $exitPath -Raw | ConvertFrom-Json}else{$null}
$errorFile=Get-Item -LiteralPath (Join-Path $experimentRoot 'stderr.log') -ErrorAction SilentlyContinue
[ordered]@{checked_at=(Get-Date).ToString('o');worker=$state;actual_process=$process | Select-Object ProcessId,ParentProcessId,Name,CommandLine;task_state=[string]$task.State;triggers=$triggers;exit_state=$exitState;progress=$progress;stderr_bytes=$errorFile.Length} | ConvertTo-Json -Depth 8
Get-Content -LiteralPath (Join-Path $experimentRoot 'stdout.log') -Tail 4
