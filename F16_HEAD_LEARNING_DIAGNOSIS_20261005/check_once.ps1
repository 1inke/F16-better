$ErrorActionPreference='Stop'
$root=$PSScriptRoot
$state=Get-Content -LiteralPath (Join-Path $root 'independent_worker_state.json') -Raw | ConvertFrom-Json
$process=Get-CimInstance Win32_Process -Filter ('ProcessId='+$state.pid)
$taskName='Trajectory-HeadLearning-Diagnosis-20261005'
$task=Get-ScheduledTask -TaskName $taskName
$triggers=([xml](Export-ScheduledTask -TaskName $taskName)).Task.Triggers.ChildNodes.Count
$exitPath=Join-Path $root 'exit_status.json'
$exitState=if(Test-Path -LiteralPath $exitPath){Get-Content -LiteralPath $exitPath -Raw | ConvertFrom-Json}else{$null}
$progressPath=Join-Path $root 'progress.json'
$progress=if(Test-Path -LiteralPath $progressPath){Get-Content -LiteralPath $progressPath -Raw | ConvertFrom-Json}else{$null}
$runs=@()
foreach($arm in @('zero','mean_std')) {foreach($seed in @(7,8,9)) {
    $dir=Join-Path $root ('runs\'+$arm+'\seed'+$seed)
    $history=Join-Path $dir 'history.json'
    $last=if(Test-Path -LiteralPath $history){@((Get-Content -LiteralPath $history -Raw | ConvertFrom-Json))[-1].epoch}else{$null}
    $runs+=@{arm=$arm;seed=$seed;last_recorded_epoch=$last;complete=(Test-Path -LiteralPath (Join-Path $dir 'done.json'))}
}}
$errorFile=Get-Item -LiteralPath (Join-Path $root 'stderr.log') -ErrorAction SilentlyContinue
[ordered]@{checked_at=(Get-Date).ToString('o');worker=$state;actual_process=$process | Select-Object ProcessId,ParentProcessId,Name,CommandLine;task_state=[string]$task.State;triggers=$triggers;exit_state=$exitState;progress=$progress;runs=$runs;stderr_bytes=$errorFile.Length} | ConvertTo-Json -Depth 8
