$ErrorActionPreference='Stop'
$root=$PSScriptRoot
$state=Get-Content -LiteralPath (Join-Path $root 'independent_worker_state.json') -Raw | ConvertFrom-Json
$process=Get-CimInstance Win32_Process -Filter ('ProcessId='+$state.pid)
$task=Get-ScheduledTask -TaskName 'Trajectory-TrainedAccelProjection-EvaluationRecovery-20261006'
$triggers=([xml](Export-ScheduledTask -TaskName 'Trajectory-TrainedAccelProjection-EvaluationRecovery-20261006')).Task.Triggers.ChildNodes.Count
$exitPath=Join-Path $root 'results\exit_status.json'
$exitState=if(Test-Path -LiteralPath $exitPath){Get-Content -LiteralPath $exitPath -Raw | ConvertFrom-Json}else{$null}
$records=@()
foreach($arm in @('zero','mean_std')) {
    foreach($seed in @(1,2,3)) {
        $dir=Join-Path (Split-Path -Parent $root) ('results\models\'+$arm+'\seed_'+$seed+'\balanced_protected')
        $history=Join-Path $dir 'history.csv'
        $epoch=if(Test-Path -LiteralPath $history){@((Import-Csv -LiteralPath $history))[-1].epoch}else{0}
        $complete=Test-Path -LiteralPath (Join-Path $dir 'complete.json')
        $records+=@{arm=$arm;seed=$seed;last_completed_epoch=$epoch;complete=$complete}
    }
}
$errorFile=Get-Item -LiteralPath (Join-Path $root 'stderr.log') -ErrorAction SilentlyContinue
$progressPath=Join-Path $root 'progress.json'
$progress=if(Test-Path -LiteralPath $progressPath){Get-Content -LiteralPath $progressPath -Raw | ConvertFrom-Json}else{$null}
[ordered]@{checked_at=(Get-Date).ToString('o');worker=$state;progress=$progress;actual_process=$process | Select-Object ProcessId,ParentProcessId,Name,CommandLine;task_state=[string]$task.State;triggers=$triggers;exit_state=$exitState;trainings=$records;stderr_bytes=$errorFile.Length} | ConvertTo-Json -Depth 8
Get-Content -LiteralPath (Join-Path $root 'stdout.log') -Tail 12
