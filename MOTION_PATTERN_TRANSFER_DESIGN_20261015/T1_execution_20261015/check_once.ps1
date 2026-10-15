$ErrorActionPreference='Stop'
$root=$PSScriptRoot;$name='Trajectory-MotionPatternT1Responsibility-20261015'
$task=Get-ScheduledTask -TaskName $name;[xml]$xml=Export-ScheduledTask -TaskName $name
$processes=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'"|Where-Object {$_.CommandLine -and $_.CommandLine.Contains($root) -and $_.CommandLine.Contains('independent_worker.py')}|ForEach-Object {
 $parent=Get-CimInstance Win32_Process -Filter ("ProcessId="+$_.ParentProcessId)
 [pscustomobject]@{ProcessId=$_.ProcessId;ParentProcessId=$_.ParentProcessId;ParentName=$parent.Name;CreationDate=$_.CreationDate;CommandLine=$_.CommandLine}
})
$r=[ordered]@{checked_at=(Get-Date).ToString('o');task=$name;task_state=[string]$task.State;triggers=$xml.SelectNodes("//*[local-name()='Triggers']/*").Count;processes=$processes}
foreach($file in @('worker_state','exit_status','progress','RESULT')){
 $p=Join-Path $root ($file+'.json')
 if(Test-Path $p){$z=Get-Content -LiteralPath $p -Raw|ConvertFrom-Json
  if($file -eq 'RESULT'){$r[$file]=@{status=$z.status;production_trainings=$z.production_trainings;saved_jobs=$z.saved_jobs;epochs=$z.epochs;decision=$z.decision}}else{$r[$file]=$z}
 }else{$r[$file]=$null}
}
$histories=@()
foreach($arm in @('dynamic_winner','fixed_motion_responsibility')){foreach($seed in @(42,43,44)){
 $d=Join-Path $root ('runs/'+$arm+'/seed_'+$seed);$p=Join-Path $d 'history.json';$n=0
 if(Test-Path $p){$n=@(Get-Content -LiteralPath $p -Raw|ConvertFrom-Json).Count}
 $histories+=@{arm=$arm;seed=$seed;completed_epochs=$n;done=(Test-Path (Join-Path $d 'done.json'))}
}}
$r.histories=$histories
$p=Join-Path $root 'worker_stderr.log';$r.stderr_bytes=if(Test-Path $p){(Get-Item $p).Length}else{0}
$r|ConvertTo-Json -Depth 12
