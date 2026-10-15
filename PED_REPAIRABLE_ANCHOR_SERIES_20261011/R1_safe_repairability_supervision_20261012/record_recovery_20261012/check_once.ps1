$ErrorActionPreference='Stop'
$root=$PSScriptRoot;$name='Trajectory-SafeRepairabilitySupervision-R1-RecordRecovery-20261012'
$task=Get-ScheduledTask -TaskName $name
[xml]$xml=Export-ScheduledTask -TaskName $name
$processes=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {$_.CommandLine -and $_.CommandLine.Contains($root) -and $_.CommandLine.Contains('independent_worker.py')} | Select-Object ProcessId,ParentProcessId,CreationDate,CommandLine)
$r=[ordered]@{checked_at=(Get-Date).ToString('o');task=$name;task_state=[string]$task.State;triggers=$xml.SelectNodes("//*[local-name()='Triggers']/*").Count;processes=$processes}
foreach($file in @('worker_state','exit_status','progress','RESULT')){$p=Join-Path $root ($file+'.json');if(Test-Path $p){$z=Get-Content -LiteralPath $p -Raw|ConvertFrom-Json;if($file -eq 'RESULT'){$r[$file]=$z.status}else{$r[$file]=$z}}else{$r[$file]=$null}}
$p=Join-Path $root 'worker_stderr.log';$r.stderr_bytes=if(Test-Path $p){(Get-Item $p).Length}else{0}
$r|ConvertTo-Json -Depth 9
