$ErrorActionPreference='Stop'
$experimentRoot=$PSScriptRoot
$taskName='Trajectory-GroupedRecordingTime-20261006'
$task=Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
$processes=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {$_.CommandLine -and $_.CommandLine.Contains($experimentRoot) -and $_.CommandLine.Contains('independent_worker.py')} | Select-Object ProcessId,ParentProcessId,CreationDate,CommandLine)
$result=[ordered]@{checked_at=(Get-Date).ToString('o');task_name=$taskName;task_state=$null;triggers=$null;processes=$processes;worker=$null;exit=$null;progress=$null;stderr_bytes=0;histories=@()}
if($task){$result.task_state=[string]$task.State;$xml=[xml](Export-ScheduledTask -TaskName $taskName);$result.triggers=@($xml.SelectNodes('/*[local-name()="Task"]/*[local-name()="Triggers"]/*')).Count}
foreach($pair in @(@('worker','worker_state.json'),@('exit','results/exit_status.json'),@('progress','progress.json'))){$file=Join-Path $experimentRoot $pair[1];if(Test-Path -LiteralPath $file){$result[$pair[0]]=Get-Content -LiteralPath $file -Encoding UTF8 -Raw | ConvertFrom-Json}}
$stderr=Join-Path $experimentRoot 'logs/worker_stderr.log';if(Test-Path -LiteralPath $stderr){$result.stderr_bytes=(Get-Item -LiteralPath $stderr).Length}
foreach($arm in @('random','grouped')){foreach($seed in @(1,2,3)){$file=Join-Path $experimentRoot "results/models/$arm/seed_$seed/balanced_protected/history.csv";$rows=0;if(Test-Path -LiteralPath $file){$rows=@(Import-Csv -LiteralPath $file).Count};$result.histories+=@([ordered]@{arm=$arm;seed=$seed;epochs=$rows})}}
$result | ConvertTo-Json -Depth 12
