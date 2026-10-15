$ErrorActionPreference='Stop'
$root=$PSScriptRoot
$name='Trajectory-PortfolioCalibrationReplay-20261007'
$t=Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
$procs=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {$_.CommandLine -and $_.CommandLine.Contains($root) -and $_.CommandLine.Contains('independent_worker.py')} | Select-Object ProcessId,ParentProcessId,CreationDate,CommandLine)
$result=[ordered]@{checked_at=(Get-Date).ToString('o');task_name=$name;task_state=$null;triggers=$null;processes=$procs;worker=$null;exit=$null;progress=$null;stderr_bytes=0}
if($t){$result.task_state=[string]$t.State;$xml=[xml](Export-ScheduledTask -TaskName $name);$result.triggers=@($xml.SelectNodes('/*[local-name()="Task"]/*[local-name()="Triggers"]/*')).Count}
foreach($pair in @(@('worker','worker_state.json'),@('exit','exit_status.json'),@('progress','progress.json'))){$p=Join-Path $root $pair[1];if(Test-Path $p){$result[$pair[0]]=Get-Content $p -Raw -Encoding UTF8 | ConvertFrom-Json}}
$p=Join-Path $root 'logs/worker_stderr.log';if(Test-Path $p){$result.stderr_bytes=(Get-Item $p).Length}
$result | ConvertTo-Json -Depth 12
