$ErrorActionPreference='Stop'
$root=$PSScriptRoot;$name='Trajectory-PortfolioCalibrationContractRecovery-20261007'
$t=Get-ScheduledTask -TaskName $name
$xml=[xml](Export-ScheduledTask -TaskName $name)
$p=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {$_.CommandLine -and $_.CommandLine.Contains($root) -and $_.CommandLine.Contains('independent_worker.py')} | Select-Object ProcessId,ParentProcessId,CreationDate,CommandLine)
function ReadNew($name){$f=Join-Path $root $name;if(Test-Path -LiteralPath $f){Get-Content -LiteralPath $f -Raw -Encoding UTF8 | ConvertFrom-Json}else{$null}}
[pscustomobject]@{checked_at=(Get-Date).ToString('o');task_name=$name;task_state=[string]$t.State;triggers=$xml.SelectNodes("/*[local-name()='Task']/*[local-name()='Triggers']/*").Count;processes=$p;worker=(ReadNew 'worker_state.json');exit=(ReadNew 'exit_status.json');progress=(ReadNew 'progress.json');stderr_bytes=(Get-Item -LiteralPath (Join-Path $root 'worker_stderr.log')).Length} | ConvertTo-Json -Depth 7
