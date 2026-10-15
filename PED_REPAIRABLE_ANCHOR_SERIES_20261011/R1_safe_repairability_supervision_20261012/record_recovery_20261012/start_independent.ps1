$ErrorActionPreference='Stop'
$root=$PSScriptRoot;$name='Trajectory-SafeRepairabilitySupervision-R1-RecordRecovery-20261012'
$pre=Get-Content -LiteralPath (Join-Path $root 'PREFLIGHT.json') -Raw | ConvertFrom-Json
if($pre.status -ne 'PASS' -or $pre.planning_minutes -gt 30){throw 'Preflight/planning gate'}
$manifest=Get-Content -LiteralPath (Join-Path $root 'source_manifest.json') -Raw | ConvertFrom-Json
foreach($item in $manifest.files.PSObject.Properties){if((Get-FileHash -LiteralPath $item.Name -Algorithm SHA256).Hash.ToLower() -ne $item.Value){throw ('Source mismatch '+$item.Name)}}
$active=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {$_.CommandLine -and $_.CommandLine.Contains($root) -and $_.CommandLine.Contains('independent_worker.py')})
if($active.Count){throw 'Same worker active'}
if((Test-Path (Join-Path $root 'worker_state.json')) -or (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue)){throw 'Worker/task exists'}
$action=New-ScheduledTaskAction -Execute 'C:\coding_toobox\envs\astra2\python.exe' -Argument ('-B -u "'+(Join-Path $root 'independent_worker.py')+'"') -WorkingDirectory $root
$settings=New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew
$principal=New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $name -Action $action -Settings $settings -Principal $principal | Out-Null
Start-ScheduledTask -TaskName $name
[pscustomobject]@{task=$name;requested_at=(Get-Date).ToString('o');triggers=0} | ConvertTo-Json
