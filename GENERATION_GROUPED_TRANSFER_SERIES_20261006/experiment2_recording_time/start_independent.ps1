$ErrorActionPreference='Stop'
$experimentRoot=$PSScriptRoot
$taskName='Trajectory-GroupedRecordingTime-20261006'
$python='C:\coding_toobox\envs\astra2\python.exe'
$preflight=Get-Content -LiteralPath (Join-Path $experimentRoot 'preflight.json') -Encoding UTF8 -Raw | ConvertFrom-Json
if($preflight.status -ne 'PASS' -or $preflight.resume_status -ne 'PASS'){throw 'Actual preflight/resume required'}
foreach($p in $preflight.source_hashes.PSObject.Properties){if((Get-FileHash -LiteralPath $p.Name -Algorithm SHA256).Hash.ToLower() -ne $p.Value){throw "Source mismatch: $($p.Name)"}}
$active=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {$_.CommandLine -and $_.CommandLine.Contains($experimentRoot) -and $_.CommandLine.Contains('independent_worker.py')})
if($active.Count -gt 0){throw 'This exact experiment already has an active worker'}
$existing=Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if($existing){if($existing.State -eq 'Running'){throw 'Same task running'};throw 'Existing task must be explicitly audited; no silent re-registration'}
$action=New-ScheduledTaskAction -Execute $python -Argument ('-B -u "'+(Join-Path $experimentRoot 'independent_worker.py')+'"') -WorkingDirectory $experimentRoot
$settings=New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew
$principal=New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $taskName -Action $action -Settings $settings -Principal $principal | Out-Null
Start-ScheduledTask -TaskName $taskName
[pscustomobject]@{task_name=$taskName;requested_at=(Get-Date).ToString('o');triggers=0} | ConvertTo-Json -Compress
