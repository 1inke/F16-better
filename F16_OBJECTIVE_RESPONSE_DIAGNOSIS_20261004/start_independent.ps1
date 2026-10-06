$ErrorActionPreference='Stop'
$experimentRoot=$PSScriptRoot
$taskName='Trajectory-ObjectiveResponse-Diagnosis-20261004'
$pythonExe='C:\coding_toobox\envs\astra2\python.exe'
$worker=Join-Path $experimentRoot 'independent_worker.py'
$active=Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' -and $_.CommandLine -match '(GENERATION_.*_SERIES|F16_.*DIAGNOSIS|F16_CUE_RESPONSE_CONCORDANCE)' }
if ($active) { throw 'Another experiment is active; no duplicate worker' }
$pre=Get-Content -LiteralPath (Join-Path $experimentRoot 'preflight.json') -Raw | ConvertFrom-Json
if ($pre.status -ne 'PASS') { throw 'Preflight not PASS' }
foreach($prop in $pre.source_hashes.PSObject.Properties) { if ((Get-FileHash -LiteralPath $prop.Name -Algorithm SHA256).Hash.ToLower() -ne $prop.Value) { throw ('Source changed '+$prop.Name) } }
foreach($name in @('MEMORY_PREFLIGHT.json','resume_check.json')) { if ((Get-Content -LiteralPath (Join-Path $experimentRoot $name) -Raw | ConvertFrom-Json).status -ne 'PASS') {throw ('Check not PASS '+$name)} }
$exitPath=Join-Path $experimentRoot 'exit_status.json'
if ((Test-Path -LiteralPath $exitPath) -and ((Get-Content -LiteralPath $exitPath -Raw | ConvertFrom-Json).status -eq 'SUCCESS')) { throw 'Already completed' }
$existing=Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if($existing) {
    if ($existing.State -eq 'Running') {throw 'Task active'}
    if ($existing.Actions.Execute -ne $pythonExe -or $existing.Actions.Arguments -ne ('-B -u "'+$worker+'"') -or $existing.Triggers) { throw 'Task mismatch' }
} else {
    $action=New-ScheduledTaskAction -Execute $pythonExe -Argument ('-B -u "'+$worker+'"') -WorkingDirectory $experimentRoot
    $settings=New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew
    $principal=New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
    Register-ScheduledTask -TaskName $taskName -Action $action -Settings $settings -Principal $principal -Description 'Human approved saved-path original objective diagnosis FIT/CAL only; no model forward or optimizer; on demand.' | Out-Null
}
Start-ScheduledTask -TaskName $taskName
[ordered]@{task_name=$taskName;requested_at=(Get-Date).ToString('o')} | ConvertTo-Json
