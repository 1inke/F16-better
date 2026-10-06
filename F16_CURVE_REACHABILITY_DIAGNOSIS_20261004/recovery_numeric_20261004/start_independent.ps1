$ErrorActionPreference='Stop'
$experimentRoot=$PSScriptRoot
$taskName='Trajectory-CurveReachability-NumericRecovery-20261004'
$pythonExe='C:\coding_toobox\envs\astra2\python.exe'
$worker=Join-Path $experimentRoot 'independent_worker.py'
$active=Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' -and $_.CommandLine -match '(GENERATION_.*_SERIES|F16_.*DIAGNOSIS|F16_CUE_RESPONSE_CONCORDANCE)' }
if ($active) { throw 'Another experiment is active; no duplicate worker' }
$pre=Get-Content -LiteralPath (Join-Path $experimentRoot 'preflight.json') -Raw | ConvertFrom-Json
if ($pre.status -ne 'PASS' -or $pre.numeric_repair_preflight -ne 'PASS') { throw 'Recovery preflight not PASS' }
$numeric=Join-Path $experimentRoot 'NUMERIC_REPAIR_PREFLIGHT.json'
if ((Get-FileHash -LiteralPath $numeric -Algorithm SHA256).Hash.ToLower() -ne $pre.numeric_repair_preflight_SHA) { throw 'Numeric proof SHA mismatch' }
if ((Get-Content -LiteralPath (Join-Path $experimentRoot 'MEMORY_PREFLIGHT.json') -Raw | ConvertFrom-Json).status -ne 'PASS_FULL_INPUT_CHUNK_MEMORY_AND_ARITHMETIC') { throw 'Resource preflight not PASS' }
foreach($prop in $pre.source_hashes.PSObject.Properties) { if ((Get-FileHash -LiteralPath $prop.Name -Algorithm SHA256).Hash.ToLower() -ne $prop.Value) { throw ('Source changed '+$prop.Name) } }
if ((Get-Content -LiteralPath (Join-Path $experimentRoot 'resume_check.json') -Raw | ConvertFrom-Json).status -ne 'PASS') { throw 'Analysis resume not PASS' }
$exitPath=Join-Path $experimentRoot 'exit_status.json'
if ((Test-Path -LiteralPath $exitPath) -and ((Get-Content -LiteralPath $exitPath -Raw | ConvertFrom-Json).status -eq 'SUCCESS')) { throw 'Already completed' }
$existing=Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if($existing) {
    if($existing.State -eq 'Running'){throw 'Task active'}
    if($existing.Actions.Execute -ne $pythonExe -or $existing.Actions.Arguments -ne ('-B -u "'+$worker+'"')) { throw 'Task mismatch' }
    if($existing.Triggers){throw 'Unexpected triggers'}
} else {
    $action=New-ScheduledTaskAction -Execute $pythonExe -Argument ('-B -u "'+$worker+'"') -WorkingDirectory $experimentRoot
    $settings=New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew
    $principal=New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
    Register-ScheduledTask -TaskName $taskName -Action $action -Settings $settings -Principal $principal -Description 'Human approved numerical reliability repair and fixed-model FIT/CAL curve diagnosis recovery, no training, on demand.' | Out-Null
}
Start-ScheduledTask -TaskName $taskName
[ordered]@{task_name=$taskName;requested_at=(Get-Date).ToString('o')} | ConvertTo-Json
