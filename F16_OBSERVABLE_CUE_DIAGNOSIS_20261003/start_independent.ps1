$ErrorActionPreference='Stop'
$experimentRoot=$PSScriptRoot
$taskName='Trajectory-ObservableCues-Diagnosis-20261003'
$pythonExe='C:\coding_toobox\envs\astra2\python.exe'
$worker=Join-Path $experimentRoot 'independent_worker.py'
$preflight=Get-Content -LiteralPath (Join-Path $experimentRoot 'preflight.json') -Raw | ConvertFrom-Json
if ($preflight.status -ne 'PASS') { throw 'Required preflight not PASS' }
foreach ($entry in $preflight.source_hashes.PSObject.Properties) {
    $digest=(Get-FileHash -LiteralPath $entry.Name -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($digest -ne $entry.Value) { throw ('Preflight source changed: '+$entry.Name) }
}
$activeRuns=Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' -and $_.CommandLine -match '(GENERATION_.*_SERIES|F16_OBSERVABLE_CUE_DIAGNOSIS_20261003)' }
if ($activeRuns) { throw 'Existing relevant Python worker active; do not duplicate' }
if (Test-Path -LiteralPath (Join-Path $experimentRoot 'RESULT.json')) { throw 'Diagnosis already complete' }
$existing=Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing) {
    if ($existing.State -eq 'Running') { throw 'Task already active' }
    if ($existing.Actions.Execute -ne $pythonExe -or $existing.Actions.Arguments -ne ('-B -u "'+$worker+'"')) { throw 'Task target mismatch' }
} else {
    $action=New-ScheduledTaskAction -Execute $pythonExe -Argument ('-B -u "'+$worker+'"') -WorkingDirectory $experimentRoot
    $settings=New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew
    $principal=New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
    Register-ScheduledTask -TaskName $taskName -Action $action -Settings $settings -Principal $principal -Description 'Human-authorized frozen F16 observable-cue diagnostic probes; new-folder-only; on-demand with no recurring triggers.' | Out-Null
}
Start-ScheduledTask -TaskName $taskName
[ordered]@{task_name=$taskName;started_at=(Get-Date).ToString('o')} | ConvertTo-Json
