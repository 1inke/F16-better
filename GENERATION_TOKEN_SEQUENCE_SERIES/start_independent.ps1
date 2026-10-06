$ErrorActionPreference='Stop'
$experimentRoot=$PSScriptRoot
$taskName='Trajectory-TokenSequence-Cycle4'
$pythonExe='C:\coding_toobox\envs\astra2\python.exe'
$worker=Join-Path $experimentRoot 'independent_worker.py'
$activeRuns=Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' -and $_.CommandLine -match 'GENERATION_.*_SERIES' }
if ($activeRuns) { throw 'Generation experiment active' }
$audit=Get-Content -LiteralPath (Join-Path $experimentRoot 'preflight.json') -Raw | ConvertFrom-Json
foreach ($property in $audit.source_hashes.PSObject.Properties) { if ((Get-FileHash -LiteralPath $property.Name -Algorithm SHA256).Hash.ToLower() -ne $property.Value) { throw ('Source/checkpoint lock mismatch '+$property.Name) } }
if ((Get-Content -LiteralPath (Join-Path $experimentRoot 'resume_check.json') -Raw | ConvertFrom-Json).status -ne 'PASS') { throw 'Resume check missing' }
$exitFile=Join-Path $experimentRoot 'results\exit_status.json'
if ((Test-Path -LiteralPath $exitFile) -and ((Get-Content -LiteralPath $exitFile -Raw | ConvertFrom-Json).status -eq 'SUCCESS')) { throw 'Already complete' }
$existing=Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing) {
    if ($existing.State -eq 'Running') { throw 'Task active' }
    if ($existing.Actions.Execute -ne $pythonExe -or $existing.Actions.Arguments -ne ('-u "'+$worker+'"')) { throw 'Task target mismatch' }
} else {
    $action=New-ScheduledTaskAction -Execute $pythonExe -Argument ('-u "'+$worker+'"') -WorkingDirectory $experimentRoot
    $settings=New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew
    $principal=New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
    Register-ScheduledTask -TaskName $taskName -Action $action -Settings $settings -Principal $principal -Description 'User-authorized candidate geometry generation experiment; on-demand only; no recurring trigger.' | Out-Null
}
Start-ScheduledTask -TaskName $taskName
[ordered]@{task_name=$taskName;started_at=(Get-Date).ToString('o')} | ConvertTo-Json
