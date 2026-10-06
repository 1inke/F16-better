param([switch]$Probe)
$ErrorActionPreference='Stop'
$experimentRoot=$PSScriptRoot
$taskName='Trajectory-NativeDecoder-Resume-Cycle2'
if ($Probe) { $taskName+='-Probe' }
$pythonExe='C:\coding_toobox\envs\astra2\python.exe'
$worker=Join-Path $experimentRoot 'independent_worker.py'
if (-not $Probe) {
    $activeRuns=Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' -and $_.CommandLine -match 'GENERATION_NATIVE_DECODER' }
    if ($activeRuns) { throw 'Native experiment/probe already active; do not duplicate' }
    $exitFile=Join-Path $experimentRoot 'results\exit_status.json'
    if (Test-Path -LiteralPath $exitFile) {
        if ((Get-Content -LiteralPath $exitFile -Raw | ConvertFrom-Json).status -eq 'SUCCESS') { throw 'Experiment already completed' }
    }
    $audit=Get-Content -LiteralPath (Join-Path $experimentRoot 'preflight.json') -Raw | ConvertFrom-Json
    foreach ($property in $audit.source_hashes.PSObject.Properties) {
        if ((Get-FileHash -LiteralPath $property.Name -Algorithm SHA256).Hash.ToLower() -ne $property.Value) { throw ('Source lock mismatch: '+$property.Name) }
    }
    if ((Get-Content -LiteralPath (Join-Path $experimentRoot 'resume_check.json') -Raw | ConvertFrom-Json).status -ne 'PASS') { throw 'Resume verification missing' }
    Get-ChildItem -LiteralPath (Join-Path $experimentRoot 'results\models') -Recurse -Filter complete.json | ForEach-Object {
        $manifest=Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json
        foreach ($property in $manifest.model_hashes.PSObject.Properties) {
            if ((Get-FileHash -LiteralPath $property.Name -Algorithm SHA256).Hash.ToLower() -ne $property.Value) { throw ('Completed checkpoint hash mismatch: '+$property.Name) }
        }
    }
}
$existing=Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing) {
    if ($existing.State -eq 'Running') { throw 'Task already running' }
    if ($existing.Actions.Execute -ne $pythonExe -or $existing.Actions.Arguments -notlike '*independent_worker.py*') { throw 'Task name occupied by another action' }
} else {
    $arguments='-u "'+$worker+'"'
    if ($Probe) { $arguments+=' --probe' }
    $action=New-ScheduledTaskAction -Execute $pythonExe -Argument $arguments -WorkingDirectory $experimentRoot
    $settings=New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew
    $userIdentity=[System.Security.Principal.WindowsIdentity]::GetCurrent().Name
    $principal=New-ScheduledTaskPrincipal -UserId $userIdentity -LogonType Interactive -RunLevel Limited
    Register-ScheduledTask -TaskName $taskName -Action $action -Settings $settings -Principal $principal -Description 'User-authorized trajectory experiment; on-demand only, no recurring trigger; resume existing checkpoints.' | Out-Null
}
Start-ScheduledTask -TaskName $taskName
[ordered]@{task_name=$taskName;requested_at=(Get-Date).ToString('o');launcher='Windows Task Scheduler';recurring=$false} | ConvertTo-Json
