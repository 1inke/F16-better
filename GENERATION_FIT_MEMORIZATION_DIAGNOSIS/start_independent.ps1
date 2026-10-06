$ErrorActionPreference='Stop'
$taskName='Trajectory-FitMemorization-Diagnosis'
$pythonExe='C:\coding_toobox\envs\astra2\python.exe'
$worker=Join-Path $PSScriptRoot 'independent_worker.py'
if (Test-Path (Join-Path $PSScriptRoot 'RESULT.json')) { throw 'Already complete' }
$audit=Get-Content (Join-Path $PSScriptRoot 'preflight.json') -Raw | ConvertFrom-Json
if ($audit.status -ne 'PASS') { throw 'Preflight required' }
foreach ($property in $audit.source_hashes.PSObject.Properties) { if ((Get-FileHash -LiteralPath $property.Name -Algorithm SHA256).Hash.ToLower() -ne $property.Value) { throw 'Source mismatch' } }
$existing=Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing) { if ($existing.State -eq 'Running') { throw 'Already running' }; if ($existing.Actions.Arguments -ne ('-B -u "'+$worker+'"')) { throw 'Target mismatch' } }
else {
    $action=New-ScheduledTaskAction -Execute $pythonExe -Argument ('-B -u "'+$worker+'"') -WorkingDirectory $PSScriptRoot
    $settings=New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew
    $principal=New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
    Register-ScheduledTask -TaskName $taskName -Action $action -Settings $settings -Principal $principal | Out-Null
}
Start-ScheduledTask -TaskName $taskName
