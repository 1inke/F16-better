$ErrorActionPreference='Stop'
$root=$PSScriptRoot;$name='Trajectory-K3SingleReplacement-20261007'
$pre=Get-Content (Join-Path $root 'PREFLIGHT.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if($pre.status -ne 'PASS'){throw 'Actual preflight required'}
foreach($item in $pre.files.PSObject.Properties){if((Get-FileHash -LiteralPath $item.Name -Algorithm SHA256).Hash.ToLower() -ne $item.Value){throw "Source mismatch: $($item.Name)"}}
$active=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {$_.CommandLine -and $_.CommandLine.Contains($root) -and $_.CommandLine.Contains('independent_worker.py')})
if($active.Count){throw 'Same worker active'}
if(Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue){throw 'Same task exists'}
$action=New-ScheduledTaskAction -Execute 'C:\coding_toobox\envs\astra2\python.exe' -Argument ('-B -u "'+(Join-Path $root 'independent_worker.py')+'"') -WorkingDirectory $root
$settings=New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew
$principal=New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $name -Action $action -Settings $settings -Principal $principal | Out-Null
Start-ScheduledTask -TaskName $name
[pscustomobject]@{task=$name;requested_at=(Get-Date).ToString('o');triggers=0} | ConvertTo-Json
