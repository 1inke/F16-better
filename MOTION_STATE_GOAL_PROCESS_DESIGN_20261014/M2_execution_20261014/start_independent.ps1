$ErrorActionPreference='Stop'
$root=$PSScriptRoot;$name='Trajectory-MotionStateM2JointGoalProcess-20261014'
$pre=Get-Content -LiteralPath (Join-Path $root 'LAUNCH_GATE.json') -Raw|ConvertFrom-Json
$approval=Get-Content -LiteralPath (Join-Path $root 'BUDGET_EXCEPTION.json') -Raw|ConvertFrom-Json
$review=Get-Content -LiteralPath (Join-Path $root 'prelaunch_review/REVIEW.json') -Raw|ConvertFrom-Json
if($pre.status -ne 'PASS' -or $approval.status -ne 'HUMAN_APPROVED_SINGLE_ITEM_BUDGET_EXCEPTION' -or $pre.planning_minutes -ne $approval.approved_planning_minutes -or $pre.planning_minutes -gt $approval.launch_budget_limit_minutes -or $review.status -ne 'PASS'){throw 'M2 explicit single-item budget/source gate'}
$manifest=Get-Content -LiteralPath (Join-Path $root 'source_manifest.json') -Raw|ConvertFrom-Json
if($pre.source_sha -ne $manifest.source_sha -or $review.source_sha -ne $manifest.source_sha){throw 'Launch-source binding mismatch'}
foreach($item in $manifest.files.PSObject.Properties){if((Get-FileHash -LiteralPath $item.Name -Algorithm SHA256).Hash.ToLower() -ne $item.Value){throw ('Source mismatch '+$item.Name)}}
$active=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'"|Where-Object {$_.CommandLine -and $_.CommandLine.Contains($root) -and $_.CommandLine.Contains('independent_worker.py')})
if($active.Count -or (Test-Path (Join-Path $root 'worker_state.json')) -or (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue)){throw 'Existing worker/task; no duplicate'}
$action=New-ScheduledTaskAction -Execute 'C:\coding_toobox\envs\astra2\python.exe' -Argument ('-B -u "'+(Join-Path $root 'independent_worker.py')+'"') -WorkingDirectory $root
$settings=New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew
$principal=New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $name -Action $action -Settings $settings -Principal $principal|Out-Null
Start-ScheduledTask -TaskName $name
[pscustomobject]@{task=$name;requested_at=(Get-Date).ToString('o');triggers=0}|ConvertTo-Json
