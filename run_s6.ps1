# Local Fine Reranking (S6) Execution Helper
param(
    [string]$Stage = "all"
)

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$PythonExe = "C:\coding_toobox\envs\astra2\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

Write-Host "[S6 RUNNER] Working Directory: $ProjectRoot" -ForegroundColor Cyan
Write-Host "[S6 RUNNER] Python: $PythonExe" -ForegroundColor Cyan
Write-Host "[S6 RUNNER] Stage: $Stage" -ForegroundColor Cyan

& $PythonExe "SELECTOR_LOCAL_RERANK_SERIES/run_local_rerank_series.py" --stage $Stage
