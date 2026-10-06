param(
    [switch]$DryRun,
    [switch]$Smoke
)

$ErrorActionPreference = "Stop"
$Python = "C:\coding_toobox\envs\astra2\python.exe"
$Root = "C:\Users\linke\Documents\New project\CONDITIONAL_ENDPOINT_QUANTIZATION_SERIES"
$Runner = Join-Path $Root "run_ceq1.py"
$Tests = Join-Path $Root "test_ceq1.py"

Write-Host "[1/3] Lock baseline, split identity, and no-test contract"
& $Python $Runner --preflight-only
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[2/3] Run metric, geometry, and probability-mass unit tests"
& $Python $Tests
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($DryRun) {
    Write-Host "[DRY-RUN PASS] Formal CEQ1 evaluation was not started."
    exit 0
}

Write-Host "[3/3] Estimate the train-only conditional future distribution and evaluate it"
if ($Smoke) {
    & $Python $Runner --device cuda:0 --smoke --bootstrap-repetitions 200
} else {
    & $Python $Runner --device cuda:0 --bootstrap-repetitions 10000
}
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($Smoke) {
    Write-Host "[DONE] Smoke report: CONDITIONAL_ENDPOINT_QUANTIZATION_SERIES\smoke_output\CEQ1_REPORT.md"
} else {
    Write-Host "[DONE] Formal report: CONDITIONAL_ENDPOINT_QUANTIZATION_SERIES\results\ceq1_eth_seed1\CEQ1_REPORT.md"
}
