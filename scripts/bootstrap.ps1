<#
.SYNOPSIS
  One-shot setup of pit-two-clocks on a new Windows PC.

.DESCRIPTION
  Sets the per-user environment variables that keep the virtualenv and all data OUTSIDE the
  (cloud-synced / git) project folder, installs uv if missing, lets uv fetch Python 3.12
  (.python-version), installs the locked dependencies (uv.lock), and rebuilds the synthetic
  dataset + demo run so the dashboard works immediately.

  Re-running is safe. Real Sharadar data is not copied between PCs: re-download it with
  `uv run pitlab data update --full` (needs SHARADAR_API_KEY in .env), or copy P:\quant_data
  across. -ScheduleUpdates registers the daily/weekly update tasks (scripts\register_tasks.ps1).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\bootstrap.ps1
  powershell -ExecutionPolicy Bypass -File scripts\bootstrap.ps1 -VenvRoot D:\Python_venv -DataRoot D:\quant_data
#>
param(
    [string]$VenvRoot = "P:\Python_venv",
    [string]$DataRoot = "P:\quant_data",
    [switch]$SkipDemo,
    [switch]$ScheduleUpdates   # also register the daily/weekly Sharadar update tasks
)
$ErrorActionPreference = "Stop"
$proj = Split-Path -Parent $PSScriptRoot
$name = Split-Path -Leaf $proj
$venv = Join-Path $VenvRoot $name

foreach ($d in @($VenvRoot, $DataRoot)) { New-Item -ItemType Directory -Force $d | Out-Null }

# persistent (new terminals) + current session
[Environment]::SetEnvironmentVariable("UV_PROJECT_ENVIRONMENT", $venv, "User")
[Environment]::SetEnvironmentVariable("PITLAB_DATA_ROOT", $DataRoot, "User")
# keep the mypy cache (tens of MB of churn) out of the synced folder too
$mypyCache = Join-Path $VenvRoot ".cache\mypy"
[Environment]::SetEnvironmentVariable("MYPY_CACHE_DIR", $mypyCache, "User")
$env:UV_PROJECT_ENVIRONMENT = $venv
$env:PITLAB_DATA_ROOT = $DataRoot
$env:MYPY_CACHE_DIR = $mypyCache
Write-Host "UV_PROJECT_ENVIRONMENT = $venv"
Write-Host "PITLAB_DATA_ROOT       = $DataRoot  (project data under $DataRoot\$name)"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "installing uv ..."
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

Push-Location $proj
try {
    uv sync --all-groups
    if (-not (Test-Path ".env")) {
        Copy-Item ".env.example" ".env"
        Write-Host "created .env from .env.example - add SHARADAR_API_KEY if you have one"
    }
    if (-not $SkipDemo) {
        uv run pitlab data synthetic
        uv run pitlab data validate
        uv run pitlab run --run-id demo
    }
    uv run pytest -q
}
finally { Pop-Location }

if ($ScheduleUpdates) {
    & (Join-Path $PSScriptRoot "register_tasks.ps1")
    Write-Host "first real-data load: uv run pitlab data update --full  (or wait for Sunday's task)"
}

Write-Host "`nDone. Open a NEW terminal (so the env vars apply), then: uv run pitlab dashboard"
