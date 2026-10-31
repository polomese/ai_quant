<#
.SYNOPSIS
  Run one Sharadar update (daily incremental, or -Full bulk refresh) and log it.
  Called by the scheduled tasks that scripts/register_tasks.ps1 creates; safe to run by hand.

  Logs: $env:PITLAB_DATA_ROOT\pit-two-clocks\logs\update_<stamp>.log (kept 60 days)
  Status of the last run: ...\logs\last_update.json
#>
param([switch]$Full)
$ErrorActionPreference = "Stop"
$proj = Split-Path -Parent $PSScriptRoot
$name = Split-Path -Leaf $proj

# scheduled tasks inherit the user's environment; fall back to the documented defaults
foreach ($v in @("UV_PROJECT_ENVIRONMENT", "PITLAB_DATA_ROOT", "MYPY_CACHE_DIR")) {
    if (-not [Environment]::GetEnvironmentVariable($v)) {
        $u = [Environment]::GetEnvironmentVariable($v, "User")
        if ($u) { Set-Item "env:$v" $u }
    }
}
if (-not $env:PITLAB_DATA_ROOT) { throw "PITLAB_DATA_ROOT is not set - run scripts\bootstrap.ps1 first" }
if (-not $env:UV_PROJECT_ENVIRONMENT) { throw "UV_PROJECT_ENVIRONMENT is not set - run scripts\bootstrap.ps1 first" }

$logs = Join-Path $env:PITLAB_DATA_ROOT "$name\logs"
New-Item -ItemType Directory -Force $logs | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$log = Join-Path $logs ("update_{0}{1}.log" -f $stamp, $(if ($Full) { "_full" } else { "" }))
$env:PYTHONIOENCODING = "utf-8"

$pitlab = Join-Path $env:UV_PROJECT_ENVIRONMENT "Scripts\pitlab.exe"
$cliArgs = "data update" + $(if ($Full) { " --full" } else { "" })
Push-Location $proj   # .env (API key) and config/ are resolved from the project folder
try {
    # redirect inside cmd: Windows PowerShell 5.1 turns a native program's stderr (our log
    # output) into terminating errors under ErrorActionPreference=Stop
    cmd /c "`"$pitlab`" $cliArgs > `"$log`" 2>&1"
    $code = $LASTEXITCODE
}
finally { Pop-Location }

Get-ChildItem $logs -Filter "update_*.log" | Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-60) } |
    Remove-Item -Force -ErrorAction SilentlyContinue
exit $code
