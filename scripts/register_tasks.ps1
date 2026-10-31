<#
.SYNOPSIS
  Register (or re-register) the Windows scheduled tasks that keep Sharadar data current.

  pitlab-sharadar-daily   every day  at -DailyAt   : incremental top-off
  pitlab-sharadar-weekly  Sundays    at -WeeklyAt  : full bulk refresh (repairs deletions and
                                                     anything older than the delta cap)

  Tasks run as the current user only while logged on (no stored password), and catch up after
  a missed start (PC was off). Sharadar publishes after the US close; 09:00 UAE time is well
  after that. Remove with -Unregister.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\register_tasks.ps1
  powershell -ExecutionPolicy Bypass -File scripts\register_tasks.ps1 -DailyAt 07:30
  powershell -ExecutionPolicy Bypass -File scripts\register_tasks.ps1 -Unregister
#>
param(
    [string]$DailyAt = "09:00",
    [string]$WeeklyAt = "10:00",
    [switch]$Unregister
)
$ErrorActionPreference = "Stop"
$script = Join-Path $PSScriptRoot "update_data.ps1"
$names = @("pitlab-sharadar-daily", "pitlab-sharadar-weekly")

foreach ($n in $names) {
    if (Get-ScheduledTask -TaskName $n -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $n -Confirm:$false
    }
}
if ($Unregister) { Write-Host "removed: $($names -join ', ')"; return }

$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 3) `
    -MultipleInstances IgnoreNew
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive

function New-UpdateAction([string]$extra) {
    New-ScheduledTaskAction -Execute "powershell.exe" `
        -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$script`" $extra"
}

Register-ScheduledTask -TaskName $names[0] -Description "pit-two-clocks: daily Sharadar incremental update" `
    -Action (New-UpdateAction "") -Trigger (New-ScheduledTaskTrigger -Daily -At $DailyAt) `
    -Settings $settings -Principal $principal | Out-Null
Register-ScheduledTask -TaskName $names[1] -Description "pit-two-clocks: weekly Sharadar full bulk refresh" `
    -Action (New-UpdateAction "-Full") -Trigger (New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At $WeeklyAt) `
    -Settings $settings -Principal $principal | Out-Null

Get-ScheduledTask -TaskName $names | Select-Object TaskName, State,
    @{n = "Next run"; e = { (Get-ScheduledTaskInfo $_.TaskName).NextRunTime } } | Format-Table -AutoSize
