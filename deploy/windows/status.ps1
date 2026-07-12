# WindsurfAPI 状态:RUNNING/STOPPED + PID + 端口 + 面板地址 + 日志尾。
$ErrorActionPreference = 'Continue'
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$LogDir = Join-Path $Root 'logs'
$PidFile = Join-Path $LogDir 'windsurfapi.pid'
$EnvPath = Join-Path $Root '.env'

# 从 .env / 环境变量读 PORT(勿硬编码)。
$port = $env:PORT
if (-not $port -and (Test-Path $EnvPath)) {
  foreach ($line in (Get-Content $EnvPath -Encoding UTF8)) {
    $t = $line.Trim()
    if ($t -match '^PORT\s*=\s*(\d+)') { $port = $Matches[1]; break }
  }
}
if (-not $port) { $port = '3003' }

$nodePid = $null
if (Test-Path $PidFile) { $nodePid = (Get-Content $PidFile -Raw).Trim() }
$alive = $nodePid -and (Get-Process -Id $nodePid -ErrorAction SilentlyContinue)

# 端口监听探测(即使 pidfile 失效也能发现)。
$portListening = $false
try {
  $listeners = [System.Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpListeners()
  $portListening = [bool]($listeners | Where-Object { $_.Port -eq [int]$port })
} catch { }

Write-Host ''
if ($alive) {
  Write-Host "状态:  RUNNING" -ForegroundColor Green
  Write-Host "PID:    $nodePid"
} elseif ($portListening) {
  Write-Host "状态:  运行中但 pidfile 缺失/失效(端口 $port 在监听)" -ForegroundColor Yellow
} else {
  Write-Host "状态:  STOPPED" -ForegroundColor Red
}
$portState = if ($portListening) { '监听中' } else { '未监听' }
Write-Host "端口:  $port ($portState)"
Write-Host "面板:  http://127.0.0.1:$port/dashboard"

$date = Get-Date -Format 'yyyy-MM-dd'
$supLog = Join-Path $LogDir "supervisor-$date.log"
if (Test-Path $supLog) {
  Write-Host ''
  Write-Host "--- $supLog 末 15 行 ---" -ForegroundColor DarkGray
  Get-Content $supLog -Tail 15 -Encoding UTF8
}
Write-Host ''
