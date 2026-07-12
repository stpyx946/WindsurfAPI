# 停止 WindsurfAPI。
# 顺序关键:先杀监督者(powershell/wscript,否则它会立刻重拉 node),再停 node。
# node 停机:先试 taskkill(不带 /F)+ 短等 3s,超时即 /F 兜底。K7 起 accounts.json
# 是 fsync 原子写 + dirty-flush 持续落盘,/F 安全(不截断、最多丢一个 flush 周期)。
$ErrorActionPreference = 'Continue'
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$PidFile = Join-Path $Root 'logs\windsurfapi.pid'

# ── 1. 先结束计划任务(如果用 schtasks 装的),它是监督者的父 ──
schtasks /end /tn WindsurfAPI 2>$null | Out-Null

# ── 2. 杀掉监督循环脚本(否则会重拉 node)──
#   监督循环可能活在三种进程里:
#     - run.ps1 直接跑(run.bat 前台)
#     - run.vbs 起的 wscript(run-background 隐藏后台)
#     - start.ps1 —— 它末尾用 `& run.ps1` 内联执行监督循环(同一进程),
#       所以进程命令行是 start.ps1 而非 run.ps1。漏了它会导致 stop 杀掉
#       node 后,start.ps1 里的循环立刻重拉一个新 node(停不掉)。
$killedSup = 0
Get-CimInstance Win32_Process -Filter "Name='powershell.exe' OR Name='wscript.exe'" -ErrorAction SilentlyContinue | ForEach-Object {
  $cl = $_.CommandLine
  if ($cl -and ($cl -match 'run\.ps1' -or $cl -match 'run\.vbs' -or $cl -match 'start\.ps1')) {
    try { Stop-Process -Id $_.ProcessId -Force -ErrorAction Stop; $killedSup++ } catch { }
  }
}
if ($killedSup -gt 0) { Write-Host "已停止 $killedSup 个监督进程。" -ForegroundColor Yellow }

# ── 3. 优雅停 node ─────────────────────────────────────────
$nodePid = $null
if (Test-Path $PidFile) {
  $nodePid = (Get-Content $PidFile -Raw).Trim()
}

if ($nodePid -and (Get-Process -Id $nodePid -ErrorAction SilentlyContinue)) {
  Write-Host "正在停止 node (pid=$nodePid)..." -ForegroundColor White
  # Windows 现实:node 是无窗口 console 进程,外部脚本用 taskkill(不带 /F)发的
  # WM_CLOSE 它收不到,SIGINT/SIGTERM 也无法可靠跨 console 投递 → 优雅关几乎必然
  # 超时。K7 起 accounts.json 是 fsync 原子写 + 池态由 30s dirty-flush 持续落盘,
  # 强杀(/F)不会截断文件、最多丢一个 flush 周期的自愈态 → 直接 /F 是安全的。
  # 先给一次不带 /F 的尝试(极少数情形 node 能响应)+ 短等待 3s,超时即 /F 兜底,
  # 不再干等 8s。真正的"抽干在途 SSE"优雅停走前台 Ctrl-C / 关窗口(run.ps1 循环)。
  taskkill /PID $nodePid /T 2>$null | Out-Null
  $waited = 0
  while ((Get-Process -Id $nodePid -ErrorAction SilentlyContinue) -and $waited -lt 3) {
    Start-Sleep -Seconds 1; $waited++
  }
  if (Get-Process -Id $nodePid -ErrorAction SilentlyContinue) {
    Write-Host '强制结束(K7:accounts.json 原子写 + dirty-flush,安全)...' -ForegroundColor Yellow
    taskkill /F /PID $nodePid /T 2>$null | Out-Null
  }
  Write-Host 'node 已停止。' -ForegroundColor Green
} else {
  Write-Host 'pidfile 无有效 node 进程(可能已停)。' -ForegroundColor DarkGray
}

if (Test-Path $PidFile) { Remove-Item $PidFile -Force -ErrorAction SilentlyContinue }
Write-Host 'WindsurfAPI 已停止。' -ForegroundColor Green
