# 注册开机自启计划任务(登录时触发,隐藏分离进程)。
# 零外部依赖:schtasks + wscript。/rl limited 普通权限即可(端口 3003 免管理员)。
$ErrorActionPreference = 'Stop'
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
$TaskName = 'WindsurfAPI'
$Vbs = (Resolve-Path (Join-Path $PSScriptRoot 'run.vbs')).Path
$Tr = "wscript `"$Vbs`""

# 已存在则先删,保证幂等。
schtasks /query /tn $TaskName 2>$null | Out-Null
if ($LASTEXITCODE -eq 0) {
  schtasks /delete /tn $TaskName /f | Out-Null
  Write-Host "已移除旧任务 $TaskName" -ForegroundColor Yellow
}

schtasks /create /tn $TaskName /sc onlogon /rl limited /tr $Tr /f
if ($LASTEXITCODE -eq 0) {
  Write-Host ''
  Write-Host "计划任务 '$TaskName' 已注册:登录 Windows 时自动隐藏后台启动。" -ForegroundColor Green
  Write-Host '现在可立即启动一次(无需重登):run-background.bat' -ForegroundColor White
  Write-Host '管理:status.bat 看状态,stop.bat 停,uninstall-task.bat 卸载任务。' -ForegroundColor White
} else {
  Write-Host "注册计划任务失败(exit=$LASTEXITCODE)。" -ForegroundColor Red
}
