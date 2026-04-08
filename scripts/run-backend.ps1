#Requires -Version 5.0
# 在项目根目录执行: .\scripts\run-backend.ps1
# 环境变量 BACKEND_PORT 可改端口（默认 8000）
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location (Join-Path $Root "backServer")

$Port = if ($env:BACKEND_PORT) { $env:BACKEND_PORT } else { "8000" }
$PortNum = [int]$Port

try {
    $listeners = @(Get-NetTCPConnection -LocalPort $PortNum -State Listen -ErrorAction SilentlyContinue)
    foreach ($l in $listeners) {
        Write-Host "端口 $Port 已被占用 (PID: $($l.OwningProcess))，结束进程…" -ForegroundColor Yellow
        Stop-Process -Id $l.OwningProcess -Force -ErrorAction SilentlyContinue
    }
    if ($listeners.Count -gt 0) {
        Start-Sleep -Milliseconds 400
    }
}
catch {
    Write-Host "提示: 无法自动释放端口（需 Windows 10+ / PowerShell 5+）。若 Address already in use，请用任务管理器或 netstat 结束占用 $Port 的进程。" -ForegroundColor DarkYellow
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "错误: 未找到 uv。请先安装: https://docs.astral.sh/uv/" -ForegroundColor Red
    exit 1
}

& uv run uvicorn main:app --reload --host 0.0.0.0 --port $PortNum
