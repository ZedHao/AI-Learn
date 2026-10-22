#Requires -Version 5.0
# 在项目根目录执行: .\scripts\run-backend.ps1
# 可选：在命令行末尾传入多个模型目录（相对仓库根或绝对路径），将设置 QWEN_ENGINE_PATHS，引擎 id 为 m0、m1…
# 示例: .\scripts\run-backend.ps1 aiBaseModel\qwen3-vl\4b-instruct aiBaseModel\qwen3-vl\8b-instruct
# 环境变量 BACKEND_PORT 可改端口（默认 8000）
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ModelPaths
)

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

if ($ModelPaths -and $ModelPaths.Count -gt 0) {
    $lines = New-Object System.Collections.Generic.List[string]
    foreach ($rel in $ModelPaths) {
        if ([string]::IsNullOrWhiteSpace($rel)) { continue }
        if ([System.IO.Path]::IsPathRooted($rel)) {
            $abs = [System.IO.Path]::GetFullPath($rel)
        }
        else {
            $abs = [System.IO.Path]::GetFullPath((Join-Path $Root $rel))
        }
        $lines.Add($abs)
    }
    $env:QWEN_ENGINE_PATHS = ($lines -join "`n")
    if ([string]::IsNullOrWhiteSpace($env:QWEN_ALLOW_EMPTY_START)) {
        $env:QWEN_ALLOW_EMPTY_START = "0"
    }
}
elseif ([string]::IsNullOrWhiteSpace($env:QWEN_ALLOW_EMPTY_START)) {
    # 克隆后 aiBaseModel 常为空；未指定模型路径时允许无模型启动（/api/chat/stream 返回 503）
    $env:QWEN_ALLOW_EMPTY_START = "1"
}

& uv run uvicorn main:app --reload --host 0.0.0.0 --port $PortNum
