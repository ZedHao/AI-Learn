#Requires -Version 5.0
# 在项目根目录执行: .\scripts\run-front.ps1
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location (Join-Path $Root "front")

if (-not (Test-Path "node_modules")) {
    Write-Host "正在安装前端依赖 (npm install)…" -ForegroundColor Cyan
    npm install
}

npm run dev
