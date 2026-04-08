#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/front"

if [[ ! -d node_modules ]]; then
  echo "正在安装前端依赖 (npm install)…"
  npm install
fi

exec npm run dev
