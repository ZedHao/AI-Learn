#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/backServer"

PORT="${BACKEND_PORT:-8000}"

if ! command -v lsof >/dev/null 2>&1; then
  echo "警告: 未找到 lsof，无法自动释放端口。请手动结束占用 ${PORT} 的进程。" >&2
else
  _pids="$(lsof -t -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)"
  if [ -n "${_pids}" ]; then
    _shown="${_pids//$'\n'/, }"
    echo "检测到端口 ${PORT} 已被占用 (PID: ${_shown})，执行 kill -9 …" >&2
    # 多个 PID 时 lsof -t 为多行，展开为多个 kill 参数
    # shellcheck disable=SC2086
    kill -9 ${_pids} 2>/dev/null || true
    sleep 0.4
  fi
fi

exec uv run uvicorn main:app --reload --host 0.0.0.0 --port "$PORT"
