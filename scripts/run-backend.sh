#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/backServer"

PORT="${BACKEND_PORT:-8000}"

# 未传模型路径时：允许无模型启动（与 run-backend.ps1 一致），避免克隆后无权重即崩溃
if [ "$#" -eq 0 ]; then
  export QWEN_ALLOW_EMPTY_START="${QWEN_ALLOW_EMPTY_START:-1}"
fi

# 传入任意个模型目录（相对仓库根或绝对路径）时写入 QWEN_ENGINE_PATHS，加载几个算几个（引擎 id 为 m0、m1…）
# 示例: ./scripts/run-backend.sh aiBaseModel/qwen3-vl/8b-instruct
if [ "$#" -gt 0 ]; then
  export QWEN_ALLOW_EMPTY_START="${QWEN_ALLOW_EMPTY_START:-0}"
  export QWEN_ENGINE_PATHS=""
  for rel in "$@"; do
    [ -z "$rel" ] && continue
    case "$rel" in
      /*) abs="$rel" ;;
      *)
        # Git Bash / MSYS: /e/... 视为绝对路径
        if [[ "$rel" =~ ^[A-Za-z]:[\\/] ]] || [[ "$rel" =~ ^/[a-z]/ ]]; then
          abs="$rel"
        else
          abs="$ROOT/$rel"
        fi
        ;;
    esac
    QWEN_ENGINE_PATHS+="$abs"$'\n'
  done
  export QWEN_ENGINE_PATHS
fi

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
