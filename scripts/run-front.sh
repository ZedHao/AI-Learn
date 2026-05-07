#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/front"

# 部分环境（尤其 Windows + Git Bash + Node 24）下，继承的变量会导致 npm 以
# “Exit handler never called!” 异常结束。安装/启动前在子进程中剥离常见干扰项。
# 若你确实需要 NODE_OPTIONS（如内存上限），可 export FRONT_KEEP_NODE_OPTIONS=1
_front_npm_env() {
  export MSYS2_ARG_CONV_EXCL='*'
  export CYGWIN_ARG_CONV_EXCL='*'
  if [[ "${FRONT_KEEP_NODE_OPTIONS:-}" != "1" ]]; then
    unset NODE_OPTIONS 2>/dev/null || true
  fi
  # 避免把其它工具里的 npm 配置带进本项目
  unset npm_config_prefix 2>/dev/null || true
  unset npm_config_global 2>/dev/null || true
}

# 仅有空目录或安装中断时，node_modules 存在但无 vite，仍会跳过安装导致「vite 不是命令」
if [[ ! -d node_modules ]] || [[ ! -f node_modules/vite/package.json ]]; then
  _lvl="${FRONT_NPM_LOGLEVEL:-verbose}"
  _log="$ROOT/front/npm-install-last.log"
  echo "正在安装前端依赖（细则日志：npm --loglevel ${_lvl}，并写入 ${_log}）…"
  echo "  更啰嗦可设: FRONT_NPM_LOGLEVEL=silly"
  echo "  Bash 逐行跟踪可设: FRONT_BASH_TRACE=1"
  (
    _front_npm_env
    if [[ "${FRONT_BASH_TRACE:-}" == "1" ]]; then
      set -x
    fi
    echo "---- npm install 诊断 ----"
    echo "PWD=$PWD"
    command -v node 2>/dev/null && node -v || echo "node: 未找到"
    command -v npm 2>/dev/null && npm -v || echo "npm: 未找到"
    echo "npm 配置摘要:"
    npm config list -l 2>/dev/null | head -n 40 || true
    echo "（完整配置见: npm config list -l）"
    echo "--------------------------"
    set -o pipefail
    # --foreground-scripts：install 钩子/postinstall 的输出直接可见，便于看卡在哪一步
    # --timing：结束时打印各阶段耗时
    npm install --loglevel="$_lvl" --foreground-scripts --timing 2>&1 | tee "$_log"
  )
  echo "若仍失败，请把终端最后几十行与文件 npm-install-last.log 一并反馈；npm 自述日志通常在:"
  echo "  Windows: %LocalAppData%\\\\npm-cache\\\\_logs\\\\ 下最新的 debug-0.log"
fi

_front_npm_env
exec npm run dev
