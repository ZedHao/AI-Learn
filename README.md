# 基于 Qwen3-4B / 8B 进行微调投资咨询 Agent

## 依赖与环境

项目根目录的 Python 依赖使用 [uv](https://github.com/astral-sh/uv) 管理。对话后端在 `backServer/` 下另有独立虚拟环境。

安装 uv（若尚未安装）：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

根目录（可选，用于其它脚本）：

```bash
uv sync
source .venv/bin/activate   # macOS / Linux
# Windows: .venv\Scripts\activate
```

## 基座模型目录（aiBaseModel）

4B / 8B 等基座模型统一放在仓库根目录的 **`aiBaseModel/`** 下（该目录下大文件已由 `.gitignore` 排除，仅提交占位文件，权重需本机自行下载或拷贝）。

推荐目录名：

- `aiBaseModel/Qwen3-4B-Base`
- `aiBaseModel/Qwen3-8B-Base`

使用 Hugging Face 下载示例（需在已安装 `huggingface_hub` 的环境中执行，例如根目录 `uv run python`）：

```bash
cd /path/to/AI-Learn
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-4B-Base', local_dir='./aiBaseModel/Qwen3-4B-Base')"
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-8B-Base', local_dir='./aiBaseModel/Qwen3-8B-Base')"
```

切换后端加载的模型（默认 4B）：启动前设置环境变量 `QWEN_MODEL_NAME`，例如使用 8B：

```bash
export QWEN_MODEL_NAME=Qwen3-8B-Base
```

也可直接指定完整路径（优先级最高）：

```bash
export QWEN_MODEL_PATH=/你的路径/某个模型目录
```

## 启动对话服务（前后端分离）

### 方式一：Shell 脚本（推荐）

在项目根目录执行：

**后端**（FastAPI + 本地 Qwen，SSE 流式接口，默认端口 `8000`）：

```bash
chmod +x scripts/run-backend.sh scripts/run-front.sh   # 首次需要
./scripts/run-backend.sh
```

**前端**（Vue + Vite，默认端口 `5173`）：

```bash
./scripts/run-front.sh
```

浏览器访问：<http://localhost:5173>。开发环境下 Vite 会将 `/api` 代理到后端 `http://127.0.0.1:8000`。

### 方式二：等价命令行

**后端：**

```bash
cd backServer
uv sync
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**前端：**

```bash
cd front
npm install
npm run dev
```

## 相关目录

| 路径 | 说明 |
|------|------|
| `aiBaseModel/` | 本地基座模型（不入库） |
| `backServer/` | 对话 API（SSE） |
| `front/` | Vue 对话页 |
| `scripts/` | 启动脚本 |
