# 基于 Qwen 多模型本地对话（可对比 4B / 8B 或任意多个权重）

**后端**（不传参数时沿用自动 4b/8b 解析；无模型且允许空启动时仍可起服务）：

```bash
./scripts/run-backend.sh  aiBaseModel/qwen3-vl/8b-instruct 
```

带多个模型路径的示例见上文「指定加载哪些模型」。

**前端**（Vue + Vite，默认端口 `5173`）：

```bash
./scripts/run-front.sh 
```


## 依赖与环境

项目根目录的 Python 依赖使用 [uv](https://github.com/astral-sh/uv) 统一管理（含对话后端 `backServer/` 与 RAG 等，单一 `pyproject.toml` 在仓库根目录）。

安装 uv（若尚未安装）：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
echo 'export PATH="$PATH:$HOME/.local/bin"' >> ~/.bashrc && source ~/.bashrc
```

根目录（可选，用于其它脚本）：

```bash
uv sync
source .venv/bin/activate   # macOS / Linux
# Windows: .venv\Scripts\activate source .venv/Scripts/activate .venv/Scripts/activate
```

## 基座模型目录（aiBaseModel）

基座模型放在仓库根目录的 **`aiBaseModel/`** 下（大文件由 `.gitignore` 排除，权重需本机下载或拷贝）。

常见布局示例（Qwen3-VL 等）：

- `aiBaseModel/qwen3-vl/4b-instruct`
- `aiBaseModel/qwen3-vl/8b-instruct`

### 指定加载哪些模型（推荐）

**在一条命令里传入几个模型目录，就加载几个**（引擎 id 依次为 **`m0`、`m1`、`m2`…**，见 `GET /api/health` 的 `engine_order`）。路径为**相对仓库根**或**绝对路径**：

```bash
chmod +x scripts/run-backend.sh scripts/run-front.sh   # 首次需要
./scripts/run-backend.sh \
  aiBaseModel/qwen3-vl/4b-instruct \
  aiBaseModel/qwen3-vl/8b-instruct
```

传入参数时脚本默认 **`QWEN_ALLOW_EMPTY_START=0`**（须至少成功加载一个模型）；仅起 HTTP、暂不加载权重时可设 `QWEN_ALLOW_EMPTY_START=1`。

**Windows（PowerShell）**：

```powershell
.\scripts\run-backend.ps1 aiBaseModel\qwen3-vl\4b-instruct aiBaseModel\qwen3-vl\8b-instruct
```

**等价：环境变量 `QWEN_ENGINE_PATHS`**（每行一个目录，可在 `backServer` 下启动 uvicorn）：

```bash
export QWEN_ENGINE_PATHS="$(printf '%s\n' \
  "$PWD/aiBaseModel/qwen3-vl/4b-instruct" \
  "$PWD/aiBaseModel/qwen3-vl/8b-instruct")"
cd backServer && uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 不设置 QWEN_ENGINE_PATHS 时（自动 4b / 8b 两槽）

未设置 `QWEN_ENGINE_PATHS` 时，后端按 **`4b` / `8b`** 两槽自动解析（默认尝试 `qwen3-vl/4b-instruct` 等，亦兼容 `Qwen3-4B-Base`），可用 `QWEN_MODEL_4B_DIR`、`QWEN_MODEL_8B_DIR` 或 `QWEN_MODEL_PATH`（4b 槽）覆盖。

Hugging Face 下载示例：

```bash
cd /path/to/AI-Learn
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-4B-Base', local_dir='./aiBaseModel/Qwen3-4B-Base')"
```

## 启动对话服务（前后端分离）

### 方式一：Shell 脚本（推荐）


浏览器访问：<http://localhost:5173>。Vite 将 `/api` 代理到 `http://127.0.0.1:8000`；前端根据 `/api/health` 的 **`engine_order`** 动态渲染对比栏。

### 方式二：等价命令行

**后端：**

```bash
cd backServer
uv sync   # 在仓库根目录解析 pyproject.toml（可从任意子目录向上找到）
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**前端：**

仓库根目录与 `front/.npmrc` 均已指向 **npmmirror**（`https://registry.npmmirror.com/`）；在根目录或 `front/` 下执行 `npm install` 都会走国内镜像。若需官方源，将对应 `.npmrc` 里的 `registry` 改为 `https://registry.npmjs.org/`。全局配置可编辑用户主目录下的 `.npmrc`（见根目录 `.npmrc` 内注释）。

```bash
cd front
npm install
npm run dev
```

### NVIDIA GPU（如 RTX 4070）不走 CPU

若日志里出现 **`torch ...+cpu`** 或 **`CPU（无 GPU 加速）`**，说明当前虚拟环境里是 **CPU 版 PyTorch**，与是否插了显卡无关。

根目录 `pyproject.toml` 中 `torch` / `torchvision` 通过 **explicit** 索引从 **`https://download.pytorch.org/whl/cu124`** 安装 CUDA 12.4 版；其余包默认走 **清华 PyPI 镜像**（`pypi.tuna.tsinghua.edu.cn/simple`）。若需改用南大 PyTorch 镜像，可将 `[[tool.uv.index]]` 中 `pytorch-cu124` 的 `url` 改为 `https://mirrors.nju.edu.cn/pytorch/whl/cu124`。在**仓库根目录**同步依赖（或在子目录执行，uv 会向上找到根 `pyproject.toml`）：

```bash
cd /path/to/AI-Learn
uv sync
```

其余 PyPI 包若仍慢，可让本次终端的 **默认 PyPI 索引** 走清华（`torch` 仍走 pyproject 里的 TUNA PyTorch 源）：

```bash
# Linux / macOS / Git Bash
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
uv sync
```

```powershell
# Windows PowerShell
$env:UV_DEFAULT_INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple"
uv sync
```

完成后可用下面命令自检（应显示 `cuda_available True` 与 `+cu124` 等，而非 `+cpu`）：

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

若仍为 CPU 版，请确认本机已安装 **NVIDIA 驱动**（PowerShell / CMD 中 `nvidia-smi` 可用），并删除 `.venv` 后再次 `uv sync`。

## 相关目录

| 路径 | 说明 |
|------|------|
| `aiBaseModel/` | 本地基座模型（不入库） |
| `backServer/` | 对话 API（SSE） |
| `front/` | Vue 对话页 |
| `scripts/` | 启动脚本 |
