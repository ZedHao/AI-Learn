"""
本地 Qwen 流式对话 API：
- 默认加载 aiBaseModel 下 4B / 8B 两槽；
- 若设置环境变量 QWEN_ENGINE_PATHS（每行一个模型目录，相对仓库根或绝对路径），则仅加载所列目录，引擎 id 为 m0、m1…（与 scripts/run-backend.sh 传参一致）。
"""
from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path
from threading import Thread
from typing import Any, Literal

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

try:
    from transformers import Qwen2VLForConditionalGeneration, Qwen3VLForConditionalGeneration
except ImportError:
    Qwen2VLForConditionalGeneration = None  # type: ignore[misc, assignment]
    Qwen3VLForConditionalGeneration = None  # type: ignore[misc, assignment]

_REPO_ROOT = Path(__file__).resolve().parent.parent
_AI_BASE_MODEL = _REPO_ROOT / "aiBaseModel"

_BUILTIN_SYSTEM_PROMPT = """你是通义千问（Qwen）助手，正在用户本机运行。
请用简洁、自然的简体中文回答。
- 如实介绍身份：通义千问系列大语言模型；不要编造不存在的内部代号或机构背书。
- 除非用户明确要求，不要输出程序源码、FXML、Android 日志（如 MotionEvent）、乱码技术转储或大量无意义标点。
- 回答紧扣用户问题，避免冗长罗列与重复套话。"""


def _default_system_prompt() -> str:
    return os.environ.get("CHAT_SYSTEM_PROMPT", _BUILTIN_SYSTEM_PROMPT).strip()


def _messages_with_system(messages: list[dict]) -> list[dict]:
    if not messages:
        return messages
    base = _default_system_prompt()
    if messages[0]["role"] == "system":
        merged = f"{base}\n\n{messages[0]['content']}"
        return [{"role": "system", "content": merged}, *messages[1:]]
    return [{"role": "system", "content": base}, *messages]


class _GarbageLogStoppingCriteria(StoppingCriteria):
    _SUBSTRINGS = (
        "MotionEvent{",
        "otionEvent{",
        "FXMLComponent",
        "FXMLLoader",
        "alwaysSyncTouchSlop",
        "metaState=0 buttonState",
    )

    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        seq = input_ids[0]
        if seq.numel() < 8:
            return False
        tail = seq[-256:]
        text = self.tokenizer.decode(tail, skip_special_tokens=True)
        return any(s in text for s in self._SUBSTRINGS)


def _resolve_4b_dir() -> Path:
    if p := os.environ.get("QWEN_MODEL_4B_DIR") or os.environ.get("QWEN_MODEL_PATH"):
        return Path(p).expanduser().resolve()
    name = os.environ.get("QWEN_MODEL_4B_NAME", "Qwen3-4B-Base")
    return (_AI_BASE_MODEL / name).resolve()


def _resolve_8b_dir() -> Path:
    if p := os.environ.get("QWEN_MODEL_8B_DIR"):
        return Path(p).expanduser().resolve()
    name = os.environ.get("QWEN_MODEL_8B_NAME", "Qwen3-8B-Base")
    return (_AI_BASE_MODEL / name).resolve()


def _uses_engine_paths_env() -> bool:
    raw = os.environ.get("QWEN_ENGINE_PATHS", "").strip()
    if not raw:
        return False
    for line in raw.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return True
    return False


def _engine_load_entries() -> list[tuple[str, Path]]:
    """(引擎 id, 模型目录)。QWEN_ENGINE_PATHS 模式为 m0、m1…；否则为 4b、8b。"""
    if not _uses_engine_paths_env():
        return [("4b", _resolve_4b_dir()), ("8b", _resolve_8b_dir())]
    out: list[Path] = []
    seen: set[Path] = set()
    for line in os.environ.get("QWEN_ENGINE_PATHS", "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = Path(line).expanduser()
        if not p.is_absolute():
            p = (_REPO_ROOT / p).resolve()
        else:
            p = p.resolve()
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return [(f"m{i}", p) for i, p in enumerate(out)]


def _hf_model_kind(model_dir: Path) -> Literal["causal_lm", "qwen3_vl", "qwen2_vl"]:
    cfg_path = model_dir / "config.json"
    if not cfg_path.is_file():
        return "causal_lm"
    try:
        with cfg_path.open(encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return "causal_lm"
    mt = (data.get("model_type") or "").lower()
    if mt == "qwen3_vl":
        return "qwen3_vl"
    if mt == "qwen2_vl":
        return "qwen2_vl"
    for arch in data.get("architectures") or []:
        a = str(arch)
        if "Qwen3VL" in a:
            return "qwen3_vl"
        if "Qwen2VL" in a and "Omni" not in a:
            return "qwen2_vl"
    return "causal_lm"


def _messages_to_qwen_vl_text_only(messages: list[dict]) -> list[dict]:
    """VL 引擎：仅文本对话时转成 processor 所需 content 列表格式。"""
    out: list[dict] = []
    for m in messages:
        c = m["content"]
        if isinstance(c, str):
            out.append({"role": m["role"], "content": [{"type": "text", "text": c}]})
        elif isinstance(c, list):
            out.append({"role": m["role"], "content": c})
        else:
            out.append({"role": m["role"], "content": [{"type": "text", "text": str(c)}]})
    return out


app = FastAPI(title="Qwen Local Chat", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_engines: dict[str, dict[str, Any]] = {}
_engine_order: list[str] = []
_device: str | None = None
_device_info: dict[str, Any] = {}


def _build_device_info(dev: str) -> dict[str, Any]:
    info: dict[str, Any] = {
        "torch_device": dev,
        "backend": dev,
        "label_zh": "CPU",
        "platform_os": platform.system(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
    }
    if dev == "cuda":
        info["label_zh"] = "NVIDIA CUDA"
        if torch.cuda.is_available():
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
            info["gpu_count"] = torch.cuda.device_count()
    elif dev == "mps":
        info["label_zh"] = "Apple Silicon（MPS / Metal）"
        info["note_zh"] = "适用于 M1 / M2 / M3 / M4 等，走 Metal 加速而非 CUDA"
    else:
        info["label_zh"] = "CPU（无 GPU 加速）"
    return info


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _log_cuda_availability_hint() -> None:
    """torch 落在 CPU 时打印原因与处理建议（有显卡却 CPU 多为安装了 CPU 版 PyTorch）。"""
    ver = getattr(torch, "__version__", "") or ""
    cuda_ver = getattr(torch.version, "cuda", None)
    print(
        "[qwen-chat] 提示: torch.cuda.is_available() 为 False，当前使用 CPU 推理。"
        f" torch={ver} torch.version.cuda={cuda_ver!s}",
        flush=True,
    )
    if "+cpu" in ver or cuda_ver is None:
        print(
            "[qwen-chat] 多为「CPU 版 PyTorch」：请在仓库根目录执行 uv sync（见根目录 pyproject 中 pytorch-cu124 源），"
            "并确认 nvidia-smi 可用、驱动已装。",
            flush=True,
        )
    else:
        print(
            "[qwen-chat] 当前 PyTorch 为 CUDA 构建但仍不可用：请检查 nvidia-smi、是否多 Python 环境用错解释器、WSL 是否已装 CUDA。",
            flush=True,
        )


def _pick_dtype(dev: str) -> torch.dtype:
    if dev == "cuda":
        return torch.bfloat16
    if dev == "mps":
        return torch.bfloat16
    return torch.float32


def _load_one(engine_key: str, model_dir: Path, dtype: torch.dtype, device: str) -> None:
    kind = _hf_model_kind(model_dir)
    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    processor = None
    if kind in ("qwen3_vl", "qwen2_vl"):
        processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
    kwargs: dict = {"trust_remote_code": True, "dtype": dtype}
    if kind == "qwen3_vl":
        if Qwen3VLForConditionalGeneration is None:
            raise RuntimeError("当前 transformers 不支持 Qwen3-VL，请升级 transformers")
        model = Qwen3VLForConditionalGeneration.from_pretrained(str(model_dir), **kwargs)
    elif kind == "qwen2_vl":
        if Qwen2VLForConditionalGeneration is None:
            raise RuntimeError("当前 transformers 不支持 Qwen2-VL，请升级 transformers")
        model = Qwen2VLForConditionalGeneration.from_pretrained(str(model_dir), **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(str(model_dir), **kwargs)
    model = model.to(device)
    model.eval()
    _engines[engine_key] = {
        "tokenizer": tok,
        "processor": processor,
        "model": model,
        "path": model_dir,
        "name": model_dir.name,
        "kind": kind,
    }
    print(
        f"[qwen-chat] 已加载 engine={engine_key} kind={kind} path={model_dir}",
        flush=True,
    )


def _first_loaded_engine_id() -> str:
    for k in _engine_order:
        if k in _engines:
            return k
    return next(iter(_engines), "4b")


def load_models() -> None:
    global _device, _device_info, _engine_order
    if _engines:
        return
    _device = _pick_device()
    _device_info = _build_device_info(_device)
    if _device == "cpu":
        _log_cuda_availability_hint()
    dtype = _pick_dtype(_device)

    entries = _engine_load_entries()
    _engine_order = [k for k, _ in entries]
    if _uses_engine_paths_env() and not entries:
        raise RuntimeError(
            "环境变量 QWEN_ENGINE_PATHS 已设置但未解析到任何有效目录路径（每行一个，# 开头为注释）。"
        )

    for key, d in entries:
        if not d.is_dir():
            print(f"[qwen-chat] 跳过 engine={key}（目录不存在）: {d}", flush=True)
            continue
        if any(pack["path"] == d for pack in _engines.values()):
            print(f"[qwen-chat] 跳过 engine={key}（与已加载模型同路径）: {d}", flush=True)
            continue
        try:
            _load_one(key, d, dtype, _device)
        except Exception as e:
            print(f"[qwen-chat] engine={key} 加载失败: {e}", flush=True)

    if not _engines:
        if os.environ.get("QWEN_ALLOW_EMPTY_START", "").lower() in ("1", "true", "yes"):
            print(
                "[qwen-chat] 警告: 未加载任何模型；服务已启动，/api/chat/stream 将返回 503。"
                " 请传入模型路径，例如: ./scripts/run-backend.sh aiBaseModel/qwen3-vl/8b-instruct",
                flush=True,
            )
            return
        hint = ""
        if _uses_engine_paths_env():
            hint = " 已设置 QWEN_ENGINE_PATHS 但未成功加载任何有效目录（请检查路径是否存在且含 config.json）。"
        raise RuntimeError(
            "未能加载任何模型。可将模型目录作为参数传给 scripts/run-backend.sh，"
            "或设置环境变量 QWEN_ENGINE_PATHS（每行一个路径，相对仓库根或绝对路径），引擎 id 为 m0、m1…；"
            "亦可放置 Qwen3-4B-Base / Qwen3-8B-Base 到 aiBaseModel/，或设置 QWEN_MODEL_4B_DIR / QWEN_MODEL_8B_DIR。"
            f"{hint}"
            "开发调试可设置 QWEN_ALLOW_EMPTY_START=1。"
        )

    print(
        "[qwen-chat] 推理设备: "
        f"{_device_info.get('label_zh')} (torch_device={_device}) "
        f"{_device_info.get('cuda_device_name') or ''}",
        flush=True,
    )


@app.on_event("startup")
def _startup() -> None:
    load_models()


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(..., min_length=1)


class RagControl(BaseModel):
    """可选：启用后由 rag 包检索 doc/caibao 并注入 system（见 rag.service）。"""

    enabled: bool = False
    backend: Literal["faiss", "elasticsearch"] = "faiss"
    top_k: int = Field(5, ge=1, le=30)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1)
    engine: str = Field(
        "4b",
        description="引擎 id：默认 4b/8b；使用 QWEN_ENGINE_PATHS 时为 m0、m1…（见 GET /api/health）",
    )
    max_new_tokens: int = Field(1024, ge=1, le=8192)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    repetition_penalty: float = Field(1.15, ge=1.0, le=2.0)
    no_repeat_ngram_size: int = Field(4, ge=0, le=16)
    rag: RagControl | None = None


class RagRebuildBody(BaseModel):
    """POST /api/rag/rebuild 请求体（转发给 rag.service）。"""

    write_elasticsearch: bool = Field(False, description="是否同步写入 Elasticsearch")


def _sse_chunk(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _generate_sse(request: ChatRequest):
    eng = (request.engine or "").strip() or _first_loaded_engine_id()
    if not eng or eng not in _engines:
        raise HTTPException(
            status_code=503,
            detail=f"引擎 {eng or '(未指定)'} 未加载或 id 无效（参见 /api/health 的 engine_order）",
        )
    pack = _engines[eng]
    tokenizer = pack["tokenizer"]
    processor = pack.get("processor")
    model = pack["model"]
    kind = pack.get("kind", "causal_lm")
    assert _device is not None

    raw_msgs = [m.model_dump() for m in request.messages]
    messages = _messages_with_system(raw_msgs)

    rag_cfg = request.rag
    if rag_cfg and rag_cfg.enabled:
        from rag.service import get_rag_service, inject_rag_into_messages, last_user_plain_text

        q = last_user_plain_text(raw_msgs)
        try:
            ctx, req_dbg, resp_dbg = get_rag_service().retrieve(
                query=q,
                backend=rag_cfg.backend,
                top_k=rag_cfg.top_k,
            )
        except Exception as e:
            ctx = ""
            req_dbg = {"backend": rag_cfg.backend, "top_k": rag_cfg.top_k, "query_text": q}
            resp_dbg = {"hits": [], "error": f"retrieve_exception:{e}"}
        yield _sse_chunk({"type": "rag", "request": req_dbg, "response": resp_dbg})
        messages = inject_rag_into_messages(messages, ctx)

    use_garbage_stop = os.environ.get("DISABLE_GARBAGE_STOP", "").lower() not in ("1", "true", "yes")

    if kind in ("qwen3_vl", "qwen2_vl"):
        if processor is None:
            raise HTTPException(status_code=500, detail="视觉引擎未加载 processor")
        vl_msgs = _messages_to_qwen_vl_text_only(messages)
        stream_tok = processor.tokenizer
        try:
            inputs = processor.apply_chat_template(
                vl_msgs,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"VL 构建输入失败: {e}") from e
        inputs = inputs.to(model.device)
        streamer = TextIteratorStreamer(
            stream_tok,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        gen_kwargs: dict[str, Any] = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": request.max_new_tokens,
            "pad_token_id": stream_tok.pad_token_id,
            "eos_token_id": stream_tok.eos_token_id,
            "repetition_penalty": request.repetition_penalty,
        }
        if use_garbage_stop:
            gen_kwargs["stopping_criteria"] = StoppingCriteriaList([_GarbageLogStoppingCriteria(stream_tok)])
        if request.no_repeat_ngram_size > 0:
            gen_kwargs["no_repeat_ngram_size"] = request.no_repeat_ngram_size
        if request.temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = request.temperature
            gen_kwargs["top_p"] = request.top_p
        else:
            gen_kwargs["do_sample"] = False
    else:
        try:
            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=False,
                enable_thinking=False,
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"构建对话失败: {e}") from e

        input_ids = input_ids.to(model.device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "streamer": streamer,
            "max_new_tokens": request.max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "repetition_penalty": request.repetition_penalty,
        }
        if use_garbage_stop:
            gen_kwargs["stopping_criteria"] = StoppingCriteriaList([_GarbageLogStoppingCriteria(tokenizer)])
        if request.no_repeat_ngram_size > 0:
            gen_kwargs["no_repeat_ngram_size"] = request.no_repeat_ngram_size
        if request.temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = request.temperature
            gen_kwargs["top_p"] = request.top_p
        else:
            gen_kwargs["do_sample"] = False

    def _run_generate() -> None:
        with torch.inference_mode():
            model.generate(**gen_kwargs)

    worker = Thread(target=_run_generate, daemon=True)
    worker.start()

    try:
        for text in streamer:
            if text:
                yield _sse_chunk({"type": "token", "text": text})
    except Exception as e:
        yield _sse_chunk({"type": "error", "message": str(e)})
        yield "data: [DONE]\n\n"
        return

    worker.join(timeout=1.0)
    yield "data: [DONE]\n\n"


@app.post("/api/chat/stream")
def chat_stream(request: ChatRequest):
    return StreamingResponse(
        _generate_sse(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/rag/status")
def rag_status():
    """RAG 索引状态（委托 rag.service，便于前端展示）。"""
    from rag.service import get_rag_service

    return get_rag_service().status()


@app.post("/api/rag/rebuild")
def rag_rebuild(body: RagRebuildBody):
    """重建向量索引（委托 rag.service）。"""
    from rag.service import get_rag_service

    print(
        f"[rag-index] HTTP POST /api/rag/rebuild | write_elasticsearch={body.write_elasticsearch}",
        flush=True,
    )
    try:
        result = get_rag_service().build_all(write_es=body.write_elasticsearch)
        print("[rag-index] HTTP /api/rag/rebuild 成功", flush=True)
        return {"ok": True, "result": result}
    except Exception as e:
        print(f"[rag-index] HTTP /api/rag/rebuild 失败: {e}", flush=True)
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/api/health")
def health():
    engines_out: dict[str, Any] = {}
    for key, d in _engine_load_entries():
        if key in _engines:
            p = _engines[key]["path"]
            knd = _engines[key].get("kind", "causal_lm")
            engines_out[key] = {
                "loaded": True,
                "model_dir": str(p),
                "model_name": _engines[key]["name"],
                "kind": knd,
                "supports_vision": knd in ("qwen3_vl", "qwen2_vl"),
            }
        else:
            uk = _hf_model_kind(d)
            engines_out[key] = {
                "loaded": False,
                "model_dir": str(d),
                "model_name": d.name,
                "kind": uk,
                "supports_vision": uk in ("qwen3_vl", "qwen2_vl"),
            }
    first_loaded = next((k for k in _engine_order if k in _engines), None)
    primary = _engines[first_loaded] if first_loaded else None
    return {
        "ok": True,
        "ai_base_model_root": str(_AI_BASE_MODEL),
        "engines": engines_out,
        "engine_order": list(_engine_order),
        "default_engine": first_loaded or (_engine_order[0] if _engine_order else None),
        "primary_model_name": primary["name"] if primary else None,
        "loaded": len(_engines) > 0,
        "device": _device_info if _device else None,
    }
