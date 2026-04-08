"""
本地 Qwen 流式对话 API：可同时加载 4B / 8B，经 engine 分流；SSE 推送 token。
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
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

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


app = FastAPI(title="Qwen Local Chat", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# engine -> {tokenizer, model, path}
_engines: dict[str, dict[str, Any]] = {}
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


def _pick_dtype(dev: str) -> torch.dtype:
    if dev == "cuda":
        return torch.bfloat16
    if dev == "mps":
        return torch.bfloat16
    return torch.float32


def _load_one(engine_key: str, model_dir: Path, dtype: torch.dtype, device: str) -> None:
    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    kwargs: dict = {"trust_remote_code": True, "dtype": dtype}
    # 单进程加载多个完整模型时不用 device_map="auto"，避免 accelerate 与多权重争用
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), **kwargs)
    model = model.to(device)
    model.eval()
    _engines[engine_key] = {
        "tokenizer": tok,
        "model": model,
        "path": model_dir,
        "name": model_dir.name,
    }
    print(
        f"[qwen-chat] 已加载 engine={engine_key} path={model_dir}",
        flush=True,
    )


def load_models() -> None:
    global _device, _device_info
    if _engines:
        return
    _device = _pick_device()
    _device_info = _build_device_info(_device)
    dtype = _pick_dtype(_device)

    dirs = [("4b", _resolve_4b_dir()), ("8b", _resolve_8b_dir())]
    for key, d in dirs:
        if not d.is_dir():
            print(f"[qwen-chat] 跳过 engine={key}（目录不存在）: {d}", flush=True)
            continue
        try:
            _load_one(key, d, dtype, _device)
        except Exception as e:
            print(f"[qwen-chat] engine={key} 加载失败: {e}", flush=True)

    if not _engines:
        raise RuntimeError(
            "未能加载任何模型。请将 Qwen3-4B-Base / Qwen3-8B-Base 放入 aiBaseModel/，"
            "或设置 QWEN_MODEL_4B_DIR / QWEN_MODEL_8B_DIR。"
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


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1)
    engine: Literal["4b", "8b"] = "4b"
    max_new_tokens: int = Field(1024, ge=1, le=8192)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    repetition_penalty: float = Field(1.15, ge=1.0, le=2.0)
    no_repeat_ngram_size: int = Field(4, ge=0, le=16)


def _sse_chunk(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _generate_sse(request: ChatRequest):
    eng = request.engine
    if eng not in _engines:
        raise HTTPException(
            status_code=503,
            detail=f"引擎 {eng} 未加载（模型目录可能不存在或加载失败）",
        )
    pack = _engines[eng]
    tokenizer = pack["tokenizer"]
    model = pack["model"]
    assert _device is not None

    messages = _messages_with_system([m.model_dump() for m in request.messages])
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
    use_garbage_stop = os.environ.get("DISABLE_GARBAGE_STOP", "").lower() not in ("1", "true", "yes")
    gen_kwargs: dict = {
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


@app.get("/api/health")
def health():
    engines_out: dict[str, Any] = {}
    for key in ("4b", "8b"):
        if key in _engines:
            p = _engines[key]["path"]
            engines_out[key] = {
                "loaded": True,
                "model_dir": str(p),
                "model_name": _engines[key]["name"],
            }
        else:
            d = _resolve_4b_dir() if key == "4b" else _resolve_8b_dir()
            engines_out[key] = {
                "loaded": False,
                "model_dir": str(d),
                "model_name": d.name,
            }
    primary = _engines.get("4b") or _engines.get("8b")
    return {
        "ok": True,
        "ai_base_model_root": str(_AI_BASE_MODEL),
        "engines": engines_out,
        "primary_model_name": primary["name"] if primary else None,
        "loaded": len(_engines) > 0,
        "device": _device_info if _device else None,
    }
