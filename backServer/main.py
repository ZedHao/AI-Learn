"""
本地 Qwen 流式对话 API，使用 SSE（text/event-stream）推送 token。
默认从仓库根目录下的 aiBaseModel/<QWEN_MODEL_NAME> 加载权重。
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
    """合并默认 system 提示，约束 Base 模型少续写代码/日志垃圾。"""
    if not messages:
        return messages
    base = _default_system_prompt()
    if messages[0]["role"] == "system":
        merged = f"{base}\n\n{messages[0]['content']}"
        return [{"role": "system", "content": merged}, *messages[1:]]
    return [{"role": "system", "content": base}, *messages]


class _GarbageLogStoppingCriteria(StoppingCriteria):
    """在生成中检测典型预训练垃圾续写片段并提前结束（仅匹配极窄模式，降低误伤）。"""

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


def _resolve_model_dir() -> Path:
    explicit = os.environ.get("QWEN_MODEL_PATH")
    if explicit:
        return Path(explicit).expanduser().resolve()
    name = os.environ.get("QWEN_MODEL_NAME", "Qwen3-4B-Base")
    return (_AI_BASE_MODEL / name).resolve()


MODEL_DIR = _resolve_model_dir()

app = FastAPI(title="Qwen Local Chat", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_tokenizer = None
_model = None
_device: str | None = None
_device_info: dict[str, Any] = {}


def _build_device_info(dev: str) -> dict[str, Any]:
    """供 /api/health 与启动日志使用：区分 NVIDIA CUDA、Apple MPS、CPU。"""
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


def load_model() -> None:
    global _tokenizer, _model, _device, _device_info
    if _model is not None:
        return
    if not MODEL_DIR.is_dir():
        raise RuntimeError(f"模型目录不存在: {MODEL_DIR}")
    _device = _pick_device()
    _device_info = _build_device_info(_device)
    dtype = _pick_dtype(_device)
    _tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    kwargs: dict = {"trust_remote_code": True, "dtype": dtype}
    if _device == "cuda":
        kwargs["device_map"] = "auto"
    _model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), **kwargs)
    if _device != "cuda":
        _model = _model.to(_device)
    _model.eval()
    print(
        "[qwen-chat] 推理设备: "
        f"{_device_info.get('label_zh')} (torch_device={_device}) "
        f"{_device_info.get('cuda_device_name') or ''}",
        flush=True,
    )


@app.on_event("startup")
def _startup() -> None:
    load_model()


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1)
    max_new_tokens: int = Field(1024, ge=1, le=8192)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    # Base 模型容易陷入重复循环，默认略加重惩罚；对话模型可调低到 1.0～1.05
    repetition_penalty: float = Field(1.15, ge=1.0, le=2.0)
    # 禁止连续重复同样长度的 n-gram，0 表示关闭（由 transformers 处理）
    no_repeat_ngram_size: int = Field(4, ge=0, le=16)


def _sse_chunk(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _generate_sse(request: ChatRequest):
    assert _tokenizer is not None and _model is not None and _device is not None

    messages = _messages_with_system([m.model_dump() for m in request.messages])
    try:
        # transformers>=5 默认 return_dict=True，会得到 BatchEncoding；generate 需要 Tensor
        # Qwen3：enable_thinking=False 会在 assistant 开头写入空 thinking 槽位，利于正常对话续写
        input_ids = _tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=False,
            enable_thinking=False,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"构建对话失败: {e}") from e

    input_ids = input_ids.to(_model.device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    streamer = TextIteratorStreamer(
        _tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    use_garbage_stop = os.environ.get("DISABLE_GARBAGE_STOP", "").lower() not in ("1", "true", "yes")
    gen_kwargs: dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "streamer": streamer,
        "max_new_tokens": request.max_new_tokens,
        "pad_token_id": _tokenizer.pad_token_id,
        "eos_token_id": _tokenizer.eos_token_id,
        "repetition_penalty": request.repetition_penalty,
    }
    if use_garbage_stop:
        gen_kwargs["stopping_criteria"] = StoppingCriteriaList([_GarbageLogStoppingCriteria(_tokenizer)])
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
            _model.generate(**gen_kwargs)

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
    return {
        "ok": True,
        "ai_base_model_root": str(_AI_BASE_MODEL),
        "model_dir": str(MODEL_DIR),
        "model_name": MODEL_DIR.name,
        "loaded": _model is not None,
        "device": _device_info if _device else None,
    }
