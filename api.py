from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse

from config import settings
from local_llm import LocalLLM
from models import CompletionRequest, CompletionResponse, Usage

logger = logging.getLogger(__name__)

_llm_instance: LocalLLM | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application lifespan: startup and shutdown events."""
    # Startup
    global _llm_instance
    logger.info("Запус, инициализация модели: %s", settings.model_name)
    _llm_instance = LocalLLM(model_name=settings.model_name)
    logger.info("Модель инициализирована")
    
    yield


app = FastAPI(title="Local LLM API", version="1.0.0", lifespan=lifespan)


def get_llm() -> LocalLLM:
    global _llm_instance
    if _llm_instance is None:
        raise HTTPException(
            status_code=503,
            detail="Модель еще не проинициализирована, подождите пока сервер запустится.",
        )
    return _llm_instance


@app.get("/health")
def health() -> Dict[str, Any]:
    model_initialized = _llm_instance is not None
    return {
        "status": "ok",
        "model_name": settings.model_name,
        "model_initialized": model_initialized,
    }


@app.post("/completions/create", response_model=CompletionResponse)
async def create_completion(
    request: CompletionRequest, llm: LocalLLM = Depends(get_llm)
) -> CompletionResponse:
    logger.info("Получен запрос с %d сообщениями", len(request.messages))

    messages_payload = [
        {"role": msg.role.value, "content": msg.content} for msg in request.messages
    ]

    try:
        start = time.perf_counter()
        result = await llm.generate_chat(
            messages=messages_payload,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ошибка при генерации: %s", exc)
        raise HTTPException(status_code=500, detail="Generation failed") from exc

    usage = Usage(
        prompt_tokens=result["prompt_tokens"],
        completion_tokens=result["completion_tokens"],
        total_tokens=result["total_tokens"],
    )

    response = CompletionResponse(
        text=result["text"],
        model=request.model or settings.model_name,
        usage=usage,
        latency_ms=latency_ms,
    )

    logger.info(
        "Сгенерировано: prompt_tokens=%d, completion_tokens=%d, latency_ms=%.2f",
        usage.prompt_tokens,
        usage.completion_tokens,
        latency_ms,
    )
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):  # type: ignore[override]
    logger.warning("HTTP error %s: %s", exc.status_code, exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc: Exception):  # type: ignore[override]
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


