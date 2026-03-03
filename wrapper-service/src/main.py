"""
包装服务主程序
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app
from contextlib import asynccontextmanager
import httpx

from .config import get_settings
from .utils.logging import setup_logging, get_logger
from .utils.cache import ThreadSafeLRUCache
from .utils.circuit_breaker import CircuitBreaker
from .utils.http_pool import get_http_pool
from .utils.exceptions import (
    WrapperServiceError,
    ServiceUnavailableError,
    CircuitBreakerError,
)
from .utils import metrics


# 初始化配置和日志
settings = get_settings()
setup_logging(settings.log_level, settings.json_logs)
logger = get_logger(__name__)

# 初始化缓存
cache = ThreadSafeLRUCache(
    max_size=settings.cache_max_size, ttl_seconds=settings.cache_ttl
)

# 初始化熔断器
embedding_breaker = CircuitBreaker(
    failure_threshold=5, timeout=60.0, half_open_max_calls=3
)

llm_breaker = CircuitBreaker(failure_threshold=5, timeout=60.0, half_open_max_calls=3)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动
    logger.info("wrapper_service_starting", port=settings.port)
    metrics.service_info.info(
        {
            "version": "1.0.0",
            "embedding_url": settings.embedding_service_url,
            "llm_url": settings.llm_service_url,
        }
    )
    yield
    # 关闭
    logger.info("wrapper_service_stopping")
    await get_http_pool().close()


app = FastAPI(title="Embedding Wrapper Service", version="1.0.0", lifespan=lifespan)

# 挂载Prometheus指标端点
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.exception_handler(WrapperServiceError)
async def wrapper_error_handler(request: Request, exc: WrapperServiceError):
    """统一异常处理"""
    logger.error(
        "wrapper_service_error",
        error=str(exc),
        status_code=exc.status_code,
        path=request.url.path,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.message, "details": exc.details},
    )


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "cache_stats": cache.get_stats(),
        "circuit_breakers": {
            "embedding": embedding_breaker.state.value,
            "llm": llm_breaker.state.value,
        },
    }


@app.post("/v1/embeddings")
@metrics.track_request("POST", "/v1/embeddings")
async def create_embeddings(request: Request):
    """
    创建文本嵌入向量

    请求体示例:
    {
        "input": "text to embed",
        "model": "qwen3-embedding"
    }
    """
    body = await request.json()
    text = body.get("input", "")

    # 检查缓存
    cache_key = f"emb:{text}"
    cached = cache.get(cache_key)
    if cached:
        metrics.cache_hits.inc()
        logger.debug("cache_hit", key=cache_key)
        return cached

    metrics.cache_misses.inc()

    # 调用后端服务（带熔断保护）
    try:

        async def call_embedding_service():
            pool = get_http_pool()
            response = await pool.post(
                f"{settings.embedding_service_url}/v1/embeddings", json=body
            )
            response.raise_for_status()
            return response.json()

        result = embedding_breaker.call(call_embedding_service)

        # 缓存结果
        cache.put(cache_key, result)
        return result

    except CircuitBreakerError as e:
        logger.error("circuit_breaker_open", service="embedding")
        raise ServiceUnavailableError("Embedding service unavailable")
    except httpx.HTTPError as e:
        logger.error("backend_error", service="embedding", error=str(e))
        metrics.backend_errors.labels(
            service="embedding", error_type=type(e).__name__
        ).inc()
        raise ServiceUnavailableError(f"Embedding service error: {str(e)}")


@app.post("/v1/chat/completions")
@metrics.track_request("POST", "/v1/chat/completions")
async def create_chat_completion(request: Request):
    """
    创建聊天补全

    请求体示例:
    {
        "messages": [{"role": "user", "content": "Hello"}],
        "model": "qwen3"
    }
    """
    body = await request.json()

    # 调用后端服务（带熔断保护）
    try:

        async def call_llm_service():
            pool = get_http_pool()
            response = await pool.post(
                f"{settings.llm_service_url}/v1/chat/completions", json=body
            )
            response.raise_for_status()
            return response.json()

        result = llm_breaker.call(call_llm_service)
        return result

    except CircuitBreakerError as e:
        logger.error("circuit_breaker_open", service="llm")
        raise ServiceUnavailableError("LLM service unavailable")
    except httpx.HTTPError as e:
        logger.error("backend_error", service="llm", error=str(e))
        metrics.backend_errors.labels(service="llm", error_type=type(e).__name__).inc()
        raise ServiceUnavailableError(f"LLM service error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings.port)
