"""
包装服务主程序
"""

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
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
    AuthError,
    PermissionDeniedError,
)
from .utils.auth import require_auth, require_permission, Permission
from .utils.connection_pool import SurrealDBConnectionPool
from .utils.memory_manager import MemoryManager


# 初始化配置和日志
settings = get_settings()
setup_logging(settings.log_level, settings.json_logs)
logger = get_logger(__name__)

# 初始化缓存
cache = ThreadSafeLRUCache(max_size=settings.cache_max_size, ttl_seconds=settings.cache_ttl)

# 初始化熔断器
embedding_breaker = CircuitBreaker(failure_threshold=5, timeout=60.0, half_open_max_calls=3)

llm_breaker = CircuitBreaker(failure_threshold=5, timeout=60.0, half_open_max_calls=3)
# SurrealDB连接池和记忆管理器（在lifespan中初始化）
surrealdb_pool: SurrealDBConnectionPool | None = None
memory_manager: MemoryManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动
    logger.info("wrapper_service_starting", port=settings.port)
    # 初始化SurrealDB连接池
    global surrealdb_pool, memory_manager
    surrealdb_pool = SurrealDBConnectionPool(
        url=settings.surrealdb_url,
        namespace=settings.surrealdb_namespace,
        database=settings.surrealdb_database,
        username=settings.surrealdb_username,
        password=settings.surrealdb_password,
        pool_size=settings.surrealdb_pool_size,
        max_overflow=settings.surrealdb_max_overflow,
    )
    await surrealdb_pool.initialize()

    # 初始化记忆管理器
    memory_manager = MemoryManager(
        pool=surrealdb_pool,
        embedding_service_url=settings.embedding_service_url,
    )
    logger.info("surrealdb_initialized")
    yield
    # 关闭
    logger.info("wrapper_service_stopping")
    await get_http_pool().close()
    # 关闭SurrealDB连接池
    if memory_manager:
        await memory_manager.close()
    if surrealdb_pool:
        await surrealdb_pool.close()
    logger.info("surrealdb_closed")


app = FastAPI(title="Embedding Wrapper Service", version="1.0.0", lifespan=lifespan)


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
    # 检查SurrealDB健康状态
    surrealdb_healthy = False
    if surrealdb_pool:
        try:
            async with surrealdb_pool.acquire() as conn:
                from .utils.surrealdb_client import SurrealDBClient

                client = SurrealDBClient(conn)
                surrealdb_healthy = await client.health_check()
        except Exception as e:
            logger.error("surrealdb_health_check_failed", error=str(e))

    return {
        "status": "healthy",
        "cache_stats": cache.get_stats(),
        "circuit_breakers": {
            "embedding": embedding_breaker.state.value,
            "llm": llm_breaker.state.value,
        },
        "surrealdb": "healthy" if surrealdb_healthy else "unhealthy",
    }


@app.post("/v1/embeddings")
async def create_embeddings(
    request: Request,
    permissions: list[str] = Depends(require_auth),
):
    """
    创建文本嵌入向量 (需要 read 权限)

    请求体示例:
    {
        "input": "text to embed",
        "model": "qwen3-embedding"
    }
    """
    # 权限检查
    if Permission.READ not in permissions and Permission.ADMIN not in permissions:
        raise PermissionDeniedError("Requires read permission")
    body = await request.json()
    text = body.get("input", "")

    # 检查缓存
    cache_key = f"emb:{text}"
    cached = cache.get(cache_key)
    if cached:
        logger.debug("cache_hit", key=cache_key)
        return cached

    # 调用后端服务（带熔断保护）
    try:

        async def call_embedding_service():
            pool = get_http_pool()
            response = await pool.post(f"{settings.embedding_service_url}/v1/embeddings", json=body)
            response.raise_for_status()
            return response.json()

        result = embedding_breaker.call(call_embedding_service)

        # 缓存结果
        cache.set(cache_key, result)
        return result

    except CircuitBreakerError as e:
        logger.error("circuit_breaker_open", service="embedding")
        raise ServiceUnavailableError("Embedding service unavailable")
    except httpx.HTTPError as e:
        logger.error("backend_error", service="embedding", error=str(e))
        raise ServiceUnavailableError(f"Embedding service error: {str(e)}")


@app.post("/v1/chat/completions")
async def create_chat_completion(
    request: Request,
    permissions: list[str] = Depends(require_auth),
):
    """
    创建聊天补全 (需要 read 权限)

    请求体示例:
    {
        "messages": [{"role": "user", "content": "Hello"}],
        "model": "qwen3"
    }
    """
    # 权限检查
    if Permission.READ not in permissions and Permission.ADMIN not in permissions:
        raise PermissionDeniedError("Requires read permission")
    body = await request.json()

    # 调用后端服务（带熔断保护）
    try:

        async def call_llm_service():
            pool = get_http_pool()
            response = await pool.post(f"{settings.llm_service_url}/v1/chat/completions", json=body)
            response.raise_for_status()
            return response.json()

        result = llm_breaker.call(call_llm_service)
        return result

    except CircuitBreakerError as e:
        logger.error("circuit_breaker_open", service="llm")
        raise ServiceUnavailableError("LLM service unavailable")
    except httpx.HTTPError as e:
        logger.error("backend_error", service="llm", error=str(e))
        raise ServiceUnavailableError(f"LLM service error: {str(e)}")


@app.post("/api/v1/memories")
async def upload_memories(
    request: Request,
    permissions: list[str] = Depends(require_auth),
):
    """
    批量上传记忆 (需要 write 权限)

    请求体示例:
    {
        "memories": [
            {
                "content": "memory text",
                "metadata": {},
                "entities": [],
                "relations": []
            }
        ]
    }
    """
    # 权限检查
    if Permission.WRITE not in permissions and Permission.ADMIN not in permissions:
        raise PermissionDeniedError("Requires write permission")
    if not memory_manager:
        raise ServiceUnavailableError("Memory service not initialized")

    body = await request.json()
    memories = body.get("memories", [])

    if not memories:
        raise HTTPException(status_code=400, detail="No memories provided")

    result = await memory_manager.upload_memories(memories)
    return result


@app.post("/api/v1/memories/search")
async def search_memories(
    request: Request,
    permissions: list[str] = Depends(require_auth),
):
    """
    搜索记忆 (需要 read 权限)

    请求体示例:
    {
        "query": "search text",
        "mode": "hybrid",
        "limit": 10,
        "threshold": 0.7
    }
    """
    # 权限检查
    if Permission.READ not in permissions and Permission.ADMIN not in permissions:
        raise PermissionDeniedError("Requires read permission")
    if not memory_manager:
        raise ServiceUnavailableError("Memory service not initialized")

    body = await request.json()
    query = body.get("query", "")
    mode = body.get("mode", "hybrid")
    limit = body.get("limit", 10)
    threshold = body.get("threshold", 0.7)

    if not query:
        raise HTTPException(status_code=400, detail="Query is required")

    results = await memory_manager.search_memories(query, mode, limit, threshold)
    return {"results": results, "total": len(results)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings.port)  # nosec B104
