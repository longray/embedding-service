"""
最小化包装服务主程序

使用 SurrealDB 长期连接 + FastAPI lifespan 管理
集成缓存和HTTP连接池，不使用熔断器。
"""

import asyncio
import inspect
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from surrealdb import Surreal
import uvicorn

from .config import config
from .utils.cache import ThreadSafeLRUCache, hash_text
from .utils.http_pool import get_http_pool, close_http_pool
from .utils.memory_manager import MemoryManager
from .utils.exceptions import WrapperServiceError, ValidationError


# ==================== 全局状态 ====================

embedding_cache: Optional[ThreadSafeLRUCache] = None
memory_manager: Optional[MemoryManager] = None


# ==================== 数据模型 ====================


class EmbeddingRequest(BaseModel):
    input: str = Field(..., description="要嵌入的文本")
    model: str = Field(default="Qwen3-Embedding-0.6B", description="模型名称")


class MemoryUploadRequest(BaseModel):
    memories: list[dict] = Field(..., description="记忆列表")


class MemorySearchRequest(BaseModel):
    query: str = Field(..., description="搜索查询")
    mode: str = Field(default="hybrid", description="搜索模式")
    limit: int = Field(default=10, ge=1, le=100)
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    metadata_filters: Optional[dict[str, Any]] = Field(default=None, description="元数据精确过滤条件")


# ==================== SurrealDB 管理器 ====================


class SurrealDBManager:
    _instance = None
    _db: Optional[Any] = None
    _lock = asyncio.Lock()

    @classmethod
    async def get_instance(cls):
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    async def connect(self):
        if self._db is None:
            db: Any = Surreal(config.surrealdb.url)
            if hasattr(db, "connect"):
                connect_result = db.connect(config.surrealdb.url)
                if inspect.isawaitable(connect_result):
                    await connect_result
            signin_result = db.signin(
                {
                    "username": config.surrealdb.username,
                    "password": config.surrealdb.password,
                }
            )
            if inspect.isawaitable(signin_result):
                await signin_result
            use_result = db.use(config.surrealdb.namespace, config.surrealdb.database)
            if inspect.isawaitable(use_result):
                await use_result
            self._db = db

    async def disconnect(self):
        if self._db:
            close_result = self._db.close()
            if inspect.isawaitable(close_result):
                await close_result
            self._db = None

    @property
    def db(self):
        if self._db is None:
            raise RuntimeError("数据库未连接")
        return self._db


# ==================== FastAPI 生命周期 ====================


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_cache, memory_manager

    print("[Startup] 初始化服务...")

    if config.cache.enabled:
        embedding_cache = ThreadSafeLRUCache(
            max_size=config.cache.max_size,
            ttl_seconds=config.cache.ttl_seconds,
        )
        print(f"[Startup] 缓存已启用")

    await get_http_pool(
        max_connections=config.http.max_connections,
        max_keepalive_connections=config.http.max_keepalive_connections,
        timeout=config.http.timeout,
        connect_timeout=config.http.connect_timeout,
        max_retries=config.http.max_retries,
    )
    print("[Startup] HTTP连接池已初始化")

    db_manager = await SurrealDBManager.get_instance()
    await db_manager.connect()
    print("[Startup] SurrealDB已连接")

    memory_manager = MemoryManager(
        db=db_manager.db,
        embedding_service_url=config.service.embedding_service_url,
    )
    print("[Startup] MemoryManager已初始化")

    yield

    print("[Shutdown] 关闭服务...")
    await close_http_pool()
    await db_manager.disconnect()


app = FastAPI(title="Minimal Wrapper Service", version="2.0.0", lifespan=lifespan)


@app.middleware("http")
async def access_log_middleware(request: Request, call_next):
    response = await call_next(request)
    if request.url.path == "/favicon.ico":
        return response
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    http_version = request.scope.get("http_version", "1.1")
    print(f'[{timestamp}] "{request.method} {request.url.path} HTTP/{http_version}" {response.status_code}')
    return response


# ==================== 异常处理 ====================


@app.exception_handler(WrapperServiceError)
async def wrapper_exception_handler(request: Request, exc: WrapperServiceError):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.message, "details": exc.details})


# ==================== 健康检查 ====================


async def check_embedding_service_health():
    try:
        http_pool = await get_http_pool()
        response = await http_pool.get(
            f"{config.service.embedding_service_url}/health",
            timeout=2.0,
        )
        if response.status_code == 200:
            return response.json()
    except Exception:  # nosec B110 - 健康检查失败时静默返回 None
        return None
    return None


async def check_surrealdb_health():
    try:
        db_manager = await SurrealDBManager.get_instance()
        await db_manager.db.query("SELECT * FROM $version")
        return {"status": "healthy"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


# ==================== API 端点 ====================


@app.get("/health")
async def health_check():
    embedding_health = await check_embedding_service_health()
    surrealdb_health = await check_surrealdb_health()

    result = {
        "status": "healthy",
        "service": "minimal-wrapper",
        "version": "2.0.0",
        "port": config.port,
        "embedding_service": embedding_health or {"status": "unhealthy"},
        "surrealdb": surrealdb_health,
        "search_thresholds": {
            "vector": config.search.vector_threshold,
            "hybrid": config.search.hybrid_threshold,
            "keyword": config.search.keyword_threshold,
        },
    }

    if embedding_cache:
        result["cache_stats"] = embedding_cache.get_stats()

    return result


@app.post("/v1/embeddings")
async def create_embedding(request: EmbeddingRequest):
    global embedding_cache

    cache_key = hash_text(request.input)

    if embedding_cache:
        cached = embedding_cache.get(cache_key)
        if cached:
            return cached

    try:
        http_pool = await get_http_pool()
        response = await http_pool.post(
            f"{config.service.embedding_service_url}/v1/embeddings",
            json={"input": request.input, "model": request.model},
        )
        response.raise_for_status()
        data = response.json()

        if embedding_cache:
            embedding_cache.set(cache_key, data)

        return data

    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Embedding服务错误: {str(e)}")


@app.post("/api/v1/memories")
async def upload_memories(request: MemoryUploadRequest):
    if not memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        result = await memory_manager.upload_memories(request.memories)
        return result
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")


@app.post("/api/v1/memories/search")
async def search_memories(request: MemorySearchRequest):
    if not memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        if request.threshold is not None:
            resolved_threshold = request.threshold
        elif request.mode == "vector":
            resolved_threshold = config.search.vector_threshold
        elif request.mode == "hybrid":
            resolved_threshold = config.search.hybrid_threshold
        else:
            resolved_threshold = config.search.keyword_threshold

        result = await memory_manager.search_memories(
            query=request.query,
            mode=request.mode,
            limit=request.limit,
            threshold=resolved_threshold,
            metadata_filters=request.metadata_filters,
        )
        return result
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host=config.host, port=config.port, access_log=False)
