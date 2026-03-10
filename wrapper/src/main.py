"""
最小化包装服务主程序

使用 SurrealDB 长期连接 + FastAPI lifespan 管理
集成缓存和HTTP连接池，不使用熔断器。
支持 Schema 自动初始化和多租户隔离。
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from surrealdb import AsyncSurreal

from .config import config
from .utils.cache import ThreadSafeLRUCache, hash_text
from .utils.exceptions import ValidationError, WrapperServiceError
from .utils.http_pool import close_http_pool, get_http_pool
from .utils.memory_manager import MemoryManager

logger = logging.getLogger(__name__)


# ==================== 全局状态 ====================

embedding_cache: ThreadSafeLRUCache | None = None
memory_manager: MemoryManager | None = None


# ==================== 数据模型 ====================


class EmbeddingRequest(BaseModel):
    input: str = Field(..., description="要嵌入的文本")
    model: str = Field(default="Qwen3-Embedding-0.6B", description="模型名称")


class MemoryUploadRequest(BaseModel):
    memories: list[dict] = Field(..., description="记忆列表")
    tenant_id: str = Field(default="default", description="租户ID")


class MemorySearchRequest(BaseModel):
    query: str = Field(..., description="搜索查询")
    mode: str = Field(default="hybrid", description="搜索模式")
    limit: int = Field(default=10, ge=1, le=100)
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    tenant_id: str = Field(default="default", description="租户ID")


# ==================== SurrealDB 管理器 ====================


class SurrealDBManager:
    _instance = None
    _db: Any = None  # AsyncSurreal SDK 返回联合类型，使用 Any 避免类型检查误报
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
            self._db = AsyncSurreal(config.surrealdb.url)
            await self._db.connect()
            await self._db.signin(
                {
                    "username": config.surrealdb.username,
                    "password": config.surrealdb.password,
                }
            )
            await self._db.use(config.surrealdb.namespace, config.surrealdb.database)

    async def disconnect(self):
        if self._db:
            await self._db.close()
            self._db = None

    @property
    def db(self):
        if self._db is None:
            raise RuntimeError("数据库未连接")
        return self._db

    async def _db_query(self, sql: str, params: dict[str, Any] | None = None) -> Any:
        """执行 SurrealQL 查询的辅助方法"""
        if params:
            return await self.db.query(sql, params)
        return await self.db.query(sql)

    # ==================== Schema 初始化 ====================

    async def ensure_schema(self):
        """确保数据库 Schema 已初始化（幂等操作 + migration lock + fail-fast）

        调用时机：lifespan 启动阶段，connect() 之后、MemoryManager 初始化之前。
        Schema 初始化失败将导致服务直接退出（SystemExit(1)），确保不会在残缺状态下接受请求。
        """
        lock_acquired = False
        try:
            lock_acquired = await self._acquire_migration_lock()
            if not lock_acquired:
                logger.info("[Schema] 其他实例正在执行 migration，跳过")
                return

            # 检查是否已初始化
            result = await self._db_query("SELECT * FROM schema_version ORDER BY applied_at DESC LIMIT 1")
            if result and isinstance(result, list) and len(result) > 0:
                version = "unknown"
                if isinstance(result[0], dict):
                    version = result[0].get("version", "unknown")
                elif isinstance(result[0], list) and len(result[0]) > 0:
                    version = result[0][0].get("version", "unknown")
                logger.info("[Schema] 当前版本: %s", version)
                return

            # 首次初始化
            logger.info("[Schema] 首次初始化，执行 init_surrealdb.surql...")
            init_script = Path(__file__).parent.parent / "scripts" / "init_surrealdb.surql"

            # 兼容项目根目录的 scripts/ 和 wrapper/scripts/
            if not init_script.exists():
                init_script = Path(__file__).parent.parent.parent / "scripts" / "init_surrealdb.surql"

            if not init_script.exists():
                raise FileNotFoundError(f"初始化脚本不存在: {init_script}")

            sql = init_script.read_text(encoding="utf-8")

            # 拆分为单条语句逐条执行（query() 对多语句只返回最后结果）
            statements = [s.strip() for s in sql.split(";") if s.strip()]
            for stmt in statements:
                # 跳过纯注释块
                lines = [line for line in stmt.split("\n") if not line.strip().startswith("--")]
                if not any(line.strip() for line in lines):
                    continue
                await self._db_query(stmt)

            logger.info("[Schema] 初始化完成")

        except SystemExit:
            raise
        except Exception as e:
            # fail-fast：Schema 初始化失败必须终止服务
            logger.critical("[Schema] 初始化失败，服务无法启动: %s", e)
            raise SystemExit(1) from e
        finally:
            if lock_acquired:
                await self._release_migration_lock()

    async def _acquire_migration_lock(self) -> bool:
        """获取 migration 锁（基于 SurrealDB 记录）

        使用 UPSERT + WHERE + TTL 实现：
        - UPSERT: 幂等，不抛 AlreadyExistsError
        - WHERE: 只在未锁定或锁已过期时获取
        - TTL (locked_until): 防止进程崩溃后永久死锁
        """
        try:
            result = await self._db_query(
                "UPSERT migration_lock:global SET "
                "  locked = true, "
                "  locked_by = $instance_id, "
                "  locked_at = time::now(), "
                "  locked_until = time::now() + 5m "
                "WHERE locked = false OR locked_until < time::now()",
                {"instance_id": str(id(self))},
            )
            # UPSERT + WHERE 返回空列表表示条件不满足（锁被持有且未过期）
            return bool(result and len(result) > 0)
        except ConnectionError:
            logger.error("[Schema] 无法连接 SurrealDB，获取锁失败")
            raise
        except Exception as e:
            logger.warning("[Schema] 获取锁异常: %s", e)
            return False

    async def _release_migration_lock(self):
        """释放 migration 锁"""
        try:
            await self._db_query("UPDATE migration_lock:global SET locked = false")
        except Exception:  # nosec B110 # noqa: S110 - 锁释放失败不影响正常流程（TTL 会自动过期）
            pass


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
        print("[Startup] 缓存已启用")

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

    # Schema 自动初始化（在 MemoryManager 之前，失败则服务退出）
    await db_manager.ensure_schema()
    print("[Startup] Schema已验证")

    memory_manager = MemoryManager(
        db=db_manager.db,
        embedding_service_url=config.service.embedding_service_url,
        search_config=config.search,
    )
    print("[Startup] MemoryManager已初始化")

    yield

    print("[Shutdown] 关闭服务...")
    await close_http_pool()
    await db_manager.disconnect()


app = FastAPI(title="Minimal Wrapper Service", version="2.0.0", lifespan=lifespan)


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
        raise HTTPException(status_code=502, detail=f"Embedding服务错误: {e!s}") from e


@app.post("/api/v1/memories")
async def upload_memories(request: MemoryUploadRequest):
    if not memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        result = await memory_manager.upload_memories(
            request.memories,
            tenant_id=request.tenant_id,
        )
        return result
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.message) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"上传失败: {e!s}") from e


@app.post("/api/v1/memories/search")
async def search_memories(request: MemorySearchRequest):
    if not memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        result = await memory_manager.search_memories(
            query=request.query,
            mode=request.mode,
            limit=request.limit,
            threshold=request.threshold,
            tenant_id=request.tenant_id,
        )
        return result
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.message) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"搜索失败: {e!s}") from e


if __name__ == "__main__":
    uvicorn.run(app, host=config.host, port=config.port)
