"""
最小化包装服务主程序

使用 SurrealDB 长期连接 + FastAPI lifespan 管理
集成缓存和HTTP连接池，不使用熔断器。
支持 Schema 自动初始化/升级、多租户隔离和图关系操作。
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from surrealdb import AsyncSurreal

from . import state
from .config import config
from .models import (  # noqa: F401 — re-exported for backward compatibility
    ConflictResolutionRequest,
    EmbeddingRequest,
    MemoryItem,
    MemorySearchRequest,
    MemoryUploadRequest,
    RelationCreateRequest,
    RelationQueryRequest,
    SyncFingerprint,
    SyncFullRequest,
    SyncFullResponse,
    SyncIncrementalRequest,
    SyncIncrementalResponse,
    SyncPreviewRequest,
    SyncPreviewResponse,
)
from .routers import audit, embeddings, health, lookup, memories, projects, relations, search, stubs, sync, websocket
from .utils.cache import ThreadSafeLRUCache
from .utils.exceptions import WrapperServiceError
from .utils.http_pool import close_http_pool, get_http_pool
from .utils.meili_client import MeilisearchClient
from .utils.memory_manager import MemoryManager
from .utils.tracing import init_tracing, shutdown_tracing

logger = logging.getLogger(__name__)


# ==================== SurrealDB 管理器 ====================


class SurrealDBManager:
    _instance = None
    _db: Any = None
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

    async def reconnect(self):
        """重新认证 SurrealDB 会话（用于 SessionExpired 自动恢复）"""
        logger.info("[SurrealDBManager] 重新认证会话...")
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
        logger.info("[SurrealDBManager] 会话重新认证成功")

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

    SCHEMA_TARGET_VERSION = "2.4.1"

    async def ensure_schema(self):
        """确保数据库 Schema 已初始化或已升级（幂等操作 + migration lock + fail-fast）

        调用时机：lifespan 启动阶段，connect() 之后、MemoryManager 初始化之前。
        Schema 初始化失败将导致服务直接退出（SystemExit(1)），确保不会在残缺状态下接受请求。
        """
        lock_acquired = False
        try:
            lock_acquired = await self._acquire_migration_lock()
            if not lock_acquired:
                logger.info("[Schema] 其他实例正在执行 migration，跳过")
                return

            current_version = await self._get_current_schema_version()
            if current_version == self.SCHEMA_TARGET_VERSION:
                logger.info("[Schema] 当前版本: %s（已是最新）", current_version)
                return

            action = "升级" if current_version else "首次初始化"
            logger.info(
                "[Schema] %s: %s -> %s，执行 init_surrealdb.surql...",
                action,
                current_version or "(none)",
                self.SCHEMA_TARGET_VERSION,
            )
            await self._apply_init_script()
            logger.info("[Schema] %s完成", action)

            if config.surrealdb.use_runtime_credentials:
                logger.info("[Security] 切换到运行时用户凭据（EDITOR 权限）")
                await self._db.signin(
                    {
                        "namespace": config.surrealdb.namespace,
                        "database": config.surrealdb.database,
                        "username": config.surrealdb.runtime_username,
                        "password": config.surrealdb.runtime_password,
                    }
                )
                logger.info("[Security] 已切换到运行时用户: %s", config.surrealdb.runtime_username)

        except SystemExit:
            raise
        except Exception as e:
            logger.critical("[Schema] 初始化失败，服务无法启动: %s", e)
            raise SystemExit(1) from e
        finally:
            if lock_acquired:
                await self._release_migration_lock()

    async def _get_current_schema_version(self) -> str | None:
        """获取当前 Schema 版本，返回 None 表示未初始化"""
        result = await self._db_query("SELECT * FROM schema_version ORDER BY applied_at DESC LIMIT 1")
        if result and isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], dict):
                return result[0].get("version")
            if isinstance(result[0], list) and len(result[0]) > 0:
                return result[0][0].get("version")
        return None

    async def _apply_init_script(self) -> None:
        """执行 init_surrealdb.surql 脚本（支持初始化和升级）

        脚本设计为幂等：
        - 普通定义使用 IF NOT EXISTS，安全重复执行
        - Analyzer/Index 使用 REMOVE + DEFINE，确保升级生效
        - UPSERT 版本号确保最终一致性
        """
        init_script = Path(__file__).parent.parent / "scripts" / "init_surrealdb.surql"

        if not init_script.exists():
            init_script = Path(__file__).parent.parent.parent / "scripts" / "init_surrealdb.surql"

        if not init_script.exists():
            raise FileNotFoundError(f"初始化脚本不存在: {init_script}")

        sql = init_script.read_text(encoding="utf-8")

        statements = [s.strip() for s in sql.split(";") if s.strip()]
        for stmt in statements:
            lines = [line for line in stmt.split("\n") if not line.strip().startswith("--")]
            if not any(line.strip() for line in lines):
                continue
            await self._db_query(stmt)

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
    print("[Startup] 初始化服务...")

    if config.cache.enabled:
        state.embedding_cache = ThreadSafeLRUCache(
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

    await db_manager.ensure_schema()
    print("[Startup] Schema已验证")

    state.memory_manager = MemoryManager(
        db=db_manager.db,
        embedding_service_url=config.service.embedding_service_url,
        search_config=config.search,
        reauthenticate_fn=db_manager.reconnect,
    )
    print("[Startup] MemoryManager已初始化")

    if config.meilisearch.enabled:
        try:
            state.meili_client = MeilisearchClient(
                url=config.meilisearch.url,
                api_key=config.meilisearch.api_key,
                index_name=config.meilisearch.index_name,
                timeout=config.meilisearch.timeout,
            )
            await state.meili_client.connect()
            await state.meili_client.ensure_index()
            await state.meili_client.configure_index()
            state.memory_manager.set_meili_client(state.meili_client)
            print("[Startup] Meilisearch已连接并配置")
        except Exception as e:
            logger.warning("[Startup] Meilisearch 初始化失败，回退到 SurrealDB 搜索: %s", e)
            state.meili_client = None

    init_tracing(app, config.telemetry)

    yield

    print("[Shutdown] 关闭服务...")
    shutdown_tracing()
    if state.meili_client:
        await state.meili_client.close()
    await close_http_pool()
    await db_manager.disconnect()


# ==================== FastAPI App ====================

app = FastAPI(title="Minimal Wrapper Service", version="2.6.0", lifespan=lifespan)

app.include_router(health.router)
app.include_router(embeddings.router)
app.include_router(lookup.router)  # 必须在 memories 之前
app.include_router(memories.router)
app.include_router(search.router)
app.include_router(relations.router)
app.include_router(projects.router)
app.include_router(audit.router)
app.include_router(sync.router)
app.include_router(websocket.router)
app.include_router(stubs.router)


# ==================== 异常处理 ====================


@app.exception_handler(WrapperServiceError)
async def wrapper_exception_handler(request: Request, exc: WrapperServiceError):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.message, "details": exc.details})


if __name__ == "__main__":
    uvicorn.run(app, host=config.host, port=config.port)
