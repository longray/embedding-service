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
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from surrealdb import AsyncSurreal

from .config import config
from .utils.auth import verify_websocket_token
from .utils.cache import ThreadSafeLRUCache, hash_text
from .utils.exceptions import ValidationError, WrapperServiceError
from .utils.http_pool import close_http_pool, get_http_pool
from .utils.meili_client import MeilisearchClient
from .utils.memory_manager import MemoryManager
from .utils.tracing import init_tracing, shutdown_tracing

logger = logging.getLogger(__name__)


# ==================== 全局状态 ====================

embedding_cache: ThreadSafeLRUCache | None = None
memory_manager: MemoryManager | None = None
meili_client: MeilisearchClient | None = None


# ==================== 数据模型 ====================


class EmbeddingRequest(BaseModel):
    input: str = Field(..., description="要嵌入的文本")
    model: str = Field(default="Qwen3-Embedding-0.6B", description="模型名称")


class MemoryItem(BaseModel):
    content: str = Field(..., min_length=1, description="记忆内容 (L2)")
    abstract: str | None = Field(default=None, description="摘要 (L0, ≤100字符)")
    overview: str | None = Field(default=None, description="概览 (L1, ≤500字符)")
    type: str = Field(default="general", description="记忆类型")
    tags: list[str] = Field(default_factory=list, description="标签列表")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元数据")
    project_id: str = Field(default="global", description="项目ID")
    source: str = Field(default="api", description="来源")
    source_id: str | None = Field(default=None, description="来源ID")
    local_id: str | None = Field(default=None, description="插件端本地ID (ULID)")
    source_timestamp: str | None = Field(default=None, description="来源时间戳")
    classification_confidence: float | None = Field(default=None, description="分类置信度")


class MemoryUploadRequest(BaseModel):
    memories: list[MemoryItem] = Field(..., description="记忆列表")
    tenant_id: str = Field(default="default", description="租户ID")


class MemorySearchRequest(BaseModel):
    query: str = Field(..., description="搜索查询")
    mode: str = Field(default="hybrid", description="搜索模式")
    limit: int = Field(default=10, ge=1, le=100)
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    level: int = Field(default=2, ge=0, le=2, description="返回层级: 0=abstract, 1=abstract+overview, 2=full")
    tenant_id: str = Field(default="default", description="租户ID")


class RelationCreateRequest(BaseModel):
    from_id: str = Field(..., description="源记忆 ID")
    to_id: str = Field(..., description="目标记忆 ID")
    relationship_type: str = Field(default="related", description="关系类型")
    weight: float = Field(default=0.5, ge=0.0, le=1.0, description="关系权重")
    tenant_id: str = Field(default="default", description="租户ID")
    description: str | None = Field(default=None, description="关系描述")


class RelationQueryRequest(BaseModel):
    direction: str = Field(default="both", description="查询方向 (outgoing/incoming/both)")
    relationship_type: str | None = Field(default=None, description="按关系类型过滤")
    tenant_id: str = Field(default="default", description="租户ID")
    limit: int = Field(default=50, ge=1, le=200)


class GraphTraversalRequest(BaseModel):
    depth: int = Field(default=1, ge=1, le=3, description="遍历深度")
    relationship_type: str | None = Field(default=None, description="按关系类型过滤")
    tenant_id: str = Field(default="default", description="租户ID")
    limit: int = Field(default=20, ge=1, le=100)


# ==================== Sync Data Models (Phase B) ====================


class SyncFingerprint(BaseModel):
    """文件指纹模型，用于增量同步"""

    path: str = Field(..., description="文件路径")
    mtime: int = Field(..., description="修改时间戳（毫秒）")
    hash: str = Field(..., description="文件内容哈希（MD5）")
    source_id: str = Field(..., description="记忆唯一标识")


class SyncPreviewRequest(BaseModel):
    """同步预览请求"""

    fingerprints: list[SyncFingerprint] = Field(..., description="本地文件指纹列表")
    tenant_id: str = Field(default="default", description="租户ID")


class SyncPreviewResponse(BaseModel):
    """同步预览响应"""

    synced: int = Field(default=0, description="成功同步数量")
    to_upload: list[dict] = Field(default_factory=list, description="需要上传的条目")
    to_delete: list[str] = Field(default_factory=list, description="需要删除的source_id列表")
    conflicts: list[dict] = Field(default_factory=list, description="冲突列表")


class SyncIncrementalRequest(BaseModel):
    """增量同步请求（已弃用，请使用 SyncPreviewRequest）"""

    fingerprints: list[SyncFingerprint] = Field(..., description="本地文件指纹列表")
    tenant_id: str = Field(default="default", description="租户ID")


class SyncIncrementalResponse(BaseModel):
    """增量同步响应（已弃用，请使用 SyncPreviewResponse）"""

    synced: int = Field(default=0, description="成功同步数量")
    to_upload: list[dict] = Field(default_factory=list, description="需要上传的条目")
    to_delete: list[str] = Field(default_factory=list, description="需要删除的source_id列表")
    conflicts: list[dict] = Field(default_factory=list, description="冲突列表")


class SyncFullRequest(BaseModel):
    """全量同步请求"""

    memories: list[MemoryItem] = Field(..., description="记忆列表")
    tenant_id: str = Field(default="default", description="租户ID")


class SyncFullResponse(BaseModel):
    total: int = Field(..., description="总数")
    success: int = Field(..., description="成功数")
    failed: int = Field(..., description="失败数")
    updated: int = Field(default=0, description="更新数")
    skipped: list[dict] = Field(default_factory=list, description="去重跳过的条目")
    errors: list[str] = Field(default_factory=list, description="错误列表")


class ConflictResolutionRequest(BaseModel):
    """冲突解决请求"""

    resolution: str = Field(..., description="解决策略: use_local | use_remote | keep_both")
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

    # 目标 Schema 版本（与 init_surrealdb.surql 中的 UPSERT 保持一致）
    SCHEMA_TARGET_VERSION = "2.3.0"

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

            # 检查当前版本
            current_version = await self._get_current_schema_version()
            if current_version == self.SCHEMA_TARGET_VERSION:
                logger.info("[Schema] 当前版本: %s（已是最新）", current_version)
                return

            # 需要初始化或升级
            action = "升级" if current_version else "首次初始化"
            logger.info(
                "[Schema] %s: %s -> %s，执行 init_surrealdb.surql...",
                action,
                current_version or "(none)",
                self.SCHEMA_TARGET_VERSION,
            )
            await self._apply_init_script()
            logger.info("[Schema] %s完成", action)

            # Phase 3C: 切换到运行时用户凭据（安全加固）
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
            # fail-fast：Schema 初始化失败必须终止服务
            logger.critical("[Schema] 初始化失败，服务��法启动: %s", e)
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
    global embedding_cache, memory_manager, meili_client

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

    # Meilisearch 初始化（可选，失败不影响服务启动，回退到纯 SurrealDB 搜索）
    if config.meilisearch.enabled:
        try:
            meili_client = MeilisearchClient(
                url=config.meilisearch.url,
                api_key=config.meilisearch.api_key,
                index_name=config.meilisearch.index_name,
                timeout=config.meilisearch.timeout,
            )
            await meili_client.connect()
            await meili_client.ensure_index()
            await meili_client.configure_index()
            memory_manager.set_meili_client(meili_client)
            print("[Startup] Meilisearch已连接并配置")
        except Exception as e:
            logger.warning("[Startup] Meilisearch 初始化失败，回退到 SurrealDB 搜索: %s", e)
            meili_client = None

    # OpenTelemetry 追踪（可选，失败不影响服务启动）
    init_tracing(app, config.telemetry)

    yield

    print("[Shutdown] 关闭服务...")
    shutdown_tracing()
    if meili_client:
        await meili_client.close()
    await close_http_pool()
    await db_manager.disconnect()

    app = FastAPI(title="Minimal Wrapper Service", version="2.4.1", lifespan=lifespan)


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
        "version": "2.4.1",
        "port": config.port,
        "embedding_service": embedding_health or {"status": "unhealthy"},
        "surrealdb": surrealdb_health,
        "meilisearch": (await meili_client.health()) if meili_client else {"status": "disabled"},
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
            [m.model_dump() for m in request.memories],
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
            level=request.level,
            tenant_id=request.tenant_id,
        )
        return result
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.message) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"搜索失败: {e!s}") from e


@app.post("/api/v1/memories/relations")
async def create_relation(request: RelationCreateRequest):
    """创建记忆间的图关系"""
    if not memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        result = await memory_manager.create_relation(
            from_id=request.from_id,
            to_id=request.to_id,
            relationship_type=request.relationship_type,
            weight=request.weight,
            tenant_id=request.tenant_id,
            description=request.description,
        )
        return result
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.message) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建关系失败: {e!s}") from e


@app.delete("/api/v1/memories/clear")
async def clear_memories(request: Request):
    if not memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    api_key = request.headers.get("WRAPPER_MEILI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing WRAPPER_MEILI_API_KEY header")

    if api_key != config.meilisearch.api_key:
        raise HTTPException(status_code=403, detail="Invalid WRAPPER_MEILI_API_KEY")

    client = meili_client
    if not client:
        client = MeilisearchClient(
            url=config.meilisearch.url,
            api_key=config.meilisearch.api_key,
            index_name=config.meilisearch.index_name,
            timeout=config.meilisearch.timeout,
        )
        await client.connect()

    try:
        logger.warning("[Clear] 清空 Meilisearch...")

        await client.delete_all_documents()
        logger.info("[Clear] Meilisearch 已清空")

        logger.warning("[Clear] 清空 SurrealDB...")
        await memory_manager._db.query("DELETE memory;")
        await memory_manager._db.query("DELETE memory_relation;")
        await memory_manager._db.query("DELETE conflict;")

        logger.info("[Clear] SurrealDB 已清空")

        return {"success": True, "message": "所有记忆数据已清空"}
    except Exception as e:
        logger.error(f"[Clear] 清空失败: {e}")
        raise HTTPException(status_code=500, detail=f"清空失败: {e!s}") from e


@app.post("/api/v1/memories/{memory_id}/relations")
async def get_relations(memory_id: str, request: RelationQueryRequest):
    """查询记忆的关联关系"""
    if not memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        result = await memory_manager.get_relations(
            memory_id=memory_id,
            direction=request.direction,
            relationship_type=request.relationship_type,
            tenant_id=request.tenant_id,
            limit=request.limit,
        )
        return {"relations": result, "total": len(result), "memory_id": memory_id}
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.message) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询关系失败: {e!s}") from e


@app.delete("/api/v1/memories/relations/{relation_id}")
async def delete_relation(relation_id: str, tenant_id: str = "default"):
    """删除指定的关系"""
    if not memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        deleted = await memory_manager.delete_relation(
            relation_id=relation_id,
            tenant_id=tenant_id,
        )
        if not deleted:
            raise HTTPException(status_code=404, detail="关系不存在或无权删除")
        return {"deleted": True, "relation_id": relation_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除关系失败: {e!s}") from e


@app.post("/api/v1/memories/{memory_id}/graph")
async def graph_traversal(memory_id: str, request: GraphTraversalRequest):
    """图遍历：获取关联的记忆内容"""
    if not memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        result = await memory_manager.get_related_memories(
            memory_id=memory_id,
            depth=request.depth,
            relationship_type=request.relationship_type,
            tenant_id=request.tenant_id,
            limit=request.limit,
        )
        return {"memories": result, "total": len(result), "source": memory_id, "depth": request.depth}
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.message) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"图遍历失败: {e!s}") from e


# ==================== Sync Routes (Phase B) ====================


@app.post("/api/v1/sync/preview", response_model=SyncPreviewResponse)
async def sync_preview(request: SyncPreviewRequest):
    """同步预览：比对指纹，返回变更指令（不执行上传）"""
    if not memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        result = await memory_manager.sync_preview(
            fingerprints=[f.model_dump() for f in request.fingerprints],
            tenant_id=request.tenant_id,
        )
        return SyncPreviewResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"同步预览失败: {e!s}") from e


@app.post("/api/v1/sync/incremental", response_model=SyncIncrementalResponse)
async def sync_incremental_legacy(request: SyncIncrementalRequest):
    """增量同步（已弃用，请使用 /api/v1/sync/preview）"""
    return await sync_preview(SyncPreviewRequest(**request.model_dump()))


@app.post("/api/v1/sync/full", response_model=SyncFullResponse)
async def sync_full(request: SyncFullRequest):
    """全量同步：上传所有记忆"""
    if not memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        result = await memory_manager.sync_full(
            memories=[m.model_dump() for m in request.memories],
            tenant_id=request.tenant_id,
        )
        return SyncFullResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"全量同步失败: {e!s}") from e


@app.get("/api/v1/sync/fingerprints")
async def get_server_fingerprints(tenant_id: str = "default"):
    """获取服务端所有记忆的指纹"""
    if not memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        fingerprints = await memory_manager.get_fingerprints(tenant_id=tenant_id)
        return {"fingerprints": fingerprints, "count": len(fingerprints)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取指纹失败: {e!s}") from e


@app.post("/api/v1/sync/conflicts/{conflict_id}/resolve")
async def resolve_conflict_endpoint(conflict_id: str, request: ConflictResolutionRequest):
    """解决同步冲突"""
    if not memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        result = await memory_manager.resolve_conflict(
            conflict_id=conflict_id,
            resolution=request.resolution,
            tenant_id=request.tenant_id,
        )
        return {"resolved": True, "action": request.resolution, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"解决冲突失败: {e!s}") from e


# ==================== WebSocket 实时推送 (Phase 3D) ====================


@app.websocket("/ws/memories/live")
async def websocket_live_memories(websocket: WebSocket, tenant_id: str = "default", token: str | None = None):
    """WebSocket 端点：实时推送记忆变更通知

    连接后自动订阅指定租户的 memory 表变更，推送 CREATE/UPDATE/DELETE 通知。
    认证：通过 token 查询参数传递（可选，取决于 WRAPPER_WEBSOCKET_TOKEN 配置）。
    """
    # 认证检查
    if not verify_websocket_token(token):
        await websocket.close(code=1008, reason="Unauthorized")
        logger.warning("[WebSocket] 认证失败，拒绝连接")
        return

    await websocket.accept()
    query_uuid = None

    try:
        db_manager = await SurrealDBManager.get_instance()

        # 启动 LIVE SELECT 查询（过滤租户）
        query_result = await db_manager.db.query(
            "LIVE SELECT * FROM memory WHERE tenant_id = $tenant_id",
            {"tenant_id": tenant_id},
        )
        query_uuid = query_result[0]["result"]

        logger.info("[WebSocket] 客户端已连接，订阅租户: %s, query_uuid: %s", tenant_id, query_uuid)

        # 订阅并转发通知
        async for notification in db_manager.db.subscribe_live(query_uuid):
            await websocket.send_json(notification)

    except WebSocketDisconnect:
        logger.info("[WebSocket] 客户端断开连接")
    except Exception as e:
        logger.error("[WebSocket] 错误: %s", e)
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:  # nosec B110 - 发送错误失败时静默
            pass
    finally:
        # 清理 LIVE 查询
        if query_uuid:
            try:
                await db_manager.db.kill(query_uuid)
                logger.info("[WebSocket] 已停止 LIVE 查询: %s", query_uuid)
            except Exception:  # nosec B110 - kill 失败不影响断开
                pass


# ==================== HNSW Tuning API (Phase C-B1) ====================


@app.get("/api/v1/hnsw/stats")
async def get_hnsw_stats(tenant_id: str = "default"):
    """Get HNSW index statistics and recommendations"""
    if not memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        stats = await memory_manager.get_memory_stats(tenant_id)
        return {
            "status": "success",
            "stats": stats,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取统计失败: {e}") from e


@app.post("/api/v1/hnsw/optimize")
async def optimize_hnsw(tenant_id: str = "default"):
    """Auto-optimize HNSW parameters without rebuilding"""
    if not memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        result = await memory_manager.optimize_hnsw(tenant_id)
        return {
            "status": "success",
            "result": result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"优化失败: {e}") from e


@app.post("/api/v1/hnsw/rebuild")
async def rebuild_hnsw(tenant_id: str = "default", force: bool = False):
    """Rebuild HNSW index with optimal parameters"""
    if not memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        result = await memory_manager.rebuild_hnsw_index(tenant_id, force)
        return {
            "status": "success",
            "result": result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重建失败: {e}") from e


# ==================== Embedding Cache API (Phase C-B2) ====================


@app.get("/api/v1/cache/stats")
async def get_cache_stats():
    """Get embedding cache statistics"""
    if not memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        stats = await memory_manager.get_cache_stats()
        return {"status": "success", "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取缓存统计失败: {e}") from e


@app.post("/api/v1/cache/clear")
async def clear_cache():
    """Clear embedding cache"""
    if not memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        result = await memory_manager.clear_embedding_cache()
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清除缓存失败: {e}") from e


@app.post("/api/v1/cache/warmup")
async def warmup_cache(tenant_id: str = "default", limit: int = 100):
    """Preload embeddings for recent memories into cache"""
    if not memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        result = await memory_manager.warmup_embedding_cache(tenant_id, limit)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预热缓存失败: {e}") from e


# ==================== Prefetch API (Phase C-B3) ====================


@app.post("/api/v1/prefetch/related")
async def prefetch_related(memory_id: str, tenant_id: str = "default", depth: int = 1, limit: int = 10):
    """Prefetch embeddings for memories related to the given memory"""
    if not memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        result = await memory_manager.prefetch_related_memories(memory_id, tenant_id, depth, limit)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预取失败: {e}") from e


@app.post("/api/v1/prefetch/popular")
async def prefetch_popular(tenant_id: str = "default", top_n: int = 20):
    """Prefetch embeddings for popular memories"""
    if not memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        result = await memory_manager.prefetch_popular_queries(tenant_id, top_n)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预取失败: {e}") from e


# ==================== Advanced Analysis API ====================


@app.post("/api/v1/memories/{memory_id}/analyze/code")
async def analyze_memory_code(memory_id: str, tenant_id: str = "default"):
    """Analyze code content in a memory"""
    if not memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        result = await memory_manager.analyze_memory_code(memory_id, tenant_id)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"代码分析失败: {e}") from e


@app.post("/api/v1/memories/cluster/leiden")
async def cluster_memories_leiden(tenant_id: str = "default", content_threshold: float = 0.75, max_clusters: int = 20):
    """Cluster memories using Leiden algorithm"""
    if not memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        result = await memory_manager.cluster_memories_leiden(
            tenant_id=tenant_id, content_threshold=content_threshold, max_clusters=max_clusters
        )
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"聚类分析失败: {e}") from e


# ==================== Advanced Analysis API ====================


@app.post("/api/v1/memories/{memory_id}/analyze/code")
async def analyze_memory_code(memory_id: str, tenant_id: str = "default"):
    """Analyze code content in a memory"""
    if not memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        result = await memory_manager.analyze_memory_code(memory_id, tenant_id)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"代码分析失败: {e}") from e


@app.post("/api/v1/memories/cluster/leiden")
async def cluster_memories_leiden(tenant_id: str = "default", content_threshold: float = 0.75, max_clusters: int = 20):
    """Cluster memories using Leiden algorithm"""
    if not memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        result = await memory_manager.cluster_memories_leiden(
            tenant_id=tenant_id, content_threshold=content_threshold, max_clusters=max_clusters
        )
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"聚类分析失败: {e}") from e


# ==================== v2.4.0 L0/L1/L2 Access Log API ====================


class AccessLogEntry(BaseModel):
    entry_id: str = Field(..., description="记忆ID")
    timestamp: str = Field(..., description="访问时间戳")
    type: str = Field(default="read", description="访问类型")


class AccessLogRequest(BaseModel):
    entries: list[AccessLogEntry] = Field(..., description="访问日志条目列表")
    tenant_id: str = Field(default="default", description="租户ID")


@app.post("/api/v1/access-log")
async def report_access_log(request: AccessLogRequest):
    """接收访问日志（用于分析记忆使用频率）"""
    if not memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        result = await memory_manager.report_access_log(
            [entry.model_dump() for entry in request.entries],
            tenant_id=request.tenant_id,
        )
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"记录访问日志失败: {e}") from e


if __name__ == "__main__":
    uvicorn.run(app, host=config.host, port=config.port)
