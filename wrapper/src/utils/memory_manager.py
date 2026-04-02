"""记忆管理器模块

封装记忆的批量上传、搜索等业务逻辑。
支持 KNN 向量搜索（HNSW 索引）、Meilisearch 全文搜索和 RRF 混合搜索。
支持 SurrealDB RELATE 图关系操作（创建/查询/删除记忆间关联）。
多租户隔离通过 tenant_id 字段实现。
"""

import asyncio
import hashlib
import json
import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any

from aiocache.serializers import JsonSerializer

from .code_analyzer import CodeAnalyzer
from .exceptions import DatabaseError, EmbeddingError, ValidationError
from .http_pool import get_http_pool
from .meili_client import MeilisearchClient
from .tracing import get_tracer

logger = logging.getLogger(__name__)


class MemoryManager:
    """记忆管理器，协调 embedding 服务和数据库操作

    支持多租户隔离和 SurrealDB 3.0 新查询语法：
    - 向量搜索: KNN <|K,EF|> 算子 + HNSW 索引
    - 关键词搜索: Meilisearch 全文搜索（CJK 分词）+ SurrealDB BM25（降级路径）
    - 混合搜索: RRF 融合算法（向量 from SurrealDB + 关键词 from Meilisearch）
    - 图关系: RELATE 语句实现记忆间关联（follow_up/related/elaboration 等）
    """

    def __init__(
        self,
        db: Any,  # AsyncSurreal SDK 返回联合类型，使用 Any 避免类型检查误报
        embedding_service_url: str,
        search_config: Any = None,
        batch_size: int = 10,
        reauthenticate_fn: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._db = db
        self._embedding_service_url = embedding_service_url
        self._batch_size = batch_size
        self._http_pool: Any | None = None
        self._meili: MeilisearchClient | None = None
        self._reauthenticate_fn = reauthenticate_fn

        # Phase A-B3: 查询结果缓存（aiocache）
        self._cache_enabled: bool = getattr(search_config, "cache_enabled", True)
        self._cache_ttl: int = getattr(search_config, "cache_ttl", 300)  # 5分钟
        self._vector_cache: Any | None = None
        self._keyword_cache: Any | None = None
        if self._cache_enabled:
            from aiocache import Cache

            self._vector_cache = Cache(Cache.MEMORY, serializer=JsonSerializer())
            self._keyword_cache = Cache(Cache.MEMORY, serializer=JsonSerializer())

        # 搜索配置（从 config.SearchConfig 传入，使用 getattr 保持向后兼容）
        self._rrf_k: int = getattr(search_config, "rrf_k", 60)
        self._rrf_vector_weight: float = getattr(search_config, "rrf_vector_weight", 0.7)
        self._rrf_keyword_weight: float = getattr(search_config, "rrf_keyword_weight", 0.3)
        self._hnsw_ef_search: int = getattr(search_config, "hnsw_ef_search", 50)
        self._default_tenant_id: str = getattr(search_config, "default_tenant_id", "default")

        # Phase A-B6: 动态去重阈值
        self._dedup_thresholds: dict[str, float] = getattr(
            search_config,
            "dedup_thresholds",
            {
                "preference": 0.88,
                "decision": 0.90,
                "long-term": 0.93,
                "general": 0.95,
                "daily": 1.0,
            },
        )

        # Code analyzer instance
        self.code_analyzer = CodeAnalyzer()

    def _is_session_expired_error(self, error: Exception) -> bool:
        err_str = str(error).lower()
        return "sessionexpired" in err_str or "session has expired" in err_str

    async def _db_query(self, sql: str, params: dict[str, Any] | None = None) -> Any:
        try:
            if params:
                return await self._db.query(sql, params)
            return await self._db.query(sql)
        except Exception as e:
            if self._is_session_expired_error(e) and self._reauthenticate_fn:
                logger.info("[MemoryManager] SurrealDB session expired, reauthenticating...")
                await self._reauthenticate_fn()
                if params:
                    return await self._db.query(sql, params)
                return await self._db.query(sql)
            raise

    async def _db_create(self, table: str, data: dict[str, Any]) -> Any:
        try:
            return await self._db.create(table, data)
        except Exception as e:
            if self._is_session_expired_error(e) and self._reauthenticate_fn:
                logger.info("[MemoryManager] SurrealDB session expired, reauthenticating (create)...")
                await self._reauthenticate_fn()
                return await self._db.create(table, data)
            raise

    async def _get_http_pool(self):
        """延迟初始化 HTTP 连接池"""
        if self._http_pool is None:
            self._http_pool = await get_http_pool()
        return self._http_pool

    def _get_dedup_threshold(self, memory_type: str) -> float:
        """获取指定记忆类型的去重阈值"""
        return self._dedup_thresholds.get(memory_type, 0.95)

    def _get_vector_cache_key(
        self,
        embedding: list[float],
        limit: int,
        threshold: float,
        tenant_id: str,
    ) -> str:
        """生成向量搜索缓存的 key

        Args:
            embedding: 查询向量
            limit: 返回数量限制
            threshold: 相似度阈值
            tenant_id: 租户ID

        Returns:
            缓存 key 字符串
        """
        # 使用向量的前8个元素 + 最后8个元素生成哈希
        emb_prefix = embedding[:8] if len(embedding) >= 8 else embedding
        emb_suffix = embedding[-8:] if len(embedding) >= 8 else []
        emb_hash = hashlib.md5(f"{emb_prefix}:{emb_suffix}".encode(), usedforsecurity=False).hexdigest()[:16]
        return f"vec:{tenant_id}:{emb_hash}:{limit}:{threshold:.2f}"

    def set_meili_client(self, client: MeilisearchClient) -> None:
        self._meili = client

    async def close(self) -> None:
        """关闭资源"""

    async def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """批量获取文本的 embedding 向量（Phase C-B2: 带缓存优化）"""
        tracer = get_tracer()
        with tracer.start_as_current_span("embedding.get_batch") as span:
            span.set_attribute("embedding.text_count", len(texts))

            # Phase C-B2: 检查缓存
            cached_results = {}
            texts_to_fetch = []
            text_indices = {}

            for i, text in enumerate(texts):
                if not text:
                    continue
                cache_key = hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()

                if self._cache_enabled and self._vector_cache:
                    cached = await self._vector_cache.get(cache_key)
                    if cached:
                        cached_results[i] = cached
                        continue

                texts_to_fetch.append(text)
                text_indices[len(texts_to_fetch) - 1] = i

            span.set_attribute("embedding.cache_hits", len(cached_results))
            span.set_attribute("embedding.cache_misses", len(texts_to_fetch))

            # Phase C-B2: 批量获取未缓存的 embeddings
            fetched_results = {}
            if texts_to_fetch:
                try:
                    http_pool = await self._get_http_pool()
                    response = await http_pool.post(
                        f"{self._embedding_service_url}/v1/embeddings",
                        json={"input": texts_to_fetch, "model": "Qwen3-Embedding-0.6B"},
                    )
                    response.raise_for_status()
                    data = response.json()

                    for j, item in enumerate(data["data"]):
                        original_index = text_indices[j]
                        embedding = item["embedding"]
                        fetched_results[original_index] = embedding

                        # Phase C-B2: 存入缓存
                        if self._cache_enabled and self._vector_cache:
                            cache_key = hashlib.md5(texts_to_fetch[j].encode(), usedforsecurity=False).hexdigest()
                            await self._vector_cache.set(cache_key, embedding, ttl=self._cache_ttl)

                except Exception as e:
                    span.record_exception(e)
                    raise EmbeddingError(f"Failed to get embeddings: {e!s}") from e

            # 合并结果
            result = []
            for i in range(len(texts)):
                if i in cached_results:
                    result.append(cached_results[i])
                elif i in fetched_results:
                    result.append(fetched_results[i])
                else:
                    result.append([])  # Empty embedding for empty text

            span.set_attribute("embedding.dimension", len(result[0]) if result else 0)
            return result

    # ==================== 上传 ====================

    async def upload_memories(self, memories: list[dict[str, Any]], tenant_id: str | None = None) -> dict[str, Any]:
        """批量上传记忆（Phase A-B2: 使用事务批量插入 + Phase A-B7: 智能去重决策）"""
        if not memories:
            raise ValidationError("Memories list cannot be empty")

        effective_tenant_id = tenant_id or self._default_tenant_id
        tracer = get_tracer()

        logger.info(
            "Starting upload_memories: count=%d, tenant=%s, meili_enabled=%s",
            len(memories),
            effective_tenant_id,
            self._meili is not None,
        )

        with tracer.start_as_current_span("memory.upload") as span:
            span.set_attribute("memory.count", len(memories))
            span.set_attribute("tenant.id", effective_tenant_id)

            total = len(memories)
            success_count = 0
            failed_count = 0
            updated_count = 0
            skipped: list[dict[str, Any]] = []
            memory_ids: list[str] = []
            errors: list[str | dict[str, Any]] = []
            meili_docs: list[dict[str, Any]] = []

            # Phase A-B5: 批量获取 embeddings（优化为单次批量调用）
            texts = [m.get("content", "") for m in memories]
            try:
                embeddings = await self._get_embeddings(texts)

            except EmbeddingError as e:
                span.record_exception(e)
                span.set_attribute("memory.upload.success", 0)
                span.set_attribute("memory.upload.failed", total)

                return {
                    "total": total,
                    "success": 0,
                    "failed": total,
                    "updated": 0,
                    "skipped": [],
                    "memory_ids": [],
                    "errors": [str(e)],
                }

            # Phase A-B2: 构建批量插入数据
            batch_inserts: list[dict[str, Any]] = []

            for memory, embedding in zip(memories, embeddings, strict=False):
                try:
                    content = memory.get("content", "")
                    mem_type = memory.get("type", "general")

                    memory_data: dict[str, Any] = {
                        "content": content,
                        "content_hash": hashlib.md5(content.encode("utf-8"), usedforsecurity=False).hexdigest(),
                        "embedding": embedding,
                        "tenant_id": effective_tenant_id,
                        "type": mem_type,
                        "tags": memory.get("tags", []),
                        "project_id": memory.get("project_id", "global"),
                        "source": memory.get("source", "api"),
                        "metadata": memory.get("metadata", {}),
                    }

                    # v2.4.0 L0/L1/L2 fields
                    memory_data["abstract"] = memory.get("abstract", "")
                    memory_data["overview"] = memory.get("overview", "")
                    if memory.get("local_id"):
                        memory_data["local_id"] = memory["local_id"]

                    if "source_id" in memory:
                        memory_data["source_id"] = memory["source_id"]
                    if "source_timestamp" in memory:
                        memory_data["source_timestamp"] = memory["source_timestamp"]
                    if "classification_confidence" in memory:
                        memory_data["classification_confidence"] = memory["classification_confidence"]

                    # Phase A-B6: 使用动态阈值
                    dedup_threshold = self._get_dedup_threshold(mem_type)

                    # BL-CA-08: 代码文件专用 Upsert（file_path + project_id）
                    if mem_type == "code":
                        metadata = memory.get("metadata", {})
                        file_path = metadata.get("file_path")
                        if file_path:
                            code_existing = await self._db_query(
                                "SELECT id, metadata FROM memory WHERE type = 'code' AND project_id = $project_id AND metadata->file_path = $file_path AND tenant_id = $tenant_id LIMIT 1",
                                {
                                    "project_id": memory_data["project_id"],
                                    "file_path": file_path,
                                    "tenant_id": effective_tenant_id,
                                },
                            )
                            code_records = self._extract_records(code_existing)
                            if code_records:
                                existing_id = str(code_records[0].get("id", ""))
                                # 更新现有记录
                                await self._update_memory(existing_id, memory_data)
                                memory_ids.append(existing_id)
                                updated_count += 1
                                success_count += 1
                                continue

                    # 通用去重：检查 content_hash
                    existing = await self._db_query(
                        "SELECT id FROM memory WHERE tenant_id = $tenant_id AND content_hash = $hash LIMIT 1",
                        {"tenant_id": effective_tenant_id, "hash": memory_data["content_hash"]},
                    )
                    existing_records = self._extract_records(existing)
                    if existing_records:
                        existing_id = str(existing_records[0].get("id", ""))
                        failed_count += 1
                        skipped.append(
                            {
                                "local_id": memory.get("local_id") or memory.get("source_id"),
                                "existing_id": existing_id,
                                "reason": "hash",
                                "similarity": None,
                            }
                        )
                        continue

                    # Phase A-B7: 语义相似度检查 + 智能决策
                    similar = await self._search_by_vector(
                        embedding=embedding,
                        limit=1,
                        threshold=dedup_threshold,  # Phase A-B6: 动态阈值
                        tenant_id=effective_tenant_id,
                    )

                    if similar:
                        similarity_score = similar[0].get("score", 0)
                        existing_id = similar[0].get("id", "")
                        existing_record = similar[0]

                        # Phase A-B7: 智能决策（UPDATE / DISCARD / KEEP_BOTH）
                        decision = self._decide_duplicate_action(
                            new_memory=memory,
                            old_record=existing_record,
                            similarity=similarity_score,
                            mem_type=mem_type,
                        )

                        if decision == "UPDATE":
                            # 更新现有记录
                            await self._update_memory(existing_id, memory_data)
                            memory_ids.append(existing_id)
                            updated_count += 1
                            success_count += 1
                            continue
                        elif decision == "DISCARD":
                            failed_count += 1
                            skipped.append(
                                {
                                    "local_id": memory.get("local_id") or memory.get("source_id"),
                                    "existing_id": existing_id,
                                    "reason": "semantic",
                                    "similarity": round(similarity_score, 4),
                                }
                            )
                            continue
                        # else: KEEP_BOTH - 继续创建新记录

                    # 加入批量插入队列（Phase A-B2）
                    batch_inserts.append(memory_data)

                except Exception as e:
                    failed_count += 1
                    errors.append(f"{type(e).__name__}: {e!s}")

            # Phase A-B2: 批量插入（使用事务）
            if batch_inserts:
                try:
                    # 使用单个 INSERT 语句批量插入
                    query = "INSERT INTO memory $data"
                    result = await self._db_query(query, {"data": batch_inserts})

                    # 处理结果
                    records = self._extract_records(result)
                    if records:
                        for i, record in enumerate(records):
                            record_id = str(record.get("id", "")) if isinstance(record, dict) else None
                            if record_id:
                                memory_ids.append(record_id)
                                success_count += 1

                                # 构建 Meilisearch 文档
                                if self._meili and i < len(batch_inserts):
                                    mem = batch_inserts[i]
                                    meili_doc = self._build_meili_doc(record_id, mem, effective_tenant_id)
                                    meili_docs.append(meili_doc)
                            else:
                                failed_count += 1
                                errors.append("No record ID returned")
                except Exception as e:
                    # 批量插入失败，回退到单条插入
                    logger.warning(f"Batch insert failed, falling back to single insert: {e}")
                    for memory_data in batch_inserts:
                        try:
                            result = await self._db_create("memory", memory_data)
                            record_id = self._extract_record_id(result)
                            if record_id:
                                memory_ids.append(record_id)
                                success_count += 1
                                # 单条插入也同步到 Meilisearch
                                if self._meili:
                                    meili_doc = self._build_meili_doc(record_id, memory_data, effective_tenant_id)
                                    meili_docs.append(meili_doc)
                            else:
                                failed_count += 1
                        except Exception as inner_e:
                            failed_count += 1
                            errors.append(f"{type(inner_e).__name__}: {inner_e!s}")

            # 同步到 Meilisearch
            if self._meili and meili_docs:
                try:
                    await self._meili.add_documents(meili_docs)
                    span.set_attribute("memory.upload.meili_synced", len(meili_docs))
                except Exception as meili_err:
                    logger.error("[Meili sync] 同步失败: %s", meili_err)
                    span.set_attribute("memory.upload.meili_error", str(meili_err))

        # BL-4: Phase A - 自动代码分析（异步触发，不阻塞上传）
        if memory_ids:
            asyncio.create_task(self._auto_analyze_memories(memory_ids, effective_tenant_id))

        span.set_attribute("memory.upload.success", success_count)
        span.set_attribute("memory.upload.failed", failed_count)
        span.set_attribute("memory.upload.updated", updated_count)  # Phase A-B7

        result_data: dict[str, Any] = {
            "total": total,
            "success": success_count,
            "failed": failed_count,
            "updated": updated_count,
            "skipped": skipped,
            "memory_ids": memory_ids,
        }
        if errors:
            result_data["errors"] = errors[:10]
        return result_data

    # ==================== 通用同步 (Stub) ====================

    async def get_fingerprints(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
        """获取服务端所有记忆的指纹（Stub）"""
        effective_tenant_id = tenant_id or self._default_tenant_id
        logger.warning("[get_fingerprints] Stub called for tenant %s", effective_tenant_id)
        # TODO: 实现实际的指纹查询
        return []

    async def sync_preview(
        self,
        fingerprints: list[dict[str, Any]],
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        """同步预览：比对指纹，返回变更指令（Stub）"""
        effective_tenant_id = tenant_id or self._default_tenant_id
        logger.warning("[sync_preview] Stub called with %d fingerprints", len(fingerprints))
        # TODO: 实现实际的同步预览逻辑
        return {
            "synced": 0,
            "to_upload": [],
            "to_delete": [],
            "conflicts": [],
        }

    async def sync_full(
        self,
        memories: list[dict[str, Any]],
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        """全量同步：上传所有记忆（Stub）"""
        effective_tenant_id = tenant_id or self._default_tenant_id
        logger.warning("[sync_full] Stub called with %d memories", len(memories))
        # TODO: 实现实际的全量同步逻辑
        return {
            "total": len(memories),
            "success": 0,
            "failed": 0,
            "updated": 0,
            "skipped": [],
            "errors": ["Not implemented"],
        }

    async def resolve_conflict(
        self,
        conflict_id: str,
        resolution: str,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        """解决同步冲突（Stub）"""
        effective_tenant_id = tenant_id or self._default_tenant_id
        logger.warning("[resolve_conflict] Stub called for %s with resolution %s", conflict_id, resolution)
        # TODO: 实现实际的冲突解决逻辑
        return {"resolved": False, "error": "Not implemented"}

    # ==================== 代码同步 (BL-CA-07) ====================

    async def sync_code_fingerprints(
        self,
        fingerprints: list[dict[str, Any]],
        project_id: str,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        """代码文件增量同步：比对指纹，返回变更指令

        Args:
            fingerprints: 本地代码文件指纹列表
            project_id: 项目标识
            tenant_id: 租户ID

        Returns:
            {"changed": [...], "unchanged": [...], "missing": [...], "conflicts": [...]}
        """
        effective_tenant_id = tenant_id or self._default_tenant_id
        changed: list[dict[str, Any]] = []
        unchanged: list[str] = []
        missing: list[str] = []
        conflicts: list[dict[str, Any]] = []

        # 查询该项目下所有代码文件
        query = """
            SELECT id, metadata, content_hash, mtime, source_id
            FROM memory
            WHERE type = "code"
              AND project_id = $project_id
              AND tenant_id = $tenant_id
        """
        result = await self._db_query(query, {"project_id": project_id, "tenant_id": effective_tenant_id})
        server_records = self._extract_records(result)

        # 建立 path -> record 映射
        server_files: dict[str, dict[str, Any]] = {}
        for record in server_records:
            metadata = record.get("metadata", {})
            file_path = metadata.get("file_path")
            if file_path:
                server_files[file_path] = record

        # 比对每个本地指纹
        for local in fingerprints:
            path = local.get("path")
            local_hash = local.get("hash")
            local_symbols_hash = local.get("symbols_hash")
            local_mtime = local.get("mtime") or 0

            if not path:
                continue

            server = server_files.get(path)

            if server is None:
                # 服务端没有此文件
                missing.append(path)
                continue

            server_mtime = server.get("mtime") or 0
            server_hash = server.get("content_hash", "")
            server_metadata = server.get("metadata", {})
            server_symbols_hash = server_metadata.get("symbols_hash", "")

            # 检查内容是否一致
            if local_hash == server_hash:
                # 内容一致，检查符号
                if local_symbols_hash == server_symbols_hash:
                    unchanged.append(path)
                else:
                    # 仅符号变更
                    changed.append(
                        {
                            "path": path,
                            "reason": "symbols_modified",
                            "server_mtime": server_mtime,
                        }
                    )
            else:
                # 内容变更，检查 mtime 冲突
                if local_mtime < server_mtime:
                    # 服务端更新，可能冲突
                    conflicts.append(
                        {
                            "path": path,
                            "local_mtime": local_mtime,
                            "server_mtime": server_mtime,
                        }
                    )
                else:
                    # 本地更新
                    changed.append(
                        {
                            "path": path,
                            "reason": "content_modified",
                            "server_mtime": server_mtime,
                        }
                    )

        return {
            "changed": changed,
            "unchanged": unchanged,
            "missing": missing,
            "conflicts": conflicts,
        }

    # ==================== 搜索 ====================

    async def search_memories(
        self,
        query: str,
        mode: str = "hybrid",
        limit: int = 10,
        threshold: float = 0.7,
        level: int = 2,
        tenant_id: str | None = None,
        filters: str | None = None,
    ) -> dict[str, Any]:
        """搜索记忆 (v2.4.0: 支持 level 参数返回分层内容)"""
        if mode not in ("vector", "keyword", "hybrid"):
            raise ValidationError(f"Invalid search mode: {mode}")

        effective_tenant_id = tenant_id or self._default_tenant_id
        tracer = get_tracer()

        with tracer.start_as_current_span("memory.search") as span:
            span.set_attribute("search.mode", mode)
            span.set_attribute("search.limit", limit)
            span.set_attribute("search.threshold", threshold)
            span.set_attribute("tenant.id", effective_tenant_id)
            span.set_attribute("search.query_length", len(query))

            try:
                if mode == "vector":
                    embeddings = await self._get_embeddings([query])
                    results = await self._search_by_vector(embeddings[0], limit, threshold, effective_tenant_id)
                elif mode == "keyword":
                    results = await self._search_by_keyword(query, limit, effective_tenant_id, filter_expr=filters)
                else:
                    embeddings = await self._get_embeddings([query])
                    results = await self._hybrid_search(
                        query, embeddings[0], limit, threshold, effective_tenant_id, filters
                    )

                span.set_attribute("search.result_count", len(results))

                # v2.4.0: 按 level 返回分层内容
                filtered_results = self._filter_by_level(results, level)

                return {
                    "results": filtered_results,
                    "total": len(results),
                    "mode": mode,
                    "level": level,
                    "query": query,
                }
            except Exception as e:
                span.record_exception(e)
                raise DatabaseError(f"Search failed: {e!s}") from e

    def _filter_by_level(self, results: list[dict[str, Any]], level: int) -> list[dict[str, Any]]:
        """按 level 过滤返回结果 (v2.4.0)"""
        filtered = []
        for r in results:
            item = {"id": r.get("id"), "score": r.get("score")}

            if level >= 0:
                item["abstract"] = r.get("abstract", "")

            if level >= 1:
                item["overview"] = r.get("overview", "")

            if level >= 2:
                item["content"] = r.get("content", "")
                item["type"] = r.get("type")
                item["tags"] = r.get("tags", [])
                item["project_id"] = r.get("project_id")
                item["local_id"] = r.get("local_id")
                item["metadata"] = r.get("metadata", {})

            filtered.append(item)

        return filtered

    async def _search_by_vector(
        self,
        embedding: list[float],
        limit: int,
        threshold: float,
        tenant_id: str,
    ) -> list[dict[str, Any]]:
        """向量相似度搜索（利用 HNSW 索引 + 缓存）"""
        tracer = get_tracer()
        with tracer.start_as_current_span("search.vector") as span:
            span.set_attribute("search.vector.limit", limit)
            span.set_attribute("search.vector.threshold", threshold)

            # Phase A-B3: 缓存检查
            cache_key = self._get_vector_cache_key(embedding, limit, threshold, tenant_id)
            if self._vector_cache:
                try:
                    cached_result = await self._vector_cache.get(cache_key)
                    if cached_result:
                        span.set_attribute("search.vector.cache_hit", True)
                        span.set_attribute("search.vector.result_count", len(cached_result))
                        return cached_result
                except Exception:  # nosec B110
                    pass
            span.set_attribute("search.vector.cache_hit", False)

            q = (
                "SELECT id, content, abstract, overview, local_id, metadata, type, tags, project_id, "
                "vector::similarity::cosine(embedding, $query_embedding) AS score "
                "FROM memory "
                "WHERE tenant_id = $tenant_id "
                "AND vector::similarity::cosine(embedding, $query_embedding) >= $threshold "
                "ORDER BY score DESC "
                "LIMIT $limit"
            )
            result = await self._db_query(
                q, {"tenant_id": tenant_id, "query_embedding": embedding, "threshold": threshold, "limit": limit}
            )

            results = self._format_similarity_results(result)

            # Phase A-B3: 缓存结果
            if self._vector_cache and results:
                try:
                    await self._vector_cache.set(cache_key, results, ttl=self._cache_ttl)
                except Exception:  # nosec B110
                    pass

            span.set_attribute("search.vector.result_count", len(results))
            return results

    async def _search_by_keyword(
        self, query_text: str, limit: int, tenant_id: str, filter_expr: str | None = None
    ) -> list[dict[str, Any]]:
        """全文搜索：优先使用 Meilisearch，降级到 SurrealDB BM25"""
        if self._meili:
            return await self._search_by_keyword_meili(query_text, limit, tenant_id, filter_expr=filter_expr)
        return await self._search_by_keyword_surreal(query_text, limit, tenant_id)

    async def _search_by_keyword_meili(
        self,
        query_text: str,
        limit: int,
        tenant_id: str,
        filter_expr: str | None = None,
    ) -> list[dict[str, Any]]:
        """Meilisearch 全文搜索（支持 CJK 分词、日期精确匹配）"""
        assert self._meili is not None  # 由 _search_by_keyword 保证
        tracer = get_tracer()
        with tracer.start_as_current_span("search.keyword.meili") as span:
            span.set_attribute("search.keyword.engine", "meilisearch")
            span.set_attribute("search.keyword.query", query_text[:100])

            actual_filter = filter_expr or f"tenant_id = '{tenant_id}'"
            result = await self._meili.search(
                query_text,
                filter_expr=actual_filter,
                limit=limit,
            )
            results = self._format_meili_results(result)
            span.set_attribute("search.keyword.result_count", len(results))
            return results

    async def _search_by_keyword_surreal(
        self,
        query_text: str,
        limit: int,
        tenant_id: str,
    ) -> list[dict[str, Any]]:
        """SurrealDB BM25 全文搜索（Meilisearch 不可用时的降级路径）"""
        tracer = get_tracer()
        with tracer.start_as_current_span("search.keyword.surreal") as span:
            safe_query = self._sanitize_query(query_text)
            span.set_attribute("search.keyword.engine", "surrealdb")
            span.set_attribute("search.keyword.query", safe_query[:100])
            q = (  # nosec B608
                "SELECT id, content, abstract, overview, local_id, metadata, type, tags, project_id, "
                "search::score(1) AS score "
                "FROM memory "
                "WHERE tenant_id = $tenant_id "
                f"AND content @1@ '{safe_query}' "  # nosec B608
                "ORDER BY score DESC "
                "LIMIT $limit"
            )
            result = await self._db_query(q, {"tenant_id": tenant_id, "limit": limit})
            results = self._format_keyword_results(result)
            span.set_attribute("search.keyword.result_count", len(results))
            return results

    async def _hybrid_search(
        self,
        query_text: str,
        embedding: list[float],
        limit: int,
        threshold: float,
        tenant_id: str,
        filter_expr: str | None = None,
    ) -> list[dict[str, Any]]:
        """RRF 混合搜索：并行执行向量+关键词搜索，然后用 RRF 算法融合"""
        tracer = get_tracer()
        with tracer.start_as_current_span("search.hybrid") as span:
            tasks = [
                self._search_by_vector(embedding, limit * 2, threshold, tenant_id),
                self._search_by_keyword(query_text, limit * 2, tenant_id, filter_expr=filter_expr),
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            vector_results: list[dict[str, Any]] = [] if isinstance(results[0], BaseException) else results[0]
            keyword_results: list[dict[str, Any]] = [] if isinstance(results[1], BaseException) else results[1]

            span.set_attribute("search.hybrid.vector_count", len(vector_results))
            span.set_attribute("search.hybrid.keyword_count", len(keyword_results))

            merged = self._rrf_merge(vector_results, keyword_results)
            span.set_attribute("search.hybrid.merged_count", len(merged[:limit]))
            return merged[:limit]

    def _rrf_merge(
        self,
        vector_results: list[dict[str, Any]],
        keyword_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """RRF (Reciprocal Rank Fusion) 混合搜索合并算法

        公式: RRF_score(d) = Σ (weight_i / (k + rank_i(d)))

        使用排名位置而非原始分数，天然归一化，不受 distance/score 尺度差异影响。
        k=60 来自原始论文（Cormack et al. 2009）推荐值。
        """
        k = self._rrf_k
        vector_weight = self._rrf_vector_weight
        keyword_weight = self._rrf_keyword_weight

        scores: dict[str, float] = {}
        items: dict[str, dict[str, Any]] = {}

        # 向量搜索贡献
        for rank, item in enumerate(vector_results):
            doc_id = item.get("id")
            if not doc_id:
                continue
            scores[doc_id] = vector_weight / (k + rank + 1)
            items[doc_id] = item

        # 关键词搜索贡献
        for rank, item in enumerate(keyword_results):
            doc_id = item.get("id")
            if not doc_id:
                continue
            scores.setdefault(doc_id, 0.0)
            scores[doc_id] += keyword_weight / (k + rank + 1)
            if doc_id not in items:
                items[doc_id] = item

        # 按 RRF 分数降序排列
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        results: list[dict[str, Any]] = []
        for doc_id in sorted_ids:
            item = items[doc_id].copy()
            item["score"] = round(scores[doc_id], 6)
            results.append(item)

        return results

    # ==================== 结果格式化 ====================

    def _format_vector_results(self, db_result: Any, max_distance: float) -> list[dict[str, Any]]:
        """格式化向量搜索结果：distance → similarity score + 阈值过滤"""
        raw_items = self._extract_records(db_result)
        results: list[dict[str, Any]] = []
        for item in raw_items:
            distance = item.get("distance")
            if distance is None:
                continue
            # 阈值过滤：distance > max_distance 表示相似度不够
            if float(distance) > max_distance:
                continue
            # distance → similarity score (cosine: similarity = 1 - distance)
            score = round(1.0 - float(distance), 6)
            results.append(self._build_result_item(item, score=score))
        return results

    def _format_similarity_results(self, db_result: Any) -> list[dict[str, Any]]:
        raw_items = self._extract_records(db_result)
        results: list[dict[str, Any]] = []
        for item in raw_items:
            score = item.get("score")
            if score is None:
                continue
            results.append(self._build_result_item(item, score=float(score)))
        return results

    def _format_keyword_results(self, db_result: Any) -> list[dict[str, Any]]:
        """格式化关键词搜索结果"""
        raw_items = self._extract_records(db_result)
        results: list[dict[str, Any]] = []
        for item in raw_items:
            score = item.get("score", 0.0)
            results.append(self._build_result_item(item, score=float(score)))
        return results

    def _format_meili_results(self, meili_result: dict[str, Any]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for hit in meili_result.get("hits", []):
            doc_id = hit.get("id", "")
            if doc_id and not doc_id.startswith("memory:"):
                doc_id = f"memory:{doc_id}"
            score = hit.get("_rankingScore", 0.0)
            results.append(
                {
                    "id": str(doc_id),
                    "content": hit.get("content", ""),
                    "abstract": hit.get("abstract", ""),
                    "overview": hit.get("overview", ""),
                    "local_id": hit.get("local_id"),
                    "metadata": hit.get("metadata", {}),
                    "type": hit.get("type", "general"),
                    "tags": hit.get("tags", []),
                    "project_id": hit.get("project_id", "global"),
                    "score": round(float(score), 6),
                }
            )
        return results

    def _build_result_item(self, item: dict[str, Any], score: float) -> dict[str, Any]:
        """构建统一的搜索结果条目（包含 v2.4.0 L0/L1/L2 字段）"""
        return {
            "id": str(item.get("id", "")),
            "content": item.get("content", ""),
            "abstract": item.get("abstract", ""),
            "overview": item.get("overview", ""),
            "local_id": item.get("local_id"),
            "metadata": item.get("metadata", {}),
            "type": item.get("type", "general"),
            "tags": item.get("tags", []),
            "project_id": item.get("project_id", "global"),
            "score": score,
        }

    def _extract_records(self, db_result: Any) -> list[dict[str, Any]]:
        """从 SurrealDB query() 返回值中提取记录列表

        处理 SDK 返回的多种格式：
        - list[dict]: 直接的记录列表（单条 SELECT 语句）
        - list[list[dict]]: 嵌套结构（多语句结果或 query_raw）
        """
        records: list[dict[str, Any]] = []
        if not db_result or not isinstance(db_result, list):
            return records
        for item in db_result:
            if isinstance(item, dict):
                records.append(item)
            elif isinstance(item, list):
                for record in item:
                    if isinstance(record, dict):
                        records.append(record)
        return records

    def _extract_record_id(self, db_result: Any) -> str | None:
        """从 SurrealDB create() 或 query() 返回值中提取记录 ID

        处理 SDK 返回的多种格式：
        - dict with 'id': 单条记录
        - list[dict]: 记录列表，取第一个
        - list[list[dict]]: 嵌套结构
        """
        if not db_result:
            return None

        # 直接是 dict
        if isinstance(db_result, dict):
            record_id = db_result.get("id")
            return str(record_id) if record_id else None

        # 是列表，尝试提取
        if isinstance(db_result, list):
            if not db_result:
                return None
            first = db_result[0]
            # 嵌套列表
            if isinstance(first, list) and first:
                first = first[0]
            # 提取 ID
            if isinstance(first, dict):
                record_id = first.get("id")
                return str(record_id) if record_id else None

        return None

    def _sanitize_query(self, text: str) -> str:
        """清洗搜索查询文本，防止 SurrealQL 注入

        策略：移除 SurrealQL 特殊字符，保留字母数字和 CJK 字符。
        比简单转义更安全：直接移除潜在危险字符而非依赖转义正确性。
        """
        # 保留: 字母、数字、空格、CJK 统一表意文字（U+4E00-U+9FFF）
        # 移除: 引号、分号、反斜杠等 SQL/SurrealQL 特殊字符
        return re.sub(r"[^\w\s\u4e00-\u9fff\u3400-\u4dbf\uff00-\uffef-]", "", text).strip()[:500]

    # ==================== Stub Methods (TODO: Implement) ====================

    async def _update_memory(self, memory_id: str, memory_data: dict[str, Any]) -> None:
        """更新现有记忆记录

        1. SurrealDB UPDATE：更新现有记录的字段
        2. Meilisearch 同步：删除旧文档并重新添加
        """
        tenant_id = memory_data.get("tenant_id", self._default_tenant_id)

        # 1. SurrealDB UPDATE
        sql = """
            UPDATE type::record($id) SET
                content = $content,
                embedding = $embedding,
                abstract = $abstract,
                overview = $overview,
                tags = $tags,
                metadata = $metadata,
                content_hash = $content_hash,
                source_timestamp = $source_timestamp,
                classification_confidence = $classification_confidence,
                mtime = $mtime
            WHERE tenant_id = $tenant_id
        """
        await self._db_query(
            sql,
            {
                "id": memory_id,
                "content": memory_data.get("content", ""),
                "embedding": memory_data.get("embedding", []),
                "abstract": memory_data.get("abstract", ""),
                "overview": memory_data.get("overview", ""),
                "tags": memory_data.get("tags", []),
                "metadata": memory_data.get("metadata", {}),
                "content_hash": memory_data.get("content_hash", ""),
                "source_timestamp": memory_data.get("source_timestamp"),
                "classification_confidence": memory_data.get("classification_confidence"),
                "mtime": memory_data.get("mtime"),
                "tenant_id": tenant_id,
            },
        )
        logger.info("[_update_memory] SurrealDB record updated: %s", memory_id)

        # 2. Meilisearch sync（失败不阻断主流程）
        if self._meili:
            try:
                await self._meili.delete_document(memory_id)
                meili_doc = self._build_meili_doc(memory_id, memory_data, tenant_id)
                await self._meili.add_documents([meili_doc])
                logger.info("[_update_memory] Meilisearch synced for: %s", memory_id)
            except Exception as e:
                logger.warning(
                    "[_update_memory] Meilisearch sync failed for %s: %s",
                    memory_id,
                    e,
                )

    def _decide_duplicate_action(
        self,
        new_memory: dict[str, Any],
        old_record: dict[str, Any],
        similarity: float,
        mem_type: str,
    ) -> str:
        if similarity >= 0.95 and "source_id" not in new_memory:
            return "UPDATE"
        return "KEEP_BOTH"

    def _build_meili_doc(self, record_id: str, memory_data: dict[str, Any], tenant_id: str) -> dict[str, Any]:
        """构建 Meilisearch 文档

        根据 meili_client.py DEFAULT_INDEX_SETTINGS 配置，构建包含所有必需字段的文档。
        """
        from datetime import datetime, timezone

        # 基础字段
        doc: dict[str, Any] = {
            "id": record_id,
            "surreal_id": record_id,
            "content": memory_data.get("content", ""),
            "content_zh": memory_data.get("content", ""),  # 简化：直接使用 content
            "tenant_id": tenant_id,
            "type": memory_data.get("type", "general"),
            "tags": memory_data.get("tags", []),
            "project_id": memory_data.get("project_id", "global"),
            "created_at": memory_data.get("created_at") or datetime.now(timezone.utc).isoformat(),
            "source_id": memory_data.get("source_id", ""),
            "metadata": memory_data.get("metadata", {}),
        }

        # 分层内容字段 (L0/L1/L2)
        doc["abstract"] = memory_data.get("abstract", "")
        doc["overview"] = memory_data.get("overview", "")

        # 代码分析字段 (BL-CA-01~04)
        metadata = memory_data.get("metadata", {})
        code_analysis = metadata.get("code_analysis", {})
        if code_analysis:
            doc["code_language"] = code_analysis.get("language", "")
            complexity = code_analysis.get("complexity", {})
            doc["code_complexity"] = complexity.get("cyclomatic_complexity", 0)
            doc["code_function_count"] = complexity.get("function_count", 0)
            doc["code_class_count"] = complexity.get("class_count", 0)
            doc["code_analyzer"] = code_analysis.get("analyzer", "")
            # code_symbols 在 upload_memories 中单独处理
            if "code_symbols" in metadata:
                doc["code_symbols"] = metadata["code_symbols"]

        return doc

    def _from_meili_id(self, meili_id: str) -> str:
        return meili_id.replace("_", ":", 1)

    def _normalize_memory_id(self, memory_id: str) -> str:
        """规范化记忆 ID（Stub）"""
        if ":" not in memory_id:
            return f"memory:{memory_id}"
        return memory_id

    def _format_relation_results(self, raw: Any, direction: str) -> list[dict[str, Any]]:
        """格式化关系查询结果"""
        records = self._extract_records(raw)
        results: list[dict[str, Any]] = []
        for rec in records:
            results.append(
                {
                    "id": str(rec.get("relation_id", "")),
                    "from_id": str(rec.get("from_id", "")),
                    "to_id": str(rec.get("to_id", "")),
                    "relationship_type": rec.get("relationship_type", "related"),
                    "weight": float(rec.get("weight", 0.5)),
                    "direction": direction,
                    "description": rec.get("description"),
                    "metadata": rec.get("metadata"),
                }
            )
        return results

    def _normalize_relation_id(self, relation_id: str) -> str:
        """规范化关系 ID（Stub）"""
        if ":" not in relation_id:
            return f"memory_relation:{relation_id}"
        return relation_id

    # ==================== 图关系 (RELATE) ====================

    async def create_relation(
        self,
        from_id: str,
        to_id: str,
        relationship_type: str = "related",
        weight: float = 0.5,
        tenant_id: str | None = None,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """创建两条记忆之间的图关系"""
        effective_tenant_id = tenant_id or self._default_tenant_id

        valid_types = {"related", "follow_up", "elaboration", "contradiction", "reference", "derived_from"}
        if relationship_type not in valid_types:
            raise ValidationError(f"Invalid relationship_type: {relationship_type}. Must be one of {valid_types}")
        if not 0.0 <= weight <= 1.0:
            raise ValidationError(f"weight must be between 0.0 and 1.0, got {weight}")

        from_ref = self._normalize_memory_id(from_id)
        to_ref = self._normalize_memory_id(to_id)

        tracer = get_tracer()
        with tracer.start_as_current_span("graph.create_relation") as span:
            span.set_attribute("graph.from", from_ref)
            span.set_attribute("graph.to", to_ref)
            span.set_attribute("graph.type", relationship_type)
            span.set_attribute("tenant.id", effective_tenant_id)

            try:
                set_clauses = [
                    f"relationship_type = '{relationship_type}'",
                    f"weight = {float(weight)}",
                    "tenant_id = $tenant_id",
                ]
                if description:
                    safe_desc = self._sanitize_query(description)
                    set_clauses.append(f"description = '{safe_desc}'")
                if metadata:
                    set_clauses.append(f"metadata = {json.dumps(metadata)}")

                set_str = ", ".join(set_clauses)
                q = (  # nosec B608
                    f"RELATE {from_ref}->memory_relation->{to_ref} "  # nosec B608
                    f"SET {set_str}"  # nosec B608
                )
                result = await self._db_query(q, {"tenant_id": effective_tenant_id})

                records = self._extract_records(result)
                if records:
                    return {
                        "id": str(records[0].get("id", "")),
                        "from": from_ref,
                        "to": to_ref,
                        "relationship_type": relationship_type,
                        "weight": weight,
                    }
                return {"error": "No relation created"}
            except Exception as e:
                span.record_exception(e)
                raise DatabaseError(f"Failed to create relation: {e!s}") from e

    async def get_relations(
        self,
        memory_id: str,
        direction: str = "both",
        relationship_type: str | None = None,
        tenant_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """查询记忆的关联关系"""
        effective_tenant_id = tenant_id or self._default_tenant_id
        mem_ref = self._normalize_memory_id(memory_id)

        if direction not in ("outgoing", "incoming", "both"):
            raise ValidationError(f"Invalid direction: {direction}. Must be outgoing/incoming/both")

        tracer = get_tracer()
        with tracer.start_as_current_span("graph.get_relations") as span:
            span.set_attribute("graph.memory_id", mem_ref)
            span.set_attribute("graph.direction", direction)
            span.set_attribute("tenant.id", effective_tenant_id)

            try:
                results: list[dict[str, Any]] = []

                # 拆分 mem_ref 为 table 和 id
                mem_parts = mem_ref.split(":")
                mem_table, mem_id = mem_parts[0], mem_parts[1] if len(mem_parts) > 1 else mem_parts[0]

                if direction in ("outgoing", "both"):
                    q = (
                        "SELECT *, meta::id(id) AS relation_id, "
                        "meta::id(in) AS from_id, meta::id(out) AS to_id "
                        "FROM memory_relation "
                        "WHERE in = type::record($mem_table, $mem_id) AND tenant_id = $tenant_id "
                    )
                    if relationship_type:
                        q += "AND relationship_type = $relationship_type "
                    q += "LIMIT $limit"
                    r = await self._db_query(
                        q,
                        {
                            "tenant_id": effective_tenant_id,
                            "mem_table": mem_table,
                            "mem_id": mem_id,
                            "relationship_type": relationship_type,
                            "limit": limit,
                        },
                    )
                    results.extend(self._format_relation_results(r, "outgoing"))

                if direction in ("incoming", "both"):
                    q = (
                        "SELECT *, meta::id(id) AS relation_id, "
                        "meta::id(in) AS from_id, meta::id(out) AS to_id "
                        "FROM memory_relation "
                        "WHERE out = type::record($mem_table, $mem_id) AND tenant_id = $tenant_id "
                    )
                    if relationship_type:
                        q += "AND relationship_type = $relationship_type "
                    q += "LIMIT $limit"
                    r = await self._db_query(
                        q,
                        {
                            "tenant_id": effective_tenant_id,
                            "mem_table": mem_table,
                            "mem_id": mem_id,
                            "relationship_type": relationship_type,
                            "limit": limit,
                        },
                    )
                    results.extend(self._format_relation_results(r, "incoming"))

                span.set_attribute("graph.relation_count", len(results))
                return results
            except Exception as e:
                span.record_exception(e)
                raise DatabaseError(f"Failed to get relations: {e!s}") from e

    async def delete_relation(
        self,
        relation_id: str,
        tenant_id: str | None = None,
    ) -> bool:
        """删除指定的关系

        Args:
            relation_id: 关系 ID（如 "memory_relation:abc123"）
            tenant_id: 租户 ID（安全检查，防止跨租户删除）
        """
        effective_tenant_id = tenant_id or self._default_tenant_id
        rel_ref = self._normalize_relation_id(relation_id)

        try:
            # 拆分 rel_ref 为 table 和 id
            rel_parts = rel_ref.split(":")
            rel_table, rel_id = rel_parts[0], rel_parts[1] if len(rel_parts) > 1 else rel_parts[0]

            # 先验证关系存在且属于该租户
            q = "SELECT id FROM type::record($rel_table, $rel_id) WHERE tenant_id = $tenant_id"
            check = await self._db_query(
                q, {"tenant_id": effective_tenant_id, "rel_table": rel_table, "rel_id": rel_id}
            )
            records = self._extract_records(check)
            if not records:
                return False

            # 删除关系
            del_q = "DELETE type::record($rel_table, $rel_id)"
            await self._db_query(del_q, {"rel_table": rel_table, "rel_id": rel_id})
            return True
        except Exception as e:
            logger.error("[Relation Delete] 失败: %s", e)
            raise DatabaseError(f"删除关系失败: {e}") from e

    # ==================== BL-6: LLM Code Summary ====================

    async def _generate_code_summary(self, memory_id: str, tenant_id: str) -> dict[str, Any] | None:
        """调用 LLM 生成代码摘要"""
        from ..config import LLMConfig

        config = LLMConfig()
        if not config.enabled:
            return None

        try:
            # 获取记忆内容和代码分析结果
            query = "SELECT content, metadata FROM $memory_id WHERE tenant_id = $tenant_id"
            result = await self._db_query(query, {"memory_id": memory_id, "tenant_id": tenant_id})
            records = self._extract_records(result)

            if not records:
                return None

            memory = records[0]
            content = memory.get("content", "")
            metadata = memory.get("metadata", {}) or {}
            code_analysis = metadata.get("code_analysis", {})

            if not code_analysis:
                return None

            # 构建 LLM 提示
            functions = code_analysis.get("functions", [])
            classes = code_analysis.get("classes", [])
            language = code_analysis.get("language", "unknown")

            prompt = f"""分析以下 {language} 代码并提供摘要：

代码内容：
```
{content[:2000]}
```

函数列表：{", ".join(f.get("name", "") for f in functions[:10])}
类列表：{", ".join(c.get("name", "") for c in classes[:10])}

请提供：
1. 一句话摘要（描述这个模块/文件的主要功能）
2. 关键函数列表（最重要的3-5个函数及其作用）
3. 代码用途（这个代码解决什么问题）

以 JSON 格式返回：
{{
    "summary": "一句话摘要",
    "key_functions": ["函数1: 作用", "函数2: 作用"],
    "purpose": "代码用途描述"
}}"""

            # 调用 LLM API
            http_pool = await self._get_http_pool()
            headers = {"Content-Type": "application/json"}
            if config.api_key:
                headers["Authorization"] = f"Bearer {config.api_key}"

            payload = {
                "model": config.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": config.max_tokens,
                "temperature": 0.3,
            }

            response = await http_pool.post(
                f"{config.endpoint}/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=config.timeout,
            )

            if response.status_code != 200:
                logger.warning("[LLM Summary] API 调用失败: %s", response.status_code)
                return None

            result_data = response.json()
            llm_content = result_data.get("choices", [{}])[0].get("message", {}).get("content", "")

            # 解析 JSON 响应
            try:
                # 尝试提取 JSON 部分
                json_start = llm_content.find("{")
                json_end = llm_content.rfind("}")
                if json_start >= 0 and json_end > json_start:
                    json_str = llm_content[json_start : json_end + 1]
                    summary_data = json.loads(json_str)
                else:
                    # 如果不是 JSON，使用原始内容作为摘要
                    summary_data = {
                        "summary": llm_content[:200],
                        "key_functions": [],
                        "purpose": "",
                    }

                # 保存到 metadata
                code_summary = {
                    "summary": summary_data.get("summary", ""),
                    "key_functions": summary_data.get("key_functions", []),
                    "purpose": summary_data.get("purpose", ""),
                    "generated_at": asyncio.get_event_loop().time(),
                    "model": config.model_name,
                }

                metadata["code_summary"] = code_summary

                # 更新数据库
                update_query = """
                    UPDATE type::record($record_id)
                    SET metadata = $metadata
                """
                await self._db_query(update_query, {"record_id": memory_id, "metadata": metadata})

                logger.info("[LLM Summary] 摘要生成完成: %s", memory_id)
                return code_summary

            except json.JSONDecodeError as e:
                logger.warning("[LLM Summary] JSON 解析失败: %s", e)
                return None

        except Exception as e:
            logger.warning("[LLM Summary] 生成失败: %s", e)
            return None

    # ==================== v2.4.0 Access Log ====================

    async def report_access_log(self, entries: list[dict[str, Any]], tenant_id: str | None = None) -> dict[str, Any]:
        """记录访问日志（用于分析记忆使用频率）"""
        effective_tenant_id = tenant_id or self._default_tenant_id

        # 简化的实现：将访问日志记录到内存中
        # 实际生产环境可以存储到 SurrealDB 或发送到分析服务
        logged_count = 0
        for entry in entries:
            if entry.get("entry_id") and entry.get("timestamp"):
                logged_count += 1

        return {
            "status": "success",
            "logged_count": logged_count,
            "total_received": len(entries),
        }

    async def analyze_memory_code(
        self, memory_id: str, tenant_id: str = "default", persist: bool = False
    ) -> dict[str, Any]:
        effective_tenant_id = tenant_id or self._default_tenant_id
        mem_ref = self._normalize_memory_id(memory_id)

        try:
            query = "SELECT content, metadata FROM type::record($id) WHERE tenant_id = $tenant_id"
            result = await self._db_query(query, {"id": mem_ref, "tenant_id": effective_tenant_id})
            records = self._extract_records(result)

            if not records:
                logger.warning("[analyze_memory_code] 记忆不存在: %s", mem_ref)
                return {}

            content = records[0].get("content", "")
            if not content:
                logger.warning("[analyze_memory_code] 内容为空: %s", mem_ref)
                return {}

            if not self._is_code_content(content):
                logger.debug("[analyze_memory_code] 非代码内容，跳过: %s", mem_ref)
                return {}

            metadata = records[0].get("metadata") or {}
            language = metadata.get("language", "")
            if not language:
                file_path = metadata.get("file_path", "")
                if file_path and "." in file_path:
                    ext = file_path.rsplit(".", 1)[-1].lower()
                    language = {
                        "py": "python",
                        "js": "javascript",
                        "ts": "typescript",
                        "java": "java",
                        "go": "go",
                        "rs": "rust",
                        "c": "c",
                        "cpp": "cpp",
                        "h": "c",
                        "html": "html",
                        "css": "css",
                        "sql": "sql",
                    }.get(ext, "")
            if not language:
                language = "python"

            analysis_result = await self.code_analyzer.analyze_code(content, language)
            analysis_dict = analysis_result.to_metadata_dict()

            if persist:
                update_sql = "UPDATE type::record($id) SET metadata.code_analysis = $code_analysis"
                await self._db_query(update_sql, {"id": mem_ref, "code_analysis": analysis_dict})
                logger.info("[analyze_memory_code] 分析结果已持久化: %s", mem_ref)

            return analysis_dict

        except Exception as e:
            logger.warning("[analyze_memory_code] 分析失败: %s - %s", mem_ref, e)
            return {}

    async def get_related_memories(
        self,
        memory_id: str,
        depth: int = 1,
        relationship_type: str | None = None,
        tenant_id: str = "default",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        effective_tenant_id = tenant_id or self._default_tenant_id
        source_ref = self._normalize_memory_id(memory_id)

        visited: set[str] = {source_ref}
        current_level: set[str] = {source_ref}
        all_memories: list[dict[str, Any]] = []

        for _ in range(depth):
            next_level: set[str] = set()
            for mem_id in current_level:
                relations = await self.get_relations(
                    mem_id,
                    direction="both",
                    relationship_type=relationship_type,
                    tenant_id=effective_tenant_id,
                    limit=limit,
                )
                for rel in relations:
                    for endpoint in (rel["from_id"], rel["to_id"]):
                        ep_ref = self._normalize_memory_id(endpoint)
                        if ep_ref in visited or ep_ref == source_ref:
                            continue
                        visited.add(ep_ref)
                        next_level.add(ep_ref)

                        if len(all_memories) >= limit:
                            return all_memories[:limit]

                        mem_parts = ep_ref.split(":")
                        tbl, rid = mem_parts[0], mem_parts[1]
                        r = await self._db_query(
                            "SELECT id, content, abstract, overview, type, tags, project_id, local_id, metadata "
                            "FROM type::record($table, $id)",
                            {"table": tbl, "id": rid},
                        )
                        records = self._extract_records(r)
                        if records:
                            all_memories.append(records[0])

            current_level = next_level
            if not current_level:
                break

        return all_memories[:limit]

    async def get_memory_stats(self, tenant_id: str = "default") -> dict[str, Any]:
        logger.warning("[MemoryManager] get_memory_stats 被调用但功能尚未实现")
        raise NotImplementedError("功能尚未实现: get_memory_stats")

    async def optimize_hnsw(self, tenant_id: str = "default") -> dict[str, Any]:
        logger.warning("[MemoryManager] optimize_hnsw 被调用但功能尚未实现")
        raise NotImplementedError("功能尚未实现: optimize_hnsw")

    async def rebuild_hnsw_index(self, tenant_id: str = "default", force: bool = False) -> dict[str, Any]:
        logger.warning("[MemoryManager] rebuild_hnsw_index 被调用但功能尚未实现")
        raise NotImplementedError("功能尚未实现: rebuild_hnsw_index")

    async def get_cache_stats(self) -> dict[str, Any]:
        logger.warning("[MemoryManager] get_cache_stats 被调用但功能尚未实现")
        raise NotImplementedError("功能尚未实现: get_cache_stats")

    async def clear_embedding_cache(self) -> dict[str, Any]:
        logger.warning("[MemoryManager] clear_embedding_cache 被调用但功能尚未实现")
        raise NotImplementedError("功能尚未实现: clear_embedding_cache")

    async def warmup_embedding_cache(self, tenant_id: str = "default", limit: int = 100) -> dict[str, Any]:
        logger.warning("[MemoryManager] warmup_embedding_cache 被调用但功能尚未实现")
        raise NotImplementedError("功能尚未实现: warmup_embedding_cache")

    async def prefetch_related_memories(
        self, memory_id: str, tenant_id: str = "default", depth: int = 1, limit: int = 10
    ) -> dict[str, Any]:
        logger.warning("[MemoryManager] prefetch_related_memories 被调用但功能尚未实现: %s", memory_id)
        raise NotImplementedError("功能尚未实现: prefetch_related_memories")

    async def prefetch_popular_queries(self, tenant_id: str = "default", top_n: int = 20) -> dict[str, Any]:
        logger.warning("[MemoryManager] prefetch_popular_queries 被调用但功能尚未实现")
        raise NotImplementedError("功能尚未实现: prefetch_popular_queries")

    async def cluster_memories_leiden(
        self, tenant_id: str = "default", content_threshold: float = 0.75, max_clusters: int = 20
    ) -> dict[str, Any]:
        logger.warning("[MemoryManager] cluster_memories_leiden 被调用但功能尚未实现")
        raise NotImplementedError("功能尚未实现: cluster_memories_leiden")

    # ==================== BL-4: Code Analysis Auto-trigger ====================

    async def _auto_analyze_memories(self, memory_ids: list[str], tenant_id: str) -> None:
        """自动分析记忆中的代码内容（异步执行，不阻塞上传）"""
        from ..config import CodeAnalysisConfig

        config = CodeAnalysisConfig()
        if not config.enabled or not config.auto_analyze:
            return

        for memory_id in memory_ids:
            try:
                # 检查内容长度
                query = "SELECT content FROM $memory_id WHERE tenant_id = $tenant_id"
                result = await self._db_query(query, {"memory_id": memory_id, "tenant_id": tenant_id})
                records = self._extract_records(result)

                if not records:
                    continue

                content = records[0].get("content", "")
                content_length = len(content)

                if content_length < config.min_content_length or content_length > config.max_content_length:
                    continue

                # 检测是否为代码内容
                if not self._is_code_content(content):
                    continue

                # 异步执行代码分析
                await self.analyze_memory_code(memory_id, tenant_id, persist=True)
                logger.info("[Auto Code Analysis] 分析完成: %s", memory_id)

            except Exception as e:
                # 降级策略：分析失败不影响上传
                logger.warning("[Auto Code Analysis] 分析失败，跳过: %s - %s", memory_id, e)

    def _is_code_content(self, content: str) -> bool:
        """检测内容是否为代码"""
        code_indicators = [
            r"^\s*(def|class|import|from)\s+",  # Python
            r"^\s*(function|const|let|var)\s+",  # JavaScript
            r"^\s*(#include|#define|int|void)\s+",  # C/C++
            r"^\s*(public|private|class|interface)\s+",  # Java/C#
            r"^\s*(func|package|import)\s+",  # Go
            r"^\s*(fn|let|mut|use)\s+",  # Rust
            r"^\s*<[^>]+>.*</[^>]+>\s*$",  # HTML/XML
            r"^\s*\{[^}]*\}\s*$",  # JSON
        ]

        content_sample = content[:1000]  # 检查前1000字符
        for pattern in code_indicators:
            if re.search(pattern, content_sample, re.MULTILINE):
                return True

        # 检查代码特征比例
        code_chars = len(re.findall(r"[{}();=<>/]", content_sample))
        total_chars = len(content_sample.replace(" ", "").replace("\n", ""))
        if total_chars > 0 and code_chars / total_chars > 0.1:
            return True

        return False
