"""上传、更新、删除、嵌入"""

import asyncio
import hashlib
import logging
from typing import Any

from wrapper.src.utils.exceptions import EmbeddingError, ValidationError
from wrapper.src.utils.tracing import get_tracer

logger = logging.getLogger(__name__)


class CrudMixin:
    """上传更新相关方法"""

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

                    # BL-CA-35: 同步 file_path 到顶层字段（用于 lookup 查询）
                    metadata = memory.get("metadata", {})
                    if metadata.get("file_path"):
                        memory_data["file_path"] = metadata["file_path"]

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
                                "SELECT id, metadata FROM memory WHERE type = 'code' AND project_id = $project_id AND metadata.file_path = $file_path AND tenant_id = $tenant_id LIMIT 1",
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
                            # 新文件：继续到批量插入（不 continue）

                    # 代码分析数据跳过去重（插件端建议方案1）
                    # 原因：代码分析的本质是版本追踪，每次分析都是独立的
                    # 同一文件多次分析应保留历史，不同项目相同代码应独立存储
                    if mem_type == "code":
                        # 代码数据直接进入批量插入队列，不检查 hash 去重
                        batch_inserts.append(memory_data)
                        continue

                    # 通用去重：检查 content_hash（仅非代码类型）
                    existing = await self._db_query(
                        "SELECT id FROM memory WHERE tenant_id = $tenant_id AND content_hash = $hash LIMIT 1",
                        {"tenant_id": effective_tenant_id, "hash": memory_data["content_hash"]},
                    )
                    existing_records = self._extract_records(existing)
                    if existing_records:
                        existing_id = str(existing_records[0].get("id", ""))

                        # TC-LOOKUP-001: 更新已有记录的 source_id 和 local_id
                        await self._update_memory(existing_id, memory_data)
                        memory_ids.append(existing_id)
                        updated_count += 1
                        success_count += 1
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

            # Phase A-B2: 批量插入（分批处理，每批50条）
            if batch_inserts:
                BATCH_SIZE = 50
                total_batches = (len(batch_inserts) + BATCH_SIZE - 1) // BATCH_SIZE

                for batch_idx in range(total_batches):
                    start_idx = batch_idx * BATCH_SIZE
                    end_idx = min(start_idx + BATCH_SIZE, len(batch_inserts))
                    current_batch = batch_inserts[start_idx:end_idx]

                    try:
                        query = "INSERT INTO memory $data"
                        result = await self._db_query(query, {"data": current_batch})

                        records = self._extract_records(result)
                        if records:
                            for i, record in enumerate(records):
                                record_id = str(record.get("id", "")) if isinstance(record, dict) else None
                                if record_id:
                                    memory_ids.append(record_id)
                                    success_count += 1

                                    if self._meili:
                                        mem = current_batch[i]
                                        meili_doc = self._build_meili_doc(record_id, mem, effective_tenant_id)
                                        meili_docs.append(meili_doc)
                                else:
                                    failed_count += 1
                                    errors.append(f"Batch {batch_idx + 1}: No record ID returned")
                    except Exception as e:
                        logger.warning(
                            f"Batch {batch_idx + 1}/{total_batches} insert failed, falling back to single insert: {e}"
                        )
                        for memory_data in current_batch:
                            try:
                                result = await self._db_create("memory", memory_data)
                                record_id = self._extract_record_id(result)
                                if record_id:
                                    memory_ids.append(record_id)
                                    success_count += 1
                                    if self._meili:
                                        meili_doc = self._build_meili_doc(record_id, memory_data, effective_tenant_id)
                                        meili_docs.append(meili_doc)
                                else:
                                    failed_count += 1
                            except Exception as inner_e:
                                failed_count += 1
                                errors.append(f"{type(inner_e).__name__}: {inner_e!s}")

            # 同步到 Meilisearch（分批处理，每批50条）
            if self._meili and meili_docs:
                MEILI_BATCH_SIZE = 50
                total_meili_batches = (len(meili_docs) + MEILI_BATCH_SIZE - 1) // MEILI_BATCH_SIZE
                meili_synced_count = 0
                meili_failed_count = 0

                for meili_batch_idx in range(total_meili_batches):
                    start_idx = meili_batch_idx * MEILI_BATCH_SIZE
                    end_idx = min(start_idx + MEILI_BATCH_SIZE, len(meili_docs))
                    current_meili_batch = meili_docs[start_idx:end_idx]

                    try:
                        await self._meili.add_documents(current_meili_batch)
                        meili_synced_count += len(current_meili_batch)
                        logger.info(
                            f"[Meili sync] Batch {meili_batch_idx + 1}/{total_meili_batches} synced: {len(current_meili_batch)} docs"
                        )
                    except Exception as meili_err:
                        meili_failed_count += len(current_meili_batch)
                        logger.error(
                            f"[Meili sync] Batch {meili_batch_idx + 1}/{total_meili_batches} failed: {meili_err}"
                        )

                span.set_attribute("memory.upload.meili_synced", meili_synced_count)
                span.set_attribute("memory.upload.meili_failed", meili_failed_count)
                if meili_failed_count > 0:
                    span.set_attribute("memory.upload.meili_error", f"{meili_failed_count} docs failed to sync")

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
                source_id = $source_id,
                local_id = $local_id,
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
                "source_id": memory_data.get("source_id"),
                "local_id": memory_data.get("local_id"),
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
