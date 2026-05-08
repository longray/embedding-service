"""Meilisearch 客户端（SDK 版本）

使用官方 meilisearch SDK 替代 httpx REST 调用。
保持与旧版 API 兼容，提供异步接口。

设计原则：
- 使用 meilisearch-python SDK（v0.40+）
- 异步优先（适配 FastAPI lifespan）
- 保持向后兼容（与旧版 meili_client.py 接口一致）
- 生产级错误处理和日志
"""

import asyncio
import logging
from typing import Any, ClassVar

from meilisearch.client import Client
from meilisearch.errors import MeilisearchApiError

logger = logging.getLogger(__name__)


class MeilisearchError(Exception):
    """Meilisearch 操作异常"""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class MeilisearchClient:
    """Meilisearch SDK 客户端（异步包装）

    Usage:
        client = MeilisearchClient(url="http://localhost:7700", api_key="your_api_key")
        await client.connect()
        await client.ensure_index()
        await client.configure_index({...})
        results = await client.search("query")
        await client.close()
    """

    # 记忆索引的默认配置
    DEFAULT_INDEX_SETTINGS: ClassVar[dict[str, Any]] = {
        # 将更多字段设为可搜索字段，支持代码/元数据等多类型搜索
        "searchableAttributes": [
            "content_zh",
            "title_zh",
            "tags_zh",
            "content_search",
            "code",
            "content",
            "code_symbols",
        ],
        "filterableAttributes": [
            "tenant_id",
            "type",
            "tags",
            "project_id",
            "file_path",
            "date",
            "ip_address",
            "email",
            "version",
            "created_at",
            "source_id",
            "code_language",
            "code_complexity",
            "code_function_count",
            "code_class_count",
            "code_analyzer",
            "code_has_exports",
        ],
        "sortableAttributes": ["date", "created_at", "code_complexity", "code_function_count"],
        "nonSeparatorTokens": [".", "-", "@", ":", "/", "_"],
        "localizedAttributes": [{"locales": ["zho"], "attributePatterns": ["*_zh"]}],
        "typoTolerance": {"enabled": True, "disableOnAttributes": ["file_path", "version", "email", "ip_address"]},
        "dictionary": [
            "v1",
            "v2",
            "v3",
            "v4",
            "v5",
            "alpha",
            "beta",
            "rc",
            "release",
            "snapshot",
            "python",
            "java",
            "javascript",
            "typescript",
            "go",
            "rust",
            "cpp",
            "csharp",
            "ruby",
            "php",
            "swift",
            "kotlin",
            "scala",
            "http",
            "https",
            "api",
            "www",
            "localhost",
            "com",
            "cn",
            "org",
            "net",
            "io",
            "dev",
            "get",
            "post",
            "put",
            "delete",
            "patch",
            "class",
            "interface",
            "enum",
            "struct",
            "function",
            "method",
            "property",
            "attribute",
            "import",
            "export",
            "require",
            "include",
            "public",
            "private",
            "protected",
            "static",
            "async",
            "await",
            "promise",
            "callback",
            "django",
            "flask",
            "fastapi",
            "spring",
            "react",
            "vue",
            "angular",
            "next",
            "nuxt",
            "tensorflow",
            "pytorch",
            "sklearn",
            "ID",
            "NO",
            "NUM",
            "CODE",
            "KEY",
            "ORD",
            "PAY",
            "TRK",
            "INV",
            "USR",
            "2025",
            "2026",
            "2027",
            "2028",
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
            "192",
            "168",
            "172",
            "10",
            "127",
            "0",
            "1",
        ],
    }

    def __init__(
        self,
        url: str = "http://localhost:7700",
        api_key: str | None = None,
        index_name: str = "memories",
        timeout: float = 30.0,
        task_poll_interval: float = 0.5,
        task_timeout: float = 60.0,
    ):
        self._url = url.rstrip("/")
        self._api_key = api_key
        self._index_name = index_name
        self._timeout = int(timeout)  # SDK 使用整数秒
        self._task_poll_interval = task_poll_interval
        self._task_timeout = task_timeout
        self._client: Client | None = None

    # ==================== 生命周期 ====================

    async def connect(self) -> None:
        """初始化 SDK 客户端并验证连接"""
        self._client = Client(
            url=self._url,
            api_key=self._api_key,
            timeout=self._timeout,
        )

        # 验证 Meilisearch 可达
        try:
            health = await asyncio.to_thread(self._client.health)
            logger.info("[Meilisearch] 已连接: %s, 状态: %s", self._url, health.get("status"))
        except Exception as e:
            await self.close()
            raise MeilisearchError(f"无法连接 Meilisearch ({self._url}): {e}") from e

    async def close(self) -> None:
        """关闭 SDK 客户端"""
        self._client = None
        logger.info("[Meilisearch] 连接已关闭")

    @property
    def client(self) -> Client:
        """获取 SDK 客户端（未初始化则抛异常）"""
        if self._client is None:
            raise MeilisearchError("Meilisearch 客户端未初始化，请先调用 connect()")
        return self._client

    # ==================== 索引管理 ====================

    async def ensure_index(self, primary_key: str = "id") -> None:
        """确保索引存在（幂等操作）"""
        try:
            await asyncio.to_thread(
                self.client.create_index,
                uid=self._index_name,
                options={"primaryKey": primary_key},
            )
            logger.info("[Meilisearch] 索引已创建: %s", self._index_name)
        except Exception as e:
            if "already exists" in str(e).lower() or "index_already_exists" in str(e):
                logger.info("[Meilisearch] 索引已存在: %s", self._index_name)
            else:
                raise MeilisearchError(f"创建索引失败: {e}") from e

    async def configure_index(self, settings: dict[str, Any] | None = None) -> None:
        """配置索引设置

        Args:
            settings: 索引配置字典。为 None 时使用 DEFAULT_INDEX_SETTINGS。
        """
        effective_settings = settings or self.DEFAULT_INDEX_SETTINGS

        index = self.client.index(self._index_name)
        task = await asyncio.to_thread(index.update_settings, effective_settings)
        await asyncio.to_thread(self.client.wait_for_task, task.task_uid)
        logger.info("[Meilisearch] 索引配置已更新: %s", list(effective_settings.keys()))

    # ==================== ID 转换 ====================

    def _to_meili_id(self, surreal_id: str) -> str:
        """SurrealDB ID → Meilisearch ID

        转换规则:
        - memory:abc123 → memory_abc123
        - 只替换第一个冒号，保留后续字符
        """
        return surreal_id.replace(":", "_", 1)

    def _from_meili_id(self, meili_id: str) -> str:
        """Meilisearch ID → SurrealDB ID

        转换规则:
        - memory_abc123 → memory:abc123
        - 将第一个下划线还原为冒号
        """
        return meili_id.replace("_", ":", 1)

    # ==================== 文档管理 ====================

    async def add_documents(
        self,
        documents: list[dict[str, Any]],
        primary_key: str = "id",
        *,
        wait: bool = True,
    ) -> dict[str, Any]:
        """添加或更新文档（upsert 语义）

        Args:
            documents: 文档列表，每个文档必须包含主键字段
            primary_key: 主键字段名
            wait: 是否等待任务完成

        Returns:
            任务信息字典
        """
        if not documents:
            return {"status": "skipped", "reason": "empty documents list"}

        converted_docs = []
        for doc in documents:
            converted_doc = doc.copy()
            if "id" in converted_doc:
                converted_doc["id"] = self._to_meili_id(converted_doc["id"])
            converted_docs.append(converted_doc)

        index = self.client.index(self._index_name)
        task = await asyncio.to_thread(
            index.add_documents,
            documents=converted_docs,
            primary_key=primary_key,
        )

        if wait:
            await asyncio.to_thread(self.client.wait_for_task, task.task_uid)

        return {"taskUid": task.task_uid, "status": "succeeded" if wait else "enqueued"}

    async def delete_all_documents(self) -> None:
        """删除所有文档"""
        index = self.client.index(self._index_name)
        task = await asyncio.to_thread(index.delete_all_documents)
        await asyncio.to_thread(self.client.wait_for_task, task.task_uid)

    async def delete_document(self, document_id: str) -> None:
        """删除单个文档"""
        meili_id = self._to_meili_id(document_id)
        index = self.client.index(self._index_name)
        task = await asyncio.to_thread(index.delete_document, meili_id)
        await asyncio.to_thread(self.client.wait_for_task, task.task_uid)

    async def delete_documents_by_filter(self, filter_expr: str) -> None:
        """通过过滤条件批量删除文档

        Args:
            filter_expr: Meilisearch 过滤表达式，如 "tenant_id = 'default'"
        """
        index = self.client.index(self._index_name)
        task = await asyncio.to_thread(index.delete_documents, filter_expr)
        await asyncio.to_thread(self.client.wait_for_task, task.task_uid)

    # ==================== 搜索 ====================

    async def search(
        self,
        query: str,
        *,
        filter_expr: str | None = None,
        limit: int = 10,
        offset: int = 0,
        sort: list[str] | None = None,
        attributes_to_retrieve: list[str] | None = None,
        show_ranking_score: bool = True,
    ) -> dict[str, Any]:
        """全文搜索

        Args:
            query: 搜索查询文本
            filter_expr: 过滤表达式，如 "tenant_id = 'default' AND type = 'general'"
            limit: 返回结果数量上限
            offset: 结果偏移量
            sort: 排序规则列表，如 ["created_at:desc"]
            attributes_to_retrieve: 要返回的字段列表
            show_ranking_score: 是否返回排名分数

        Returns:
            搜索结果字典，包含 hits, estimatedTotalHits, query 等字段
        """
        index = self.client.index(self._index_name)

        search_params: dict[str, Any] = {
            "limit": limit,
            "offset": offset,
            "showRankingScore": show_ranking_score,
        }
        if filter_expr:
            search_params["filter"] = filter_expr
        if sort:
            search_params["sort"] = sort
        if attributes_to_retrieve:
            search_params["attributesToRetrieve"] = attributes_to_retrieve

        result = await asyncio.to_thread(index.search, query, search_params)

        # 转换 ID 格式
        hits = []
        for hit in result.get("hits", []):
            if "id" in hit:
                hit["id"] = self._from_meili_id(hit["id"])
            hits.append(hit)

        return {
            "hits": hits,
            "estimatedTotalHits": result.get("estimatedTotalHits"),
            "totalHits": result.get("totalHits"),
            "limit": result.get("limit"),
            "offset": result.get("offset"),
            "processingTimeMs": result.get("processingTimeMs"),
            "query": result.get("query"),
        }

    # ==================== 健康检查 ====================

    async def health(self) -> dict[str, Any]:
        """检查 Meilisearch 健康状态"""
        try:
            health = await asyncio.to_thread(self.client.health)
            return {"status": health.get("status")}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def get_stats(self) -> dict[str, Any]:
        """获取索引统计信息"""
        try:
            index = self.client.index(self._index_name)
            stats = await asyncio.to_thread(index.get_stats)
            return {
                "numberOfDocuments": getattr(stats, "number_of_documents", None),
                "isIndexing": getattr(stats, "is_indexing", None),
                "fieldDistribution": getattr(stats, "field_distribution", None),
            }
        except Exception as e:
            return {"error": str(e)}

    async def get_settings(self) -> dict[str, Any]:
        """获取索引设置"""
        try:
            index = self.client.index(self._index_name)
            settings = await asyncio.to_thread(index.get_settings)
            return {
                "searchableAttributes": getattr(settings, "searchable_attributes", []),
                "filterableAttributes": getattr(settings, "filterable_attributes", []),
                "sortableAttributes": getattr(settings, "sortable_attributes", []),
                "typoTolerance": getattr(settings, "typo_tolerance", {}),
                "dictionary": getattr(settings, "dictionary", []),
            }
        except Exception as e:
            return {"error": str(e)}

    # ==================== 文档删除 ====================

    async def delete_all_documents(self) -> None:
        """删除索引中的所有文档
        
        注意: Meilisearch 1.13.3 中 delete_all_documents 可能不工作，
        使用删除并重建索引的方式确保数据被清空。
        """
        try:
            # 方法1: 尝试使用 delete_all_documents
            index = self.client.index(self._index_name)
            task = await asyncio.to_thread(index.delete_all_documents)
            await asyncio.to_thread(self.client.wait_for_task, task.task_uid)
            
            # 验证是否删除成功
            stats = await asyncio.to_thread(index.get_stats)
            if stats.number_of_documents == 0:
                logger.info(f"[Meilisearch] 已删除索引 {self._index_name} 中的所有文档")
                return
            
            # 方法2: 如果还有文档，删除并重建索引
            logger.warning(f"[Meilisearch] delete_all_documents 未完全清空，使用重建索引方式")
            await asyncio.to_thread(self.client.delete_index, self._index_name)
            await asyncio.to_thread(
                self.client.create_index, 
                self._index_name, 
                {"primaryKey": "id"}
            )
            # 重新配置索引
            await self.configure_index()
            logger.info(f"[Meilisearch] 已重建索引 {self._index_name}")
            
        except Exception as e:
            logger.error(f"[Meilisearch] 删除所有文档失败: {e}")
            raise

    async def delete_document(self, document_id: str) -> None:
        """删除单个文档

        Args:
            document_id: 文档 ID (SurrealDB 格式，如 "memory:xxx")
        """
        try:
            index = self.client.index(self._index_name)
            meili_id = self._to_meili_id(document_id)
            task = await asyncio.to_thread(index.delete_document, meili_id)
            await asyncio.to_thread(self.client.wait_for_task, task.task_uid)
            logger.debug(f"[Meilisearch] 已删除文档: {document_id}")
        except Exception as e:
            logger.error(f"[Meilisearch] 删除文档 {document_id} 失败: {e}")
            raise

    async def delete_documents_by_filter(self, filter_expr: str) -> None:
        """通过过滤条件批量删除文档

        Args:
            filter_expr: 过滤表达式，如 "tenant_id = 'default'"
                        空字符串 "" 表示删除所有文档
        """
        try:
            index = self.client.index(self._index_name)
            if filter_expr:
                # 使用 filter 删除匹配的文档
                task = await asyncio.to_thread(index.delete_documents, {"filter": filter_expr})
            else:
                # 空 filter 删除所有文档
                task = await asyncio.to_thread(index.delete_all_documents)
            await asyncio.to_thread(self.client.wait_for_task, task.task_uid)
            logger.info(f"[Meilisearch] 已通过 filter 删除文档: {filter_expr or 'all'}")
        except Exception as e:
            logger.error(f"[Meilisearch] 通过 filter 删除文档失败: {e}")
            raise
