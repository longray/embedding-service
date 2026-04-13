"""Meilisearch SDK 客户端 (v0.40+)

使用官方 meilisearch SDK 替代 httpx REST 调用。
提供索引配置、文档管理和全文搜索功能。
"""

import logging
from typing import Any, ClassVar

from meilisearch.client import Client
from meilisearch.errors import MeilisearchApiError

logger = logging.getLogger(__name__)


class MeilisearchSDKError(Exception):
    """Meilisearch SDK 操作异常"""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class MeilisearchSDKClient:
    """Meilisearch SDK 客户端 (v0.40+)

    Usage:
        client = MeilisearchSDKClient(url="http://localhost:7700", api_key="your_api_key")
        client.connect()
        client.ensure_index()
        client.configure_index({...})
        results = client.search("query")
        client.close()
    """

    DEFAULT_INDEX_SETTINGS: ClassVar[dict[str, Any]] = {
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
        timeout: int = 30,
    ):
        self._url = url.rstrip("/")
        self._api_key = api_key
        self._index_name = index_name
        self._timeout = timeout
        self._client: Client | None = None

    def connect(self) -> None:
        """初始化 SDK 客户端并验证连接"""
        self._client = Client(
            url=self._url,
            api_key=self._api_key,
            timeout=self._timeout,
        )

        try:
            health = self._client.health()
            logger.info("[Meilisearch SDK] 已连接: %s, 状态: %s", self._url, health.get("status"))
        except Exception as e:
            self.close()
            raise MeilisearchSDKError(f"无法连接 Meilisearch ({self._url}): {e}") from e

    def close(self) -> None:
        """关闭 SDK 客户端"""
        self._client = None
        logger.info("[Meilisearch SDK] 连接已关闭")

    @property
    def client(self) -> Client:
        """获取 SDK 客户端"""
        if self._client is None:
            raise MeilisearchSDKError("Meilisearch 客户端未初始化，请先调用 connect()")
        return self._client

    def ensure_index(self, primary_key: str = "id") -> None:
        """确保索引存在（幂等操作）"""
        try:
            self.client.create_index(
                uid=self._index_name,
                options={"primaryKey": primary_key},
            )
            logger.info("[Meilisearch SDK] 索引已创建: %s", self._index_name)
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info("[Meilisearch SDK] 索引已存在: %s", self._index_name)
            else:
                raise MeilisearchSDKError(f"创建索引失败: {e}") from e

    def configure_index(self, settings: dict[str, Any] | None = None) -> None:
        """配置索引设置"""
        effective_settings = settings or self.DEFAULT_INDEX_SETTINGS

        index = self.client.index(self._index_name)
        index.update_settings(effective_settings)
        logger.info("[Meilisearch SDK] 索引配置已更新: %s", list(effective_settings.keys()))

    def _to_meili_id(self, surreal_id: str) -> str:
        """SurrealDB ID → Meilisearch ID"""
        return surreal_id.replace(":", "_", 1)

    def _from_meili_id(self, meili_id: str) -> str:
        """Meilisearch ID → SurrealDB ID"""
        return meili_id.replace("_", ":", 1)

    def add_documents(
        self,
        documents: list[dict[str, Any]],
        primary_key: str = "id",
        *,
        wait: bool = True,
    ) -> dict[str, Any]:
        """添加或更新文档"""
        if not documents:
            return {"status": "skipped", "reason": "empty documents list"}

        converted_docs = []
        for doc in documents:
            converted_doc = doc.copy()
            if "id" in converted_doc:
                converted_doc["id"] = self._to_meili_id(converted_doc["id"])
            converted_docs.append(converted_doc)

        index = self.client.index(self._index_name)
        task = index.add_documents(
            documents=converted_docs,
            primary_key=primary_key,
        )

        if wait:
            self.client.wait_for_task(task.task_uid)

        return {"taskUid": task.task_uid, "status": "enqueued"}

    def batch_add_documents(
        self,
        documents: list[dict[str, Any]],
        primary_key: str = "id",
        batch_size: int = 100,
        *,
        wait: bool = True,
    ) -> dict[str, Any]:
        """批量添加文档（分批处理）

        Args:
            documents: 文档列表
            primary_key: 主键字段名
            batch_size: 每批处理的文档数，默认 100
            wait: 是否等待所有任务完成

        Returns:
            批量处理结果
        """
        if not documents:
            return {"status": "skipped", "reason": "empty documents list", "processed": 0}

        total = len(documents)
        processed = 0
        task_uids = []

        for i in range(0, total, batch_size):
            batch = documents[i : i + batch_size]
            result = self.add_documents(batch, primary_key, wait=False)
            task_uids.append(result["taskUid"])
            processed += len(batch)
            logger.debug("[Meilisearch SDK] 已提交批次: %d/%d", processed, total)

        if wait:
            for task_uid in task_uids:
                self.client.wait_for_task(task_uid)

        return {
            "status": "enqueued",
            "processed": processed,
            "total": total,
            "batches": len(task_uids),
            "taskUids": task_uids,
        }

    def batch_delete_documents(
        self,
        document_ids: list[str],
        batch_size: int = 100,
        *,
        wait: bool = True,
    ) -> dict[str, Any]:
        """批量删除文档（分批处理）

        Args:
            document_ids: 文档 ID 列表
            batch_size: 每批处理的文档数，默认 100
            wait: 是否等待所有任务完成

        Returns:
            批量处理结果
        """
        if not document_ids:
            return {"status": "skipped", "reason": "empty document_ids list", "processed": 0}

        total = len(document_ids)
        processed = 0
        task_uids = []

        for i in range(0, total, batch_size):
            batch = document_ids[i : i + batch_size]
            meili_ids = [self._to_meili_id(doc_id) for doc_id in batch]
            index = self.client.index(self._index_name)
            task = index.delete_documents(meili_ids)
            task_uids.append(task.task_uid)
            processed += len(batch)
            logger.debug("[Meilisearch SDK] 已提交删除批次: %d/%d", processed, total)

        if wait:
            for task_uid in task_uids:
                self.client.wait_for_task(task_uid)

        return {
            "status": "enqueued",
            "processed": processed,
            "total": total,
            "batches": len(task_uids),
            "taskUids": task_uids,
        }

    def delete_all_documents(self) -> None:
        """删除所有文档"""
        index = self.client.index(self._index_name)
        task = index.delete_all_documents()
        self.client.wait_for_task(task.task_uid)

    def delete_document(self, document_id: str) -> None:
        """删除单个文档"""
        meili_id = self._to_meili_id(document_id)
        index = self.client.index(self._index_name)
        task = index.delete_document(meili_id)
        self.client.wait_for_task(task.task_uid)

    def delete_documents_by_filter(self, filter_expr: str) -> None:
        """通过过滤条件批量删除文档"""
        index = self.client.index(self._index_name)
        task = index.delete_documents(filter_expr)
        self.client.wait_for_task(task.task_uid)

    def search(
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
        """全文搜索"""
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

        result = index.search(query, search_params)

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

    def health(self) -> dict[str, Any]:
        """检查健康状态"""
        try:
            health = self.client.health()
            return {"status": health.get("status")}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def get_stats(self) -> dict[str, Any]:
        """获取索引统计信息"""
        try:
            index = self.client.index(self._index_name)
            stats = index.get_stats()
            return {
                "numberOfDocuments": getattr(stats, "number_of_documents", None),
                "isIndexing": getattr(stats, "is_indexing", None),
                "fieldDistribution": getattr(stats, "field_distribution", None),
            }
        except Exception as e:
            return {"error": str(e)}

    def get_settings(self) -> dict[str, Any]:
        """获取索引设置"""
        try:
            index = self.client.index(self._index_name)
            settings = index.get_settings()
            return {
                "searchableAttributes": getattr(settings, "searchable_attributes", []),
                "filterableAttributes": getattr(settings, "filterable_attributes", []),
                "sortableAttributes": getattr(settings, "sortable_attributes", []),
                "typoTolerance": getattr(settings, "typo_tolerance", {}),
                "dictionary": getattr(settings, "dictionary", []),
            }
        except Exception as e:
            return {"error": str(e)}

    def reset_settings(self) -> None:
        """重置索引设置为默认值"""
        index = self.client.index(self._index_name)
        task = index.reset_settings()
        self.client.wait_for_task(task.task_uid)
        logger.info("[Meilisearch SDK] 索引设置已重置: %s", self._index_name)
