"""Meilisearch 异步客户端

使用 httpx 直接调用 Meilisearch REST API，避免额外 SDK 依赖。
提供索引配置、文档管理和全文搜索功能。

设计原则：
- 不新增依赖（复用项目已有的 httpx）
- 异步优先（适配 FastAPI lifespan）
- 任务轮询（Meilisearch 写操作是异步任务）
- 生产级错误处理和日志
"""

import asyncio
import logging
from typing import Any, ClassVar

import httpx

logger = logging.getLogger(__name__)


class MeilisearchError(Exception):
    """Meilisearch 操作异常"""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class MeilisearchClient:
    """异步 Meilisearch 客户端

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
        ],  # 新增
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
        ],
        "sortableAttributes": ["date", "created_at", "code_complexity", "code_function_count"],
        # 让日期格式 2026-03-11 在全文搜索时保持整体，不被 - 分割
        "nonSeparatorTokens": [".", "-", "@", ":", "/", "_"],
        "localizedAttributes": [{"locales": ["zho"], "attributePatterns": ["*_zh"]}],
        "typoTolerance": {"enabled": True, "disableOnAttributes": ["file_path", "version", "email", "ip_address"]},
        "dictionary": [
            # 版本前缀
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
            # 编程语言
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
            # 常见命名
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
            # 代码术语
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
            # 框架/库
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
            # 常见 ID 前缀
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
            # 时间
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
            # IP 段
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
        self._timeout = timeout
        self._task_poll_interval = task_poll_interval
        self._task_timeout = task_timeout
        self._client: httpx.AsyncClient | None = None

    # ==================== 生命周期 ====================

    async def connect(self) -> None:
        """初始化 HTTP 客户端并验证连接"""
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        self._client = httpx.AsyncClient(  # nosec B113 - timeout is set via httpx.Timeout
            base_url=self._url,
            headers=headers,
            timeout=httpx.Timeout(self._timeout),
        )

        # 验证 Meilisearch 可达
        try:
            resp = await self._client.get("/health")
            resp.raise_for_status()
            health = resp.json()
            logger.info("[Meilisearch] 已连接: %s, 状态: %s", self._url, health.get("status"))
        except Exception as e:
            await self.close()
            raise MeilisearchError(f"无法连接 Meilisearch ({self._url}): {e}") from e

    async def close(self) -> None:
        """关闭 HTTP 客户端"""
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.info("[Meilisearch] 连接已关闭")

    @property
    def client(self) -> httpx.AsyncClient:
        """获取 HTTP 客户端��未初始化则抛异常）"""
        if self._client is None:
            raise MeilisearchError("Meilisearch 客户端未初始化，请先调用 connect()")
        return self._client

    # ==================== 索引管理 ====================

    async def ensure_index(self, primary_key: str = "id") -> None:
        """确保索引存在（幂等操作）"""
        try:
            resp = await self.client.post(
                "/indexes",
                json={"uid": self._index_name, "primaryKey": primary_key},
            )
            if resp.status_code == 202:
                task = resp.json()
                await self._wait_for_task(task["taskUid"])
                logger.info("[Meilisearch] 索引已创建: %s", self._index_name)
            else:
                # Meilisearch 返回 202 表示任务已入队
                # 如果索引已存在，任务完成后不会报错
                resp.raise_for_status()
        except MeilisearchError as e:
            # index_already_exists 不是错误
            if "index_already_exists" in str(e):
                logger.info("[Meilisearch] 索引已存在: %s", self._index_name)
            else:
                raise

    async def configure_index(self, settings: dict[str, Any] | None = None) -> None:
        """配置索引设置

        Args:
            settings: 索引配置字典。为 None 时使用 DEFAULT_INDEX_SETTINGS。
        """
        effective_settings = settings or self.DEFAULT_INDEX_SETTINGS

        resp = await self.client.patch(
            f"/indexes/{self._index_name}/settings",
            json=effective_settings,
        )
        resp.raise_for_status()
        task = resp.json()
        await self._wait_for_task(task["taskUid"])
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

        resp = await self.client.post(
            f"/indexes/{self._index_name}/documents",
            json=converted_docs,
            params={"primaryKey": primary_key},
        )
        resp.raise_for_status()
        task_info = resp.json()

        if wait:
            return await self._wait_for_task(task_info["taskUid"])

        return task_info

    async def delete_all_documents(self) -> None:
        resp = await self.client.delete(
            f"/indexes/{self._index_name}/documents",
        )
        resp.raise_for_status()
        task_info = resp.json()
        await self._wait_for_task(task_info["taskUid"])

    async def delete_document(self, document_id: str) -> None:
        """删除单个文档"""
        resp = await self.client.delete(
            f"/indexes/{self._index_name}/documents/{document_id}",
        )
        resp.raise_for_status()
        task_info = resp.json()
        await self._wait_for_task(task_info["taskUid"])

    async def delete_documents_by_filter(self, filter_expr: str) -> None:
        """通过过滤条件批量删除文档

        Args:
            filter_expr: Meilisearch 过滤表达式，如 "tenant_id = 'default'"
        """
        resp = await self.client.post(
            f"/indexes/{self._index_name}/documents/delete",
            json={"filter": filter_expr},
        )
        resp.raise_for_status()
        task_info = resp.json()
        await self._wait_for_task(task_info["taskUid"])

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
        body: dict[str, Any] = {
            "q": query,
            "limit": limit,
            "offset": offset,
            "showRankingScore": show_ranking_score,
        }
        if filter_expr:
            body["filter"] = filter_expr
        if sort:
            body["sort"] = sort
        if attributes_to_retrieve:
            body["attributesToRetrieve"] = attributes_to_retrieve

        resp = await self.client.post(
            f"/indexes/{self._index_name}/search",
            json=body,
        )
        resp.raise_for_status()
        result = resp.json()

        if "hits" in result:
            for hit in result["hits"]:
                if "id" in hit:
                    hit["id"] = self._from_meili_id(hit["id"])

        return result

    # ==================== 任务管理 ====================

    async def _wait_for_task(self, task_uid: int) -> dict[str, Any]:
        """等待异步任务完成

        Meilisearch 的写操作（索引/文档/设置）都是异步任务，
        需要轮询 GET /tasks/{taskUid} 直到状态变为 succeeded 或 failed。
        """
        loop = asyncio.get_event_loop()
        deadline = loop.time() + self._task_timeout

        while loop.time() < deadline:
            resp = await self.client.get(f"/tasks/{task_uid}")
            resp.raise_for_status()
            task = resp.json()

            status = task.get("status")
            if status == "succeeded":
                return task
            if status == "failed":
                error = task.get("error", {})
                error_msg = error.get("message", "Unknown error")
                error_code = error.get("code", "unknown")
                raise MeilisearchError(
                    f"任务失败 (uid={task_uid}, code={error_code}): {error_msg}",
                )
            # status 为 "enqueued" 或 "processing"，继续轮询
            await asyncio.sleep(self._task_poll_interval)

        raise MeilisearchError(f"任务超时 (uid={task_uid}): 超过 {self._task_timeout}s")

    # ==================== 健康检查 ====================

    async def health(self) -> dict[str, Any]:
        """检查 Meilisearch 健康状态"""
        try:
            resp = await self.client.get("/health")
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def get_stats(self) -> dict[str, Any]:
        """获取索引统计信息"""
        try:
            resp = await self.client.get(f"/indexes/{self._index_name}/stats")
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"error": str(e)}
