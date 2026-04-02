"""MemoryManager 主类 — 编排层

通过多继承 Mixin 模式将功能分散到各子模块，同时保持单一类接口。
"""

import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any

from aiocache.serializers import JsonSerializer

from ..code_analyzer import CodeAnalyzer
from ..http_pool import get_http_pool
from ..meili_client import MeilisearchClient
from .stubs import StubsMixin
from .crud import CrudMixin
from .search import SearchMixin
from .sync import SyncMixin
from .relations import RelationsMixin
from .dedup import DedupMixin
from .meili_sync import MeiliSyncMixin
from .code_analysis import CodeAnalysisMixin

logger = logging.getLogger(__name__)


class MemoryManager(
    StubsMixin,
    CrudMixin,
    SearchMixin,
    SyncMixin,
    RelationsMixin,
    DedupMixin,
    MeiliSyncMixin,
    CodeAnalysisMixin,
):
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

    def set_meili_client(self, client: MeilisearchClient) -> None:
        self._meili = client

    async def close(self) -> None:
        """关闭资源"""

    def _sanitize_query(self, text: str) -> str:
        """清洗搜索查询文本，防止 SurrealQL 注入

        策略：移除 SurrealQL 特殊字符，保留字母数字和 CJK 字符。
        比简单转义更安全：直接移除潜在危险字符而非依赖转义正确性。
        """
        # 保留: 字母、数字、空格、CJK 统一表意文字（U+4E00-U+9FFF）
        # 移除: 引号、分号、反斜杠等 SQL/SurrealQL 特殊字符
        return re.sub(r"[^\w\s\u4e00-\u9fff\u3400-\u4dbf\uff00-\uffef-]", "", text).strip()[:500]

    def _normalize_memory_id(self, memory_id: str) -> str:
        """规范化记忆 ID（Stub）"""
        if ":" not in memory_id:
            return f"memory:{memory_id}"
        return memory_id

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


# 保持向后兼容的别名
MemoryManagerConfig = None  # type: ignore[assignment,misc]
