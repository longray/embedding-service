"""9 个 NotImplementedError 占位方法"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class StubsMixin:
    """占位方法（功能尚未实现）"""

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
