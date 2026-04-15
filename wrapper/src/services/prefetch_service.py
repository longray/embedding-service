"""预取服务 - 相关记忆和热门记忆预取

支持基于关系图的关联记忆预取和基于访问统计的热门记忆预取。
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class PrefetchService:
    """预取服务

    提供相关记忆预取和热门记忆预取功能。
    """

    def __init__(self):
        self._access_stats: dict[str, int] = {}  # 简单的访问统计缓存

    async def prefetch_related(
        self,
        memory_id: str,
        tenant_id: str,
        depth: int,
        limit: int,
        db_query_fn: callable,
        extract_records_fn: callable,
    ) -> dict[str, Any]:
        """预取相关记忆

        基于关系图遍历，预取与给定记忆相关的其他记忆。

        Args:
            memory_id: 起始记忆 ID
            tenant_id: 租户 ID
            depth: 遍历深度（1-3）
            limit: 返回数量限制
            db_query_fn: 数据库查询函数
            extract_records_fn: 记录提取函数

        Returns:
            相关记忆列表
        """
        try:
            if depth < 1 or depth > 3:
                return {
                    "status": "error",
                    "message": "遍历深度必须在 1-3 之间",
                    "related_memories": [],
                    "total_fetched": 0,
                }

            # 查询直接关系（第一层）
            related_ids = await self._fetch_related_at_depth(memory_id, tenant_id, 1, db_query_fn, extract_records_fn)

            # 如果深度 > 1，继续查询间接关系
            current_depth = 1
            all_related = set(related_ids)

            while current_depth < depth and len(all_related) < limit:
                next_level_ids = set()
                for mid in list(all_related):
                    deeper_related = await self._fetch_related_at_depth(
                        mid, tenant_id, 1, db_query_fn, extract_records_fn
                    )
                    next_level_ids.update(deeper_related)

                # 排除已查询过的和原始记忆
                next_level_ids.discard(memory_id)
                next_level_ids -= all_related

                if not next_level_ids:
                    break

                all_related.update(next_level_ids)
                current_depth += 1

            # 获取记忆详情
            memory_ids = list(all_related)[:limit]
            memories = await self._fetch_memories_details(memory_ids, tenant_id, db_query_fn, extract_records_fn)

            return {
                "status": "success",
                "message": f"成功预取 {len(memories)} 个相关记忆",
                "related_memories": memories,
                "total_fetched": len(memories),
                "depth": depth,
                "memory_id": memory_id,
                "tenant_id": tenant_id,
            }

        except Exception as e:
            logger.error("[PrefetchService] 预取相关记忆失败: %s", e)
            return {
                "status": "error",
                "message": str(e),
                "related_memories": [],
                "total_fetched": 0,
            }

    async def _fetch_related_at_depth(
        self,
        memory_id: str,
        tenant_id: str,
        depth: int,
        db_query_fn: callable,
        extract_records_fn: callable,
    ) -> list[str]:
        """获取指定深度的相关记忆 ID"""
        try:
            # 查询与 memory_id 相关的记忆
            query = """
                SELECT out AS related_id
                FROM memory_relation
                WHERE in = type::record($memory_id)
                    AND relationship_type IN ['references', 'depends_on', 'calls', 'similar_to']
                LIMIT 50
            """
            result = await db_query_fn(query, {"memory_id": memory_id})
            records = extract_records_fn(result)

            related_ids = []
            for record in records:
                related_id = record.get("related_id")
                if related_id:
                    # 处理 RecordID 对象
                    if hasattr(related_id, "table_name") and hasattr(related_id, "id"):
                        related_id = f"{related_id.table_name}:{related_id.id}"
                    related_ids.append(str(related_id))

            return related_ids

        except Exception as e:
            logger.error("[PrefetchService] 获取相关记忆失败: %s", e)
            return []

    async def _fetch_memories_details(
        self,
        memory_ids: list[str],
        tenant_id: str,
        db_query_fn: callable,
        extract_records_fn: callable,
    ) -> list[dict[str, Any]]:
        """获取记忆详情"""
        if not memory_ids:
            return []

        try:
            # 批量查询记忆详情
            query = """
                SELECT
                    id,
                    content,
                    type,
                    metadata,
                    created_at,
                    updated_at
                FROM memory
                WHERE id IN array::map($memory_ids, |$id| type::record($id))
                    AND tenant_id = $tenant_id
            """
            result = await db_query_fn(query, {"memory_ids": memory_ids, "tenant_id": tenant_id})
            records = extract_records_fn(result)

            memories = []
            for record in records:
                if record:
                    mid = record.get("id")
                    if hasattr(mid, "table_name") and hasattr(mid, "id"):
                        mid = f"{mid.table_name}:{mid.id}"

                    memories.append(
                        {
                            "id": str(mid),
                            "content": record.get("content", ""),
                            "type": record.get("type", "unknown"),
                            "metadata": record.get("metadata", {}),
                            "created_at": str(record.get("created_at", "")),
                            "updated_at": str(record.get("updated_at", "")),
                        }
                    )

            return memories

        except Exception as e:
            logger.error("[PrefetchService] 获取记忆详情失败: %s", e)
            return []

    async def prefetch_popular(
        self,
        tenant_id: str,
        top_n: int,
        db_query_fn: callable,
        extract_records_fn: callable,
    ) -> dict[str, Any]:
        """预取热门记忆

        基于访问统计和最近活跃度，预取热门记忆。

        Args:
            tenant_id: 租户 ID
            top_n: 返回数量
            db_query_fn: 数据库查询函数
            extract_records_fn: 记录提取函数

        Returns:
            热门记忆列表
        """
        try:
            # 查询最近访问的记忆（基于 updated_at 作为活跃度指标）
            query = """
                SELECT
                    id,
                    content,
                    type,
                    metadata,
                    created_at,
                    updated_at
                FROM memory
                WHERE tenant_id = $tenant_id
                ORDER BY updated_at DESC
                LIMIT $limit
            """
            result = await db_query_fn(query, {"tenant_id": tenant_id, "limit": top_n})
            records = extract_records_fn(result)

            memories = []
            for record in records:
                if record:
                    mid = record.get("id")
                    if hasattr(mid, "table_name") and hasattr(mid, "id"):
                        mid = f"{mid.table_name}:{mid.id}"

                    memories.append(
                        {
                            "id": str(mid),
                            "content": record.get("content", ""),
                            "type": record.get("type", "unknown"),
                            "metadata": record.get("metadata", {}),
                            "created_at": str(record.get("created_at", "")),
                            "updated_at": str(record.get("updated_at", "")),
                        }
                    )

            return {
                "status": "success",
                "message": f"成功预取 {len(memories)} 个热门记忆",
                "popular_memories": memories,
                "total_fetched": len(memories),
                "tenant_id": tenant_id,
            }

        except Exception as e:
            logger.error("[PrefetchService] 预取热门记忆失败: %s", e)
            return {
                "status": "error",
                "message": str(e),
                "popular_memories": [],
                "total_fetched": 0,
            }


# 全局服务实例
_prefetch_service: PrefetchService | None = None


def get_prefetch_service() -> PrefetchService:
    """获取预取服务实例（单例模式）"""
    global _prefetch_service
    if _prefetch_service is None:
        _prefetch_service = PrefetchService()
    return _prefetch_service
