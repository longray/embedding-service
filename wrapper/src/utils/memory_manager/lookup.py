"""Lookup Mixin

提供记忆查询功能，支持通过 source_id、file_path、hash 查询记忆。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class LookupMixin:
    """记忆查询功能 Mixin"""

    async def lookup_by_source_id(
        self,
        source_id: str,
        tenant_id: str,
        type_filter: str | None = None,
        limit: int = 1,
    ) -> list[dict]:
        """通过 source_id 查询记忆

        Args:
            source_id: 本地生成的 ULID
            tenant_id: 租户ID
            type_filter: 类型过滤（可选）
            limit: 返回数量限制

        Returns:
            记忆记录列表
        """
        try:
            query = """
                SELECT * FROM memory
                WHERE source_id = $source_id
                    AND tenant_id = $tenant_id
                    AND ($type_filter IS NONE OR type = $type_filter)
                ORDER BY created_at DESC
                LIMIT $limit
            """

            result = await self._db_query(
                query,
                {
                    "source_id": source_id,
                    "tenant_id": tenant_id,
                    "type_filter": type_filter,
                    "limit": limit,
                },
            )

            return self._extract_records(result)

        except Exception as e:
            logger.error("[LookupMixin] source_id 查询失败: %s", e)
            return []

    async def lookup_by_hash(
        self,
        content_hash: str,
        tenant_id: str,
        type_filter: str | None = None,
        limit: int = 1,
    ) -> list[dict]:
        """通过 content_hash 查询记忆

        Args:
            content_hash: 内容哈希（32位十六进制）
            tenant_id: 租户ID
            type_filter: 类型过滤（可选）
            limit: 返回数量限制

        Returns:
            记忆记录列表
        """
        try:
            query = """
                SELECT * FROM memory
                WHERE content_hash = $content_hash
                    AND tenant_id = $tenant_id
                    AND ($type_filter IS NONE OR type = $type_filter)
                ORDER BY created_at DESC
                LIMIT $limit
            """

            result = await self._db_query(
                query,
                {
                    "content_hash": content_hash,
                    "tenant_id": tenant_id,
                    "type_filter": type_filter,
                    "limit": limit,
                },
            )

            return self._extract_records(result)

        except Exception as e:
            logger.error("[LookupMixin] hash 查询失败: %s", e)
            return []

    async def lookup_by_file_path(
        self,
        file_path: str,
        project_id: str,
        tenant_id: str,
        type_filter: str | None = None,
        limit: int = 1,
    ) -> list[dict]:
        """通过 file_path + project_id 查询记忆

        Args:
            file_path: 文件相对路径
            project_id: 项目ID
            tenant_id: 租户ID
            type_filter: 类型过滤（可选）
            limit: 返回数量限制

        Returns:
            记忆记录列表
        """
        try:
            query = """
                SELECT * FROM memory
                WHERE metadata.file_path = $file_path
                    AND project_id = $project_id
                    AND tenant_id = $tenant_id
                    AND ($type_filter IS NONE OR type = $type_filter)
                ORDER BY created_at DESC
                LIMIT $limit
            """

            result = await self._db_query(
                query,
                {
                    "file_path": file_path,
                    "project_id": project_id,
                    "tenant_id": tenant_id,
                    "type_filter": type_filter,
                    "limit": limit,
                },
            )

            return self._extract_records(result)

        except Exception as e:
            logger.error("[LookupMixin] file_path 查询失败: %s", e)
            return []
