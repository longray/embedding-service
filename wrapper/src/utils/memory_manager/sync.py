"""同步、指纹、冲突解决、代码指纹"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class SyncMixin:
    """同步相关方法"""

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
