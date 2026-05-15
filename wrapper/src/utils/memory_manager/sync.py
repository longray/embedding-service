"""同步、指纹、冲突解决、代码指纹"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class SyncMixin:
    """同步相关方法"""

    async def get_fingerprints(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
        """获取服务端所有记忆的指纹列表"""
        effective_tenant_id = tenant_id or self._default_tenant_id
        query = """
            SELECT source_id, content_hash, updated_at
            FROM memory
            WHERE tenant_id = $tenant_id
        """
        result = await self._db_query(query, {"tenant_id": effective_tenant_id})
        records = self._extract_records(result)
        return [
            {
                "source_id": r["source_id"],
                "hash": r.get("content_hash", ""),
                "mtime": r.get("updated_at", 0),
            }
            for r in records
            if r.get("source_id")
        ]

    async def sync_preview(
        self,
        fingerprints: list[dict[str, Any]],
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        """同步预览：比对指纹，返回变更指令"""
        effective_tenant_id = tenant_id or self._default_tenant_id

        server_fps = await self.get_fingerprints(tenant_id=effective_tenant_id)
        server_map: dict[str, dict[str, Any]] = {fp["source_id"]: fp for fp in server_fps}
        local_ids = {fp["source_id"] for fp in fingerprints}

        to_upload: list[dict[str, Any]] = []
        to_delete: list[str] = []
        conflicts: list[dict[str, Any]] = []

        for local in fingerprints:
            sid = local["source_id"]
            if sid not in server_map:
                to_upload.append({"source_id": sid, "reason": "new"})
            elif local.get("hash") != server_map[sid]["hash"]:
                conflict_entry = {
                    "source_id": sid,
                    "local_hash": local.get("hash", ""),
                    "server_hash": server_map[sid]["hash"],
                }
                conflicts.append(conflict_entry)
                await self._db_create(
                    "conflict",
                    {
                        "source_id": sid,
                        "local_hash": conflict_entry["local_hash"],
                        "server_hash": conflict_entry["server_hash"],
                        "status": "pending",
                        "tenant_id": effective_tenant_id,
                    },
                )

        for sid in server_map:
            if sid not in local_ids:
                to_delete.append(sid)

        return {
            "synced": 0,
            "to_upload": to_upload,
            "to_delete": to_delete,
            "conflicts": conflicts,
        }

    async def sync_full(
        self,
        memories: list[dict[str, Any]],
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        """全量同步：上传所有记忆"""
        effective_tenant_id = tenant_id or self._default_tenant_id
        result = await self.upload_memories(memories, tenant_id=effective_tenant_id)
        return result

    async def _record_conflict(
        self,
        source_id: str,
        local_hash: str,
        server_hash: str,
        tenant_id: str,
        local_content: str | None = None,
        server_content: str | None = None,
        local_mtime: int | None = None,
        server_mtime: int | None = None,
    ) -> str:
        """记录冲突到数据库，返回冲突 ID"""
        data: dict[str, Any] = {
            "source_id": source_id,
            "local_hash": local_hash,
            "server_hash": server_hash,
            "tenant_id": tenant_id,
            "status": "pending",
        }
        if local_content is not None:
            data["local_content"] = local_content
        if server_content is not None:
            data["server_content"] = server_content
        if local_mtime is not None:
            data["local_mtime"] = local_mtime
        if server_mtime is not None:
            data["server_mtime"] = server_mtime

        result = await self._db_create("conflict", data)
        # _db_create 返回 self._db.create() 的结果，通常是 [{"id": "conflict:xxx"}]
        if isinstance(result, list) and result and "id" in result[0]:
            return result[0]["id"]
        return str(result)

    async def get_conflicts(
        self,
        tenant_id: str,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """获取指定租户的冲突列表"""
        if status:
            query = "SELECT * FROM conflict WHERE tenant_id = $tenant_id AND status = $status LIMIT $limit"
            params: dict[str, Any] = {"tenant_id": tenant_id, "status": status, "limit": limit}
        else:
            query = "SELECT * FROM conflict WHERE tenant_id = $tenant_id LIMIT $limit"
            params = {"tenant_id": tenant_id, "limit": limit}
        result = await self._db_query(query, params)
        return self._extract_records(result)

    async def get_conflict_detail(
        self,
        conflict_id: str,
        tenant_id: str,
    ) -> dict[str, Any] | None:
        """获取单个冲突详情（conflict_id 可不带前缀）"""
        full_id = conflict_id if conflict_id.startswith("conflict:") else f"conflict:{conflict_id}"
        # BL-B-118: 使用 type::record() 转换 RecordID
        query = "SELECT * FROM conflict WHERE id = type::record($conflict_id) AND tenant_id = $tenant_id LIMIT 1"
        result = await self._db_query(query, {"conflict_id": full_id, "tenant_id": tenant_id})
        records = self._extract_records(result)
        return records[0] if records else None

    async def resolve_conflict(
        self,
        conflict_id: str,
        resolution: str,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        """解决同步冲突

        支持三种策略：
        - use_local: 用本地内容覆盖服务端记忆
        - use_remote: 保留服务端内容（不做修改）
        - keep_both: 保留两份，本地内容作为新记忆存入
        """
        effective_tenant_id = tenant_id or self._default_tenant_id
        resolution = resolution.lower()

        conflict = await self.get_conflict_detail(conflict_id, effective_tenant_id)
        if not conflict:
            return {"conflict_id": conflict_id, "resolved": False, "error": "Conflict not found"}

        source_id = conflict["source_id"]
        full_conflict_id = conflict["id"]

        if resolution == "use_local":
            local_content = conflict.get("local_content", "")
            update_sql = """
                UPDATE memory SET content = $content, content_hash = $content_hash
                WHERE source_id = $source_id AND tenant_id = $tenant_id
            """
            await self._db_query(
                update_sql,
                {
                    "content": local_content,
                    "content_hash": conflict.get("local_hash", ""),
                    "source_id": source_id,
                    "tenant_id": effective_tenant_id,
                },
            )

            if self._meili and local_content:
                find_sql = "SELECT id FROM memory WHERE source_id = $source_id AND tenant_id = $tenant_id LIMIT 1"
                mem_result = await self._db_query(find_sql, {"source_id": source_id, "tenant_id": effective_tenant_id})
                mem_records = self._extract_records(mem_result)
                if mem_records:
                    mem_id = mem_records[0]["id"]
                    await self._get_embeddings([local_content])
                    meili_doc = self._build_meili_doc(mem_id, conflict)
                    await self._meili.add_documents([meili_doc])

        elif resolution == "keep_both":
            local_content = conflict.get("local_content", "")
            new_source_id = f"{source_id}-local"
            new_data = {
                "content": local_content,
                "content_hash": conflict.get("local_hash", ""),
                "source_id": new_source_id,
                "tenant_id": effective_tenant_id,
                "type": "general",
            }
            await self._db_create("memory", new_data)

            if self._meili and local_content:
                await self._get_embeddings([local_content])
                meili_doc = self._build_meili_doc(new_source_id, conflict)
                await self._meili.add_documents([meili_doc])

        # BL-B-118: 使用 type::record() 转换 RecordID
        update_conflict_sql = """
            UPDATE conflict SET status = $status, resolution = $resolution
            WHERE id = type::record($conflict_id) AND tenant_id = $tenant_id
        """
        await self._db_query(
            update_conflict_sql,
            {
                "status": "resolved",
                "resolution": resolution,
                "conflict_id": full_conflict_id,
                "tenant_id": effective_tenant_id,
            },
        )

        return {
            "conflict_id": conflict_id,
            "resolution": resolution,
            "status": "resolved",
            "source_id": source_id,
        }

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
