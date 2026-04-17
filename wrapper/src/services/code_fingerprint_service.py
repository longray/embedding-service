"""代码指纹服务 (BL-B-80)

实现代码文件指纹的存储、比对和增量同步功能。
"""

import logging

from surrealdb import Surreal

from ..utils.db_utils import extract_records

logger = logging.getLogger(__name__)


class CodeFingerprintService:
    """代码指纹服务"""

    def __init__(self, db: Surreal):  # type: ignore[reportGeneralTypeIssues]
        self._db = db

    async def compare_fingerprints(
        self,
        fingerprints: list[dict[str, str]],
        tenant_id: str,
        project_id: str,
    ) -> dict[str, list[str]]:
        """比对文件指纹，返回变更分类

        Args:
            fingerprints: 文件指纹列表 [{"file": "path", "content_hash": "...", "symbols_hash": "..."}]
            tenant_id: 租户ID
            project_id: 项目ID

        Returns:
            {
                "changed_files": ["path1", ...],
                "unchanged_files": ["path2", ...],
                "new_files": ["path3", ...],
                "deleted_files": ["path4", ...]
            }
        """
        try:
            # 获取数据库中该项目的所有指纹
            query = """
                SELECT file_path, content_hash, symbols_hash
                FROM file_fingerprint
                WHERE tenant_id = $tenant_id
                  AND project_id = $project_id
            """
            result = await self._db.query(query, {"tenant_id": tenant_id, "project_id": project_id})

            # 解析现有指纹
            existing_fingerprints: dict[str, dict[str, str]] = {}
            records = extract_records(result)
            for record in records:
                existing_fingerprints[record["file_path"]] = {
                    "content_hash": record.get("content_hash", ""),
                    "symbols_hash": record.get("symbols_hash", ""),
                }

            # 分类文件
            changed_files: list[str] = []
            unchanged_files: list[str] = []
            new_files: list[str] = []
            incoming_files: set[str] = set()

            for fp in fingerprints:
                file_path = fp["file"]
                incoming_files.add(file_path)

                if file_path not in existing_fingerprints:
                    new_files.append(file_path)
                else:
                    existing = existing_fingerprints[file_path]
                    if fp["content_hash"] != existing["content_hash"] or fp["symbols_hash"] != existing["symbols_hash"]:
                        changed_files.append(file_path)
                    else:
                        unchanged_files.append(file_path)

            # 检测已删除文件
            deleted_files = [path for path in existing_fingerprints if path not in incoming_files]

            logger.info(
                "[CodeFingerprint] Compared %d files: %d changed, %d unchanged, %d new, %d deleted",
                len(fingerprints),
                len(changed_files),
                len(unchanged_files),
                len(new_files),
                len(deleted_files),
            )

            return {
                "changed_files": changed_files,
                "unchanged_files": unchanged_files,
                "new_files": new_files,
                "deleted_files": deleted_files,
            }

        except Exception as e:
            logger.error("[CodeFingerprint] Compare failed: %s", e)
            raise

    async def update_fingerprints(
        self,
        fingerprints: list[dict[str, str]],
        tenant_id: str,
        project_id: str,
    ) -> int:
        """批量更新文件指纹到数据库

        使用批量 UPSERT 替代循环，减少数据库往返次数。

        Args:
            fingerprints: 文件指纹列表
            tenant_id: 租户ID
            project_id: 项目ID

        Returns:
            更新数量
        """
        if not fingerprints:
            return 0

        try:
            # 构建批量数据
            batch_data = [
                {
                    "file_path": fp["file"],
                    "content_hash": fp["content_hash"],
                    "symbols_hash": fp["symbols_hash"],
                    "tenant_id": tenant_id,
                    "project_id": project_id,
                    "updated_at": "time::now()",
                }
                for fp in fingerprints
            ]

            # 批量 UPSERT
            query = """
                FOR $fp IN $fingerprints {
                    UPSERT file_fingerprint CONTENT {
                        file_path: $fp.file_path,
                        content_hash: $fp.content_hash,
                        symbols_hash: $fp.symbols_hash,
                        tenant_id: $fp.tenant_id,
                        project_id: $fp.project_id,
                        updated_at: time::now()
                    }
                }
            """
            await self._db.query(query, {"fingerprints": batch_data})

            logger.info("[CodeFingerprint] Batch updated %d fingerprints", len(fingerprints))
            return len(fingerprints)

        except Exception as e:
            logger.error("[CodeFingerprint] Batch update failed: %s", e)
            raise

    async def delete_fingerprints(
        self,
        file_paths: list[str],
        tenant_id: str,
        project_id: str,
    ) -> int:
        """批量删除文件指纹

        使用批量 DELETE 替代循环，减少数据库往返次数。

        Args:
            file_paths: 文件路径列表
            tenant_id: 租户ID
            project_id: 项目ID

        Returns:
            删除数量
        """
        if not file_paths:
            return 0

        try:
            # 批量 DELETE
            query = """
                DELETE FROM file_fingerprint
                WHERE file_path IN $file_paths
                  AND tenant_id = $tenant_id
                  AND project_id = $project_id
            """
            await self._db.query(
                query,
                {
                    "file_paths": file_paths,
                    "tenant_id": tenant_id,
                    "project_id": project_id,
                },
            )

            logger.info("[CodeFingerprint] Batch deleted %d fingerprints", len(file_paths))
            return len(file_paths)

        except Exception as e:
            logger.error("[CodeFingerprint] Batch delete failed: %s", e)
            raise
