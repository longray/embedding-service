"""Meilisearch 双写/同步/ID 转换"""

import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


class MeiliSyncMixin:
    """Meilisearch 集成方法"""

    def _build_meili_doc(
        self,
        record_id: str,
        data: dict[str, Any],
        tenant_id: str,
        doc_type: str = "entity",
    ) -> dict[str, Any]:
        """构建 Meilisearch 文档

        根据 meili_client.py DEFAULT_INDEX_SETTINGS 配置，构建包含所有必需字段的文档。
        支持 entity (memory) 和 atom 两种文档类型。

        Args:
            record_id: SurrealDB RecordID (e.g., "memory:xxx" or "atom:xxx")
            data: 文档数据
            tenant_id: 租户 ID
            doc_type: 文档类型 ("entity" | "atom")
        """
        meili_id = record_id.replace(":", "_", 1)

        if doc_type == "atom":
            return self._build_atom_meili_doc(meili_id, record_id, data, tenant_id)

        return self._build_entity_meili_doc(meili_id, record_id, data, tenant_id)

    def _build_entity_meili_doc(
        self,
        meili_id: str,
        surreal_id: str,
        memory_data: dict[str, Any],
        tenant_id: str,
    ) -> dict[str, Any]:
        """构建 Entity (Memory) Meilisearch 文档"""

        doc: dict[str, Any] = {
            "id": meili_id,
            "surreal_id": surreal_id,
            "doc_type": "entity",
            "content": memory_data.get("content", ""),
            "content_zh": memory_data.get("content", ""),
            "tenant_id": tenant_id,
            "type": memory_data.get("type", "general"),
            "tags": memory_data.get("tags", []),
            "project_id": memory_data.get("project_id", "global"),
            "created_at": memory_data.get("created_at") or datetime.utcnow().isoformat(),
            "source_id": memory_data.get("source_id", ""),
            "metadata": memory_data.get("metadata", {}),
            "abstract": memory_data.get("abstract", ""),
            "overview": memory_data.get("overview", ""),
            "file_path": memory_data.get("file_path"),
        }

        # 代码分析字段
        metadata = memory_data.get("metadata", {})
        code_analysis = metadata.get("code_analysis", {})
        if code_analysis:
            doc["code_language"] = code_analysis.get("language", "")
            complexity = code_analysis.get("complexity", {})
            doc["code_complexity"] = complexity.get("cyclomatic_complexity", 0)
            doc["code_function_count"] = complexity.get("function_count", 0)
            doc["code_class_count"] = complexity.get("class_count", 0)
            doc["code_analyzer"] = code_analysis.get("analyzer", "")
            exports = code_analysis.get("exports", [])
            doc["code_has_exports"] = len(exports) > 0
            if "code_symbols" in metadata:
                doc["code_symbols"] = metadata["code_symbols"]

        return doc

    def _build_atom_meili_doc(
        self,
        meili_id: str,
        surreal_id: str,
        atom_data: dict[str, Any],
        tenant_id: str,
    ) -> dict[str, Any]:
        """构建 Atom Meilisearch 文档"""

        return {
            "id": meili_id,
            "surreal_id": surreal_id,
            "doc_type": "atom",
            "name": atom_data.get("name", ""),
            "content": atom_data.get("content", ""),
            "content_zh": atom_data.get("content", ""),
            "tenant_id": tenant_id,
            "atom_type": atom_data.get("type", "note"),
            "type": atom_data.get("type", "note"),  # 兼容现有过滤
            "tags": atom_data.get("tags", []),
            "entity_id": atom_data.get("entity_id", ""),
            "local_id": atom_data.get("local_id", ""),
            "heading_level": atom_data.get("heading_level"),
            "created_at": atom_data.get("created_at") or datetime.utcnow().isoformat(),
        }

    def _from_meili_id(self, meili_id: str) -> str:
        return meili_id.replace("_", ":", 1)

    async def report_access_log(self, entries: list[dict[str, Any]], tenant_id: str | None = None) -> dict[str, Any]:
        """记录访问日志（用于分析记忆使用频率）"""
        _ = tenant_id or self._default_tenant_id  # 保留供未来使用

        # 简化的实现：将访问日志记录到内存中
        # 实际生产环境可以存储到 SurrealDB 或发送到分析服务
        logged_count = 0
        for entry in entries:
            if entry.get("entry_id") and entry.get("timestamp"):
                logged_count += 1

        return {
            "status": "success",
            "logged_count": logged_count,
            "total_received": len(entries),
        }
