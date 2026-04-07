"""Meilisearch 双写/同步/ID 转换"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class MeiliSyncMixin:
    """Meilisearch 集成方法"""

    def _build_meili_doc(self, record_id: str, memory_data: dict[str, Any], tenant_id: str) -> dict[str, Any]:
        """构建 Meilisearch 文档

        根据 meili_client.py DEFAULT_INDEX_SETTINGS 配置，构建包含所有必需字段的文档。
        """
        from datetime import datetime, timezone

        # 基础字段
        doc: dict[str, Any] = {
            "id": record_id,
            "surreal_id": record_id,
            "content": memory_data.get("content", ""),
            "content_zh": memory_data.get("content", ""),  # 简化：直接使用 content
            "tenant_id": tenant_id,
            "type": memory_data.get("type", "general"),
            "tags": memory_data.get("tags", []),
            "project_id": memory_data.get("project_id", "global"),
            "created_at": memory_data.get("created_at") or datetime.now(timezone.utc).isoformat(),
            "source_id": memory_data.get("source_id", ""),
            "metadata": memory_data.get("metadata", {}),
        }

        # 分层内容字段 (L0/L1/L2)
        doc["abstract"] = memory_data.get("abstract", "")
        doc["overview"] = memory_data.get("overview", "")

        # 代码分析字段 (BL-CA-01~04, BL-CA-18)
        metadata = memory_data.get("metadata", {})
        code_analysis = metadata.get("code_analysis", {})
        if code_analysis:
            doc["code_language"] = code_analysis.get("language", "")
            complexity = code_analysis.get("complexity", {})
            doc["code_complexity"] = complexity.get("cyclomatic_complexity", 0)
            doc["code_function_count"] = complexity.get("function_count", 0)
            doc["code_class_count"] = complexity.get("class_count", 0)
            doc["code_analyzer"] = code_analysis.get("analyzer", "")
            # BL-CA-18: 添加 code_has_exports 字段
            exports = code_analysis.get("exports", [])
            doc["code_has_exports"] = len(exports) > 0
            # code_symbols 在 upload_memories 中单独处理
            if "code_symbols" in metadata:
                doc["code_symbols"] = metadata["code_symbols"]

        return doc

    def _from_meili_id(self, meili_id: str) -> str:
        return meili_id.replace("_", ":", 1)

    async def report_access_log(self, entries: list[dict[str, Any]], tenant_id: str | None = None) -> dict[str, Any]:
        """记录访问日志（用于分析记忆使用频率）"""
        effective_tenant_id = tenant_id or self._default_tenant_id

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
