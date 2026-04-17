"""符号查询服务 (BL-B-83)

查询 memory 表中存储的代码符号信息，支持精确匹配、前缀匹配和类型过滤。
符号数据由 PrecomputeService 写入 memory.metadata.symbol_* 字段。
"""

import logging
from typing import Any

from ..models import SymbolMatch, SymbolSearchResponse
from ..utils.db_utils import extract_records

logger = logging.getLogger(__name__)

VALID_SYMBOL_TYPES = {"function", "class", "interface", "variable", "method"}


class SymbolService:
    def __init__(self, db: Any):
        self._db = db

    async def search(
        self,
        query: str,
        tenant_id: str = "default",
        symbol_type: str | None = None,
        project_id: str | None = None,
        fuzzy: bool = False,
        limit: int = 20,
    ) -> SymbolSearchResponse:
        if symbol_type and symbol_type not in VALID_SYMBOL_TYPES:
            raise ValueError(f"Invalid symbol type: {symbol_type}. Must be one of: {VALID_SYMBOL_TYPES}")

        conditions = ["tenant_id = $tenant_id"]

        if fuzzy:
            conditions.append("string::starts_with(metadata.symbol_name, $query)")
        else:
            conditions.append("metadata.symbol_name = $query")

        if symbol_type:
            conditions.append("metadata.symbol_type = $symbol_type")

        if project_id:
            conditions.append("project_id = $project_id")

        where_clause = " AND ".join(conditions)

        surreal_query = (
            f"SELECT id, metadata.symbol_name AS name, metadata.symbol_type AS type, "
            f"metadata.symbol_file AS file, metadata.symbol_line AS line, "
            f"metadata.symbol_signature AS signature, created_at "
            f"FROM memory WHERE {where_clause} "
            f"ORDER BY created_at DESC LIMIT $limit"
        )

        params: dict[str, Any] = {
            "tenant_id": tenant_id,
            "query": query,
            "limit": limit,
        }
        if symbol_type:
            params["symbol_type"] = symbol_type
        if project_id:
            params["project_id"] = project_id

        result = await self._db_query(surreal_query, params)
        records = self._extract_records(result)

        symbols = []
        for rec in records:
            symbols.append(
                SymbolMatch(
                    name=rec.get("name", ""),
                    type=rec.get("type", ""),
                    file=rec.get("file", ""),
                    line=rec.get("line", 0),
                    memory_id=str(rec.get("id", "")),
                    signature=rec.get("signature"),
                )
            )

        return SymbolSearchResponse(symbols=symbols, total=len(symbols))

    async def _db_query(self, sql: str, params: dict[str, Any] | None = None) -> Any:
        if params:
            return await self._db.query(sql, params)
        return await self._db.query(sql)

    @staticmethod
    def _extract_records(result: Any) -> list[dict]:
        """使用统一的 extract_records 工具函数"""
        return extract_records(result)
