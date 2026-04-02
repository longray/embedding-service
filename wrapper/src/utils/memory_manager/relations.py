"""图关系、遍历"""

import json
import logging
from typing import Any

from ..exceptions import DatabaseError, ValidationError
from ..tracing import get_tracer

logger = logging.getLogger(__name__)


class RelationsMixin:
    """图关系相关方法"""

    def _format_relation_results(self, raw: Any, direction: str) -> list[dict[str, Any]]:
        """格式化关系查询结果"""
        records = self._extract_records(raw)
        results: list[dict[str, Any]] = []
        for rec in records:
            results.append(
                {
                    "id": str(rec.get("relation_id", "")),
                    "from_id": str(rec.get("from_id", "")),
                    "to_id": str(rec.get("to_id", "")),
                    "relationship_type": rec.get("relationship_type", "related"),
                    "weight": float(rec.get("weight", 0.5)),
                    "direction": direction,
                    "description": rec.get("description"),
                    "metadata": rec.get("metadata"),
                }
            )
        return results

    def _normalize_relation_id(self, relation_id: str) -> str:
        """规范化关系 ID（Stub）"""
        if ":" not in relation_id:
            return f"memory_relation:{relation_id}"
        return relation_id

    async def create_relation(
        self,
        from_id: str,
        to_id: str,
        relationship_type: str = "related",
        weight: float = 0.5,
        tenant_id: str | None = None,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """创建两条记忆之间的图关系"""
        effective_tenant_id = tenant_id or self._default_tenant_id

        valid_types = {"related", "follow_up", "elaboration", "contradiction", "reference", "derived_from"}
        if relationship_type not in valid_types:
            raise ValidationError(f"Invalid relationship_type: {relationship_type}. Must be one of {valid_types}")
        if not 0.0 <= weight <= 1.0:
            raise ValidationError(f"weight must be between 0.0 and 1.0, got {weight}")

        from_ref = self._normalize_memory_id(from_id)
        to_ref = self._normalize_memory_id(to_id)

        tracer = get_tracer()
        with tracer.start_as_current_span("graph.create_relation") as span:
            span.set_attribute("graph.from", from_ref)
            span.set_attribute("graph.to", to_ref)
            span.set_attribute("graph.type", relationship_type)
            span.set_attribute("tenant.id", effective_tenant_id)

            try:
                set_clauses = [
                    f"relationship_type = '{relationship_type}'",
                    f"weight = {float(weight)}",
                    "tenant_id = $tenant_id",
                ]
                if description:
                    safe_desc = self._sanitize_query(description)
                    set_clauses.append(f"description = '{safe_desc}'")
                if metadata:
                    set_clauses.append(f"metadata = {json.dumps(metadata)}")

                set_str = ", ".join(set_clauses)
                q = (  # nosec B608
                    f"RELATE {from_ref}->memory_relation->{to_ref} "  # nosec B608
                    f"SET {set_str}"  # nosec B608
                )
                result = await self._db_query(q, {"tenant_id": effective_tenant_id})

                records = self._extract_records(result)
                if records:
                    return {
                        "id": str(records[0].get("id", "")),
                        "from": from_ref,
                        "to": to_ref,
                        "relationship_type": relationship_type,
                        "weight": weight,
                    }
                return {"error": "No relation created"}
            except Exception as e:
                span.record_exception(e)
                raise DatabaseError(f"Failed to create relation: {e!s}") from e

    async def get_relations(
        self,
        memory_id: str,
        direction: str = "both",
        relationship_type: str | None = None,
        tenant_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """查询记忆的关联关系"""
        effective_tenant_id = tenant_id or self._default_tenant_id
        mem_ref = self._normalize_memory_id(memory_id)

        if direction not in ("outgoing", "incoming", "both"):
            raise ValidationError(f"Invalid direction: {direction}. Must be outgoing/incoming/both")

        tracer = get_tracer()
        with tracer.start_as_current_span("graph.get_relations") as span:
            span.set_attribute("graph.memory_id", mem_ref)
            span.set_attribute("graph.direction", direction)
            span.set_attribute("tenant.id", effective_tenant_id)

            try:
                results: list[dict[str, Any]] = []

                # 拆分 mem_ref 为 table 和 id
                mem_parts = mem_ref.split(":")
                mem_table, mem_id = mem_parts[0], mem_parts[1] if len(mem_parts) > 1 else mem_parts[0]

                if direction in ("outgoing", "both"):
                    q = (
                        "SELECT *, meta::id(id) AS relation_id, "
                        "meta::id(in) AS from_id, meta::id(out) AS to_id "
                        "FROM memory_relation "
                        "WHERE in = type::record($mem_table, $mem_id) AND tenant_id = $tenant_id "
                    )
                    if relationship_type:
                        q += "AND relationship_type = $relationship_type "
                    q += "LIMIT $limit"
                    r = await self._db_query(
                        q,
                        {
                            "tenant_id": effective_tenant_id,
                            "mem_table": mem_table,
                            "mem_id": mem_id,
                            "relationship_type": relationship_type,
                            "limit": limit,
                        },
                    )
                    results.extend(self._format_relation_results(r, "outgoing"))

                if direction in ("incoming", "both"):
                    q = (
                        "SELECT *, meta::id(id) AS relation_id, "
                        "meta::id(in) AS from_id, meta::id(out) AS to_id "
                        "FROM memory_relation "
                        "WHERE out = type::record($mem_table, $mem_id) AND tenant_id = $tenant_id "
                    )
                    if relationship_type:
                        q += "AND relationship_type = $relationship_type "
                    q += "LIMIT $limit"
                    r = await self._db_query(
                        q,
                        {
                            "tenant_id": effective_tenant_id,
                            "mem_table": mem_table,
                            "mem_id": mem_id,
                            "relationship_type": relationship_type,
                            "limit": limit,
                        },
                    )
                    results.extend(self._format_relation_results(r, "incoming"))

                span.set_attribute("graph.relation_count", len(results))
                return results
            except Exception as e:
                span.record_exception(e)
                raise DatabaseError(f"Failed to get relations: {e!s}") from e

    async def delete_relation(
        self,
        relation_id: str,
        tenant_id: str | None = None,
    ) -> bool:
        """删除指定的关系

        Args:
            relation_id: 关系 ID（如 "memory_relation:abc123"）
            tenant_id: 租户 ID（安全检查，防止跨租户删除）
        """
        effective_tenant_id = tenant_id or self._default_tenant_id
        rel_ref = self._normalize_relation_id(relation_id)

        try:
            # 拆分 rel_ref 为 table 和 id
            rel_parts = rel_ref.split(":")
            rel_table, rel_id = rel_parts[0], rel_parts[1] if len(rel_parts) > 1 else rel_parts[0]

            # 先验证关系存在且属于该租户
            q = "SELECT id FROM type::record($rel_table, $rel_id) WHERE tenant_id = $tenant_id"
            check = await self._db_query(
                q, {"tenant_id": effective_tenant_id, "rel_table": rel_table, "rel_id": rel_id}
            )
            records = self._extract_records(check)
            if not records:
                return False

            # 删除关系
            del_q = "DELETE type::record($rel_table, $rel_id)"
            await self._db_query(del_q, {"rel_table": rel_table, "rel_id": rel_id})
            return True
        except Exception as e:
            logger.error("[Relation Delete] 失败: %s", e)
            raise DatabaseError(f"删除关系失败: {e}") from e

    async def get_related_memories(
        self,
        memory_id: str,
        depth: int = 1,
        relationship_type: str | None = None,
        tenant_id: str = "default",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        effective_tenant_id = tenant_id or self._default_tenant_id
        source_ref = self._normalize_memory_id(memory_id)

        visited: set[str] = {source_ref}
        current_level: set[str] = {source_ref}
        all_memories: list[dict[str, Any]] = []

        for _ in range(depth):
            next_level: set[str] = set()
            for mem_id in current_level:
                relations = await self.get_relations(
                    mem_id,
                    direction="both",
                    relationship_type=relationship_type,
                    tenant_id=effective_tenant_id,
                    limit=limit,
                )
                for rel in relations:
                    for endpoint in (rel["from_id"], rel["to_id"]):
                        ep_ref = self._normalize_memory_id(endpoint)
                        if ep_ref in visited or ep_ref == source_ref:
                            continue
                        visited.add(ep_ref)
                        next_level.add(ep_ref)

                        if len(all_memories) >= limit:
                            return all_memories[:limit]

                        mem_parts = ep_ref.split(":")
                        tbl, rid = mem_parts[0], mem_parts[1]
                        r = await self._db_query(
                            "SELECT id, content, abstract, overview, type, tags, project_id, local_id, metadata "
                            "FROM type::record($table, $id)",
                            {"table": tbl, "id": rid},
                        )
                        records = self._extract_records(r)
                        if records:
                            all_memories.append(records[0])

            current_level = next_level
            if not current_level:
                break

        return all_memories[:limit]
