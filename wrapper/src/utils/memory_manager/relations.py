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

    async def create_call_relations_batch(
        self,
        calls: list[dict[str, Any]],
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        """批量创建调用关系 (BL-CA-20)

        Args:
            calls: 调用关系列表，每项包含 caller_memory_id, callee_memory_id, line, column, file_path
            tenant_id: 租户 ID

        Returns:
            {"created": int, "errors": list[dict]}
        """
        effective_tenant_id = tenant_id or self._default_tenant_id

        # 限制批量大小
        if len(calls) > 100:
            raise ValidationError(f"Batch size exceeds maximum of 100, got {len(calls)}")

        created = 0
        errors = []

        tracer = get_tracer()
        with tracer.start_as_current_span("graph.create_call_relations_batch") as span:
            span.set_attribute("batch.size", len(calls))
            span.set_attribute("tenant.id", effective_tenant_id)

            for idx, call in enumerate(calls):
                try:
                    caller_id = call.get("caller_memory_id")
                    callee_id = call.get("callee_memory_id")
                    line = call.get("line")
                    column = call.get("column")
                    file_path = call.get("file_path")

                    if not caller_id or not callee_id:
                        errors.append(
                            {
                                "index": idx,
                                "error": "Missing caller_memory_id or callee_memory_id",
                                "call": call,
                            }
                        )
                        continue

                    # 验证 caller_memory_id 存在
                    caller_ref = self._normalize_memory_id(caller_id)
                    check_caller = await self._db_query(
                        "SELECT id FROM memory WHERE id = type::record($caller_ref)",
                        {"caller_ref": caller_ref},
                    )
                    if not self._extract_records(check_caller):
                        errors.append(
                            {
                                "index": idx,
                                "caller_memory_id": caller_id,
                                "error": "Caller memory not found",
                            }
                        )
                        continue

                    # 验证 callee_memory_id 存在
                    callee_ref = self._normalize_memory_id(callee_id)
                    check_callee = await self._db_query(
                        "SELECT id FROM memory WHERE id = type::record($callee_ref)",
                        {"callee_ref": callee_ref},
                    )
                    if not self._extract_records(check_callee):
                        errors.append(
                            {
                                "index": idx,
                                "callee_memory_id": callee_id,
                                "error": "Callee memory not found",
                            }
                        )
                        continue

                    # 构建 metadata
                    metadata = {}
                    if line is not None:
                        metadata["line"] = line
                    if column is not None:
                        metadata["column"] = column
                    if file_path:
                        metadata["file_path"] = file_path

                    # 创建调用关系
                    await self.create_relation(
                        from_id=caller_id,
                        to_id=callee_id,
                        relationship_type="calls",
                        weight=0.8,  # 调用关系权重较高
                        tenant_id=effective_tenant_id,
                        description=f"Call from {caller_id} to {callee_id}",
                        metadata=metadata if metadata else None,
                    )
                    created += 1

                except Exception as e:
                    errors.append(
                        {
                            "index": idx,
                            "call": call,
                            "error": str(e),
                        }
                    )

            span.set_attribute("batch.created", created)
            span.set_attribute("batch.errors", len(errors))

        return {
            "status": "success" if created > 0 else "partial_success" if errors else "error",
            "created": created,
            "total": len(calls),
            "errors": errors,
        }

    async def get_call_references(
        self,
        memory_id: str,
        tenant_id: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """查询谁调用了该符号 (BL-CA-21)

        查询所有调用该函数的代码位置。

        Args:
            memory_id: 被调用的函数记忆 ID
            tenant_id: 租户 ID
            limit: 最大返回数量

        Returns:
            {"references": list[dict], "total": int}
        """
        effective_tenant_id = tenant_id or self._default_tenant_id
        mem_ref = self._normalize_memory_id(memory_id)

        tracer = get_tracer()
        with tracer.start_as_current_span("graph.get_call_references") as span:
            span.set_attribute("graph.memory_id", mem_ref)
            span.set_attribute("tenant.id", effective_tenant_id)

            try:
                # 查询 incoming 关系（谁调用了我）
                relations = await self.get_relations(
                    memory_id=memory_id,
                    direction="incoming",
                    relationship_type="calls",
                    tenant_id=tenant_id,
                    limit=limit,
                )

                references = []
                for rel in relations:
                    # 获取调用者的详细信息
                    caller_id = rel.get("from_id")
                    if not caller_id:
                        continue

                    caller_ref = self._normalize_memory_id(caller_id)
                    caller_info = await self._db_query(
                        "SELECT id, metadata.file_path AS file_path, metadata.code_analysis.functions AS functions "
                        "FROM memory WHERE id = type::record($caller_ref)",
                        {"caller_ref": caller_ref},
                    )
                    caller_records = self._extract_records(caller_info)

                    if caller_records:
                        caller = caller_records[0]
                        metadata = rel.get("metadata", {}) or {}

                        # 提取调用者函数名
                        caller_function = ""
                        functions = caller.get("functions", []) or []
                        if functions and len(functions) > 0:
                            caller_function = functions[0].get("name", "")

                        references.append(
                            {
                                "memory_id": caller_id,
                                "file_path": caller.get("file_path", ""),
                                "line": metadata.get("line"),
                                "column": metadata.get("column"),
                                "caller_function": caller_function,
                                "confidence": 0.95,  # 预留字段
                            }
                        )

                span.set_attribute("graph.reference_count", len(references))

                return {
                    "status": "success",
                    "memory_id": memory_id,
                    "references": references,
                    "total": len(references),
                }
            except Exception as e:
                span.record_exception(e)
                raise DatabaseError(f"Failed to get call references: {e!s}") from e

    async def get_call_dependencies(
        self,
        memory_id: str,
        tenant_id: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """查询该符号依赖了谁 (BL-CA-22)

        查询该函数调用了哪些其他函数。

        Args:
            memory_id: 调用者的函数记忆 ID
            tenant_id: 租户 ID
            limit: 最大返回数量

        Returns:
            {"dependencies": list[dict], "total": int}
        """
        effective_tenant_id = tenant_id or self._default_tenant_id
        mem_ref = self._normalize_memory_id(memory_id)

        tracer = get_tracer()
        with tracer.start_as_current_span("graph.get_call_dependencies") as span:
            span.set_attribute("graph.memory_id", mem_ref)
            span.set_attribute("tenant.id", effective_tenant_id)

            try:
                # 查询 outgoing 关系（我调用了谁）
                relations = await self.get_relations(
                    memory_id=memory_id,
                    direction="outgoing",
                    relationship_type="calls",
                    tenant_id=tenant_id,
                    limit=limit,
                )

                dependencies = []
                for rel in relations:
                    # 获取被调用者的详细信息
                    callee_id = rel.get("to_id")
                    if not callee_id:
                        continue

                    callee_ref = self._normalize_memory_id(callee_id)
                    callee_info = await self._db_query(
                        "SELECT id, metadata.file_path AS file_path, metadata.code_analysis.functions AS functions, "
                        "metadata.code_analysis.imports AS imports "
                        "FROM memory WHERE id = type::record($callee_ref)",
                        {"callee_ref": callee_ref},
                    )
                    callee_records = self._extract_records(callee_info)

                    if callee_records:
                        callee = callee_records[0]
                        metadata = rel.get("metadata", {}) or {}

                        # 提取被调用者函数名
                        callee_function = ""
                        functions = callee.get("functions", []) or []
                        if functions and len(functions) > 0:
                            callee_function = functions[0].get("name", "")

                        # 判断依赖类型
                        file_path = callee.get("file_path", "")
                        dep_type = self._classify_dependency(file_path, callee.get("imports", []))

                        dependencies.append(
                            {
                                "memory_id": callee_id,
                                "file_path": file_path,
                                "line": metadata.get("line"),
                                "column": metadata.get("column"),
                                "callee_function": callee_function,
                                "type": dep_type,
                            }
                        )

                span.set_attribute("graph.dependency_count", len(dependencies))

                return {
                    "status": "success",
                    "memory_id": memory_id,
                    "dependencies": dependencies,
                    "total": len(dependencies),
                }
            except Exception as e:
                span.record_exception(e)
                raise DatabaseError(f"Failed to get call dependencies: {e!s}") from e

    def _classify_dependency(self, file_path: str, imports: list) -> str:
        """分类依赖类型"""
        if not file_path:
            return "unknown"

        # 内置模块判断
        builtin_patterns = ["node:", "os", "sys", "fs", "path", "util"]
        for pattern in builtin_patterns:
            if pattern in file_path.lower():
                return "builtin"

        # 相对导入判断
        if file_path.startswith(".") or file_path.startswith("/"):
            return "internal"

        # 外部包判断（简化逻辑）
        if "/node_modules/" in file_path or "/site-packages/" in file_path:
            return "external"

        return "internal"
