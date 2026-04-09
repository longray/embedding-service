"""审计日志 Mixin

提供审计日志记录和查询功能。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from ...utils.exceptions import DatabaseError

logger = logging.getLogger(__name__)


class AuditMixin:
    """审计日志功能 Mixin"""

    async def log_audit_event(
        self,
        action: str,
        resource_type: str | None = None,
        resource_id: str | None = None,
        details: dict[str, Any] | None = None,
        user_id: str | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
        tenant_id: str = "default",
    ) -> dict[str, Any]:
        """记录审计日志事件

        Args:
            action: 操作类型 (memory_create, memory_read, etc.)
            resource_type: 资源类型 (memory, relation, etc.)
            resource_id: 资源ID
            details: 详细信息
            user_id: 用户ID
            ip_address: 客户端IP
            user_agent: 客户端User-Agent
            tenant_id: 租户ID

        Returns:
            记录结果
        """
        try:
            # Build query using SET clause (more reliable for object fields)
            set_clauses = [
                "timestamp = time::now()",
                "action = $action",
                "tenant_id = $tenant_id",
            ]
            params = {
                "action": action,
                "tenant_id": tenant_id,
            }

            if user_id:
                set_clauses.append("user_id = $user_id")
                params["user_id"] = user_id
            if resource_type:
                set_clauses.append("resource_type = $resource_type")
                params["resource_type"] = resource_type
            if resource_id:
                set_clauses.append("resource_id = $resource_id")
                params["resource_id"] = resource_id
            if details:
                # Store details as JSON string
                set_clauses.append("details = $details")
                params["details"] = json.dumps(details)
            if ip_address:
                set_clauses.append("ip_address = $ip_address")
                params["ip_address"] = ip_address
            if user_agent:
                set_clauses.append("user_agent = $user_agent")
                params["user_agent"] = user_agent

            set_clause = ", ".join(set_clauses)
            query = f"CREATE audit_log SET {set_clause}"

            result = await self._db_query(query, params)

            records = self._extract_records(result)
            if records:
                return {
                    "status": "success",
                    "audit_log_id": str(records[0].get("id", "")),
                    "timestamp": records[0].get("timestamp", ""),
                }
            else:
                return {
                    "status": "error",
                    "message": "创建审计日志失败：无返回数据",
                }

        except Exception as e:
            logger.error("[AuditMixin] 记录审计日志失败: %s", e)
            return {
                "status": "error",
                "message": str(e),
            }

    async def query_audit_logs(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        user_id: str | None = None,
        action: str | None = None,
        resource_type: str | None = None,
        resource_id: str | None = None,
        tenant_id: str = "default",
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """查询审计日志

        Args:
            start_date: 开始时间
            end_date: 结束时间
            user_id: 用户ID过滤
            action: 操作类型过滤
            resource_type: 资源类型过滤
            resource_id: 资源ID过滤
            tenant_id: 租户ID
            limit: 返回数量限制
            offset: 分页偏移

        Returns:
            审计日志列表和总数
        """
        try:
            # Build WHERE 条件
            conditions = ["tenant_id = $tenant_id"]
            params = {"tenant_id": tenant_id, "limit": limit, "offset": offset}

            if start_date:
                conditions.append("timestamp >= $start_date")
                params["start_date"] = start_date.isoformat()

            if end_date:
                conditions.append("timestamp <= $end_date")
                params["end_date"] = end_date.isoformat()

            if user_id:
                conditions.append("user_id = $user_id")
                params["user_id"] = user_id

            if action:
                conditions.append("action = $action")
                params["action"] = action

            if resource_type:
                conditions.append("resource_type = $resource_type")
                params["resource_type"] = resource_type

            if resource_id:
                conditions.append("resource_id = $resource_id")
                params["resource_id"] = resource_id

            where_clause = " AND ".join(conditions)

            # Query total count
            count_query = f"""
                SELECT count() AS total FROM audit_log
                WHERE {where_clause}
                GROUP ALL
            """
            count_result = await self._db.query(count_query, params)
            count_records = self._extract_records(count_result)
            total = count_records[0].get("total", 0) if count_records else 0

            # Query data
            query = f"""
                SELECT * FROM audit_log
                WHERE {where_clause}
                ORDER BY timestamp DESC
                LIMIT $limit
                START $offset
            """
            result = await self._db.query(query, params)
            records = self._extract_records(result)

            return {
                "status": "success",
                "total": total,
                "logs": records,
                "limit": limit,
                "offset": offset,
            }

        except Exception as e:
            logger.error("[AuditMixin] 查询审计日志失败: %s", e)
            return {
                "status": "error",
                "message": str(e),
            }

    async def cleanup_audit_logs(
        self,
        retention_days: int = 90,
        tenant_id: str = "default",
    ) -> dict[str, Any]:
        """清理过期审计日志

        Args:
            retention_days: 保留天数（默认90天）
            tenant_id: 租户ID

        Returns:
            清理结果
        """
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=retention_days)

            query = """
                DELETE FROM audit_log
                WHERE tenant_id = $tenant_id
                    AND timestamp < $cutoff_date
            """

            result = await self._db.query(
                query,
                {
                    "tenant_id": tenant_id,
                    "cutoff_date": cutoff_date.isoformat(),
                },
            )

            # SurrealDB DELETE returns deleted records
            deleted_count = len(self._extract_records(result))

            return {
                "status": "success",
                "deleted_count": deleted_count,
                "retention_days": retention_days,
                "cutoff_date": cutoff_date.isoformat(),
            }

        except Exception as e:
            logger.error("[AuditMixin] 清理审计日志失败: %s", e)
            return {
                "status": "error",
                "message": str(e),
            }
