"""审计日志端点"""

from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from .. import state
from ..models import AuditLogRequest

router = APIRouter(prefix="/api/v1", tags=["audit"])


@router.post("/audit/log")
async def create_audit_log(request: AuditLogRequest):
    """记录审计日志事件

    用于手动记录重要的业务操作审计事件。
    """
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        result = await state.memory_manager.log_audit_event(
            action=request.action,
            resource_type=request.resource_type,
            resource_id=request.resource_id,
            details=request.details,
            user_id=request.user_id,
            ip_address=request.ip_address,
            user_agent=request.user_agent,
            tenant_id=request.tenant_id,
        )

        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("message", "记录审计日志失败"))

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"记录审计日志失败: {e}") from e


@router.get("/audit/logs")
async def query_audit_logs(
    start_date: datetime | None = Query(None, description="开始时间 (ISO 8601)"),
    end_date: datetime | None = Query(None, description="结束时间 (ISO 8601)"),
    user_id: str | None = Query(None, description="用户ID过滤"),
    action: str | None = Query(None, description="操作类型过滤"),
    resource_type: str | None = Query(None, description="资源类型过滤"),
    resource_id: str | None = Query(None, description="资源ID过滤"),
    tenant_id: str = Query(default="default", description="租户ID"),
    limit: int = Query(default=100, ge=1, le=1000, description="返回数量限制"),
    offset: int = Query(default=0, ge=0, description="分页偏移"),
):
    """查询审计日志

    支持时间范围、用户、操作类型等多维度过滤。
    """
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        result = await state.memory_manager.query_audit_logs(
            start_date=start_date,
            end_date=end_date,
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            tenant_id=tenant_id,
            limit=limit,
            offset=offset,
        )

        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("message", "查询审计日志失败"))

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询审计日志失败: {e}") from e


@router.delete("/audit/logs")
async def cleanup_audit_logs(
    retention_days: int = Query(default=90, ge=1, description="保留天数"),
    tenant_id: str = Query(default="default", description="租户ID"),
):
    """清理过期审计日志

    删除指定保留天数之前的审计日志（默认保留90天）。
    """
    if not state.memory_manager:
        raise HTTPException(status_code=503, detail="MemoryManager未初始化")

    try:
        result = await state.memory_manager.cleanup_audit_logs(
            retention_days=retention_days,
            tenant_id=tenant_id,
        )

        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("message", "清理审计日志失败"))

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清理审计日志失败: {e}") from e
