"""Audit 日志端点测试

验证 BL-T-1: Audit 日志端点测试
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from wrapper.src.routers.audit import create_audit_log, query_audit_logs, cleanup_audit_logs
from wrapper.src.models import AuditLogRequest


class TestCreateAuditLog:
    """POST /api/v1/audit/log 端点测试"""

    @pytest.fixture
    def mock_memory_manager(self):
        """模拟 MemoryManager"""
        mm = MagicMock()
        mm.log_audit_event = AsyncMock(
            return_value={
                "status": "success",
                "audit_log_id": "audit:123",
                "timestamp": "2026-04-15T10:00:00Z",
            }
        )
        return mm

    @pytest.mark.asyncio
    async def test_create_audit_log_success(self, mock_memory_manager):
        """测试正常记录审计日志"""
        request = AuditLogRequest(
            action="memory_create",
            resource_type="memory",
            resource_id="memory:456",
            details={"content_length": 100},
            user_id="user:123",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            tenant_id="default",
        )

        with patch("wrapper.src.routers.audit.state.memory_manager", mock_memory_manager):
            result = await create_audit_log(request)

        assert result["status"] == "success"
        assert result["audit_log_id"] == "audit:123"
        mock_memory_manager.log_audit_event.assert_called_once_with(
            action="memory_create",
            resource_type="memory",
            resource_id="memory:456",
            details={"content_length": 100},
            user_id="user:123",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            tenant_id="default",
        )

    @pytest.mark.asyncio
    async def test_create_audit_log_minimal_fields(self, mock_memory_manager):
        """测试只提供必填字段"""
        request = AuditLogRequest(action="memory_read")

        with patch("wrapper.src.routers.audit.state.memory_manager", mock_memory_manager):
            result = await create_audit_log(request)

        assert result["status"] == "success"
        mock_memory_manager.log_audit_event.assert_called_once_with(
            action="memory_read",
            resource_type=None,
            resource_id=None,
            details=None,
            user_id=None,
            ip_address=None,
            user_agent=None,
            tenant_id="default",
        )

    @pytest.mark.asyncio
    async def test_create_audit_log_memory_manager_not_initialized(self):
        """测试 MemoryManager 未初始化"""
        request = AuditLogRequest(action="memory_create")

        with patch("wrapper.src.routers.audit.state.memory_manager", None):
            with pytest.raises(HTTPException) as exc_info:
                await create_audit_log(request)

        assert exc_info.value.status_code == 503
        assert "MemoryManager未初始化" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_create_audit_log_database_error(self, mock_memory_manager):
        """测试数据库写入失败"""
        mock_memory_manager.log_audit_event = AsyncMock(
            return_value={
                "status": "error",
                "message": "数据库连接失败",
            }
        )

        request = AuditLogRequest(action="memory_create")

        with patch("wrapper.src.routers.audit.state.memory_manager", mock_memory_manager):
            with pytest.raises(HTTPException) as exc_info:
                await create_audit_log(request)

        assert exc_info.value.status_code == 500
        assert "数据库连接失败" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_create_audit_log_large_details(self, mock_memory_manager):
        """测试大 details 字段"""
        large_details = {"content": "x" * 10000}
        request = AuditLogRequest(
            action="memory_create",
            details=large_details,
        )

        with patch("wrapper.src.routers.audit.state.memory_manager", mock_memory_manager):
            result = await create_audit_log(request)

        assert result["status"] == "success"
        call_args = mock_memory_manager.log_audit_event.call_args[1]
        assert call_args["details"] == large_details


class TestQueryAuditLogs:
    """GET /api/v1/audit/logs 端点测试"""

    @pytest.fixture
    def mock_memory_manager(self):
        """模拟 MemoryManager"""
        mm = MagicMock()
        mm.query_audit_logs = AsyncMock(
            return_value={
                "status": "success",
                "total": 2,
                "logs": [
                    {
                        "id": "audit:1",
                        "action": "memory_create",
                        "user_id": "user:123",
                        "timestamp": "2026-04-15T10:00:00Z",
                    },
                    {
                        "id": "audit:2",
                        "action": "memory_read",
                        "user_id": "user:456",
                        "timestamp": "2026-04-15T09:00:00Z",
                    },
                ],
                "limit": 100,
                "offset": 0,
            }
        )
        return mm

    @pytest.mark.asyncio
    async def test_query_audit_logs_no_filters(self, mock_memory_manager):
        """测试无参数查询全部日志"""
        with patch("wrapper.src.routers.audit.state.memory_manager", mock_memory_manager):
            result = await query_audit_logs()

        assert result["status"] == "success"
        assert result["total"] == 2
        assert len(result["logs"]) == 2
        # Verify the mock was called (FastAPI Query objects are passed, not raw values)
        mock_memory_manager.query_audit_logs.assert_called_once()
        call_args = mock_memory_manager.query_audit_logs.call_args[1]
        # Check that default values are used
        assert call_args["tenant_id"].default == "default"
        assert call_args["limit"].default == 100
        assert call_args["offset"].default == 0

    @pytest.mark.asyncio
    async def test_query_audit_logs_with_time_range(self, mock_memory_manager):
        """测试按时间范围过滤"""
        start_date = datetime(2026, 4, 1)
        end_date = datetime(2026, 4, 30)

        with patch("wrapper.src.routers.audit.state.memory_manager", mock_memory_manager):
            result = await query_audit_logs(
                start_date=start_date,
                end_date=end_date,
            )

        assert result["status"] == "success"
        call_args = mock_memory_manager.query_audit_logs.call_args[1]
        assert call_args["start_date"] == start_date
        assert call_args["end_date"] == end_date

    @pytest.mark.asyncio
    async def test_query_audit_logs_with_user_filter(self, mock_memory_manager):
        """测试按 user_id 过滤"""
        with patch("wrapper.src.routers.audit.state.memory_manager", mock_memory_manager):
            result = await query_audit_logs(user_id="user:123")

        assert result["status"] == "success"
        call_args = mock_memory_manager.query_audit_logs.call_args[1]
        assert call_args["user_id"] == "user:123"

    @pytest.mark.asyncio
    async def test_query_audit_logs_with_action_filter(self, mock_memory_manager):
        """测试按 action 过滤"""
        with patch("wrapper.src.routers.audit.state.memory_manager", mock_memory_manager):
            result = await query_audit_logs(action="memory_create")

        assert result["status"] == "success"
        call_args = mock_memory_manager.query_audit_logs.call_args[1]
        assert call_args["action"] == "memory_create"

    @pytest.mark.asyncio
    async def test_query_audit_logs_with_resource_type_filter(self, mock_memory_manager):
        """测试按 resource_type 过滤"""
        with patch("wrapper.src.routers.audit.state.memory_manager", mock_memory_manager):
            result = await query_audit_logs(resource_type="memory")

        assert result["status"] == "success"
        call_args = mock_memory_manager.query_audit_logs.call_args[1]
        assert call_args["resource_type"] == "memory"

    @pytest.mark.asyncio
    async def test_query_audit_logs_with_combined_filters(self, mock_memory_manager):
        """测试组合过滤条件"""
        start_date = datetime(2026, 4, 1)

        with patch("wrapper.src.routers.audit.state.memory_manager", mock_memory_manager):
            result = await query_audit_logs(
                start_date=start_date,
                user_id="user:123",
                action="memory_create",
                resource_type="memory",
            )

        assert result["status"] == "success"
        call_args = mock_memory_manager.query_audit_logs.call_args[1]
        assert call_args["start_date"] == start_date
        assert call_args["user_id"] == "user:123"
        assert call_args["action"] == "memory_create"
        assert call_args["resource_type"] == "memory"

    @pytest.mark.asyncio
    async def test_query_audit_logs_with_pagination(self, mock_memory_manager):
        """测试分页查询"""
        with patch("wrapper.src.routers.audit.state.memory_manager", mock_memory_manager):
            result = await query_audit_logs(limit=10, offset=20)

        assert result["status"] == "success"
        call_args = mock_memory_manager.query_audit_logs.call_args[1]
        assert call_args["limit"] == 10
        assert call_args["offset"] == 20

    @pytest.mark.asyncio
    async def test_query_audit_logs_empty_result(self, mock_memory_manager):
        """测试无结果返回空列表"""
        mock_memory_manager.query_audit_logs = AsyncMock(
            return_value={
                "status": "success",
                "total": 0,
                "logs": [],
                "limit": 100,
                "offset": 0,
            }
        )

        with patch("wrapper.src.routers.audit.state.memory_manager", mock_memory_manager):
            result = await query_audit_logs(user_id="nonexistent")

        assert result["status"] == "success"
        assert result["total"] == 0
        assert result["logs"] == []

    @pytest.mark.asyncio
    async def test_query_audit_logs_memory_manager_not_initialized(self):
        """测试 MemoryManager 未初始化"""
        with patch("wrapper.src.routers.audit.state.memory_manager", None):
            with pytest.raises(HTTPException) as exc_info:
                await query_audit_logs()

        assert exc_info.value.status_code == 503


class TestCleanupAuditLogs:
    """DELETE /api/v1/audit/logs 端点测试"""

    @pytest.fixture
    def mock_memory_manager(self):
        """模拟 MemoryManager"""
        mm = MagicMock()
        mm.cleanup_audit_logs = AsyncMock(
            return_value={
                "status": "success",
                "deleted_count": 100,
                "retention_days": 90,
                "cutoff_date": "2026-01-15T00:00:00Z",
            }
        )
        return mm

    @pytest.mark.asyncio
    async def test_cleanup_audit_logs_default_retention(self, mock_memory_manager):
        """测试默认保留天数（90天）清理"""
        with patch("wrapper.src.routers.audit.state.memory_manager", mock_memory_manager):
            result = await cleanup_audit_logs()

        assert result["status"] == "success"
        assert result["deleted_count"] == 100
        assert result["retention_days"] == 90
        mock_memory_manager.cleanup_audit_logs.assert_called_once()
        call_args = mock_memory_manager.cleanup_audit_logs.call_args[1]
        assert call_args["retention_days"].default == 90
        assert call_args["tenant_id"].default == "default"

    @pytest.mark.asyncio
    async def test_cleanup_audit_logs_custom_retention(self, mock_memory_manager):
        """测试自定义保留天数"""
        mock_memory_manager.cleanup_audit_logs = AsyncMock(
            return_value={
                "status": "success",
                "deleted_count": 50,
                "retention_days": 30,
                "cutoff_date": "2026-03-16T00:00:00Z",
            }
        )

        with patch("wrapper.src.routers.audit.state.memory_manager", mock_memory_manager):
            result = await cleanup_audit_logs(retention_days=30)

        assert result["status"] == "success"
        assert result["retention_days"] == 30
        call_args = mock_memory_manager.cleanup_audit_logs.call_args[1]
        assert call_args["retention_days"] == 30

    @pytest.mark.asyncio
    async def test_cleanup_audit_logs_memory_manager_not_initialized(self):
        """测试 MemoryManager 未初始化"""
        with patch("wrapper.src.routers.audit.state.memory_manager", None):
            with pytest.raises(HTTPException) as exc_info:
                await cleanup_audit_logs()

        assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_cleanup_audit_logs_database_error(self, mock_memory_manager):
        """测试清理失败"""
        mock_memory_manager.cleanup_audit_logs = AsyncMock(
            return_value={
                "status": "error",
                "message": "清理失败：数据库错误",
            }
        )

        with patch("wrapper.src.routers.audit.state.memory_manager", mock_memory_manager):
            with pytest.raises(HTTPException) as exc_info:
                await cleanup_audit_logs()

        assert exc_info.value.status_code == 500
        assert "清理失败" in exc_info.value.detail


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
