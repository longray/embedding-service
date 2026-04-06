"""测试 CLI 诊断工具"""

import pytest
from unittest.mock import AsyncMock, patch
import httpx


class TestDiagnose:
    """测试诊断工具"""

    @pytest.mark.asyncio
    async def test_check_http_service_success(self):
        """测试 HTTP 服务检查成功"""
        from scripts.diagnose import check_http_service

        with patch("httpx.AsyncClient.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.json = AsyncMock(return_value={"status": "healthy"})
            mock_get.return_value = mock_response

            ok, data = await check_http_service("Test", "http://localhost:18000")

            assert ok is True
            assert data is not None

    @pytest.mark.asyncio
    async def test_check_http_service_connection_refused(self):
        """测试 HTTP 服务连接拒绝"""
        from scripts.diagnose import check_http_service

        with patch("httpx.AsyncClient.get") as mock_get:
            mock_get.side_effect = httpx.ConnectError("Connection refused")

            ok, data = await check_http_service("Test", "http://localhost:18000")

            assert ok is False
            assert data["error"] == "连接拒绝"

    @pytest.mark.asyncio
    async def test_check_http_service_timeout(self):
        """测试 HTTP 服务超时"""
        from scripts.diagnose import check_http_service

        with patch("httpx.AsyncClient.get") as mock_get:
            mock_get.side_effect = httpx.TimeoutException("Timeout")

            ok, data = await check_http_service("Test", "http://localhost:18000")

            assert ok is False
            assert data["error"] == "连接超时"
