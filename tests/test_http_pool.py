"""Unit tests for wrapper.src.utils.http_pool"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import httpx
from wrapper.src.utils.http_pool import HTTPClientPool


class TestHTTPClientPool:
    @pytest.mark.asyncio
    async def test_create_client(self):
        pool = HTTPClientPool(max_connections=10, timeout=5.0)
        client = await pool.get_client()
        assert client is not None
        assert isinstance(client, httpx.AsyncClient)
        await pool.close()

    @pytest.mark.asyncio
    async def test_get_client_returns_same_instance(self):
        pool = HTTPClientPool()
        client1 = await pool.get_client()
        client2 = await pool.get_client()
        assert client1 is client2
        await pool.close()

    @pytest.mark.asyncio
    async def test_close(self):
        pool = HTTPClientPool()
        await pool.get_client()
        await pool.close()
        assert pool._client is None

    @pytest.mark.asyncio
    async def test_close_when_no_client(self):
        pool = HTTPClientPool()
        await pool.close()  # Should not raise
        assert pool._client is None

    @pytest.mark.asyncio
    async def test_request_method(self):
        pool = HTTPClientPool()
        mock_response = MagicMock()
        mock_response.status_code = 200
        with patch.object(pool, "get_client", new_callable=AsyncMock) as mock_get:
            mock_client = AsyncMock()
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_get.return_value = mock_client
            response = await pool.request("GET", "http://example.com")
            assert response.status_code == 200
            mock_client.request.assert_called_once_with("GET", "http://example.com")
        await pool.close()
