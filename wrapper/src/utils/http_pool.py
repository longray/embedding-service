"""HTTP连接池管理 - 提高性能和资源利用"""

import asyncio
import httpx
from typing import Optional


class HTTPClientPool:
    """HTTP客户端连接池"""

    def __init__(
        self,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
        timeout: float = 30.0,
        connect_timeout: float = 5.0,
        max_retries: int = 3,
    ):
        self._client: Optional[httpx.AsyncClient] = None
        self._lock = asyncio.Lock()
        self._config = {
            "limits": httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_keepalive_connections,
            ),
            "timeout": httpx.Timeout(timeout, connect=connect_timeout),
            "transport": httpx.AsyncHTTPTransport(retries=max_retries),
        }

    async def get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            async with self._lock:
                if self._client is None:
                    self._client = httpx.AsyncClient(**self._config)  # nosec B113 - timeout in self._config
        return self._client

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    async def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        client = await self.get_client()
        return await client.request(method, url, **kwargs)

    async def get(self, url: str, **kwargs) -> httpx.Response:
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs) -> httpx.Response:
        return await self.request("POST", url, **kwargs)


_pool: Optional[HTTPClientPool] = None
_pool_lock = asyncio.Lock()


async def get_http_pool(
    max_connections: int = 100,
    max_keepalive_connections: int = 20,
    timeout: float = 30.0,
    connect_timeout: float = 5.0,
    max_retries: int = 3,
) -> HTTPClientPool:
    global _pool
    if _pool is None:
        async with _pool_lock:
            if _pool is None:
                _pool = HTTPClientPool(
                    max_connections=max_connections,
                    max_keepalive_connections=max_keepalive_connections,
                    timeout=timeout,
                    connect_timeout=connect_timeout,
                    max_retries=max_retries,
                )
    return _pool


async def close_http_pool():
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
