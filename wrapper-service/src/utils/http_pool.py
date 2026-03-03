"""
HTTP连接池管理 - 提高性能和资源利用
"""

import httpx
from typing import Optional
import threading


class HTTPClientPool:
    """
    HTTP客户端连接池

    特性：
    - 连接复用
    - 超时控制
    - 自动重试
    - 线程安全
    """

    def __init__(
        self,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        self._client: Optional[httpx.AsyncClient] = None
        self._lock = threading.Lock()
        self._config = {
            "limits": httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_keepalive_connections,
            ),
            "timeout": httpx.Timeout(timeout),
            "transport": httpx.AsyncHTTPTransport(retries=max_retries),
        }

    async def get_client(self) -> httpx.AsyncClient:
        """获取或创建客户端实例"""
        if self._client is None:
            with self._lock:
                if self._client is None:
                    self._client = httpx.AsyncClient(**self._config)
        return self._client

    async def close(self):
        """关闭连接池"""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """发送HTTP请求"""
        client = await self.get_client()
        return await client.request(method, url, **kwargs)

    async def get(self, url: str, **kwargs) -> httpx.Response:
        """GET请求"""
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs) -> httpx.Response:
        """POST请求"""
        return await self.request("POST", url, **kwargs)


# 全局连接池实例
_pool: Optional[HTTPClientPool] = None
_pool_lock = threading.Lock()


def get_http_pool() -> HTTPClientPool:
    """获取全局连接池实例（单例）"""
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = HTTPClientPool()
    return _pool
