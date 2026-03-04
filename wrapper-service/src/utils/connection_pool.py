"""
SurrealDB 连接池模块

实现异步连接池，管理 SurrealDB WebSocket 连接的复用和生命周期。
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from surrealdb import AsyncSurreal

logger = logging.getLogger(__name__)


class SurrealDBConnectionPool:
    """SurrealDB 异步连接池"""

    def __init__(
        self,
        url: str,
        namespace: str,
        database: str,
        username: str,
        password: str,
        pool_size: int = 10,
        max_overflow: int = 5,
    ) -> None:
        self._url = url
        self._namespace = namespace
        self._database = database
        self._username = username
        self._password = password
        self._pool_size = pool_size
        self._max_overflow = max_overflow

        self._pool: asyncio.Queue[AsyncSurreal] = asyncio.Queue(maxsize=pool_size)
        self._active_count = 0
        self._lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self) -> None:
        """预创建连接池中的连接"""
        if self._initialized:
            return
        for _ in range(self._pool_size):
            conn = await self._create_connection()
            await self._pool.put(conn)
        self._initialized = True
        logger.info("SurrealDB连接池已初始化", extra={"pool_size": self._pool_size})

    async def _create_connection(self) -> AsyncSurreal:
        """创建并认证一个新连接"""
        conn = AsyncSurreal(self._url)
        await conn.connect()
        await conn.signin({"username": self._username, "password": self._password})
        await conn.use(self._namespace, self._database)
        return conn

    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[AsyncSurreal, None]:
        """从池中获取连接，使用完毕后自动归还"""
        conn: AsyncSurreal | None = None
        overflow = False

        try:
            # 尝试从池中非阻塞获取
            try:
                conn = self._pool.get_nowait()
            except asyncio.QueueEmpty:
                # 池已空，检查是否可以创建溢出连接
                async with self._lock:
                    if self._active_count < self._pool_size + self._max_overflow:
                        conn = await self._create_connection()
                        self._active_count += 1
                        overflow = True
                    else:
                        # 等待池中有可用连接
                        conn = await asyncio.wait_for(self._pool.get(), timeout=10.0)

            yield conn

        except Exception as e:
            logger.error("连接池获取连接失败", extra={"error": str(e)})
            raise
        finally:
            if conn is not None:
                if overflow:
                    # 溢出连接用完即关闭
                    async with self._lock:
                        self._active_count -= 1
                    await conn.close()
                else:
                    # 归还连接到池
                    await self._pool.put(conn)

    async def close(self) -> None:
        """关闭池中所有连接"""
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                await conn.close()
            except asyncio.QueueEmpty:
                break
        self._initialized = False
        logger.info("SurrealDB连接池已关闭")
