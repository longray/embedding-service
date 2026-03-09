"""
最小化包装服务主程序

使用 SurrealDB 长期连接 + FastAPI lifespan 管理
"""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import httpx
import uvicorn
from surrealdb import AsyncSurreal

# ==================== 配置 ====================

EMBEDDING_SERVICE_URL = "http://localhost:18000"
SURREALDB_CONFIG = {
    "url": "ws://localhost:8000/rpc",
    "namespace": "memory_ns",
    "database": "memory_db",
    "username": "root",
    "password": "root",
}


# ==================== SurrealDB 连接管理器 ====================


class SurrealDBManager:
    """SurrealDB 连接管理器 - 单例模式"""

    _instance = None
    _db: AsyncSurreal = None
    _lock = asyncio.Lock()

    @classmethod
    async def get_instance(cls) -> "SurrealDBManager":
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    async def connect(self) -> None:
        if self._db is None:
            self._db = AsyncSurreal(SURREALDB_CONFIG["url"])
            await self._db.connect()
            await self._db.signin(
                {
                    "username": SURREALDB_CONFIG["username"],
                    "password": SURREALDB_CONFIG["password"],
                }
            )
            await self._db.use(
                SURREALDB_CONFIG["namespace"],
                SURREALDB_CONFIG["database"],
            )

    async def disconnect(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    @property
    def db(self) -> AsyncSurreal:
        if self._db is None:
            raise RuntimeError("数据库未连接")
        return self._db


# ==================== FastAPI 生命周期管理 ====================


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: 建立长期连接
    db_manager = await SurrealDBManager.get_instance()
    await db_manager.connect()

    yield

    # Shutdown: 关闭连接
    await db_manager.disconnect()


app = FastAPI(title="Minimal Wrapper Service", version="2.0.0", lifespan=lifespan)


# ==================== 健康检查函数 ====================


async def check_embedding_service_health():
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{EMBEDDING_SERVICE_URL}/health")
            if response.status_code == 200:
                return response.json()
    except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPError):
        pass
    return None


async def check_surrealdb_health():
    try:
        db_manager = await SurrealDBManager.get_instance()
        await db_manager.db.query("SELECT * FROM $version;")

        return {
            "status": "healthy",
            "url": SURREALDB_CONFIG["url"],
            "namespace": SURREALDB_CONFIG["namespace"],
            "database": SURREALDB_CONFIG["database"],
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "url": SURREALDB_CONFIG["url"],
            "error": str(e),
        }


# ==================== API 端点 ====================


@app.get("/health")
async def health_check():
    embedding_health = await check_embedding_service_health()
    surrealdb_health = await check_surrealdb_health()

    result = {
        "status": "healthy",
        "service": "minimal-wrapper",
        "version": "2.0.0",
        "port": 17999,
    }

    if embedding_health:
        result["embedding_service"] = {
            "status": "healthy",
            "url": EMBEDDING_SERVICE_URL,
            "service": embedding_health.get("service"),
            "version": embedding_health.get("version"),
            "device": embedding_health.get("device"),
            "cuda_available": embedding_health.get("cuda_available"),
        }
    else:
        result["embedding_service"] = {
            "status": "unhealthy",
            "url": EMBEDDING_SERVICE_URL,
            "error": "无法连接到嵌入服务",
        }

    result["surrealdb"] = surrealdb_health

    return JSONResponse(status_code=200, content=result)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=17999)  # nosec B104
