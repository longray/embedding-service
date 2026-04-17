"""健康检查端点"""

import logging

from fastapi import APIRouter

from .. import state
from ..config import config
from ..utils.http_pool import get_http_pool

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


async def check_embedding_service_health():
    try:
        http_pool = await get_http_pool()
        response = await http_pool.get(
            f"{config.service.embedding_service_url}/health",
            timeout=2.0,
        )
        if response.status_code == 200:
            return response.json()
    except Exception:  # nosec B110 - 健康检查失败时静默返回 None
        return None
    return None


async def check_surrealdb_health():
    try:
        if state.memory_manager and state.memory_manager.db:
            return {"status": "connected"}
        return {"status": "unhealthy", "error": "数据库未初始化"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@router.get("/health")
async def health_check():
    embedding_health = await check_embedding_service_health()
    surrealdb_health = await check_surrealdb_health()

    result = {
        "status": "healthy",
        "service": "minimal-wrapper",
        "version": "2.4.1",
        "port": config.port,
        "embedding_service": embedding_health or {"status": "unhealthy"},
        "surrealdb": surrealdb_health,
        "meilisearch": (await state.meili_client.health()) if state.meili_client else {"status": "disabled"},
    }

    if state.embedding_cache:
        result["cache_stats"] = state.embedding_cache.get_stats()

    return result
