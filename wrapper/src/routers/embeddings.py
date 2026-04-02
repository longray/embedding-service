"""Embedding 端点"""

from fastapi import APIRouter, HTTPException

from .. import state
from ..config import config
from ..models import EmbeddingRequest
from ..utils.cache import hash_text
from ..utils.http_pool import get_http_pool

router = APIRouter(tags=["embeddings"])


@router.post("/v1/embeddings")
async def create_embedding(request: EmbeddingRequest):
    cache_key = hash_text(request.input)

    if state.embedding_cache:
        cached = state.embedding_cache.get(cache_key)
        if cached:
            return cached

    try:
        http_pool = await get_http_pool()
        response = await http_pool.post(
            f"{config.service.embedding_service_url}/v1/embeddings",
            json={"input": request.input, "model": request.model},
        )
        response.raise_for_status()
        data = response.json()

        if state.embedding_cache:
            state.embedding_cache.set(cache_key, data)

        return data

    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Embedding服务错误: {e!s}") from e
