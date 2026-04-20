"""
BL-B-102: 分页功能简单测试

验证分页响应结构。

运行方式:
    uv run pytest tests/test_pagination_simple.py -v
"""

import pytest
import pytest_asyncio
import httpx

pytestmark = pytest.mark.e2e

WRAPPER_URL = "http://localhost:18008"
DEFAULT_TIMEOUT = 30.0


@pytest_asyncio.fixture
async def client():
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as c:
        yield c


class TestAtomPagination:
    async def test_list_atoms_pagination_structure(self, client):
        response = await client.get(
            f"{WRAPPER_URL}/api/v1/atoms",
            params={"tenant_id": "default", "page": 1, "page_size": 2}
        )

        assert response.status_code == 200
        result = response.json()

        assert "data" in result
        assert "total" in result
        assert "page" in result
        assert "page_size" in result
        assert "has_more" in result
        assert isinstance(result["data"], list)
        assert isinstance(result["total"], int)
        assert isinstance(result["page"], int)
        assert isinstance(result["page_size"], int)
        assert isinstance(result["has_more"], bool)


class TestEntityPagination:
    async def test_list_entities_pagination_structure(self, client):
        response = await client.get(
            f"{WRAPPER_URL}/api/v1/entities",
            params={"tenant_id": "default", "page": 1, "page_size": 2}
        )

        assert response.status_code == 200
        result = response.json()

        assert "data" in result
        assert "total" in result
        assert "page" in result
        assert "page_size" in result
        assert "has_more" in result


class TestReferencePagination:
    async def test_list_references_pagination_structure(self, client):
        response = await client.get(
            f"{WRAPPER_URL}/api/v1/references",
            params={"tenant_id": "default", "page": 1, "page_size": 2}
        )

        assert response.status_code == 200
        result = response.json()

        assert "data" in result
        assert "total" in result
        assert "page" in result
        assert "page_size" in result
        assert "has_more" in result
