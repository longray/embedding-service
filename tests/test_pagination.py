"""
BL-B-102: 分页功能测试

验证 Atom/Entity/Reference 列表查询的分页功能。

运行方式:
    uv run pytest tests/test_pagination.py -v
"""

import pytest
import pytest_asyncio
import httpx

pytestmark = pytest.mark.e2e

WRAPPER_URL = "http://localhost:18008"
DEFAULT_TIMEOUT = 30.0


@pytest_asyncio.fixture
async def client():
    """HTTP客户端fixture"""
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as c:
        yield c


class TestAtomPagination:
    """测试 Atom 分页功能"""

    async def test_list_atoms_pagination(self, client):
        """测试 Atom 列表分页"""
        # 创建多个 atoms
        for i in range(5):
            response = await client.post(
                f"{WRAPPER_URL}/api/v1/atoms",
                json={
                    "type": "function",
                    "content": f"def func_{i}(): pass",
                    "name": f"func_{i}",
                    "tenant_id": "test_pagination"
                }
            )
            assert response.status_code == 200

        # 测试分页查询 - 第1页，每页2条
        response = await client.get(
            f"{WRAPPER_URL}/api/v1/atoms",
            params={
                "tenant_id": "test_pagination",
                "page": 1,
                "page_size": 2
            }
        )

        assert response.status_code == 200
        result = response.json()

        # 验证分页响应结构
        assert "data" in result
        assert "total" in result
        assert "page" in result
        assert "page_size" in result
        assert "has_more" in result

        # 验证分页数据
        assert result["page"] == 1
        assert result["page_size"] == 2
        assert len(result["data"]) == 2
        assert result["total"] >= 5
        assert result["has_more"] is True

    async def test_list_atoms_backward_compatible(self, client):
        """测试 Atom 列表向后兼容（limit/offset）"""
        # 使用旧的 limit/offset 参数
        response = await client.get(
            f"{WRAPPER_URL}/api/v1/atoms",
            params={
                "tenant_id": "test_pagination",
                "limit": 2,
                "offset": 0
            }
        )

        assert response.status_code == 200
        result = response.json()

        # 验证分页响应结构
        assert "data" in result
        assert "total" in result
        assert "page" in result
        assert "page_size" in result
        assert "has_more" in result

        # 验证数据
        assert len(result["data"]) == 2


class TestEntityPagination:
    """测试 Entity 分页功能"""

    async def test_list_entities_pagination(self, client):
        """测试 Entity 列表分页"""
        # 创建多个 entities
        for i in range(3):
            response = await client.post(
                f"{WRAPPER_URL}/api/v1/entities",
                json={
                    "type": "memory",
                    "abstract": f"测试 Entity {i}",
                    "tenant_id": "test_pagination"
                }
            )
            assert response.status_code == 200

        # 测试分页查询
        response = await client.get(
            f"{WRAPPER_URL}/api/v1/entities",
            params={
                "tenant_id": "test_pagination",
                "page": 1,
                "page_size": 2
            }
        )

        assert response.status_code == 200
        result = response.json()

        # 验证分页响应结构
        assert "data" in result
        assert "total" in result
        assert "page" in result
        assert "page_size" in result
        assert "has_more" in result

        # 验证分页数据
        assert result["page"] == 1
        assert result["page_size"] == 2
        assert len(result["data"]) <= 2
        assert result["total"] >= 3


class TestReferencePagination:
    """测试 Reference 分页功能"""

    async def test_list_references_pagination(self, client):
        """测试 Reference 列表分页"""
        # 创建两个 atoms
        atoms = []
        for i in range(2):
            response = await client.post(
                f"{WRAPPER_URL}/api/v1/atoms",
                json={
                    "type": "function",
                    "content": f"def ref_func_{i}(): pass",
                    "name": f"ref_func_{i}",
                    "tenant_id": "test_pagination"
                }
            )
            assert response.status_code == 200
            atoms.append(response.json())

        # 创建 reference
        response = await client.post(
            f"{WRAPPER_URL}/api/v1/references",
            json={
                "from_id": atoms[0]["id"],
                "to_id": atoms[1]["id"],
                "type": "calls",
                "tenant_id": "test_pagination"
            }
        )
        assert response.status_code == 200

        # 测试分页查询
        response = await client.get(
            f"{WRAPPER_URL}/api/v1/references",
            params={
                "tenant_id": "test_pagination",
                "page": 1,
                "page_size": 10
            }
        )

        assert response.status_code == 200
        result = response.json()

        # 验证分页响应结构
        assert "data" in result
        assert "total" in result
        assert "page" in result
        assert "page_size" in result
        assert "has_more" in result
