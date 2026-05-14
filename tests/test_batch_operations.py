"""
BL-B-103: 批量操作测试

验证 Atom/Entity 批量创建功能。

运行方式:
    uv run pytest tests/test_batch_operations.py -v
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


class TestAtomBatchOperations:
    """测试 Atom 批量操作"""

    async def test_batch_create_atoms_success(self, client):
        """测试批量创建 Atoms 成功"""
        response = await client.post(
            f"{WRAPPER_URL}/api/v1/atoms/batch",
            json={
                "atoms": [
                    {"type": "function", "content": "def a(): pass", "name": "a"},
                    {"type": "function", "content": "def b(): pass", "name": "b"}
                ],
                "tenant_id": "test_batch"
            }
        )

        assert response.status_code == 200
        result = response.json()

        # 验证响应结构 (BL-B-105: 修复测试断言匹配实际响应格式)
        assert "atoms" in result
        assert "total" in result
        assert "created" in result
        assert "skipped" in result
        assert "errors" in result

        # 验证数据
        assert result["total"] == 2
        assert result["created"] == 2
        assert result["skipped"] == 0
        assert result["errors"] == 0
        assert len(result["atoms"]) == 2
        assert result["atoms"][0]["status"] == "created"

    async def test_batch_create_atoms_partial_failure(self, client):
        """测试批量创建 Atoms 部分失败"""
        response = await client.post(
            f"{WRAPPER_URL}/api/v1/atoms/batch",
            json={
                "atoms": [
                    {"type": "function", "content": "def c(): pass", "name": "c"},
                    {"type": "invalid_type", "content": "def d(): pass", "name": "d"}
                ],
                "tenant_id": "test_batch"
            }
        )

        assert response.status_code == 200
        result = response.json()

        # 验证部分失败 (BL-B-105: 修复测试断言匹配实际响应格式)
        assert result["total"] == 2
        assert result["created"] == 1
        assert result["errors"] == 1
        assert len(result["atoms"]) == 2
        # 找到错误的那条
        error_atoms = [a for a in result["atoms"] if a["status"] == "error"]
        assert len(error_atoms) == 1

    async def test_batch_create_atoms_exceed_limit(self, client):
        """测试批量创建 Atoms 超过限制"""
        response = await client.post(
            f"{WRAPPER_URL}/api/v1/atoms/batch",
            json={
                "atoms": [{"type": "function", "content": f"def func_{i}(): pass"} for i in range(101)],
                "tenant_id": "test_batch"
            }
        )

        assert response.status_code == 400
        result = response.json()
        # BL-B-105: 错误消息包含"最多 100 条"
        assert "最多 100 条" in result["detail"]


class TestEntityBatchOperations:
    """测试 Entity 批量操作"""

    async def test_batch_create_entities_success(self, client):
        """测试批量创建 Entities 成功"""
        response = await client.post(
            f"{WRAPPER_URL}/api/v1/entities/batch",
            json={
                "entities": [
                    {"type": "memory", "abstract": "Entity 1"},
                    {"type": "memory", "abstract": "Entity 2"}
                ],
                "tenant_id": "test_batch"
            }
        )

        assert response.status_code == 200
        result = response.json()

        # 验证响应结构 (BL-B-105: 修复测试断言匹配实际响应格式)
        assert "entities" in result
        assert "total" in result
        assert "created" in result
        assert "skipped" in result
        assert "errors" in result

        # 验证数据
        assert result["total"] == 2
        assert result["created"] == 2
        assert result["skipped"] == 0
        assert result["errors"] == 0

    async def test_batch_create_entities_exceed_limit(self, client):
        """测试批量创建 Entities 超过限制"""
        response = await client.post(
            f"{WRAPPER_URL}/api/v1/entities/batch",
            json={
                "entities": [{"type": "memory", "abstract": f"Entity {i}"} for i in range(101)],
                "tenant_id": "test_batch"
            }
        )

        assert response.status_code == 400
        result = response.json()
        # BL-B-105: 错误消息包含"最多 100 条"
        assert "最多 100 条" in result["detail"]
