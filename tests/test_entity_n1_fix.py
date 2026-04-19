"""
BL-B-99: N+1 查询修复测试

验证 Entity 创建时的 Atom 验证从 N+1 查询改为批量查询。

运行方式:
    uv run pytest tests/test_entity_n1_fix.py -v
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


@pytest_asyncio.fixture
async def test_atoms(client):
    """创建测试用的 atoms"""
    atoms = []
    for i in range(5):
        response = await client.post(
            f"{WRAPPER_URL}/api/v1/atoms",
            json={
                "type": "function",
                "content": f"def test_func_{i}(): pass",
                "name": f"test_func_{i}",
                "tenant_id": "test_tenant"
            }
        )
        assert response.status_code == 200
        atoms.append(response.json())
    return atoms


class TestEntityN1Fix:
    """测试 N+1 查询修复"""

    async def test_create_entity_with_multiple_atoms(self, client, test_atoms):
        """测试创建 Entity 引用多个 Atoms"""
        atom_ids = [atom["id"] for atom in test_atoms]

        response = await client.post(
            f"{WRAPPER_URL}/api/v1/entities",
            json={
                "type": "memory",
                "abstract": "测试 Entity",
                "overview": {"key": "value"},
                "atoms": atom_ids,
                "tenant_id": "test_tenant"
            }
        )

        assert response.status_code == 200
        result = response.json()
        assert result["type"] == "memory"
        assert result["abstract"] == "测试 Entity"
        assert set(result["atoms"]) == set(atom_ids)

    async def test_create_entity_with_invalid_atoms(self, client):
        """测试创建 Entity 引用不存在的 Atoms"""
        response = await client.post(
            f"{WRAPPER_URL}/api/v1/entities",
            json={
                "type": "memory",
                "abstract": "测试 Entity",
                "atoms": ["atom:nonexistent1", "atom:nonexistent2"],
                "tenant_id": "test_tenant"
            }
        )

        assert response.status_code == 400
        result = response.json()
        assert "Atoms 不存在" in result["detail"]
        # 验证返回了所有不存在的 atoms
        assert "atom:nonexistent1" in result["detail"] or "atom:nonexistent2" in result["detail"]

    async def test_create_entity_without_atoms(self, client):
        """测试创建 Entity 不引用任何 Atoms"""
        response = await client.post(
            f"{WRAPPER_URL}/api/v1/entities",
            json={
                "type": "memory",
                "abstract": "测试 Entity 无 Atoms",
                "tenant_id": "test_tenant"
            }
        )

        assert response.status_code == 200
        result = response.json()
        assert result["type"] == "memory"
        assert result.get("atoms", []) == []

    async def test_create_entity_partial_invalid_atoms(self, client, test_atoms):
        """测试创建 Entity 部分 Atoms 存在，部分不存在"""
        valid_atom_id = test_atoms[0]["id"]
        invalid_atom_id = "atom:nonexistent"

        response = await client.post(
            f"{WRAPPER_URL}/api/v1/entities",
            json={
                "type": "memory",
                "abstract": "测试 Entity",
                "atoms": [valid_atom_id, invalid_atom_id],
                "tenant_id": "test_tenant"
            }
        )

        assert response.status_code == 400
        result = response.json()
        assert "Atoms 不存在" in result["detail"]
        assert invalid_atom_id in result["detail"]


class TestEntityN1Performance:
    """测试 N+1 修复后的性能"""

    async def test_create_entity_with_many_atoms(self, client):
        """测试创建 Entity 引用大量 Atoms（验证批量查询性能）"""
        # 创建 20 个 atoms
        atoms = []
        for i in range(20):
            response = await client.post(
                f"{WRAPPER_URL}/api/v1/atoms",
                json={
                    "type": "function",
                    "content": f"def func_{i}(): pass",
                    "name": f"func_{i}",
                    "tenant_id": "perf_test"
                }
            )
            assert response.status_code == 200
            atoms.append(response.json())

        atom_ids = [atom["id"] for atom in atoms]

        # 创建 entity 引用所有 atoms
        response = await client.post(
            f"{WRAPPER_URL}/api/v1/entities",
            json={
                "type": "memory",
                "abstract": "性能测试 Entity",
                "atoms": atom_ids,
                "tenant_id": "perf_test"
            }
        )

        assert response.status_code == 200
        result = response.json()
        assert len(result["atoms"]) == 20
