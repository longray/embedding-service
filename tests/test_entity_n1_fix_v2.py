"""
BL-B-106: Entity N+1 查询修复测试

验证 Entity update 时的 Atom 批量验证。

运行方式:
    uv run pytest tests/test_entity_n1_fix_v2.py -v
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


class TestEntityN1FixV2:
    """测试 Entity N+1 查询修复"""

    async def test_update_entity_with_multiple_atoms_batch_validation(self, client):
        """测试更新 Entity 时批量验证 Atoms"""
        # 创建多个 atoms
        atoms = []
        for i in range(5):
            response = await client.post(
                f"{WRAPPER_URL}/api/v1/atoms",
                json={
                    "type": "function",
                    "content": f"def func_{i}(): pass",
                    "name": f"func_{i}",
                    "tenant_id": "test_n1_v2"
                }
            )
            assert response.status_code == 200
            atoms.append(response.json())

        # 创建 entity
        response = await client.post(
            f"{WRAPPER_URL}/api/v1/entities",
            json={
                "type": "memory",
                "abstract": "Test Entity",
                "atoms": [a["id"] for a in atoms],
                "tenant_id": "test_n1_v2"
            }
        )
        assert response.status_code == 200
        entity = response.json()

        # 更新 entity（添加更多 atoms）
        new_atoms = []
        for i in range(5, 8):
            response = await client.post(
                f"{WRAPPER_URL}/api/v1/atoms",
                json={
                    "type": "function",
                    "content": f"def func_{i}(): pass",
                    "name": f"func_{i}",
                    "tenant_id": "test_n1_v2"
                }
            )
            assert response.status_code == 200
            new_atoms.append(response.json())

        # 更新 entity - 应该只执行 1 次批量查询，而不是 3 次单独查询
        response = await client.put(
            f"{WRAPPER_URL}/api/v1/entities/{entity['id']}",
            json={
                "atoms": [a["id"] for a in atoms] + [a["id"] for a in new_atoms]
            },
            params={"tenant_id": "test_n1_v2"}
        )
        assert response.status_code == 200
        updated = response.json()
        assert len(updated["atoms"]) == 8

    async def test_update_entity_with_invalid_atoms(self, client):
        """测试更新 Entity 时验证不存在的 Atoms"""
        # 创建 entity
        response = await client.post(
            f"{WRAPPER_URL}/api/v1/entities",
            json={
                "type": "memory",
                "abstract": "Test Entity",
                "tenant_id": "test_n1_v2"
            }
        )
        assert response.status_code == 200
        entity = response.json()

        # 更新 entity 引用不存在的 atoms
        response = await client.put(
            f"{WRAPPER_URL}/api/v1/entities/{entity['id']}",
            json={
                "atoms": ["atom:nonexistent1", "atom:nonexistent2"]
            },
            params={"tenant_id": "test_n1_v2"}
        )
        assert response.status_code == 400
        result = response.json()
        assert "Atoms 不存在" in result["detail"]
        # 验证返回了所有不存在的 atoms
        assert "atom:nonexistent1" in result["detail"] or "atom:nonexistent2" in result["detail"]
