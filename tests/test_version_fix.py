"""
BL-B-104: 版本号更新逻辑修复测试

验证 atom 更新时版本号正确递增。

运行方式:
    uv run pytest tests/test_version_fix.py -v
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


class TestAtomVersionFix:
    """测试 Atom 版本号更新修复"""

    async def test_update_atom_version_increment(self, client):
        """测试更新 Atom 时版本号递增"""
        # 创建 atom
        response = await client.post(
            f"{WRAPPER_URL}/api/v1/atoms",
            json={
                "type": "function",
                "content": "def test(): pass",
                "name": "test_func",
                "tenant_id": "test_version"
            }
        )
        assert response.status_code == 200
        atom = response.json()
        initial_version = atom.get("version", 1)

        # 更新 atom
        response = await client.put(
            f"{WRAPPER_URL}/api/v1/atoms/{atom['id']}",
            json={"content": "def test(): return 1"},
            params={"tenant_id": "test_version"}
        )
        assert response.status_code == 200
        updated = response.json()

        # 验证版本号递增
        assert updated["version"] == initial_version + 1, f"版本号应从 {initial_version} 递增到 {initial_version + 1}"

        # 再次更新
        response = await client.put(
            f"{WRAPPER_URL}/api/v1/atoms/{atom['id']}",
            json={"content": "def test(): return 2"},
            params={"tenant_id": "test_version"}
        )
        assert response.status_code == 200
        updated2 = response.json()

        # 验证版本号再次递增
        assert updated2["version"] == initial_version + 2, f"版本号应从 {initial_version + 1} 递增到 {initial_version + 2}"

    async def test_update_atom_version_persistence(self, client):
        """测试版本号持久化到数据库"""
        # 创建 atom
        response = await client.post(
            f"{WRAPPER_URL}/api/v1/atoms",
            json={
                "type": "function",
                "content": "def persist(): pass",
                "name": "persist_func",
                "tenant_id": "test_version"
            }
        )
        assert response.status_code == 200
        atom = response.json()

        # 更新多次
        for i in range(3):
            response = await client.put(
                f"{WRAPPER_URL}/api/v1/atoms/{atom['id']}",
                json={"content": f"def persist(): return {i}"},
                params={"tenant_id": "test_version"}
            )
            assert response.status_code == 200

        # 查询最终版本
        response = await client.get(
            f"{WRAPPER_URL}/api/v1/atoms/{atom['id']}",
            params={"tenant_id": "test_version"}
        )
        assert response.status_code == 200
        final = response.json()

        # 验证版本号为 4（初始 1 + 3 次更新）
        assert final["version"] == 4, f"版本号应为 4，实际为 {final['version']}"
