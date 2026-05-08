#!/usr/bin/env python3
"""测试 Entity UPDATE 是否生效"""

import asyncio
import httpx


async def test_entity_update():
    """测试创建 Entity 后 atoms 是否正确关联"""

    async with httpx.AsyncClient() as client:
        # 1. 创建 Entity
        print("=== 步骤 1: 创建 Entity ===")
        request_data = {
            "type": "memory",
            "abstract": "测试 UPDATE 修复",
            "atoms": [
                {
                    "type": "section",
                    "name": "测试 Atom",
                    "content": "测试内容",
                    "local_id": "01KTEST01SE00000000000001"
                }
            ]
        }

        response = await client.post(
            "http://localhost:18008/api/v1/entities",
            json=request_data
        )

        if response.status_code != 200:
            print(f"❌ 创建失败: {response.status_code}")
            print(response.text)
            return False

        data = response.json()
        entity_id = data.get('id')
        print(f"✅ Entity 创建成功: {entity_id}")

        # 2. 获取 Entity（验证 atoms 不为空）
        print(f"\n=== 步骤 2: 获取 Entity {entity_id} ===")
        response = await client.get(
            f"http://localhost:18008/api/v1/entities/{entity_id}?level=2"
        )

        if response.status_code != 200:
            print(f"❌ 获取失败: {response.status_code}")
            return False

        data = response.json()
        atoms = data.get('atoms', [])
        print(f"Atoms count: {len(atoms)}")

        if len(atoms) == 0:
            print("❌ atoms 为空！UPDATE 未生效")
            return False

        for i, atom in enumerate(atoms):
            if isinstance(atom, dict):
                print(f"\nAtom {i+1}:")
                print(f"  - id: {atom.get('id')}")
                print(f"  - local_id: {atom.get('local_id')}")

        # 3. 验证 local_id 正确
        if atoms and isinstance(atoms[0], dict):
            local_id = atoms[0].get('local_id')
            if local_id == "01KTEST01SE00000000000001":
                print(f"\n✅ local_id 正确: {local_id}")
                return True
            else:
                print(f"\n❌ local_id 错误")
                print(f"  期望: 01KTEST01SE00000000000001")
                print(f"  实际: {local_id}")
                return False

        return False


if __name__ == "__main__":
    result = asyncio.run(test_entity_update())
    exit(0 if result else 1)
