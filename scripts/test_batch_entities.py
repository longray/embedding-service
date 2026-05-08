#!/usr/bin/env python3
"""测试批量创建 Entity"""

import asyncio
import httpx


async def test_batch_create_entities():
    """测试批量创建 Entities"""

    async with httpx.AsyncClient() as client:
        # 批量创建 Entity 请求
        request_data = {
            "entities": [
                {
                    "type": "memory",
                    "abstract": "测试 Entity 1",
                    "overview": {"key": "value1"},
                    "atoms": [
                        {
                            "type": "section",
                            "name": "测试 Atom 1",
                            "content": "测试内容 1",
                            "local_id": "01KTEST01SE00000000000001"
                        }
                    ]
                },
                {
                    "type": "memory",
                    "abstract": "测试 Entity 2",
                    "overview": {"key": "value2"},
                    "atoms": [
                        {
                            "type": "section",
                            "name": "测试 Atom 2",
                            "content": "测试内容 2",
                            "local_id": "01KTEST02SE00000000000002"
                        }
                    ]
                }
            ]
        }

        # 发送请求
        response = await client.post(
            "http://localhost:18008/api/v1/entities/batch",
            json=request_data
        )

        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"\nTotal: {data.get('total')}")
            print(f"Success: {data.get('success_count')}")
            print(f"Failed: {data.get('failed_count')}")

            for i, entity in enumerate(data.get('success', [])):
                print(f"\nEntity {i+1}:")
                print(f"  - ID: {entity.get('id')}")
                print(f"  - Abstract: {entity.get('abstract')}")

                atoms = entity.get('atoms', [])
                print(f"  - Atoms count: {len(atoms)}")

                for j, atom in enumerate(atoms):
                    if isinstance(atom, dict):
                        print(f"    - Atom {j+1}: id={atom.get('id')}, local_id={atom.get('local_id')}")

            # 验证所有 local_id 是否正确返回
            all_correct = True
            expected_local_ids = ["01KTEST01SE00000000000001", "01KTEST02SE00000000000002"]
            for i, entity in enumerate(data.get('success', [])):
                atoms = entity.get('atoms', [])
                if atoms and isinstance(atoms[0], dict):
                    local_id = atoms[0].get('local_id')
                    if local_id != expected_local_ids[i]:
                        print(f"\n❌ FAILED: Entity {i+1} local_id 不匹配 (期望: {expected_local_ids[i]}, 实际: {local_id})")
                        all_correct = False

            if all_correct and data.get('success_count') == 2:
                print("\n✅ SUCCESS: 批量创建成功，所有 local_id 正确返回！")
                return True
            else:
                print("\n❌ FAILED: 批量创建失败或 local_id 不匹配")
                return False
        else:
            print(f"\n❌ FAILED: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return False


if __name__ == "__main__":
    result = asyncio.run(test_batch_create_entities())
    exit(0 if result else 1)
