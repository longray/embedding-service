#!/usr/bin/env python3
"""调试 local_id 保存问题"""

import asyncio
import httpx


async def test_local_id_save():
    """测试 local_id 是否正确保存到数据库"""

    async with httpx.AsyncClient() as client:
        # 创建 Entity 请求，使用完整格式的 local_id
        request_data = {
            "type": "memory",
            "abstract": "调试 local_id 保存",
            "atoms": [
                {
                    "type": "section",
                    "name": "测试 Atom",
                    "content": "测试内容",
                    "local_id": "01KTEST01SE00000000000001"  # 完整格式
                }
            ]
        }

        print("=== 发送请求 ===")
        print(f"local_id: {request_data['atoms'][0]['local_id']}")

        response = await client.post(
            "http://localhost:18008/api/v1/entities",
            json=request_data
        )

        print(f"\n=== 响应 ===")
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"Entity ID: {data.get('id')}")

            atoms = data.get('atoms', [])
            print(f"\nAtoms count: {len(atoms)}")

            for i, atom in enumerate(atoms):
                if isinstance(atom, dict):
                    print(f"\nAtom {i+1}:")
                    print(f"  - id: {atom.get('id')}")
                    print(f"  - local_id: {atom.get('local_id')}")
                    print(f"  - local_id 长度: {len(atom.get('local_id', ''))}")

                    # 验证 local_id 是否为完整格式
                    local_id = atom.get('local_id')
                    if local_id == "01KTEST01SE00000000000001":
                        print(f"  - ✅ local_id 正确（完整格式）")
                    else:
                        print(f"  - ❌ local_id 错误")
                        print(f"    期望: 01KTEST01SE00000000000001")
                        print(f"    实际: {local_id}")


if __name__ == "__main__":
    asyncio.run(test_local_id_save())
