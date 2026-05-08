#!/usr/bin/env python3
"""测试 Atom local_id 是否正确返回"""

import asyncio
import httpx


async def test_create_entity_with_atoms():
    """测试创建 Entity 时 atoms 是否包含 local_id"""
    
    async with httpx.AsyncClient() as client:
        # 创建 Entity 请求
        request_data = {
            "type": "memory",
            "abstract": "测试 Entity",
            "overview": {"key": "value"},
            "atoms": [
                {
                    "type": "section",
                    "name": "测试 Atom",
                    "content": "测试内容",
                    "local_id": "01KTEST01SE00000000000001"
                }
            ]
        }
        
        # 发送请求
        response = await client.post(
            "http://localhost:18008/api/v1/entities",
            json=request_data
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\nEntity ID: {data.get('id')}")
            print(f"Abstract: {data.get('abstract')}")
            
            atoms = data.get('atoms', [])
            print(f"\nAtoms count: {len(atoms)}")
            
            for i, atom in enumerate(atoms):
                print(f"\nAtom {i+1}:")
                if isinstance(atom, dict):
                    print(f"  - id: {atom.get('id')}")
                    print(f"  - local_id: {atom.get('local_id')}")
                else:
                    print(f"  - {atom} (unexpected type: {type(atom)})")
            
            # 验证 local_id 是否正确返回
            if atoms and isinstance(atoms[0], dict):
                local_id = atoms[0].get('local_id')
                if local_id == "01KTEST01SE00000000000001":
                    print("\n✅ SUCCESS: local_id 正确返回！")
                    return True
                else:
                    print(f"\n❌ FAILED: local_id 不匹配 (期望: 01KTEST01SE00000000000001, 实际: {local_id})")
                    return False
            else:
                print("\n❌ FAILED: atoms 格式不正确")
                return False
        else:
            print(f"\n❌ FAILED: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return False


if __name__ == "__main__":
    result = asyncio.run(test_create_entity_with_atoms())
    exit(0 if result else 1)
