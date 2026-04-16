"""测试审计日志 API (P3-5)"""

import asyncio
import httpx


async def test_audit_api():
    """测试审计日志 API"""
    base_url = "http://localhost:18008"

    print("=" * 60)
    print("测试审计日志 API (P3-5)")
    print("=" * 60)

    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        # 1. 测试创建审计日志
        print("\n1. 测试创建审计日志...")
        try:
            response = await client.post(
                "/api/v1/audit/log",
                json={
                    "action": "memory_create",
                    "resource_type": "memory",
                    "resource_id": "memory:test-123",
                    "details": {"content_length": 100, "language": "python"},
                    "user_id": "user-001",
                    "ip_address": "192.168.1.100",
                    "user_agent": "TestClient/1.0",
                    "tenant_id": "default",
                },
            )
            print(f"   状态码: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ 创建成功: {data}")
            else:
                print(f"⚠️ 创建失败: {response.text}")
        except Exception as e:
            print(f"❌ 请求失败: {e}")

        # 2. 测试查询审计日志
        print("\n2. 测试查询审计日志...")
        try:
            response = await client.get(
                "/api/v1/audit/logs",
                params={
                    "action": "memory_create",
                    "tenant_id": "default",
                    "limit": 10,
                },
            )
            print(f"   状态码: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ 查询成功")
                print(f"   总数: {data.get('total', 0)}")
                print(f"   返回: {len(data.get('logs', []))} 条")
            else:
                print(f"⚠️ 查询失败: {response.text}")
        except Exception as e:
            print(f"❌ 请求失败: {e}")

        # 3. 测试清理审计日志（保留 30 天）
        print("\n3. 测试清理审计日志...")
        try:
            response = await client.delete(
                "/api/v1/audit/logs",
                params={
                    "retention_days": 30,
                    "tenant_id": "default",
                },
            )
            print(f"   状态码: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ 清理成功: {data}")
            else:
                print(f"⚠️ 清理失败: {response.text}")
        except Exception as e:
            print(f"❌ 请求失败: {e}")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_audit_api())
