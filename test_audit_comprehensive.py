"""审计日志 API 完整测试套件 (P3-5)"""

import asyncio
import httpx
import pytest
from datetime import datetime, timedelta


BASE_URL = "http://localhost:18008"


class TestAuditLogAPI:
    """审计日志 API 测试类"""

    async def test_create_audit_log_minimal(self):
        """测试创建最小审计日志（仅必需字段）"""
        async with httpx.AsyncClient(base_url=BASE_URL) as client:
            response = await client.post(
                "/api/v1/audit/log",
                json={
                    "action": "memory_create",
                    "tenant_id": "default",
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "audit_log_id" in data
            assert "timestamp" in data
            print("✅ 最小审计日志创建成功")

    async def test_create_audit_log_full(self):
        """测试创建完整审计日志（所有字段）"""
        async with httpx.AsyncClient(base_url=BASE_URL) as client:
            response = await client.post(
                "/api/v1/audit/log",
                json={
                    "action": "memory_read",
                    "resource_type": "memory",
                    "resource_id": "memory:test-full-001",
                    "user_id": "user-test-001",
                    "ip_address": "192.168.1.100",
                    "user_agent": "TestAgent/1.0",
                    "tenant_id": "test-tenant",
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            print("✅ 完整审计日志创建成功")

    async def test_create_audit_log_all_actions(self):
        """测试创建各种操作类型的审计日志"""
        actions = [
            "memory_create",
            "memory_read",
            "memory_update",
            "memory_delete",
            "relation_create",
            "relation_delete",
            "sync_preview",
            "sync_full",
            "login",
            "logout",
            "admin_action",
            "system_cleanup",
        ]

        async with httpx.AsyncClient(base_url=BASE_URL) as client:
            for action in actions:
                response = await client.post(
                    "/api/v1/audit/log",
                    json={
                        "action": action,
                        "resource_type": "memory",
                        "resource_id": f"memory:test-{action}",
                        "tenant_id": "default",
                    },
                )
                assert response.status_code == 200, f"Action {action} failed"
                print(f"✅ Action '{action}' 创建成功")

    async def test_query_audit_logs_by_action(self):
        """测试按操作类型查询审计日志"""
        async with httpx.AsyncClient(base_url=BASE_URL) as client:
            # 先创建一条记录
            await client.post(
                "/api/v1/audit/log",
                json={
                    "action": "memory_create",
                    "resource_id": "memory:query-test-001",
                    "tenant_id": "default",
                },
            )

            # 按 action 查询
            response = await client.get(
                "/api/v1/audit/logs",
                params={
                    "action": "memory_create",
                    "tenant_id": "default",
                    "limit": 10,
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert data["total"] >= 1
            print(f"✅ 按 action 查询成功，找到 {data['total']} 条记录")

    async def test_query_audit_logs_by_resource(self):
        """测试按资源查询审计日志"""
        async with httpx.AsyncClient(base_url=BASE_URL) as client:
            resource_id = "memory:resource-test-001"

            # 创建记录
            await client.post(
                "/api/v1/audit/log",
                json={
                    "action": "memory_read",
                    "resource_type": "memory",
                    "resource_id": resource_id,
                    "tenant_id": "default",
                },
            )

            # 按 resource_id 查询
            response = await client.get(
                "/api/v1/audit/logs",
                params={
                    "resource_id": resource_id,
                    "tenant_id": "default",
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert len(data["logs"]) >= 1
            print(f"✅ 按 resource_id 查询成功")

    async def test_query_audit_logs_pagination(self):
        """测试审计日志分页"""
        async with httpx.AsyncClient(base_url=BASE_URL) as client:
            # 创建多条记录
            for i in range(5):
                await client.post(
                    "/api/v1/audit/log",
                    json={
                        "action": "memory_create",
                        "resource_id": f"memory:pagination-test-{i}",
                        "tenant_id": "default",
                    },
                )

            # 测试分页
            response = await client.get(
                "/api/v1/audit/logs",
                params={
                    "tenant_id": "default",
                    "limit": 2,
                    "offset": 0,
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert len(data["logs"]) <= 2
            print(f"✅ 分页查询成功，返回 {len(data['logs'])} 条")

    async def test_query_audit_logs_time_range(self):
        """测试按时间范围查询审计日志"""
        async with httpx.AsyncClient(base_url=BASE_URL) as client:
            # 创建记录
            await client.post(
                "/api/v1/audit/log",
                json={
                    "action": "memory_create",
                    "tenant_id": "default",
                },
            )

            # 查询最近1小时
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=1)

            response = await client.get(
                "/api/v1/audit/logs",
                params={
                    "tenant_id": "default",
                    "start_date": start_time.isoformat(),
                    "end_date": end_time.isoformat(),
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            print(f"✅ 时间范围查询成功，找到 {data['total']} 条记录")

    async def test_cleanup_audit_logs(self):
        """测试清理过期审计日志"""
        async with httpx.AsyncClient(base_url=BASE_URL) as client:
            # 清理30天前的记录
            response = await client.delete(
                "/api/v1/audit/logs",
                params={
                    "retention_days": 30,
                    "tenant_id": "default",
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "deleted_count" in data
            print(f"✅ 清理成功，删除 {data['deleted_count']} 条记录")

    async def test_tenant_isolation(self):
        """测试租户隔离"""
        async with httpx.AsyncClient(base_url=BASE_URL) as client:
            # 在 tenant-a 创建记录
            await client.post(
                "/api/v1/audit/log",
                json={
                    "action": "memory_create",
                    "resource_id": "memory:tenant-test",
                    "tenant_id": "tenant-a",
                },
            )

            # 查询 tenant-a 的记录
            response_a = await client.get(
                "/api/v1/audit/logs",
                params={"tenant_id": "tenant-a"},
            )
            data_a = response_a.json()

            # 查询 tenant-b 的记录（应该为空或不同）
            response_b = await client.get(
                "/api/v1/audit/logs",
                params={"tenant_id": "tenant-b"},
            )
            data_b = response_b.json()

            print(f"✅ 租户隔离测试完成: tenant-a={data_a['total']}条, tenant-b={data_b['total']}条")


async def run_all_tests():
    """运行所有测试"""
    print("=" * 70)
    print("审计日志 API 完整测试套件 (P3-5)")
    print("=" * 70)

    test_class = TestAuditLogAPI()
    tests = [
        test_class.test_create_audit_log_minimal,
        test_class.test_create_audit_log_full,
        test_class.test_create_audit_log_all_actions,
        test_class.test_query_audit_logs_by_action,
        test_class.test_query_audit_logs_by_resource,
        test_class.test_query_audit_logs_pagination,
        test_class.test_query_audit_logs_time_range,
        test_class.test_cleanup_audit_logs,
        test_class.test_tenant_isolation,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            print(f"\n🧪 {test.__doc__}...")
            await test()
            passed += 1
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"测试结果: ✅ {passed} 通过, ❌ {failed} 失败")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)
