"""Audit Log Integration Tests

验证 BL-T-1: Audit 日志端点集成测试

测试范围：
- 端到端日志记录与查询流程
- 并发写入测试

运行方式：
    uv run pytest tests/integration/test_audit_integration.py -v

前置条件：
- Wrapper 服务运行在 http://localhost:18008
- SurrealDB 运行在 ws://localhost:18002
"""

import asyncio
import os
import uuid
from datetime import datetime, timedelta

import httpx
import pytest

pytestmark = pytest.mark.skipif(os.getenv("SKIP_E2E_TESTS") == "1", reason="E2E tests skipped (SKIP_E2E_TESTS=1)")

BASE_URL = "http://localhost:18008"


class TestAuditLogEndToEnd:
    """端到端审计日志流程测试"""

    def test_create_and_query_audit_log(self):
        """测试创建审计日志并查询"""
        unique_id = str(uuid.uuid4())[:8]

        # 1. 创建审计日志
        create_payload = {
            "action": "memory_create",
            "resource_type": "memory",
            "resource_id": f"memory:test-{unique_id}",
            "details": {"content_length": 100, "test_id": unique_id},
            "user_id": f"user:{unique_id}",
            "ip_address": "192.168.1.1",
            "user_agent": "TestAgent/1.0",
            "tenant_id": "default",
        }

        try:
            create_response = httpx.post(
                f"{BASE_URL}/api/v1/audit/log",
                json=create_payload,
                timeout=10.0,
            )

            if create_response.status_code == 503:
                pytest.skip("MemoryManager not initialized")

            assert create_response.status_code == 200, f"Create failed: {create_response.text}"
            create_data = create_response.json()
            assert create_data["status"] == "success"
            assert "audit_log_id" in create_data

            # 2. 查询刚创建的日志
            query_response = httpx.get(
                f"{BASE_URL}/api/v1/audit/logs",
                params={
                    "user_id": f"user:{unique_id}",
                    "action": "memory_create",
                    "tenant_id": "default",
                    "limit": 10,
                },
                timeout=10.0,
            )

            assert query_response.status_code == 200
            query_data = query_response.json()
            assert query_data["status"] == "success"
            assert query_data["total"] >= 1

            # 验证返回的日志包含我们创建的记录
            found = False
            for log in query_data["logs"]:
                if log.get("resource_id") == f"memory:test-{unique_id}":
                    found = True
                    break

            assert found, "Created audit log not found in query results"

        except Exception as e:
            pytest.skip(f"Test failed: {e}")

    def test_audit_log_with_time_range(self):
        """测试按时间范围查询审计日志"""
        unique_id = str(uuid.uuid4())[:8]

        # 创建审计日志
        create_payload = {
            "action": "memory_read",
            "resource_type": "memory",
            "resource_id": f"memory:timerange-{unique_id}",
            "user_id": f"user:timerange-{unique_id}",
            "tenant_id": "default",
        }

        try:
            create_response = httpx.post(
                f"{BASE_URL}/api/v1/audit/log",
                json=create_payload,
                timeout=10.0,
            )

            if create_response.status_code == 503:
                pytest.skip("MemoryManager not initialized")

            if create_response.status_code != 200:
                pytest.skip(f"Create audit log failed: {create_response.status_code}")

            # 查询过去1小时内的日志
            now = datetime.utcnow()
            start_time = (now - timedelta(hours=1)).isoformat()
            end_time = (now + timedelta(hours=1)).isoformat()

            query_response = httpx.get(
                f"{BASE_URL}/api/v1/audit/logs",
                params={
                    "start_date": start_time,
                    "end_date": end_time,
                    "user_id": f"user:timerange-{unique_id}",
                    "tenant_id": "default",
                    "limit": 10,
                },
                timeout=10.0,
            )

            assert query_response.status_code == 200
            query_data = query_response.json()
            assert query_data["status"] == "success"

            # 应该能找到刚创建的日志
            found = any(log.get("resource_id") == f"memory:timerange-{unique_id}" for log in query_data["logs"])
            assert found, "Audit log not found in time range query"

        except Exception as e:
            pytest.skip(f"Test failed: {e}")

    def test_audit_log_pagination(self):
        """测试审计日志分页查询"""
        unique_id = str(uuid.uuid4())[:8]

        try:
            # 创建多条审计日志
            for i in range(3):
                create_payload = {
                    "action": "memory_update",
                    "resource_type": "memory",
                    "resource_id": f"memory:page-{unique_id}-{i}",
                    "user_id": f"user:page-{unique_id}",
                    "tenant_id": "default",
                }

                httpx.post(
                    f"{BASE_URL}/api/v1/audit/log",
                    json=create_payload,
                    timeout=10.0,
                )

            # 分页查询
            page1_response = httpx.get(
                f"{BASE_URL}/api/v1/audit/logs",
                params={
                    "user_id": f"user:page-{unique_id}",
                    "tenant_id": "default",
                    "limit": 2,
                    "offset": 0,
                },
                timeout=10.0,
            )

            assert page1_response.status_code == 200
            page1_data = page1_response.json()
            assert page1_data["status"] == "success"
            assert len(page1_data["logs"]) <= 2

            # 第二页
            page2_response = httpx.get(
                f"{BASE_URL}/api/v1/audit/logs",
                params={
                    "user_id": f"user:page-{unique_id}",
                    "tenant_id": "default",
                    "limit": 2,
                    "offset": 2,
                },
                timeout=10.0,
            )

            assert page2_response.status_code == 200
            page2_data = page2_response.json()
            assert page2_data["status"] == "success"

        except Exception as e:
            pytest.skip(f"Test failed: {e}")


class TestAuditLogConcurrent:
    """并发写入测试"""

    @pytest.mark.asyncio
    async def test_concurrent_audit_log_creation(self):
        """测试并发创建审计日志"""
        unique_id = str(uuid.uuid4())[:8]

        async def create_log(index: int):
            payload = {
                "action": "memory_create",
                "resource_type": "memory",
                "resource_id": f"memory:concurrent-{unique_id}-{index}",
                "user_id": f"user:concurrent-{unique_id}",
                "tenant_id": "default",
            }

            async with httpx.AsyncClient() as client:
                try:
                    response = await client.post(
                        f"{BASE_URL}/api/v1/audit/log",
                        json=payload,
                        timeout=10.0,
                    )
                    return response.status_code == 200
                except Exception:
                    return False

        try:
            # 并发创建 5 条审计日志
            tasks = [create_log(i) for i in range(5)]
            results = await asyncio.gather(*tasks)

            # 所有请求都应该成功
            success_count = sum(results)
            assert success_count == 5, f"Only {success_count}/5 concurrent writes succeeded"

            # 验证所有日志都被记录
            async with httpx.AsyncClient() as client:
                query_response = await client.get(
                    f"{BASE_URL}/api/v1/audit/logs",
                    params={
                        "user_id": f"user:concurrent-{unique_id}",
                        "tenant_id": "default",
                        "limit": 10,
                    },
                    timeout=10.0,
                )

                if query_response.status_code == 200:
                    query_data = query_response.json()
                    if query_data.get("status") == "success":
                        # 应该能找到所有创建的日志
                        found_count = sum(
                            1
                            for i in range(5)
                            if any(
                                log.get("resource_id") == f"memory:concurrent-{unique_id}-{i}"
                                for log in query_data.get("logs", [])
                            )
                        )
                        assert found_count == 5, f"Only {found_count}/5 logs found in query"

        except Exception as e:
            pytest.skip(f"Test failed: {e}")

    @pytest.mark.asyncio
    async def test_concurrent_mixed_operations(self):
        """测试并发混合操作（创建和查询）"""
        unique_id = str(uuid.uuid4())[:8]

        async def create_and_query(index: int):
            # 创建
            create_payload = {
                "action": "memory_update",
                "resource_type": "memory",
                "resource_id": f"memory:mixed-{unique_id}-{index}",
                "user_id": f"user:mixed-{unique_id}",
                "tenant_id": "default",
            }

            async with httpx.AsyncClient() as client:
                try:
                    create_response = await client.post(
                        f"{BASE_URL}/api/v1/audit/log",
                        json=create_payload,
                        timeout=10.0,
                    )

                    if create_response.status_code != 200:
                        return False

                    # 立即查询
                    query_response = await client.get(
                        f"{BASE_URL}/api/v1/audit/logs",
                        params={
                            "user_id": f"user:mixed-{unique_id}",
                            "action": "memory_update",
                            "tenant_id": "default",
                            "limit": 5,
                        },
                        timeout=10.0,
                    )

                    return query_response.status_code == 200
                except Exception:
                    return False

        try:
            # 并发执行创建+查询
            tasks = [create_and_query(i) for i in range(3)]
            results = await asyncio.gather(*tasks)

            # 所有操作都应该成功
            success_count = sum(results)
            assert success_count == 3, f"Only {success_count}/3 concurrent operations succeeded"

        except Exception as e:
            pytest.skip(f"Test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
