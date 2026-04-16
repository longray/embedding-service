"""测试项目代码分析端点 (BL-CA-23/25)"""

import asyncio
import httpx
import sys


async def test_project_endpoints():
    """测试项目地图和统计端点"""
    base_url = "http://localhost:18008"
    project_id = "test-project"
    tenant_id = "default"

    print("=" * 60)
    print("测试项目代码分析端点 (BL-CA-23/25)")
    print("=" * 60)

    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        # 测试健康检查
        print("\n1. 测试健康检查...")
        try:
            response = await client.get("/health")
            if response.status_code == 200:
                print("✅ 服务正常运行")
            else:
                print(f"⚠️ 健康检查返回: {response.status_code}")
        except Exception as e:
            print(f"❌ 健康检查失败: {e}")
            print("请确保服务已启动: uv run python -m wrapper.src.main")
            return False

        # 测试项目地图端点 (BL-CA-23)
        print(f"\n2. 测试项目地图端点 (BL-CA-23)...")
        print(f"   GET /api/v1/projects/{project_id}/map")
        try:
            response = await client.get(f"/api/v1/projects/{project_id}/map", params={"tenant_id": tenant_id})
            print(f"   状态码: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                print(f"✅ 请求成功")
                print(f"   响应键: {list(data.keys())}")

                if data.get("status") == "success":
                    stats = data.get("statistics", {})
                    print(f"   文件数: {stats.get('total_files', 0)}")
                    print(f"   函数数: {stats.get('total_functions', 0)}")
                    print(f"   类数: {stats.get('total_classes', 0)}")
                else:
                    print(f"   状态: {data.get('status')}")
            else:
                print(f"⚠️ 请求返回: {response.status_code}")
                print(f"   响应: {response.text[:200]}")
        except Exception as e:
            print(f"❌ 请求失败: {e}")

        # 测试项目统计端点 (BL-CA-25)
        print(f"\n3. 测试项目统计端点 (BL-CA-25)...")
        print(f"   GET /api/v1/projects/{project_id}/stats")
        try:
            response = await client.get(f"/api/v1/projects/{project_id}/stats", params={"tenant_id": tenant_id})
            print(f"   状态码: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                print(f"✅ 请求成功")
                print(f"   响应键: {list(data.keys())}")

                if data.get("status") == "success":
                    print(f"   项目ID: {data.get('project_id')}")
                    print(f"   总文件数: {data.get('total_files', 0)}")
                    print(f"   总函数数: {data.get('total_functions', 0)}")
                    print(f"   总类数: {data.get('total_classes', 0)}")
                    print(f"   平均复杂度: {data.get('avg_complexity', 0)}")
                    print(f"   最大复杂度: {data.get('max_complexity', 0)}")
                else:
                    print(f"   状态: {data.get('status')}")
            else:
                print(f"⚠️ 请求返回: {response.status_code}")
                print(f"   响应: {response.text[:200]}")
        except Exception as e:
            print(f"❌ 请求失败: {e}")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
    return True


if __name__ == "__main__":
    result = asyncio.run(test_project_endpoints())
    sys.exit(0 if result else 1)
