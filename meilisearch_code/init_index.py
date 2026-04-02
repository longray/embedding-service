# -*- coding: utf-8 -*-
"""
Meilisearch 索引初始化脚本
场景：代码/开发者工具
策略：双字段 + 零副作用
端口：18003

使用方法:
    python init_index.py
"""

import sys
import time
import json
from pathlib import Path
from config import MeiliConfig
from meilisearch.errors import MeilisearchApiError, MeilisearchTimeoutError


def check_server_health(max_retries=30, delay=2):
    """等待 Meilisearch 服务启动"""
    import requests

    print("⏳ 等待 Meilisearch 服务启动...")

    for i in range(max_retries):
        try:
            response = requests.get(f"{MeiliConfig.HOST}/health", timeout=5)
            if response.status_code == 200:
                print("✅ Meilisearch 服务已就绪")
                return True
        except requests.RequestException:
            # 连接失败，继续重试
            pass
        print(f"   重试 {i + 1}/{max_retries}...")
        time.sleep(delay)

    print("❌ Meilisearch 服务启动超时")
    return False


def check_version_compatibility():
    """检查版本兼容性"""
    print("\n🔍 检查版本兼容性...")
    success, message = MeiliConfig.check_version()

    if success:
        print(f"✅ 版本检查通过：{message}")
        return True
    else:
        print(f"❌ 版本检查失败：{message}")
        return False


def create_or_update_index():
    """创建或更新索引配置"""
    print("\n" + "=" * 70)
    print("🔧 开始配置 Meilisearch 索引")
    print("=" * 70)

    try:
        client = MeiliConfig.get_client()
        index = client.index(MeiliConfig.INDEX_NAME)

        # 1. 创建索引
        print(f"\n📁 索引名称：{MeiliConfig.INDEX_NAME}")
        try:
            client.create_index(MeiliConfig.INDEX_NAME, {"primaryKey": "id"})
            print("✅ 索引创建成功")
        except MeilisearchApiError as e:
            if "index_already_exists" in str(e).lower():
                print("ℹ️  索引已存在，将更新配置")
            else:
                raise

        # 2. 应用完整配置
        print("\n⚙️  应用索引配置...")
        print(f"   - 端口：{MeiliConfig.HOST}")
        print(f"   - nonSeparatorTokens: {len(MeiliConfig.SETTINGS['nonSeparatorTokens'])} 个字符")
        print(f"   - dictionary: {len(MeiliConfig.SETTINGS['dictionary'])} 个词条")
        print(f"   - localizedAttributes: {MeiliConfig.SETTINGS['localizedAttributes']}")

        task = index.update_settings(MeiliConfig.SETTINGS)
        client.wait_for_task(task.task_uid, timeout_in_ms=60000)
        print("✅ 配置更新成功")

        # 3. 验证配置
        print("\n🔍 验证配置...")
        current_settings = index.get_settings()

        assertions = [
            (
                current_settings.get("nonSeparatorTokens"),
                MeiliConfig.SETTINGS["nonSeparatorTokens"],
                "nonSeparatorTokens",
            ),
            (len(current_settings.get("localizedAttributes", [])) > 0, True, "localizedAttributes"),
            (len(current_settings.get("dictionary", [])) > 0, True, "dictionary"),
        ]

        for actual, expected, name in assertions:
            if name == "nonSeparatorTokens":
                if set(actual) == set(expected):
                    print(f"   ✅ {name} 验证通过")
                else:
                    print(f"   ❌ {name} 验证失败")
                    return False
            elif actual == expected or (isinstance(expected, bool) and bool(actual) == expected):
                print(f"   ✅ {name} 验证通过")
            else:
                print(f"   ❌ {name} 验证失败")
                return False

        # 4. 添加测试文档（双字段策略示例）
        print("\n📝 添加测试文档（双字段策略）...")
        test_documents = [
            {
                "id": 1,
                "title_zh": "用户服务类",
                "description_zh": "处理用户认证和授权的公共服务类",
                "content_zh": "包含登录、注册、权限验证等功能",
                "tags_zh": ["用户", "认证", "服务"],
                "file_path": "src/main/java/com/example/app/UserService.java",
                "version": "v2.1.0",
                "language": "java",
                "project_name": "example-app",
                "email": "developer@example.com",
                "ip_address": "192.168.1.100",
                "status": "active",
                "created_at": 1710230400,
                "updated_at": 1710230400,
                "file_size": 15360,
                "line_count": 520,
                "file_name_search": "UserService java",
                "class_name_search": "UserService",
                "method_name_search": "login register authenticate authorize",
                "namespace_search": "com example app",
                "code_content_search": "public class UserService implements AuthService",
            },
            {
                "id": 2,
                "title_zh": "API 路由配置",
                "description_zh": "定义 RESTful API 端点路由",
                "content_zh": "包含用户、订单、支付等模块的路由配置",
                "tags_zh": ["API", "路由", "配置"],
                "file_path": "src/api/routes.py",
                "version": "v2.1.0",
                "language": "python",
                "project_name": "example-app",
                "email": "backend@example.com",
                "ip_address": "10.0.0.1",
                "status": "active",
                "created_at": 1711929600,
                "updated_at": 1711929600,
                "file_size": 8192,
                "line_count": 280,
                "file_name_search": "routes python",
                "class_name_search": "APIRouter",
                "method_name_search": "get post put delete",
                "namespace_search": "src api",
                "code_content_search": "from fastapi import APIRouter router = APIRouter",
            },
            {
                "id": 3,
                "title_zh": "数据库连接工具",
                "description_zh": "封装数据库连接和查询操作",
                "content_zh": "支持 MySQL、PostgreSQL、SQLite 等多种数据库",
                "tags_zh": ["数据库", "连接", "工具"],
                "file_path": "src/utils/database.ts",
                "version": "v1.9.5",
                "language": "typescript",
                "project_name": "example-app",
                "email": "db@example.com",
                "ip_address": "172.16.0.50",
                "status": "deprecated",
                "created_at": 1733011200,
                "updated_at": 1733011200,
                "file_size": 12288,
                "line_count": 450,
                "file_name_search": "database typescript",
                "class_name_search": "DatabaseConnection",
                "method_name_search": "connect query execute transaction",
                "namespace_search": "src utils",
                "code_content_search": "export class DatabaseConnection implements Connection",
            },
            {
                "id": 4,
                "title_zh": "版本发布说明",
                "description_zh": "v2.1.0 版本的发布说明和变更日志",
                "content_zh": "新增用户认证模块，修复 IP 解析 bug，优化邮箱验证",
                "tags_zh": ["版本", "发布", "日志"],
                "file_path": "docs/release/v2.1.0.md",
                "version": "v2.1.0",
                "language": "markdown",
                "project_name": "example-app",
                "email": "release@example.com",
                "ip_address": "192.168.1.1",
                "status": "published",
                "created_at": 1710316800,
                "updated_at": 1710316800,
                "file_size": 4096,
                "line_count": 150,
                "file_name_search": "release markdown",
                "class_name_search": "",
                "method_name_search": "",
                "namespace_search": "docs release",
                "code_content_search": "version v2.1.0 release notes changelog",
            },
        ]

        task = index.add_documents(test_documents)
        client.wait_for_task(task.task_uid, timeout_in_ms=60000)
        print(f"✅ 已添加 {len(test_documents)} 条测试文档")

        # 5. 获取索引统计
        print("\n📊 索引统计:")
        stats = index.get_stats()
        print(f"   - 文档数量：{stats.number_of_documents}")
        print(f"   - 索引大小：不可用（新版本 API 不提供）")

        print("\n" + "=" * 70)
        print("🎉 索引初始化完成！")
        print("=" * 70)
        return True

    except MeilisearchTimeoutError as e:
        print(f"\n❌ 连接超时：{e}")
        return False
    except MeilisearchApiError as e:
        print(f"\n❌ API 错误：{e}")
        return False
    except Exception as e:
        print(f"\n❌ 未知错误：{e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("\n" + "🚀" * 35)
    print("   Meilisearch 索引初始化脚本")
    print("   代码/开发者工具场景 | 端口 18003 | 零副作用")
    print("🚀" * 35)

    if not check_server_health():
        print("\n💡 提示：请先启动 Meilisearch 服务")
        print("   Docker: docker-compose up -d")
        sys.exit(1)

    if not check_version_compatibility():
        sys.exit(1)

    success = create_or_update_index()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
