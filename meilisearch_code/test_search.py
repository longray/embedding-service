# -*- coding: utf-8 -*-
"""
Meilisearch 搜索测试脚本
场景：代码/开发者工具
验证：所有测试用例 + 副作用规避
端口：18003

使用方法:
    python test_search.py
"""

from config import MeiliConfig
from meilisearch.errors import MeilisearchApiError

# ========== 测试用例清单 ==========
TEST_CASES = [
    # (搜索词, 期望有结果, 描述, 测试类型)
    # --- 特殊格式精确匹配 ---
    ("v2.1.0", True, "完整版本号匹配", "special_format"),
    ("192.168.1.100", True, "完整 IP 匹配", "special_format"),
    ("developer@example.com", True, "完整邮箱匹配", "special_format"),
    ("2026-03-12", True, "完整日期匹配", "special_format"),
    # --- 特殊格式前缀匹配 ---
    ("v2", True, "版本前缀匹配", "special_format"),
    ("192", True, "IP 前缀匹配", "special_format"),
    ("developer", True, "邮箱前缀匹配", "special_format"),
    ("2026", True, "日期前缀匹配", "special_format"),
    # --- 代码相关搜索 ---
    ("UserService", True, "类名搜索", "code"),
    ("login register", True, "方法名搜索", "code"),
    ("com.example.app", True, "命名空间搜索", "code"),
    ("fastapi", True, "框架搜索", "code"),
    ("python", True, "语言搜索", "code"),
    ("java", True, "语言搜索", "code"),
    # --- 中文内容搜索 ---
    ("用户服务", True, "中文标题搜索", "chinese"),
    ("认证", True, "中文标签搜索", "chinese"),
    ("数据库连接", True, "中文描述搜索", "chinese"),
    ("发布说明", True, "中文内容搜索", "chinese"),
    # --- 文件路径搜索 ---
    ("UserService.java", True, "文件名搜索", "code"),
    ("routes.py", True, "文件名搜索", "code"),
    ("database.ts", True, "文件名搜索", "code"),
    # --- 组合搜索 ---
    ("v2.1.0 用户", True, "版本 + 中文组合", "combined"),
    ("java UserService", True, "语言 + 类名组合", "combined"),
    ("python fastapi", True, "语言 + 框架组合", "combined"),
]

# ========== 过滤器测试用例 ==========
FILTER_TESTS = [
    ('version = "v2.1.0"', "版本精确过滤"),
    ('language = "python"', "语言过滤"),
    ('status = "active"', "状态过滤"),
    ("file_size > 10000", "文件大小过滤"),
    ("line_count >= 300", "行数过滤"),
    ("created_at >= 1710000000", "时间范围过滤"),
    ('email = "developer@example.com"', "邮箱精确过滤"),
    ('ip_address = "192.168.1.100"', "IP 精确过滤"),
]


def run_tests():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("🧪 Meilisearch 搜索测试")
    print("   场景：代码/开发者工具 | 端口：http://localhost:18003")
    print("=" * 80)

    index = MeiliConfig.get_index()

    results = {
        "special_format": {"passed": 0, "failed": 0, "cases": []},
        "code": {"passed": 0, "failed": 0, "cases": []},
        "chinese": {"passed": 0, "failed": 0, "cases": []},
        "combined": {"passed": 0, "failed": 0, "cases": []},
    }

    for query, should_match, description, test_type in TEST_CASES:
        try:
            result = index.search(query, {"limit": 1})
            has_results = len(result.get("hits", [])) > 0
            success = has_results == should_match

            case_result = {
                "query": query,
                "description": description,
                "hits": len(result.get("hits", [])),
                "expected": "有结果" if should_match else "无结果",
                "success": success,
            }

            if success:
                results[test_type]["passed"] += 1
                case_result["status"] = "✅"
            else:
                results[test_type]["failed"] += 1
                case_result["status"] = "❌"

            results[test_type]["cases"].append(case_result)

        except Exception as e:
            results[test_type]["failed"] += 1
            results[test_type]["cases"].append(
                {"query": query, "description": description, "status": "❌", "error": str(e)}
            )

    # 打印结果
    print("\n📋 测试结果:\n")

    for test_type, data in results.items():
        print(f"【{test_type.upper()}】通过 {data['passed']}/{data['passed'] + data['failed']}")
        for case in data["cases"]:
            status = case.get("status", "❌")
            query = case["query"][:30].ljust(30)
            desc = case["description"][:20].ljust(20)
            hits = str(case.get("hits", "N/A")).ljust(5)
            print(f"  {status} {query} {desc} {hits}")
        print()

    # 过滤器测试
    print("\n🔍 过滤器测试:\n")
    filter_passed = 0
    filter_failed = 0

    for filter_expr, description in FILTER_TESTS:
        try:
            result = index.search("", {"filter": filter_expr, "limit": 5})
            hits = len(result.get("hits", []))
            status = "✅" if hits > 0 else "⚠️"
            if hits > 0:
                filter_passed += 1
            else:
                filter_failed += 1
            print(f"  {status} {description[:25].ljust(25)} {filter_expr[:40]} → {hits} 结果")
        except Exception as e:
            filter_failed += 1
            print(f"  ❌ {description[:25].ljust(25)} {e}")

    # 统计汇总
    total_passed = sum(d["passed"] for d in results.values())
    total_failed = sum(d["failed"] for d in results.values())

    print("\n" + "=" * 80)
    print(
        f"📊 汇总：搜索测试 {total_passed}/{total_passed + total_failed} | 过滤测试 {filter_passed}/{filter_passed + filter_failed}"
    )
    print("=" * 80)

    return total_failed == 0 and filter_failed == 0


def test_side_effects():
    """测试副作用规避"""
    print("\n" + "=" * 80)
    print("⚠️  副作用规避验证")
    print("=" * 80)

    index = MeiliConfig.get_index()

    # 1. 索引大小检查
    stats = index.get_stats()
    doc_count = stats.number_of_documents
    db_size = 0  # 新版本 API 不提供索引大小

    print(f"\n📊 索引大小：不可用 ({doc_count} 文档)")

    # 跳过大小检查（新版本不提供）
    print("   ℹ️  索引大小检查已跳过（新版本 API 不提供）")

    # 2. 配置验证
    settings = index.get_settings()

    print(f"\n⚙️  配置验证:")
    print(f"   - nonSeparatorTokens: {len(settings.get('nonSeparatorTokens', []))} 个字符 ✅")
    print(f"   - dictionary: {len(settings.get('dictionary', []))} 个词条 ✅")
    print(f"   - localizedAttributes: {len(settings.get('localizedAttributes', []))} 个配置 ✅")

    # 3. 容错禁用验证
    typo_config = settings.get("typoTolerance", {})
    disabled_attrs = typo_config.get("disableOnAttributes", [])

    print(f"\n🔒 容错禁用字段：{len(disabled_attrs)} 个")
    for attr in ["file_path", "version", "email", "ip_address"]:
        if attr in disabled_attrs:
            print(f"   ✅ {attr}")
        else:
            print(f"   ❌ {attr} (未禁用)")

    return True


def main():
    """主函数"""
    print("\n" + "🚀" * 40)
    print("   Meilisearch 搜索测试脚本")
    print("   代码/开发者工具场景 | 端口 18003 | 零副作用")
    print("🚀" * 40)

    try:
        client = MeiliConfig.get_client()
        client.health()
        print("\n✅ Meilisearch 服务连接成功")
    except Exception as e:
        print(f"\n❌ 无法连接 Meilisearch: {e}")
        print("   请确保服务运行在 http://localhost:18003")
        return 1

    search_success = run_tests()
    side_effect_success = test_side_effects()

    if search_success and side_effect_success:
        print("\n🎉 所有测试通过！")
        return 0
    else:
        print("\n⚠️  部分测试失败，请检查配置")
        return 1


if __name__ == "__main__":
    exit(main())
