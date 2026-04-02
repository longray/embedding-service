# -*- coding: utf-8 -*-
"""
Meilisearch 索引监控脚本
端口：18003

使用方法:
    python monitor_index.py
"""

from config import MeiliConfig


def check_index_health():
    """检查索引健康状态"""
    print("\n" + "=" * 60)
    print("📊 Meilisearch 索引健康检查")
    print("=" * 60)

    try:
        client = MeiliConfig.get_client()
        index = MeiliConfig.get_index()

        # 1. 服务健康
        health = client.health()
        print(f"\n✅ 服务状态：{health.get('status', 'unknown')}")

        # 2. 索引统计
        stats = index.get_stats()
        stats_dict = stats.model_dump()
        doc_count = stats_dict.get("number_of_documents", 0)
        db_size = stats_dict.get("raw_document_db_size", 0)

        print(f"\n📁 索引统计:")
        print(f"   - 文档数量：{doc_count:,}")
        print(f"   - 索引大小：{db_size / 1024 / 1024:.2f} MB")

        # 3. 阈值检查
        print(f"\n⚠️  阈值检查:")

        size_limit = MeiliConfig.MAX_INDEX_SIZE_GB * 1024**3
        if db_size < size_limit:
            print(f"   ✅ 索引大小：{db_size / size_limit * 100:.1f}% / {MeiliConfig.MAX_INDEX_SIZE_GB}GB")
        else:
            print(f"   ❌ 索引大小超限：{db_size / size_limit * 100:.1f}%")

        doc_limit = MeiliConfig.MAX_DOCUMENTS
        if doc_count < doc_limit:
            print(f"   ✅ 文档数量：{doc_count / doc_limit * 100:.1f}% / {doc_limit:,}")
        else:
            print(f"   ❌ 文档数量超限：{doc_count / doc_limit * 100:.1f}%")

        # 4. 配置检查
        settings = index.get_settings()
        print(f"\n⚙️  配置检查:")
        print(f"   - nonSeparatorTokens: {len(settings.get('nonSeparatorTokens', []))} 个")
        print(f"   - dictionary: {len(settings.get('dictionary', []))} 个")
        print(f"   - localizedAttributes: {len(settings.get('localizedAttributes', []))} 个")

        # 5. 版本检查
        version = client.get_version()
        print(f"\n📦 版本信息:")
        print(f"   - Meilisearch: {version.get('pkgVersion', 'unknown')}")
        print(f"   - Commit SHA: {version.get('commitSha', 'unknown')[:8]}")

        print("\n" + "=" * 60)
        print("✅ 索引健康检查完成")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n❌ 检查失败：{e}")
        return False


if __name__ == "__main__":
    check_index_health()
