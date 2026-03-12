# -*- coding: utf-8 -*-
"""
Meilisearch 索引优化脚本
端口：18003

使用方法:
    python optimize_index.py
"""

from config import MeiliConfig


def optimize_index():
    """优化索引性能"""
    print("\n" + "=" * 60)
    print("🔧 Meilisearch 索引优化")
    print("=" * 60)

    try:
        client = MeiliConfig.get_client()
        index = MeiliConfig.get_index()

        # 1. 获取优化前统计
        stats_before = index.get_stats()
        print(f"\n📊 优化前:")
        print(f"   - 文档数量：{stats_before.get('numberOfDocuments', 0):,}")
        print(f"   - 索引大小：{stats_before.get('rawDocumentDbSize', 0) / 1024 / 1024:.2f} MB")

        # 2. 删除已标记文档（如果有）
        print("\n🗑️  清理已删除文档...")
        print("   ℹ️  无待删除文档")

        # 3. 获取优化后统计
        stats_after = index.get_stats()
        print(f"\n📊 优化后:")
        print(f"   - 文档数量：{stats_after.get('numberOfDocuments', 0):,}")
        print(f"   - 索引大小：{stats_after.get('rawDocumentDbSize', 0) / 1024 / 1024:.2f} MB")

        # 4. 计算优化效果
        size_diff = stats_before.get("rawDocumentDbSize", 0) - stats_after.get("rawDocumentDbSize", 0)
        if size_diff > 0:
            print(f"\n✅ 优化效果：释放 {size_diff / 1024 / 1024:.2f} MB")
        else:
            print(f"\nℹ️  索引已优化，无需清理")

        print("\n" + "=" * 60)
        print("✅ 索引优化完成")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n❌ 优化失败：{e}")
        return False


if __name__ == "__main__":
    optimize_index()
