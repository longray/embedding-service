"""Meilisearch SDK 异步客户端测试

测试范围：
- 异步连接/关闭
- 异步索引管理
- 异步文档操作
- 异步搜索

运行方式：
    uv run pytest tests/test_meili_async.py -v
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from wrapper.src.utils.meili_sdk_client import (
    MeilisearchSDKClient,
    AsyncMeilisearchSDKClient,
)


class TestAsyncMeilisearchSDKClient:
    """异步 Meilisearch SDK 客户端测试"""

    @pytest.fixture
    def async_client(self):
        """创建 AsyncMeilisearchSDKClient 实例"""
        return AsyncMeilisearchSDKClient(
            url="http://localhost:7700",
            api_key="test_key",
            index_name="test_index",
        )

    @pytest.mark.asyncio
    async def test_async_client_init(self, async_client):
        """测试异步客户端初始化"""
        assert async_client._sync_client is not None

    @pytest.mark.asyncio
    async def test_connect(self, async_client):
        """测试异步连接"""
        with patch.object(
            async_client._sync_client,
            "connect",
            MagicMock(),
        ) as mock_connect:
            await async_client.connect()
            mock_connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_close(self, async_client):
        """测试异步关闭"""
        with patch.object(
            async_client._sync_client,
            "close",
            MagicMock(),
        ) as mock_close:
            await async_client.close()
            mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_index(self, async_client):
        """测试异步确保索引存在"""
        with patch.object(
            async_client._sync_client,
            "ensure_index",
            MagicMock(),
        ) as mock_ensure:
            await async_client.ensure_index(primary_key="id")
            mock_ensure.assert_called_once_with("id")

    @pytest.mark.asyncio
    async def test_configure_index(self, async_client):
        """测试异步配置索引"""
        settings = {"searchableAttributes": ["title"]}
        with patch.object(
            async_client._sync_client,
            "configure_index",
            MagicMock(),
        ) as mock_configure:
            await async_client.configure_index(settings)
            mock_configure.assert_called_once_with(settings)

    @pytest.mark.asyncio
    async def test_add_documents(self, async_client):
        """测试异步添加文档"""
        docs = [{"id": "1", "title": "Test"}]
        with patch.object(
            async_client._sync_client,
            "add_documents",
            return_value={"taskUid": 1},
        ) as mock_add:
            result = await async_client.add_documents(docs)
            mock_add.assert_called_once_with(docs, "id", wait=True)
            assert result == {"taskUid": 1}

    @pytest.mark.asyncio
    async def test_search(self, async_client):
        """测试异步搜索"""
        expected_result = {
            "hits": [{"id": "1", "title": "Test"}],
            "estimatedTotalHits": 1,
        }
        with patch.object(
            async_client._sync_client,
            "search",
            return_value=expected_result,
        ) as mock_search:
            result = await async_client.search("test query")
            mock_search.assert_called_once_with(
                "test query",
                filter_expr=None,
                limit=10,
                offset=0,
                sort=None,
                attributes_to_retrieve=None,
                show_ranking_score=True,
            )
            assert result == expected_result

    @pytest.mark.asyncio
    async def test_health(self, async_client):
        """测试异步健康检查"""
        with patch.object(
            async_client._sync_client,
            "health",
            return_value={"status": "available"},
        ) as mock_health:
            result = await async_client.health()
            mock_health.assert_called_once()
            assert result == {"status": "available"}

    @pytest.mark.asyncio
    async def test_get_stats(self, async_client):
        """测试异步获取统计信息"""
        with patch.object(
            async_client._sync_client,
            "get_stats",
            return_value={"numberOfDocuments": 100},
        ) as mock_stats:
            result = await async_client.get_stats()
            mock_stats.assert_called_once()
            assert result == {"numberOfDocuments": 100}

    @pytest.mark.asyncio
    async def test_delete_document(self, async_client):
        """测试异步删除文档"""
        with patch.object(
            async_client._sync_client,
            "delete_document",
            MagicMock(),
        ) as mock_delete:
            await async_client.delete_document("doc_1")
            mock_delete.assert_called_once_with("doc_1")

    @pytest.mark.asyncio
    async def test_delete_all_documents(self, async_client):
        """测试异步删除所有文档"""
        with patch.object(
            async_client._sync_client,
            "delete_all_documents",
            MagicMock(),
        ) as mock_delete:
            await async_client.delete_all_documents()
            mock_delete.assert_called_once()


class TestAsyncMeilisearchSDKClientEdgeCases:
    """异步客户端边界情况测试"""

    @pytest.mark.asyncio
    async def test_search_with_filters(self):
        """测试带过滤条件的搜索"""
        client = AsyncMeilisearchSDKClient()

        with patch.object(
            client._sync_client,
            "search",
            return_value={"hits": []},
        ) as mock_search:
            await client.search(
                "query",
                filter_expr="tenant_id = 'default'",
                limit=5,
                offset=10,
                sort=["created_at:desc"],
            )

            mock_search.assert_called_once_with(
                "query",
                filter_expr="tenant_id = 'default'",
                limit=5,
                offset=10,
                sort=["created_at:desc"],
                attributes_to_retrieve=None,
                show_ranking_score=True,
            )

    @pytest.mark.asyncio
    async def test_batch_add_documents(self):
        """测试批量添加文档"""
        client = AsyncMeilisearchSDKClient()
        docs = [{"id": str(i)} for i in range(250)]

        with patch.object(
            client._sync_client,
            "batch_add_documents",
            return_value={"processed": 250},
        ) as mock_batch:
            result = await client.batch_add_documents(docs, batch_size=100)

            mock_batch.assert_called_once_with(docs, "id", 100, wait=True)
            assert result == {"processed": 250}

    @pytest.mark.asyncio
    async def test_batch_delete_documents(self):
        """测试批量删除文档"""
        client = AsyncMeilisearchSDKClient()
        ids = [f"doc_{i}" for i in range(250)]

        with patch.object(
            client._sync_client,
            "batch_delete_documents",
            return_value={"processed": 250},
        ) as mock_batch:
            result = await client.batch_delete_documents(ids, batch_size=100)

            mock_batch.assert_called_once_with(ids, 100, wait=True)
            assert result == {"processed": 250}

    @pytest.mark.asyncio
    async def test_get_settings(self):
        """测试获取设置"""
        client = AsyncMeilisearchSDKClient()

        with patch.object(
            client._sync_client,
            "get_settings",
            return_value={"searchableAttributes": ["title"]},
        ) as mock_settings:
            result = await client.get_settings()

            mock_settings.assert_called_once()
            assert result == {"searchableAttributes": ["title"]}

    @pytest.mark.asyncio
    async def test_reset_settings(self):
        """测试重置设置"""
        client = AsyncMeilisearchSDKClient()

        with patch.object(
            client._sync_client,
            "reset_settings",
            MagicMock(),
        ) as mock_reset:
            await client.reset_settings()
            mock_reset.assert_called_once()
