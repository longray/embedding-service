"""Code Search Index 配置测试

测试范围：
- 代码搜索索引配置
- 代码术语词典
- 代码标识符搜索
- 代码搜索功能

运行方式：
    uv run pytest tests/test_code_search_index.py -v
"""

import pytest
from unittest.mock import MagicMock, patch

from wrapper.src.utils.meili_sdk_client import (
    MeilisearchSDKClient,
    AsyncMeilisearchSDKClient,
)


class TestCodeSearchIndexConfig:
    """代码搜索索引配置测试"""

    def test_code_search_settings_defined(self):
        """测试代码搜索索引配置已定义"""
        assert hasattr(MeilisearchSDKClient, "CODE_SEARCH_INDEX_SETTINGS")
        settings = MeilisearchSDKClient.CODE_SEARCH_INDEX_SETTINGS

        # 验证关键配置项
        assert "searchableAttributes" in settings
        assert "filterableAttributes" in settings
        assert "sortableAttributes" in settings
        assert "nonSeparatorTokens" in settings
        assert "typoTolerance" in settings
        assert "dictionary" in settings

    def test_code_search_searchable_attributes(self):
        """测试可搜索属性"""
        settings = MeilisearchSDKClient.CODE_SEARCH_INDEX_SETTINGS
        attrs = settings["searchableAttributes"]

        assert "file_path" in attrs
        assert "code_content" in attrs
        assert "code_symbols" in attrs
        assert "function_names" in attrs
        assert "class_names" in attrs
        assert "variable_names" in attrs

    def test_code_search_filterable_attributes(self):
        """测试可过滤属性"""
        settings = MeilisearchSDKClient.CODE_SEARCH_INDEX_SETTINGS
        attrs = settings["filterableAttributes"]

        assert "code_language" in attrs
        assert "file_path" in attrs
        assert "code_complexity" in attrs
        assert "function_count" in attrs
        assert "class_count" in attrs
        assert "is_test_file" in attrs
        assert "is_config_file" in attrs

    def test_code_search_non_separator_tokens(self):
        """测试非分隔符标记"""
        settings = MeilisearchSDKClient.CODE_SEARCH_INDEX_SETTINGS
        tokens = settings["nonSeparatorTokens"]

        assert "." in tokens
        assert "-" in tokens
        assert "_" in tokens
        assert "::" in tokens  # C++ 作用域
        assert "->" in tokens  # 指针访问

    def test_code_search_dictionary(self):
        """测试代码术语词典"""
        settings = MeilisearchSDKClient.CODE_SEARCH_INDEX_SETTINGS
        dictionary = settings["dictionary"]

        # 验证编程语言
        assert "python" in dictionary
        assert "javascript" in dictionary
        assert "typescript" in dictionary
        assert "rust" in dictionary
        assert "golang" in dictionary

        # 验证框架
        assert "react" in dictionary
        assert "django" in dictionary
        assert "fastapi" in dictionary

        # 验证代码术语
        assert "class" in dictionary
        assert "function" in dictionary
        assert "async" in dictionary
        assert "await" in dictionary

        # 验证至少有 100 个词
        assert len(dictionary) >= 100

    def test_code_search_typo_tolerance(self):
        """测试拼写容错配置"""
        settings = MeilisearchSDKClient.CODE_SEARCH_INDEX_SETTINGS
        typo = settings["typoTolerance"]

        assert typo["enabled"] is True
        assert "file_path" in typo["disableOnAttributes"]
        assert "function_names" in typo["disableOnAttributes"]


class TestCodeSearchIndexMethods:
    """代码搜索索引方法测试"""

    @pytest.fixture
    def client(self):
        """创建 MeilisearchSDKClient 实例"""
        return MeilisearchSDKClient(
            url="http://localhost:7700",
            api_key="test_key",
        )

    def test_configure_code_search_index(self, client):
        """测试配置代码搜索索引"""
        # Mock 客户端连接
        client._client = MagicMock()

        with (
            patch.object(client._client, "create_index") as mock_create,
            patch.object(client._client, "index") as mock_index,
        ):
            mock_create.return_value = None
            mock_index.return_value.update_settings = MagicMock()

            client.configure_code_search_index("code_search_index")

            mock_create.assert_called_once()
            mock_index.assert_called_once()

    def test_search_code(self, client):
        """测试代码搜索"""
        expected_result = {
            "hits": [{"file_path": "test.py", "code_content": "def test(): pass"}],
            "estimatedTotalHits": 1,
        }

        # Mock 客户端连接
        client._client = MagicMock()

        with patch.object(client._client, "index") as mock_index:
            mock_index.return_value.search = MagicMock(return_value=expected_result)

            result = client.search_code("def test")

            assert result["estimatedTotalHits"] == 1
            mock_index.assert_called_once_with("code_search_index")

    def test_search_code_with_filters(self, client):
        """测试带过滤条件的代码搜索"""
        expected_result = {"hits": [], "estimatedTotalHits": 0}

        # Mock 客户端连接
        client._client = MagicMock()

        with patch.object(client._client, "index") as mock_index:
            mock_index.return_value.search = MagicMock(return_value=expected_result)

            result = client.search_code(
                "class",
                language="python",
                file_path="src/",
                limit=5,
            )

            # 验证搜索被调用
            mock_index.return_value.search.assert_called_once()
            # 验证返回结果包含预期的键
            assert "hits" in result
            assert "estimatedTotalHits" in result


class TestAsyncCodeSearchIndex:
    """异步代码搜索索引测试"""

    @pytest.fixture
    def async_client(self):
        """创建 AsyncMeilisearchSDKClient 实例"""
        return AsyncMeilisearchSDKClient(
            url="http://localhost:7700",
            api_key="test_key",
        )

    @pytest.mark.asyncio
    async def test_async_configure_code_search_index(self, async_client):
        """测试异步配置代码搜索索引"""
        with patch.object(
            async_client._sync_client,
            "configure_code_search_index",
            MagicMock(),
        ) as mock_configure:
            await async_client.configure_code_search_index("code_search_index")
            mock_configure.assert_called_once_with("code_search_index")

    @pytest.mark.asyncio
    async def test_async_search_code(self, async_client):
        """测试异步代码搜索"""
        expected_result = {
            "hits": [{"file_path": "test.py"}],
            "estimatedTotalHits": 1,
        }

        with patch.object(
            async_client._sync_client,
            "search_code",
            return_value=expected_result,
        ) as mock_search:
            result = await async_client.search_code("def test")

            mock_search.assert_called_once_with("def test", language=None, file_path=None, limit=10, offset=0)
            assert result["estimatedTotalHits"] == 1

    @pytest.mark.asyncio
    async def test_async_search_code_with_filters(self, async_client):
        """测试异步带过滤条件的代码搜索"""
        with patch.object(
            async_client._sync_client,
            "search_code",
            return_value={"hits": []},
        ) as mock_search:
            await async_client.search_code(
                "class",
                language="python",
                file_path="src/",
                limit=5,
            )

            mock_search.assert_called_once_with(
                "class",
                language="python",
                file_path="src/",
                limit=5,
                offset=0,
            )
