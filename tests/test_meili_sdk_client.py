"""Tests for MeilisearchSDKClient"""

import pytest
from unittest.mock import MagicMock, patch

from wrapper.src.utils.meili_sdk_client import MeilisearchSDKClient, MeilisearchSDKError


class TestMeilisearchSDKClientInitialization:
    """Test MeilisearchSDKClient initialization"""

    def test_basic_initialization(self):
        """Test basic initialization"""
        client = MeilisearchSDKClient(
            url="http://localhost:7700",
            api_key="test_key",
            index_name="test_index",
            timeout=60,
        )

        assert client._url == "http://localhost:7700"
        assert client._api_key == "test_key"
        assert client._index_name == "test_index"
        assert client._timeout == 60
        assert client._client is None

    def test_default_initialization(self):
        """Test initialization with default values"""
        client = MeilisearchSDKClient()

        assert client._url == "http://localhost:7700"
        assert client._api_key is None
        assert client._index_name == "memories"
        assert client._timeout == 30

    def test_url_trailing_slash_removed(self):
        """Test that trailing slash is removed from URL"""
        client = MeilisearchSDKClient(url="http://localhost:7700/")
        assert client._url == "http://localhost:7700"


class TestMeilisearchSDKClientConnect:
    """Test MeilisearchSDKClient connect method"""

    def test_connect_success(self):
        """Test successful connection"""
        client = MeilisearchSDKClient()

        with patch("wrapper.src.utils.meili_sdk_client.Client") as mock_client:
            mock_instance = MagicMock()
            mock_instance.health.return_value = {"status": "available"}
            mock_client.return_value = mock_instance

            client.connect()

            assert client._client is not None
            mock_client.assert_called_once_with(
                url="http://localhost:7700",
                api_key=None,
                timeout=30,
            )

    def test_connect_failure(self):
        """Test connection failure"""
        client = MeilisearchSDKClient()

        with patch("wrapper.src.utils.meili_sdk_client.Client") as mock_client:
            mock_instance = MagicMock()
            mock_instance.health.side_effect = Exception("Connection refused")
            mock_client.return_value = mock_instance

            with pytest.raises(MeilisearchSDKError, match="无法连接 Meilisearch"):
                client.connect()


class TestMeilisearchSDKClientClose:
    """Test MeilisearchSDKClient close method"""

    def test_close(self):
        """Test closing client"""
        client = MeilisearchSDKClient()

        with patch("wrapper.src.utils.meili_sdk_client.Client") as mock_client:
            mock_instance = MagicMock()
            mock_instance.health.return_value = {"status": "available"}
            mock_client.return_value = mock_instance

            client.connect()
            client.close()

            assert client._client is None


class TestMeilisearchSDKClientEnsureIndex:
    """Test MeilisearchSDKClient ensure_index method"""

    def test_ensure_index_creates_new(self):
        """Test creating new index"""
        client = MeilisearchSDKClient()

        with patch("wrapper.src.utils.meili_sdk_client.Client") as mock_client:
            mock_instance = MagicMock()
            mock_instance.health.return_value = {"status": "available"}
            mock_instance.create_index = MagicMock()
            mock_client.return_value = mock_instance

            client.connect()
            client.ensure_index()

            mock_instance.create_index.assert_called_once_with(
                uid="memories",
                options={"primaryKey": "id"},
            )

    def test_ensure_index_already_exists(self):
        """Test when index already exists - should not raise"""
        client = MeilisearchSDKClient()

        with patch("wrapper.src.utils.meili_sdk_client.Client") as mock_client:
            mock_instance = MagicMock()
            mock_instance.health.return_value = {"status": "available"}
            # Use a simple Exception with "already exists" in the message
            mock_instance.create_index.side_effect = Exception("index already exists")
            mock_client.return_value = mock_instance

            client.connect()
            # Should not raise
            client.ensure_index()


class TestMeilisearchSDKClientIDConversion:
    """Test ID conversion methods"""

    def test_to_meili_id(self):
        """Test SurrealDB ID to Meilisearch ID conversion"""
        client = MeilisearchSDKClient()

        assert client._to_meili_id("memory:abc123") == "memory_abc123"
        assert client._to_meili_id("tenant:user:123") == "tenant_user:123"

    def test_from_meili_id(self):
        """Test Meilisearch ID to SurrealDB ID conversion"""
        client = MeilisearchSDKClient()

        assert client._from_meili_id("memory_abc123") == "memory:abc123"
        assert client._from_meili_id("tenant_user:123") == "tenant:user:123"


class TestMeilisearchSDKClientAddDocuments:
    """Test MeilisearchSDKClient add_documents method"""

    def test_add_documents_empty(self):
        """Test adding empty document list"""
        client = MeilisearchSDKClient()

        result = client.add_documents([])

        assert result["status"] == "skipped"
        assert result["reason"] == "empty documents list"

    def test_add_documents_with_id_conversion(self):
        """Test adding documents with ID conversion"""
        client = MeilisearchSDKClient()
        docs = [{"id": "memory:abc123", "content": "test"}]

        with patch("wrapper.src.utils.meili_sdk_client.Client") as mock_client:
            mock_instance = MagicMock()
            mock_instance.health.return_value = {"status": "available"}

            mock_task = MagicMock()
            mock_task.task_uid = 123
            mock_index = MagicMock()
            mock_index.add_documents.return_value = mock_task
            mock_instance.index.return_value = mock_index
            mock_instance.wait_for_task = MagicMock()
            mock_client.return_value = mock_instance

            client.connect()
            result = client.add_documents(docs)

            assert result["taskUid"] == 123
            mock_index.add_documents.assert_called_once()
            call_args = mock_index.add_documents.call_args
            assert call_args.kwargs["documents"][0]["id"] == "memory_abc123"


class TestMeilisearchSDKClientSearch:
    """Test MeilisearchSDKClient search method"""

    def test_search_basic(self):
        """Test basic search"""
        client = MeilisearchSDKClient()

        with patch("wrapper.src.utils.meili_sdk_client.Client") as mock_client:
            mock_instance = MagicMock()
            mock_instance.health.return_value = {"status": "available"}

            mock_result = {
                "hits": [{"id": "memory_abc123", "content": "test"}],
                "estimatedTotalHits": 1,
                "totalHits": 1,
                "limit": 10,
                "offset": 0,
                "processingTimeMs": 5,
                "query": "test",
            }

            mock_index = MagicMock()
            mock_index.search.return_value = mock_result
            mock_instance.index.return_value = mock_index
            mock_client.return_value = mock_instance

            client.connect()
            result = client.search("test")

            assert result["query"] == "test"
            assert result["estimatedTotalHits"] == 1
            assert len(result["hits"]) == 1
            assert result["hits"][0]["id"] == "memory:abc123"

    def test_search_with_filter(self):
        """Test search with filter"""
        client = MeilisearchSDKClient()

        with patch("wrapper.src.utils.meili_sdk_client.Client") as mock_client:
            mock_instance = MagicMock()
            mock_instance.health.return_value = {"status": "available"}

            mock_result = {
                "hits": [],
                "estimatedTotalHits": 0,
                "totalHits": 0,
                "limit": 10,
                "offset": 0,
                "processingTimeMs": 2,
                "query": "test",
            }

            mock_index = MagicMock()
            mock_index.search.return_value = mock_result
            mock_instance.index.return_value = mock_index
            mock_client.return_value = mock_instance

            client.connect()
            client.search("test", filter_expr="tenant_id = 'default'")

            mock_index.search.assert_called_once()
            # Verify search was called with query and params containing filter
            call_args = mock_index.search.call_args
            assert call_args[0][0] == "test"
            assert "filter" in call_args[0][1]


class TestMeilisearchSDKClientHealth:
    """Test MeilisearchSDKClient health method"""

    def test_health_healthy(self):
        """Test health check when healthy"""
        client = MeilisearchSDKClient()

        with patch("wrapper.src.utils.meili_sdk_client.Client") as mock_client:
            mock_instance = MagicMock()
            mock_instance.health.return_value = {"status": "available"}
            mock_client.return_value = mock_instance

            client.connect()
            result = client.health()

            assert result["status"] == "available"

    def test_health_unhealthy(self):
        """Test health check when unhealthy"""
        client = MeilisearchSDKClient()

        with patch("wrapper.src.utils.meili_sdk_client.Client") as mock_client:
            mock_instance = MagicMock()
            mock_instance.health.return_value = {"status": "available"}
            mock_client.return_value = mock_instance

            client.connect()

        # Simulate health check failure by patching the internal client
        with patch.object(client, "_client") as mock_internal_client:
            mock_internal_client.health.side_effect = Exception("Connection refused")
            result = client.health()

            assert result["status"] == "unhealthy"
            assert "error" in result
