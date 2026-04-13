"""Tests for ChangeFeed client"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from wrapper.src.utils.changefeed_client import ChangeFeedClient


class TestChangeFeedClientInitialization:
    """Test ChangeFeedClient initialization"""

    def test_basic_initialization(self):
        """Test basic initialization"""
        client = ChangeFeedClient(
            url="ws://localhost:18002",
            namespace="test_ns",
            database="test_db",
            username="root",
            password="root",
        )

        assert client._url == "ws://localhost:18002"
        assert client._namespace == "test_ns"
        assert client._database == "test_db"
        assert client._username == "root"
        assert client._password == "root"
        assert client._db is None
        assert client._listening is False

    def test_default_initialization(self):
        """Test initialization with default values"""
        client = ChangeFeedClient()

        assert client._url == "ws://localhost:18002"
        assert client._namespace == "memory_ns"
        assert client._database == "memory_db"
        assert client._username == "root"
        assert client._password == "root"


class TestChangeFeedClientConnect:
    """Test ChangeFeedClient connect method"""

    @pytest.mark.asyncio
    async def test_connect_success(self):
        """Test successful connection"""
        client = ChangeFeedClient()

        with patch("wrapper.src.utils.changefeed_client.Surreal") as mock_surreal:
            mock_instance = AsyncMock()
            mock_instance.connect = AsyncMock()
            mock_instance.signin = AsyncMock()
            mock_instance.use = AsyncMock()
            mock_surreal.return_value = mock_instance

            await client.connect()

            assert client._db is not None
            mock_instance.connect.assert_called_once()
            mock_instance.signin.assert_called_once()
            mock_instance.use.assert_called_once_with("memory_ns", "memory_db")

    @pytest.mark.asyncio
    async def test_close(self):
        """Test closing connection"""
        client = ChangeFeedClient()

        with patch("wrapper.src.utils.changefeed_client.Surreal") as mock_surreal:
            mock_instance = AsyncMock()
            mock_instance.connect = AsyncMock()
            mock_instance.signin = AsyncMock()
            mock_instance.use = AsyncMock()
            mock_instance.close = AsyncMock()
            mock_surreal.return_value = mock_instance

            await client.connect()
            await client.close()

            mock_instance.close.assert_called_once()
            assert client._db is None


class TestChangeFeedClientSubscribe:
    """Test ChangeFeedClient subscribe methods"""

    @pytest.mark.asyncio
    async def test_subscribe_to_changes(self):
        """Test subscribing to changes"""
        client = ChangeFeedClient()

        with patch("wrapper.src.utils.changefeed_client.Surreal") as mock_surreal:
            mock_instance = AsyncMock()
            mock_instance.connect = AsyncMock()
            mock_instance.signin = AsyncMock()
            mock_instance.use = AsyncMock()
            mock_instance.query = AsyncMock(return_value="query_uuid_123")
            mock_surreal.return_value = mock_instance

            await client.connect()

            async def callback(data):
                pass

            query_id = await client.subscribe_to_changes("atom", callback)

            assert query_id == "query_uuid_123"
            assert "query_uuid_123" in client._subscriptions
            mock_instance.query.assert_called_once_with("LIVE SELECT * FROM atom")

    @pytest.mark.asyncio
    async def test_subscribe_not_connected(self):
        """Test subscribing when not connected"""
        client = ChangeFeedClient()

        async def callback(data):
            pass

        with pytest.raises(RuntimeError, match="Not connected to SurrealDB"):
            await client.subscribe_to_changes("atom", callback)

    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        """Test unsubscribing from changes"""
        client = ChangeFeedClient()

        with patch("wrapper.src.utils.changefeed_client.Surreal") as mock_surreal:
            mock_instance = AsyncMock()
            mock_instance.connect = AsyncMock()
            mock_instance.signin = AsyncMock()
            mock_instance.use = AsyncMock()
            mock_instance.query = AsyncMock()
            mock_surreal.return_value = mock_instance

            await client.connect()
            await client.unsubscribe("query_uuid_123")

            mock_instance.query.assert_called_with("KILL query_uuid_123")


class TestChangeFeedClientListening:
    """Test ChangeFeedClient listening methods"""

    @pytest.mark.asyncio
    async def test_start_listening(self):
        """Test starting to listen"""
        client = ChangeFeedClient()

        with patch("wrapper.src.utils.changefeed_client.Surreal") as mock_surreal:
            mock_instance = AsyncMock()
            mock_instance.connect = AsyncMock()
            mock_instance.signin = AsyncMock()
            mock_instance.use = AsyncMock()
            mock_surreal.return_value = mock_instance

            await client.connect()

            # Start listening in background
            task = asyncio.create_task(client.start_listening())
            await asyncio.sleep(0.1)

            assert client._listening is True

            client.stop_listening()
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass

    def test_stop_listening(self):
        """Test stopping listening"""
        client = ChangeFeedClient()
        client._listening = True

        client.stop_listening()

        assert client._listening is False


class TestChangeFeedClientVerification:
    """Test ChangeFeedClient verification methods"""

    @pytest.mark.asyncio
    async def test_get_changefeed_info(self):
        """Test getting ChangeFeed info"""
        client = ChangeFeedClient()

        with patch("wrapper.src.utils.changefeed_client.Surreal") as mock_surreal:
            mock_instance = AsyncMock()
            mock_instance.connect = AsyncMock()
            mock_instance.signin = AsyncMock()
            mock_instance.use = AsyncMock()
            mock_instance.query = AsyncMock(return_value={"changefeed": True})
            mock_surreal.return_value = mock_instance

            await client.connect()
            result = await client.get_changefeed_info("atom")

            assert result == {"changefeed": True}
            mock_instance.query.assert_called_with("INFO FOR TABLE atom")

    @pytest.mark.asyncio
    async def test_verify_changefeed_enabled_true(self):
        """Test verifying ChangeFeed is enabled"""
        client = ChangeFeedClient()

        with patch("wrapper.src.utils.changefeed_client.Surreal") as mock_surreal:
            mock_instance = AsyncMock()
            mock_instance.connect = AsyncMock()
            mock_instance.signin = AsyncMock()
            mock_instance.use = AsyncMock()
            mock_instance.query = AsyncMock(return_value={"changefeed": "enabled"})
            mock_surreal.return_value = mock_instance

            await client.connect()
            result = await client.verify_changefeed_enabled("atom")

            assert result is True

    @pytest.mark.asyncio
    async def test_verify_changefeed_enabled_false(self):
        """Test verifying ChangeFeed is disabled"""
        client = ChangeFeedClient()

        with patch("wrapper.src.utils.changefeed_client.Surreal") as mock_surreal:
            mock_instance = AsyncMock()
            mock_instance.connect = AsyncMock()
            mock_instance.signin = AsyncMock()
            mock_instance.use = AsyncMock()
            mock_instance.query = AsyncMock(return_value={"fields": []})
            mock_surreal.return_value = mock_instance

            await client.connect()
            result = await client.verify_changefeed_enabled("atom")

            assert result is False


import asyncio
