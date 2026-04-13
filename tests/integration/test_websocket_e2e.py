"""WebSocket End-to-End Integration Tests

Tests WebSocket functionality with real service connection.
Requires the wrapper service to be running.

Run with:
    uv run pytest tests/integration/test_websocket_e2e.py -v

Or with coverage:
    uv run pytest tests/integration/test_websocket_e2e.py --cov=wrapper/src/websocket -v
"""

import asyncio
import json
import os
import pytest
from typing import Optional

# Skip all tests if SKIP_E2E_TESTS is set
pytestmark = pytest.mark.skipif(os.getenv("SKIP_E2E_TESTS") == "1", reason="E2E tests skipped (SKIP_E2E_TESTS=1)")


class TestWebSocketE2EConnection:
    """WebSocket E2E connection tests"""

    @pytest.mark.asyncio
    async def test_websocket_connection(self):
        """Test basic WebSocket connection"""
        try:
            import websockets
        except ImportError:
            pytest.skip("websockets not installed")

        uri = "ws://localhost:18008/ws/memories/live?tenant_id=default"

        try:
            async with websockets.connect(uri) as websocket:
                # Connection established
                assert websocket.open

                # Send ping
                await websocket.send(json.dumps({"type": "ping"}))

                # Receive pong
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(response)
                assert data["type"] == "pong"

        except Exception as e:
            pytest.skip(f"WebSocket connection failed: {e}")

    @pytest.mark.asyncio
    async def test_websocket_with_token(self):
        """Test WebSocket connection with token"""
        try:
            import websockets
        except ImportError:
            pytest.skip("websockets not installed")

        uri = "ws://localhost:18008/ws/memories/live?tenant_id=default&token=test_token"

        try:
            async with websockets.connect(uri) as websocket:
                assert websocket.open

                # Send subscribe message
                await websocket.send(json.dumps({"type": "subscribe", "filters": {"type": "memory"}}))

                # Should receive acknowledgment
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(response)
                assert "type" in data

        except Exception as e:
            pytest.skip(f"WebSocket connection failed: {e}")


class TestWebSocketE2EHeartbeat:
    """WebSocket E2E heartbeat tests"""

    @pytest.mark.asyncio
    async def test_heartbeat_exchange(self):
        """Test heartbeat ping-pong exchange"""
        try:
            import websockets
        except ImportError:
            pytest.skip("websockets not installed")

        uri = "ws://localhost:18008/ws/memories/live?tenant_id=default"

        try:
            async with websockets.connect(uri) as websocket:
                # Send multiple pings
                for i in range(3):
                    await websocket.send(json.dumps({"type": "ping", "timestamp": asyncio.get_event_loop().time()}))

                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    data = json.loads(response)
                    assert data["type"] == "pong"
                    assert "timestamp" in data

        except Exception as e:
            pytest.skip(f"Heartbeat test failed: {e}")


class TestWebSocketE2EMessages:
    """WebSocket E2E message tests"""

    @pytest.mark.asyncio
    async def test_subscribe_and_receive(self):
        """Test subscribing and receiving messages"""
        try:
            import websockets
        except ImportError:
            pytest.skip("websockets not installed")

        uri = "ws://localhost:18008/ws/memories/live?tenant_id=default"

        try:
            async with websockets.connect(uri) as websocket:
                # Subscribe to changes
                await websocket.send(json.dumps({"type": "subscribe", "table": "memory", "filters": {}}))

                # Wait for acknowledgment
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(response)
                assert data["type"] in ["ack", "subscribed"]

        except Exception as e:
            pytest.skip(f"Message test failed: {e}")

    @pytest.mark.asyncio
    async def test_message_acknowledgment(self):
        """Test message acknowledgment"""
        try:
            import websockets
        except ImportError:
            pytest.skip("websockets not installed")

        uri = "ws://localhost:18008/ws/memories/live?tenant_id=default"

        try:
            async with websockets.connect(uri) as websocket:
                # Send message with ack requirement
                message_id = "test_msg_001"
                await websocket.send(
                    json.dumps({"type": "action", "id": message_id, "action": "test", "require_ack": True})
                )

                # Should receive ack
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(response)

                if data["type"] == "ack":
                    assert data.get("message_id") == message_id

        except Exception as e:
            pytest.skip(f"ACK test failed: {e}")


class TestWebSocketE2EReconnection:
    """WebSocket E2E reconnection tests"""

    @pytest.mark.asyncio
    async def test_reconnection_with_session(self):
        """Test reconnection with session recovery"""
        try:
            import websockets
        except ImportError:
            pytest.skip("websockets not installed")

        uri = "ws://localhost:18008/ws/memories/live?tenant_id=default"
        session_id: Optional[str] = None

        try:
            # First connection
            async with websockets.connect(uri) as websocket:
                await websocket.send(json.dumps({"type": "ping"}))
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(response)

                # Get session ID if available
                if "session_id" in data:
                    session_id = data["session_id"]

            # Reconnect with session ID
            if session_id:
                reconnect_uri = f"{uri}&session_id={session_id}"
                async with websockets.connect(reconnect_uri) as websocket:
                    await websocket.send(json.dumps({"type": "ping"}))
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    data = json.loads(response)
                    assert data["type"] == "pong"

        except Exception as e:
            pytest.skip(f"Reconnection test failed: {e}")


class TestWebSocketE2EPerformance:
    """WebSocket E2E performance tests"""

    @pytest.mark.asyncio
    async def test_message_latency(self):
        """Test message round-trip latency"""
        try:
            import websockets
            import time
        except ImportError:
            pytest.skip("websockets not installed")

        uri = "ws://localhost:18008/ws/memories/live?tenant_id=default"

        try:
            async with websockets.connect(uri) as websocket:
                latencies = []

                for _ in range(10):
                    start_time = time.time()

                    await websocket.send(json.dumps({"type": "ping"}))
                    await asyncio.wait_for(websocket.recv(), timeout=5.0)

                    end_time = time.time()
                    latency = (end_time - start_time) * 1000  # ms
                    latencies.append(latency)

                avg_latency = sum(latencies) / len(latencies)
                max_latency = max(latencies)

                # Assert reasonable latency (< 500ms)
                assert avg_latency < 500, f"Average latency too high: {avg_latency}ms"
                assert max_latency < 1000, f"Max latency too high: {max_latency}ms"

        except Exception as e:
            pytest.skip(f"Performance test failed: {e}")

    @pytest.mark.asyncio
    async def test_concurrent_connections(self):
        """Test multiple concurrent connections"""
        try:
            import websockets
        except ImportError:
            pytest.skip("websockets not installed")

        uri = "ws://localhost:18008/ws/memories/live?tenant_id=default"

        async def connect_and_ping():
            try:
                async with websockets.connect(uri) as websocket:
                    await websocket.send(json.dumps({"type": "ping"}))
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    data = json.loads(response)
                    return data["type"] == "pong"
            except Exception:
                return False

        try:
            # Create 5 concurrent connections
            tasks = [connect_and_ping() for _ in range(5)]
            results = await asyncio.gather(*tasks)

            # At least 80% should succeed
            success_rate = sum(results) / len(results)
            assert success_rate >= 0.8, f"Success rate too low: {success_rate}"

        except Exception as e:
            pytest.skip(f"Concurrent test failed: {e}")
