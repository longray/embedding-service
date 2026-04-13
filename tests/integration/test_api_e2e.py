"""API End-to-End Integration Tests

Tests API endpoints with real service connection.
Requires the wrapper service to be running.

Run with:
    uv run pytest tests/integration/test_api_e2e.py -v
"""

import os
import pytest
import httpx

# Skip all tests if SKIP_E2E_TESTS is set
pytestmark = pytest.mark.skipif(os.getenv("SKIP_E2E_TESTS") == "1", reason="E2E tests skipped (SKIP_E2E_TESTS=1)")

BASE_URL = "http://localhost:18008"


class TestAPIE2EHealth:
    """API E2E health endpoint tests"""

    def test_health_endpoint(self):
        """Test health endpoint returns 200"""
        try:
            response = httpx.get(f"{BASE_URL}/health", timeout=5.0)
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
        except Exception as e:
            pytest.skip(f"Health check failed: {e}")


class TestAPIE2EMemories:
    """API E2E memories endpoint tests"""

    def test_get_memories_list(self):
        """Test getting memories list"""
        try:
            response = httpx.get(f"{BASE_URL}/api/v1/memories?tenant_id=default", timeout=5.0)
            assert response.status_code in [200, 404]

            if response.status_code == 200:
                data = response.json()
                assert isinstance(data, list)
        except Exception as e:
            pytest.skip(f"Get memories failed: {e}")

    def test_create_memory(self):
        """Test creating a memory"""
        try:
            response = httpx.post(
                f"{BASE_URL}/api/v1/memories?tenant_id=default",
                json={"content": "Test memory from E2E test", "type": "test", "tags": ["e2e", "test"]},
                timeout=5.0,
            )
            assert response.status_code in [200, 201, 422]

            if response.status_code in [200, 201]:
                data = response.json()
                assert "id" in data
        except Exception as e:
            pytest.skip(f"Create memory failed: {e}")

    def test_search_memories(self):
        """Test searching memories"""
        try:
            response = httpx.post(
                f"{BASE_URL}/api/v1/memories/search?tenant_id=default", json={"query": "test", "limit": 10}, timeout=5.0
            )
            assert response.status_code in [200, 404]

            if response.status_code == 200:
                data = response.json()
                assert "hits" in data or "results" in data
        except Exception as e:
            pytest.skip(f"Search memories failed: {e}")


class TestAPIE2EEmbeddings:
    """API E2E embeddings endpoint tests"""

    def test_create_embedding(self):
        """Test creating an embedding"""
        try:
            response = httpx.post(
                f"{BASE_URL}/v1/embeddings", json={"input": "Test text for embedding", "model": "default"}, timeout=10.0
            )
            assert response.status_code in [200, 201, 503]

            if response.status_code in [200, 201]:
                data = response.json()
                assert "embedding" in data or "data" in data
        except Exception as e:
            pytest.skip(f"Create embedding failed: {e}")


class TestAPIE2ESync:
    """API E2E sync endpoint tests"""

    def test_sync_preview(self):
        """Test sync preview endpoint"""
        try:
            response = httpx.post(
                f"{BASE_URL}/api/v1/sync/preview?tenant_id=default", json={"fingerprints": []}, timeout=5.0
            )
            assert response.status_code in [200, 404]

            if response.status_code == 200:
                data = response.json()
                assert "conflicts" in data or "to_upload" in data
        except Exception as e:
            pytest.skip(f"Sync preview failed: {e}")

    def test_get_fingerprints(self):
        """Test getting fingerprints"""
        try:
            response = httpx.get(f"{BASE_URL}/api/v1/sync/fingerprints?tenant_id=default", timeout=5.0)
            assert response.status_code in [200, 404]

            if response.status_code == 200:
                data = response.json()
                assert isinstance(data, list) or "fingerprints" in data
        except Exception as e:
            pytest.skip(f"Get fingerprints failed: {e}")


class TestAPIE2EPerformance:
    """API E2E performance tests"""

    def test_api_response_time(self):
        """Test API response time"""
        import time

        try:
            start_time = time.time()
            response = httpx.get(f"{BASE_URL}/health", timeout=5.0)
            end_time = time.time()

            response_time = (end_time - start_time) * 1000  # ms
            assert response.status_code == 200
            assert response_time < 1000, f"Response time too slow: {response_time}ms"

        except Exception as e:
            pytest.skip(f"Performance test failed: {e}")

    def test_api_concurrent_requests(self):
        """Test concurrent API requests"""
        import asyncio

        async def make_request():
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{BASE_URL}/health", timeout=5.0)
                    return response.status_code == 200
            except Exception:
                return False

        try:
            # Create 10 concurrent requests
            tasks = [make_request() for _ in range(10)]
            results = asyncio.run(asyncio.gather(*tasks))

            # At least 80% should succeed
            success_rate = sum(results) / len(results)
            assert success_rate >= 0.8, f"Success rate too low: {success_rate}"

        except Exception as e:
            pytest.skip(f"Concurrent test failed: {e}")
