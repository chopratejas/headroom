"""Tests for CCR endpoints in the proxy server.

These tests verify the /v1/retrieve endpoints work correctly.
"""

import json

import pytest

# Skip if fastapi not available
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from headroom.cache.compression_store import get_compression_store, reset_compression_store
from headroom.proxy.server import ProxyConfig, create_app


@pytest.fixture
def client():
    """Create test client with fresh compression store."""
    reset_compression_store()
    config = ProxyConfig(
        optimize=False,  # Disable optimization for simpler tests
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )
    app = create_app(config)
    with TestClient(app) as client:
        yield client
    reset_compression_store()


@pytest.fixture
def client_with_data(client):
    """Test client with pre-populated compression store."""
    store = get_compression_store()

    # Store some test data
    items = [{"id": i, "content": f"Item {i} about Python programming"} for i in range(100)]
    store.store(
        original=json.dumps(items),
        compressed=json.dumps(items[:10]),
        original_tokens=1000,
        compressed_tokens=100,
        original_item_count=100,
        compressed_item_count=10,
        tool_name="test_tool",
    )

    return client


class TestCCRRetrieveEndpoint:
    """Test the /v1/retrieve POST endpoint."""

    def test_retrieve_requires_hash(self, client):
        """Request without hash should return 400."""
        response = client.post("/v1/retrieve", json={})
        assert response.status_code == 400
        assert "hash required" in response.json()["detail"]

    def test_retrieve_nonexistent_hash(self, client):
        """Request with nonexistent hash should return 404."""
        response = client.post("/v1/retrieve", json={"hash": "nonexistent123"})
        assert response.status_code == 404
        assert "not found or expired" in response.json()["detail"]

    def test_retrieve_full_content(self, client):
        """Full retrieval returns original content."""
        store = get_compression_store()
        items = [{"id": i} for i in range(50)]
        hash_key = store.store(
            original=json.dumps(items),
            compressed="[]",
            original_item_count=50,
            compressed_item_count=0,
        )

        response = client.post("/v1/retrieve", json={"hash": hash_key})
        assert response.status_code == 200

        data = response.json()
        assert data["hash"] == hash_key
        assert data["original_item_count"] == 50
        assert "original_content" in data

        # Verify content is correct
        retrieved_items = json.loads(data["original_content"])
        assert len(retrieved_items) == 50
        assert retrieved_items[0]["id"] == 0

    def test_retrieve_with_search(self, client):
        """Search retrieval filters by query."""
        store = get_compression_store()
        items = [
            {"id": 1, "text": "Python programming language"},
            {"id": 2, "text": "JavaScript web development"},
            {"id": 3, "text": "Python data science"},
            {"id": 4, "text": "Java enterprise"},
        ]
        hash_key = store.store(
            original=json.dumps(items),
            compressed="[]",
            original_item_count=4,
            compressed_item_count=0,
        )

        response = client.post(
            "/v1/retrieve", json={"hash": hash_key, "query": "Python programming"}
        )
        assert response.status_code == 200

        data = response.json()
        assert data["hash"] == hash_key
        assert data["query"] == "Python programming"
        assert "results" in data
        assert data["count"] >= 1

    def test_retrieve_increments_count(self, client):
        """Each retrieval increments the retrieval count."""
        store = get_compression_store()
        hash_key = store.store(original="[]", compressed="[]")

        # First retrieval
        response1 = client.post("/v1/retrieve", json={"hash": hash_key})
        assert response1.status_code == 200
        count1 = response1.json()["retrieval_count"]

        # Second retrieval
        response2 = client.post("/v1/retrieve", json={"hash": hash_key})
        assert response2.status_code == 200
        count2 = response2.json()["retrieval_count"]

        assert count2 > count1


class TestCCRRetrieveGetEndpoint:
    """Test the /v1/retrieve/{hash_key} GET endpoint."""

    def test_get_retrieve_full(self, client):
        """GET retrieval returns full content."""
        store = get_compression_store()
        items = [{"id": i} for i in range(20)]
        hash_key = store.store(
            original=json.dumps(items),
            compressed="[]",
            original_item_count=20,
            compressed_item_count=0,
            tool_name="get_test_tool",
        )

        response = client.get(f"/v1/retrieve/{hash_key}")
        assert response.status_code == 200

        data = response.json()
        assert data["hash"] == hash_key
        assert data["original_item_count"] == 20
        assert data["tool_name"] == "get_test_tool"

    def test_get_retrieve_with_query(self, client):
        """GET retrieval with query parameter invokes search."""
        store = get_compression_store()
        # Create items with distinctive content
        items = [
            {"id": 1, "msg": "Python programming language tutorial for beginners"},
            {"id": 2, "msg": "JavaScript web development framework guide"},
            {"id": 3, "msg": "Python data science machine learning pandas"},
            {"id": 4, "msg": "Java enterprise application development"},
        ]
        hash_key = store.store(
            original=json.dumps(items),
            compressed="[]",
        )

        response = client.get(f"/v1/retrieve/{hash_key}?query=Python programming")
        assert response.status_code == 200

        data = response.json()
        assert data["query"] == "Python programming"
        # Response includes search results structure
        assert "results" in data
        assert "count" in data
        # Results should be a list (may be empty if BM25 threshold not met)
        assert isinstance(data["results"], list)

    def test_get_retrieve_nonexistent(self, client):
        """GET with nonexistent hash returns 404."""
        response = client.get("/v1/retrieve/nonexistent123")
        assert response.status_code == 404


class TestCCRStatsEndpoint:
    """Test the /v1/retrieve/stats endpoint."""

    def test_stats_empty_store(self, client):
        """Stats with empty store returns zeros."""
        response = client.get("/v1/retrieve/stats")
        assert response.status_code == 200

        data = response.json()
        assert "store" in data
        assert data["store"]["entry_count"] == 0
        assert "recent_retrievals" in data

    def test_stats_with_entries(self, client):
        """Stats reflect store contents."""
        store = get_compression_store()

        # Add some entries
        store.store(original="[1]", compressed="[]", original_tokens=100)
        store.store(original="[2]", compressed="[]", original_tokens=200)

        response = client.get("/v1/retrieve/stats")
        assert response.status_code == 200

        data = response.json()
        assert data["store"]["entry_count"] == 2
        assert data["store"]["total_original_tokens"] == 300

    def test_stats_tracks_retrievals(self, client):
        """Stats include recent retrieval events."""
        import json as json_module

        store = get_compression_store()

        # Use non-empty content so search actually logs
        content = json_module.dumps(
            [
                {"id": "1", "name": "test item", "value": 100},
                {"id": "2", "name": "another item", "value": 200},
            ]
        )
        hash_key = store.store(
            original=content,
            compressed=content,
            tool_name="stats_test_tool",
        )

        # Make some retrievals
        client.post("/v1/retrieve", json={"hash": hash_key})  # Full retrieval
        client.post("/v1/retrieve", json={"hash": hash_key, "query": "test"})  # Search retrieval

        response = client.get("/v1/retrieve/stats")
        assert response.status_code == 200

        data = response.json()
        assert data["store"]["total_retrievals"] >= 2
        assert len(data["recent_retrievals"]) >= 2

        # Verify we have both retrieval types (no double-logging of full)
        retrieval_types = [r["retrieval_type"] for r in data["recent_retrievals"]]
        assert "full" in retrieval_types
        assert "search" in retrieval_types


class TestCCRIntegration:
    """Integration tests for CCR with proxy."""

    def test_health_endpoint(self, client):
        """Health endpoint works."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_stats_endpoint(self, client):
        """Stats endpoint includes CCR-relevant info."""
        response = client.get("/stats")
        assert response.status_code == 200
        # Proxy stats endpoint is separate from CCR stats
        data = response.json()
        assert "requests" in data
        assert "tokens" in data


class TestCCREdgeCases:
    """Edge cases for CCR endpoints."""

    def test_retrieve_empty_content(self, client):
        """Retrieve works with empty content."""
        store = get_compression_store()
        hash_key = store.store(original="[]", compressed="[]")

        response = client.post("/v1/retrieve", json={"hash": hash_key})
        assert response.status_code == 200
        assert response.json()["original_content"] == "[]"

    def test_retrieve_large_content(self, client):
        """Retrieve works with large content."""
        store = get_compression_store()
        items = [{"id": i, "data": "x" * 100} for i in range(1000)]
        hash_key = store.store(
            original=json.dumps(items),
            compressed=json.dumps(items[:10]),
            original_item_count=1000,
        )

        response = client.post("/v1/retrieve", json={"hash": hash_key})
        assert response.status_code == 200

        data = response.json()
        assert data["original_item_count"] == 1000

    def test_search_no_matches(self, client):
        """Search with no matches returns empty results."""
        store = get_compression_store()
        items = [{"id": 1, "text": "hello world"}]
        hash_key = store.store(original=json.dumps(items), compressed="[]")

        response = client.post("/v1/retrieve", json={"hash": hash_key, "query": "xyznonexistent"})
        assert response.status_code == 200

        data = response.json()
        assert data["count"] == 0
        assert data["results"] == []

    def test_unicode_content(self, client):
        """Unicode content is handled correctly."""
        store = get_compression_store()
        items = [
            {"id": 1, "text": "日本語テキスト"},
            {"id": 2, "text": "Émoji 🎉 test"},
        ]
        hash_key = store.store(original=json.dumps(items, ensure_ascii=False), compressed="[]")

        response = client.post("/v1/retrieve", json={"hash": hash_key})
        assert response.status_code == 200

        data = response.json()
        retrieved = json.loads(data["original_content"])
        assert retrieved[0]["text"] == "日本語テキスト"
        assert "🎉" in retrieved[1]["text"]
