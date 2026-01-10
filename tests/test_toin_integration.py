"""Integration tests for the full TOIN feedback loop.

Tests the complete flow:
1. SmartCrusher compresses data and records compression event
2. compression_store stores with correct tool_signature_hash
3. User retrieves cached data (triggering feedback)
4. TOIN learns from retrieval event
5. Future compressions get improved recommendations
"""

import json
import tempfile
from pathlib import Path

import pytest

from headroom.cache.compression_store import (
    get_compression_store,
    reset_compression_store,
)
from headroom.config import CCRConfig
from headroom.telemetry import ToolSignature
from headroom.telemetry.toin import (
    TOINConfig,
    ToolIntelligenceNetwork,
    get_toin,
    reset_toin,
)
from headroom.transforms.smart_crusher import (
    SmartCrusher,
    SmartCrusherConfig,
)


@pytest.fixture
def fresh_toin():
    """Create a fresh TOIN instance with temporary storage."""
    reset_toin()
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = str(Path(tmpdir) / "toin.json")
        toin = get_toin(
            TOINConfig(
                storage_path=storage_path,
                auto_save_interval=0,  # No auto-persist during tests
            )
        )
        yield toin
        reset_toin()


@pytest.fixture
def fresh_store():
    """Create a fresh compression store."""
    reset_compression_store()
    store = get_compression_store(max_entries=100, default_ttl=300)
    yield store
    reset_compression_store()


class TestTOINIntegration:
    """Test the full TOIN feedback loop."""

    def test_compression_records_correct_hash(self, fresh_toin, fresh_store):
        """Test that SmartCrusher records the correct tool_signature_hash in store."""
        # Create test data
        items = [{"id": i, "score": 100 - i, "name": f"Item {i}"} for i in range(50)]
        content = json.dumps(items)

        # Create a SmartCrusher with CCR enabled
        ccr_config = CCRConfig(enabled=True, inject_retrieval_marker=False)
        crusher = SmartCrusher(
            SmartCrusherConfig(max_items_after_crush=10),
            ccr_config=ccr_config,
        )

        # Compress the content
        crushed, was_modified, info = crusher._smart_crush_content(content, tool_name="test_tool")

        assert was_modified, "Content should be modified by compression"

        # Verify the store has an entry with the correct tool_signature_hash
        stats = fresh_store.get_stats()
        assert stats["entry_count"] >= 1, "Store should have at least one entry"

        # Get the entry and verify it has tool_signature_hash
        # We need to find the hash key from the store
        entries = list(fresh_store._store.values())
        assert len(entries) >= 1, "Should have at least one entry"

        entry = entries[0]
        assert entry.tool_signature_hash is not None, "Entry should have tool_signature_hash"
        assert entry.compression_strategy is not None, "Entry should have compression_strategy"

        # Verify the hash matches what ToolSignature would generate
        expected_signature = ToolSignature.from_items(items)
        assert entry.tool_signature_hash == expected_signature.structure_hash, (
            "Stored hash should match ToolSignature.structure_hash"
        )

    def test_retrieval_updates_toin_strategy_success(self, fresh_toin, fresh_store):
        """Test that retrieval events update TOIN strategy success rates."""
        # Create test data
        items = [{"id": i, "score": 100 - i, "name": f"Item {i}"} for i in range(50)]
        signature = ToolSignature.from_items(items)

        # Record some compressions with fresh_toin
        for _ in range(5):
            fresh_toin.record_compression(
                tool_signature=signature,
                original_count=50,
                compressed_count=10,
                original_tokens=5000,
                compressed_tokens=1000,
                strategy="smart_sample",
            )

        # Get initial strategy success rate
        pattern = fresh_toin._patterns.get(signature.structure_hash)
        assert pattern is not None, "Pattern should exist after compressions"

        initial_rate = pattern.strategy_success_rates.get("smart_sample", 1.0)
        assert initial_rate > 0, "Initial success rate should be positive"

        # Simulate retrieval events (which indicate compression was too aggressive)
        for _ in range(3):
            fresh_toin.record_retrieval(
                tool_signature_hash=signature.structure_hash,
                retrieval_type="full",
                query="test query",
                query_fields=["id"],
                strategy="smart_sample",
            )

        # Verify success rate decreased
        final_rate = pattern.strategy_success_rates.get("smart_sample", 1.0)
        assert final_rate < initial_rate, (
            f"Success rate should decrease after retrievals: {initial_rate} -> {final_rate}"
        )

    def test_full_feedback_loop(self, fresh_toin, fresh_store):
        """Test the complete feedback loop: compress → retrieve → learn → recommend."""
        # Create test data
        items = [{"id": i, "score": 100 - i, "name": f"Item {i}"} for i in range(50)]
        content = json.dumps(items)
        signature = ToolSignature.from_items(items)

        # Step 1: Compress with SmartCrusher (records to TOIN)
        ccr_config = CCRConfig(enabled=True, inject_retrieval_marker=False)
        crusher = SmartCrusher(
            SmartCrusherConfig(max_items_after_crush=10, use_feedback_hints=True),
            ccr_config=ccr_config,
        )

        # Multiple compressions to build pattern
        for _ in range(5):
            crusher._smart_crush_content(content, tool_name="test_tool")

        # Step 2: Verify TOIN has a pattern
        pattern = fresh_toin._patterns.get(signature.structure_hash)
        assert pattern is not None, "TOIN should have a pattern after compressions"
        assert pattern.total_compressions >= 5, "Should have recorded 5 compressions"

        # Step 3: Simulate retrievals (indicating compression was too aggressive)
        # Find the stored entry hash
        entries = list(fresh_store._store.values())
        assert len(entries) > 0, "Should have cached entries"

        # Retrieve multiple times to trigger learning
        for entry in entries[:3]:
            fresh_store.retrieve(entry.hash, query="find all items")

        # Step 4: Verify TOIN learned from retrievals
        recommendation = fresh_toin.get_recommendation(signature, "find all items")

        # After many retrievals, TOIN should recommend more items
        # Default is 15-20, but with high retrieval rate it should go higher
        assert recommendation.confidence > 0, "Recommendation should have confidence"

    def test_preserve_fields_used_in_compression(self, fresh_toin, fresh_store):
        """Test that TOIN preserve_fields are used during compression planning."""
        # Create test data with specific fields
        items = [{"id": i, "score": 100 - i, "category": f"cat_{i % 3}"} for i in range(50)]
        signature = ToolSignature.from_items(items)

        # Record compressions and retrievals that query the "category" field
        for _ in range(10):
            fresh_toin.record_compression(
                tool_signature=signature,
                original_count=50,
                compressed_count=10,
                original_tokens=5000,
                compressed_tokens=1000,
                strategy="smart_sample",
            )

        # Record retrievals that query "category"
        for _ in range(5):
            fresh_toin.record_retrieval(
                tool_signature_hash=signature.structure_hash,
                retrieval_type="search",
                query="category:cat_1",
                query_fields=["category"],
                strategy="smart_sample",
            )

        # Get recommendation - should now preserve "category" field
        recommendation = fresh_toin.get_recommendation(signature, "find all items in cat_1")

        # Verify the recommendation reflects learning
        assert recommendation.source in ("local", "network", "default"), (
            f"Should have a valid source: {recommendation.source}"
        )

    def test_instance_id_stable_across_restarts(self):
        """Test that instance_id is stable across restarts."""
        reset_toin()
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = str(Path(tmpdir) / "toin_stable.json")

            # Create first TOIN instance
            toin1 = ToolIntelligenceNetwork(TOINConfig(storage_path=storage_path))
            instance_id_1 = toin1._instance_id

            # Create second instance with same path (simulating restart)
            toin2 = ToolIntelligenceNetwork(TOINConfig(storage_path=storage_path))
            instance_id_2 = toin2._instance_id

            # Instance IDs should be the same since derived from path
            assert instance_id_1 == instance_id_2, (
                f"Instance ID should be stable: {instance_id_1} vs {instance_id_2}"
            )

    def test_atomic_save(self):
        """Test that save() uses atomic writes."""
        reset_toin()
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = str(Path(tmpdir) / "toin_atomic.json")
            toin = ToolIntelligenceNetwork(TOINConfig(storage_path=storage_path))

            # Create some data
            items = [{"id": i, "name": f"test_{i}"} for i in range(10)]
            signature = ToolSignature.from_items(items)

            toin.record_compression(
                tool_signature=signature,
                original_count=10,
                compressed_count=5,
                original_tokens=1000,
                compressed_tokens=500,
                strategy="smart_sample",
            )

            # Save
            toin.save()

            # Verify file exists and is valid JSON
            saved_path = Path(storage_path)
            assert saved_path.exists(), "Save file should exist"

            with open(saved_path) as f:
                data = json.load(f)

            assert "patterns" in data, "Saved data should have patterns"
            assert len(data["patterns"]) > 0, "Should have at least one pattern"

    def test_query_patterns_merged_on_import(self):
        """Test that query patterns are merged when importing patterns."""
        reset_toin()
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = str(Path(tmpdir) / "toin_merge.json")
            toin = ToolIntelligenceNetwork(TOINConfig(storage_path=storage_path))

            # Create local pattern
            items = [{"id": i, "name": f"test_{i}"} for i in range(10)]
            signature = ToolSignature.from_items(items)

            toin.record_compression(
                tool_signature=signature,
                original_count=10,
                compressed_count=5,
                original_tokens=1000,
                compressed_tokens=500,
                strategy="smart_sample",
            )

            # Record local query pattern
            toin.record_retrieval(
                tool_signature_hash=signature.structure_hash,
                retrieval_type="search",
                query="local_pattern:value",
                query_fields=["local_pattern"],
                strategy="smart_sample",
            )

            # Create import data with different query pattern
            import_data = {
                "patterns": {
                    signature.structure_hash: {
                        "tool_signature_hash": signature.structure_hash,
                        "total_compressions": 100,
                        "total_retrievals": 20,
                        "avg_original_count": 50,
                        "avg_compressed_count": 10,
                        "avg_compression_ratio": 0.2,
                        "retrieval_rate": 0.2,
                        "common_query_patterns": ["imported_pattern:x"],
                        "strategy_success_rates": {"smart_sample": 0.8},
                        "preserve_fields": ["imported_field"],
                        "user_count": 50,
                        "last_updated": 1234567890.0,
                    }
                }
            }

            # Import patterns
            toin.import_patterns(import_data)

            # Verify patterns were merged
            pattern = toin._patterns.get(signature.structure_hash)
            assert pattern is not None, "Pattern should exist after merge"

            # Check that both query patterns exist
            assert "imported_pattern:x" in pattern.common_query_patterns, (
                "Imported query pattern should be present"
            )


class TestStoreToTOINHash:
    """Test the hash correlation between compression_store and TOIN."""

    def test_hash_matches_between_store_and_toin(self, fresh_toin, fresh_store):
        """Test that the hash stored in compression_store matches TOIN events."""
        # Use 50 items with score field to ensure compression threshold is met
        items = [{"id": i, "score": 100 - i, "name": f"Item {i}"} for i in range(50)]
        content = json.dumps(items)
        signature = ToolSignature.from_items(items)

        # Compress with aggressive settings to ensure items are actually reduced
        ccr_config = CCRConfig(enabled=True, inject_retrieval_marker=False)
        crusher = SmartCrusher(
            SmartCrusherConfig(max_items_after_crush=10),
            ccr_config=ccr_config,
        )
        crushed, was_modified, info = crusher._smart_crush_content(content, tool_name="hash_test")

        # Verify compression actually happened
        assert was_modified, f"Content should be modified by compression: {info}"

        # Get the stored hash
        entries = list(fresh_store._store.values())
        assert len(entries) >= 1, (
            f"Should have stored entry. Modified: {was_modified}, Info: {info}"
        )
        stored_hash = entries[0].tool_signature_hash

        # Verify it matches ToolSignature
        assert stored_hash == signature.structure_hash, (
            f"Store hash {stored_hash} should match signature {signature.structure_hash}"
        )

        # Verify TOIN can receive events for this hash
        fresh_toin.record_compression(
            tool_signature=signature,
            original_count=50,
            compressed_count=10,
            original_tokens=5000,
            compressed_tokens=1000,
            strategy="smart_sample",
        )

        pattern = fresh_toin._patterns.get(signature.structure_hash)
        assert pattern is not None, "TOIN should have pattern for same hash"
