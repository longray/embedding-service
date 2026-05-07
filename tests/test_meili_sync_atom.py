"""Tests for MeiliSyncMixin._build_meili_doc with Atom support."""

import pytest
from wrapper.src.utils.memory_manager.meili_sync import MeiliSyncMixin


class TestMeiliSyncMixin:
    """Test MeiliSyncMixin document building."""

    @pytest.fixture
    def mixin(self):
        """Create a MeiliSyncMixin instance for testing."""
        # Create a minimal mixin instance
        mixin = MeiliSyncMixin()
        mixin._default_tenant_id = "default"
        return mixin

    def test_build_entity_meili_doc(self, mixin):
        """Test building entity (memory) document."""
        record_id = "memory:abc123"
        memory_data = {
            "content": "Test content",
            "type": "general",
            "tags": ["test"],
            "abstract": "Test abstract",
        }
        tenant_id = "default"

        doc = mixin._build_meili_doc(record_id, memory_data, tenant_id, doc_type="entity")

        assert doc["id"] == "memory_abc123"
        assert doc["surreal_id"] == "memory:abc123"
        assert doc["doc_type"] == "entity"
        assert doc["content"] == "Test content"
        assert doc["type"] == "general"
        assert doc["tags"] == ["test"]
        assert doc["abstract"] == "Test abstract"

    def test_build_atom_meili_doc(self, mixin):
        """Test building atom document."""
        record_id = "atom:xyz789"
        atom_data = {
            "name": "Test Atom",
            "content": "Atom content",
            "type": "section",
            "tags": ["atom-tag"],
            "entity_id": "entity:parent123",
            "local_id": "01KTEST01",
            "heading_level": 2,
        }
        tenant_id = "default"

        doc = mixin._build_meili_doc(record_id, atom_data, tenant_id, doc_type="atom")

        assert doc["id"] == "atom_xyz789"
        assert doc["surreal_id"] == "atom:xyz789"
        assert doc["doc_type"] == "atom"
        assert doc["name"] == "Test Atom"
        assert doc["content"] == "Atom content"
        assert doc["atom_type"] == "section"
        assert doc["type"] == "section"  # 兼容字段
        assert doc["entity_id"] == "entity:parent123"
        assert doc["local_id"] == "01KTEST01"
        assert doc["heading_level"] == 2

    def test_build_meili_doc_default_to_entity(self, mixin):
        """Test that _build_meili_doc defaults to entity type."""
        record_id = "memory:test123"
        data = {"content": "Test"}
        tenant_id = "default"

        # Should default to entity when doc_type not specified
        doc = mixin._build_meili_doc(record_id, data, tenant_id)

        assert doc["doc_type"] == "entity"

    def test_id_conversion_for_atom(self, mixin):
        """Test that atom IDs are properly converted."""
        record_id = "atom:my-atom-id-123"
        atom_data = {"name": "Test", "content": "Content"}

        doc = mixin._build_meili_doc(record_id, atom_data, "default", doc_type="atom")

        # ID should have colon replaced with underscore
        assert doc["id"] == "atom_my-atom-id-123"
        # surreal_id should preserve original format
        assert doc["surreal_id"] == "atom:my-atom-id-123"

    def test_atom_doc_has_required_fields(self, mixin):
        """Test that atom document has all required Meilisearch fields."""
        record_id = "atom:test123"
        atom_data = {
            "name": "Test Name",
            "content": "Test Content",
            "type": "section",
            "tags": ["tag1"],
            "entity_id": "entity:parent",
            "local_id": "LOCAL001",
            "heading_level": 3,
        }

        doc = mixin._build_meili_doc(record_id, atom_data, "default", doc_type="atom")

        # Check all required fields from DEFAULT_INDEX_SETTINGS
        required_fields = [
            "id", "surreal_id", "doc_type", "name", "content", "content_zh",
            "tenant_id", "atom_type", "type", "tags", "entity_id", "local_id",
            "heading_level", "created_at",
        ]
        for field in required_fields:
            assert field in doc, f"Missing required field: {field}"

    def test_entity_doc_has_code_analysis_fields(self, mixin):
        """Test that entity document includes code analysis fields."""
        record_id = "memory:test123"
        memory_data = {
            "content": "Test",
            "metadata": {
                "code_analysis": {
                    "language": "python",
                    "complexity": {"cyclomatic_complexity": 5, "function_count": 3},
                    "analyzer": "tree-sitter",
                    "exports": ["func1", "func2"],
                },
                "code_symbols": ["Symbol1", "Symbol2"],
            },
        }

        doc = mixin._build_meili_doc(record_id, memory_data, "default", doc_type="entity")

        assert doc["code_language"] == "python"
        assert doc["code_complexity"] == 5
        assert doc["code_function_count"] == 3
        assert doc["code_analyzer"] == "tree-sitter"
        assert doc["code_has_exports"] is True
        assert doc["code_symbols"] == ["Symbol1", "Symbol2"]

    def test_atom_doc_does_not_have_code_analysis(self, mixin):
        """Test that atom document does not include code analysis fields."""
        record_id = "atom:test123"
        atom_data = {"name": "Test", "content": "Content"}

        doc = mixin._build_meili_doc(record_id, atom_data, "default", doc_type="atom")

        # Atom docs should not have code analysis fields
        code_fields = ["code_language", "code_complexity", "code_function_count",
                      "code_class_count", "code_analyzer", "code_has_exports", "code_symbols"]
        for field in code_fields:
            assert field not in doc, f"Atom doc should not have {field}"

    def test_created_at_is_set_for_atom(self, mixin):
        """Test that created_at is set for atom documents."""
        record_id = "atom:test123"
        atom_data = {"name": "Test"}

        doc = mixin._build_meili_doc(record_id, atom_data, "default", doc_type="atom")

        assert "created_at" in doc
        assert isinstance(doc["created_at"], str)

    def test_empty_atom_data(self, mixin):
        """Test building atom document with minimal/empty data."""
        record_id = "atom:empty123"
        atom_data = {}  # Empty data

        doc = mixin._build_meili_doc(record_id, atom_data, "default", doc_type="atom")

        # Should have default values
        assert doc["id"] == "atom_empty123"
        assert doc["name"] == ""
        assert doc["content"] == ""
        assert doc["atom_type"] == "note"  # Default type
        assert doc["tags"] == []
        assert doc["entity_id"] == ""
        assert doc["local_id"] == ""
        assert doc["heading_level"] is None
        assert "created_at" in doc

    def test_empty_entity_data(self, mixin):
        """Test building entity document with minimal/empty data."""
        record_id = "memory:empty456"
        memory_data = {}  # Empty data

        doc = mixin._build_meili_doc(record_id, memory_data, "default", doc_type="entity")

        # Should have default values
        assert doc["id"] == "memory_empty456"
        assert doc["content"] == ""
        assert doc["type"] == "general"  # Default type
        assert doc["tags"] == []
        assert doc["abstract"] == ""
        assert doc["overview"] == ""
        assert doc["file_path"] is None
        assert "created_at" in doc

    def test_record_id_with_multiple_colons(self, mixin):
        """Test ID conversion with multiple colons in record ID."""
        # SurrealDB IDs can have multiple colons, only first should be replaced
        record_id = "atom:ns:db:table:id"
        atom_data = {"name": "Test"}

        doc = mixin._build_meili_doc(record_id, atom_data, "default", doc_type="atom")

        # Only first colon should be replaced
        assert doc["id"] == "atom_ns:db:table:id"
        assert doc["surreal_id"] == "atom:ns:db:table:id"

    def test_record_id_with_underscore(self, mixin):
        """Test ID conversion with underscore in ID."""
        record_id = "atom:test_id_123"
        atom_data = {"name": "Test"}

        doc = mixin._build_meili_doc(record_id, atom_data, "default", doc_type="atom")

        assert doc["id"] == "atom_test_id_123"
        assert doc["surreal_id"] == "atom:test_id_123"

    def test_special_characters_in_content(self, mixin):
        """Test handling special characters in content."""
        record_id = "atom:special"
        atom_data = {
            "name": "Test with emoji 🎉 and unicode 中文",
            "content": "Special chars: <>&\"'\n\t\\",
            "type": "section",
        }

        doc = mixin._build_meili_doc(record_id, atom_data, "default", doc_type="atom")

        assert doc["name"] == "Test with emoji 🎉 and unicode 中文"
        assert doc["content"] == "Special chars: <>&\"'\n\t\\"
        assert doc["content_zh"] == "Special chars: <>&\"'\n\t\\"
