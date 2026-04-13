"""Tests for port migration documentation updates"""

import pytest
import re


class TestPortMigrationDocs:
    """Test port migration documentation updates"""

    def test_readme_mentions_new_port(self):
        """Test README mentions new port 18008"""
        try:
            with open("README.md", "r", encoding="utf-8") as f:
                content = f.read()
            assert "18008" in content
        except FileNotFoundError:
            pytest.skip("README.md not found")

    def test_readme_mentions_legacy_port(self):
        """Test README mentions legacy port 17999"""
        try:
            with open("README.md", "r", encoding="utf-8") as f:
                content = f.read()
            # Should mention both ports for clarity
            assert "17999" in content or "旧端口" in content
        except FileNotFoundError:
            pytest.skip("README.md not found")

    def test_start_guide_mentions_new_port(self):
        """Test START_GUIDE mentions new port 18008"""
        try:
            with open("docs/START_GUIDE.md", "r", encoding="utf-8") as f:
                content = f.read()
            assert "18008" in content
        except FileNotFoundError:
            pytest.skip("docs/START_GUIDE.md not found")

    def test_api_spec_mentions_new_port(self):
        """Test API_SPECIFICATION mentions new port 18008"""
        try:
            with open("docs/API_SPECIFICATION.md", "r", encoding="utf-8") as f:
                content = f.read()
            assert "18008" in content
        except FileNotFoundError:
            pytest.skip("docs/API_SPECIFICATION.md not found")

    def test_readme_examples_use_new_port(self):
        """Test README examples use new port"""
        try:
            with open("README.md", "r", encoding="utf-8") as f:
                content = f.read()
            # Check that curl examples use 18008
            curl_examples = re.findall(r"curl.*localhost:(\d+)", content)
            for port in curl_examples:
                assert port == "18008", f"Found old port {port} in curl example"
        except FileNotFoundError:
            pytest.skip("README.md not found")

    def test_start_guide_examples_use_new_port(self):
        """Test START_GUIDE examples use new port"""
        try:
            with open("docs/START_GUIDE.md", "r", encoding="utf-8") as f:
                content = f.read()
            # Check that wrapper curl examples use 18008 (not embedding 18000)
            curl_examples = re.findall(r"curl.*localhost:(\d+)", content)
            for port in curl_examples:
                # Skip embedding service port 18000
                if port == "18000":
                    continue
                assert port == "18008", f"Found old port {port} in curl example"
        except FileNotFoundError:
            pytest.skip("docs/START_GUIDE.md not found")

    def test_environment_variable_docs_updated(self):
        """Test environment variable documentation updated"""
        try:
            with open("docs/API_SPECIFICATION.md", "r", encoding="utf-8") as f:
                content = f.read()
            # Check WRAPPER_PORT default is 18008
            assert "WRAPPER_PORT" in content
            assert "18008" in content
        except FileNotFoundError:
            pytest.skip("docs/API_SPECIFICATION.md not found")

    def test_dual_port_mentioned(self):
        """Test dual port support is mentioned"""
        try:
            with open("README.md", "r", encoding="utf-8") as f:
                content = f.read()
            # Should mention dual port or parallel support
            assert any(term in content for term in ["双端口", "并行", "legacy", "旧端口"])
        except FileNotFoundError:
            pytest.skip("README.md not found")
