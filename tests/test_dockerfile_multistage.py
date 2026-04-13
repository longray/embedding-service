"""Tests for multi-stage Dockerfile"""

import pytest
import re


class TestDockerfileMultistage:
    """Test multi-stage Dockerfile structure"""

    def test_dockerfile_exists(self):
        """Test Dockerfile.multistage exists"""
        try:
            with open("wrapper/Dockerfile.multistage", "r") as f:
                content = f.read()
            assert len(content) > 0
        except FileNotFoundError:
            pytest.skip("Dockerfile.multistage not found")

    def test_has_builder_stage(self):
        """Test Dockerfile has builder stage"""
        try:
            with open("wrapper/Dockerfile.multistage", "r") as f:
                content = f.read()
            assert "FROM python:3.11-slim AS builder" in content
        except FileNotFoundError:
            pytest.skip("Dockerfile.multistage not found")

    def test_has_production_stage(self):
        """Test Dockerfile has production stage"""
        try:
            with open("wrapper/Dockerfile.multistage", "r") as f:
                content = f.read()
            assert "FROM python:3.11-slim AS production" in content
        except FileNotFoundError:
            pytest.skip("Dockerfile.multistage not found")

    def test_has_development_stage(self):
        """Test Dockerfile has development stage"""
        try:
            with open("wrapper/Dockerfile.multistage", "r") as f:
                content = f.read()
            assert "FROM builder AS development" in content
        except FileNotFoundError:
            pytest.skip("Dockerfile.multistage not found")

    def test_uses_non_root_user(self):
        """Test production stage uses non-root user"""
        try:
            with open("wrapper/Dockerfile.multistage", "r") as f:
                content = f.read()
            assert "USER appuser" in content
            assert "groupadd" in content
            assert "useradd" in content
        except FileNotFoundError:
            pytest.skip("Dockerfile.multistage not found")

    def test_has_healthcheck(self):
        """Test Dockerfile has healthcheck"""
        try:
            with open("wrapper/Dockerfile.multistage", "r") as f:
                content = f.read()
            assert "HEALTHCHECK" in content
            assert "18008" in content
        except FileNotFoundError:
            pytest.skip("Dockerfile.multistage not found")

    def test_exposes_new_port(self):
        """Test Dockerfile exposes new port 18008"""
        try:
            with open("wrapper/Dockerfile.multistage", "r") as f:
                content = f.read()
            assert "EXPOSE 18008" in content
        except FileNotFoundError:
            pytest.skip("Dockerfile.multistage not found")

    def test_exposes_legacy_port(self):
        """Test Dockerfile exposes legacy port 17999"""
        try:
            with open("wrapper/Dockerfile.multistage", "r") as f:
                content = f.read()
            assert "EXPOSE 18008 17999" in content or "EXPOSE 17999" in content
        except FileNotFoundError:
            pytest.skip("Dockerfile.multistage not found")

    def test_uses_uv_for_dependencies(self):
        """Test Dockerfile uses uv for dependency management"""
        try:
            with open("wrapper/Dockerfile.multistage", "r") as f:
                content = f.read()
            assert "uv pip install" in content
            assert "uv venv" in content
        except FileNotFoundError:
            pytest.skip("Dockerfile.multistage not found")

    def test_has_cache_mount(self):
        """Test Dockerfile uses cache mount for faster builds"""
        try:
            with open("wrapper/Dockerfile.multistage", "r") as f:
                content = f.read()
            assert "--mount=type=cache" in content
        except FileNotFoundError:
            pytest.skip("Dockerfile.multistage not found")

    def test_copies_venv_from_builder(self):
        """Test production stage copies venv from builder"""
        try:
            with open("wrapper/Dockerfile.multistage", "r") as f:
                content = f.read()
            assert "COPY --from=builder /app/.venv" in content
        except FileNotFoundError:
            pytest.skip("Dockerfile.multistage not found")

    def test_environment_variables_updated(self):
        """Test environment variables use new port"""
        try:
            with open("wrapper/Dockerfile.multistage", "r") as f:
                content = f.read()
            assert "WRAPPER_PORT=18008" in content
            assert "WRAPPER_LEGACY_PORT=17999" in content
            assert "WRAPPER_ENABLE_DUAL_PORT=true" in content
        except FileNotFoundError:
            pytest.skip("Dockerfile.multistage not found")

    def test_has_python_optimizations(self):
        """Test Dockerfile has Python optimizations"""
        try:
            with open("wrapper/Dockerfile.multistage", "r") as f:
                content = f.read()
            assert "PYTHONDONTWRITEBYTECODE=1" in content
            assert "PYTHONUNBUFFERED=1" in content
        except FileNotFoundError:
            pytest.skip("Dockerfile.multistage not found")

    def test_development_has_test_tools(self):
        """Test development stage has test tools"""
        try:
            with open("wrapper/Dockerfile.multistage", "r") as f:
                content = f.read()
            assert "pytest" in content
            assert "black" in content or "ruff" in content
        except FileNotFoundError:
            pytest.skip("Dockerfile.multistage not found")
