"""Tests for port migration configuration"""

import os
import pytest

from wrapper.src.config import AppConfig, load_config


class TestPortMigrationConfig:
    """Test port migration configuration"""

    def test_default_port_is_18008(self):
        """Test default port is 18008"""
        cfg = AppConfig()
        assert cfg.port == 18008

    def test_legacy_port_is_17999(self):
        """Test legacy port is 17999"""
        cfg = AppConfig()
        assert cfg.legacy_port == 17999

    def test_dual_port_enabled_by_default(self):
        """Test dual port is enabled by default"""
        cfg = AppConfig()
        assert cfg.enable_dual_port is True

    def test_dual_port_duration_default(self):
        """Test dual port duration is 14 days by default"""
        cfg = AppConfig()
        assert cfg.dual_port_duration_days == 14


class TestPortMigrationEnvVars:
    """Test port migration environment variables"""

    def test_port_from_env(self):
        """Test port can be set from environment variable"""
        os.environ["WRAPPER_PORT"] = "18009"
        cfg = load_config()
        assert cfg.port == 18009
        del os.environ["WRAPPER_PORT"]

    def test_legacy_port_from_env(self):
        """Test legacy port can be set from environment variable"""
        os.environ["WRAPPER_LEGACY_PORT"] = "17998"
        cfg = load_config()
        assert cfg.legacy_port == 17998
        del os.environ["WRAPPER_LEGACY_PORT"]

    def test_enable_dual_port_from_env(self):
        """Test dual port can be disabled from environment variable"""
        os.environ["WRAPPER_ENABLE_DUAL_PORT"] = "false"
        cfg = load_config()
        assert cfg.enable_dual_port is False
        del os.environ["WRAPPER_ENABLE_DUAL_PORT"]

    def test_dual_port_duration_from_env(self):
        """Test dual port duration can be set from environment variable"""
        os.environ["WRAPPER_DUAL_PORT_DURATION_DAYS"] = "7"
        cfg = load_config()
        assert cfg.dual_port_duration_days == 7
        del os.environ["WRAPPER_DUAL_PORT_DURATION_DAYS"]

    def test_port_migration_backward_compatibility(self):
        """Test backward compatibility with old WRAPPER_PORT"""
        os.environ["WRAPPER_PORT"] = "17999"
        cfg = load_config()
        assert cfg.port == 17999
        del os.environ["WRAPPER_PORT"]


class TestPortMigrationValidation:
    """Test port migration validation"""

    def test_ports_are_different(self):
        """Test new and legacy ports are different"""
        cfg = AppConfig()
        assert cfg.port != cfg.legacy_port

    def test_ports_are_positive(self):
        """Test ports are positive integers"""
        cfg = AppConfig()
        assert cfg.port > 0
        assert cfg.legacy_port > 0

    def test_ports_are_valid_range(self):
        """Test ports are in valid range (1024-65535)"""
        cfg = AppConfig()
        assert 1024 <= cfg.port <= 65535
        assert 1024 <= cfg.legacy_port <= 65535

    def test_dual_port_duration_positive(self):
        """Test dual port duration is positive"""
        cfg = AppConfig()
        assert cfg.dual_port_duration_days > 0
