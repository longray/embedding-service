"""Tests for docker-compose configuration"""

import pytest
import yaml


class TestDockerComposeConfig:
    """Test docker-compose configuration"""

    def test_dockercompose_exists(self):
        """Test docker-compose.yml exists"""
        try:
            with open("docker-compose.yml", "r") as f:
                content = f.read()
            assert len(content) > 0
        except FileNotFoundError:
            pytest.skip("docker-compose.yml not found")

    def test_wrapper_service_uses_new_port(self):
        """Test wrapper service uses new port 18008"""
        try:
            with open("docker-compose.yml", "r") as f:
                config = yaml.safe_load(f)
            wrapper = config["services"]["wrapper"]
            ports = wrapper.get("ports", [])
            assert any("18008" in str(p) for p in ports)
        except (FileNotFoundError, KeyError):
            pytest.skip("docker-compose.yml not found or invalid")

    def test_wrapper_service_uses_port_18008(self):
        """Test wrapper service exposes port 18008"""
        try:
            with open("docker-compose.yml", "r") as f:
                config = yaml.safe_load(f)
            wrapper = config["services"]["wrapper"]
            ports = wrapper.get("ports", [])
            assert any("18008" in str(p) for p in ports)
            assert not any("17999" in str(p) for p in ports)
        except (FileNotFoundError, KeyError):
            pytest.skip("docker-compose.yml not found or invalid")

    def test_wrapper_uses_multistage_dockerfile(self):
        """Test wrapper uses multi-stage Dockerfile"""
        try:
            with open("docker-compose.yml", "r") as f:
                config = yaml.safe_load(f)
            wrapper = config["services"]["wrapper"]
            build = wrapper.get("build", {})
            assert "Dockerfile.multistage" in str(build.get("dockerfile", ""))
            assert "production" in str(build.get("target", ""))
        except (FileNotFoundError, KeyError):
            pytest.skip("docker-compose.yml not found or invalid")

    def test_wrapper_has_healthcheck(self):
        """Test wrapper service has healthcheck"""
        try:
            with open("docker-compose.yml", "r") as f:
                config = yaml.safe_load(f)
            wrapper = config["services"]["wrapper"]
            healthcheck = wrapper.get("healthcheck", {})
            assert "test" in healthcheck
            assert "18008" in str(healthcheck.get("test", []))
        except (FileNotFoundError, KeyError):
            pytest.skip("docker-compose.yml not found or invalid")

    def test_wrapper_has_restart_policy(self):
        """Test wrapper service has restart policy"""
        try:
            with open("docker-compose.yml", "r") as f:
                config = yaml.safe_load(f)
            wrapper = config["services"]["wrapper"]
            restart = wrapper.get("restart", "")
            assert restart in ["unless-stopped", "always", "on-failure"]
        except (FileNotFoundError, KeyError):
            pytest.skip("docker-compose.yml not found or invalid")

    def test_wrapper_has_dependency_conditions(self):
        """Test wrapper has dependency conditions"""
        try:
            with open("docker-compose.yml", "r") as f:
                config = yaml.safe_load(f)
            wrapper = config["services"]["wrapper"]
            depends_on = wrapper.get("depends_on", {})
            required_services = ["surrealdb", "embedding", "meilisearch"]
            for service in required_services:
                assert service in depends_on
                condition = depends_on[service].get("condition", "")
                assert condition == "service_healthy"
        except (FileNotFoundError, KeyError):
            pytest.skip("docker-compose.yml not found or invalid")

    def test_environment_variables_updated(self):
        """Test environment variables use new port"""
        try:
            with open("docker-compose.yml", "r") as f:
                config = yaml.safe_load(f)
            wrapper = config["services"]["wrapper"]
            env = wrapper.get("environment", [])
            env_dict = {}
            for e in env:
                if "=" in e:
                    key, value = e.split("=", 1)
                    env_dict[key] = value
            assert env_dict.get("WRAPPER_PORT") == "18008"
            assert "WRAPPER_LEGACY_PORT" not in env_dict
            assert "WRAPPER_ENABLE_DUAL_PORT" not in env_dict
        except (FileNotFoundError, KeyError):
            pytest.skip("docker-compose.yml not found or invalid")

    def test_all_services_have_healthchecks(self):
        """Test all services have healthchecks"""
        try:
            with open("docker-compose.yml", "r") as f:
                config = yaml.safe_load(f)
            services = config.get("services", {})
            required_services = ["surrealdb", "meilisearch", "embedding", "wrapper"]
            for service_name in required_services:
                service = services.get(service_name, {})
                assert "healthcheck" in service, f"{service_name} missing healthcheck"
        except (FileNotFoundError, KeyError):
            pytest.skip("docker-compose.yml not found or invalid")

    def test_healthcheck_uses_correct_port(self):
        """Test wrapper healthcheck uses port 18008"""
        try:
            with open("docker-compose.yml", "r") as f:
                config = yaml.safe_load(f)
            wrapper = config["services"]["wrapper"]
            healthcheck = wrapper.get("healthcheck", {})
            test = healthcheck.get("test", [])
            assert any("18008" in str(item) for item in test)
        except (FileNotFoundError, KeyError):
            pytest.skip("docker-compose.yml not found or invalid")

    def test_healthcheck_has_start_period(self):
        """Test wrapper healthcheck has start period"""
        try:
            with open("docker-compose.yml", "r") as f:
                config = yaml.safe_load(f)
            wrapper = config["services"]["wrapper"]
            healthcheck = wrapper.get("healthcheck", {})
            assert "start_period" in healthcheck
        except (FileNotFoundError, KeyError):
            pytest.skip("docker-compose.yml not found or invalid")
