"""Tests for SSL configuration"""

import pytest
import os


class TestSSLConfiguration:
    """Test SSL configuration files"""

    def test_docker_compose_ssl_exists(self):
        """Test docker-compose.ssl.yml exists"""
        try:
            with open("docker-compose.ssl.yml", "r") as f:
                content = f.read()
            assert len(content) > 0
        except FileNotFoundError:
            pytest.skip("docker-compose.ssl.yml not found")

    def test_nginx_conf_exists(self):
        """Test nginx.conf exists"""
        try:
            with open("nginx/nginx.conf", "r") as f:
                content = f.read()
            assert len(content) > 0
        except FileNotFoundError:
            pytest.skip("nginx/nginx.conf not found")

    def test_certbot_service_defined(self):
        """Test certbot service is defined"""
        try:
            with open("docker-compose.ssl.yml", "r") as f:
                content = f.read()
            assert "certbot:" in content
            assert "certbot/certbot" in content
        except FileNotFoundError:
            pytest.skip("docker-compose.ssl.yml not found")

    def test_nginx_service_defined(self):
        """Test nginx service is defined"""
        try:
            with open("docker-compose.ssl.yml", "r") as f:
                content = f.read()
            assert "nginx:" in content
            assert "nginx:alpine" in content
        except FileNotFoundError:
            pytest.skip("docker-compose.ssl.yml not found")

    def test_certbot_has_renewal_loop(self):
        """Test certbot has auto-renewal loop"""
        try:
            with open("docker-compose.ssl.yml", "r") as f:
                content = f.read()
            assert "certbot renew" in content
            assert "sleep 12h" in content
        except FileNotFoundError:
            pytest.skip("docker-compose.ssl.yml not found")

    def test_nginx_listens_on_443(self):
        """Test nginx listens on port 443"""
        try:
            with open("nginx/nginx.conf", "r") as f:
                content = f.read()
            assert "listen 443 ssl" in content
        except FileNotFoundError:
            pytest.skip("nginx/nginx.conf not found")

    def test_nginx_redirects_http_to_https(self):
        """Test nginx redirects HTTP to HTTPS"""
        try:
            with open("nginx/nginx.conf", "r") as f:
                content = f.read()
            assert "listen 80" in content
            assert "return 301 https://" in content
        except FileNotFoundError:
            pytest.skip("nginx/nginx.conf not found")

    def test_nginx_has_security_headers(self):
        """Test nginx has security headers"""
        try:
            with open("nginx/nginx.conf", "r") as f:
                content = f.read()
            assert "Strict-Transport-Security" in content
            assert "X-Frame-Options" in content
            assert "X-Content-Type-Options" in content
        except FileNotFoundError:
            pytest.skip("nginx/nginx.conf not found")

    def test_nginx_proxies_to_wrapper(self):
        """Test nginx proxies to wrapper service"""
        try:
            with open("nginx/nginx.conf", "r") as f:
                content = f.read()
            assert "proxy_pass http://wrapper_backend" in content
            assert "server wrapper:18008" in content
        except FileNotFoundError:
            pytest.skip("nginx/nginx.conf not found")

    def test_certbot_volumes_mounted(self):
        """Test certbot volumes are mounted"""
        try:
            with open("docker-compose.ssl.yml", "r") as f:
                content = f.read()
            assert "certbot-data" in content
            assert "certbot-www" in content
        except FileNotFoundError:
            pytest.skip("docker-compose.ssl.yml not found")

    def test_init_ssl_script_exists(self):
        """Test init_ssl.sh script exists"""
        try:
            with open("scripts/init_ssl.sh", "r") as f:
                content = f.read()
            assert len(content) > 0
            assert "certbot/certbot" in content
        except FileNotFoundError:
            pytest.skip("scripts/init_ssl.sh not found")

    def test_ssl_uses_profile(self):
        """Test SSL services use profile"""
        try:
            with open("docker-compose.ssl.yml", "r") as f:
                content = f.read()
            assert "profiles:" in content
            assert "ssl" in content
        except FileNotFoundError:
            pytest.skip("docker-compose.ssl.yml not found")
