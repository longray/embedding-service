"""Unit tests for wrapper.src.utils.auth"""

import os
import pytest
from unittest.mock import patch

pytestmark = pytest.mark.unit
from wrapper.src.utils.auth import verify_websocket_token, get_websocket_token


class TestVerifyWebsocketToken:
    def test_no_token_configured_allows_all(self):
        with patch.dict(os.environ, {"WRAPPER_WEBSOCKET_TOKEN": ""}, clear=False):
            # Remove the env var temporarily
            os.environ.pop("WRAPPER_WEBSOCKET_TOKEN", None)
            assert verify_websocket_token("anything") is True
            assert verify_websocket_token(None) is True

    def test_correct_token(self):
        with patch.dict(os.environ, {"WRAPPER_WEBSOCKET_TOKEN": "secret123"}):
            assert verify_websocket_token("secret123") is True

    def test_incorrect_token(self):
        with patch.dict(os.environ, {"WRAPPER_WEBSOCKET_TOKEN": "secret123"}):
            assert verify_websocket_token("wrong") is False

    def test_none_token_when_configured(self):
        with patch.dict(os.environ, {"WRAPPER_WEBSOCKET_TOKEN": "secret123"}):
            assert verify_websocket_token(None) is False


class TestGetWebsocketToken:
    def test_returns_env_value(self):
        with patch.dict(os.environ, {"WRAPPER_WEBSOCKET_TOKEN": "test_token"}):
            assert get_websocket_token() == "test_token"

    def test_returns_none_when_not_set(self):
        os.environ.pop("WRAPPER_WEBSOCKET_TOKEN", None)
        assert get_websocket_token() is None
