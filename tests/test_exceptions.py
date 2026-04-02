"""Unit tests for wrapper.src.utils.exceptions"""

import pytest
from wrapper.src.utils.exceptions import (
    WrapperServiceError,
    ServiceUnavailableError,
    ValidationError,
    EmbeddingError,
    DatabaseError,
)


class TestWrapperServiceError:
    def test_default_status_code(self):
        err = WrapperServiceError("test error")
        assert err.message == "test error"
        assert err.status_code == 500
        assert err.details == {}
        assert str(err) == "test error"

    def test_custom_status_code(self):
        err = WrapperServiceError("test", status_code=418)
        assert err.status_code == 418

    def test_with_details(self):
        err = WrapperServiceError("test", details={"key": "value"})
        assert err.details == {"key": "value"}

    def test_is_exception(self):
        err = WrapperServiceError("test")
        assert isinstance(err, Exception)


class TestServiceUnavailableError:
    def test_status_code_503(self):
        err = ServiceUnavailableError("embedding")
        assert err.status_code == 503
        assert "embedding" in err.message

    def test_with_details(self):
        err = ServiceUnavailableError("llm", details={"latency_ms": 5000})
        assert err.details["latency_ms"] == 5000

    def test_inherits_wrapper_error(self):
        err = ServiceUnavailableError("test")
        assert isinstance(err, WrapperServiceError)


class TestValidationError:
    def test_status_code_400(self):
        err = ValidationError("invalid input")
        assert err.status_code == 400

    def test_inherits_wrapper_error(self):
        err = ValidationError("test")
        assert isinstance(err, WrapperServiceError)


class TestEmbeddingError:
    def test_status_code_502(self):
        err = EmbeddingError("model failed")
        assert err.status_code == 502

    def test_inherits_wrapper_error(self):
        err = EmbeddingError("test")
        assert isinstance(err, WrapperServiceError)


class TestDatabaseError:
    def test_status_code_500(self):
        err = DatabaseError("query failed")
        assert err.status_code == 500

    def test_inherits_wrapper_error(self):
        err = DatabaseError("test")
        assert isinstance(err, WrapperServiceError)
