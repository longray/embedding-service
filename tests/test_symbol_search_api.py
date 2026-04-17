"""BL-B-83 符号查询 API 单元测试

纯 mock 测试，无需启动数据库或外部服务。
覆盖：精确查询、类型过滤、前缀匹配、项目过滤、参数验证、错误处理。
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from wrapper.src.models import SymbolMatch, SymbolSearchResponse
from wrapper.src.services.symbol_service import VALID_SYMBOL_TYPES, SymbolService

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_db():
    db = MagicMock()
    db.query = AsyncMock()
    return db


@pytest.fixture
def service(mock_db):
    return SymbolService(db=mock_db)


def _make_db_result(records: list[dict]) -> list[dict]:
    return [{"result": records}]


# ============================================================================
# SymbolService._extract_records
# ============================================================================


class TestExtractRecords:
    def test_none_input(self):
        assert SymbolService._extract_records(None) == []

    def test_empty_list(self):
        assert SymbolService._extract_records([]) == []

    def test_result_format(self):
        data = [{"result": [{"id": "mem:1"}]}]
        result = SymbolService._extract_records(data)
        assert len(result) == 1
        assert result[0]["id"] == "mem:1"

    def test_result_format_empty(self):
        data = [{"result": []}]
        assert SymbolService._extract_records(data) == []

    def test_result_format_single_dict(self):
        data = [{"result": {"id": "mem:1"}}]
        result = SymbolService._extract_records(data)
        assert len(result) == 1
        assert result[0]["id"] == "mem:1"

    def test_result_format_none_value(self):
        data = [{"result": None}]
        assert SymbolService._extract_records(data) == []

    def test_direct_list_format(self):
        data = [[{"id": "mem:1"}]]
        result = SymbolService._extract_records(data)
        assert len(result) == 1

    def test_direct_dict_format(self):
        data = [{"id": "mem:1"}]
        result = SymbolService._extract_records(data)
        assert len(result) == 1


# ============================================================================
# SymbolService.search — 精确查询
# ============================================================================


class TestSymbolSearchExact:
    @pytest.mark.asyncio
    async def test_exact_match_returns_symbol(self, service, mock_db):
        mock_db.query.return_value = _make_db_result(
            [
                {
                    "id": "mem:abc123",
                    "name": "main",
                    "type": "function",
                    "file": "src/main.py",
                    "line": 10,
                    "signature": "def main():",
                }
            ]
        )

        result = await service.search(query="main", tenant_id="default")

        assert result.total == 1
        assert len(result.symbols) == 1
        sym = result.symbols[0]
        assert isinstance(sym, SymbolMatch)
        assert sym.name == "main"
        assert sym.type == "function"
        assert sym.file == "src/main.py"
        assert sym.line == 10
        assert sym.memory_id == "mem:abc123"
        assert sym.signature == "def main():"

    @pytest.mark.asyncio
    async def test_exact_match_no_results(self, service, mock_db):
        mock_db.query.return_value = _make_db_result([])

        result = await service.search(query="nonexistent")

        assert result.total == 0
        assert result.symbols == []

    @pytest.mark.asyncio
    async def test_exact_match_multiple_results(self, service, mock_db):
        mock_db.query.return_value = _make_db_result(
            [
                {"id": "mem:1", "name": "parse", "type": "function", "file": "a.py", "line": 5},
                {"id": "mem:2", "name": "parse", "type": "method", "file": "b.py", "line": 20},
            ]
        )

        result = await service.search(query="parse")

        assert result.total == 2

    @pytest.mark.asyncio
    async def test_query_uses_correct_params(self, service, mock_db):
        mock_db.query.return_value = _make_db_result([])

        await service.search(query="hello", tenant_id="mytenant")

        mock_db.query.assert_called_once()
        call_args = mock_db.query.call_args
        params = call_args[0][1]
        assert params["query"] == "hello"
        assert params["tenant_id"] == "mytenant"


# ============================================================================
# SymbolService.search — 类型过滤
# ============================================================================


class TestSymbolSearchTypeFilter:
    @pytest.mark.asyncio
    async def test_filter_by_function(self, service, mock_db):
        mock_db.query.return_value = _make_db_result(
            [{"id": "mem:1", "name": "main", "type": "function", "file": "main.py", "line": 1}]
        )

        result = await service.search(query="main", symbol_type="function")

        assert result.total == 1
        call_args = mock_db.query.call_args
        params = call_args[0][1]
        assert params["symbol_type"] == "function"

    @pytest.mark.asyncio
    async def test_filter_by_class(self, service, mock_db):
        mock_db.query.return_value = _make_db_result(
            [{"id": "mem:1", "name": "User", "type": "class", "file": "models.py", "line": 5}]
        )

        result = await service.search(query="User", symbol_type="class")

        assert result.total == 1

    @pytest.mark.asyncio
    async def test_invalid_type_raises(self, service, mock_db):
        with pytest.raises(ValueError, match="Invalid symbol type"):
            await service.search(query="test", symbol_type="invalid_type")

    @pytest.mark.asyncio
    async def test_all_valid_types_accepted(self, service, mock_db):
        mock_db.query.return_value = _make_db_result([])

        for st in VALID_SYMBOL_TYPES:
            await service.search(query="x", symbol_type=st)

        assert mock_db.query.call_count == len(VALID_SYMBOL_TYPES)


# ============================================================================
# SymbolService.search — 前缀模糊匹配
# ============================================================================


class TestSymbolSearchFuzzy:
    @pytest.mark.asyncio
    async def test_prefix_match(self, service, mock_db):
        mock_db.query.return_value = _make_db_result(
            [
                {"id": "mem:1", "name": "parseJSON", "type": "function", "file": "a.js", "line": 1},
                {"id": "mem:2", "name": "parseXML", "type": "function", "file": "b.js", "line": 2},
            ]
        )

        result = await service.search(query="parse", fuzzy=True)

        assert result.total == 2

    @pytest.mark.asyncio
    async def test_fuzzy_query_includes_starts_with(self, service, mock_db):
        mock_db.query.return_value = _make_db_result([])

        await service.search(query="get", fuzzy=True)

        sql = mock_db.query.call_args[0][0]
        assert "string::starts_with" in sql


# ============================================================================
# SymbolService.search — 项目过滤
# ============================================================================


class TestSymbolSearchProjectFilter:
    @pytest.mark.asyncio
    async def test_project_filter(self, service, mock_db):
        mock_db.query.return_value = _make_db_result(
            [{"id": "mem:1", "name": "foo", "type": "function", "file": "bar.py", "line": 1}]
        )

        await service.search(query="foo", project_id="my-project")

        call_args = mock_db.query.call_args
        params = call_args[0][1]
        assert params["project_id"] == "my-project"

    @pytest.mark.asyncio
    async def test_project_filter_in_sql(self, service, mock_db):
        mock_db.query.return_value = _make_db_result([])

        await service.search(query="foo", project_id="my-project")

        sql = mock_db.query.call_args[0][0]
        assert "project_id = $project_id" in sql


# ============================================================================
# SymbolService.search — limit 参数
# ============================================================================


class TestSymbolSearchLimit:
    @pytest.mark.asyncio
    async def test_limit_passed_to_query(self, service, mock_db):
        mock_db.query.return_value = _make_db_result([])

        await service.search(query="x", limit=5)

        params = mock_db.query.call_args[0][1]
        assert params["limit"] == 5

    @pytest.mark.asyncio
    async def test_default_limit(self, service, mock_db):
        mock_db.query.return_value = _make_db_result([])

        await service.search(query="x")

        params = mock_db.query.call_args[0][1]
        assert params["limit"] == 20


# ============================================================================
# SymbolSearchResponse 模型测试
# ============================================================================


class TestSymbolSearchResponseModel:
    def test_empty_response(self):
        resp = SymbolSearchResponse(symbols=[], total=0)
        assert resp.symbols == []
        assert resp.total == 0

    def test_response_with_symbols(self):
        symbols = [
            SymbolMatch(name="main", type="function", file="main.py", line=1, memory_id="mem:1"),
            SymbolMatch(name="User", type="class", file="models.py", line=5, memory_id="mem:2"),
        ]
        resp = SymbolSearchResponse(symbols=symbols, total=2)
        assert resp.total == 2
        assert resp.symbols[0].name == "main"

    def test_model_dump(self):
        sym = SymbolMatch(name="f", type="function", file="a.py", line=1, memory_id="mem:1", signature="def f():")
        resp = SymbolSearchResponse(symbols=[sym], total=1)
        d = resp.model_dump()
        assert d["total"] == 1
        assert d["symbols"][0]["name"] == "f"
        assert d["symbols"][0]["signature"] == "def f():"


# ============================================================================
# Router 集成测试 (mock state)
# ============================================================================


class TestSymbolSearchRouter:
    @pytest.mark.asyncio
    async def test_router_service_unavailable(self):
        with patch("wrapper.src.routers.symbols.state") as mock_state:
            mock_state.memory_manager = None

            from fastapi import FastAPI
            from fastapi.testclient import TestClient

            from wrapper.src.routers.symbols import router

            app = FastAPI()
            app.include_router(router)
            client = TestClient(app)

            response = client.get("/api/v1/symbols/search?query=main")
            assert response.status_code == 503

    @pytest.mark.asyncio
    async def test_router_missing_query_param(self):
        with patch("wrapper.src.routers.symbols.state") as mock_state:
            mock_state.memory_manager = MagicMock()

            from fastapi import FastAPI
            from fastapi.testclient import TestClient

            from wrapper.src.routers.symbols import router

            app = FastAPI()
            app.include_router(router)
            client = TestClient(app)

            response = client.get("/api/v1/symbols/search")
            assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_router_success(self):
        with patch("wrapper.src.routers.symbols.state") as mock_state:
            mock_db = MagicMock()
            mock_db.query = AsyncMock(
                return_value=_make_db_result(
                    [{"id": "mem:1", "name": "main", "type": "function", "file": "main.py", "line": 10}]
                )
            )
            mm = MagicMock()
            mm._db = mock_db
            mock_state.memory_manager = mm

            from fastapi import FastAPI
            from fastapi.testclient import TestClient

            from wrapper.src.routers.symbols import router

            app = FastAPI()
            app.include_router(router)
            client = TestClient(app)

            response = client.get("/api/v1/symbols/search?query=main")
            assert response.status_code == 200
            data = response.json()
            assert data["total"] == 1
            assert data["symbols"][0]["name"] == "main"
            assert data["symbols"][0]["type"] == "function"
            assert data["symbols"][0]["file"] == "main.py"
            assert data["symbols"][0]["line"] == 10
            assert data["symbols"][0]["memory_id"] == "mem:1"

    @pytest.mark.asyncio
    async def test_router_with_type_filter(self):
        with patch("wrapper.src.routers.symbols.state") as mock_state:
            mock_db = MagicMock()
            mock_db.query = AsyncMock(return_value=_make_db_result([]))
            mm = MagicMock()
            mm._db = mock_db
            mock_state.memory_manager = mm

            from fastapi import FastAPI
            from fastapi.testclient import TestClient

            from wrapper.src.routers.symbols import router

            app = FastAPI()
            app.include_router(router)
            client = TestClient(app)

            response = client.get("/api/v1/symbols/search?query=main&type=function")
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_router_with_fuzzy(self):
        with patch("wrapper.src.routers.symbols.state") as mock_state:
            mock_db = MagicMock()
            mock_db.query = AsyncMock(return_value=_make_db_result([]))
            mm = MagicMock()
            mm._db = mock_db
            mock_state.memory_manager = mm

            from fastapi import FastAPI
            from fastapi.testclient import TestClient

            from wrapper.src.routers.symbols import router

            app = FastAPI()
            app.include_router(router)
            client = TestClient(app)

            response = client.get("/api/v1/symbols/search?query=main&fuzzy=true")
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_router_with_project_filter(self):
        with patch("wrapper.src.routers.symbols.state") as mock_state:
            mock_db = MagicMock()
            mock_db.query = AsyncMock(return_value=_make_db_result([]))
            mm = MagicMock()
            mm._db = mock_db
            mock_state.memory_manager = mm

            from fastapi import FastAPI
            from fastapi.testclient import TestClient

            from wrapper.src.routers.symbols import router

            app = FastAPI()
            app.include_router(router)
            client = TestClient(app)

            response = client.get("/api/v1/symbols/search?query=main&project_id=test-project")
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_router_invalid_type(self):
        with patch("wrapper.src.routers.symbols.state") as mock_state:
            mock_db = MagicMock()
            mock_db.query = AsyncMock(return_value=_make_db_result([]))
            mm = MagicMock()
            mm._db = mock_db
            mock_state.memory_manager = mm

            from fastapi import FastAPI
            from fastapi.testclient import TestClient

            from wrapper.src.routers.symbols import router

            app = FastAPI()
            app.include_router(router)
            client = TestClient(app)

            response = client.get("/api/v1/symbols/search?query=main&type=invalid_type")
            assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_router_db_error(self):
        with patch("wrapper.src.routers.symbols.state") as mock_state:
            mock_db = MagicMock()
            mock_db.query = AsyncMock(side_effect=RuntimeError("DB connection lost"))
            mm = MagicMock()
            mm._db = mock_db
            mock_state.memory_manager = mm

            from fastapi import FastAPI
            from fastapi.testclient import TestClient

            from wrapper.src.routers.symbols import router

            app = FastAPI()
            app.include_router(router)
            client = TestClient(app)

            response = client.get("/api/v1/symbols/search?query=main")
            assert response.status_code == 500
