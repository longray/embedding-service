"""测试 transaction 上下文管理器"""

import pytest
import pytest_asyncio


class MockDB:
    """模拟 SurrealDB 连接"""

    def __init__(self):
        self.queries = []
        self.should_fail = False

    async def query(self, sql, params=None):
        self.queries.append(sql)
        if self.should_fail and "COMMIT" in sql:
            raise Exception("Commit failed")
        return []


@pytest_asyncio.fixture
async def mock_db():
    return MockDB()


@pytest.mark.asyncio
class TestTransaction:
    """测试 transaction 上下文管理器"""

    async def test_transaction_commit(self, mock_db):
        """测试事务成功提交"""
        from wrapper.src.utils.transaction import transaction

        async with transaction(mock_db, "Test"):
            await mock_db.query("INSERT INTO test { name: 'test' }")

        assert "BEGIN TRANSACTION" in mock_db.queries
        assert "INSERT INTO test { name: 'test' }" in mock_db.queries
        assert "COMMIT TRANSACTION" in mock_db.queries
        assert "CANCEL TRANSACTION" not in mock_db.queries

    async def test_transaction_rollback_on_error(self, mock_db):
        """测试事务出错时回滚"""
        from wrapper.src.utils.transaction import transaction

        with pytest.raises(ValueError):
            async with transaction(mock_db, "Test"):
                await mock_db.query("INSERT INTO test { name: 'test' }")
                raise ValueError("Test error")

        assert "BEGIN TRANSACTION" in mock_db.queries
        assert "CANCEL TRANSACTION" in mock_db.queries
        assert "COMMIT TRANSACTION" not in mock_db.queries

    async def test_transaction_yield_db(self, mock_db):
        """测试上下文管理器 yield 数据库连接"""
        from wrapper.src.utils.transaction import transaction

        async with transaction(mock_db, "Test") as db:
            assert db is mock_db
            await db.query("SELECT * FROM test")

        assert "SELECT * FROM test" in mock_db.queries
