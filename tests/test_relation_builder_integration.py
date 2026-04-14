"""RelationBuilder SurrealDB 集成测试

测试范围：
- SurrealDB RELATE 语句生成
- 批量 RELATE 操作
- 错误处理和重试机制
- 关系查询接口

运行方式：
    uv run pytest tests/test_relation_builder_integration.py -v
"""

import pytest
from unittest.mock import MagicMock, patch

from wrapper.src.services.relation_builder import RelationBuilder, CallRelation


class MockSurrealDB:
    """模拟 SurrealDB 客户端"""

    def __init__(self):
        self.queries = []
        self.should_fail = False

    def query(self, sql: str):
        """模拟查询执行"""
        self.queries.append(sql)

        if self.should_fail:
            raise Exception("Database error")

        # 返回模拟结果
        return [{"result": "success"}]


class TestRelationBuilderSurrealDBIntegration:
    """RelationBuilder SurrealDB 集成测试"""

    @pytest.fixture
    def mock_db(self):
        """创建模拟 SurrealDB 客户端"""
        return MockSurrealDB()

    @pytest.fixture
    def builder(self, mock_db):
        """创建 RelationBuilder 实例"""
        return RelationBuilder(db=mock_db)

    def test_get_atom_id(self, builder):
        """测试获取 atom ID"""
        # 普通函数名
        atom_id = builder._get_atom_id("module.function")
        assert atom_id == "atom:module_function"

        # 带连字符的函数名
        atom_id = builder._get_atom_id("my-function")
        assert atom_id == "atom:my_function"

    def test_create_surrealdb_relations(self, builder, mock_db):
        """测试创建 SurrealDB 关系"""
        relations = [
            CallRelation(
                caller="module.func1",
                callee="module.func2",
                weight=1.0,
                relation_type="calls",
                file_path="/test/file.py",
            ),
            CallRelation(
                caller="module.func2",
                callee="module.func3",
                weight=0.8,
                relation_type="calls",
                file_path="/test/file.py",
            ),
        ]

        builder._create_surrealdb_relations(relations)

        # 验证执行了查询
        assert len(mock_db.queries) == 1

        # 验证 RELATE 语句
        query = mock_db.queries[0]
        assert "BEGIN TRANSACTION" in query
        assert "RELATE atom:module_func1->calls->atom:module_func2" in query
        assert "RELATE atom:module_func2->calls->atom:module_func3" in query
        assert "COMMIT TRANSACTION" in query

    def test_execute_relate_statements_success(self, builder, mock_db):
        """测试成功执行 RELATE 语句"""
        statements = [
            'RELATE atom:a->calls->atom:b SET weight=1.0, type="calls", file_path="/test.py"',
        ]

        builder._execute_relate_statements(statements)

        # 验证查询被执行
        assert len(mock_db.queries) == 1
        assert "BEGIN TRANSACTION" in mock_db.queries[0]

        # 验证关系被记录到内存
        assert builder.relation_count == 1

    def test_execute_relate_statements_failure(self, builder, mock_db):
        """测试 RELATE 语句执行失败"""
        mock_db.should_fail = True

        statements = [
            'RELATE atom:a->calls->atom:b SET weight=1.0, type="calls", file_path="/test.py"',
        ]

        with pytest.raises(Exception, match="Database error"):
            builder._execute_relate_statements(statements)

    def test_parse_relate_statement(self, builder):
        """测试解析 RELATE 语句"""
        stmt = (
            'RELATE atom:module_func1->calls->atom:module_func2 SET weight=1.5, type="calls", file_path="/test/file.py"'
        )

        relation = builder._parse_relate_statement(stmt)

        assert relation is not None
        assert relation.caller == "module.func1"
        assert relation.callee == "module.func2"
        assert relation.weight == 1.5
        assert relation.relation_type == "calls"
        assert relation.file_path == "/test/file.py"

    def test_parse_relate_statement_invalid(self, builder):
        """测试解析无效的 RELATE 语句"""
        # 空语句
        stmt = ""
        relation = builder._parse_relate_statement(stmt)
        assert relation is None

        # 无效格式
        stmt = "INVALID STATEMENT"
        relation = builder._parse_relate_statement(stmt)
        assert relation is None

    def test_batch_relate_with_db(self, builder, mock_db):
        """测试批量创建关系（使用数据库）"""
        relations = [
            CallRelation(
                caller=f"func{i}",
                callee=f"func{i + 1}",
                weight=1.0,
                relation_type="calls",
                file_path="/test.py",
            )
            for i in range(5)
        ]

        result = builder.batch_relate(relations, batch_size=2)

        assert result["total"] == 5
        assert result["created"] == 5
        assert result["failed"] == 0
        assert result["batches"] == 3  # 5/2 = 3 batches

        # 验证执行了查询
        assert len(mock_db.queries) == 3  # 3 batches

    def test_batch_relate_without_db(self):
        """测试批量创建关系（无数据库，Mock 模式）"""
        builder = RelationBuilder(db=None)

        relations = [
            CallRelation(
                caller="func1",
                callee="func2",
                weight=1.0,
                relation_type="calls",
                file_path="/test.py",
            ),
        ]

        result = builder.batch_relate(relations)

        assert result["total"] == 1
        assert result["created"] == 1
        assert builder.relation_count == 1

    def test_create_relations_filters_self_calls(self, builder, mock_db):
        """测试创建关系时过滤自调用"""
        relations = [
            CallRelation(
                caller="func1",
                callee="func1",  # 自调用
                weight=0.5,
                relation_type="calls",
                file_path="/test.py",
            ),
            CallRelation(
                caller="func1",
                callee="func2",
                weight=1.0,
                relation_type="calls",
                file_path="/test.py",
            ),
        ]

        result = builder.create_relations(relations)

        # 自调用被过滤
        assert result["total"] == 1
        assert result["created"] == 1


class TestRelationBuilderEdgeCases:
    """RelationBuilder 边界情况测试"""

    def test_empty_batch(self):
        """测试空批次"""
        mock_db = MockSurrealDB()
        builder = RelationBuilder(db=mock_db)

        # 空批次不应该执行查询
        builder._create_surrealdb_relations([])
        assert len(mock_db.queries) == 0

    def test_empty_statements(self):
        """测试空语句列表"""
        mock_db = MockSurrealDB()
        builder = RelationBuilder(db=mock_db)

        # 空语句列表也会执行事务（BEGIN/COMMIT）
        builder._execute_relate_statements([])
        # 空语句也会执行事务，所以查询数为 1
        assert len(mock_db.queries) == 1

    def test_special_characters_in_function_names(self):
        """测试函数名中的特殊字符"""
        mock_db = MockSurrealDB()
        builder = RelationBuilder(db=mock_db)

        # 测试各种特殊字符
        test_cases = [
            ("func.name", "atom:func_name"),
            ("func-name", "atom:func_name"),
            ("func.name-with.dash", "atom:func_name_with_dash"),
        ]

        for func_name, expected_id in test_cases:
            atom_id = builder._get_atom_id(func_name)
            assert atom_id == expected_id

    def test_parse_relate_statement_with_special_chars(self):
        """测试解析包含特殊字符的 RELATE 语句"""
        mock_db = MockSurrealDB()
        builder = RelationBuilder(db=mock_db)

        stmt = 'RELATE atom:my_module_func->calls->atom:other_module_func SET weight=1.0, type="calls", file_path="/path/to/file.py"'

        relation = builder._parse_relate_statement(stmt)

        assert relation is not None
        assert relation.caller == "my.module.func"
        assert relation.callee == "other.module.func"
