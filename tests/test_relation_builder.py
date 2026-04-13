"""关系构建器测试"""

import pytest

from wrapper.src.services.relation_builder import CallRelation, RelationBuilder


class TestRelationBuilder:
    """RelationBuilder 单元测试"""

    @pytest.fixture
    def builder(self):
        """创建 RelationBuilder 实例"""
        return RelationBuilder()

    def test_initialization(self, builder):
        """测试初始化"""
        assert builder.relation_count == 0
        assert builder.get_relations() == []

    def test_extract_calls(self, builder):
        """测试调用关系提取"""
        ast = {
            "symbols": [
                {"type": "function", "name": "main"},
                {"type": "function", "name": "helper"},
            ],
            "root_node": {
                "type": "module",
                "children": [
                    {
                        "type": "call_expression",
                        "children": [{"type": "identifier", "text": "helper"}],
                    }
                ],
            },
        }

        relations = builder.extract_calls(ast, "/test.py")

        assert len(relations) > 0
        assert any(r.callee == "helper" for r in relations)

    def test_self_call_filtering(self, builder):
        """测试自调用过滤"""
        relations = [
            CallRelation("func_a", "func_b", 1.0, "calls", "/test.py"),
            CallRelation("func_a", "func_a", 1.0, "calls", "/test.py"),  # 自调用
        ]

        result = builder.create_relations(relations)

        assert result["total"] == 1  # 过滤后只剩 1 条

    def test_batch_relate(self, builder):
        """测试批量创建关系"""
        relations = [CallRelation(f"caller_{i}", f"callee_{i}", 1.0, "calls", "/test.py") for i in range(250)]

        result = builder.batch_relate(relations, batch_size=100)

        assert result["total"] == 250
        assert result["created"] == 250
        assert result["batches"] == 3

    def test_calculate_weight(self, builder):
        """测试权重计算"""
        # 递归调用
        weight = builder._calculate_weight("func_a", "func_a", "/test.py")
        assert weight == 0.5

        # 普通调用
        weight = builder._calculate_weight("func_a", "func_b", "/test.py")
        assert weight == 1.0

    def test_get_relations(self, builder):
        """测试获取关系"""
        relations = [
            CallRelation("a", "b", 1.0, "calls", "/test.py"),
        ]

        builder.batch_relate(relations)

        result = builder.get_relations()

        assert len(result) == 1
        assert result[0].caller == "a"

    def test_clear_relations(self, builder):
        """测试清除关系"""
        relations = [CallRelation("a", "b", 1.0, "calls", "/test.py")]
        builder.batch_relate(relations)

        builder.clear_relations()

        assert builder.relation_count == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
