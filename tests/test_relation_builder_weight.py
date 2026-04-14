"""RelationBuilder WeightCalculator 集成测试

测试范围：
- WeightCalculator 集成
- 权重计算
- 权重持久化

运行方式：
    uv run pytest tests/test_relation_builder_weight.py -v
"""

import pytest
from unittest.mock import MagicMock

from wrapper.src.services.relation_builder import RelationBuilder, CallRelation
from wrapper.src.services.weight_calculator import WeightCalculator, WeightFactors


class TestRelationBuilderWeightIntegration:
    """RelationBuilder WeightCalculator 集成测试"""

    @pytest.fixture
    def builder(self):
        """创建 RelationBuilder 实例"""
        return RelationBuilder(db=None, skip_cycles=True)

    def test_weight_calculator_initialized(self, builder):
        """测试 WeightCalculator 已初始化"""
        assert builder.weight_calculator is not None
        assert isinstance(builder.weight_calculator, WeightCalculator)

    def test_calculate_weight_uses_weight_calculator(self, builder):
        """测试 _calculate_weight 使用 WeightCalculator"""
        weight = builder._calculate_weight("func_a", "func_b", "/test.py")

        # 验证权重在合理范围内
        assert 0.0 <= weight <= 1.0

        # 验证权重已保存
        relation_id = "func_a->func_b"
        saved_weight = builder.weight_calculator.get_weight(relation_id)
        assert saved_weight is not None
        assert saved_weight == weight

    def test_calculate_weight_self_call(self, builder):
        """测试自调用权重"""
        weight = builder._calculate_weight("func_a", "func_a", "/test.py")

        # 自调用应该返回 0.5
        assert weight == 0.5

    def test_calculate_weight_different_relations(self, builder):
        """测试不同关系的权重"""
        weight1 = builder._calculate_weight("func_a", "func_b", "/test.py")
        weight2 = builder._calculate_weight("func_b", "func_c", "/test.py")
        weight3 = builder._calculate_weight("func_a", "func_c", "/test.py")

        # 所有权重都应该在 [0, 1] 范围内
        assert 0.0 <= weight1 <= 1.0
        assert 0.0 <= weight2 <= 1.0
        assert 0.0 <= weight3 <= 1.0

        # 验证权重已保存
        assert builder.weight_calculator.get_weight("func_a->func_b") == weight1
        assert builder.weight_calculator.get_weight("func_b->func_c") == weight2
        assert builder.weight_calculator.get_weight("func_a->func_c") == weight3

    def test_weight_calculator_property(self, builder):
        """测试 weight_calculator property"""
        wc = builder.weight_calculator

        # 应该返回同一个实例
        assert wc is builder._weight_calculator

        # 可以使用 WeightCalculator 的方法
        factors = WeightFactors(frequency=5, complexity=3, param_count=2, is_cross_file=True)
        weight = wc.calculate_weight(factors)
        assert 0.0 <= weight <= 1.0

    def test_weight_persistence(self, builder):
        """测试权重持久化"""
        # 计算权重
        weight = builder._calculate_weight("func_a", "func_b", "/test.py")

        # 获取保存的权重
        saved_weight = builder.weight_calculator.get_weight("func_a->func_b")
        assert saved_weight == weight

        # 获取所有权重
        all_weights = builder.weight_calculator.get_all_weights()
        assert "func_a->func_b" in all_weights
        assert all_weights["func_a->func_b"] == weight

    def test_weight_count(self, builder):
        """测试权重计数"""
        assert builder.weight_calculator.weight_count == 0

        builder._calculate_weight("func_a", "func_b", "/test.py")
        assert builder.weight_calculator.weight_count == 1

        builder._calculate_weight("func_b", "func_c", "/test.py")
        assert builder.weight_calculator.weight_count == 2

    def test_clear_weights(self, builder):
        """测试清除权重"""
        builder._calculate_weight("func_a", "func_b", "/test.py")
        assert builder.weight_calculator.weight_count == 1

        builder.weight_calculator.clear_weights()
        assert builder.weight_calculator.weight_count == 0
        assert builder.weight_calculator.get_weight("func_a->func_b") is None

    def test_get_top_relations(self, builder):
        """测试获取权重最高的关系"""
        # 创建多个关系
        builder._calculate_weight("func_a", "func_b", "/test.py")
        builder._calculate_weight("func_b", "func_c", "/test.py")
        builder._calculate_weight("func_c", "func_d", "/test.py")

        # 获取权重最高的关系
        top_relations = builder.weight_calculator.get_top_relations(n=2)
        assert len(top_relations) == 2

        # 验证返回的是字典
        assert isinstance(top_relations, dict)

    def test_weight_calculation_with_factors(self, builder):
        """测试使用不同因子计算权重"""
        # 使用 WeightCalculator 直接计算
        factors_low = WeightFactors(frequency=1, complexity=1, param_count=0, is_cross_file=False)
        factors_high = WeightFactors(frequency=10, complexity=10, param_count=5, is_cross_file=True)

        weight_low = builder.weight_calculator.calculate_weight(factors_low)
        weight_high = builder.weight_calculator.calculate_weight(factors_high)

        # 高因子应该产生更高的权重
        assert weight_low < weight_high
        assert 0.0 <= weight_low <= 1.0
        assert 0.0 <= weight_high <= 1.0


class TestRelationBuilderWeightEdgeCases:
    """权重计算边界情况测试"""

    def test_weight_with_empty_relation_id(self):
        """测试空关系 ID"""
        builder = RelationBuilder(db=None, skip_cycles=True)

        # 保存权重
        builder.weight_calculator.save_weight("", 0.5)
        assert builder.weight_calculator.get_weight("") == 0.5

    def test_weight_with_special_characters(self):
        """测试特殊字符关系 ID"""
        builder = RelationBuilder(db=None, skip_cycles=True)

        # 保存权重
        builder.weight_calculator.save_weight("func::test->other", 0.7)
        assert builder.weight_calculator.get_weight("func::test->other") == 0.7

    def test_remove_weight(self):
        """测试删除权重"""
        builder = RelationBuilder(db=None, skip_cycles=True)

        builder.weight_calculator.save_weight("test->other", 0.5)
        assert builder.weight_calculator.weight_count == 1

        # 删除权重
        result = builder.weight_calculator.remove_weight("test->other")
        assert result is True
        assert builder.weight_calculator.weight_count == 0

        # 再次删除应该返回 False
        result = builder.weight_calculator.remove_weight("test->other")
        assert result is False

    def test_calculate_weight_from_relation(self):
        """测试从关系信息计算权重"""
        builder = RelationBuilder(db=None, skip_cycles=True)

        weight = builder.weight_calculator.calculate_weight_from_relation(
            caller="func_a",
            callee="func_b",
            frequency=5,
            complexity=3,
            param_count=2,
            is_cross_file=True,
        )

        assert 0.0 <= weight <= 1.0
        assert builder.weight_calculator.get_weight("func_a->func_b") == weight
