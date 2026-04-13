"""权重计算器测试"""

import pytest

from wrapper.src.services.weight_calculator import WeightCalculator, WeightFactors


class TestWeightCalculator:
    """WeightCalculator 单元测试"""

    @pytest.fixture
    def calculator(self):
        """创建 WeightCalculator 实例"""
        return WeightCalculator()

    def test_initialization(self, calculator):
        """测试初始化"""
        assert calculator.weight_count == 0
        assert calculator.get_all_weights() == {}

    def test_calculate_weight_base(self, calculator):
        """测试基础权重"""
        factors = WeightFactors()
        weight = calculator.calculate_weight(factors)

        assert weight == 0.5

    def test_calculate_weight_with_frequency(self, calculator):
        """测试频率加成"""
        factors = WeightFactors(frequency=10)
        weight = calculator.calculate_weight(factors)

        # 基础 0.5 + 频率加成 min(10*0.05, 0.3) = 0.3
        assert weight == 0.8

    def test_calculate_weight_with_complexity(self, calculator):
        """测试复杂度加成"""
        factors = WeightFactors(complexity=10)
        weight = calculator.calculate_weight(factors)

        # 基础 0.5 + 复杂度加成 min(10*0.02, 0.2) = 0.2
        assert weight == 0.7

    def test_calculate_weight_with_params(self, calculator):
        """测试参数加成"""
        factors = WeightFactors(param_count=10)
        weight = calculator.calculate_weight(factors)

        # 基础 0.5 + 参数加成 min(10*0.01, 0.1) = 0.1
        assert weight == 0.6

    def test_calculate_weight_cross_file(self, calculator):
        """测试跨文件加成"""
        factors = WeightFactors(is_cross_file=True)
        weight = calculator.calculate_weight(factors)

        # 基础 0.5 + 跨文件加成 0.1
        assert weight == 0.6

    def test_calculate_weight_combined(self, calculator):
        """测试组合因子"""
        factors = WeightFactors(
            frequency=5,
            complexity=5,
            param_count=5,
            is_cross_file=True,
        )
        weight = calculator.calculate_weight(factors)

        # 基础 0.5 + 频率 0.25 + 复杂度 0.1 + 参数 0.05 + 跨文件 0.1 = 1.0
        assert weight == 1.0

    def test_calculate_weight_normalization(self, calculator):
        """测试归一化"""
        factors = WeightFactors(
            frequency=100,  # 会被限制
            complexity=100,  # 会被限制
            param_count=100,  # 会被限制
            is_cross_file=True,
        )
        weight = calculator.calculate_weight(factors)

        # 应该被归一化到 1.0
        assert weight == 1.0

    def test_calculate_weight_from_relation(self, calculator):
        """测试从关系计算权重"""
        weight = calculator.calculate_weight_from_relation(
            caller="func_a",
            callee="func_b",
            frequency=5,
            complexity=3,
            param_count=2,
            is_cross_file=False,
        )

        assert 0 <= weight <= 1
        assert calculator.get_weight("func_a->func_b") == weight

    def test_save_and_get_weight(self, calculator):
        """测试保存和获取权重"""
        calculator.save_weight("a->b", 0.8)

        weight = calculator.get_weight("a->b")

        assert weight == 0.8

    def test_get_weight_nonexistent(self, calculator):
        """测试获取不存在的权重"""
        weight = calculator.get_weight("nonexistent")

        assert weight is None

    def test_remove_weight(self, calculator):
        """测试删除权重"""
        calculator.save_weight("a->b", 0.8)

        result = calculator.remove_weight("a->b")

        assert result is True
        assert calculator.get_weight("a->b") is None

    def test_remove_weight_nonexistent(self, calculator):
        """测试删除不存在的权重"""
        result = calculator.remove_weight("nonexistent")

        assert result is False

    def test_clear_weights(self, calculator):
        """测试清除权重"""
        calculator.save_weight("a->b", 0.8)
        calculator.save_weight("c->d", 0.9)

        calculator.clear_weights()

        assert calculator.weight_count == 0

    def test_get_top_relations(self, calculator):
        """测试获取权重最高的关系"""
        calculator.save_weight("a->b", 0.5)
        calculator.save_weight("c->d", 0.9)
        calculator.save_weight("e->f", 0.7)

        top = calculator.get_top_relations(n=2)

        assert len(top) == 2
        assert "c->d" in top  # 最高
        assert "e->f" in top  # 第二高

    def test_weight_range(self, calculator):
        """测试权重范围"""
        # 最小权重
        factors_min = WeightFactors()
        weight_min = calculator.calculate_weight(factors_min)
        assert weight_min >= 0

        # 最大权重
        factors_max = WeightFactors(
            frequency=1000,
            complexity=1000,
            param_count=1000,
            is_cross_file=True,
        )
        weight_max = calculator.calculate_weight(factors_max)
        assert weight_max <= 1


class TestWeightFactors:
    """WeightFactors 测试"""

    def test_default_values(self):
        """测试默认值"""
        factors = WeightFactors()

        assert factors.frequency == 0
        assert factors.complexity == 0
        assert factors.param_count == 0
        assert factors.is_cross_file is False

    def test_custom_values(self):
        """测试自定义值"""
        factors = WeightFactors(
            frequency=10,
            complexity=5,
            param_count=3,
            is_cross_file=True,
        )

        assert factors.frequency == 10
        assert factors.complexity == 5
        assert factors.param_count == 3
        assert factors.is_cross_file is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
