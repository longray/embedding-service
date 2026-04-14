"""RelationBuilder CycleDetector 集成测试

测试范围：
- CycleDetector 集成
- 创建关系前检测循环
- 发现循环时记录警告
- 支持跳过循环关系创建

运行方式：
    uv run pytest tests/test_relation_builder_cycle.py -v
"""

import pytest
import logging
from unittest.mock import MagicMock, patch

from wrapper.src.services.relation_builder import RelationBuilder, CallRelation
from wrapper.src.services.cycle_detector import CycleDetector, Cycle


class TestRelationBuilderCycleIntegration:
    """RelationBuilder CycleDetector 集成测试"""

    @pytest.fixture
    def builder(self):
        """创建 RelationBuilder 实例"""
        return RelationBuilder(db=None, skip_cycles=True)

    def test_cycle_detector_initialized(self, builder):
        """测试 CycleDetector 已初始化"""
        assert builder._cycle_detector is not None
        assert isinstance(builder._cycle_detector, CycleDetector)

    def test_skip_cycles_default(self, builder):
        """测试默认跳过循环"""
        assert builder.skip_cycles is True

    def test_set_skip_cycles(self, builder):
        """测试设置 skip_cycles"""
        builder.skip_cycles = False
        assert builder.skip_cycles is False

        builder.skip_cycles = True
        assert builder.skip_cycles is True

    def test_detect_cycles_no_cycles(self, builder):
        """测试检测无循环"""
        relations = [
            CallRelation("a", "b", 1.0, "calls", "/test.py"),
            CallRelation("b", "c", 1.0, "calls", "/test.py"),
            CallRelation("c", "d", 1.0, "calls", "/test.py"),
        ]

        cycles = builder.detect_cycles(relations)

        assert len(cycles) == 0
        assert builder.has_cycles(relations) is False

    def test_detect_cycles_with_cycle(self, builder):
        """测试检测有循环"""
        relations = [
            CallRelation("a", "b", 1.0, "calls", "/test.py"),
            CallRelation("b", "c", 1.0, "calls", "/test.py"),
            CallRelation("c", "a", 1.0, "calls", "/test.py"),  # 形成循环
        ]

        cycles = builder.detect_cycles(relations)

        assert len(cycles) == 1
        assert cycles[0].length == 3
        assert builder.has_cycles(relations) is True

    def test_detect_cycles_multiple_cycles(self, builder):
        """测试检测多个循环"""
        relations = [
            CallRelation("a", "b", 1.0, "calls", "/test.py"),
            CallRelation("b", "a", 1.0, "calls", "/test.py"),  # 循环 1: a->b->a
            CallRelation("c", "d", 1.0, "calls", "/test.py"),
            CallRelation("d", "c", 1.0, "calls", "/test.py"),  # 循环 2: c->d->c
        ]

        cycles = builder.detect_cycles(relations)

        assert len(cycles) == 2

    def test_filter_cycle_relations_skip_enabled(self, builder):
        """测试过滤循环关系（skip_cycles=True）"""
        relations = [
            CallRelation("a", "b", 1.0, "calls", "/test.py"),
            CallRelation("b", "c", 1.0, "calls", "/test.py"),
            CallRelation("c", "a", 1.0, "calls", "/test.py"),  # 循环
            CallRelation("d", "e", 1.0, "calls", "/test.py"),  # 非循环
        ]

        non_cycle, cycle_rels = builder.filter_cycle_relations(relations)

        assert len(non_cycle) == 1
        assert non_cycle[0].caller == "d"
        assert len(cycle_rels) == 3

    def test_filter_cycle_relations_skip_disabled(self, builder):
        """测试过滤循环关系（skip_cycles=False）"""
        builder.skip_cycles = False

        relations = [
            CallRelation("a", "b", 1.0, "calls", "/test.py"),
            CallRelation("b", "c", 1.0, "calls", "/test.py"),
            CallRelation("c", "a", 1.0, "calls", "/test.py"),  # 循环
        ]

        non_cycle, cycle_rels = builder.filter_cycle_relations(relations)

        # skip_cycles=False 时不过滤
        assert len(non_cycle) == 3
        assert len(cycle_rels) == 0

    def test_filter_cycle_relations_no_cycles(self, builder):
        """测试过滤无循环的关系"""
        relations = [
            CallRelation("a", "b", 1.0, "calls", "/test.py"),
            CallRelation("b", "c", 1.0, "calls", "/test.py"),
        ]

        non_cycle, cycle_rels = builder.filter_cycle_relations(relations)

        assert len(non_cycle) == 2
        assert len(cycle_rels) == 0

    def test_get_cycles(self, builder):
        """测试获取循环"""
        relations = [
            CallRelation("a", "b", 1.0, "calls", "/test.py"),
            CallRelation("b", "a", 1.0, "calls", "/test.py"),
        ]

        builder.detect_cycles(relations)
        cycles = builder.get_cycles()

        assert len(cycles) == 1
        assert cycles[0].length == 2

    def test_clear_cycles(self, builder):
        """测试清除循环"""
        relations = [
            CallRelation("a", "b", 1.0, "calls", "/test.py"),
            CallRelation("b", "a", 1.0, "calls", "/test.py"),
        ]

        builder.detect_cycles(relations)
        assert len(builder.get_cycles()) == 1

        builder.clear_cycles()
        assert len(builder.get_cycles()) == 0

    def test_create_relations_with_cycles_filtered(self, builder):
        """测试创建关系时过滤循环"""
        relations = [
            CallRelation("a", "b", 1.0, "calls", "/test.py"),
            CallRelation("b", "c", 1.0, "calls", "/test.py"),
            CallRelation("c", "a", 1.0, "calls", "/test.py"),  # 循环
        ]

        # 手动过滤循环
        non_cycle, _ = builder.filter_cycle_relations(relations)

        # 只创建非循环关系
        result = builder.create_relations(non_cycle)

        assert result["total"] == 0  # 所有关系都是循环，被过滤后为空

    def test_create_relations_with_cycles_not_filtered(self, builder):
        """测试创建关系时不过滤循环"""
        builder.skip_cycles = False

        relations = [
            CallRelation("a", "b", 1.0, "calls", "/test.py"),
            CallRelation("b", "a", 1.0, "calls", "/test.py"),  # 循环
        ]

        # 不过滤，直接创建
        result = builder.create_relations(relations)

        assert result["total"] == 2  # 创建所有关系


class TestRelationBuilderCycleEdgeCases:
    """RelationBuilder 循环检测边界情况测试"""

    def test_empty_relations(self):
        """测试空关系列表"""
        builder = RelationBuilder(db=None, skip_cycles=True)

        cycles = builder.detect_cycles([])
        assert len(cycles) == 0

        non_cycle, cycle_rels = builder.filter_cycle_relations([])
        assert len(non_cycle) == 0
        assert len(cycle_rels) == 0

    def test_single_relation_no_cycle(self):
        """测试单条关系无循环"""
        builder = RelationBuilder(db=None, skip_cycles=True)

        relations = [
            CallRelation("a", "b", 1.0, "calls", "/test.py"),
        ]

        cycles = builder.detect_cycles(relations)
        assert len(cycles) == 0

    def test_self_call_filtered_by_create_relations(self):
        """测试自调用被 create_relations 过滤"""
        builder = RelationBuilder(db=None, skip_cycles=True)

        relations = [
            CallRelation("a", "a", 0.5, "calls", "/test.py"),  # 自调用
        ]

        # create_relations 会过滤自调用
        result = builder.create_relations(relations)

        assert result["total"] == 0  # 自调用被过滤

    def test_complex_cycle(self):
        """测试复杂循环"""
        builder = RelationBuilder(db=None, skip_cycles=True)

        # a -> b -> c -> d -> b (循环: b->c->d->b)
        relations = [
            CallRelation("a", "b", 1.0, "calls", "/test.py"),
            CallRelation("b", "c", 1.0, "calls", "/test.py"),
            CallRelation("c", "d", 1.0, "calls", "/test.py"),
            CallRelation("d", "b", 1.0, "calls", "/test.py"),  # 回到 b，形成循环
        ]

        cycles = builder.detect_cycles(relations)

        assert len(cycles) == 1
        assert cycles[0].length == 3  # b->c->d->b

    def test_multiple_disjoint_cycles(self):
        """测试多个不相交的循环"""
        builder = RelationBuilder(db=None, skip_cycles=True)

        relations = [
            # 循环 1
            CallRelation("a", "b", 1.0, "calls", "/test.py"),
            CallRelation("b", "a", 1.0, "calls", "/test.py"),
            # 循环 2
            CallRelation("x", "y", 1.0, "calls", "/test.py"),
            CallRelation("y", "z", 1.0, "calls", "/test.py"),
            CallRelation("z", "x", 1.0, "calls", "/test.py"),
        ]

        cycles = builder.detect_cycles(relations)

        assert len(cycles) == 2

    def test_cycle_with_non_cycle_relations(self):
        """测试混合循环和非循环关系"""
        builder = RelationBuilder(db=None, skip_cycles=True)

        relations = [
            CallRelation("a", "b", 1.0, "calls", "/test.py"),
            CallRelation("b", "c", 1.0, "calls", "/test.py"),
            CallRelation("c", "a", 1.0, "calls", "/test.py"),  # 循环
            CallRelation("x", "y", 1.0, "calls", "/test.py"),  # 非循环
            CallRelation("y", "z", 1.0, "calls", "/test.py"),  # 非循环
        ]

        non_cycle, cycle_rels = builder.filter_cycle_relations(relations)

        assert len(non_cycle) == 2  # x->y, y->z
        assert len(cycle_rels) == 3  # a->b, b->c, c->a
