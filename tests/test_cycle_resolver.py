"""CycleResolver 测试

测试范围：
- 循环类型分类
- 循环打破策略
- 循环标记（跳过/警告/错误）
- 循环依赖报告生成

运行方式：
    uv run pytest tests/test_cycle_resolver.py -v
"""

import pytest
from unittest.mock import MagicMock, patch

from wrapper.src.services.cycle_resolver import (
    CycleResolver,
    CycleType,
    CycleAction,
    CycleInfo,
    CycleReport,
)
from wrapper.src.services.cycle_detector import Cycle
from wrapper.src.services.relation_builder import CallRelation


class TestCycleResolverClassification:
    """循环类型分类测试"""

    @pytest.fixture
    def resolver(self):
        """创建 CycleResolver 实例"""
        return CycleResolver()

    def test_classify_direct_cycle(self, resolver):
        """测试直接循环分类"""
        cycle = Cycle(path=["a", "b", "a"], length=2)
        cycle_type = resolver.classify_cycle(cycle)
        assert cycle_type == CycleType.DIRECT

    def test_classify_indirect_cycle(self, resolver):
        """测试间接循环分类"""
        cycle = Cycle(path=["a", "b", "c", "a"], length=3)
        cycle_type = resolver.classify_cycle(cycle)
        assert cycle_type == CycleType.INDIRECT

    def test_classify_self_cycle(self, resolver):
        """测试自调用循环分类"""
        cycle = Cycle(path=["a", "a"], length=1)
        cycle_type = resolver.classify_cycle(cycle)
        assert cycle_type == CycleType.SELF

    def test_classify_complex_cycle(self, resolver):
        """测试复杂循环分类"""
        cycle = Cycle(path=["a", "b", "c", "d", "e", "a"], length=5)
        cycle_type = resolver.classify_cycle(cycle)
        assert cycle_type == CycleType.COMPLEX


class TestCycleResolverSeverity:
    """严重程度计算测试"""

    @pytest.fixture
    def resolver(self):
        """创建 CycleResolver 实例"""
        return CycleResolver()

    def test_severity_direct_cycle(self, resolver):
        """测试直接循环严重程度"""
        cycle = Cycle(path=["a", "b", "a"], length=2)
        severity = resolver.calculate_severity(cycle, CycleType.DIRECT)
        assert severity == 3  # 基础2 + 直接1

    def test_severity_indirect_cycle(self, resolver):
        """测试间接循环严重程度"""
        cycle = Cycle(path=["a", "b", "c", "a"], length=3)
        severity = resolver.calculate_severity(cycle, CycleType.INDIRECT)
        assert severity == 3  # 基础3

    def test_severity_complex_cycle(self, resolver):
        """测试复杂循环严重程度"""
        cycle = Cycle(path=["a", "b", "c", "d", "e", "a"], length=5)
        severity = resolver.calculate_severity(cycle, CycleType.COMPLEX)
        assert severity == 5  # 基础3 + 复杂2

    def test_severity_long_cycle(self, resolver):
        """测试长循环严重程度"""
        cycle = Cycle(path=["a", "b", "c", "d", "e", "f", "g", "a"], length=7)
        severity = resolver.calculate_severity(cycle, CycleType.INDIRECT)
        assert severity == 4  # 基础3 + 长度>5加1 = 4


class TestCycleResolverBreakSuggestion:
    """循环打破建议测试"""

    @pytest.fixture
    def resolver(self):
        """创建 CycleResolver 实例"""
        return CycleResolver()

    def test_suggest_break_edge(self, resolver):
        """测试建议打破边"""
        cycle = Cycle(path=["a", "b", "c", "a"], length=3)
        suggested = resolver.suggest_break_edge(cycle)
        assert suggested == ("c", "a")  # 最后一条边

    def test_suggest_break_edge_short_cycle(self, resolver):
        """测试短循环建议"""
        cycle = Cycle(path=["a", "b", "a"], length=2)
        suggested = resolver.suggest_break_edge(cycle)
        assert suggested == ("b", "a")

    def test_suggest_break_edge_empty(self, resolver):
        """测试空循环建议"""
        cycle = Cycle(path=["a"], length=0)
        suggested = resolver.suggest_break_edge(cycle)
        assert suggested is None


class TestCycleResolverResolution:
    """循环解决测试"""

    @pytest.fixture
    def resolver(self):
        """创建 CycleResolver 实例"""
        return CycleResolver(default_action=CycleAction.SKIP)

    def test_resolve_single_cycle(self, resolver):
        """测试解决单个循环"""
        cycle = Cycle(path=["a", "b", "a"], length=2)
        cycle_infos = resolver.resolve_cycles([cycle])

        assert len(cycle_infos) == 1
        assert cycle_infos[0].cycle_type == CycleType.DIRECT
        assert cycle_infos[0].action == CycleAction.SKIP

    def test_resolve_multiple_cycles(self, resolver):
        """测试解决多个循环"""
        cycles = [
            Cycle(path=["a", "b", "a"], length=2),
            Cycle(path=["c", "d", "e", "c"], length=3),
        ]
        cycle_infos = resolver.resolve_cycles(cycles)

        assert len(cycle_infos) == 2
        assert cycle_infos[0].cycle_type == CycleType.DIRECT
        assert cycle_infos[1].cycle_type == CycleType.INDIRECT

    def test_resolve_with_custom_action(self, resolver):
        """测试自定义动作"""
        resolver.severity_threshold = 5  # 提高阈值，避免 WARN 转为 ERROR
        cycle = Cycle(path=["a", "b", "a"], length=2)
        cycle_infos = resolver.resolve_cycles([cycle], action=CycleAction.WARN)

        assert cycle_infos[0].action == CycleAction.WARN

    def test_resolve_high_severity_error(self, resolver):
        """测试高严重程度转为错误"""
        resolver.severity_threshold = 3
        # 创建一个严重程度为4的复杂循环
        cycle = Cycle(path=["a", "b", "c", "d", "e", "a"], length=5)
        cycle_infos = resolver.resolve_cycles([cycle], action=CycleAction.WARN)

        # 严重程度 >= threshold，WARN 转为 ERROR
        assert cycle_infos[0].action == CycleAction.ERROR


class TestCycleResolverApply:
    """应用解决策略测试"""

    @pytest.fixture
    def resolver(self):
        """创建 CycleResolver 实例"""
        return CycleResolver()

    def test_apply_skip_action(self, resolver):
        """测试 SKIP 动作"""
        relations = [
            CallRelation("a", "b", 1.0, "calls", "/test.py"),
            CallRelation("b", "a", 1.0, "calls", "/test.py"),  # 循环
            CallRelation("c", "d", 1.0, "calls", "/test.py"),  # 非循环
        ]

        cycle = Cycle(path=["a", "b", "a"], length=2)
        cycle_info = CycleInfo(
            cycle=cycle,
            cycle_type=CycleType.DIRECT,
            action=CycleAction.SKIP,
            severity=3,
            description="test",
        )

        kept, removed, resolved = resolver.apply_resolution(relations, [cycle_info])

        assert len(kept) == 1
        assert kept[0].caller == "c"
        assert len(removed) == 2  # a->b 和 b->a
        assert len(resolved) == 1

    def test_apply_break_action(self, resolver):
        """测试 BREAK 动作"""
        relations = [
            CallRelation("a", "b", 1.0, "calls", "/test.py"),
            CallRelation("b", "a", 1.0, "calls", "/test.py"),
        ]

        cycle = Cycle(path=["a", "b", "a"], length=2)
        cycle_info = CycleInfo(
            cycle=cycle,
            cycle_type=CycleType.DIRECT,
            action=CycleAction.BREAK,
            severity=3,
            description="test",
            suggested_break=("b", "a"),
        )

        kept, removed, resolved = resolver.apply_resolution(relations, [cycle_info])

        assert len(kept) == 1
        assert len(removed) == 1
        assert removed[0].caller == "b"
        assert len(resolved) == 1

    def test_apply_warn_action(self, resolver):
        """测试 WARN 动作"""
        relations = [
            CallRelation("a", "b", 1.0, "calls", "/test.py"),
            CallRelation("b", "a", 1.0, "calls", "/test.py"),
        ]

        cycle = Cycle(path=["a", "b", "a"], length=2)
        cycle_info = CycleInfo(
            cycle=cycle,
            cycle_type=CycleType.DIRECT,
            action=CycleAction.WARN,
            severity=3,
            description="test",
        )

        kept, removed, resolved = resolver.apply_resolution(relations, [cycle_info])

        # WARN 不移除关系
        assert len(kept) == 2
        assert len(removed) == 0
        assert len(resolved) == 0  # 未解决


class TestCycleResolverReport:
    """报告生成测试"""

    @pytest.fixture
    def resolver(self):
        """创建 CycleResolver 实例"""
        return CycleResolver()

    @pytest.fixture
    def sample_cycles(self):
        """创建示例循环"""
        return [
            CycleInfo(
                cycle=Cycle(path=["a", "b", "a"], length=2),
                cycle_type=CycleType.DIRECT,
                action=CycleAction.SKIP,
                severity=3,
                description="Direct cycle",
            ),
            CycleInfo(
                cycle=Cycle(path=["c", "d", "e", "c"], length=3),
                cycle_type=CycleType.INDIRECT,
                action=CycleAction.WARN,
                severity=3,
                description="Indirect cycle",
            ),
            CycleInfo(
                cycle=Cycle(path=["x", "y", "z", "w", "x"], length=4),
                cycle_type=CycleType.COMPLEX,
                action=CycleAction.ERROR,
                severity=5,
                description="Complex cycle",
            ),
        ]

    def test_generate_report(self, resolver, sample_cycles):
        """测试报告生成"""
        report = resolver.generate_report(sample_cycles)

        assert report.total_cycles == 3
        assert len(report.cycles_by_type) == 3
        assert len(report.cycles_by_action) == 3
        assert len(report.resolved_cycles) == 1  # SKIP
        assert len(report.unresolved_cycles) == 2  # WARN, ERROR

    def test_report_recommendations(self, resolver, sample_cycles):
        """测试报告建议"""
        report = resolver.generate_report(sample_cycles)

        assert len(report.recommendations) > 0
        # 应该包含复杂循环建议
        assert any("复杂循环" in r for r in report.recommendations)
        # 应该包含错误建议
        assert any("错误" in r for r in report.recommendations)

    def test_report_no_cycles(self, resolver):
        """测试无循环报告"""
        report = resolver.generate_report([])

        assert report.total_cycles == 0
        assert len(report.recommendations) == 1
        assert "未发现" in report.recommendations[0]


class TestCycleResolverProperties:
    """属性测试"""

    def test_default_action_property(self):
        """测试 default_action 属性"""
        resolver = CycleResolver(default_action=CycleAction.SKIP)
        assert resolver.default_action == CycleAction.SKIP

        resolver.default_action = CycleAction.WARN
        assert resolver.default_action == CycleAction.WARN

    def test_severity_threshold_property(self):
        """测试 severity_threshold 属性"""
        resolver = CycleResolver(severity_threshold=3)
        assert resolver.severity_threshold == 3

        resolver.severity_threshold = 5
        assert resolver.severity_threshold == 5

        # 测试边界
        resolver.severity_threshold = 10
        assert resolver.severity_threshold == 5  # 最高5

        resolver.severity_threshold = 0
        assert resolver.severity_threshold == 1  # 最低1


class TestCycleResolverEdgeCases:
    """边界情况测试"""

    @pytest.fixture
    def resolver(self):
        """创建 CycleResolver 实例"""
        return CycleResolver()

    def test_empty_cycles(self, resolver):
        """测试空循环列表"""
        cycle_infos = resolver.resolve_cycles([])
        assert len(cycle_infos) == 0

    def test_clear_cycle_infos(self, resolver):
        """测试清除循环信息"""
        cycle = Cycle(path=["a", "b", "a"], length=2)
        resolver.resolve_cycles([cycle])
        assert len(resolver.get_cycle_infos()) == 1

        resolver.clear_cycle_infos()
        assert len(resolver.get_cycle_infos()) == 0

    def test_generate_description(self, resolver):
        """测试描述生成"""
        cycle = Cycle(path=["a", "b", "c", "a"], length=3)
        desc = resolver._generate_description(cycle, CycleType.INDIRECT, 3)
        assert "INDIRECT循环" in desc
        assert "3个节点" in desc
        assert "a -> b -> c -> a" in desc
        assert "[严重度:3]" in desc
