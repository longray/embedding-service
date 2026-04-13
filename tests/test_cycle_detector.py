"""循环检测器测试"""

import pytest

from wrapper.src.services.cycle_detector import Cycle, CycleDetector
from wrapper.src.services.relation_builder import CallRelation


class TestCycleDetector:
    """CycleDetector 单元测试"""

    @pytest.fixture
    def detector(self):
        """创建 CycleDetector 实例"""
        return CycleDetector()

    def test_initialization(self, detector):
        """测试初始化"""
        assert detector.get_cycle_count() == 0
        assert detector.get_cycles() == []

    def test_no_cycles(self, detector):
        """测试无循环图"""
        relations = [
            CallRelation("a", "b", 1.0, "calls", "/test.py"),
            CallRelation("b", "c", 1.0, "calls", "/test.py"),
        ]

        cycles = detector.detect_cycles(relations)

        assert len(cycles) == 0

    def test_simple_cycle(self, detector):
        """测试简单循环 A->B->A"""
        relations = [
            CallRelation("a", "b", 1.0, "calls", "/test.py"),
            CallRelation("b", "a", 1.0, "calls", "/test.py"),
        ]

        cycles = detector.detect_cycles(relations)

        assert len(cycles) == 1
        assert cycles[0].length == 2
        assert "a" in cycles[0].path
        assert "b" in cycles[0].path

    def test_complex_cycle(self, detector):
        """测试复杂循环 A->B->C->A"""
        relations = [
            CallRelation("a", "b", 1.0, "calls", "/test.py"),
            CallRelation("b", "c", 1.0, "calls", "/test.py"),
            CallRelation("c", "a", 1.0, "calls", "/test.py"),
        ]

        cycles = detector.detect_cycles(relations)

        assert len(cycles) == 1
        assert cycles[0].length == 3
        assert "a" in cycles[0].path
        assert "b" in cycles[0].path
        assert "c" in cycles[0].path

    def test_multiple_cycles(self, detector):
        """测试多个循环"""
        relations = [
            CallRelation("a", "b", 1.0, "calls", "/test.py"),
            CallRelation("b", "a", 1.0, "calls", "/test.py"),
            CallRelation("c", "d", 1.0, "calls", "/test.py"),
            CallRelation("d", "c", 1.0, "calls", "/test.py"),
        ]

        cycles = detector.detect_cycles(relations)

        assert len(cycles) == 2

    def test_self_call(self, detector):
        """测试自调用"""
        relations = [
            CallRelation("a", "a", 1.0, "calls", "/test.py"),
        ]

        cycles = detector.detect_cycles(relations)

        # 自调用在图论中也是循环（长度为1）
        assert len(cycles) == 1
        assert cycles[0].length == 1

    def test_empty_relations(self, detector):
        """测试空关系列表"""
        cycles = detector.detect_cycles([])

        assert len(cycles) == 0

    def test_has_cycles(self, detector):
        """测试 has_cycles 方法"""
        # 无循环
        relations_no_cycle = [
            CallRelation("a", "b", 1.0, "calls", "/test.py"),
        ]
        assert detector.has_cycles(relations_no_cycle) is False

        # 有循环
        relations_with_cycle = [
            CallRelation("a", "b", 1.0, "calls", "/test.py"),
            CallRelation("b", "a", 1.0, "calls", "/test.py"),
        ]
        assert detector.has_cycles(relations_with_cycle) is True

    def test_clear_cycles(self, detector):
        """测试清除循环"""
        relations = [
            CallRelation("a", "b", 1.0, "calls", "/test.py"),
            CallRelation("b", "a", 1.0, "calls", "/test.py"),
        ]

        detector.detect_cycles(relations)
        assert detector.get_cycle_count() == 1

        detector.clear_cycles()
        assert detector.get_cycle_count() == 0

    def test_cycle_path_format(self, detector):
        """测试循环路径格式"""
        relations = [
            CallRelation("a", "b", 1.0, "calls", "/test.py"),
            CallRelation("b", "a", 1.0, "calls", "/test.py"),
        ]

        cycles = detector.detect_cycles(relations)

        assert len(cycles) == 1
        # 路径应该包含起始节点两次，形成闭环
        assert cycles[0].path[0] == cycles[0].path[-1]


class TestCycleDetectorPerformance:
    """性能测试"""

    def test_large_graph_no_cycles(self):
        """测试大图无循环"""
        detector = CycleDetector()

        # 创建链式结构：a->b->c->...->z
        relations = [CallRelation(chr(97 + i), chr(97 + i + 1), 1.0, "calls", "/test.py") for i in range(25)]

        cycles = detector.detect_cycles(relations)

        assert len(cycles) == 0

    def test_large_graph_with_cycle(self):
        """测试大图有循环"""
        detector = CycleDetector()

        # 创建链式结构，最后形成循环
        relations = [CallRelation(chr(97 + i), chr(97 + i + 1), 1.0, "calls", "/test.py") for i in range(25)]
        # 添加循环：z->a
        relations.append(CallRelation("z", "a", 1.0, "calls", "/test.py"))

        cycles = detector.detect_cycles(relations)

        assert len(cycles) == 1
        assert cycles[0].length == 26


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
