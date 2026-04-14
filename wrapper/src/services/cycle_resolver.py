"""循环依赖解决策略

定义循环类型分类，实现循环打破策略，支持循环标记和报告生成。
"""

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple

from .cycle_detector import Cycle
from .relation_builder import CallRelation

logger = logging.getLogger(__name__)


class CycleType(Enum):
    """循环类型分类"""

    DIRECT = auto()  # 直接循环: A -> B -> A
    INDIRECT = auto()  # 间接循环: A -> B -> C -> A
    SELF = auto()  # 自调用: A -> A (已在 RelationBuilder 中过滤)
    COMPLEX = auto()  # 复杂循环: 多个循环交织


class CycleAction(Enum):
    """循环处理动作"""

    SKIP = auto()  # 跳过循环关系
    WARN = auto()  # 记录警告但保留
    ERROR = auto()  # 抛出错误
    BREAK = auto()  # 打破循环（删除一条边）


@dataclass
class CycleInfo:
    """循环详细信息

    Attributes:
        cycle: 原始循环对象
        cycle_type: 循环类型
        action: 处理动作
        severity: 严重程度 (1-5, 5 最严重)
        description: 循环描述
        suggested_break: 建议打破的边
    """

    cycle: Cycle
    cycle_type: CycleType
    action: CycleAction
    severity: int
    description: str
    suggested_break: Optional[Tuple[str, str]] = None


@dataclass
class CycleReport:
    """循环依赖报告

    Attributes:
        total_cycles: 总循环数
        cycles_by_type: 按类型分类的循环
        cycles_by_action: 按动作分类的循环
        resolved_cycles: 已解决的循环
        unresolved_cycles: 未解决的循环
        recommendations: 优化建议
    """

    total_cycles: int = 0
    cycles_by_type: Dict[CycleType, List[CycleInfo]] = field(default_factory=dict)
    cycles_by_action: Dict[CycleAction, List[CycleInfo]] = field(default_factory=dict)
    resolved_cycles: List[CycleInfo] = field(default_factory=list)
    unresolved_cycles: List[CycleInfo] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class CycleResolver:
    """循环依赖解决器

    提供循环类型分类、打破策略和报告生成功能。

    Attributes:
        default_action: 默认处理动作
        severity_threshold: 严重程度阈值
    """

    def __init__(
        self,
        default_action: CycleAction = CycleAction.SKIP,
        severity_threshold: int = 3,
    ):
        """初始化循环解决器

        Args:
            default_action: 默认处理动作
            severity_threshold: 严重程度阈值 (1-5)
        """
        self._default_action = default_action
        self._severity_threshold = severity_threshold
        self._logger = logging.getLogger(__name__)
        self._cycle_infos: List[CycleInfo] = []

        self._logger.debug("[CycleResolver] 初始化")

    def classify_cycle(self, cycle: Cycle) -> CycleType:
        """分类循环类型

        Args:
            cycle: 循环对象

        Returns:
            循环类型
        """
        length = cycle.length

        if length == 2:
            return CycleType.DIRECT
        elif length == 1:
            return CycleType.SELF
        elif length <= 4:
            return CycleType.INDIRECT
        else:
            return CycleType.COMPLEX

    def calculate_severity(self, cycle: Cycle, cycle_type: CycleType) -> int:
        """计算循环严重程度

        评分标准:
        - 基础分: 循环长度
        - 直接循环: +1
        - 复杂循环: +2
        - 长度 > 5: +1

        Args:
            cycle: 循环对象
            cycle_type: 循环类型

        Returns:
            严重程度 (1-5)
        """
        severity = min(cycle.length, 3)  # 基础分，最高 3

        if cycle_type == CycleType.DIRECT:
            severity += 1
        elif cycle_type == CycleType.COMPLEX:
            severity += 2

        if cycle.length > 5:
            severity += 1

        return min(severity, 5)  # 最高 5

    def suggest_break_edge(self, cycle: Cycle) -> Optional[Tuple[str, str]]:
        """建议打破循环的边

        策略: 选择权重最低的边，或者选择最后一条边

        Args:
            cycle: 循环对象

        Returns:
            建议打破的边 (caller, callee)，如果没有则返回 None
        """
        if len(cycle.path) < 2:
            return None

        # 返回循环的最后一条边
        # 实际应用中可以根据权重、调用频率等因素选择
        return (cycle.path[-2], cycle.path[-1])

    def resolve_cycles(
        self,
        cycles: List[Cycle],
        action: Optional[CycleAction] = None,
    ) -> List[CycleInfo]:
        """解决循环依赖

        Args:
            cycles: 循环列表
            action: 处理动作，None 则使用默认动作

        Returns:
            循环信息列表
        """
        action = action or self._default_action
        self._cycle_infos = []

        self._logger.info("[CycleResolver] 开始解决 %d 个循环", len(cycles))

        for cycle in cycles:
            cycle_type = self.classify_cycle(cycle)
            severity = self.calculate_severity(cycle, cycle_type)
            suggested_break = self.suggest_break_edge(cycle)

            # 根据严重程度调整动作
            final_action = action
            if severity >= self._severity_threshold:
                if action == CycleAction.WARN:
                    final_action = CycleAction.ERROR

            description = self._generate_description(cycle, cycle_type, severity)

            cycle_info = CycleInfo(
                cycle=cycle,
                cycle_type=cycle_type,
                action=final_action,
                severity=severity,
                description=description,
                suggested_break=suggested_break,
            )

            self._cycle_infos.append(cycle_info)

            self._logger.warning(
                "[CycleResolver] %s [%s] 严重程度=%d 动作=%s",
                description,
                cycle_type.name,
                severity,
                final_action.name,
            )

        return self._cycle_infos

    def _generate_description(self, cycle: Cycle, cycle_type: CycleType, severity: int) -> str:
        """生成循环描述"""
        path_str = " -> ".join(cycle.path)
        return f"{cycle_type.name}循环 ({cycle.length}个节点): {path_str} [严重度:{severity}]"

    def apply_resolution(
        self,
        relations: List[CallRelation],
        cycle_infos: List[CycleInfo],
    ) -> Tuple[List[CallRelation], List[CallRelation], List[CycleInfo]]:
        """应用解决策略

        Args:
            relations: 原始关系列表
            cycle_infos: 循环信息列表

        Returns:
            (保留的关系, 移除的关系, 已解决的循环信息)
        """
        # 收集需要移除的边
        edges_to_remove: Set[Tuple[str, str]] = set()
        resolved_cycles: List[CycleInfo] = []
        unresolved_cycles: List[CycleInfo] = []

        for info in cycle_infos:
            if info.action == CycleAction.SKIP:
                # 跳过整个循环的所有边
                for i in range(len(info.cycle.path) - 1):
                    edges_to_remove.add((info.cycle.path[i], info.cycle.path[i + 1]))
                resolved_cycles.append(info)

            elif info.action == CycleAction.BREAK:
                # 只打破建议的边
                if info.suggested_break:
                    edges_to_remove.add(info.suggested_break)
                resolved_cycles.append(info)

            elif info.action == CycleAction.WARN:
                # 只警告，不移除
                unresolved_cycles.append(info)

            elif info.action == CycleAction.ERROR:
                # 错误级别，不移除但标记为未解决
                unresolved_cycles.append(info)

        # 分离关系
        kept_relations = []
        removed_relations = []

        for rel in relations:
            if (rel.caller, rel.callee) in edges_to_remove:
                removed_relations.append(rel)
            else:
                kept_relations.append(rel)

        self._logger.info(
            "[CycleResolver] 应用解决策略: 保留=%d, 移除=%d, 解决=%d, 未解决=%d",
            len(kept_relations),
            len(removed_relations),
            len(resolved_cycles),
            len(unresolved_cycles),
        )

        return kept_relations, removed_relations, resolved_cycles

    def generate_report(
        self,
        cycle_infos: Optional[List[CycleInfo]] = None,
    ) -> CycleReport:
        """生成循环依赖报告

        Args:
            cycle_infos: 循环信息列表，None 则使用上次解决的结果

        Returns:
            循环依赖报告
        """
        cycle_infos = cycle_infos or self._cycle_infos

        report = CycleReport(total_cycles=len(cycle_infos))

        # 按类型分类
        for info in cycle_infos:
            if info.cycle_type not in report.cycles_by_type:
                report.cycles_by_type[info.cycle_type] = []
            report.cycles_by_type[info.cycle_type].append(info)

            # 按动作分类
            if info.action not in report.cycles_by_action:
                report.cycles_by_action[info.action] = []
            report.cycles_by_action[info.action].append(info)

            # 区分已解决和未解决
            if info.action in (CycleAction.SKIP, CycleAction.BREAK):
                report.resolved_cycles.append(info)
            else:
                report.unresolved_cycles.append(info)

        # 生成建议
        report.recommendations = self._generate_recommendations(report)

        return report

    def _generate_recommendations(self, report: CycleReport) -> List[str]:
        """生成优化建议"""
        recommendations = []

        if report.total_cycles == 0:
            recommendations.append("✅ 未发现循环依赖")
            return recommendations

        # 按类型建议
        if CycleType.COMPLEX in report.cycles_by_type:
            count = len(report.cycles_by_type[CycleType.COMPLEX])
            recommendations.append(f"⚠️ 发现 {count} 个复杂循环，建议重构相关模块")

        if CycleType.DIRECT in report.cycles_by_type:
            count = len(report.cycles_by_type[CycleType.DIRECT])
            recommendations.append(f"💡 {count} 个直接循环可以通过提取公共模块解决")

        # 按动作建议
        if CycleAction.ERROR in report.cycles_by_action:
            count = len(report.cycles_by_action[CycleAction.ERROR])
            recommendations.append(f"🚨 {count} 个循环被标记为错误，需要立即处理")

        if report.unresolved_cycles:
            recommendations.append(f"⏳ {len(report.unresolved_cycles)} 个循环待解决，建议设置处理策略")

        return recommendations

    def get_cycle_infos(self) -> List[CycleInfo]:
        """获取所有循环信息"""
        return list(self._cycle_infos)

    def clear_cycle_infos(self) -> None:
        """清除循环信息"""
        self._cycle_infos.clear()
        self._logger.debug("[CycleResolver] 清除循环信息")

    @property
    def default_action(self) -> CycleAction:
        """默认处理动作"""
        return self._default_action

    @default_action.setter
    def default_action(self, value: CycleAction) -> None:
        """设置默认处理动作"""
        self._default_action = value

    @property
    def severity_threshold(self) -> int:
        """严重程度阈值"""
        return self._severity_threshold

    @severity_threshold.setter
    def severity_threshold(self, value: int) -> None:
        """设置严重程度阈值"""
        self._severity_threshold = max(1, min(value, 5))
