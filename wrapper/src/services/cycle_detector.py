"""循环检测器

使用 DFS 算法检测代码中的循环依赖（circular dependencies）。
时间复杂度: O(V+E)
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set

from .relation_builder import CallRelation

logger = logging.getLogger(__name__)


@dataclass
class Cycle:
    """循环信息

    Attributes:
        path: 循环路径（函数名列表）
        length: 循环长度
    """

    path: List[str]
    length: int


class CycleDetector:
    """循环检测器

    使用 DFS 算法检测调用图中的循环依赖。
    使用三色标记法：
    - 白色：未访问
    - 灰色：正在访问（在递归栈中）
    - 黑色：已访问完成

    Attributes:
        cycles: 检测到的循环列表
    """

    def __init__(self):
        """初始化循环检测器"""
        self._cycles: List[Cycle] = []
        self._logger = logging.getLogger(__name__)

        self._logger.debug("[CycleDetector] 初始化")

    def detect_cycles(self, relations: List[CallRelation]) -> List[Cycle]:
        """检测循环

        从调用关系中检测循环依赖。

        Args:
            relations: 调用关系列表

        Returns:
            检测到的循环列表
        """
        if not relations:
            return []

        # 构建有向图
        graph = self._build_graph(relations)

        self._logger.debug(
            "[CycleDetector] 开始检测: nodes=%d, edges=%d",
            len(graph),
            len(relations),
        )

        # 初始化
        self._cycles = []
        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        path: List[str] = []

        # 对每个未访问的节点进行 DFS
        for node in graph:
            if node not in visited:
                self._dfs(node, graph, visited, rec_stack, path)

        self._logger.info(
            "[CycleDetector] 检测完成: 发现 %d 个循环",
            len(self._cycles),
        )

        return self._cycles

    def _build_graph(self, relations: List[CallRelation]) -> Dict[str, List[str]]:
        """构建有向图

        从调用关系构建邻接表表示的有向图。

        Args:
            relations: 调用关系列表

        Returns:
            邻接表表示的图
        """
        graph = defaultdict(list)

        for rel in relations:
            graph[rel.caller].append(rel.callee)

        return graph

    def _dfs(
        self,
        node: str,
        graph: Dict[str, List[str]],
        visited: Set[str],
        rec_stack: Set[str],
        path: List[str],
    ) -> None:
        """深度优先搜索

        使用三色标记法检测循环。

        Args:
            node: 当前节点
            graph: 有向图
            visited: 已访问节点集合（黑色）
            rec_stack: 递归栈（灰色）
            path: 当前路径
        """
        # 标记为正在访问（灰色）
        visited.add(node)
        rec_stack.add(node)
        path.append(node)

        # 遍历邻居
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                # 未访问，继续 DFS
                self._dfs(neighbor, graph, visited, rec_stack, path)
            elif neighbor in rec_stack:
                # 发现回边，存在循环
                cycle_path = self._extract_cycle(path, neighbor)
                cycle = Cycle(path=cycle_path, length=len(cycle_path) - 1)
                self._cycles.append(cycle)

                self._logger.warning(
                    "[CycleDetector] 检测到循环: %s (长度: %d)",
                    " -> ".join(cycle_path),
                    cycle.length,
                )

        # 回溯，标记为已访问完成（黑色）
        path.pop()
        rec_stack.remove(node)

    def _extract_cycle(self, path: List[str], start_node: str) -> List[str]:
        """从路径中提取循环

        Args:
            path: 当前 DFS 路径
            start_node: 循环起始节点

        Returns:
            循环路径（包含起始节点两次，形成闭环）
        """
        start_idx = path.index(start_node)
        cycle = path[start_idx:] + [start_node]
        return cycle

    def has_cycles(self, relations: List[CallRelation]) -> bool:
        """检查是否存在循环

        Args:
            relations: 调用关系列表

        Returns:
            是否存在循环
        """
        cycles = self.detect_cycles(relations)
        return len(cycles) > 0

    def get_cycle_count(self) -> int:
        """获取循环数量"""
        return len(self._cycles)

    def get_cycles(self) -> List[Cycle]:
        """获取所有循环"""
        return list(self._cycles)

    def clear_cycles(self) -> None:
        """清除循环记录"""
        self._cycles.clear()
        self._logger.debug("[CycleDetector] 清除循环记录")
