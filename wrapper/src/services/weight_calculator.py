"""权重计算器

计算调用关系的权重，用于图遍历优先级。

权重因子：
- 调用频率（frequency）
- 代码复杂度（complexity）
- 参数数量（param_count）
- 是否跨文件（is_cross_file）
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class WeightFactors:
    """权重因子

    Attributes:
        frequency: 调用频率（调用次数）
        complexity: 代码复杂度（圈复杂度）
        param_count: 参数数量
        is_cross_file: 是否跨文件调用
    """

    frequency: int = 0
    complexity: int = 0
    param_count: int = 0
    is_cross_file: bool = False


class WeightCalculator:
    """权重计算器

    计算调用关系的权重，用于图遍历优先级。
    权重范围：[0, 1]

    Attributes:
        weights: 权重缓存
    """

    def __init__(self):
        """初始化权重计算器"""
        self._weights: Dict[str, float] = {}
        self._logger = logging.getLogger(__name__)

        self._logger.debug("[WeightCalculator] 初始化")

    def calculate_weight(self, factors: WeightFactors) -> float:
        """计算权重

        根据权重因子计算调用关系的权重。

        计算公式：
        - 基础权重：0.5
        - 频率加成：min(frequency * 0.05, 0.3)
        - 复杂度加成：min(complexity * 0.02, 0.2)
        - 参数加成：min(param_count * 0.01, 0.1)
        - 跨文件加成：0.1（如果是跨文件）
        - 总权重 = 基础 + 各项加成
        - 归一化到 [0, 1]

        Args:
            factors: 权重因子

        Returns:
            权重值 [0, 1]
        """
        # 基础权重
        base_weight = 0.5

        # 各项加成（带上限）
        freq_bonus = min(factors.frequency * 0.05, 0.3)
        complexity_bonus = min(factors.complexity * 0.02, 0.2)
        param_bonus = min(factors.param_count * 0.01, 0.1)
        cross_file_bonus = 0.1 if factors.is_cross_file else 0.0

        # 总权重
        total = base_weight + freq_bonus + complexity_bonus + param_bonus + cross_file_bonus

        # 归一化到 [0, 1]
        weight = min(total, 1.0)

        self._logger.debug(
            "[WeightCalculator] 计算权重: base=%.2f, freq=%.2f, complexity=%.2f, param=%.2f, cross=%.2f, total=%.2f",
            base_weight,
            freq_bonus,
            complexity_bonus,
            param_bonus,
            cross_file_bonus,
            weight,
        )

        return weight

    def calculate_weight_from_relation(
        self,
        caller: str,
        callee: str,
        frequency: int = 1,
        complexity: int = 1,
        param_count: int = 0,
        is_cross_file: bool = False,
    ) -> float:
        """从关系信息计算权重

        Args:
            caller: 调用者
            callee: 被调用者
            frequency: 调用频率
            complexity: 代码复杂度
            param_count: 参数数量
            is_cross_file: 是否跨文件

        Returns:
            权重值
        """
        factors = WeightFactors(
            frequency=frequency,
            complexity=complexity,
            param_count=param_count,
            is_cross_file=is_cross_file,
        )

        weight = self.calculate_weight(factors)

        # 保存权重
        relation_id = f"{caller}->{callee}"
        self.save_weight(relation_id, weight)

        return weight

    def save_weight(self, relation_id: str, weight: float) -> None:
        """保存权重

        Args:
            relation_id: 关系 ID
            weight: 权重值
        """
        self._weights[relation_id] = weight
        self._logger.debug("[WeightCalculator] 保存权重: %s=%.2f", relation_id, weight)

    def get_weight(self, relation_id: str) -> Optional[float]:
        """获取权重

        Args:
            relation_id: 关系 ID

        Returns:
            权重值，如果不存在返回 None
        """
        return self._weights.get(relation_id)

    def remove_weight(self, relation_id: str) -> bool:
        """删除权重

        Args:
            relation_id: 关系 ID

        Returns:
            是否成功删除
        """
        if relation_id in self._weights:
            del self._weights[relation_id]
            self._logger.debug("[WeightCalculator] 删除权重: %s", relation_id)
            return True
        return False

    def clear_weights(self) -> None:
        """清除所有权重"""
        self._weights.clear()
        self._logger.debug("[WeightCalculator] 清除所有权重")

    def get_all_weights(self) -> Dict[str, float]:
        """获取所有权重

        Returns:
            关系 ID -> 权重的映射
        """
        return dict(self._weights)

    @property
    def weight_count(self) -> int:
        """权重数量"""
        return len(self._weights)

    def get_top_relations(self, n: int = 10) -> Dict[str, float]:
        """获取权重最高的关系

        Args:
            n: 返回数量

        Returns:
            权重最高的 n 个关系
        """
        sorted_weights = sorted(
            self._weights.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return dict(sorted_weights[:n])
