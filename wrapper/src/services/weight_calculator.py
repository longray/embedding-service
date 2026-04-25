"""权重计算器

计算调用关系的权重，用于图遍历优先级。
支持内存缓存和数据库持久化。

权重因子：
- 调用频率（frequency）
- 代码复杂度（complexity）
- 参数数量（param_count）
- 是否跨文件（is_cross_file）
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..utils.db_utils import extract_records

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
        weights: 权重缓存（内存）
        db: 数据库连接（可选，用于持久化）
    """

    def __init__(self, db: Any = None):
        """初始化权重计算器

        Args:
            db: 数据库连接（SurrealDB 或其他），用于持久化
        """
        self._weights: Dict[str, float] = {}
        self._db = db
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

        # 保存权重到内存
        relation_id = f"{caller}->{callee}"
        self.save_weight(relation_id, weight)

        return weight

    def save_weight(self, relation_id: str, weight: float) -> None:
        """保存权重到内存

        Args:
            relation_id: 关系 ID
            weight: 权重值
        """
        self._weights[relation_id] = weight
        self._logger.debug("[WeightCalculator] 保存权重到内存: %s=%.2f", relation_id, weight)

    async def save_weight_to_db(
        self,
        caller: str,
        callee: str,
        weight: float,
        tenant_id: str = "default",
    ) -> bool:
        """保存权重到数据库

        将权重保存到 SurrealDB 的 reference 表中。

        Args:
            caller: 调用者（entity ID）
            callee: 被调用者（entity ID）
            weight: 权重值
            tenant_id: 租户 ID

        Returns:
            是否保存成功
        """
        if self._db is None:
            self._logger.warning("[WeightCalculator] 数据库连接未设置，无法持久化权重")
            return False

        try:
            # 更新 reference 表的 weight 字段
            query = """
                UPDATE reference
                SET weight = $weight
                WHERE in = $caller AND out = $callee AND tenant_id = $tenant_id
            """
            await self._db.query(
                query,
                {
                    "weight": weight,
                    "caller": caller,
                    "callee": callee,
                    "tenant_id": tenant_id,
                },
            )
            self._logger.debug(
                "[WeightCalculator] 权重已持久化到 DB: %s->%s=%.2f",
                caller,
                callee,
                weight,
            )
            return True
        except Exception as e:
            self._logger.error("[WeightCalculator] 保存权重到 DB 失败: %s", e)
            return False

    async def persist_all_weights(self, tenant_id: str = "default") -> Dict[str, int]:
        """持久化所有内存中的权重到数据库

        Args:
            tenant_id: 租户 ID

        Returns:
            统计信息 {"success": 成功数, "failed": 失败数}
        """
        if self._db is None:
            self._logger.warning("[WeightCalculator] 数据库连接未设置，无法持久化权重")
            return {"success": 0, "failed": 0}

        success_count = 0
        failed_count = 0

        for relation_id, weight in self._weights.items():
            # 解析 relation_id (格式: "caller->callee")
            if "->" not in relation_id:
                continue

            parts = relation_id.split("->", 1)
            if len(parts) != 2:
                continue

            caller, callee = parts

            # 保存到 DB
            result = await self.save_weight_to_db(caller, callee, weight, tenant_id)
            if result:
                success_count += 1
            else:
                failed_count += 1

        self._logger.info(
            "[WeightCalculator] 批量持久化完成: success=%d, failed=%d",
            success_count,
            failed_count,
        )

        return {"success": success_count, "failed": failed_count}

    def get_weight(self, relation_id: str) -> Optional[float]:
        """获取权重（从内存）

        Args:
            relation_id: 关系 ID

        Returns:
            权重值，如果不存在返回 None
        """
        return self._weights.get(relation_id)

    async def get_weight_from_db(
        self,
        caller: str,
        callee: str,
        tenant_id: str = "default",
    ) -> Optional[float]:
        """从数据库获取权重

        Args:
            caller: 调用者（entity ID）
            callee: 被调用者（entity ID）
            tenant_id: 租户 ID

        Returns:
            权重值，如果不存在返回 None
        """
        if self._db is None:
            return None

        try:
            query = """
                SELECT weight FROM reference
                WHERE in = $caller AND out = $callee AND tenant_id = $tenant_id
                LIMIT 1
            """
            result = await self._db.query(
                query,
                {"caller": caller, "callee": callee, "tenant_id": tenant_id},
            )

            records = extract_records(result)
            if records:
                weight = records[0].get("weight")
                self._logger.debug(
                    "[WeightCalculator] 从 DB 获取权重: %s->%s=%.2f",
                    caller,
                    callee,
                    weight,
                )
                return weight
        except Exception as e:
            self._logger.error("[WeightCalculator] 从 DB 获取权重失败: %s", e)

        return None

    def remove_weight(self, relation_id: str) -> bool:
        """删除权重（从内存）

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
        """清除所有权重（内存）"""
        self._weights.clear()
        self._logger.debug("[WeightCalculator] 清除所有权重")

    def get_all_weights(self) -> Dict[str, float]:
        """获取所有权重（内存）

        Returns:
            关系 ID -> 权重的映射
        """
        return dict(self._weights)

    @property
    def weight_count(self) -> int:
        """权重数量（内存）"""
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

    async def load_weights_from_db(
        self,
        tenant_id: str = "default",
        limit: int = 1000,
    ) -> int:
        """从数据库加载权重到内存

        Args:
            tenant_id: 租户 ID
            limit: 最大加载数量

        Returns:
            加载的权重数量
        """
        if self._db is None:
            self._logger.warning("[WeightCalculator] 数据库连接未设置，无法加载权重")
            return 0

        try:
            query = """
                SELECT in, out, weight FROM reference
                WHERE tenant_id = $tenant_id AND weight IS NOT NULL
                LIMIT $limit
            """
            result = await self._db.query(
                query,
                {"tenant_id": tenant_id, "limit": limit},
            )

            loaded_count = 0
            for record in extract_records(result):
                caller = str(record.get("in", ""))
                callee = str(record.get("out", ""))
                weight = record.get("weight")

                if caller and callee and weight is not None:
                    relation_id = f"{caller}->{callee}"
                    self._weights[relation_id] = float(weight)
                    loaded_count += 1

            self._logger.info(
                "[WeightCalculator] 从 DB 加载权重: %d 条",
                loaded_count,
            )
            return loaded_count
        except Exception as e:
            self._logger.error("[WeightCalculator] 从 DB 加载权重失败: %s", e)
            return 0
