"""关系构建器

从 AST 中提取函数调用关系，创建 RELATE 关系。
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CallRelation:
    """调用关系"""

    caller: str
    callee: str
    weight: float
    relation_type: str
    file_path: str


class RelationBuilder:
    """关系构建器

    从 AST 中提取函数调用关系，批量创建关系。

    Attributes:
        db: 数据库连接
        relations: 待创建的关系列表
    """

    def __init__(self, db: Any = None):
        """初始化关系构建器

        Args:
            db: 数据库连接（可选）
        """
        self._db = db
        self._relations: List[CallRelation] = []
        self._logger = logging.getLogger(__name__)

        self._logger.debug("[RelationBuilder] 初始化")

    def extract_calls(self, ast: Dict, file_path: str) -> List[CallRelation]:
        """从 AST 提取调用关系

        Args:
            ast: AST 解析结果
            file_path: 文件路径

        Returns:
            调用关系列表
        """
        relations = []
        root_node = ast.get("root_node")

        if not root_node:
            return relations

        # 提取当前文件的函数定义
        current_functions = self._extract_function_names(ast)

        # 遍历 AST 查找调用表达式
        for node in self._walk_tree(root_node):
            if node.get("type") in ("call_expression", "call"):
                callee = self._extract_callee_name(node)
                if callee:
                    for caller in current_functions:
                        relation = CallRelation(
                            caller=caller,
                            callee=callee,
                            weight=self._calculate_weight(caller, callee, file_path),
                            relation_type="calls",
                            file_path=file_path,
                        )
                        relations.append(relation)
                        self._logger.debug(
                            "[RelationBuilder] 提取关系: %s -> %s",
                            caller,
                            callee,
                        )

        return relations

    def _walk_tree(self, node: Dict) -> List[Dict]:
        """遍历 AST 树"""
        nodes = [node]
        children = node.get("children", [])
        for child in children:
            nodes.extend(self._walk_tree(child))
        return nodes

    def _extract_function_names(self, ast: Dict) -> List[str]:
        """提取函数名"""
        names = []
        symbols = ast.get("symbols", [])
        for symbol in symbols:
            if symbol.get("type") == "function":
                names.append(symbol.get("name", ""))
        return names

    def _extract_callee_name(self, node: Dict) -> Optional[str]:
        """提取被调用函数名"""
        # 简化处理，从 call_expression 中提取函数名
        for child in node.get("children", []):
            if child.get("type") in ("identifier", "property_identifier"):
                return child.get("text", "")
        return None

    def create_relations(self, relations: List[CallRelation]) -> Dict[str, Any]:
        """创建关系

        过滤自调用，批量创建关系。

        Args:
            relations: 调用关系列表

        Returns:
            创建结果统计
        """
        # 过滤自调用
        filtered = [r for r in relations if r.caller != r.callee]

        self._logger.info(
            "[RelationBuilder] 过滤自调用: %d -> %d",
            len(relations),
            len(filtered),
        )

        # 批量创建
        return self.batch_relate(filtered)

    def batch_relate(
        self,
        relations: List[CallRelation],
        batch_size: int = 100,
    ) -> Dict[str, Any]:
        """批量创建关系

        Args:
            relations: 调用关系列表
            batch_size: 批次大小，默认 100

        Returns:
            创建结果统计
        """
        total = len(relations)
        created = 0
        failed = 0
        batches = (total + batch_size - 1) // batch_size

        self._logger.info(
            "[RelationBuilder] 批量创建关系: total=%d, batches=%d",
            total,
            batches,
        )

        for i in range(0, total, batch_size):
            batch = relations[i : i + batch_size]
            batch_num = i // batch_size + 1

            try:
                self._create_batch(batch)
                created += len(batch)
                self._logger.debug(
                    "[RelationBuilder] 批次 %d/%d 完成: %d 条",
                    batch_num,
                    batches,
                    len(batch),
                )
            except Exception as e:
                failed += len(batch)
                self._logger.error(
                    "[RelationBuilder] 批次 %d/%d 失败: %s",
                    batch_num,
                    batches,
                    e,
                )

        return {
            "total": total,
            "created": created,
            "failed": failed,
            "batches": batches,
        }

    def _create_batch(self, batch: List[CallRelation]) -> None:
        """创建一批关系"""
        if self._db is None:
            # Mock 模式，只记录到内存
            self._relations.extend(batch)
            return

        # TODO: 实际 DB 集成（后续任务）
        # 使用 SurrealDB RELATE 创建关系
        pass

    def _calculate_weight(self, caller: str, callee: str, file_path: str) -> float:
        """计算关系权重

        基础权重计算：
        - 同一文件内调用：1.0
        - 跨文件调用：0.8
        - 递归调用：0.5

        Args:
            caller: 调用者
            callee: 被调用者
            file_path: 文件路径

        Returns:
            权重值
        """
        # 递归调用检测
        if caller == callee:
            return 0.5

        # 基础权重
        weight = 1.0

        # TODO: 跨文件调用检测（需要更多上下文）
        # 暂时返回基础权重

        return weight

    def get_relations(self) -> List[CallRelation]:
        """获取所有关系"""
        return list(self._relations)

    def clear_relations(self) -> None:
        """清除关系缓存"""
        self._relations.clear()
        self._logger.debug("[RelationBuilder] 清除关系缓存")

    @property
    def relation_count(self) -> int:
        """关系数量"""
        return len(self._relations)
