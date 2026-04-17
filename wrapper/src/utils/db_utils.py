"""数据库工具函数

提供 SurrealDB 结果解析等通用工具函数。
"""

from typing import Any


def extract_records(db_result: Any) -> list[dict[str, Any]]:
    """从 SurrealDB query() 返回值中提取记录列表

    处理 SDK 返回的多种格式：
    - list[dict]: 直接的记录列表（单条 SELECT 语句）
    - list[list[dict]]: 嵌套结构（多语句结果或 query_raw）
    - list[dict{"result": [...]}]: 标准查询结果格式

    Args:
        db_result: SurrealDB 查询返回的原始结果

    Returns:
        提取后的记录列表，每个记录是一个字典
    """
    records: list[dict[str, Any]] = []

    if not db_result or not isinstance(db_result, list):
        return records

    for item in db_result:
        if isinstance(item, dict):
            # 处理标准查询结果格式 {"result": [...]}
            if "result" in item:
                inner = item["result"]
                if isinstance(inner, list):
                    records.extend(inner)
                elif inner:
                    records.append(inner)
            else:
                records.append(item)
        elif isinstance(item, list):
            # 处理嵌套列表
            for record in item:
                if isinstance(record, dict):
                    records.append(record)

    return records
