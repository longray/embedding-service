"""SurrealDB 数据库辅助函数"""

from typing import Any


def parse_surrealdb_result(result: dict | list | None) -> dict | None:
    """解析 SurrealDB 返回结果，提取记录字典。

    Args:
        result: SurrealDB 查询结果，可能是 dict、list 或 None

    Returns:
        解析后的记录字典，如果结果为空则返回 None
    """
    if result is None:
        return None

    if isinstance(result, dict):
        return result
    elif isinstance(result, list) and result:
        record = result[0]
        if isinstance(record, list) and record:
            record = record[0]
        return record if isinstance(record, dict) else None
    return None


def extract_record_id(record: dict | Any) -> str:
    """从记录中提取字符串格式的 RecordID。

    Args:
        record: 包含 id 字段的记录字典，或直接的 RecordID 对象

    Returns:
        格式化的 RecordID 字符串，如 "atom:abc123"
    """
    if not record:
        return ""

    raw_id = record.get("id") if isinstance(record, dict) else record
    if raw_id and not isinstance(raw_id, list) and hasattr(raw_id, "table_name"):
        return f"{raw_id.table_name}:{raw_id.id}"
    return str(raw_id) if raw_id else ""


def parse_record_id(record_id_str: str) -> tuple[str, str] | None:
    """解析 RecordID 字符串为表名和 ID。

    Args:
        record_id_str: RecordID 字符串，如 "atom:abc123"

    Returns:
        (table_name, id) 元组，如果格式无效则返回 None
    """
    if ":" not in record_id_str:
        return None
    parts = record_id_str.split(":", 1)
    return parts[0], parts[1]
