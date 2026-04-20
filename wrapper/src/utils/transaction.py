"""SurrealDB 事务上下文管理器"""

import logging
from contextlib import asynccontextmanager
from typing import Any


@asynccontextmanager
async def transaction(db: Any, logger_name: str = "db"):
    """SurrealDB 事务上下文管理器。

    自动处理事务的开始、提交和回滚。

    Args:
        db: SurrealDB 数据库连接
        logger_name: 日志记录器名称，用于错误日志

    Yields:
        db: 数据库连接对象

    Example:
        async with transaction(db, "Atom"):
            result = await db.create("atom", atom_data)
            # 自动处理 COMMIT 或 CANCEL
    """
    logger = logging.getLogger(logger_name)
    await db.query("BEGIN TRANSACTION")
    try:
        yield db
        await db.query("COMMIT TRANSACTION")
    except Exception:
        try:
            await db.query("CANCEL TRANSACTION")
        except Exception as cancel_error:
            logger.error("[%s] 事务回滚失败: %s", logger_name, cancel_error)
        raise
