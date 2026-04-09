"""
SurrealDB 数据库初始化脚本

从零开始初始化 SurrealDB 数据库，包括：
- 创建命名空间和数据库
- 执行 schema 初始化脚本
- 验证表和索引创建成功
- 创建运行时用户（可选）

用法:
    uv run python scripts/init_database.py

环境变量:
    SURREAL_URL          SurrealDB WebSocket URL (默认: ws://localhost:18002)
    SURREAL_NS           命名空间 (默认: memory_ns)
    SURREAL_DB           数据库 (默认: memory_db)
    SURREAL_USER         用户名 (默认: root)
    SURREAL_PASS         密码 (默认: root)
    CREATE_RUNTIME_USER  是否创建运行时用户 (默认: true)

示例:
    # 默认配置初始化
    uv run python scripts/init_database.py

    # 自定义配置初始化
    export SURREAL_URL=ws://localhost:18002
    export SURREAL_NS=memory_ns
    export SURREAL_DB=memory_db
    uv run python scripts/init_database.py

    # 仅验证 schema（不重新初始化）
    uv run python scripts/init_database.py --verify-only
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any

from surrealdb import AsyncSurreal

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("init_db")


class DatabaseInitializer:
    """SurrealDB 数据库初始化器"""

    def __init__(
        self,
        url: str,
        namespace: str,
        database: str,
        username: str,
        password: str,
        create_runtime_user: bool = True,
    ):
        self.url = url
        self.namespace = namespace
        self.database = database
        self.username = username
        self.password = password
        self.should_create_runtime_user = create_runtime_user
        self.db: Any = None  # SurrealDB SDK 返回联合类型，使用 Any 避免类型检查误报

    async def connect(self) -> None:
        """连接到 SurrealDB"""
        try:
            self.db = AsyncSurreal(self.url)
            await self.db.connect()
            await self.db.signin(
                {
                    "username": self.username,
                    "password": self.password,
                }
            )
            logger.info("✅ 已连接到 SurrealDB: %s", self.url)
        except Exception as e:
            logger.error("❌ 连接 SurrealDB 失败: %s", e)
            raise

    async def disconnect(self) -> None:
        """断开连接"""
        if self.db:
            await self.db.close()
            logger.info("🔌 已断开 SurrealDB 连接")

    async def ensure_namespace_and_database(self) -> None:
        """确保命名空间和数据库存在"""
        try:
            # 使用 root 用户查询现有命名空间和数据库
            await self.db.use("root", "root")

            # 检查命名空间是否存在
            ns_query = "SELECT * FROM namespace WHERE name = $name"
            ns_result = await self.db.query(ns_query, {"name": self.namespace})

            if not ns_result or not ns_result[0]:
                logger.info("📦 创建命名空间: %s", self.namespace)
                await self.db.query(f"DEFINE NAMESPACE IF NOT EXISTS {self.namespace}")
            else:
                logger.info("📦 命名空间已存在: %s", self.namespace)

            # 检查数据库是否存在
            db_query = "SELECT * FROM database WHERE name = $name"
            db_result = await self.db.query(db_query, {"name": self.database})

            if not db_result or not db_result[0]:
                logger.info("🗄️  创建数据库: %s.%s", self.namespace, self.database)
                await self.db.query(f"USE NS {self.namespace}; DEFINE DATABASE IF NOT EXISTS {self.database}")
            else:
                logger.info("🗄️  数据库已存在: %s.%s", self.namespace, self.database)

            # 切换到目标命名空间和数据库
            await self.db.use(self.namespace, self.database)
            logger.info("✅ 已切换到: %s.%s", self.namespace, self.database)

        except Exception as e:
            logger.error("❌ 创建命名空间/数据库失败: %s", e)
            raise

    async def apply_schema(self) -> None:
        """执行 schema 初始化脚本"""
        try:
            # 查找初始化脚本
            script_path = Path(__file__).parent / "init_surrealdb.surql"
            if not script_path.exists():
                raise FileNotFoundError(f"初始化脚本不存在: {script_path}")

            logger.info("📜 读取初始化脚本: %s", script_path)
            sql = script_path.read_text(encoding="utf-8")

            # 拆分为单条语句逐条执行
            statements = [s.strip() for s in sql.split(";") if s.strip()]
            logger.info("📜 执行 %d 条 SQL 语句...", len(statements))

            success_count = 0
            for i, stmt in enumerate(statements, 1):
                try:
                    # 跳过纯注释块
                    lines = [line for line in stmt.split("\n") if not line.strip().startswith("--")]
                    if not any(line.strip() for line in lines):
                        continue

                    # 执行语句
                    result = await self.db.query(stmt)
                    success_count += 1

                    # 每 50 条打印一次进度
                    if i % 50 == 0:
                        logger.info("  进度: %d/%d (%.1f%%)", i, len(statements), i / len(statements) * 100)

                except Exception as e:
                    # 某些语句可能因已存在而失败（幂等性）
                    if "already exists" in str(e).lower():
                        logger.debug("  语句已存在，跳过: %s", stmt[:50])
                        success_count += 1
                    else:
                        logger.warning("  ⚠️  语句执行失败 (第 %d 条): %s", i, e)
                        logger.debug("  语句内容: %s", stmt[:100])

            logger.info("✅ Schema 初始化完成: %d/%d 条语句执行成功", success_count, len(statements))

        except Exception as e:
            logger.error("❌ Schema 初始化失败: %s", e)
            raise

    async def verify_schema(self) -> bool:
        """验证 schema 是否正确初始化"""
        try:
            logger.info("🔍 验证 Schema...")

            # 检查必需的表 - SurrealDB 3.0 使用 INFO FOR DB 获取表列表
            required_tables = ["memory", "memory_relation", "project", "schema_version", "conflict"]
            db_info = await self.db.query("INFO FOR DB")

            tables = []
            if db_info and isinstance(db_info, dict) and "tables" in db_info:
                tables = list(db_info["tables"].keys())
            elif db_info and isinstance(db_info, list) and len(db_info) > 0:
                # 处理嵌套结果
                info = db_info[0] if isinstance(db_info[0], dict) else {}
                if "tables" in info:
                    tables = list(info["tables"].keys())

            logger.info("  📋 当前表: %s", ", ".join(tables) if tables else "(无)")

            for table in required_tables:
                if table in tables:
                    logger.info("  ✅ 表存在: %s", table)
                else:
                    logger.error("  ❌ 表不存在: %s", table)
                    return False

            # 检查 schema_version
            try:
                version_result = await self.db.query("SELECT * FROM schema_version LIMIT 1")
                if version_result and isinstance(version_result, list) and len(version_result) > 0:
                    version = version_result[0].get("version", "unknown")
                    logger.info("  ✅ Schema 版本: %s", version)
                else:
                    logger.warning("  ⚠️  无法获取 Schema 版本（可能首次初始化）")
            except Exception as e:
                logger.warning("  ⚠️  检查 Schema 版本时出错: %s", e)

            # 检查 HNSW 索引 - SurrealDB 3.0 使用 INFO FOR TABLE
            try:
                table_info = await self.db.query("INFO FOR TABLE memory")
                if table_info and isinstance(table_info, dict) and "indexes" in table_info:
                    indexes = table_info["indexes"]
                    if "memory_embedding_hnsw" in indexes:
                        logger.info("  ✅ HNSW 索引存在: memory_embedding_hnsw")
                    else:
                        logger.warning("  ⚠️  HNSW 索引不存在: memory_embedding_hnsw")
                else:
                    logger.warning("  ⚠️  无法获取表索引信息")
            except Exception as e:
                logger.warning("  ⚠️  检查索引时出错: %s", e)

            logger.info("✅ Schema 验证通过")
            return True

        except Exception as e:
            logger.error("❌ Schema 验证失败: %s", e)
            return False

    async def create_runtime_user(self) -> None:
        """创建运行时用户（如果启用）"""
        if not self.create_runtime_user:
            logger.info("⏭️  跳过运行时用户创建")
            return

        try:
            logger.info("👤 创建运行时用户...")

            # 运行时用户已在 init_surrealdb.surql 中定义
            # 这里只是验证用户是否存在
            user_query = "SELECT * FROM users WHERE name = 'runtime_user'"
            user_result = await self.db.query(user_query)

            if user_result and user_result[0]:
                logger.info("  ✅ 运行时用户已存在: runtime_user")
                logger.info("  ⚠️  默认密码: change_me_in_production (请立即修改!)")
            else:
                logger.warning("  ⚠️  运行时用户不存在，请检查 init_surrealdb.surql")

        except Exception as e:
            logger.warning("⚠️  验证运行时用户失败: %s", e)

    async def initialize(self, verify_only: bool = False) -> None:
        """完整的初始化流程"""
        logger.info("=" * 60)
        logger.info("SurrealDB 数据库初始化")
        logger.info("=" * 60)
        logger.info("URL: %s", self.url)
        logger.info("Namespace: %s", self.namespace)
        logger.info("Database: %s", self.database)
        logger.info("User: %s", self.username)
        logger.info("")

        try:
            # 1. 连接
            await self.connect()

            # 2. 创建命名空间和数据库
            await self.ensure_namespace_and_database()

            if verify_only:
                # 3. 仅验证 schema
                success = await self.verify_schema()
                if not success:
                    logger.error("❌ Schema 验证失败")
                    sys.exit(1)
                logger.info("✅ 验证完成，Schema 正常")
            else:
                # 3. 应用 schema
                await self.apply_schema()

                # 4. 验证 schema
                success = await self.verify_schema()
                if not success:
                    logger.error("❌ Schema 验证失败")
                    sys.exit(1)

                # 5. 创建运行时用户
                await self.create_runtime_user()

            logger.info("")
            logger.info("=" * 60)
            logger.info("✅ 数据库初始化完成!")
            logger.info("=" * 60)

        except Exception as e:
            logger.error("")
            logger.error("=" * 60)
            logger.error("❌ 数据库初始化失败!")
            logger.error("=" * 60)
            logger.error("错误: %s", e)
            sys.exit(1)
        finally:
            await self.disconnect()


async def main() -> None:
    """主函数"""
    parser = argparse.ArgumentParser(description="SurrealDB 数据库初始化脚本")
    parser.add_argument("--verify-only", action="store_true", help="仅验证 schema，不重新初始化")
    parser.add_argument("--no-runtime-user", action="store_true", help="不创建运行时用户")
    args = parser.parse_args()

    # 从环境变量读取配置
    url = os.getenv("SURREAL_URL", "ws://localhost:18002")
    namespace = os.getenv("SURREAL_NS", "memory_ns")
    database = os.getenv("SURREAL_DB", "memory_db")
    username = os.getenv("SURREAL_USER", "root")
    password = os.getenv("SURREAL_PASS", "root")
    create_runtime_user = not args.no_runtime_user and os.getenv("CREATE_RUNTIME_USER", "true").lower() == "true"

    # 创建初始化器
    initializer = DatabaseInitializer(
        url=url,
        namespace=namespace,
        database=database,
        username=username,
        password=password,
        create_runtime_user=create_runtime_user,
    )

    # 执行初始化
    await initializer.initialize(verify_only=args.verify_only)


if __name__ == "__main__":
    asyncio.run(main())
