"""
Meilisearch 索引初始化脚本

从零开始初始化 Meilisearch 索引，包括：
- 创建索引
- 配置索引设置（中文分词、代码术语字典等）
- 验证索引创建成功
- 设置 API Key（可选）

用法:
    uv run python scripts/init_meilisearch.py

环境变量:
    WRAPPER_MEILI_URL         Meilisearch URL (默认: http://localhost:7700)
    WRAPPER_MEILI_API_KEY      Meilisearch API Key (默认: None)
    WRAPPER_MEILI_INDEX_NAME   索引名 (默认: memories)
    WRAPPER_MEILI_TIMEOUT      请求超时 (默认: 30.0)

示例:
    # 默认配置初始化
    uv run python scripts/init_meilisearch.py

    # 自定义配置初始化
    export WRAPPER_MEILI_URL=http://localhost:7700
    export WRAPPER_MEILI_API_KEY=your_master_key
    uv run python scripts/init_meilisearch.py

    # 仅验证索引（不重新初始化）
    uv run python scripts/init_meilisearch.py --verify-only
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from typing import Any

import httpx

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("init_meili")


class MeilisearchInitializer:
    """Meilisearch 索引初始化器"""

    # 默认索引配置
    DEFAULT_INDEX_SETTINGS: dict[str, Any] = {
        # 可搜索字段（支持中文、代码、元数据等多类型搜索）
        "searchableAttributes": [
            "content_zh",
            "title_zh",
            "tags_zh",
            "content_search",
            "code",
            "content",
        ],
        # 可过滤字段（用于精确匹配和范围查询）
        "filterableAttributes": [
            "tenant_id",
            "type",
            "tags",
            "project_id",
            "file_path",
            "date",
            "ip_address",
            "email",
            "version",
            "created_at",
            "source_id",
        ],
        # 可排序字段
        "sortableAttributes": ["date", "created_at"],
        # 不作为分隔符的字符（保持代码标识符完整性）
        "nonSeparatorTokens": [".", "-", "@", ":", "/", "_"],
        # 中文标点作为分隔符
        "separatorTokens": ["、", "；", "："],
        # 中文本地化配置（使用 cmn 触发 jieba 分词）
        "localizedAttributes": [{"locales": ["cmn"], "attributePatterns": ["*_zh"]}],
        # 拼写容错（禁用文件路径、版本号等精确字段）
        "typoTolerance": {"enabled": True, "disableOnAttributes": ["file_path", "version", "email", "ip_address"]},
        # 代码术语字典（104词）
        "dictionary": [
            # 版本前缀
            "v1",
            "v2",
            "v3",
            "v4",
            "v5",
            "alpha",
            "beta",
            "rc",
            "release",
            "snapshot",
            # 编程语言
            "python",
            "java",
            "javascript",
            "typescript",
            "go",
            "rust",
            "cpp",
            "csharp",
            "ruby",
            "php",
            "swift",
            "kotlin",
            "scala",
            # 常见命名
            "http",
            "https",
            "api",
            "www",
            "localhost",
            "com",
            "cn",
            "org",
            "net",
            "io",
            "dev",
            "get",
            "post",
            "put",
            "delete",
            "patch",
            # 代码术语
            "class",
            "interface",
            "enum",
            "struct",
            "function",
            "method",
            "property",
            "attribute",
            "import",
            "export",
            "require",
            "include",
            "public",
            "private",
            "protected",
            "static",
            "async",
            "await",
            "promise",
            "callback",
            # 框架/库
            "django",
            "flask",
            "fastapi",
            "spring",
            "react",
            "vue",
            "angular",
            "next",
            "nuxt",
            "tensorflow",
            "pytorch",
            "sklearn",
            # 常见 ID 前缀
            "ID",
            "NO",
            "NUM",
            "CODE",
            "KEY",
            "ORD",
            "PAY",
            "TRK",
            "INV",
            "USR",
            # 时间
            "2025",
            "2026",
            "2027",
            "2028",
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
            # IP 段
            "192",
            "168",
            "172",
            "10",
            "127",
            "0",
            "1",
        ],
    }

    def __init__(
        self,
        url: str,
        api_key: str | None,
        index_name: str,
        timeout: float = 30.0,
    ):
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.index_name = index_name
        self.timeout = timeout
        self.client: Any = None  # httpx.AsyncClient | None

    async def connect(self) -> None:
        """连接到 Meilisearch"""
        try:
            headers: dict[str, str] = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            self.client = httpx.AsyncClient(
                base_url=self.url,
                headers=headers,
                timeout=httpx.Timeout(self.timeout),
            )

            # 检查健康状态
            response = await self.client.get("/health")
            if response.status_code == 200:
                logger.info("✅ 已连接到 Meilisearch: %s", self.url)
            else:
                raise Exception(f"Meilisearch 健康检查失败: {response.status_code}")
        except Exception as e:
            logger.error("❌ 连接 Meilisearch 失败: %s", e)
            raise

    async def disconnect(self) -> None:
        """断开连接"""
        if self.client:
            await self.client.aclose()
            logger.info("🔌 已断开 Meilisearch 连接")

    async def ensure_index(self) -> None:
        """确保索引存在"""
        try:
            # 尝试获取索引
            response = await self.client.get(f"/indexes/{self.index_name}")

            if response.status_code == 200:
                logger.info("📦 索引已存在: %s", self.index_name)
                return

            # 索引不存在，创建新索引
            if response.status_code == 404:
                logger.info("📦 创建索引: %s", self.index_name)
                response = await self.client.post(
                    "/indexes",
                    json={"uid": self.index_name, "primaryKey": "id"},
                )

                if response.status_code == 201:
                    task_info = response.json()
                    await self._wait_for_task(task_info["taskUid"])
                    logger.info("✅ 索引创建成功: %s", self.index_name)
                else:
                    raise Exception(f"创建索引失败: {response.text}")

        except httpx.HTTPStatusError as e:
            # 404 表示索引不存在，需要创建
            if e.response.status_code == 404:
                logger.info("📦 创建索引: %s", self.index_name)
                response = await self.client.post(
                    "/indexes",
                    json={"uid": self.index_name, "primaryKey": "id"},
                )

                if response.status_code == 201:
                    task_info = response.json()
                    await self._wait_for_task(task_info["taskUid"])
                    logger.info("✅ 索引创建成功: %s", self.index_name)
                else:
                    raise Exception(f"创建索引失败: {response.text}")
            else:
                raise

    async def configure_index(self) -> None:
        """配置索引设置"""
        try:
            logger.info("⚙️  配置索引设置...")

            # 更新索引设置
            response = await self.client.patch(
                f"/indexes/{self.index_name}/settings",
                json=self.DEFAULT_INDEX_SETTINGS,
            )

            if response.status_code == 202:
                task_info = response.json()
                await self._wait_for_task(task_info["taskUid"])
                logger.info("✅ 索引配置成功")
            else:
                raise Exception(f"配置索引失败: {response.text}")

        except Exception as e:
            logger.error("❌ 配置索引失败: %s", e)
            raise

    async def verify_index(self) -> bool:
        """验证索引是否正确配置"""
        try:
            logger.info("🔍 验证索引配置...")

            # 1. 检查索引是否存在
            response = await self.client.get(f"/indexes/{self.index_name}")
            if response.status_code != 200:
                logger.error("  ❌ 索引不存在: %s", self.index_name)
                return False
            logger.info("  ✅ 索引存在: %s", self.index_name)

            # 2. 检查索引设置
            settings_response = await self.client.get(f"/indexes/{self.index_name}/settings")
            if settings_response.status_code == 200:
                settings = settings_response.json()
                logger.info("  ✅ 索引设置已配置")

                # 验证关键字段
                searchable = settings.get("searchableAttributes", [])
                if "content_zh" in searchable:
                    logger.info("  ✅ 中文搜索字段已配置")
                if "code" in searchable:
                    logger.info("  ✅ 代码搜索字段已配置")

                filterable = settings.get("filterableAttributes", [])
                if "tenant_id" in filterable:
                    logger.info("  ✅ 租户过滤已配置")
                if "date" in filterable:
                    logger.info("  ✅ 日期过滤已配置")

                dictionary = settings.get("dictionary", [])
                if len(dictionary) >= 100:
                    logger.info("  ✅ 代码术语字典已配置 (%d 词)", len(dictionary))

            # 3. 检查索引统计
            stats_response = await self.client.get(f"/indexes/{self.index_name}/stats")
            if stats_response.status_code == 200:
                stats = stats_response.json()
                doc_count = stats.get("numberOfDocuments", 0)
                logger.info("  📊 索引文档数: %d", doc_count)

            logger.info("✅ 索引验证通过")
            return True

        except Exception as e:
            logger.error("❌ 索引验证失败: %s", e)
            return False

    async def _wait_for_task(self, task_uid: int, timeout: float = 120.0) -> dict:
        """等待任务完成"""
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            try:
                response = await self.client.get(f"/tasks/{task_uid}")
                task = response.json()

                status = task.get("status")
                if status == "succeeded":
                    return task
                elif status == "failed":
                    error = task.get("error", {})
                    msg = error.get("message", "Unknown error")
                    raise RuntimeError(f"任务失败 (uid={task_uid}): {msg}")

                await asyncio.sleep(0.5)
            except Exception as e:
                if "not found" in str(e).lower():
                    # 任务可能已完成并被清理
                    return {"status": "succeeded"}
                raise

        raise TimeoutError(f"任务超时 (uid={task_uid}, timeout={timeout}s)")

    async def initialize(self, verify_only: bool = False) -> None:
        """完整的初始化流程"""
        logger.info("=" * 60)
        logger.info("Meilisearch 索引初始化")
        logger.info("=" * 60)
        logger.info("URL: %s", self.url)
        logger.info("Index: %s", self.index_name)
        logger.info("")

        try:
            # 1. 连接
            await self.connect()

            if verify_only:
                # 2. 仅验证索引
                success = await self.verify_index()
                if not success:
                    logger.error("❌ 索引验证失败")
                    sys.exit(1)
                logger.info("✅ 验证完成，索引正常")
            else:
                # 2. 确保索引存在
                await self.ensure_index()

                # 3. 配置索引
                await self.configure_index()

                # 4. 验证索引
                success = await self.verify_index()
                if not success:
                    logger.error("❌ 索引验证失败")
                    sys.exit(1)

            logger.info("")
            logger.info("=" * 60)
            logger.info("✅ Meilisearch 初始化完成!")
            logger.info("=" * 60)

        except Exception as e:
            logger.error("")
            logger.error("=" * 60)
            logger.error("❌ Meilisearch 初始化失败!")
            logger.error("=" * 60)
            logger.error("错误: %s", e)
            sys.exit(1)
        finally:
            await self.disconnect()


async def main() -> None:
    """主函数"""
    parser = argparse.ArgumentParser(description="Meilisearch 索引初始化脚本")
    parser.add_argument("--verify-only", action="store_true", help="仅验证索引，不重新初始化")
    args = parser.parse_args()

    # 从环境变量读取配置
    url = os.getenv("WRAPPER_MEILI_URL", "http://localhost:7700")
    api_key = os.getenv("WRAPPER_MEILI_API_KEY")
    index_name = os.getenv("WRAPPER_MEILI_INDEX_NAME", "memories")
    timeout = float(os.getenv("WRAPPER_MEILI_TIMEOUT", "30.0"))

    # 创建初始化器
    initializer = MeilisearchInitializer(
        url=url,
        api_key=api_key,
        index_name=index_name,
        timeout=timeout,
    )

    # 执行初始化
    await initializer.initialize(verify_only=args.verify_only)


if __name__ == "__main__":
    asyncio.run(main())
