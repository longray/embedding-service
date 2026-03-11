"""SurrealDB → Meilisearch 数据迁移脚本

将 SurrealDB memory 表中的现有记忆批量同步到 Meilisearch 索引。
幂等操作：重复执行不会产生重复文档（Meilisearch upsert 语义）。

用法:
    uv run python scripts/migrate_to_meilisearch.py [--batch-size 100] [--tenant-id default]

环境变量:
    SURREAL_URL          SurrealDB WebSocket URL (默认: ws://localhost:18800)
    SURREAL_NS           命名空间 (默认: memory)
    SURREAL_DB           数据库 (默认: memory)
    SURREAL_USER         用户名 (默认: root)
    SURREAL_PASS         密码 (默认: root)
    WRAPPER_MEILI_URL    Meilisearch URL (默认: http://localhost:7700)
    WRAPPER_MEILI_API_KEY  Meilisearch API Key (默认: None)
    WRAPPER_MEILI_INDEX_NAME  索引名 (默认: memories)
"""

import argparse
import asyncio
import logging
import os
import sys
import time

import httpx
from surrealdb import AsyncSurreal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("migrate")


class MeilisearchMigrator:
    """Meilisearch 批量写入客户端（迁移专用，精简版）"""

    def __init__(self, url: str, api_key: str | None, index_name: str):
        self._url = url.rstrip("/")
        self._index_name = index_name
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            base_url=self._url,
            headers=headers,
            timeout=httpx.Timeout(60.0),
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def health(self) -> bool:
        try:
            resp = await self._client.get("/health")
            return resp.status_code == 200
        except Exception:
            return False

    async def ensure_index(self) -> None:
        """确保索引存在（幂等：已存在则跳过）"""
        resp = await self._client.post(
            "/indexes",
            json={"uid": self._index_name, "primaryKey": "id"},
        )
        if resp.status_code == 202:
            task = resp.json()
            try:
                await self._wait_for_task(task["taskUid"])
            except RuntimeError as e:
                if "already exists" in str(e):
                    logger.info("索引 '%s' 已存在，跳过创建", self._index_name)
                else:
                    raise
        # 其他状态码（如 409）也表示索引已存在，不报错

    async def add_documents(self, docs: list[dict]) -> dict:
        """批量添加文档"""
        resp = await self._client.post(
            f"/indexes/{self._index_name}/documents",
            json=docs,
            params={"primaryKey": "id"},
        )
        resp.raise_for_status()
        task_info = resp.json()
        return await self._wait_for_task(task_info["taskUid"])

    async def get_stats(self) -> dict:
        """获取索引统计"""
        resp = await self._client.get(f"/indexes/{self._index_name}/stats")
        if resp.status_code == 200:
            return resp.json()
        return {}

    async def _wait_for_task(self, task_uid: int, timeout: float = 120.0) -> dict:
        """等待任务完成"""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            resp = await self._client.get(f"/tasks/{task_uid}")
            resp.raise_for_status()
            task = resp.json()
            status = task.get("status")
            if status == "succeeded":
                return task
            if status == "failed":
                error = task.get("error", {})
                msg = error.get("message", "Unknown error")
                raise RuntimeError(f"Meilisearch 任务失败 (uid={task_uid}): {msg}")
            await asyncio.sleep(0.5)
        raise TimeoutError(f"Meilisearch 任务超时 (uid={task_uid})")


def to_meili_id(surreal_id: str) -> str:
    """SurrealDB record ID → Meilisearch 主键"""
    sid = str(surreal_id)
    if ":" in sid:
        return sid.split(":", 1)[1]
    return sid


def build_meili_doc(record: dict) -> dict:
    """从 SurrealDB 记录构建 Meilisearch 文档（不含 embedding 向量）"""
    record_id = str(record.get("id", ""))
    doc: dict = {
        "id": to_meili_id(record_id),
        "surreal_id": record_id,
        "content": record.get("content", ""),
        "tenant_id": record.get("tenant_id", "default"),
        "type": record.get("type", "general"),
        "tags": record.get("tags", []),
        "project_id": record.get("project_id", "global"),
    }
    if record.get("source_id"):
        doc["source_id"] = record["source_id"]
    if record.get("source_timestamp"):
        doc["date"] = str(record["source_timestamp"])
    if record.get("created_at"):
        doc["created_at"] = str(record["created_at"])
    return doc


async def count_records(db: AsyncSurreal, tenant_id: str | None) -> int:
    """统计 SurrealDB 记忆总数"""
    if tenant_id:
        result = await db.query(
            "SELECT count() AS total FROM memory WHERE tenant_id = $tenant_id GROUP ALL",
            {"tenant_id": tenant_id},
        )
    else:
        result = await db.query("SELECT count() AS total FROM memory GROUP ALL")

    # 解析结果
    if isinstance(result, list):
        for item in result:
            if isinstance(item, dict) and "total" in item:
                return int(item["total"])
            if isinstance(item, list):
                for sub in item:
                    if isinstance(sub, dict) and "total" in sub:
                        return int(sub["total"])
    return 0


async def fetch_batch(
    db: AsyncSurreal,
    offset: int,
    batch_size: int,
    tenant_id: str | None,
) -> list[dict]:
    """分页读取 SurrealDB 记忆"""
    if tenant_id:
        q = (
            "SELECT id, content, tenant_id, type, tags, project_id, "
            "source_id, source_timestamp, created_at "
            "FROM memory WHERE tenant_id = $tenant_id "
            f"ORDER BY created_at ASC LIMIT {batch_size} START {offset}"
        )
        result = await db.query(q, {"tenant_id": tenant_id})
    else:
        q = (
            "SELECT id, content, tenant_id, type, tags, project_id, "
            "source_id, source_timestamp, created_at "
            f"FROM memory ORDER BY created_at ASC LIMIT {batch_size} START {offset}"
        )
        result = await db.query(q)

    # 提取记录
    records: list[dict] = []
    if isinstance(result, list):
        for item in result:
            if isinstance(item, dict) and "id" in item:
                records.append(item)
            elif isinstance(item, list):
                for sub in item:
                    if isinstance(sub, dict) and "id" in sub:
                        records.append(sub)
    return records


async def migrate(
    surreal_url: str,
    surreal_ns: str,
    surreal_db: str,
    surreal_user: str,
    surreal_pass: str,
    meili_url: str,
    meili_api_key: str | None,
    meili_index: str,
    batch_size: int,
    tenant_id: str | None,
) -> None:
    """执行迁移"""
    logger.info("=" * 60)
    logger.info("SurrealDB → Meilisearch 数据迁移")
    logger.info("=" * 60)
    logger.info("SurrealDB: %s (ns=%s, db=%s)", surreal_url, surreal_ns, surreal_db)
    logger.info("Meilisearch: %s (index=%s)", meili_url, meili_index)
    logger.info("批量大小: %d", batch_size)
    if tenant_id:
        logger.info("租户过滤: %s", tenant_id)
    else:
        logger.info("租户过滤: 全部")

    # 1. 连接 Meilisearch
    meili = MeilisearchMigrator(meili_url, meili_api_key, meili_index)
    if not await meili.health():
        logger.error("❌ 无法连接 Meilisearch: %s", meili_url)
        await meili.close()
        sys.exit(1)
    logger.info("✅ Meilisearch 已连接")
    await meili.ensure_index()

    # 2. 连接 SurrealDB
    db = AsyncSurreal(surreal_url)
    try:
        await db.signin({"username": surreal_user, "password": surreal_pass})
        await db.use(surreal_ns, surreal_db)
        logger.info("✅ SurrealDB 已连接")
    except Exception as e:
        logger.error("❌ 无法连接 SurrealDB: %s", e)
        await meili.close()
        sys.exit(1)

    # 3. 统计总数
    total = await count_records(db, tenant_id)
    logger.info("📊 总记忆数: %d", total)

    if total == 0:
        logger.info("没有需要迁移的记忆，退出")
        await meili.close()
        return

    # 4. 分批迁移
    migrated = 0
    failed = 0
    start_time = time.monotonic()

    offset = 0
    while offset < total:
        records = await fetch_batch(db, offset, batch_size, tenant_id)
        if not records:
            break

        # 构建 Meilisearch 文档
        meili_docs = [build_meili_doc(r) for r in records]

        try:
            await meili.add_documents(meili_docs)
            migrated += len(meili_docs)
            elapsed = time.monotonic() - start_time
            rate = migrated / elapsed if elapsed > 0 else 0
            logger.info(
                "  ✅ 批次 %d-%d / %d (%.1f docs/s)",
                offset + 1,
                min(offset + len(records), total),
                total,
                rate,
            )
        except Exception as e:
            failed += len(meili_docs)
            logger.error("  ❌ 批次 %d-%d 失败: %s", offset + 1, offset + len(records), e)

        offset += batch_size

    # 5. 统计结果
    elapsed = time.monotonic() - start_time
    logger.info("=" * 60)
    logger.info("迁移完成!")
    logger.info("  总记录: %d", total)
    logger.info("  已迁移: %d", migrated)
    logger.info("  失败:   %d", failed)
    logger.info("  耗时:   %.1f 秒", elapsed)

    # 6. 验证
    stats = await meili.get_stats()
    if stats:
        logger.info("  Meilisearch 索引文档数: %s", stats.get("numberOfDocuments", "unknown"))

    await meili.close()

    if failed > 0:
        logger.warning("⚠️ 有 %d 条记忆迁移失败，请重新运行迁移脚本（幂等操作）", failed)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="SurrealDB → Meilisearch 数据迁移")
    parser.add_argument("--batch-size", type=int, default=100, help="每批迁移数量 (默认: 100)")
    parser.add_argument("--tenant-id", type=str, default=None, help="只迁移指定租户 (默认: 全部)")
    args = parser.parse_args()

    asyncio.run(
        migrate(
            surreal_url=os.getenv("SURREAL_URL", "ws://localhost:18800"),
            surreal_ns=os.getenv("SURREAL_NS", "memory"),
            surreal_db=os.getenv("SURREAL_DB", "memory"),
            surreal_user=os.getenv("SURREAL_USER", "root"),
            surreal_pass=os.getenv("SURREAL_PASS", "root"),
            meili_url=os.getenv("WRAPPER_MEILI_URL", "http://localhost:7700"),
            meili_api_key=os.getenv("WRAPPER_MEILI_API_KEY"),
            meili_index=os.getenv("WRAPPER_MEILI_INDEX_NAME", "memories"),
            batch_size=args.batch_size,
            tenant_id=args.tenant_id,
        )
    )


if __name__ == "__main__":
    main()
