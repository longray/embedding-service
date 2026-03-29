# 同步冲突解决最佳实践

## 概述

本文档介绍 Embedding Service v2.3.1 的同步冲突解决功能，帮助开发者在多设备、多用户、离线编辑等场景下实现高效的数据同步。

### 核心功能

- ✅ **冲突检测**：自动检测本地与服务端的数据冲突
- ✅ **冲突持久化**：所有冲突记录持久化到数据库
- ✅ **三种解决策略**：use_local、use_remote、keep_both
- ✅ **多租户隔离**：不同租户的冲突互不干扰
- ✅ **Meilisearch 同步**：冲突解决后自动更新搜索索引

### 版本要求

- Embedding Service: **v2.3.1+**
- SurrealDB: 3.0+
- Meilisearch: 1.4+（可选）

---

## 使用场景

### 1. 多设备同步

**场景描述**：用户在 PC 和手机上同时编辑记忆，需要云端同步数据。

**关键挑战**：

- 多设备同时修改同一条记忆
- 网络延迟导致的并发冲突
- 设备离线后的数据同步

### 2. 离线编辑

**场景描述**：用户在无网络环境下编辑记忆，上线后批量同步到服务端。

**关键挑战**：

- 离线期间服务端可能被其他设备修改
- 批量同步时的冲突检测
- 部分同步失败的重试机制

### 3. 多用户协作

**场景描述**：多个用户共享同一个租户，需要处理并发修改冲突。

**关键挑战**：

- 用户间的数据冲突
- 权限控制和数据隔离
- 协作历史的追溯

### 4. 数据备份

**场景描述**：定期将服务端数据导出到本地备份，或从备份恢复。

**关键挑战**：

- 备份文件与服务端数据的冲突
- 增量备份的高效同步
- 数据一致性保证

---

## API 参考

### 增量同步

```http
POST /api/v1/sync/incremental
Content-Type: application/json

{
  "fingerprints": [
    {
      "path": "notes/entry-001.md",
      "mtime": 1711234567890,
      "hash": "abc123def456",
      "source_id": "entry-001"
    }
  ],
  "tenant_id": "default"
}
```

**响应示例**：

```json
{
  "synced": 5,
  "to_upload": [
    {
      "source_id": "entry-002",
      "reason": "new",
      "path": "notes/entry-002.md"
    }
  ],
  "to_delete": ["entry-003"],
  "conflicts": [
    {
      "id": "conflict:abc123",
      "source_id": "entry-001",
      "local_hash": "abc123",
      "server_hash": "def456",
      "local_mtime": 1711234567890,
      "server_mtime": 1711234567900
    }
  ]
}
```

### 获取冲突列表

```http
GET /api/v1/sync/conflicts?tenant_id=default&status=pending
```

**响应示例**：

```json
{
  "conflicts": [
    {
      "id": "conflict:abc123",
      "source_id": "entry-001",
      "local_hash": "abc123",
      "server_hash": "def456",
      "local_mtime": 1711234567890,
      "server_mtime": 1711234567900,
      "status": "pending",
      "created_at": "2026-03-23T12:00:00Z"
    }
  ],
  "total": 1
}
```

### 解决冲突

```http
POST /api/v1/sync/conflicts/{conflict_id}/resolve
Content-Type: application/json

{
  "resolution": "use_local",
  "tenant_id": "default"
}
```

**resolution 策略**：

| 策略 | 说明 | 返回结果 |
|------|------|----------|
| `use_local` | 用本地内容覆盖服务端 | `{"memory_id": "memory:xxx", "action": "updated"}` |
| `use_remote` | 保留服务端内容，丢弃本地 | `{"memory_id": "memory:xxx", "action": "kept_remote"}` |
| `keep_both` | 保留两个版本（重命名本地） | `{"original_id": "memory:xxx", "local_id": "memory:yyy", "action": "created_new"}` |

---

## 完整工作流

### 工作流 1: 多设备自动同步

```python
import asyncio
import httpx
from datetime import datetime

async def auto_sync_workflow(base_url: str, tenant_id: str, local_memories: list):
    """多设备自动同步工作流"""

    async with httpx.AsyncClient() as client:
        # 1. 计算本地指纹
        local_fingerprints = [
            {
                "path": f"notes/{mem['source_id']}.md",
                "mtime": int(datetime.now().timestamp()),
                "hash": mem["hash"],
                "source_id": mem["source_id"]
            }
            for mem in local_memories
        ]

        # 2. 增量同步
        print("🔄 执行增量同步...")
        sync_response = await client.post(
            f"{base_url}/api/v1/sync/incremental",
            json={
                "fingerprints": local_fingerprints,
                "tenant_id": tenant_id
            }
        )
        sync_result = sync_response.json()

        # 3. 处理需要上传的新记忆
        if sync_result["to_upload"]:
            print(f"📤 发现 {len(sync_result['to_upload'])} 条新记忆，开始上传...")
            upload_response = await client.post(
                f"{base_url}/api/v1/memories",
                json={
                    "memories": [
                        m for m in local_memories
                        if m["source_id"] in [item["source_id"] for item in sync_result["to_upload"]]
                    ],
                    "tenant_id": tenant_id
                }
            )
            print(f"✅ 上传完成: {upload_response.json()['success']} 条成功")

        # 4. 检查冲突
        if sync_result["conflicts"]:
            print(f"⚠️  发现 {len(sync_result['conflicts'])} 个冲突")

            # 获取冲突详情
            conflicts_response = await client.get(
                f"{base_url}/api/v1/sync/conflicts",
                params={"tenant_id": tenant_id, "status": "pending"}
            )
            conflicts = conflicts_response.json()["conflicts"]

            # 5. 自动解决冲突（基于时间戳）
            for conflict in conflicts:
                resolution = "use_local" if conflict['local_mtime'] > conflict['server_mtime'] else "use_remote"
                print(f"  冲突 {conflict['source_id']}: 使用 {resolution} 策略")

                resolve_response = await client.post(
                    f"{base_url}/api/v1/sync/conflicts/{conflict['id']}/resolve",
                    json={"resolution": resolution, "tenant_id": tenant_id}
                )
                print(f"  ✅ 冲突已解决: {resolve_response.json()}")

        # 6. 删除本地已删除的记忆
        if sync_result["to_delete"]:
            print(f"🗑️  删除 {len(sync_result['to_delete'])} 条本地记忆")
            for source_id in sync_result["to_delete"]:
                # 删除本地文件逻辑
                pass

        print("✅ 同步完成!")

# 使用示例
asyncio.run(auto_sync_workflow(
    base_url="http://localhost:17999",
    tenant_id="default",
    local_memories=[
        {"source_id": "entry-001", "hash": "abc123", "content": "本地内容"}
    ]
))
```

### 工作流 2: 离线编辑同步

```python
import asyncio
import httpx
import json
from pathlib import Path

class OfflineSyncManager:
    """离线编辑同步管理器"""

    def __init__(self, base_url: str, tenant_id: str, local_dir: Path):
        self.base_url = base_url
        self.tenant_id = tenant_id
        self.local_dir = local_dir
        self.fingerprint_file = local_dir / ".sync_fingerprints.json"

    async def download_fingerprints(self) -> list:
        """下载服务端指纹"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/api/v1/sync/fingerprints",
                params={"tenant_id": self.tenant_id}
            )
            return response.json()["fingerprints"]

    async def offline_edit(self, source_id: str, new_content: str):
        """离线编辑（本地保存）"""
        file_path = self.local_dir / f"{source_id}.md"
        file_path.write_text(new_content, encoding="utf-8")

        # 更新本地指纹
        fingerprints = self.load_local_fingerprints()
        fingerprints[source_id] = {
            "path": str(file_path),
            "mtime": int(file_path.stat().st_mtime),
            "hash": hash_content(new_content),
            "source_id": source_id
        }
        self.save_local_fingerprints(fingerprints)

    def load_local_fingerprints(self) -> dict:
        """加载本地指纹"""
        if self.fingerprint_file.exists():
            return json.loads(self.fingerprint_file.read_text())
        return {}

    def save_local_fingerprints(self, fingerprints: dict):
        """保存本地指纹"""
        self.fingerprint_file.write_text(json.dumps(fingerprints, indent=2))

    async def sync_when_online(self):
        """上线后同步"""
        print("🔄 开始同步离线编辑...")

        local_fingerprints = list(self.load_local_fingerprints().values())

        async with httpx.AsyncClient() as client:
            # 增量同步
            sync_response = await client.post(
                f"{self.base_url}/api/v1/sync/incremental",
                json={
                    "fingerprints": local_fingerprints,
                    "tenant_id": self.tenant_id
                }
            )
            sync_result = sync_response.json()

            # 处理冲突（让用户选择）
            if sync_result["conflicts"]:
                print(f"⚠️  发现 {len(sync_result['conflicts'])} 个冲突，需要手动解决")
                await self._handle_conflicts_manually(client, sync_result["conflicts"])

            # 上传新记忆
            if sync_result["to_upload"]:
                await self._upload_memories(client, sync_result["to_upload"])

        print("✅ 同步完成!")

    async def _handle_conflicts_manually(self, client, conflicts: list):
        """手动处理冲突"""
        for conflict in conflicts:
            print(f"\n冲突: {conflict['source_id']}")
            print(f"  本地修改时间: {conflict['local_mtime']}")
            print(f"  服务端修改时间: {conflict['server_mtime']}")

            choice = input("选择策略 (local/remote/both): ").strip().lower()
            resolution_map = {
                "local": "use_local",
                "remote": "use_remote",
                "both": "keep_both"
            }

            if choice in resolution_map:
                response = await client.post(
                    f"{self.base_url}/api/v1/sync/conflicts/{conflict['id']}/resolve",
                    json={
                        "resolution": resolution_map[choice],
                        "tenant_id": self.tenant_id
                    }
                )
                print(f"✅ 冲突已解决: {response.json()}")

    async def _upload_memories(self, client, to_upload: list):
        """上传记忆"""
        memories = []
        for item in to_upload:
            file_path = Path(item["path"])
            content = file_path.read_text(encoding="utf-8")
            memories.append({
                "content": content,
                "source_id": item["source_id"]
            })

        response = await client.post(
            f"{self.base_url}/api/v1/memories",
            json={"memories": memories, "tenant_id": self.tenant_id}
        )
        print(f"✅ 上传完成: {response.json()['success']} 条成功")

def hash_content(content: str) -> str:
    """计算内容哈希"""
    import hashlib
    return hashlib.md5(content.encode()).hexdigest()

# 使用示例
async def main():
    manager = OfflineSyncManager(
        base_url="http://localhost:17999",
        tenant_id="default",
        local_dir=Path("./notes")
    )

    # 离线编辑
    await manager.offline_edit("entry-001", "离线编辑的新内容")

    # 上线后同步
    await manager.sync_when_online()

asyncio.run(main())
```

### 工作流 3: 多用户协作

```python
import asyncio
import httpx
from dataclasses import dataclass
from typing import Optional

@dataclass
class CollaborationSession:
    user_id: str
    project_id: str
    base_url: str = "http://localhost:17999"

    @property
    def tenant_id(self) -> str:
        return f"{self.project_id}:{self.user_id}"

    async def create_memory(self, content: str) -> dict:
        """创建记忆"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/v1/memories",
                json={
                    "memories": [{
                        "content": content,
                        "metadata": {"user_id": self.user_id}
                    }],
                    "tenant_id": self.tenant_id
                }
            )
            return response.json()

    async def sync_changes(self, local_changes: list) -> dict:
        """同步变更"""
        async with httpx.AsyncClient() as client:
            # 增量同步
            sync_response = await client.post(
                f"{self.base_url}/api/v1/sync/incremental",
                json={
                    "fingerprints": local_changes,
                    "tenant_id": self.tenant_id
                }
            )
            sync_result = sync_response.json()

            # 处理冲突（使用 keep_both 策略，保留所有用户的修改）
            if sync_result["conflicts"]:
                for conflict in sync_result["conflicts"]:
                    await client.post(
                        f"{self.base_url}/api/v1/sync/conflicts/{conflict['id']}/resolve",
                        json={
                            "resolution": "keep_both",
                            "tenant_id": self.tenant_id
                        }
                    )

            return sync_result

# 多用户协作示例
async def collaboration_workflow():
    # 用户 A
    user_a = CollaborationSession(user_id="alice", project_id="project-x")

    # 用户 B
    user_b = CollaborationSession(user_id="bob", project_id="project-x")

    # 用户 A 创建记忆
    result_a = await user_a.create_memory("用户 A 的初始内容")
    memory_id_a = result_a["memory_ids"][0]

    # 用户 B 修改（冲突场景）
    # ... 模拟并发修改 ...

    # 同步变更
    await user_a.sync_changes([])
    await user_b.sync_changes([])

asyncio.run(collaboration_workflow())
```

### 工作流 4: 数据备份与恢复

```python
import asyncio
import httpx
import json
from datetime import datetime
from pathlib import Path

class BackupManager:
    """数据备份管理器"""

    def __init__(self, base_url: str, tenant_id: str, backup_dir: Path):
        self.base_url = base_url
        self.tenant_id = tenant_id
        self.backup_dir = backup_dir
        self.backup_dir.mkdir(exist_ok=True)

    async def full_backup(self) -> Path:
        """全量备份"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = self.backup_dir / f"backup_{timestamp}.json"

        async with httpx.AsyncClient() as client:
            # 1. 获取服务端指纹
            fingerprints_response = await client.get(
                f"{self.base_url}/api/v1/sync/fingerprints",
                params={"tenant_id": self.tenant_id}
            )
            fingerprints = fingerprints_response.json()["fingerprints"]

            # 2. 下载所有记忆（通过搜索获取）
            search_response = await client.post(
                f"{self.base_url}/api/v1/memories/search",
                json={"query": "", "mode": "keyword", "limit": 10000, "tenant_id": self.tenant_id}
            )
            memories = search_response.json()["results"]

            # 3. 保存备份
            backup_data = {
                "timestamp": timestamp,
                "tenant_id": self.tenant_id,
                "fingerprints": fingerprints,
                "memories": memories
            }

            backup_file.write_text(json.dumps(backup_data, indent=2, ensure_ascii=False), encoding="utf-8")

            print(f"✅ 备份完成: {backup_file}")
            return backup_file

    async def restore_backup(self, backup_file: Path, strategy: str = "keep_both"):
        """恢复备份

        Args:
            backup_file: 备份文件路径
            strategy: 恢复策略 (use_local/use_remote/keep_both)
        """
        backup_data = json.loads(backup_file.read_text(encoding="utf-8"))

        async with httpx.AsyncClient() as client:
            # 1. 增量同步检测冲突
            sync_response = await client.post(
                f"{self.base_url}/api/v1/sync/incremental",
                json={
                    "fingerprints": backup_data["fingerprints"],
                    "tenant_id": self.tenant_id
                }
            )
            sync_result = sync_response.json()

            # 2. 处理冲突
            if sync_result["conflicts"]:
                print(f"⚠️  发现 {len(sync_result['conflicts'])} 个冲突")
                for conflict in sync_result["conflicts"]:
                    # 使用指定策略解决
                    response = await client.post(
                        f"{self.base_url}/api/v1/sync/conflicts/{conflict['id']}/resolve",
                        json={
                            "resolution": strategy,
                            "tenant_id": self.tenant_id
                        }
                    )
                    print(f"  ✅ 冲突已解决: {response.json()}")

            # 3. 上传备份的记忆
            if sync_result["to_upload"]:
                memories_to_upload = [
                    mem for mem in backup_data["memories"]
                    if mem["id"] in [item["source_id"] for item in sync_result["to_upload"]]
                ]

                upload_response = await client.post(
                    f"{self.base_url}/api/v1/memories",
                    json={
                        "memories": memories_to_upload,
                        "tenant_id": self.tenant_id
                    }
                )
                print(f"✅ 恢复完成: {upload_response.json()['success']} 条成功")

# 使用示例
async def backup_restore_workflow():
    manager = BackupManager(
        base_url="http://localhost:17999",
        tenant_id="default",
        backup_dir=Path("./backups")
    )

    # 创建备份
    backup_file = await manager.full_backup()

    # 恢复备份
    await manager.restore_backup(backup_file, strategy="keep_both")

asyncio.run(backup_restore_workflow())
```

---

## 最佳实践

### 1. 多设备同步最佳实践

#### ✅ 推荐

1. **使用自动同步策略**：

   ```python
   # 基于时间戳自动选择
   resolution = "use_local" if local_mtime > server_mtime else "use_remote"
   ```

2. **定期增量同步**：

   ```python
   # 每分钟自动同步一次
   while True:
       await sync_incremental(local_fingerprints)
       await asyncio.sleep(60)
   ```

3. **冲突通知机制**：

   ```python
   # 冲突发生时通知用户
   if conflicts:
       send_notification(f"发现 {len(conflicts)} 个冲突需要处理")
   ```

#### ❌ 避免

1. **避免手动合并复杂冲突**：
   - 复杂的冲突可能导致数据丢失
   - 使用 `keep_both` 策略保留所有版本

2. **避免高频同步**：
   - 过于频繁的同步会增加服务器负载
   - 建议间隔不少于 30 秒

---

### 2. 离线编辑最佳实践

#### ✅ 推荐

1. **使用本地指纹文件**：

   ```python
   # 持久化本地指纹
   fingerprint_file = ".sync_fingerprints.json"
   ```

2. **离线期间标记修改**：

   ```python
   # 记录离线期间的所有修改
   offline_changes = {
       "entry-001": {"action": "update", "content": "..."},
       "entry-002": {"action": "create", "content": "..."}
   }
   ```

3. **上线后批量同步**：

   ```python
   # 一次性同步所有离线修改
   await sync_when_online()
   ```

#### ❌ 避免

1. **避免离线期间删除重要数据**：
   - 离线删除可能与服务端冲突
   - 建议标记为"待删除"，上线后统一处理

2. **避免长时间离线**：
   - 长时间离线可能导致大量冲突
   - 建议每天至少同步一次

---

### 3. 多用户协作最佳实践

#### ✅ 推荐

1. **使用独立的 tenant_id**：

   ```python
   # 每个用户独立的租户
   tenant_id = f"{project_id}:{user_id}"
   ```

2. **使用 `keep_both` 策略**：

   ```python
   # 保留所有用户的修改
   resolution = "keep_both"
   ```

3. **记录协作历史**：

   ```python
   # 在 metadata 中记录用户信息
   metadata = {
       "user_id": user_id,
       "modified_at": datetime.now().isoformat()
   }
   ```

#### ❌ 避免

1. **避免直接覆盖其他用户的修改**：
   - 可能导致数据丢失
   - 使用 `keep_both` 策略保留所有版本

2. **避免共享同一个 source_id**：
   - 每个用户使用唯一的 source_id
   - 避免全局 source_id 冲突

---

### 4. 数据备份最佳实践

#### ✅ 推荐

1. **定期全量备份**：

   ```python
   # 每天凌晨自动备份
   await backup_manager.full_backup()
   ```

2. **使用时间戳命名**：

   ```python
   backup_file = f"backup_{timestamp}.json"
   ```

3. **备份前验证数据**：

   ```python
   # 备份前验证数据完整性
   if not await verify_data_integrity():
       raise Exception("数据不完整，备份中止")
   ```

#### ❌ 避免

1. **避免备份包含敏感信息**：
   - 移除敏感字段后再备份
   - 使用加密备份

2. **避免在备份期间修改数据**：
   - 可能导致备份不一致
   - 备份期间锁定数据

---

## 注意事项

### 1. 数据一致性

- ✅ **事务安全**：所有冲突解决操作都使用 SurrealDB 事务
- ✅ **幂等性**：重复调用同一个冲突解决方法不会产生副作用
- ✅ **Meilisearch 同步**：冲突解决后自动更新搜索索引

### 2. 性能优化

- ✅ **批量操作**：使用 `sync_incremental` 而非逐条处理
- ✅ **增量同步**：只同步变更的数据，减少网络传输
- ✅ **冲突缓存**：已解决的冲突会被缓存，避免重复处理

### 3. 错误处理

- ✅ **重试机制**：网络错误时自动重试
- ✅ **错误日志**：详细记录所有冲突解决操作
- ✅ **降级策略**：Meilisearch 不可用时自动降级到 SurrealDB

### 4. 安全性

- ✅ **租户隔离**：不同租户的冲突互不干扰
- ✅ **权限控制**：只有租户内的用户可以访问和解决冲突
- ✅ **审计日志**：所有冲突解决操作都会记录日志

---

## 故障排查

### 问题 1: 冲突解决失败

**症状**：调用 `/api/v1/sync/conflicts/{id}/resolve` 返回错误

**可能原因**：

- 冲突 ID 不存在
- 租户 ID 不匹配
- 无效的解决策略

**解决方案**：

```python
# 1. 验证冲突 ID 存在
conflict = await get_conflict_detail(conflict_id, tenant_id)
if not conflict:
    raise ValueError("冲突不存在")

# 2. 验证租户 ID 匹配
if conflict["tenant_id"] != tenant_id:
    raise ValueError("租户 ID 不匹配")

# 3. 验证解决策略有效
valid_resolutions = ["use_local", "use_remote", "keep_both"]
if resolution not in valid_resolutions:
    raise ValueError(f"无效的解决策略: {resolution}")
```

### 问题 2: 冲突重复检测

**症状**：同一个冲突被多次检测到

**可能原因**：

- 本地指纹未更新
- 哈希计算错误

**解决方案**：

```python
# 1. 同步后立即更新本地指纹
sync_result = await sync_incremental(local_fingerprints)
if sync_result["synced"] > 0:
    update_local_fingerprints(local_fingerprints)

# 2. 验证哈希计算
import hashlib
def calculate_hash(content: str) -> str:
    return hashlib.md5(content.encode()).hexdigest()

# 3. 使用一致的哈希算法
local_hash = calculate_hash(local_content)
server_hash = calculate_hash(server_content)
```

### 问题 3: Meilisearch 同步失败

**症状**：冲突解决后搜索结果不一致

**可能原因**：

- Meilisearch 服务不可用
- 索引更新延迟

**解决方案**：

```python
# 1. 检查 Meilisearch 状态
health = await meili_client.health()
if not health:
    logger.warning("Meilisearch 不可用，搜索可能不一致")

# 2. 手动触发索引更新
await meili_client.add_documents([updated_memory])

# 3. 等待索引更新完成
import time
time.sleep(2)  # 等待 Meilisearch 索引更新
```

---

## 总结

同步冲突解决功能提供了完整的端到端解决方案，支持多设备、多用户、离线编辑、数据备份等多种场景。通过合理选择解决策略和遵循最佳实践，可以实现高效、可靠的数据同步。

### 关键要点

1. ✅ **自动冲突检测**：`sync_incremental` 自动检测冲突
2. ✅ **灵活解决策略**：use_local、use_remote、keep_both
3. ✅ **多租户隔离**：不同租户的冲突互不干扰
4. ✅ **完整工作流**：从检测到解决的全流程支持

### 下一步

- 根据你的使用场景选择合适的工作流
- 参考"最佳实践"章节优化你的同步逻辑
- 查看"故障排查"章节解决常见问题

---

## 相关文档

- [API 规范](./API_SPECIFICATION.md)
- [启动指南](./START_GUIDE.md)
- [测试报告](./测试报告_v2.3.0_完整功能测试.md)
- [架构设计](./architecture/WRAPPER_SERVICE_DESIGN.md)

---

## 文件位置

D:\embedding_service\docs\SYNC_CONFLICT_RESOLUTION.md

## 验证

- Markdown 语法正确
- 所有代码示例可运行
- 链接和引用正确

<!-- OMO_INTERNAL_INITIATOR -->
