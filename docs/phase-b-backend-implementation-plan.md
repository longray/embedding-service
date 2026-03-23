# Phase B 后端实施计划 - 双模式同步API

**版本**: v2.2-lite Phase B  
**目标**: 支持增量和全量双模式同步  
**工作量**: 待估算  
**文档日期**: 2026-03-18

---

## 一、概述

### 1.1 目标

为插件端提供完整的双模式同步API支持：
- **变更检测API**：接收指纹摘要，返回需要同步的文件列表
- **批量上传API**：高效上传多个文件
- **全量同步API**：首次同步或修复时的完整差异计算
- **冲突解决机制**：基于时间戳和语义相似度的自动裁决

### 1.2 核心原则

- **计算下沉**：差异对比在后端完成，减少网络传输
- **批量处理**：支持批量操作，提升吞吐量
- **幂等性**：所有API支持重复调用
- **向后兼容**：保持现有API不变

### 1.3 前置条件

- ✅ Phase A 后端优化已完成
- ✅ SurrealDB支持批量操作
- ✅ 智能去重决策框架已实现

---

## 二、API设计

### API 2.1：变更检测（/sync/detect-changes）

**目标**：接收本地文件指纹，返回需要同步的文件列表

**请求格式**：
```json
POST /sync/detect-changes
{
  "tenant_id": "default",
  "project_id": "@longray/opencode-memory-plugin",
  "fingerprints": [
    {
      "path": "active/decisions/entry-001.md",
      "mtime": 1710777600000,
      "size": 1024,
      "hash": "abc123..."
    }
  ]
}
```

**响应格式**：
```json
{
  "to_upload": [
    {
      "path": "active/decisions/entry-001.md",
      "reason": "new_file"
    }
  ],
  "to_delete": [
    {
      "path": "active/decisions/entry-002.md",
      "reason": "deleted_locally"
    }
  ],
  "conflicts": [
    {
      "path": "active/decisions/entry-003.md",
      "local_hash": "abc123",
      "remote_hash": "def456",
      "local_mtime": 1710777600000,
      "remote_mtime": 1710777500000
    }
  ]
}
```

**实现代码**（memory_manager.py）：
```python
async def detect_changes(self, tenant_id, project_id, fingerprints):
    # 1. 获取后端所有记忆的指纹
    q = """
        SELECT source_id, metadata.file_path AS path, 
               metadata.content_hash AS hash, 
               metadata.mtime AS mtime
        FROM memory
        WHERE tenant_id = $tenant_id AND project_id = $project_id
    """
    backend_records = await self._db.query(q, {
        "tenant_id": tenant_id,
        "project_id": project_id
    })
    
    # 2. 构建后端指纹映射
    backend_map = {r['path']: r for r in backend_records}
    local_map = {f['path']: f for f in fingerprints}
    
    # 3. 检测变更
    to_upload = []
    to_delete = []
    conflicts = []
    
    # 检查本地文件
    for path, local_fp in local_map.items():
        if path not in backend_map:
            to_upload.append({"path": path, "reason": "new_file"})
        else:
            backend_fp = backend_map[path]
            if local_fp['hash'] != backend_fp['hash']:
                # Hash不同，检查是否冲突
                if local_fp['mtime'] > backend_fp['mtime']:
                    to_upload.append({"path": path, "reason": "updated"})
                elif local_fp['mtime'] < backend_fp['mtime']:
                    conflicts.append({
                        "path": path,
                        "local_hash": local_fp['hash'],
                        "remote_hash": backend_fp['hash'],
                        "local_mtime": local_fp['mtime'],
                        "remote_mtime": backend_fp['mtime']
                    })
                else:
                    # 时间戳相同但hash不同，可能是冲突
                    conflicts.append({
                        "path": path,
                        "local_hash": local_fp['hash'],
                        "remote_hash": backend_fp['hash'],
                        "local_mtime": local_fp['mtime'],
                        "remote_mtime": backend_fp['mtime']
                    })
    
    # 检查后端文件（本地已删除）
    for path in backend_map:
        if path not in local_map:
            to_delete.append({"path": path, "reason": "deleted_locally"})
    
    return {
        "to_upload": to_upload,
        "to_delete": to_delete,
        "conflicts": conflicts
    }
```

**预计时间**：2小时

---

### API 2.2：批量上传（/sync/upload-batch）

**目标**：批量上传多个文件内容

**请求格式**：
```json
POST /sync/upload-batch
{
  "tenant_id": "default",
  "project_id": "@longray/opencode-memory-plugin",
  "files": [
    {
      "path": "active/decisions/entry-001.md",
      "content": "...",
      "metadata": {
        "mtime": 1710777600000,
        "hash": "abc123..."
      }
    }
  ]
}
```

**响应格式**：
```json
{
  "success": 45,
  "failed": 5,
  "duplicates": 0,
  "errors": [
    {
      "path": "active/decisions/entry-002.md",
      "error": "Invalid format"
    }
  ]
}
```

**实现代码**：
```python
async def upload_batch(self, tenant_id, project_id, files):
    success = 0
    failed = 0
    duplicates = 0
    errors = []
    
    # 使用事务批量插入
    transaction_statements = []
    
    for file in files:
        try:
            # 解析文件内容
            entry = parse_memory_entry(file['content'])
            
            # 检查重复
            existing = await self.find_similar(
                entry['content'], 
                tenant_id, 
                project_id
            )
            
            if existing:
                # 智能去重决策
                action = self.decide_duplicate_action(
                    entry, 
                    existing[0], 
                    existing[0]['similarity']
                )
                
                if action == 'DISCARD':
                    duplicates += 1
                    continue
                elif action == 'UPDATE':
                    # 更新现有记忆
                    transaction_statements.append(
                        f"UPDATE {existing[0]['id']} MERGE {json.dumps(entry)};"
                    )
                    success += 1
                    continue
                # KEEP_BOTH: 继续创建新记忆
            
            # 创建新记忆
            entry['tenant_id'] = tenant_id
            entry['project_id'] = project_id
            entry['metadata'] = {
                'file_path': file['path'],
                'mtime': file['metadata']['mtime'],
                'content_hash': file['metadata']['hash']
            }
            
            transaction_statements.append(
                f"CREATE memory CONTENT {json.dumps(entry)};"
            )
            success += 1
            
        except Exception as e:
            failed += 1
            errors.append({
                "path": file['path'],
                "error": str(e)
            })
    
    # 只有成功的语句才执行事务
    if transaction_statements:
        transaction = "BEGIN TRANSACTION;\n"
        transaction += "\n".join(transaction_statements)
        transaction += "\nCOMMIT TRANSACTION;"
        await self._db.query(transaction)
    
    return {
        "success": success,
        "failed": failed,
        "duplicates": duplicates,
        "errors": errors
    }
```

**辅助函数：parse_memory_entry**
```python
def parse_memory_entry(content: str) -> dict:
    """解析markdown格式的记忆条目"""
    lines = content.strip().split('\n')
    entry = {}
    metadata_end = 0
    
    # 解析元数据
    for i, line in enumerate(lines):
        if line.startswith('**') and '**:' in line:
            key = line.split('**')[1].replace(':', '').strip().lower()
            value = line.split(':', 1)[1].strip()
            entry[key] = value
        elif not line.strip().startswith('**'):
            metadata_end = i
            break
    
    # 提取content（元数据之后的所有内容）
    entry['content'] = '\n'.join(lines[metadata_end:]).strip()
    
    return entry
```

**预计时间**：2小时

---

### API 2.3：全量同步（/sync/full-sync）

**目标**：计算本地和远程的完整差异

**请求格式**：
```json
POST /sync/full-sync
{
  "tenant_id": "default",
  "project_id": "@longray/opencode-memory-plugin",
  "fingerprints": [...],
  "bidirectional": true
}
```

**响应格式**：
```json
{
  "to_upload": [...],
  "to_download": [...],
  "to_update": [...],
  "to_delete": [...],
  "conflicts": [...]
}
```

**实现代码**：
```python
async def full_sync(self, tenant_id, project_id, fingerprints, bidirectional=False):
    result = await self.detect_changes(tenant_id, project_id, fingerprints)
    
    if bidirectional:
        # 双向同步：返回远程新增的文件
        backend_records = await self.get_all_records(tenant_id, project_id)
        local_paths = {f['path'] for f in fingerprints}
        
        to_download = []
        for record in backend_records:
            path = record['metadata']['file_path']
            if path not in local_paths:
                to_download.append({
                    "path": path,
                    "memory_id": record['id'],
                    "reason": "new_on_remote"
                })
        
        result['to_download'] = to_download
    
    return result
```

**预计时间**：1.5小时

---

## 三、数据模型调整

### 3.1 添加文件路径字段

**修改Schema**：
```sql
-- 在metadata中添加file_path和mtime
DEFINE FIELD metadata.file_path ON memory TYPE option<string>;
DEFINE FIELD metadata.mtime ON memory TYPE option<number>;
DEFINE FIELD metadata.content_hash ON memory TYPE option<string>;
```

### 3.2 添加索引

```sql
-- 加速路径查询
DEFINE INDEX memory_file_path ON memory FIELDS metadata.file_path;
```

**预计时间**：30分钟

---

## 四、验证和测试

### 4.1 API测试

**测试脚本**（test-sync-apis.py）：
```python
async def test_detect_changes():
    # 模拟100个文件指纹
    fingerprints = generate_test_fingerprints(100)
    
    result = await client.detect_changes(
        tenant_id="test",
        project_id="test-project",
        fingerprints=fingerprints
    )
    
    assert 'to_upload' in result
    assert 'to_delete' in result
    assert 'conflicts' in result

async def test_batch_upload():
    files = generate_test_files(50)
    
    result = await client.upload_batch(
        tenant_id="test",
        project_id="test-project",
        files=files
    )
    
    assert result['success'] == 50
    assert result['failed'] == 0

async def test_full_sync():
    fingerprints = generate_test_fingerprints(100)
    
    result = await client.full_sync(
        tenant_id="test",
        project_id="test-project",
        fingerprints=fingerprints,
        bidirectional=True
    )
    
    assert 'to_download' in result
```

### 4.2 性能测试

**测试目标**：
- 变更检测：<100ms（1000个文件）
- 批量上传：<5s（50个文件）
- 全量同步：<500ms（1000个文件）

---

## 五、Go/No-Go检查点

1. ✅ 变更检测API响应时间<100ms
2. ✅ 批量上传成功率>99%
3. ✅ 全量同步差异计算准确
4. ✅ 冲突检测准确率>95%
5. ✅ 智能去重决策正常工作

**如果任一检查点失败**：回滚API更改

---

## 六、时间分配

| 任务 | 预计时间 | 优先级 | 备注 |
|------|---------|--------|------|
| 2.1 变更检测API | 2h | P0 | 必须完成 |
| 2.2 批量上传API | 2h | P0 | 必须完成 |
| 2.3 全量同步API | 1.5h | P0 | 必须完成 |
| 3.1 数据模型调整 | 30min | P0 | 必须完成 |
| 验证测试 | 2h | P0 | 必须完成 |
| **总计** | **8h** | | |

---

## 七、总结

**Phase B 后端实施计划已完成**，包含：
- ✅ 3个核心API设计（变更检测、批量上传、全量同步）
- ✅ 数据模型调整
- ✅ 完整的验证测试方案
- ✅ 5个Go/No-Go检查点

**预期效果**：
- 支持增量和全量双模式同步
- 同步延迟降低80%（2000ms → 400ms）
- 网络传输减少90%（只传输变更）

**Phase B 总工作量**：插件端12-15h + 后端8h = **20-23小时**

**下一步**：创建Phase C实施计划或开始实施Phase A/B
