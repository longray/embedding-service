# Backlog - BL-CA-07/08 单元测试

> **关联实现**: BL-CA-07 (code-fingerprints API) + BL-CA-08 (code upsert)  
> **测试文件**: `tests/test_code_sync.py`

---

## 使用场景（测试覆盖目标）

### 场景 1：全新项目首次同步
- 本地有 10 个代码文件
- 服务端没有任何记录
- 预期：全部 10 个文件返回 missing

### 场景 2：部分文件变更
- 本地 10 个文件，服务端已有 10 个记录
- 其中 3 个文件内容变更，2 个仅符号变更，5 个完全一致
- 预期：3 个 changed(content)，2 个 changed(symbols)，5 个 unchanged

### 场景 3：mtime 冲突检测
- 本地文件 mtime < 服务端 mtime
- 预期：返回 conflicts 列表

### 场景 4：Upsert 逻辑
- 上传已存在的代码文件（相同 file_path + project_id）
- 预期：UPDATE 而非 INSERT，返回 updated 计数

---

## Backlog 项

### BL-CA-07-TEST: sync_code_fingerprints 单元测试

| 字段 | 内容 |
|------|------|
| **目标** | 为 `MemoryManager.sync_code_fingerprints()` 编写完整的单元测试，覆盖 changed/unchanged/missing/conflicts 四种场景 |
| **涉及范围** | `tests/test_code_sync.py` 新增测试类 `TestSyncCodeFingerprints` |
| **前置依赖** | BL-CA-07 已实现、pytest + AsyncMock 可用 |
| **完成标准** | ① 测试全部 4 种分类场景 ② 测试空列表输入 ③ 测试无 file_path 的异常指纹 ④ 使用 AsyncMock 模拟 `_db_query` 和 `_extract_records` |
| **验证方式** | `uv run pytest tests/test_code_sync.py -v` 全部通过 |

**测试用例清单**:
- `test_all_missing` - 全新文件，全部 missing
- `test_all_unchanged` - 完全一致，全部 unchanged
- `test_content_changed` - 内容变更，reason="content_modified"
- `test_symbols_changed` - 符号变更，reason="symbols_modified"
- `test_mtime_conflict` - mtime 冲突，归入 conflicts
- `test_mixed_scenarios` - 混合场景（2 missing + 2 changed + 2 unchanged + 1 conflict）
- `test_empty_fingerprints` - 空列表输入
- `test_fingerprint_without_path` - 无 path 字段的指纹应被跳过

---

### BL-CA-08-TEST: code upsert 单元测试

| 字段 | 内容 |
|------|------|
| **目标** | 为 `upload_memories` 中的 code upsert 逻辑编写单元测试，验证 file_path + project_id 存在时执行 UPDATE |
| **涉及范围** | `tests/test_code_sync.py` 新增测试类 `TestCodeUpsert` |
| **前置依赖** | BL-CA-08 已实现、pytest + AsyncMock 可用 |
| **完成标准** | ① 测试 code 类型文件 upsert ② 测试非 code 类型走原有流程 ③ 测试无 file_path 的 code 记忆走原有流程 ④ 验证 updated_count 正确 |
| **验证方式** | `uv run pytest tests/test_code_sync.py -v` 全部通过 |

**测试用例清单**:
- `test_code_file_upsert_existing` - 已存在的代码文件，执行 UPDATE
- `test_code_file_insert_new` - 新代码文件，执行 INSERT
- `test_non_code_type_no_upsert` - 非 code 类型，不走 upsert 逻辑
- `test_code_without_file_path` - code 类型但无 file_path，走原有流程
- `test_upsert_updates_correct_record` - 验证更新的是正确的记录 ID

---

## 依赖关系

```text
BL-CA-07 (实现)
    ↓
BL-CA-07-TEST (测试)

BL-CA-08 (实现)
    ↓
BL-CA-08-TEST (测试)
```

## 执行顺序

1. 先实现 BL-CA-07-TEST（不依赖 BL-CA-08）
2. 再实现 BL-CA-08-TEST
3. 一起跑测试验证

---

## 技术要点

### Mock 策略

```python
# Mock _db_query 返回服务端记录
mock_db_query = AsyncMock(return_value=[
    {
        "id": "memory:abc123",
        "content_hash": "hash456",
        "mtime": 1712345000,
        "metadata": {
            "file_path": "src/index.js",
            "symbols_hash": "sym789"
        }
    }
])

# Mock _extract_records 解析结果
mock_extract = MagicMock(return_value=[...])
```

### 测试数据示例

```python
# 本地指纹
local_fingerprints = [
    {
        "path": "src/index.js",
        "hash": "hash456",  # 与 server 相同
        "symbols_hash": "sym789",  # 与 server 相同
        "mtime": 1712345000,
        "size": 1024
    },
    {
        "path": "src/new.ts",
        "hash": "hash999",  # server 没有
        "symbols_hash": "sym000",
        "mtime": 1712346000,
        "size": 512
    }
]
```

---

*创建时间: 2026-03-31*
