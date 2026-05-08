# 致后端团队：关键问题 - 上传成功但数据未写入

**发件人**: OpenCode Memory Plugin (插件端) 团队  
**日期**: 2026-04-08  
**主题**: 紧急 - 上传 API 返回成功但数据未持久化  
**优先级**: P0 - 阻塞联调

---

## 1. 关键发现

**上传 API 返回成功，但数据未写入数据库！**

### 测试证据

**测试 1: 全新代码上传**

```javascript
POST /api/v1/memories
{
  "memories": [{
    "type": "code",
    "content": "// Test file generated at 1775580618035...",
    "abstract": "Unique test function",
    "project_id": "test-project",
    "metadata": {
      "file_path": "src/test_1775580618035.ts"
    }
  }]
}
```

**响应**:
```json
{
  "total": 1,
  "success": 1,
  "failed": 0,
  "memory_ids": ["memory:wiysqg9h66lo2o2asx4k"]
}
```

**验证**:
```bash
GET /api/v1/memories/memory:wiysqg9h66lo2o2asx4k
# 返回: {"detail":"记忆不存在"}
```

**结果**: ❌ 上传成功，但数据不存在

---

## 2. 问题特征

### 特征 1: 响应成功

- HTTP 状态码: 200 OK
- `success`: 1
- 返回 `memory_id`
- 无错误信息

### 特征 2: 数据丢失

- 查询返回 404
- 搜索找不到
- 项目地图为空

### 特征 3: 可重复

- 多次测试均出现
- 不同内容、不同项目都出现
- 不是偶发问题

---

## 3. 可能原因

### 原因 A: 异步写入失败（最可能）

**流程**:
```
1. API 接收请求
2. 立即返回成功（异步处理）
3. 后台写入队列处理
4. 写入失败（无重试/无通知）
```

**证据**:
- 响应极快（<100ms）
- 无等待写入确认
- 数据最终不存在

### 原因 B: 事务回滚

**流程**:
```
1. 开始数据库事务
2. 生成 memory_id
3. 返回成功响应
4. 事务提交失败（回滚）
```

**证据**:
- memory_id 已生成
- 但数据未持久化

### 原因 C: 多节点数据不一致

**场景**:
- 上传写入节点 A
- 查询读取节点 B
- 节点间数据未同步

---

## 4. 紧急排查建议

### 4.1 检查写入队列

```bash
# 检查队列积压
redis-cli LLEN memory_upload_queue
redis-cli LRANGE memory_upload_queue 0 10

# 检查处理状态
redis-cli KEYS "upload:*"
```

### 4.2 检查数据库事务日志

```sql
-- 检查最近的事务
SELECT * FROM transaction_log 
WHERE timestamp > NOW() - INTERVAL '1 hour'
ORDER BY timestamp DESC;

-- 检查失败的写入
SELECT * FROM failed_writes
WHERE created_at > NOW() - INTERVAL '1 hour';
```

### 4.3 检查服务日志

```bash
# 查看错误日志
docker logs wrapper-service 2>&1 | grep -i "error\|fail\|exception" | tail -50

# 查看上传处理日志
docker logs wrapper-service 2>&1 | grep "uploadMemories\|writeMemory" | tail -20
```

### 4.4 检查数据一致性

```sql
-- 检查内存中的 memory_id 是否存在于数据库
SELECT id FROM memories 
WHERE id IN ('memory:wiysqg9h66lo2o2asx4k', 'memory:ihvclhn43qeqkg3f3twt');

-- 检查 hash 索引
SELECT * FROM memory_hash_index 
WHERE hash IN (SELECT content_hash FROM memories WHERE id = 'memory:wiysqg9h66lo2o2asx4k');
```

---

## 5. 临时解决方案

### 方案 A: 改为同步写入（推荐）

修改上传 API，等待写入完成后再返回：

```python
@app.post("/api/v1/memories")
async def upload_memories(memories):
    results = []
    for memory in memories:
        # 同步写入
        memory_id = await write_to_database(memory)
        
        # 验证写入
        verify = await query_database(memory_id)
        if not verify:
            raise Exception("Write failed")
        
        results.append(memory_id)
    
    return {"success": len(results), "memory_ids": results}
```

### 方案 B: 添加写入确认轮询

插件端上传后主动轮询确认：

```javascript
async function uploadWithConfirm(memory) {
  const result = await upload(memory);
  
  // 轮询确认
  for (let i = 0; i < 10; i++) {
    const exists = await query(result.memory_id);
    if (exists) return result;
    await sleep(500);
  }
  
  throw new Error("Upload confirmation timeout");
}
```

---

## 6. 联调影响

| 功能 | 状态 | 影响 |
|------|------|------|
| 代码分析 | ⚠️ 部分可用 | 分析成功，但无法保存 |
| 调用关系 | ❌ 阻塞 | 依赖 memory 存在 |
| 项目地图 | ❌ 阻塞 | 依赖 memory 存在 |
| 引用查询 | ❌ 阻塞 | 依赖 memory 存在 |

**当前状态**: 所有核心功能阻塞

---

## 7. 需要后端立即执行

1. **确认问题**: 是否能在后端复现上传成功但查询不到？
2. **检查队列**: 是否有积压的写入任务？
3. **检查日志**: 是否有写入失败的错误？
4. **修复时间**: 今天能否修复？
5. **临时方案**: 是否可以先改为同步写入？

---

## 8. 测试命令

后端可自行测试：

```bash
# 1. 上传测试
curl -X POST http://localhost:17999/api/v1/memories \
  -H "Content-Type: application/json" \
  -d '{"memories":[{"type":"code","content":"test","abstract":"test","project_id":"test","metadata":{"file_path":"test.ts"}}],"tenant_id":"default"}'

# 2. 立即查询（替换上面的 memory_id）
curl http://localhost:17999/api/v1/memories/memory:xxx

# 3. 检查数据库
# 进入数据库容器，查询 memories 表
```

---

**这是阻塞联调的关键问题，请后端紧急排查！**

---

*文档版本: v1.0*  
*日期: 2026-04-08*  
*优先级: P0*  
*状态: 紧急 - 阻塞联调*
