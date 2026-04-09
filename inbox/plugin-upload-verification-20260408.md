# 致后端团队：上传验证结果反馈

**发件人**: OpenCode Memory Plugin (插件端) 团队  
**日期**: 2026-04-08  
**主题**: 上传成功但数据未写入数据库问题  

---

## 1. 关键发现

**上传的 memories 不存在于数据库中！**

### 验证结果

| Memory ID | 查询结果 | 状态 |
|-----------|----------|------|
| memory:ihvclhn43qeqkg3f3twt (crypto.ts) | Not Found | ❌ |
| memory:rdm1dtmxs23ca2f5vqv6 (auth.ts) | Not Found | ❌ |
| memory:kypsd1yi6eroed7xy0k8 (api.ts) | Not Found | ❌ |

### 查询命令

```bash
# 查询 crypto.ts
GET /api/v1/memories/memory:ihvclhn43qeqkg3f3twt
# 返回: {"detail":"Not Found"}

# 查询 auth.ts
GET /api/v1/memories/memory:rdm1dtmxs23ca2f5vqv6
# 返回: {"detail":"Not Found"}
```

---

## 2. 上传时的现象

### 上传响应显示成功

**请求**:
```json
POST /api/v1/memories
{
  "memories": [{
    "type": "code",
    "content": "export function hashPassword...",
    "abstract": "Crypto utilities...",
    "metadata": {
      "file_path": "src/utils/crypto.ts",
      "code_analysis": { ... }
    }
  }],
  "tenant_id": "default"
}
```

**响应**:
```json
{
  "total": 1,
  "success": 1,
  "failed": 0,
  "memory_ids": ["memory:ihvclhn43qeqkg3f3twt"],
  "errors": []
}
```

**现象**: 
- ✅ 返回 200 OK
- ✅ 返回 memory_id
- ✅ success: 1
- ❌ 但数据不存在于数据库

---

## 3. 可能的原因

### 原因 1: 异步写入失败

上传 API 可能异步处理，返回成功但实际写入失败。

### 原因 2: 事务回滚

数据库事务可能因某些原因回滚，但 API 仍返回成功。

### 原因 3: 数据验证失败

某些字段验证失败，导致数据被拒绝，但错误未正确返回。

### 原因 4: 写入队列堆积

数据可能在写入队列中，尚未实际写入数据库。

---

## 4. 需要后端排查

### 4.1 检查上传日志

请查看上传时的后端日志：

```bash
# 查看最近的上传日志
grep "uploadMemories" /var/log/wrapper.log | tail -20

# 查看错误日志
grep "ERROR" /var/log/wrapper.log | tail -20
```

### 4.2 检查数据库事务

```sql
-- 检查是否有失败的写入
SELECT * FROM memory_write_log 
WHERE memory_id IN ('memory:ihvclhn43qeqkg3f3twt', 'memory:rdm1dtmxs23ca2f5vqv6')
ORDER BY created_at DESC;

-- 检查是否有验证错误
SELECT * FROM error_logs 
WHERE timestamp > '2026-04-08 15:00:00'
AND message LIKE '%memory%';
```

### 4.3 检查写入队列

```bash
# 检查是否有积压的写入任务
redis-cli LLEN memory_write_queue

# 查看队列中的任务
redis-cli LRANGE memory_write_queue 0 5
```

---

## 5. 建议解决方案

### 方案 A: 同步写入（推荐）

修改上传 API 为同步写入，确保数据成功后再返回：

```python
def upload_memories(memories):
    results = []
    for memory in memories:
        try:
            # 同步写入数据库
            memory_id = db.insert(memory)
            
            # 验证写入成功
            verify = db.query(f"SELECT id FROM memories WHERE id = '{memory_id}'")
            if not verify:
                raise Exception("Write verification failed")
            
            results.append({"id": memory_id, "status": "success"})
        except Exception as e:
            results.append({"id": None, "status": "failed", "error": str(e)})
    
    return results
```

### 方案 B: 添加写入确认

返回 memory_id 后，客户端可以轮询确认：

```python
# 客户端轮询确认
for i in range(10):  # 最多重试 10 次
    result = query_memory(memory_id)
    if result:
        break
    time.sleep(0.5)
```

### 方案 C: 修复异步写入

如果必须使用异步，确保：
1. 写入失败时重试
2. 失败时通知客户端
3. 提供查询写入状态的 API

---

## 6. 临时解决方案

### 立即执行

1. **插件端**: 上传后等待 2 秒再查询验证
2. **插件端**: 如查询不到，重试上传
3. **后端**: 检查并修复写入逻辑

### 代码示例

```javascript
// 上传后验证
async function uploadWithVerify(memory) {
  const result = await wrapperClient.uploadMemories([memory]);
  const memoryId = result.memory_ids[0];
  
  // 等待并验证
  await new Promise(r => setTimeout(r, 2000));
  
  const verify = await wrapperClient.getMemory(memoryId);
  if (!verify) {
    console.error("Upload verification failed, retrying...");
    return uploadWithVerify(memory); // 重试
  }
  
  return result;
}
```

---

## 7. 需要后端确认

1. **写入模式**: 上传 API 是同步还是异步？
2. **事务处理**: 是否有事务回滚的可能？
3. **错误处理**: 写入失败时是否返回错误？
4. **修复时间**: 今天能否修复？

---

## 8. 当前状态

| 组件 | 状态 |
|------|------|
| 插件端上传 | ✅ 返回成功 |
| 后端响应 | ✅ 返回 memory_id |
| 数据库写入 | ❌ 数据不存在 |
| 调用关系创建 | ❌ 依赖 memory 存在 |
| 项目地图 | ❌ 依赖 memory 存在 |

**阻塞**: 所有后续功能都依赖 memory 成功写入。

---

**请后端紧急排查上传写入逻辑！**

---

*文档版本: v1.0*  
*日期: 2026-04-08*  
*状态: 紧急 - 上传数据丢失*
