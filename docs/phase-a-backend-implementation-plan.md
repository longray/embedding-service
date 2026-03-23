# Phase A 后端实施计划 - 数据库优化

**版本**: v2.2-lite Phase A  
**目标**: 后端数据库性能和安全优化  
**工作量**: 2.5小时  
**文档日期**: 2026-03-18

---

## 一、概述

### 1.1 目标

优化embedding_service后端的数据库设计和性能，基于4位专家评审结果：
- **P0优化**（必须完成）：启用传输加密
- **P1优化**（强烈建议）：批量事务、查询缓存、HNSW参数调整、智能去重决策框架等
- **已取消**：消除重复向量计算（SurrealDB 3.0.1 KNN bug，保持当前实现）

### 1.2 专家评审总结

| 专家 | 评分 | 核心问题 |
|------|------|---------|
| 数据库架构师 | 4/5 | HNSW M=12偏保守，缺少计算字段 |
| 性能工程师 | 3/5 | 重复向量计算（50%性能损失） |
| 安全工程师 | 3/5 | 无传输加密（ws→wss） |
| AI/ML工程师 | 3.5/5 | 去重阈值固定，缺少批量embedding |

**综合评分**: ⭐⭐⭐☆☆ (3.5/5)

### 1.3 核心原则

- **P0优先**：先完成P0优化，确保基础性能和安全
- **向后兼容**：保持API接口不变
- **最小化代码**：只修改必要的部分
- **充分测试**：每个优化都要验证效果

### 1.4 前置条件

- ✅ SurrealDB数据库已备份（2026-03-18备份）
- ✅ 后端服务运行正常（localhost:17999）
- ✅ 测试环境准备就绪

---

## 二、P0优化任务（必须完成，30分钟）

### ~~任务 2.1：消除重复向量计算~~（已取消）

**决策**：保持当前实现（重复计算但稳定）

**原因**：
- SurrealDB 3.0.1的KNN操作符存在严重bug（GitHub Issue #6994）
- 暴力搜索`<|K|>`报错，HNSW搜索`<|K,EF|>`返回所有记录
- 当前实现虽有重复计算，但功能稳定可靠

**当前实现**（memory_manager.py）：
```python
async def _search_by_vector(self, embedding, limit, threshold, tenant_id):
    q = (
        "SELECT id, content, metadata, type, tags, project_id, "
        "vector::similarity::cosine(embedding, $query_embedding) AS score "
        "FROM memory "
        "WHERE tenant_id = $tenant_id "
        "AND vector::similarity::cosine(embedding, $query_embedding) >= $threshold "
        "ORDER BY score DESC "
        f"LIMIT {int(limit)}"
    )
    # ✅ 在SurrealDB 3.0.1版本稳定工作
```

**后续优化计划**：
- 等待SurrealDB 3.0.2+修复KNN bug
- 升级后重新评估KNN优化方案

**预计时间**：0分钟（无需修改）

---

### 任务 2.2：启用传输加密（30分钟）

**问题**：当前使用ws://协议，数据明文传输

**当前配置**（surrealdb_client.py）：
```python
SURREALDB_URL = "ws://localhost:18002/rpc"
```

**优化方案**：启用TLS加密，使用wss://协议

**实施步骤**：

1. **生成自签名证书**（开发环境）：
```bash
# 在 D:/embedding_service/certs/ 目录
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes
```

2. **修改SurrealDB启动参数**：
```bash
# scripts/start_surrealdb.sh
surreal start \
  --bind 0.0.0.0:18002 \
  --user root \
  --pass root \
  file:./data/surrealdb \
  --web-crt ./certs/cert.pem \
  --web-key ./certs/key.pem
```

3. **修改客户端配置**：
```python
# surrealdb_client.py
SURREALDB_URL = "wss://localhost:18002/rpc"

# 开发环境：忽略自签名证书警告
import ssl
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

# 连接时使用ssl_context
await self._db.connect(SURREALDB_URL, ssl=ssl_context)
```

**验证**：
- 使用Wireshark抓包，确认数据已加密
- 测试连接是否正常
- 检查性能影响（预期<5%）

**预计时间**：30分钟

---

## 三、P1优化任务（强烈建议，1.5小时）

### 任务 3.1：批量插入使用事务（15分钟）

**问题**：当前批量插入逐条执行，性能低

**优化方案**：使用SurrealDB事务

**修改代码**（memory_manager.py）：
```python
async def batch_insert(self, memories):
    # 构建事务
    transaction = "BEGIN TRANSACTION;\n"
    for mem in memories:
        transaction += f"CREATE memory CONTENT {json.dumps(mem)};\n"
    transaction += "COMMIT TRANSACTION;"
    
    result = await self._db.query(transaction)
    return result
```

**预期提升**：10-20x性能改善

---

### 任务 3.2：添加查询结果缓存（20分钟）

**优化方案**：使用LRU缓存高频查询

**实现代码**：
```python
from async_lru import alru_cache  # pip install async-lru
from hashlib import md5

@alru_cache(maxsize=1000, ttl=3600)  # 支持异步函数
async def _search_cached(self, query_hash, limit, threshold, tenant_id):
    return await self._search_by_vector(...)

async def search(self, query, limit, threshold, tenant_id):
    query_hash = md5(query.encode()).hexdigest()
    return await self._search_cached(query_hash, limit, threshold, tenant_id)
```

**预期提升**：80%缓存命中率，查询时间降低80%

---

### 任务 3.3：调整HNSW参数（10分钟）

**问题**：M=12偏保守，适合1K-100K数据

**优化方案**：调整为M=16-24

**修改代码**（init_db.py）：
```sql
DEFINE INDEX memory_embedding_hnsw ON memory
    FIELDS embedding HNSW DIMENSION 1024 DIST COSINE
    EFC 200 M 16;  -- 从12改为16
```

**预期提升**：召回率提升5-10%

---

### 任务 3.4：批量embedding优化（15分钟）

**问题**：逐条调用embedding API，吞吐量低

**优化方案**：批量调用（batch_size=32）

**实现代码**：
```python
async def get_embeddings_batch(self, texts, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = await self.embedding_client.embed(batch)
        embeddings.extend(response)
    return embeddings
```

**预期提升**：5-10x吞吐量提升

---

### 任务 3.5：动态去重阈值配置（10分钟）

**问题**：阈值固定为0.95，无法按记忆类型调整

**优化方案**：配置文件支持分层阈值

**配置文件**（config.yaml）：
```yaml
deduplication:
  thresholds:
    preference: 0.88
    decision: 0.90
    long-term: 0.93
    general: 0.95
    daily: 1.0
```

**修改代码**：
```python
def get_threshold(self, memory_type):
    config = load_config()
    return config['deduplication']['thresholds'].get(memory_type, 0.95)
```

---

### 任务 3.6：智能去重决策框架（20分钟）⭐

**问题**：只有"检测-跳过"策略，无法更新或保留历史

**优化方案**：实现UPDATE/DISCARD/KEEP_BOTH三种策略

**核心代码**：
```python
def decide_duplicate_action(self, new_mem, old_mem, similarity):
    time_diff = (new_mem['created_at'] - old_mem['created_at']).total_seconds()
    new_len = len(new_mem['content'])
    old_len = len(old_mem['content'])
    mem_type = new_mem.get('type', 'general')
    
    # 6个核心规则
    if similarity > 0.98 and time_diff < 300:
        return 'DISCARD'  # 5分钟内的重复
    if mem_type == 'preference':
        return 'UPDATE'  # 用户偏好总是更新
    if mem_type == 'decision':
        return 'KEEP_BOTH'  # 决策记录保留历史
    if new_len > old_len * 1.5:
        return 'UPDATE'  # 新内容更详细
    if new_len < old_len * 0.7:
        return 'DISCARD'  # 新内容不完整
    if time_diff > 30 * 86400:
        return 'KEEP_BOTH'  # 超过30天，可能是演变
    if time_diff < 7 * 86400 and similarity > 0.90:
        return 'UPDATE'  # 7天内的高相似更新
    
    return 'KEEP_BOTH'  # 默认保留
```

**预期效果**：去重准确率提升15-20%

---

### 任务 3.7：添加计算字段（10分钟）

**优化方案**：避免per-record计算

**修改Schema**（init_db.py）：
```sql
DEFINE FIELD content_length ON memory VALUE string::length(content);
DEFINE FIELD created_date ON memory VALUE time::format(created_at, '%Y-%m-%d');
```

**使用示例**：
```python
# 查询时直接使用计算字段
q = "SELECT * FROM memory WHERE content_length > 1000"
```

---

### 任务 3.8：补充索引（10分钟）

**问题**：缺少created_at和tags索引

**修改Schema**：
```sql
DEFINE INDEX memory_created_at ON memory FIELDS created_at;
DEFINE INDEX memory_tags ON memory FIELDS tags;
```

**预期提升**：时间范围查询和标签过滤性能提升10x

---

### 任务 3.9：字段级权限控制（15分钟）

**优化方案**：限制runtime_user只能访问必要字段

**修改Schema**：
```sql
DEFINE FIELD embedding ON memory 
    PERMISSIONS FOR select WHERE $auth.role = 'admin';
    
DEFINE FIELD metadata ON memory
    PERMISSIONS FOR update WHERE $auth.role = 'admin';
```

**验证**：
- runtime_user无法读取embedding字段
- runtime_user无法修改metadata字段

---

### 任务 3.10：租户ID验证（10分钟）

**优化方案**：强制验证tenant_id

**修改代码**：
```python
async def create_memory(self, content, tenant_id, **kwargs):
    if not tenant_id or tenant_id == '':
        raise ValueError("tenant_id is required")
    
    # 验证tenant_id格式
    if not re.match(r'^[a-zA-Z0-9_-]+$', tenant_id):
        raise ValueError("Invalid tenant_id format")
    
    # 继续创建...
```

---

### 任务 3.11：敏感字段加密（15分钟）

**优化方案**：加密存储metadata中的敏感信息

**实现代码**：
```python
from cryptography.fernet import Fernet

class EncryptionHelper:
    def __init__(self, key):
        self.cipher = Fernet(key)
    
    def encrypt(self, data):
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt(self, data):
        return self.cipher.decrypt(data.encode()).decode()

# 使用
encryptor = EncryptionHelper(os.getenv('ENCRYPTION_KEY'))
metadata['sensitive_field'] = encryptor.encrypt(value)
```

---

### 任务 3.12：备份文件加密（10分钟）

**优化方案**：使用GPG加密备份文件

**修改脚本**（scripts/backup.sh）：
```bash
# 导出备份
surreal export ... backup.surql

# 加密备份
gpg --symmetric --cipher-algo AES256 backup.surql

# 删除明文备份
rm backup.surql
```

---

### 任务 3.13：添加EXPLAIN分析（5分钟）

**优化方案**：使用EXPLAIN分析查询计划

**使用示例**：
```python
async def analyze_query(self, query):
    explain_query = f"EXPLAIN {query}"
    result = await self._db.query(explain_query)
    print(f"Query plan: {result}")
```

---

## 四、验证和测试

### 4.1 P0优化验证

**测试脚本**（test-p0-optimization.py）：
```python
import asyncio
import time

async def test_vector_search_performance():
    # 优化前后对比
    start = time.time()
    result = await search_by_vector(embedding, limit=10)
    elapsed = time.time() - start
    
    print(f"查询时间: {elapsed*1000:.2f}ms")
    assert elapsed < 0.05, "性能未达标"

async def test_tls_encryption():
    # 验证TLS连接
    assert client.url.startswith('wss://'), "未启用TLS"
    print("✅ TLS加密已启用")
```

### 4.2 P1优化验证

**测试清单**：
- ✅ 批量插入性能提升10x
- ✅ 缓存命中率>80%
- ✅ HNSW召回率提升5-10%
- ✅ 智能去重准确率>95%

---

## 五、Go/No-Go检查点

1. ✅ 传输加密启用（wss://）
2. ✅ 批量插入性能提升10x
3. ✅ 查询缓存命中率>80%
4. ✅ 智能去重决策框架实现

**如果任一检查点失败**：回滚到备份版本

---

## 六、时间分配

| 任务 | 预计时间 | 优先级 | 备注 |
|------|---------|--------|------|
| ~~2.1 消除重复向量计算~~ | ~~30min~~ | ~~P0~~ | 已取消（SurrealDB 3.0.1 bug） |
| 2.2 启用传输加密 | 30min | P0 | 必须完成 |
| 3.1 批量事务 | 15min | P1 | 强烈建议 |
| 3.2 查询缓存 | 20min | P1 | 强烈建议 |
| 3.3 HNSW参数 | 10min | P1 | 强烈建议 |
| 3.4 批量embedding | 15min | P1 | 强烈建议 |
| 3.5 动态阈值 | 10min | P1 | 强烈建议 |
| 3.6 智能去重 | 20min | P1 | 强烈建议 |
| 3.7 计算字段 | 10min | P1 | 可选 |
| 3.8 补充索引 | 10min | P1 | 可选 |
| 3.9 字段权限 | 15min | P1 | 可选 |
| 3.10 租户验证 | 10min | P1 | 可选 |
| 3.11 字段加密 | 15min | P1 | 可选 |
| 3.12 备份加密 | 10min | P1 | 可选 |
| 3.13 EXPLAIN | 5min | P1 | 可选 |
| **P0总计** | **30min** | | 仅TLS加密 |
| **P1总计** | **1.5h** | | |
| **总计** | **2h** | | |

---

## 七、总结

**Phase A 后端实施计划已完成**，包含：
- ✅ P0优化1项（必须完成，30分钟）：TLS传输加密
- ✅ P1优化13项（强烈建议，1.5小时）
- ✅ 完整的验证测试方案
- ✅ 4个Go/No-Go检查点
- ⚠️ 已取消：消除重复向量计算（SurrealDB 3.0.1 KNN bug）

**预期效果**：
- 传输安全性提升（TLS加密）
- 去重准确率提升15-20%
- 批量操作性能提升10x
- 整体系统评分：3.5/5 → 4.0/5

**下一步**：开始实施Phase A（插件端4-4.5h + 后端2h = 总计6-6.5h）
