# 记忆去重机制实施指南

## 📋 方案概述

**方案**: 内容哈希 + 复合 UNIQUE 索引（租户级去重）

**核心机制**:

- 添加 `content_hash` 字段（MD5 哈希）
- 创建 `(tenant_id, content_hash)` 复合 UNIQUE 索引
- 自动哈希触发器（CREATE/UPDATE 时自动计算）

**效果**:

- 🚫 完全阻止重复内容插入
- ⚡ 数据库层面拒绝，性能最优
- 🔒 无法绕过，安全可靠
- 🎯 租户隔离，灵活性好

---

## 🚀 实施步骤

### 步骤1: 备份数据库（必需）

```bash
# 导出当前数据
surreal export --conn ws://localhost:18002 \
  --user root --pass root \
  --ns memory_ns --db memory_db \
  backup_$(date +%Y%m%d_%H%M%S).surql
```

### 步骤2: 清理现有重复数据

```bash
cd D:\embedding_service
uv run python scripts/migrate_add_deduplication.py
```

**输出示例**:

```
🔍 步骤1: 分析重复数据...
  - 总记录数: 288
  - 唯一内容: 208
  - 重复组数: 80
  - 重复记录: 80

🗑️  步骤2: 清理 80 组重复数据...
  - 已删除: 80 条重复记录

🔨 步骤3: 生成 content_hash...
  - 已为所有记录生成哈希

✅ 迁移完成！
```

### 步骤3: 应用 Schema 变更

```bash
# 方式1: 通过 Python SDK
uv run python -c "
import asyncio
from surrealdb import AsyncSurreal

async def apply_schema():
    db = AsyncSurreal('ws://localhost:18002/rpc')
    await db.connect()
    await db.signin({'username': 'root', 'password': 'root'})
    await db.use('memory_ns', 'memory_db')
    
    with open('scripts/add_deduplication.surql', 'r', encoding='utf-8') as f:
        sql = f.read()
    
    await db.query(sql)
    print('✅ Schema 已更新')
    await db.close()

asyncio.run(apply_schema())
"

# 方式2: 通过 HTTP API
curl -X POST http://localhost:18002/sql \
  -H "Content-Type: application/json" \
  -H "NS: memory_ns" \
  -H "DB: memory_db" \
  -u "root:root" \
  --data-binary @scripts/add_deduplication.surql
```

### 步骤4: 验证去重机制

```bash
uv run python -c "
import asyncio
from surrealdb import AsyncSurreal

async def test_dedup():
    db = AsyncSurreal('ws://localhost:18002/rpc')
    await db.connect()
    await db.signin({'username': 'root', 'password': 'root'})
    await db.use('memory_ns', 'memory_db')
    
    # 第一次插入（应该成功）
    try:
        await db.query('''
            CREATE memory CONTENT {
                content: 'Test duplicate content',
                tenant_id: 'test',
                type: 'test'
            };
        ''')
        print('✅ 第一次插入成功')
    except Exception as e:
        print(f'❌ 第一次插入失败: {e}')
    
    # 第二次插入相同内容（应该失败）
    try:
        await db.query('''
            CREATE memory CONTENT {
                content: 'Test duplicate content',
                tenant_id: 'test',
                type: 'test'
            };
        ''')
        print('❌ 第二次插入成功（去重失败！）')
    except Exception as e:
        if 'already contains' in str(e):
            print('✅ 第二次插入被拒绝（去重成功！）')
        else:
            print(f'⚠️  第二次插入失败，但原因不明: {e}')
    
    await db.close()

asyncio.run(test_dedup())
"
```

---

## 📊 预期结果

### 插入重复内容时的错误

```
Database index `memory_tenant_content_unique` already contains 
[tenant_id: 'test', content_hash: 'a1b2c3d4...']
```

### 数据库状态

```sql
-- 检查字段
INFO FOR TABLE memory;
-- 应该看到 content_hash 字段

-- 检查索引
SHOW INDEX ON memory;
-- 应该看到 memory_tenant_content_unique 索引

-- 检查触发器
INFO FOR TABLE memory;
-- 应该看到 auto_content_hash 事件
```

---

## 🔄 回滚方案

如果需要撤销去重机制：

```sql
-- 1. 删除触发器
REMOVE EVENT IF EXISTS auto_content_hash ON memory;

-- 2. 删除索引
REMOVE INDEX IF EXISTS memory_tenant_content_unique ON memory;

-- 3. 删除字段（可选）
REMOVE FIELD IF EXISTS content_hash ON memory;
```

---

## ⚠️ 注意事项

1. **备份**: 执行前必须备份数据库
2. **停机时间**: 建议在维护窗口执行（约5-10分钟）
3. **数据量**: 如果记录 > 10万，分批执行迁移脚本
4. **插件更新**: 无需修改插件代码，触发器自动处理
5. **性能影响**: MD5 计算开销极小（< 1ms/记录）

---

## 📈 性能影响

| 操作 | 影响 | 说明 |
|------|------|------|
| 插入 | +1-2ms | MD5 计算 + 索引检查 |
| 查询 | 无影响 | 不涉及 content_hash |
| 更新 | +1-2ms | 重新计算哈希 |
| 存储 | +32 bytes/记录 | MD5 哈希长度 |

---

## ✅ 完成检查清单

- [ ] 备份数据库
- [ ] 运行迁移脚本清理重复数据
- [ ] 应用 Schema 变更
- [ ] 验证去重机制
- [ ] 测试插件上传功能
- [ ] 监控错误日志

---

**版本**: 2.3.1  
**创建日期**: 2026-03-16  
**兼容性**: SurrealDB >= 3.0
