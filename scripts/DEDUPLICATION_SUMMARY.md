# SurrealDB 去重机制实施总结

## 问题
SurrealDB 中存在重复的记忆数据（27.8%），需要实现去重机制防止未来重复上传。

## 解决方案
**方案**: 内容哈希 + UNIQUE 索引（应用层计算）

### 实施步骤

1. **添加 content_hash 字段**
   - 使用 MD5 哈希算法
   - 在应用层计算（插入前）

2. **创建 UNIQUE 索引**
   ```sql
   DEFINE INDEX memory_tenant_content_unique ON memory 
   FIELDS tenant_id, content_hash UNIQUE;
   ```

3. **修改应用代码**
   - 文件: `wrapper/src/utils/memory_manager.py`
   - 添加: `import hashlib`
   - 在 `upload_memories` 方法中插入前计算 content_hash

4. **删除 EVENT 触发器**
   - 原触发器 `auto_content_hash` 导致无限递归
   - 执行: `REMOVE EVENT auto_content_hash ON TABLE memory;`

## 验证结果

✅ **测试通过**
- 第一次上传唯一内容: 成功
- 第二次上传相同内容: 失败（UNIQUE 约束）

## 关键文件

- `scripts/add_deduplication.surql` - Schema 变更
- `scripts/migrate_add_deduplication.py` - 数据迁移
- `scripts/remove_event_trigger.surql` - 删除触发器
- `wrapper/src/utils/memory_manager.py` - 应用层实现

## 技术要点

1. **为什么在应用层计算 content_hash？**
   - SurrealDB EVENT 触发器是异步的
   - CREATE 时 content_hash 为 NULL，UNIQUE 检查失效
   - 触发器更新会导致无限递归

2. **去重粒度**
   - 按 `(tenant_id, content_hash)` 组合去重
   - 不同租户可以有相同内容
   - 相同租户不能有重复内容

## 实施日期
2026-03-16
