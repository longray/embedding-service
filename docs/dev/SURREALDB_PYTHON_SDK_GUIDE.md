# SurrealDB Python SDK 使用指南

> 基于实际项目经验总结的 SurrealDB Python SDK 使用最佳实践

## 快速开始

### 安装

```bash
uv pip install surrealdb
```

### 基础连接示例

```python
import asyncio
from surrealdb import AsyncSurreal


async def main():
    """SurrealDB 基础连接示例"""
    # 1. 创建连接（注意：不会自动连接）
    db = AsyncSurreal("ws://localhost:18002/rpc")
    
    try:
        # 2. 显式连接（必须！）
        await db.connect()
        
        # 3. 认证（使用 username/password，不是 user/pass）
        await db.signin({"username": "root", "password": "root"})
        
        # 4. 选择命名空间和数据库
        await db.use("memory_ns", "memory_db")
        
        # 5. 执行查询
        result = await db.query("SELECT * FROM atom LIMIT 5")
        print(result)
        
    finally:
        # 6. 关闭连接
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
```

## 关键注意事项

### 1. 必须使用 AsyncSurreal

```python
# ✅ 正确 - 使用 AsyncSurreal（异步版本）
from surrealdb import AsyncSurreal
db = AsyncSurreal("ws://localhost:18002/rpc")

# ❌ 错误 - 旧版 Surreal 类可能不可用或行为不一致
from surrealdb import Surreal  # 某些版本可能不存在
```

### 2. 必须显式调用 connect()

```python
db = AsyncSurreal("ws://localhost:18002/rpc")

# ❌ 错误 - 直接执行查询会失败
result = await db.query("SELECT * FROM atom")  # Connection not established

# ✅ 正确 - 先 connect
await db.connect()
result = await db.query("SELECT * FROM atom")
```

### 3. 认证参数使用 username/password

```python
# ❌ 错误 - 使用 user/pass
await db.signin({"user": "root", "pass": "root"})

# ✅ 正确 - 使用 username/password
await db.signin({"username": "root", "password": "root"})
```

### 4. 使用环境变量管理配置

```python
import os

SURREALDB_URL = os.getenv("SURREALDB_URL", "ws://localhost:18002/rpc")
SURREALDB_NS = os.getenv("SURREALDB_NS", "memory_ns")
SURREALDB_DB = os.getenv("SURREALDB_DB", "memory_db")
SURREALDB_USER = os.getenv("SURREALDB_USER", "root")
SURREALDB_PASS = os.getenv("SURREALDB_PASS", "root")
```

## 常用操作示例

### 查询数据

```python
# 基础查询
result = await db.query("SELECT * FROM atom LIMIT 10")

# 条件查询
result = await db.query(
    "SELECT local_id, name FROM atom WHERE name = '错误处理模式' LIMIT 1"
)

# 参数化查询（防止 SQL 注入）
name = "错误处理模式"
result = await db.query(
    "SELECT * FROM atom WHERE name = $name LIMIT 1",
    {"name": name}
)
```

### 插入数据

```python
# 插入单条记录
await db.query(
    "CREATE atom CONTENT { local_id: 'TEST001', name: '测试数据', type: 'note' }"
)

# 插入多条记录
atoms = [
    {"local_id": "TEST001", "name": "测试1"},
    {"local_id": "TEST002", "name": "测试2"},
]
for atom in atoms:
    await db.query(
        "CREATE atom CONTENT $content",
        {"content": atom}
    )
```

### 更新数据

```python
# 更新记录
await db.query(
    "UPDATE atom SET name = '新名称' WHERE local_id = 'TEST001'"
)

# 条件更新
await db.query(
    "UPDATE atom SET status = 'archived' WHERE type = 'temp'"
)
```

### 删除数据

```python
# 删除单条
await db.query("DELETE atom WHERE local_id = 'TEST001'")

# 删除所有（危险！）
await db.query("DELETE atom")
```

## 完整脚本模板

```python
#!/usr/bin/env python3
"""SurrealDB 查询脚本模板

前置条件:
    - SurrealDB 运行中

用法:
    uv run python scripts/query_surrealdb.py

示例:
    uv run python scripts/query_surrealdb.py
"""

import asyncio
import os
import logging
from surrealdb import AsyncSurreal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 配置
SURREALDB_URL = os.getenv("SURREALDB_URL", "ws://localhost:18002/rpc")
SURREALDB_NS = os.getenv("SURREALDB_NS", "memory_ns")
SURREALDB_DB = os.getenv("SURREALDB_DB", "memory_db")
SURREALDB_USER = os.getenv("SURREALDB_USER", "root")
SURREALDB_PASS = os.getenv("SURREALDB_PASS", "root")


async def query_database():
    """查询 SurrealDB 数据库"""
    db = AsyncSurreal(SURREALDB_URL)
    
    try:
        # 连接
        await db.connect()
        logger.info(f"已连接到 {SURREALDB_URL}")
        
        # 认证
        await db.signin({"username": SURREALDB_USER, "password": SURREALDB_PASS})
        logger.info("认证成功")
        
        # 选择命名空间和数据库
        await db.use(SURREALDB_NS, SURREALDB_DB)
        logger.info(f"使用命名空间: {SURREALDB_NS}, 数据库: {SURREALDB_DB}")
        
        # 执行查询
        result = await db.query("SELECT * FROM atom LIMIT 5")
        logger.info(f"查询结果: {result}")
        
        return result
        
    except Exception as e:
        logger.error(f"查询失败: {e}")
        raise
    finally:
        await db.close()
        logger.info("连接已关闭")


if __name__ == "__main__":
    asyncio.run(query_database())
```

## 故障排查

### 问题 1: Connection refused

**症状**:

```
Connection refused: [Errno 111] Connection refused
```

**解决方案**:

```bash
# 1. 检查 SurrealDB 是否运行
docker ps | grep surreal

# 2. 启动 SurrealDB
docker-compose up -d surrealdb

# 3. 检查端口
netstat -an | findstr 18002  # Windows
# 或
ss -tlnp | grep 18002        # Linux
```

### 问题 2: Authentication failed

**症状**:

```
Authentication failed: Invalid credentials
```

**解决方案**:

```python
# ❌ 错误
await db.signin({"user": "root", "pass": "root"})

# ✅ 正确
await db.signin({"username": "root", "password": "root"})
```

### 问题 3: Not connected

**症状**:

```
Not connected to SurrealDB
```

**解决方案**:

```python
# ❌ 错误 - 忘记 connect
db = AsyncSurreal("ws://localhost:18002/rpc")
result = await db.query("SELECT * FROM atom")

# ✅ 正确 - 先 connect
db = AsyncSurreal("ws://localhost:18002/rpc")
await db.connect()  # 必须显式连接
result = await db.query("SELECT * FROM atom")
```

### 问题 4: 查询结果解析

**症状**: 查询返回的数据格式不符合预期

**解决方案**:

```python
result = await db.query("SELECT * FROM atom LIMIT 5")

# SurrealDB Python SDK 返回的结果是一个列表
# 每个元素对应一个查询的结果
if result and len(result) > 0:
    # 第一个查询的结果
    records = result[0]
    for record in records:
        print(f"ID: {record.get('id')}")
        print(f"Local ID: {record.get('local_id')}")
        print(f"Name: {record.get('name')}")
```

## 项目中的实际使用

### 检查 Atom local_id 格式

```python
import asyncio
from surrealdb import AsyncSurreal


async def check_local_id():
    """检查 atom 表中的 local_id 格式"""
    db = AsyncSurreal("ws://localhost:18002/rpc")
    
    try:
        await db.connect()
        await db.signin({"username": "root", "password": "root"})
        await db.use("memory_ns", "memory_db")
        
        # 查询 local_id 字段
        result = await db.query("SELECT local_id FROM atom LIMIT 10")
        
        if result and len(result) > 0:
            records = result[0]
            for record in records:
                local_id = record.get("local_id")
                print(f"local_id: {local_id} (类型: {type(local_id).__name__})")
                
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(check_local_id())
```

## 参考资源

- [SurrealDB 官方文档](https://surrealdb.com/docs)
- [SurrealDB Python SDK GitHub](https://github.com/surrealdb/surrealdb.py)
- 项目模板: `scripts/surrealdb_query_template.py`

## 总结

使用 SurrealDB Python SDK 的关键点:

1. **使用 AsyncSurreal** - 异步版本
2. **显式 connect()** - 创建实例后必须 connect
3. **username/password** - 认证参数名
4. **环境变量** - 管理连接配置
5. **try-finally** - 确保连接关闭
