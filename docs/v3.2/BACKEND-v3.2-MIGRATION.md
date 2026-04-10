# 后端 v3.2 迁移指南

> **版本**: v3.2.0  
> **日期**: 2026-04-10  
> **状态**: 实施版  
> **目标**: 完成服务端口从 17999 到 18008 的迁移

---

## 目录

1. [迁移概述](#1-迁移概述)
2. [迁移步骤](#2-迁移步骤)
3. [配置更新](#3-配置更新)
4. [验证测试](#4-验证测试)
5. [回滚方案](#5-回滚方案)

---

## 1. 迁移概述

### 1.1 迁移范围

| 组件         | 当前端口    | 目标端口    | 说明       |
| ------------ | ----------- | ----------- | ---------- |
| FastAPI HTTP | 17999       | 18008       | 主服务端口 |
| WebSocket    | 17999       | 18008       | 同一端口   |
| Docker 映射  | 17999:17999 | 18008:18008 | 容器映射   |

### 1.2 影响范围

**后端**:

- `main.py` - 端口配置
- `config.py` - 默认端口
- `docker-compose.yml` - 端口映射

**插件端**:

- `wrapper-client.js` - 默认端口
- `agents/` - 配置更新

**文档**:

- 所有文档中的端口引用

---

## 2. 迁移步骤

### Step 1: 后端配置更新

```python
# wrapper/src/config.py
class Config:
    # 旧配置
    # PORT = 17999

    # 新配置
    PORT = 18008
    HOST = "0.0.0.0"

    # 向后兼容（可选）
    LEGACY_PORT = 17999
```

```python
# wrapper/src/main.py
import uvicorn
from config import Config

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=Config.HOST,
        port=Config.PORT,  # 18008
        reload=True
    )
```

### Step 2: Docker 配置更新

```yaml
# docker-compose.yml
services:
  memory-service:
    build: .
    ports:
      # - "17999:17999"  # 旧配置
      - "18008:18008" # 新配置
    environment:
      - PORT=18008
```

### Step 3: 插件端配置更新

```javascript
// lib/wrapper-client.js
class WrapperClient {
  constructor(options = {}) {
    // 旧配置
    // this.port = options.port || 17999;

    // 新配置
    this.port = options.port || 18008;

    // 向后兼容
    if (options.legacy) {
      this.port = 17999;
    }
  }
}
```

### Step 4: 环境变量更新

```bash
# .env
# 旧配置
# PORT=17999

# 新配置
PORT=18008
```

---

## 3. 配置更新清单

### 3.1 后端文件

| 文件                    | 更新内容               |
| ----------------------- | ---------------------- |
| `wrapper/src/config.py` | PORT = 18008           |
| `wrapper/src/main.py`   | 使用 Config.PORT       |
| `docker-compose.yml`    | ports: - "18008:18008" |
| `.env.example`          | PORT=18008             |

### 3.2 插件端文件

| 文件                    | 更新内容            |
| ----------------------- | ------------------- |
| `lib/wrapper-client.js` | defaultPort = 18008 |
| `agents/*.md`           | 端口引用更新        |
| `README.md`             | 端口说明更新        |

### 3.3 文档文件

| 文件           | 更新内容           |
| -------------- | ------------------ |
| `docs/**/*.md` | 所有 17999 → 18008 |

---

## 4. 验证测试

### 4.1 配置检查

```bash
# 全局搜索旧端口
grep -r "17999" --include="*.py" --include="*.js" --include="*.md" .

# 应该只出现在：
# - 迁移文档
# - 向后兼容代码
# - 历史记录
```

### 4.2 服务启动测试

```bash
# 启动服务
cd embedding_service/wrapper
uvicorn src.main:app --port 18008

# 测试连接
curl http://localhost:18008/health
```

### 4.3 端到端测试

```bash
# 运行完整测试套件
pytest tests/ -v

# 验证端口
netstat -an | grep 18008
```

---

## 5. 回滚方案

### 5.1 快速回滚

```bash
# 1. 停止服务
docker-compose down

# 2. 切换分支
git checkout v3.1

# 3. 启动旧版本
docker-compose up -d
```

### 5.2 配置回滚

```python
# 临时使用旧端口
export PORT=17999
python -m wrapper.src.main
```

---

## 参考文档

- [UNIFIED-ARCHITECTURE-v3.2.md](./UNIFIED-ARCHITECTURE-v3.2.md)
- [BACKEND-v3.2-IMPLEMENTATION.md](./BACKEND-v3.2-IMPLEMENTATION.md)

---

_文档版本: v3.2.0_  
_最后更新: 2026-04-10_
