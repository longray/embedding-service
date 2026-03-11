 ```python
"""
SurrealDB 长期连接 + FastAPI + Uvicorn 生命周期管理
一键复制完整示例
"""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from surrealdb import Surreal


# ==================== 数据库连接管理 ====================

class SurrealDBManager:
    """SurrealDB 连接管理器 - 单例模式"""
    _instance = None
    _db: Surreal = None
    _lock = asyncio.Lock()
    
    # 配置信息（根据实际情况修改）
    CONFIG = {
        "url": "ws://localhost:18002/rpc",  # WebSocket 连接地址
        "namespace": "test",
        "database": "test",
        "username": "root",
        "password": "root",
    }
    
    @classmethod
    async def get_instance(cls) -> "SurrealDBManager":
        """获取单例实例"""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    async def connect(self) -> None:
        """建立长期连接"""
        if self._db is None:
            self._db = Surreal(self.CONFIG["url"])
            await self._db.connect()
            await self._db.signin({
                "user": self.CONFIG["username"],
                "pass": self.CONFIG["password"]
            })
            await self._db.use(self.CONFIG["namespace"], self.CONFIG["database"])
            print(f"✅ SurrealDB 连接成功: {self.CONFIG['url']}")
    
    async def disconnect(self) -> None:
        """关闭连接"""
        if self._db is not None:
            await self._db.close()
            self._db = None
            print("🔌 SurrealDB 连接已关闭")
    
    @property
    def db(self) -> Surreal:
        """获取数据库连接实例"""
        if self._db is None:
            raise RuntimeError("数据库未连接，请先调用 connect()")
        return self._db


# ==================== FastAPI 生命周期管理 ====================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Uvicorn 生命周期管理：
    - Startup: 应用启动时建立数据库连接
    - Shutdown: 应用关闭时释放连接
    """
    # ===== Startup =====
    print("🚀 应用启动中...")
    db_manager = await SurrealDBManager.get_instance()
    await db_manager.connect()
    
    yield  # 应用运行期间保持连接
    
    # ===== Shutdown =====
    print("🛑 应用关闭中...")
    await db_manager.disconnect()


# 创建 FastAPI 应用（使用 lifespan）
app = FastAPI(
    title="SurrealDB API 示例",
    lifespan=lifespan
)


# ==================== API 接口示例 ====================

@app.get("/health")
async def health_check():
    """健康检查接口"""
    try:
        db_manager = await SurrealDBManager.get_instance()
        # 执行简单查询验证连接
        result = await db_manager.db.query("RETURN 'OK';")
        return {"status": "healthy", "db": "connected", "result": result[0]["result"]}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"数据库连接异常: {str(e)}")


@app.get("/users")
async def get_users():
    """获取所有用户"""
    try:
        db_manager = await SurrealDBManager.get_instance()
        # 使用长期连接执行查询
        users = await db_manager.db.query("SELECT * FROM user;")
        return {"data": users[0]["result"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/users")
async def create_user(name: str, email: str):
    """创建用户"""
    try:
        db_manager = await SurrealDBManager.get_instance()
        # 使用参数化查询防止注入
        result = await db_manager.db.query(
            "CREATE user SET name = $name, email = $email, created_at = time::now();",
            {"name": name, "email": email}
        )
        return {"data": result[0]["result"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/users/{user_id}")
async def get_user(user_id: str):
    """获取单个用户"""
    try:
        db_manager = await SurrealDBManager.get_instance()
        # 使用记录ID直接查询
        result = await db_manager.db.query(
            "SELECT * FROM user WHERE id = $id;",
            {"id": user_id}
        )
        if not result[0]["result"]:
            raise HTTPException(status_code=404, detail="用户不存在")
        return {"data": result[0]["result"][0]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 启动命令 ====================

"""
终端运行：
uvicorn main:app --host 0.0.0.0 --port 18002 --reload

生产环境：
uvicorn main:app --host 0.0.0.0 --port 18002 --workers 4
"""


# ==================== 可选：纯异步上下文管理器版本 ====================

"""
如果你不需要 FastAPI，只是想要一个长期连接的上下文管理器：
"""

import asyncio
from contextlib import asynccontextmanager


@asynccontextmanager
async def get_db_connection():
    """纯异步长期连接上下文管理器"""
    db = Surreal("ws://localhost:18002/rpc")
    try:
        await db.connect()
        await db.signin({"user": "root", "pass": "root"})
        await db.use("test", "test")
        print("✅ 数据库连接已建立")
        yield db
    finally:
        await db.close()
        print("🔌 数据库连接已关闭")


# 使用示例
async def standalone_example():
    """独立使用长期连接的示例"""
    async with get_db_connection() as db:
        # 在这个上下文中，连接一直保持
        result = await db.query("SELECT * FROM user;")
        print(result)
        # 可以执行多次操作，连接不会断开


if __name__ == "__main__":
    # 测试独立版本
    asyncio.run(standalone_example())
```

**安装依赖：**
```bash
pip install fastapi uvicorn surrealdb
```

**核心要点：**

| 组件 | 作用 | 生命周期 |
|------|------|----------|
| `SurrealDBManager` | 单例连接管理器 | 应用全局 |
| `lifespan` | FastAPI 生命周期钩子 | `startup` → `yield` → `shutdown` |
| `@asynccontextmanager` | 确保连接正确关闭 | `try` → `yield` → `finally` |

**执行流程：**
1. **Uvicorn 启动** → 触发 `lifespan` startup → 建立 WebSocket 连接
2. **API 请求** → 复用已建立的长期连接执行查询
3. **Uvicorn 关闭** → 触发 `lifespan` shutdown → 关闭连接释放资源