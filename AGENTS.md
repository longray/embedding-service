# Embedding Service - Agent 指南

## 项目注意事项

### Python 环境管理

**⚠️ 重要：不要删除 Python 虚拟环境**
- PyTorch 体积很大，重新下载浪费流量
- 如果包有问题，使用 `uv` 管理包修复

### 包管理

**使用 uv 管理依赖**：
```bash
# 安装包
uv pip install package_name

# 运行 Python 脚本
uv run python script.py

# 运行测试
uv run pytest tests/
```

## 项目结构

```
embedding_service/
├── src/                        # Embedding 和 LLM 服务
├── wrapper-service/            # 包装层服务
│   ├── src/                    # 包装层源码
│   │   ├── main.py            # FastAPI 主程序
│   │   └── utils/             # 工具模块
│   └── requirements.txt       # 依赖配置
└── tests/                      # 测试套件
```

## 开发命令

```bash
# 启动服务
uv run python start_services.py --with-llm

# 运行测试
uv run pytest tests/ -v

# 代码检查
uv run ruff check .
uv run pyright
```

## 最近变更

- 已移除 prometheus_client 依赖及相关监控代码
- 使用 structlog 进行日志记录
- API 认证通过环境变量 `WRAPPER_AUTH_ENABLED` 控制
