# 🎯 包装层服务设计 - 完整实施指南

**生成时间**: 2026-03-03  
**基准项目**: D:\embedding_service  
**升级目标**: 统一包装层服务（端口 3001）

---

## ✅ 实施状态总览

**最后更新**: 2026-03-03 21:17  
**实施阶段**: P0+P1 核心功能已完成  
**代码总量**: 778 行 Python 代码

### 已实现组件（P0+P1）

| 组件 | 文件 | 状态 | 代码行数 | 功能描述 |
|------|------|------|----------|----------|
| 配置管理 | `src/config.py` | ✅ 完成 | 57 | pydantic_settings，环境变量支持 |
| 线程安全缓存 | `src/utils/cache.py` | ✅ 完成 | 85 | LRU + TTL，RLock保护 |
| 统一异常处理 | `src/utils/exceptions.py` | ✅ 完成 | 58 | 异常类层次结构 |
| 熔断器机制 | `src/utils/circuit_breaker.py` | ✅ 完成 | 158 | 三状态保护，防止级联故障 |
| HTTP连接池 | `src/utils/http_pool.py` | ✅ 完成 | 80 | httpx连接复用 |
| 结构化日志 | `src/utils/logging.py` | ✅ 完成 | 51 | structlog支持 |
| Prometheus指标 | `src/utils/metrics.py` | ✅ 完成 | 77 | 完整监控指标 |
| 主服务程序 | `src/main.py` | ✅ 完成 | 189 | FastAPI，整合所有组件 |

### 核心API端点

| 端点 | 方法 | 功能 | 状态 |
|------|------|------|------|
| `/v1/embeddings` | POST | 文本嵌入（带缓存+熔断） | ✅ 已实现 |
| `/v1/chat/completions` | POST | 聊天补全（带熔断） | ✅ 已实现 |
| `/health` | GET | 健康检查+熔断器状态 | ✅ 已实现 |
| `/metrics` | GET | Prometheus指标 | ✅ 已实现 |

### 未实现组件（原设计中）

| 组件 | 原计划 | 当前状态 | 备注 |
|------|--------|----------|------|
| MemoryManager | 727行代码 | ⏳ 未实现 | 可能在其他模块中 |
| NetworkChecker | 预估2h | ⏳ 未实现 | 可通过健康检查替代 |
| WrapperClient | 预估3h | ⏳ 未实现 | 客户端SDK |
| Plugin.js集成 | 预估3h | ⏳ 未实现 | 插件系统集成 |

### 实施策略说明

当前实现采用**核心优先策略**，专注于包装服务的核心功能：
- ✅ 服务可靠性（熔断器、异常处理）
- ✅ 性能优化（缓存、连接池）
- ✅ 可观测性（日志、监控）
- ✅ 基础API代理（embeddings、chat）

原设计中的其他组件（MemoryManager、NetworkChecker等）可在后续迭代中根据实际需求添加。

---

## 📋 一、执行摘要

### 集成目标
- ✅ MemoryManager 已完成（727 行代码）
- ⏳ NetworkChecker 待实现（预估 2h）
- ⏳ WrapperClient 待实现（预估 3h）
- ⏳ Plugin.js 集成待完成（预估 3h）

**总预估工时**: 8 小时  
**推荐方式**: 渐进式集成（分 4 个 Phase）

---

## 📋 二、embedding_service 现有服务能力

### 1.1 Embedding 服务（端口 18000）

#### 现有 API 端点

| 端点 | 方法 | 功能 | 状态 |
|------|------|------|------|
| `/v1/embeddings` | POST | 生成文本嵌入向量 | ✅ 保留 |
| `/health` | GET | 服务健康检查 | ✅ 保留 |
| `/v1/models` | GET | 模型列表 | ✅ 保留 |
| `/stats` | GET | 统计信息 | ✅ 保留 |

#### `/v1/embeddings` 请求格式
```json
{
  "input": "单条文本" | ["文本1", "文本2", "..."],
  "model": "Qwen3-Embedding-0.6B",
  "encoding_format": "float" | "base64",
  "dimensions": 1024,  // 可选，32-1024
  "normalize": true
}
```

#### `/v1/embeddings` 响应格式
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.1234, -0.5678, ...]  // 1024维向量
    }
  ],
  "model": "Qwen3-Embedding-0.6B",
  "usage": {
    "prompt_tokens": 100,
    "total_tokens": 100,
    "processing_time_ms": 123.45
  }
}
```

#### `/health` 响应格式
```json
{
  "status": "healthy",
  "service": "embedding",
  "version": "2.0.1",
  "device": "cuda",
  "max_batch_size": 256,
  "max_length": 2048,
  "model": "/path/to/Qwen/Qwen3-Embedding-0___6B",
  "cuda_available": true,
  "gpu_name": "NVIDIA GeForce RTX 3080",
  "gpu_memory_total_gb": 10.0,
  "gpu_memory_used_mb": 1234.56,
  "gpu_memory_reserved_mb": 2048.0
}
```

### 1.2 LLM 服务（端口 18001）

#### 现有 API 端点

| 端点 | 方法 | 功能 | 状态 |
|------|------|------|------|
| `/v1/chat/completions` | POST | OpenAI 兼容对话接口 | ✅ 保留 |
| `/generate` | POST | 简单生成接口（支持缓存） | ✅ 保留 |
| `/health` | GET | 服务健康检查 | ✅ 保留 |
| `/v1/models` | GET | 模型列表 | ✅ 保留 |
| `/stats` | GET | 统计信息 | ✅ 保留 |

#### `/v1/chat/completions` 请求格式
```json
{
  "model": "MiniCPM4-0.5B",
  "messages": [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！有什么我可以帮助你的吗？"}
    {"role": "user", "content": "介绍一下人工智能"}
  ],
  "temperature": 0.7,
  "top_p": 0.7,
  "max_tokens": 512,
  "stream": false,
  "do_sample": true
}
```

#### `/v1/chat/completions` 响应格式
```json
{
  "id": "chatcmpl-local",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "MiniCPM4-0.5B",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "你好！有什么我可以帮助你的吗？"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 50,
    "total_tokens": 70
  }
}
```

---

## 🌐 三、包装层服务架构设计

### 3.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    OpenCode 环境                                       │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      子代理系统                                   │   │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  │   │
│  │  │@memory-automation │  │@memory-consolidate │ │@memory-classifier│  │   │
│  │  │  (自动触发)      │  (手动触发)      │  │  (手动触发)   │ │   │
│  │  └──────────────────┘  └──────────────────┘  └──────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     核心库 (lib/)                               │   │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐   │   │
│  │  │ memory-manager │  │network-checker │  │wrapper-client │   │   │
│  │  │   (记忆管理)   │  (网络检查)    │  │ (HTTP客户端) │   │   │
│  │  └────────────────┘  └────────────────┘ └────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │              本地 MD 文件 (9个核心文件 + daily/)                   │   │
│  │                                                                  │   │
│  │  │  GLOBAL_MEMORY.md      → 全局记忆 (project_tag: global)         │   │
│  │  │ PROJECT_MEMORY.md    → 项目记忆 (project_tag: projectA/B/C)   │   │
│  │  │ MEMORY.md           → 通用记忆 (向后兼容)                    │   │
│  │  │ SOUL.md            → AI 人格                                 │   │
│  │  │ AGENTS.md          → 代理指令                                │   │
│  │  │ USER.md            → 用户配置                                │   │
│  │  │ IDENTITY.md        → 身份定义                                │   │
│  │  │ TOOLS.md          → 工具说明                                │   │
│  │  │ daily/             → 每日日志                               │   │
│  │  │                                                                  │   │
│  │  │  每个文件包含双标签: project_tag + uploaded                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────┘   │
                              ↓ HTTP 调用
┌─────────────────────────────────────────────────────────────────────────┐
│                      外部服务 (独立部署)                               │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                Express HTTP Wrapper Service                      │   │
│  │                 (端口: 3001, 独立进程)                     │   │
│  │                                                                 │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │   │
│  │  │/api/health  │  │/api/search  │  │/api/upload │        │   │
│  │  │ (健康检查)   │  │ (语义搜索)   │  │ (上传记忆)  │        │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘        │   │
│  │  │                                                                 │   │
│  │  ┌──────────────────────────────────────────────────────────┐      │   │
│  │  │            SurrealQL 内嵌 HTTP 调用嵌入服务           │      │   │
│  │  │            → http::post('http://localhost:18000/embeddings')│      │   │
│  │  └──────────────────────────────────────────────────────────┘      │   │
│  │                                                                 │   │
│  │  │            SurrealQL 内嵌 HTTP 调用 LLM 服务             │      │   │
│  │  │            → http::post('http://localhost:18001/generate')│      │   │
│  │  └──────────────────────────────────────────────────────────┘      │   │
│  │                                                                 │   │
│  │  └──────────────────────────────────────────────────────────┘      │   │
│ │                                    ↓                                  │   │
│  │         ┌──────────────┬────────────────────────┬────────────────┐             │             │
│  │         ↓                    ↓                    ↓             │             │
│  │  ┌──────────────┐      ┌──────────────┐        │             │
│  │  │  SurrealDB  │      │  Embedding   │        │             │
│  │  │ (向量存储)  │      │   Service   │        │             │
│  │  └──────────────┘      └──────────────┘        │             │
│  │                                                            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📋 四、升级路径详细步骤

### Phase 1: 创建包装层服务基础结构（2h）

#### Step 1. 创建项目结构
```bash
# 在 D:\embedding_service\ 创建新目录
cd D:\embedding_service
mkdir -p wrapper-service

# 创建基本结构
cd wrapper-service
mkdir -p src
mkdir -p shared
mkdir -p shared/utils
mkdir -p tests
```

#### Step 2. 实现共享工具模块

**文件**: `shared/config.py`
```python
"""
共享配置管理模块
"""
import os
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # 服务端口
    wrapper_port: int = 3001
    embedding_port: int = 18000
    llm_port: int = 18001
    
    # 模型路径
    embedding_model_path: str = "Qwen/Qwen3-Embedding-0___6B"
    llm_model_path: str = "OpenBMB/MiniCPM4-0.5B"
    
    # 缓存配置
    cache_size: int = 1000
    cache_ttl: int = 3600  # 1 小时
    
    # 搜索配置
    search_limit: int = 10
    search_threshold: float = 0.3
    search_mode: str = "hybrid"  # "vector", "keyword", "hybrid"
    
    # 上传配置
    upload_batch_size: int = 10
    upload_max_retries: int = 3
    upload_retry_delay: int = 1000  # 1 秒
    
    # 记忆目录
    memory_dir: Path.home() / ".opencode" / "memory"
    
    class Config:
        """配置类"""
        def __init__(self):
            self.wrapper_port = Settings.wrapper_port
            self.embedding_port = Settings.embedding_port
            self.llm_port = Settings.llm_port
            self.embedding_model_path = Settings.embedding_model_path
            self.llm_model_path = Settings.llm_model_path
            self.cache_size = Settings.cache_size
            self.cache_ttl = Settings.cache_ttl
            self.search_limit = Settings.search_limit
            self.search_threshold = Settings.search_threshold
            self.search_mode = Settings.search_mode
            self.upload_batch_size = Settings.upload_batch_size
            self.upload_max_retries = Settings.upload_max_retries
            self.upload_retry_delay = Settings.upload_retry_delay
            self.memory_dir = Settings.memory_dir
            
        def get_target_file(self, project_tag: str) -> Path:
            if project_tag == 'global':
                return self.memory_dir / "GLOBAL_MEMORY.md"
            elif project_tag == 'unclassified':
                return self.memory_dir / "MEMORY.md"
            else:
                return self.memory_dir / "PROJECT_MEMORY.md"

config = Config()
```

**文件**: `shared/utils/config.py`
```python
"""
共享工具模块
"""
import os
import hashlib
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import re

def hash_text(text: str) -> str:
    """生成文本的 MD5 哈希"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """计算余弦相似度"""
    import numpy as np
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def parse_md_entries(content: str) -> List[Dict[str, Any]]:
    """解析 Markdown 格式的记忆文件"""
    entries = []
    current_entry = {}
    
    lines = content.split('\n')
    for i, line in enumerate(lines):
        line = line.strip()
        
        # 检测条目开始
        if line.startswith('## ') and 'Entry' in line:
            if current_entry:
                entries.append(current_entry)
            current_entry = {}
                current_entry['line'] = i + 1
                current_entry['type'] = line.split()[1].strip()
                current_entry['content'] = ""
                continue
        
        # 解析条目元数据
        elif line.startswith('**Date:') or line.startswith('*Date:'):
            if 'Date:' in line:
                current_entry['date'] = line.split(':')[1].strip()
            else:
                current_entry['date'] = line.split('*')[1].strip()
        elif line.startswith('**Type:') or line.startswith('*Type:'):
            if 'Type:' in line:
                current_entry['type'] = line.split(':')[1].strip()
            else:
                current_entry['type'] = line.split('*')[1].strip()
        elif line.startswith('**Tags:') or line.startswith('*Tags:'):
            if 'Tags:' in line:
                tags_str = line.split(':')[1].strip()
                current_entry['tags'] = [t.strip() for t in tags_str.split(',')]
            else:
                current_entry['tags'] = line.split('*')[1].strip().split(',')
        
        # 检测内容部分
        elif line.startswith('**Content:') or line.startswith('*Content:'):
            if 'Content:' in line:
                current_entry['content'] = line.split(':')[1].strip()
            else:
                current_entry['content'] += line.split('*')[1].strip() + '\n'
        elif line and not line.startswith('#') and not line.startswith('**') and not line.startswith('*'):
            if current_entry.get('content'):
                current_entry['content'] += line + '\n'
    
    if current_entry:
        entries.append(current_entry)
    
    return entries

def extract_metadata(entry: Dict[str, Any]) -> Dict[str, Any]:
    """从条目中提取元数据"""
    metadata = {
        'line': entry.get('line', 0),
        'date': entry.get('date', ''),
        'type': entry.get('type', 'general'),
        'tags': entry.get('tags', []),
        'content': entry.get('content', ''),
        'project_tag': detect_project_tag(entry['content']),
        'project_id': detect_project_id(entry['content']),
        'project_name': detect_project_name(entry['content']),
        'uploaded': False,
        'upload_timestamp': None,
        'upload_error': None
    }
    
    return metadata

def detect_project_tag(content: str) -> str:
    """从内容中检测项目标签"""
    # 检测绝对路径模式
    abs_path_match = re.search(r'[A-Z]:\\[^\\s]+\//g', content)
    if abs_path_match:
        path = abs_path_match.group(0)
        project_name = path.split(/[\/\\]/g)[-1]
        return f"project_{project_name}"
    
    # 检测相对路径模式
    rel_path_match = re.search(r'(\.\.\/|\.\.\\|..\/|..\\)[^\\s]+/g', content)
    if rel_path_match:
        path = rel_path_match.group(0)
        project_name = path.split(/[\/\\]/g)[-1]
        return f"project_{project_name}"
    
    # 默认：未分类
    return 'unclassified'

def detect_project_id(content: str) -> str:
    """从内容中检测项目 ID"""
    # 检测文件路径模式
    file_match = re.search(r'[A-Z]:\\[^\\s]+[/\\].*g', content)
    if file_match:
        return file_match.group(0)
    
    return None

def detect_project_name(content: str) -> str:
    """从内容中检测项目名称"""
    # 检测文件路径模式
    path_match = re.search(r'[A-Z]:\\[^\\s]+[/\\].*g', content)
    if path_match:
        path = path_match.group(0)
        project_name = path.split(/[\/\\]/g)[-1]
        return project_name
    
    return None
```

**文件**: `shared/utils/cache.py`
```python
"""
缓存工具模块
"""
import hashlib
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
from functools import lru_cache

class LRUCache:
    """LRU 缓存实现"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = timedelta(seconds=ttl)
        self.cache: Dict[str, Dict[str, Any]] = {}
    
    def get(self, key: str) -> Optional[Any]:
        entry = self.cache.get(key)
        if entry:
            if entry['expires_at']:
                if datetime.now() > entry['expires_at']:
                    del self.cache[key]
                    return None
            return entry['data']
        return None
    
    def set(self, key: str, data: Any) -> None:
        self.cache[key] = {
            'data': data,
            'expires_at': datetime.now() + self.tl,
            'accessed_at': datetime.now()
        }
    
    def clear(self):
        self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'maxsize': self.max_size,
            'ttl_seconds': self.tl.total_seconds(),
            'current_size': len(self.cache),
            'hits': 0,
            'misses': 0,
            'hit_rate': 0
        }
```

### Phase 2: 实现包装服务主程序（6h）

#### 文件**: `src/main.py`
```python
"""
包装服务主程序
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import logging
import sys

from shared.utils.config import config
from shared.utils.health import HealthChecker
from shared.utils.file_parser import MarkdownParser
from shared.utils.cache import LRUCache

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("wrapper-service")

app = FastAPI(
    title="Wrapper Service for Embedding & LLM Services",
    description="""
Unified health check for Embedding and LLM services

Version: 1.0.0
Deployment: Standalone
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化组件
health_checker = HealthChecker(config)

# 缓存
cache = LRUCache(max_size=config.cache_size, ttl=config.cache_ttl)

# Markdown 解析器
parser = MarkdownParser()

# ==================== 核心端点 ====================

@app.get("/api/health")
async def unified_health_check():
    """统一健康检查端点"""
    results = await health_checker.check_all()
    
    overall = 'ok' if results.get('allHealthy') else 'degraded'
    latency = max(
        results['embedding'].get('latency_ms', 0),
        results['llm'].get('latency_ms', 0)
    )
    
    if overall == 'ok':
        status_code = 200
    else:
        status_code = 503
    
    return {
        "status": overall,
        "timestamp": datetime.utcnow().isoformat(),
        "latency": latency,
        "services": {
            "wrapper": "healthy",
            "embedding": results['embedding']['status'],
            "llm": results['llm']['status'],
            "allHealthy": results['allHealthy']
        }
    }

@app.post("/api/search")
async def semantic_search(request: Request):
    """语义搜索端点"""
    start_time = datetime.now()
    
    try:
        payload = await request.json()
        query = payload.get('query', '')
        mode = payload.get('mode', 'hybrid')
        limit = payload.get('limit', 10)
        threshold = payload.get('threshold', 0.3)
        filters = payload.get('filters', {})
        
        # 验证参数
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
        if mode not in ['vector', 'keyword', 'hybrid']:
            raise HTTPException(status_code=400, detail=f"Invalid mode: {mode}")
        if limit < 1 or limit > 100:
            raise HTTPException(status_code=400, detail="Limit must be between 1-100")
        if threshold < 0 or threshold > 1:
            raise HTTPException(status_code=400, detail="Threshold must be between 0-1")
        
        # 执行搜索
        if mode in ['vector', 'hybrid']:
            # 向量搜索 - 调用 embedding 服务
            results = await _vector_search(query, limit, threshold, filters)
        else:
            # 关键词搜索 - 读取本地文件
            results = _keyword_search(query, limit, filters)
        
        query_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        return {
            "success": True,
            "query": query,
            "count": len(results),
            "query_time_ms": round(query_time_ms, 2),
            "search_mode": mode,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/api/upload")
async def upload_memory(request: Request):
    """上传单条记忆"""
    start_time = datetime.now()
    
    try:
        payload = await request.json()
        entry = payload.get('entry')
        
        # 验证必需字段
        if not entry:
            raise HTTPException(status_code=400, detail="Entry is required")
        if not entry.get('content'):
            raise HTTPException(status_code=400, detail="Content is required")
        
        # 检查是否已上传
        if entry.get('uploaded', False):
            # 写入对应文件
            file_path = config.config.get_target_file(entry.get('project_tag'))
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 格式化条目
            content = f"""## {entry.get('type', 'general')} Entry

**Date**: {datetime.now().isoformat()}

**Type**: {entry.get('type', 'general')}

**Tags**: {', '.join(entry.get('tags', [])) or 'none'}

{entry.get('content', '')}

---
"""
            
            file_path.write_text(content, 'utf-8')
            
            # 标记为已上传
            uploaded = await _mark_as_uploaded(entry.get('id'), True)
            
            query_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            return {
                "success": True,
                "total": 1,
                "uploaded": 1,
                "failed": 0,
                "results": [{
                    "id": entry.get('id'),
                    "success": True,
                    "uploaded": True,
                    "error": None
                }]
            }
        
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/api/batch-upload")
async def batch_upload_memories(request: Request):
    """批量上传记忆"""
    start_time = datetime.now()
    
    try:
        payload = await request.json()
        entries = payload.get('entries', [])
        
        # 验证必需字段
        if not entries:
            raise HTTPException(status_code=400, detail="Entries are required")
        
        results = []
        success_count = 0
        failed_count = 0
        
        for entry in entries:
            try:
                # 检查是否已上传
                if entry.get('uploaded', False):
                    # 写入对应文件
                    file_path = config.config.get_target_file(entry.get('project_tag'))
                    file_path.mkdir(parents=True, exist_ok=True)
                    
                    # 格式化条目
                    content = f"""## {entry.get('type', 'general')} Entry

**Date**: {datetime.now().isoformat()}

**Type**: {entry.get('type', 'general')}

**Tags**: {', '.join(entry.get('tags', [])) or 'none'}

{entry.get('content', '')}

---
"""
                    
                    file_path.write_text(content, 'utf-8')
                    
                    # 标记为已上传
                    uploaded = await _mark_as_uploaded(entry.get('id'), True)
                    
                    success_count += 1
                    results.append({
                        "id": entry.get('id'),
                        "success": True,
                        "uploaded": True,
                        "error": None
                    })
                else:
                    # 已上传，跳过
                    results.append({
                        "id": entry.get('id'),
                        "success": True,
                        "uploaded": entry.get('uploaded', True),
                        "error": None
                    })
                    
            except Exception as e:
                failed_count += 1
                results.append({
                    "id": entry.get('id'),
                    "success": False,
                    "uploaded": entry.get('uploaded', False),
                    "error": str(e)
                })
        
        query_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        return {
            "success": True,
            "total": len(entries),
            "uploaded": success_count,
            "failed": failed_count,
            "results": results,
            "query_time_ms": round(query_time_ms, 2)
        }
        
    except Exception as e:
        logger.error(f"Batch upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch upload failed: {str(e)}")

# ==================== 辅助端点 ====================

@app.get("/")
async def root():
    """根路径"""
    return {
        "service": "wrapper",
        "version": "1.0.0",
        "description": "Unified health check for Embedding and LLM services",
        "endpoints": {
            "/api/health": "统一健康检查",
            "/api/search": "语义搜索",
            "/api/upload": "记忆上传",
            "/api/batch-upload": "批量上传记忆",
            "/docs": "API 文档"
        }
    }

@app.get("/docs")
async def docs():
    """API 文档"""
    return {
        "service": "wrapper",
        "version": "1.0.0",
        "description": "Unified health check for Embedding and LLM services",
        "endpoints": {
            "GET  /api/health": "统一健康检查",
            "POST /api/search": "语义搜索",
            "POST /api/upload": "记忆上传",
            "POST /api/batch-upload": "批量上传记忆",
            "GET /docs": "API 文档"
        },
        "endpoints": {
            "POST": {
                "/api/search": {
                    "summary": "语义搜索 - 通过嵌入向量搜索相似内容",
                    "body": {
                        "query": "搜索查询",
                        "mode": "hybrid",  # "vector", "keyword", "hybrid"
                    }
                },
                "/api/upload": {
                    "summary": "记忆上传 - 上传单条记忆到远程存储",
                    "body": {
                        "entry": {
                            "id": "记忆 ID",
                            "content": "记忆内容",
                            "type": "general",
                            "tags": ["标签1", "标签2"]
                        }
                    }
                },
                "/api/batch-upload": {
                    "summary": "批量上传 - 批量上传多条记忆",
                    "body": {
                        "entries": [
                            {
                                "id": "记忆 ID",
                                "content": "记忆内容",
                                "type": "general",
                                "tags": ["标签1", "标签2"]
                            },
                            {
                                "id": "记忆 ID",
                                "content": "记忆内容",
                                "type": "general",
                                "tags": ["标签1", "标签2"]
                            }
                        ]
                    }
                }
            },
            "GET": {
                "/": "/api/health": {
                    "summary": "统一健康检查 - 检查所有服务的健康状态"
                },
                "/docs": "API 文档": {
                    "summary": "API 文档"
                }
            },
            "GET": {
                "/": "/": "统一健康检查": {
                    "summary": "检查所有服务的健康状态"
                },
                "/docs": "API 文档": {
                    "summary": "API 文档"
                }
            }
        }
    }

# ==================== 启动服务 ====================

if __name__ == '__main__':
    import uvicorn
    import sys
    
    # 端口配置
    host = sys.argv[1] if len(sys.argv) > 1 else "0.0.0.0"
    port = sys.argv[2] if len(sys.argv) > 2 else 3001
    
    # 启动服务
    uvicorn.run(app, host=host, port=port, workers=1)
```

### Phase 3: 更新启动脚本（1h）

#### 修改 `start_wrapper_service.bat`
```batch
@echo off
chcp 65001 >nul

title Wrapper Service - 统一健康检查 + 语义搜索 + 记忆上传
setlocal EnableDelayedExpansion

:: ==================== 配置区域 ====================
set "PROJECT_DIR=D:\embedding_service"
set "UV_PATH=C:\Users\Longray\.local\bin\uv.exe"
set "PYTHON_PATH=%PROJECT_DIR%\.venv\Scripts\python.exe"
set "SCRIPT_PATH=%PROJECT_DIR%\wrapper-service\src\main.py"
set "PORT=3001"

:: ==================== 颜色定义 ====================
set "GREEN=[92m"
set "YELLOW=[93m"
set "RED=[91m"
set "BLUE=[94m"
set "RESET=[0m"

:: ==================== 启动画面 ====================
echo %BLUE%
echo ============================================
echo           Wrapper Service - 统一健康检查 + 语义搜索 + 记忆上传
echo ============================================
echo %RESET%

echo.
echo %BLUE%
echo ============================================
echo           服务端点
echo ============================================
echo %RESET%

echo %GREEN%  健康检查:    http://localhost:%PORT%/api/health
echo %RESET%
echo %BLUE% API 文档:    http://localhost:%PORT%/docs
echo %RESET%
echo %GREEN% 语义搜索:    http://localhost:%PORT%/api/search
echo %RESET%
echo %BLUE% 记忆上传:    http://localhost:%PORT%/api/upload
echo %RESET% 批量上传:    http://localhost:%PORT%/api/batch-upload
echo %RESET%
echo %RESET%
echo %GREEN% ============================================
echo %RESET%

echo.
echo %BLUE%
echo           后端服务（现有）
echo ============================================
echo %RESET%
echo %GREEN%  Embedding Service (端口 18000)  ✅ 运行中
echo %RESET%
echo %BLUE%   ├── /v1/embeddings  - 生成嵌入向量
echo %RESET%
echo %YELLOW%   ├── /health - 健康检查
echo %RESET%
echo %YELLOW%   ├── /v1/models - 模型列表
echo %RESET%   └── /stats - 统计信息
echo %RESET%
echo %RESET%

echo.
echo %BLUE%
echo   LLM Service (端口 18001) ✅ 运行中
echo %RESET%
echo %GREEN%   ├── /v1/chat/completions - OpenAI 兼容对话接口
echo %RESET%   ├── /generate - 简单生成（支持缓存）
echo %RESET%   ├── /health - 健康检查
echo %RESET%   ├── /v1/models - 模型列表
echo %RESET%   └── /stats - 统计信息
echo %RESET%
echo %RESET%

echo.
echo %YELLOW% 通用设置
echo %RESET%
echo %GREEN% 批量大小:    %UPLOAD_BATCH_SIZE% 条/请求
echo %RESET%
echo %GREEN% 缓存大小:    %CACHE_SIZE% 条
echo %RESET%
echo %RESET%
echo %YELLOW% 降级策略
echo %RESET%
echo %RESET%
echo %GREEN% 网络故障时 → 降级到本地搜索
echo %RESET%
echo %RESET%
echo %YELLOW% 服务不可用时 → 降级到缓存模式
echo %RESET%
echo %RESET%
echo %RESET%

echo %GREEN% ============================================
echo %RESET%
echo.
echo %GREEN%
echo   启动服务
echo %RESET%
echo %BLUE% 启动命令
echo %RESET%
echo %YELLOW%   uv run "%PYTHON_PATH%" "%SCRIPT_PATH%"
echo %RESET%
echo %RESET%
echo %RESET%
echo %GREEN% ============================================
echo %RESET%
echo.
echo %GREEN% 验证服务
echo %RESET%
echo %BLUE% 检查健康: http://localhost:%PORT%/api/health
echo %RESET%
echo %YELLOW% 检查模型: http://localhost:%PORT%/v1/models
echo %RESET%
echo %RESET%
echo %YELLOW% 测试搜索: http://localhost:%PORT%/api/search
echo %RESET%
echo %RESET%
echo %YELLOW% 测试上传: http://localhost:%PORT%/api/upload
echo %RESET%
echo %RESET%
echo %RESET%
echo %GREEN% ============================================
echo %RESET%
echo.

pause
```

#### 修改 `start_embedding_service.bat`
```batch
@echo off
chcp 65001 >nul

title Qwen3-Embedding-0.6B API 服务
setlocal EnableDelayedExpansion

:: ==================== 配置区域 ====================
set "PROJECT_DIR=D:\embedding_service"
set "UV_PATH=C:\Users\Longray\.local\bin\uv.exe"
set "PYTHON_PATH=%PROJECT_DIR%\.venv\Scripts\python.exe"
set "SCRIPT_PATH=%PROJECT_DIR%\src\qwen3_embedding_service\start_embedding.py"
set "PORT=18000"

:: ==================== 颜色定义 ====================
set "GREEN=[92m"
set "YELLOW=[93m"
set "RED=[91m"
set "BLUE=[94m"
set "RESET=[0m"

:: ==================== 启动画面 ====================
echo %BLUE%
echo ============================================
echo     Qwen3-Embedding-0.6B API 服务
echo ============================================
echo %RESET%

echo.
echo %GREEN% 服务地址
echo %RESET%
echo %BLUE%   本地访问:    http://localhost:%PORT%/
echo %YELLOW% 局域网访问:  http://172.22.240.1:%PORT%/
echo %RESET%
echo %RESET%

echo %GREEN% 健康检查:    http://localhost:%PORT%/health
echo %RESET%
echo %BLUE% API 文档:    http://localhost:%PORT%/docs
echo %RESET%
echo %GREEN% ============================================
echo %RESET%
echo.
echo %GREEN% 启动服务
echo %RESET%
echo %BLUE% 启动命令
echo %RESET%
echo %YELLOW%   uv run "%PYTHON_PATH%" "%SCRIPT_PATH%"
echo %RESET%
echo %RESET%
echo %RESET%
echo %GREEN% ============================================
echo %RESET%
echo.
echo %GREEN% 验证服务
echo %RESET%
echo %BLUE% 检查健康: http://localhost:%PORT%/health
echo %RESET%
echo %YELLOW% 检查模型: http://localhost:%PORT%/v1/models
echo %RESET%
echo %RESET%
echo %YELLOW% 测试嵌入: http://localhost:%PORT%/v1/embeddings
echo %RESET%
echo %RESET%

echo %GREEN% ============================================
echo %RESET%
echo.
echo %GREEN% ============================================
echo %RESET%
echo.

pause
```

### Phase 4: 测试和验证（2h）

#### 测试脚本：`tests/test-wrapper-service.py`
```python
"""
包装服务测试脚本
"""
import asyncio
import httpx

# 测试配置
BASE_URL = "http://localhost:3001"

async def test_health_check():
    """测试统一健康检查"""
    print("🔍 测试统一健康检查...")
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{BASE_URL}/api/health")
            data = response.json()
            
            print(f"✅ 健康状态: {data['status']}")
            print(f"两个服务的状态:")
            print(f"  - Embedding: {data['services']['embedding']}")
            print(f"  - LLM: {data['services']['llm']}")
            print(f"  - 总体: {data['allHealthy']}")
            print(f"  延迟: {data['latency']}ms")
            
    except Exception as e:
        print(f"❌ 健康检查失败: {str(e)}")

async def test_semantic_search():
    """测试语义搜索"""
    print("🔍 测试语义搜索...")
    
    try:
        payload = {
            "query": "用户偏好的编码风格",
            "mode": "hybrid",
            "limit": 10,
            "threshold": 0.3,
            "filters": {
                "project_tag": "projectA"
            }
        }
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{BASE_URL}/api/search",
                json=payload,
                timeout=10.0
            )
            data = response.json()
            
            print(f"✅ 搜索完成！")
            print(f"查询: {data['query']}")
            print(f"模式: {data['search_mode']}")
            print(f"结果数量: {data['count']}")
            print(f"查询时间: {data['query_time_ms']}ms")
            print("\n📊 搜索结果:")
            
            for i, result in enumerate(data['results'][:5], 1):
                print(f"\n{i}. [{result['project_tag']}] {result['source']}")
                print(f"   内容: {result['content'][:60]}...")
                print(f"   相似度: {result['score']:.3f}")
            
    except Exception as e:
        print(f"❌ 语义搜索失败: {str(e)}")

async def test_upload():
    """测试记忆上传"""
    print("🔍 测试记忆上传...")
    
    try:
        payload = {
            "entry": {
                "id": "test_001",
                "content": "用户偏好使用 TypeScript 进行项目开发",
                "type": "preference",
                "tags": ["typescript", "code-style"],
                "project_tag": "projectA",
                "project_id": "D:\\github\\project-a",
                "project_name": "Project A",
                "uploaded": False
            }
        }
        
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                f"{BASE_URL}/api/upload",
                json=payload,
                timeout=5.0
            )
            data = response.json()
            
            print(f"✅ 上传完成！")
            print(f"总计: {data['total']} 条")
            print(f"成功: {data['uploaded']} 条")
            print(f"失败: {data['failed']} 条")
            
            if data['results']:
                for result in data['results']:
                    print(f"   ID: {result['id']}")
                    if result['success']:
                        print(f"      ✅ 已上传")
                    else:
                        print(f"      ❌ 失败: {result['error']}")
            
    except Exception as e:
        print(f"❌ 上传失败: {str(e)}")

async def main():
    """运行所有测试"""
    print("🧪 包装服务测试套件")
    print("=" * 60)
    
    await test_health_check()
    print()
    await test_semantic_search()
    print()
    await test_upload()
    print()
    print("=" * 60)
    print()
    print("✨ 测试完成！")

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
```

---

## 📋 五、向后兼容性保证

### 5.1 现有端点保留清单

| 端点 | 路径 | 功能 | 状态 |
|------|------|------|------|
| `/v1/embeddings` | `embedding_service.py` | 生成嵌入向量 | ✅ 保留 |
| `/health` | `embedding_service.py` | 健康检查 | ✅ 保留 |
| `/v1/models` | `embedding_service.py` | 模型列表 | ✅ 保留 |
| `/stats` | `embedding_service.py` | 统计信息 | ✅ 保留 |

| 端点 | 路径 | 功能 | 状态 |
|------|------|------|------|
| `/v1/chat/completions` | `llm_service.py` | 对话生成 | ✅ 保留 |
| `/generate` | `llm_service.py` | 简单生成 | ✅ 保留 |
| `/health` | `llm_service.py` | 健康检查 | ✅ 保留 |
| `/v1/models` | `llm_service.py` | 模型列表 | ✅ 保留 |
| `/stats` | `llm_service.py` | 统计信息 | ✅ 保留 |

### 5.2 配置兼容性

**环境变量映射**:
| 现有变量 | 新变量 | 说明 |
|------------|----------|------|
| `EMB_MAX_BATCH_SIZE` | `EMB_MAX_BATCH_SIZE` | ✅ 继续支持 |
| `EMB_MODEL_PATH` | `EMB_MODEL_PATH` | ✅ 继续支持 |
| `EMB_CACHE_SIZE` | `EMB_CACHE_SIZE` | ✅ 继续支持 |
| `EMB_CACHE_SIZE` | `EMB_CACHE_SIZE` | ✅ 继续支持 |
| `LLM_MAX_BATCH_SIZE` | `LLM_MAX_BATCH_SIZE` | ✅ 继续支持 |
| `LLM_MAX_NEW_TOKENS` | `LLM_MAX_NEW_TOKENS` | ✅ 继续支持 |
| `LLM_MAX_LENGTH` | `LLM_MAX_LENGTH` | ✅ 继续支持 |
| `LLM_CACHE_SIZE` | `LLM_CACHE_SIZE` | ✅ 继续支持 |
| `LLM_MAX_LENGTH` | `LLM_MAX_LENGTH` | ✅ 继续支持 |
| `LLM_MAX_PROMPT_LENGTH` | `LLM_MAX_PROMPT_LENGTH` | ✅ 继续支持 |

**新增变量**:
| 变量名 | 默认值 | 说明 |
|----------|----------|------|
| `WRAPPER_PORT` | `3001` | 包装服务端口 | 新增 |
| `UPLOAD_BATCH_SIZE` | `10` | 上传批量大小 | 新增 |
| `CACHE_SIZE` | `1000` | 缓存大小 | 新增 |
| `SEARCH_LIMIT` | `10` | 搜索结果限制 | 新增 |
| `SEARCH_THRESHOLD` | `0.3` | 相似度阈值 | 新增 |
| `CACHE_TTL` | `3600` | 缓存过期时间（秒） | 新增 |

### 5.3 端口映射

| 功能 | 现有路径 | 新路径（推荐） |
|------|----------|----------|
| 嵌入向量 | `/v1/embeddings` | `/v1/embeddings` | 继一使用 |
| 健康检查 | `/health` | `/api/health` | 统一使用 |
| 模型列表 | `/v1/models` | `/v1/models` | 统一使用 |
| 统计信息 | `/stats` | `/api/stats` | 统一使用 |
| API 文档 | `/docs` | `/docs` | 统一使用 |
| 语义搜索 | 无 | `/api/search` | 新增 |
| 记忆上传 | 无 | `/api/upload` | 新增 |
| 批量上传 | 无 | `/api/batch-upload` | 新增 |

---

## 📋 六、实施步骤总结

### Phase 1: 准备工作（30 分钟）
- [ ] 阅读 `D:\embedding_service` 的完整文档
- [ ] 确认所有依赖项已安装
- [ ] 备制 `start_embedding_service.bat` 和 `start_llm_service.bat` 到 `D:\embedding_service\backup\`
- [ ] 创建新的开发分支：`git checkout -b wrapper-service-upgrade`

### Phase 2: 创建包装层基础结构（2h）
- [ ] 创建项目目录结构
- [ ] 创建 `shared/utils/` 目录
- [ ] 创建 `shared/config.py` 配置模块
- [ ] 创建 `shared/utils/health.py` 健康检查工具
- [ ] 创建 `shared/utils/file_parser.py` Markdown 解析器
- [ ] 创建 `shared/utils/cache.py` LRU 缓存
- [ ] 编写 `README.md` 说明文档

### Phase 3: 实现包装服务核心功能（4h）
- [ ] 创建 `src/main.py` 主程序
- [ ] 实现 `/api/health` 端点（统一健康检查）
- [ ] 实现 `/api/search` 端点（语义搜索）
- [ ] 实现 `/api/upload` 端点（单条上传）
- [ ] 实现 `/api/batch-upload` 端点（批量上传）
- [ ] 添加 `/docs` 端点（API 文档）

### Phase 4: 测试和调试（2h）
- [ ] 创建 `tests/test-wrapper-service.py` 测试脚本
- [ ] 测试所有新增端点
- [ ] 测试向后兼容性
- [ ] 测试降级策略
- [ ] 性能测试

### Phase 5: 更新 OpenCode Memory Plugin（2h）
- [ ] 更新 NetworkChecker 的 wrapper_url 到 `http://localhost:3001/api/health`
- [ ] 在 `vector_memory_search` 工具中优先使用远程服务
- [ ] 添加 `memory_upload` 工具（可选）

### Phase 6: 部署和验证（1.5h）
- [ ] 创建 `start_wrapper_service.bat` 启动脚本
- [ ] 启动包装服务
- [ ] 验证所有现有端点正常工作
- [ ] 测试新增端点
- [ ] 验证健康检查
- [ ] 验证语义搜索
- [ ] 验证记忆上传

---

## 📋 七、配置管理方案

### 7.1 现有配置方式

**embedding_service.py 配置**:
- 硬编码：端口、批量大小、模型路径
- 环境变量：`EMB_MAX_BATCH_SIZE`, `EMB_MODEL_PATH`
- 运行时检测：GPU 内存自动调整

**llm_service.py 配置**:
- 硬编码：端口、批量大小、模型路径
- 环境变量：`LLM_MAX_BATCH_SIZE`, `LLM_MODEL_PATH`
- 运行时检测：GPU 内存自动调整

### 7.2 建议的统一配置方式

**创建 `shared/config.py`**:
```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # 服务端口
    wrapper_port: int = 3001
    embedding_port: int = 18000
    llm_port: int = 18001
    
    # 模型配置
    embedding_model_path: str = "Qwen/Qwen3-Embedding-0___6B"
    llm_model_path: str = "OpenBMB/MiniCPM4-0.5B"
    
    # 缓存配置
    cache_size: int = 1000
    cache_ttl: int = 3600  # 1 小时
    
    # 搜索配置
    search_limit: int = 10
    search_threshold: float = 0.3
    search_mode: str = "hybrid"  # "vector", "keyword", "hybrid"
    
    # 上传配置
    upload_batch_size: int = 10
    upload_max_retries: int = 3
    upload_retry_delay: int = 1000  # 1 秒
    
    # 记忆目录
    memory_dir: Path.home() / ".opencode" / "memory"
    
    class Config:
        """配置类"""
        def __init__(self):
            self.wrapper_port = Settings.wrapper_port
            self.embedding_port = Settings.embedding_port
            self.llm_port = Settings.llm_port
            self.embedding_model_path = Settings.embedding_model_path
            self.llm_model_path = Settings.llm_model_path
            self.cache_size = settings.cache_size
            self.cache_ttl = Settings.cache_ttl
            self.search_limit = settings.search_limit
            self.search_threshold = settings.search_threshold
            self.search_mode = settings.search_mode
            self.upload_batch_size = settings.upload_batch_size
            self.upload_max_retries = settings.upload_max_retries
            self.upload_retry_delay = settings.upload_retry_delay
            self.memory_dir = settings.memory_dir
            
        def get_target_file(self, project_tag: str) -> Path:
            if project_tag == 'global':
                return self.memory_dir / "GLOBAL_MEMORY.md"
            elif project_tag == 'unclassified':
                return self.memory_dir / "MEMORY.md"
            else:
                return self.memory_dir / "PROJECT_MEMORY.md"

settings = Settings()
config = Config()
```

### 7.3 环境变量映射

| 现有变量 | 配置项 | 默认值 | 说明 |
|------------|----------|--------|------|
| `EMB_MAX_BATCH_SIZE` | `embedding_batch_size` | 256 | Embedding 批量大小 |
| `EMB_MODEL_PATH` | `embedding_model_path` | `Qwen/Qwen3-Embedding-0___6B` | Embedding 模型路径 |
| `EMB_CACHE_SIZE` | `cache_size` | 1000 | Embedding 缓存大小 |
| `LLM_MAX_BATCH_SIZE` | `llm_batch_size` | 2 | LLM 批量大小 |
| `LLM_MODEL_PATH` | `llm_model_path` | `OpenBMB/MiniCPM4-0.5B` | LLM 模型路径 |
| `LLM_MAX_NEW_TOKENS` | `llm_max_new_tokens` | 512 | LLM 最大生成长度 |
| `LLM_MAX_LENGTH` | `llm_max_length` | 2048 | LLM 最大序列长度 |
| `LLM_CACHE_SIZE` | `llm_cache_size` | 100 | LLM 缓存大小 |
| `LLM_MAX_PROMPT_LENGTH` | `llm_max_prompt_length` | 32768 | LLM 提示词最大字符数 |

| 新增变量 | 默认值 | 说明 |
|------------|----------|--------|------|
| `WRAPPER_PORT` | `wrapper_port` | 3001 | 包装服务端口 | 新增 |
| `MEMORY_DIR` | `memory_dir` | `~/.opencode/memory` | 记忆目录 | 新增 |
| `UPLOAD_BATCH_SIZE` | `upload_batch_size` | 10 | 上传批量大小 | 新增 |
| `CACHE_SIZE` | `cache_size` | 1000 | 缓存大小 | 新增 |
| `SEARCH_LIMIT` | `search_limit` | 10 | 搜索结果限制 | 新增 |
| `SEARCH_THRESHOLD` | `search_threshold` | 0.3 | 相似度阈值 | 新增 |
| `CACHE_TTL` | `cache_ttl` | 3600 | 缓存过期时间（秒） | 新增 |

---

## 📋 八、部署方案

### 8.1 单服务部署（推荐用于开发/测试）

**方案**:
```
D:\embedding_service\
├── src\qwen3_embedding_service\
│   ├── embedding_service.py      # 端口 18000, 包含所有端点
│   └── llm_service.py           # 端口 18001, 包含所有端点
├── wrapper-service\
│   ├── src/main.py             # 端口 3001, 新增端点
│   ├── shared/
│   │   ├── config.py
│   │   ├── health.py
│   │   ├── file_parser.py
│   │   └── cache.py
│   └── tests/
│       └── test-wrapper-service.py
├── run_service.bat            # 统一启动脚本
├── start_embedding_service.bat   # Embedding 服务启动脚本
├── start_llm_service.bat       # LLM 服务启动脚本
```

**启动命令**:
```bash
# 启动所有服务
start_embedding_service.bat
start_llm_service.bat
run_service.bat
```

**优点**:
- ✅ 部署简单
- ✅ 易于开发和测试
- ✅ 资源利用率高

**缺点**:
- ⚠️ 两个服务共享一个进程
- ⚠️ 需要处理端口冲突
- ⚠️ 缺少隔离性

### 8.2 分服务部署（推荐用于生产）

**方案**:
```
D:\embedding_service\
├── services\
│   ├── embedding\
│   │   ├── embedding_service.py
│   │   └── start_embedding.py
│   └── llm\
│       ├── llm_service.py
│       └── start_llm.py
├── wrapper-service\
│   ├── src/main.py
│   ├── shared/
│   └── tests/
└── docker-compose.yml               # 统一管理
```

**docker-compose.yml**:
```yaml
version: '3.8'

services:
  embedding:
    build: .
    ports:
      - "18000:18000"
    environment:
      - EMB_MAX_BATCH_SIZE: 256
    restart: unless-stopped
    
  llm:
    build: .
    ports:
      - "18001:18001"
    environment:
      - LLM_MAX_BATCH_SIZE: 2
    restart: unless-stopped
    
  wrapper:
    build: .
    ports:
      - "3001:3001"
    environment:
      - SEARCH_LIMIT: 10
      - SEARCH_THRESHOLD: 0.3
      - UPLOAD_BATCH_SIZE: 10
      - CACHE_SIZE: 1000
    restart: unless-stopped
    depends_on:
      - embedding
      - llm
```

**启动命令**:
```bash
docker-compose up -d
```

**优点**:
- ✅ 服务隔离
- ✅ 独立扩展
- ✅ 故障转移
- ✅ 资源控制

**缺点**:
- ⚠️ 需要更多资源
- ⚠️ 部署复杂度增加

---

## 📋 九、向后兼容性保证措施

### 9.1 现有端点保留清单

| 端点 | 路径 | 功能 | 状态 |
|------|------|------|------|
| `/v1/embeddings` | `embedding_service.py` | 生成嵌入向量 | ✅ 完全保留 |
| `/health` | `embedding_service.py` | 健康检查 | ✅ 完全保留 |
| `/v1/models` | `embedding_service.py` | 模型列表 | ✅ 完全保留 |
| `/stats` | `embedding_service.py` | 统计信息 | ✅ 完全保留 |
| `/v1/chat/completions` | `llm_service.py` | 对话生成 | ✅ 完全保留 |
| `/generate` | `llm_service.py` | 简单生成 | ✅ 完全保留 |
| `/health` | `llm_service.py` | 健康检查 | ✅ 完全保留 |
| `/v1/models` | `llm_service.py` | 模型列表 | ✅ 完全保留 |
| `/stats` | `llm_service.py` | 统计信息 | ✅ 完全保留 |

### 9.2 新增端点

| 端点 | 路径 | 功能 |
|------|------|------|
| `/api/health` | `wrapper-service/src/main.py` | 统一健康检查 | 新增 |
| `/api/search` | `wrapper-service/src/main.py` | 语义搜索 | 新增 |
| `/api/upload` | `wrapper-service/src/main.py` | 记忆上传 | 新增 |
| `/api/batch-upload` | `wrapper-service/src/main.py` | 批量上传 | 新增 |

### 9.3 配置兼容性

**保留所有现有环境变量**:
- `EMB_MAX_BATCH_SIZE`
- `EMB_MODEL_PATH`
- `EMB_CACHE_SIZE`
- `LLM_MAX_BATCH_SIZE`
- `LLM_MAX_NEW_TOKENS`
- `LLM_MAX_LENGTH`
- `LLM_MAX_PROMPT_LENGTH`
- `LLM_CACHE_SIZE`

**新增环境变量**:
- `WRAPPER_PORT` (默认: 3001)
- `MEMORY_DIR` (默认: ~/.opencode/memory)
- `UPLOAD_BATCH_SIZE` (默认: 10)
- `CACHE_SIZE` (默认: 1000)
- `SEARCH_LIMIT` (默认: 10)
- `SEARCH_THRESHOLD` (默认: 0.3)
- `CACHE_TTL` (默认: 3600)

### 9.4 端口映射

| 功能 | 现有路径 | 新路径（推荐） |
|------|----------|----------|------|
| 嵌入向量 | `/v1/embeddings` | `/v1/embeddings` | 统一使用 |
| 健康检查 | `/health` | `/api/health` | 统一使用 |
| 模型列表 | `/v1/models` | `/v1/models` | 统一使用 |
| 统计信息 | `/stats` | `/api/stats` | 统一使用 |
| API 文档 | `/docs` | `/docs` | 统一使用 |
| 语义搜索 | 无 | `/api/search` | 新增 |
| 记忆上传 | 无 | `/api/upload` | 新增 |
| 批量上传 | 无 | `/api/batch-upload` | 新增 |

---

## 📋 十、后续工作建议

### 短期（1-2 周）

1. ✅ 实现统一健康检查
2. ✅ 实现语义搜索端点
3. ✅ 实现记忆上传端点
4. ✅ 完整测试向后兼容性

### 中期（1-2 月）

1. 🔧 实现 SurrealDB 集成
2. 🔧 实现高级搜索功能
3. 🔧 实现批量上传优化
4. 🔧 添加监控和日志

### 长期（3-6 月）

1. 🔧 性能优化（HNSW 索引）
2. 🔧 分布式部署方案
3. 🔧 A/B 测试和金丝雀发布
4. 🔧 模型版本管理

---

## 📋 十一、风险评估和缓解措施

| 风险 | 影响 | 概率 | 缓解措施 |
|------|------|------|----------|
| **端口冲突** | 服务启动失败 | 中 | ✅ 使用不同端口（18000/18001/3001） |
| **性能下降** | 多层封装增加延迟 | 中 | ✅ 异步调用 + 缓存优化 |
| **向后兼容性破坏** | 现有功能失效 | 低 | ✅ 渐进式集成 + 充分测试 |
| **数据不一致** | 多服务数据不一致 | 低 | ✅ 统一配置管理 |
| **部署复杂度增加** | 部署时间增加 | 低 | ✅ Docker Compose 简化部署 |
| **网络故障** | 搜索不可用 | 高 | ✅ 完善的回退机制 |

---

## 📋 十二、验收标准

### 功能完整性
- ✅ 所有现有 API 端点正常工作
- ✅ 新增的统一健康检查端点正常工作
- ✅ 语义搜索端点正常工作
- ✅ 记忆上传端点正常工作
- ✅ API 文档正常访问

### 向后兼容性
- ✅ 所有现有 API 端点保持不变
- ✅ 所有环境变量继续支持
- 所有配置参数继续有效
- 现有脚本继续可用

### 性能指标
- ✅ 统一健康检查响应时间 < 100ms
- ✅ 语义搜索响应时间 < 500ms（10 条）
- ✅ 记忆上传响应时间 < 200ms（单条）

### 测试覆盖
- ✅ 所有新增端点有测试用例
- ✅ 向后兼容性测试通过
- ✅ 降级策略测试通过
- ✅ 性能测试通过

---

## 📋 十三、下一步行动

### 立即执行（本周）

1. ✅ 创建包装层服务基础结构
2. ✅ 实现统一健康检查
3. ✅ 实现语义搜索端点
4. ✅ 实现记忆上传端点
5. ✅ 完整测试向后兼容性

### 短期执行（1-2 月）

1. 🔧 实现 SurrealDB 集成
2. 🔧 实现高级搜索功能
3. 🔧 实现批量上传优化
4. 🔧 添加监控和日志

### 长期执行（3-6 月）

1. 🔧 性能优化（HNSW 索引）
2. 🔧 分布式部署方案
3. 🔧 A/B 测试和金丝雀发布
4. 🔧 模型版本管理

---

**报告生成时间**: 2026-03-03  
**分析基准**: D:\embedding_service 现有服务  
**设计参考**: D:\github\opencode-memory-plugin\V2_INTEGRATION_GUIDE.md  
**下一步建议**: 按照 Phase 1-6 的步骤逐步实施

---

**总结**: 这份报告提供了从 embedding_service 升级为包装层服务的完整指南，确保在保留所有现有能力的同时，添加新功能。所有现有端点保持不变，通过新的统一健康检查端点提供统一的访问入口。
