# Meilisearch 代码搜索方案

## 环境要求

- Windows 10/11
- Docker Desktop
- Python 3.8+

## 快速开始

### 1. 启动服务
```bash
docker-compose up -d
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 初始化索引
```bash
python init_index.py
```

### 4. 运行测试
```bash
python test_search.py
```

### 5. 监控索引
```bash
python monitor_index.py
```

### 6. 优化索引
```bash
python optimize_index.py
```

## 核心特性

| 特性 | 配置 | 说明 |
|------|------|------|
| 端口 | 18003 | 自定义端口 |
| 双字段策略 | ✅ | 精确字段 + 搜索字段 |
| nonSeparatorTokens | 10 个字符 | 保留特殊格式 |
| dictionary | 80+ 词条 | 代码术语词典 |
| localizedAttributes | *_zh | 中文分词 |
| typoTolerance | 禁用精确字段 | 代码字段不容错 |
| 版本锁定 | v1.12.0 | 兼容性保证 |

## 副作用规避

| 副作用 | 规避方案 |
|--------|---------|
| 索引体积增长 | 双字段策略 |
| 部分匹配异常 | dictionary + 过滤器 |
| 中文分词不一致 | *_zh 统一命名 |
| 性能下降 | 监控 + 定期优化 |
| 版本兼容 | Docker 版本锁定 |

## 访问地址

- API: http://localhost:18003
- Master Key: meili_master_key_2026_safe

## 文档模板

```python
document = {
    "id": 1,
    "title_zh": "中文标题",
    "description_zh": "中文描述",
    "content_zh": "中文内容",
    "tags_zh": ["标签1", "标签2"],
    "file_path": "完整路径",
    "version": "v1.0.0",
    "language": "python",
    "project_name": "项目名",
    "email": "user@example.com",
    "ip_address": "192.168.1.1",
    "status": "active",
    "created_at": 1710230400,
    "updated_at": 1710230400,
    "file_size": 10240,
    "line_count": 300,
    "file_name_search": "文件名 语言",
    "class_name_search": "类名",
    "method_name_search": "方法1 方法2",
    "namespace_search": "命名 空间",
    "code_content_search": "代码内容",
}
```
