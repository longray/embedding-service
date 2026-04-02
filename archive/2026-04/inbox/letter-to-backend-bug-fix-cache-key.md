# 后端 Bug 报告：MemoryManager 缺少 _get_vector_cache_key 方法

**发件人**: OpenCode Memory Plugin  
**收件人**: Backend Team  
**日期**: 2026-03-31

---

## 问题描述

在调用 `POST /api/v1/memories` 上传记忆时，后端返回错误：

```
AttributeError: 'MemoryManager' object has no attribute '_get_vector_cache_key'
```

## 复现步骤

1. 调用 `POST /api/v1/memories` API
2. 发送包含 content, type, tags, project_id, tenant_id 的记忆数据
3. 后端返回 200 但处理失败，错误信息如上

## 影响范围

- 所有记忆上传请求都会失败
- 搜索功能无法返回新上传的记忆
- 集成测试 Checkpoint 4 失败

## 请求

请修复 `MemoryManager` 类，添加 `_get_vector_cache_key` 方法或修复相关逻辑。

---

**附件**: 错误日志

```
Upload result: {
  "total": 1,
  "success": 0,
  "failed": 1,
  "updated": 0,
  "memory_ids": [],
  "errors": [
    "AttributeError: 'MemoryManager' object has no attribute '_get_vector_cache_key'"
  ]
}
```