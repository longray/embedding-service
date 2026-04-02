# 后端 API 问题报告

**收件人**: 后端开发团队  
**发件人**: OpenCode Memory Plugin 前端团队  
**日期**: 2026-04-02  
**主题**: 关于 `/api/v1/memories` 批量上传接口返回 422 错误的报告

---

## 问题概述

在集成测试中发现，后端 API 的批量上传接口返回 **422 Unprocessable Entity** 错误，导致 4 个关键集成测试失败。

## 错误详情

### 受影响端点

- `POST /api/v1/memories` (批量上传)

### 错误响应

```json
{
  "detail": [
    {
      "type": "string_type",
      "loc": ["body", "memories", 0, "abstract"],
      "msg": "Input should be a valid string",
      "input": null
    },
    {
      "type": "string_type",
      "loc": ["body", "memories", 0, "overview"],
      "msg": "Input should be a valid string",
      "input": null
    }
  ]
}
```

### 失败测试

| 测试名称 | 状态 | 说明 |
|----------|------|------|
| should upload memory to backend | ❌ 失败 | 单条上传测试 |
| should handle batch upload | ❌ 失败 | 批量上传测试 |
| should complete full workflow | ❌ 失败 | 端到端工作流 |
| should detect and handle duplicates | ❌ 失败 | 重复检测测试 |

## 请求分析

前端发送的请求体结构：

```json
{
  "memories": [
    {
      "id": "01J...",
      "type": "code",
      "abstract": null,      // ← 问题字段
      "overview": null,      // ← 问题字段
      "content": "...",
      "tags": ["code", "javascript"],
      "metadata": {...}
    }
  ]
}
```

## 问题分析

后端 API 要求 `abstract` 和 `overview` 字段必须是**非空字符串**，但前端在某些场景下会发送 `null` 值。

### 建议修复方案

**方案 1**: 后端接受 `null` 值并自动处理（推荐）

```python
# 在 Pydantic 模型中
abstract: Optional[str] = Field(default="", nullable=True)
overview: Optional[str] = Field(default="", nullable=True)
```

**方案 2**: 后端返回更详细的错误信息

```json
{
  "error": "VALIDATION_ERROR",
  "message": "abstract and overview cannot be null",
  "code": "NULL_FIELD_NOT_ALLOWED",
  "fields": ["abstract", "overview"]
}
```

**方案 3**: 前端在发送前设置默认值

前端可以在发送前将 `null` 转换为 `""`，但这会增加前端复杂度。

## 优先级

**P1 - 高优先级**

此问题阻塞了以下功能：
- 代码分析结果的自动上传
- 批量记忆同步
- 重复记忆检测

## 复现步骤

1. 启动后端服务 (`localhost:17999`)
2. 运行集成测试: `npm test -- tests/phase-a-integration.test.js`
3. 观察 422 错误

## 环境信息

- 后端版本: 2.5.0
- 插件版本: 2.9.0
- Node.js: v22.18.0
- 测试框架: Jest

## 附件

- [ ] 完整请求/响应日志
- [ ] 测试用例代码片段
- [ ] 相关代码分析结果

---

**期待回复**: 请确认修复方案和时间表。

如有疑问，请联系前端团队。

此致  
OpenCode Memory Plugin 团队
