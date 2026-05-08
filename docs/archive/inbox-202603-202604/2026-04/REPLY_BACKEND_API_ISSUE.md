# 回复：后端 API 问题报告

**收件人**: OpenCode Memory Plugin 前端团队
**发件人**: 后端开发团队
**日期**: 2026-04-02
**回复**: 关于 `/api/v1/memories` 批量上传接口返回 422 错误

---

## 结论

后端接口定义没有问题，**422 是预期行为**。请前端按照接口契约补全必填字段。

## 接口定义说明

`POST /api/v1/memories` 的 `MemoryItem` 模型定义如下（`wrapper/src/main.py:50-63`）：

```python
class MemoryItem(BaseModel):
    content: str = Field(..., min_length=1, description="记忆内容 (L2)")
    abstract: str = Field(..., min_length=1, max_length=100, description="摘要 (L0, 建议≤100字符)")
    overview: str = Field(..., min_length=1, max_length=500, description="概览 (L1, 建议≤500字符)")
    type: str = Field(default="general", description="记忆类型")
    tags: list[str] = Field(default_factory=list, description="标签列表")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元数据")
    # ... 其余可选字段均有默认值
```

**字段分类**：

| 字段 | 必填 | 类型 | 说明 |
|------|------|------|------|
| `content` | **是** | `str` (min_length=1) | L2 完整内容 |
| `abstract` | **是** | `str` (min_length=1, max_length=100) | L0 摘要 |
| `overview` | **是** | `str` (min_length=1, max_length=500) | L1 概览 |
| `type` | 否 | `str` (默认 `"general"`) | 记忆类型 |
| `tags` | 否 | `list[str]` (默认 `[]`) | 标签 |
| `metadata` | 否 | `dict` (默认 `{}`) | 元数据 |
| `source_id` | 否 | `str \| None` (默认 `null`) | ✅ 这个才允许 null |
| `local_id` | 否 | `str \| None` (默认 `null`) | ✅ 这个才允许 null |

关键区分：`Field(...)` 表示**必填**，`Field(default=None)` 或 `str | None = Field(default=None)` 才允许 null。

## 问题根因

前端发送的请求体：

```json
{
  "abstract": null,   // ❌ 违反接口契约
  "overview": null    // ❌ 违反接口契约
}
```

接口要求 `abstract` 和 `overview` 是**非空字符串**，传 `null` 违反了类型约束，Pydantic 正确地返回了 422。

## 不修改后端的原因

1. **分层内容模型设计**：项目的 L0/L1/L2 分层模型要求每个层级都有实际内容，`abstract` 和 `overview` 是记忆检索的核心字段，用于渐进加载——搜索结果先展示 abstract（L0），用户需要更多细节时加载 overview（L1），再进一步加载 content（L2）。如果允许 null，搜索结果展示会出问题。
2. **向后兼容**：该接口自 v2.0 起定义至今，已有多个调用方依赖此契约，不应为了方便一个调用方而降低约束。
3. **对比其他字段**：模型中确实有允许 null 的字段（如 `source_id`、`local_id`），说明设计是有意为之的。

## 前端修复建议

在发送请求前，确保 `abstract` 和 `overview` 不为 null：

```typescript
// 方案 A：生成时补全（推荐）
// 在创建记忆对象时就生成 abstract 和 overview
const memory = {
  content: "...",
  abstract: content.slice(0, 100),  // 截取前100字符作为摘要
  overview: content.slice(0, 500),  // 截取前500字符作为概览
  // ...
};

// 方案 B：发送前兜底
memories.forEach(m => {
  m.abstract = m.abstract ?? m.content.slice(0, 100);
  m.overview = m.overview ?? m.content.slice(0, 500);
});
```

请前端团队按照上述方案修复后重新运行集成测试。

---

此致
后端开发团队
