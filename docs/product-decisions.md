# 产品决策备忘录 (v2.6.0 规划)

**日期**: 2026-04-02
**范围**: 本次全面审计发现的产品层面决策

---

## PD-1: abstract/overview 必填字段不放宽

**背景**: 插件端集成测试发送 `abstract: null` / `overview: null`，收到 422 错误，请求后端放宽约束。

**决策**: **不改后端，要求前端遵照接口契约。**

**理由**:

1. L0/L1/L2 分层内容模型是核心设计，abstract 和 overview 分别是 L0、L1 层展示内容，渐进加载依赖它们有实际值。null 会破坏搜索结果展示。
2. 模型中已有允许 null 的字段（`source_id`、`local_id`、`source_timestamp`），说明 `str` vs `str | None` 是有意区分。
3. 接口自 v2.0 起稳定，不应为单一调用方降低约束。
4. 前端修复成本极低（发送前兜底即可）。

**已行动**: 已回复 `inbox/REPLY_BACKEND_API_ISSUE.md`。

**影响范围**: 插件端需修复 4 个集成测试。

---

## PD-2: v2.6.0 版本范围确认

**背景**: 全面审计发现代码库存在技术债（上帝文件、类型错误、配置过时），需要规划清理版本。

**决策**: v2.6.0 定位为 **"质量治理版本"**，不新增功能。

**范围**:

| 类别 | 内容 | 用户感知 |
|------|------|----------|
| 代码结构 | memory_manager.py 拆分、main.py 路由拆分 | 无（纯重构） |
| 代码质量 | pyproject.toml 修复、meilisearch_code/ 类型修复 | 无 |
| 测试补充 | 工具模块单元测试 | 无 |
| 安全加固 | meili_client.py 默认 API Key 处理 | 无 |
| 文档治理 | 更新 ROADMAP、CHANGELOG | README 更准确 |

**对外承诺**: v2.6.0 所有 API 行为不变，纯内部质量提升。

**版本策略**: 遵循 ROADMAP.md 中的版本策略（minor 版本允许向后兼容的改进）。

---

## PD-3: meilisearch_code/ 模块定位

**背景**: `meilisearch_code/` 目录有 9 个 Pyright 错误，但不在 ruff 检查范围内。

**决策**: 该模块定位为**辅助工具**（代码搜索索引管理），非核心业务代码。修复类型错误但不需要投入额外测试资源。

**理由**: 该模块仅在运维场景手动运行（`init_index.py`、`monitor_index.py`、`optimize_index.py`），不影响 API 服务运行。

---

*本文档归档位置: `docs/product-decisions.md`*
