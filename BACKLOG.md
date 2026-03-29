# Backlog

> 后端任务追踪文档，按优先级排序。已完成任务归档至 backlog_archive.md。

**更新时间**: 2026-03-30

---

## v2.4.2 - 进行中

- [ ] BL-1 [P2] Tenant ID 不匹配 #sync #low-priority
  - 描述: 用户配置的 tenant_id 是 `longray`，但插件默认使用 `default`。导致插件上传的记忆和用户实际数据不在同一个租户下。
  - 涉及范围: 插件侧配置
  - 前置依赖: 无
  - 完成标准: 插件使用用户配置的 tenant_id 调用所有 API，上传和查询使用同一 tenant_id
  - 验证方式: 插件上传一条记忆 → 后端用 `longray` tenant_id 查询 → 能找到；后端用 `default` tenant_id 查询 → 找不到

- [ ] BL-2 [P1] 性能基线建立 #performance
  - 描述: 建立性能基准测试脚本和数据，为后续优化提供对比依据。
  - 涉及范围: `scripts/benchmark.py`
  - 前置依赖: 无
  - 完成标准: 基线数据已记录，优化方向已识别
  - 验证方式: 运行 `uv run python scripts/benchmark.py`，获取当前环境性能数据

---

## v2.5.0 - 代码分析增强

> 基于 GitNexus 设计理念，增强 Memory Stack 的代码分析能力。核心定位：**记忆级代码分析**（非仓库级）。

- [ ] BL-3 [P1] 修复代码分析增强设计文档 #docs #design
  - 描述: 修复 `docs/code-analysis-enhancement.md` 中的格式问题、事实错误和内容缺失。
  - 涉及范围: `docs/code-analysis-enhancement.md` — 全面重写
  - 前置依赖: 无
  - 完成标准: 无格式问题、术语统一、事实性数据修正、补充缺失章节
  - 验证方式: 本地 Markdown 渲染检查，对照审核报告逐项确认修复

- [ ] BL-4 [P1] 代码分析结果持久化（Phase A）#code-analysis
  - 描述: 将 `analyze_memory_code()` 的返回结果持久化到记忆的 `metadata.code_analysis` 字段，上传代码记忆时自动触发分析。
  - 涉及范围: `wrapper/src/utils/memory_manager.py`, `wrapper/src/main.py`, `wrapper/src/utils/code_analyzer.py`, `wrapper/src/config.py`
  - 前置依赖: BL-3（设计文档修复后实施更准确）
  - 完成标准: 分析结果写入 `metadata.code_analysis`，上传时自动触发，分析失败不影响上传
  - 验证方式: `uv run pytest tests/test_code_analysis_persistence.py -v`, 32个同步测试通过

- [ ] BL-5 [P2] Meilisearch 代码分析字段索引（Phase B）#code-analysis #meilisearch
  - 描述: 将代码分析结果同步到 Meilisearch 索引，支持按代码属性过滤搜索。
  - 涉及范围: `wrapper/src/utils/memory_manager.py`, `wrapper/src/utils/meili_client.py`
  - 前置依赖: BL-4（需要先有代码分析数据）
  - 完成标准: Meilisearch 文档包含 `code_analysis` 字段，支持按语言/函数名/复杂度过滤
  - 验证方式: Meilisearch 字段验证，代码搜索测试

- [ ] BL-6 [P2] LLM 代码摘要生成（Phase C）#code-analysis #llm
  - 描述: 调用外部 LLM API 为代码记忆生成自然语言摘要，存入 `metadata.code_summary`。
  - 涉及范围: `wrapper/src/config.py`, `wrapper/src/utils/memory_manager.py`, `wrapper/src/main.py`
  - 前置依赖: BL-4（需要先有代码分析结果作为 LLM 输入）
  - 完成标准: 上传代码后异步触发 LLM 摘要，LLM 调用失败不影响上传
  - 验证方式: 手动触发 LLM 摘要，获取摘要，单元测试（mock LLM 调用）

---

## 暂缓任务

- [ ] BL-7 [P3] 跨文件关系解析（Phase D）#code-analysis #future
  - 描述: 解析代码记忆间的 import/call 关系，存入 SurrealDB `relation` 表，带置信度评分。
  - 状态: ⚪ 远期考虑，可行性待评估
  - 限制: 记忆级代码分析的上传顺序不确定，需要延迟解析策略

- [ ] BL-8 [P3] 插件端代码分析工具（Phase E）#code-analysis #plugin
  - 描述: 在 OpenCode 插件中注册代码分析相关工具，与后端新增 API 对齐。
  - 涉及范围: 插件端（TypeScript）
  - 前置依赖: BL-4 ~ BL-6（后端 API 就绪后才能注册工具）
  - 状态: ⏳ 待后端 API 就绪

---

## Backlog 规范

**格式**: `- [ ] BL-{N} [{Priority}] 描述 #标签`

**优先级**: P0 = 紧急, P1 = 重要, P2 = 普通, P3 = 低优先级

**状态**: ⏳ 进行中, 📋 待开始, ⚪ 暂缓

**已完成任务**: 见 [backlog_archive.md](backlog_archive.md)

---

**历史归档**: v2.4.0 之前的已完成任务已归档至 CHANGELOG.md

---

*最后更新: 2026-03-30*
