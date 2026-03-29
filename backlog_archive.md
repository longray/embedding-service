# Backlog Archive

> 已完成任务归档，采用 Agent 手册规范格式

**归档时间**: 2026-03-30

---

## 2026-03-28 归档批次

### v2.4.1 - sync_preview conflict 检测修复

- [x] B-005 [P0] upload_memories 上传后 get_fingerprints 返回空 #bug #sync (完成于 2026-03-28)
  - [x] B-005-A [P0] SCHEMAFULL 字段未定义 (完成于 2026-03-28)
  - [x] B-005-B [P0] SurrealDB 3.0 SDK 结果解析逻辑错误 (完成于 2026-03-28)
  - [x] B-005-C [P0] get_conflict_detail 参数化表名语法错误 (完成于 2026-03-28)
- [x] B-006 [P1] SCHEMA_TARGET_VERSION 版本号未更新 #quality (完成于 2026-03-28)
- [x] B-007 [P1] FastAPI app 定义位置错误 #quality (完成于 2026-03-28)
- [x] B-008 [P1] 重复 API 端点定义 #quality (完成于 2026-03-28)
- [x] B-009 [P1] tree_sitter 导入类型错误 #quality (完成于 2026-03-28)

### v2.4.2 - API 稳定性修复

- [x] B-010 [P0] sync_preview 返回 500（to_delete 含 None）#bug #api (完成于 2026-03-28)
- [x] B-011 [P1] 项目文档三分类整理 #docs (完成于 2026-03-28)
- [x] B-012 [P1] Cache/HNSW 500 错误修复 #bug (完成于 2026-03-28)
- [x] B-014 [P1] LLM 服务并发请求导致 OOM 崩溃 #bug #llm (完成于 2026-03-28)
- [x] B-015 [P1] LLM 服务 Pydantic @validator 弃用警告 #quality #llm (完成于 2026-03-28)
- [x] B-016 [P1] LLM 服务版本号硬编码未更新 #quality #llm (完成于 2026-03-28)
- [x] B-017 [P2] wrapper 层 llm_service_url 配而不用 #cleanup (完成于 2026-03-28)
- [x] B-018 [P1] SurrealDB count(*) 语法不兼容 #bug (完成于 2026-03-28)
- [x] B-025 [P1] B608 record_id SQL 注入修复 #security (完成于 2026-03-28)
- [x] B-026 [P1] Bandit 安全扫描标记 # nosec #security (完成于 2026-03-28)

### v2.4.0 - API 行为优化

- [x] B-001 [P1] relationship_type 错误提示优化 #api (完成于 2026-03-28)
- [x] B-002 [P1] conflict resolution 大小写兼容 #api #sync (完成于 2026-03-28)
- [x] B-003 [P1] full_sync 返回 skipped 列表 #api #sync (完成于 2026-03-28)
- [x] B-004 [P1] sync_incremental → sync_preview 重命名 #api (完成于 2026-03-28)

### Markdown 质量门禁 (v2.4.3)

- [x] BL-MD-01 [P1] 建立 Markdown 检查基础配置 #docs #quality (完成于 2026-03-28)
- [x] BL-MD-02 [P1] 集成 Pre-commit 质量门禁 #docs #quality (完成于 2026-03-28)
- [x] BL-MD-03 [P1] 全局存量 Markdown 错误修复 #docs #quality (完成于 2026-03-28)
- [x] BL-MD-04 [P1] 文档体系同步更新 #docs (完成于 2026-03-28)

---

## 历史归档

> v2.4.0 之前的已完成任务已归档至 CHANGELOG.md

---

## 归档规范

**格式**: `- [x] {ID} [{Priority}] {描述} #{标签} (完成于 YYYY-MM-DD)`

**优先级**: P0 = 紧急, P1 = 重要, P2 = 普通, P3 = 低优先级

**状态**: ✅ 已完成, ⏳ 进行中, 📋 规划中, ⚪ 暂缓

---

*最后更新: 2026-03-30*
