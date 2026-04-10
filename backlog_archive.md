# Backlog Archive

> 已完成任务归档，采用 Agent 手册规范格式

**归档时间**: 2026-04-07

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

## 2026-04-10 归档批次

### 场景 9: 代码分析增强

- [x] BL-CA-34 [P1] 实现 Memory Lookup API #feature (完成于 2026-04-09)
  - 支持 source_id、file_path、hash 三种查询方式
  - 用于缓存重建和多设备同步
  - 添加 SurrealDB 索引优化查询性能
- [x] BL-CA-35 [P1] SurrealDB SessionExpired 自动重连 #bug #reliability (完成于 2026-04-10)
  - 添加 reconnect() 方法到 SurrealDBManager
  - 传入 reauthenticate_fn 到 MemoryManager
  - 自动检测会话过期并重试查询

### 场景 10: 代码分析数据上传修复

- [x] BL-CA-FIX-01 [P0] 代码数据被 hash 去重跳过 #bug (完成于 2026-04-08)
- [x] BL-CA-FIX-02 [P0] 新代码文件未写入 #bug (完成于 2026-04-08)
- [x] BL-CA-FIX-03 [P0] 相同 file_path 不更新 #bug (完成于 2026-04-08)
- [x] BL-CA-FIX-04 [P1] 字段名不一致 #quality (完成于 2026-04-08)
- [x] BL-CA-FIX-05 [P1] Schema 未初始化 #bug (完成于 2026-04-08)
- [x] BL-CA-FIX-06 [P1] 验证脚本错误 #bug (完成于 2026-04-08)
- [x] BL-CA-FIX-07 [P1] 项目地图 module_dependencies 为空 #bug (完成于 2026-04-08)
- [x] BL-CA-FIX-08 [P1] 模块依赖重复 #bug (完成于 2026-04-08)
- [x] BL-CA-FIX-09 [P1] 项目地图查询返回空 #bug (完成于 2026-04-08)
- [x] BL-CA-FIX-API-01 [P1] 字段缺失导致查询 404 #bug (完成于 2026-04-08)
- [x] BL-CA-FIX-API-02 [P1] 可选字段校验过严 #bug (完成于 2026-04-08)

### 优化项

- [x] BL-CA-OPT-01 [P1] RELATE SQL 注入防护 #security (完成于 2026-04-08)
- [x] BL-CA-OPT-02 [P0] RecordID 格式统一 #quality (完成于 2026-04-08)
- [x] BL-CA-OPT-03 [P2] 嵌套字段查询优化 #performance (完成于 2026-04-08)
- [x] BL-CA-OPT-04 [P1] 批量插入分批处理 #performance (完成于 2026-04-08)
- [x] BL-CA-OPT-05 [P2] SQL 查询规范文档 #docs (完成于 2026-04-08)
- [x] BL-CA-OPT-06 [P1] Meilisearch 同步分批 #performance (完成于 2026-04-08)
- [x] BL-CA-OPT-08 [P1] embedding 字段优化 #performance (完成于 2026-04-08)

---

## 历史归档

> v2.4.0 之前的已完成任务已归档至 CHANGELOG.md

---

## 2026-04-07 归档批次

### 场景 1: 记忆上传与搜索

- [x] BL-18 [P0] 修复测试用例适配 abstract/overview 必填 #quality (完成于 2026-04-04)
- [x] BL-19 [P0] 修复双写测试验证上传→Meilisearch 流程 #quality (完成于 2026-04-04)
- [x] BL-20 [P0] 端到端验证上传→搜索全链路（87/87）#e2e (完成于 2026-04-04)
- [x] BL-24 [P1] 修复 get_memory_summary 连接泄露 #bug (完成于 2026-04-04)
- [x] BL-25 [P2] 清理调试日志 #cleanup (完成于 2026-04-04)
- [x] BL-26 [P1] 实现智能去重决策（替代硬编码 KEEP_BOTH）#feature (完成于 2026-04-04)
- [x] BL-27 [P1] 实现 _update_memory（SurrealDB UPDATE + Meilisearch 同步）#feature (完成于 2026-04-04)

### 场景 2: 代码分析

- [x] BL-4 [P0] 代码分析结果持久化（Phase A）#feature (完成于 2026-04-03)
- [x] BL-6 [P0] LLM 代码摘要生成（Phase C）#feature (完成于 2026-04-03)
- [x] BL-28 [P0] analyze_memory_code 实现（CodeAnalyzer 集成）#feature (完成于 2026-04-04)
- [x] BL-CA-05 [P1] code_filter 添加 max_complexity 支持 #feature (完成于 2026-04-03)
- [x] BL-CA-06 [P1] 修复 v1.2 设计文档 4 个小问题 #docs (完成于 2026-04-03)
- [x] BL-CA-07 [P1] 代码文件指纹同步 API #feature (完成于 2026-04-03)
- [x] BL-CA-08 [P1] 代码文件 Upsert #feature (完成于 2026-04-03)
- [x] BL-CA-09 [P1] 代码分析集成测试补充 #quality (完成于 2026-04-03)
- [x] BL-CA-10 [P1] 代码分析 API 文档更新 #docs (完成于 2026-04-03)

### 场景 3: 多设备同步

- [x] BL-29 [P0] get_fingerprints 查询 SurrealDB 指纹 #sync (完成于 2026-04-04)
- [x] BL-30 [P0] sync_preview 三分类比对 + conflict 表写入 #sync (完成于 2026-04-04)
- [x] BL-31 [P0] sync_full 透传 upload_memories #sync (完成于 2026-04-04)
- [x] BL-32 [P0] resolve_conflict 三种策略 + 辅助方法 #sync (完成于 2026-04-04)

### 场景 4: 测试架构优化

- [x] BL-T1 [P0] 定义测试分层标记 (pytest.mark) #quality (完成于 2026-04-04)
- [x] BL-T2 [P1] 优化 conftest fixture scope #quality (完成于 2026-04-04)
- [x] BL-T3 [P0] 修复 Mixin 模式导致的 mock 断言失败 #bug (完成于 2026-04-04)
- [x] BL-T4 [P1] 修复接口变更导致的测试失败 #bug (完成于 2026-04-04)
- [x] BL-T5 [P2] 清理无效测试文件 #cleanup (完成于 2026-04-04)
- [x] BL-T6 [P0] pre-commit 配置调整 #quality (完成于 2026-04-04)
- [x] BL-T7 [P2] 合并小型测试文件 #cleanup (完成于 2026-04-04)
- [x] BL-T8 [P0] 修复 conftest 配置错误和 session scope 回归 #bug (完成于 2026-04-04)
- [x] BL-T9 [P1] LLM 服务和 SDK 变更测试条件跳过 #quality (完成于 2026-04-04)
- [x] BL-T11 [P1] 修复 wrapper 接口变更测试 #bug (完成于 2026-04-04)

### 场景 7: 系统可观测性与开发者工具

- [x] BL-OBS-00 [P0] 修复 docker-compose.yml 端口映射 #bug (完成于 2026-04-04)
- [x] BL-OBS-01 [P1] 实现 HNSW 统计端点 #feature (完成于 2026-04-04)
- [x] BL-OBS-02 [P1] 实现缓存统计端点 #feature (完成于 2026-04-04)
- [x] BL-OBS-03 [P2] 创建 CLI 诊断工具 #feature (完成于 2026-04-04)
- [x] BL-OBS-04 [P2] 创建 CONTRIBUTING.md #docs (完成于 2026-04-04)

---

## 归档规范

**格式**: `- [x] {ID} [{Priority}] {描述} #{标签} (完成于 YYYY-MM-DD)`

**优先级**: P0 = 紧急, P1 = 重要, P2 = 普通, P3 = 低优先级

**状态**: ✅ 已完成, ⏳ 进行中, 📋 规划中, ⚪ 暂缓

---

*最后更新: 2026-04-10*
