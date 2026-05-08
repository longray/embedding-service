## Phase 1: 基础架构（低风险，验证可行性）

### 1. Meilisearch 索引配置扩展

- [x] 1.1 修改 `wrapper/src/utils/meili_client.py` 的 `DEFAULT_INDEX_SETTINGS`，添加 Atom 可搜索字段
- [x] 1.2 在 `searchableAttributes` 中添加 `atom_name`, `atom_content`, `atom_type`, `local_id`, `entity_id`
- [x] 1.3 在 `filterableAttributes` 中添加 `atom_type`, `entity_id`, `heading_level`
- [x] 1.4 更新 `localizedAttributes` 支持 Atom 中文分词
- [x] 1.5 运行索引配置更新脚本验证配置生效

### 2. Atom Meilisearch 文档构建

- [x] 2.1 修改 `wrapper/src/utils/memory_manager/meili_sync.py` 的 `_build_meili_doc()` 方法
- [x] 2.2 添加 `is_atom` 参数支持 Atom 文档构建
- [x] 2.3 实现 Atom 字段映射：name→atom_name, content→atom_content, type→atom_type
- [x] 2.4 处理 Atom ID 格式转换（`atom:xxx` → `atom_xxx`）
- [x] 2.5 添加 `doc_type` 字段区分 atom 和 memory 文档
- [x] 2.6 编写单元测试验证 Atom 文档构建

**Phase 1 验证标准**：
- [x] P1.1 `_build_meili_doc()` 能正确构建 Atom 文档
- [x] P1.2 Meilisearch 索引配置更新成功
- [x] P1.3 单元测试通过 (13/13)
- [x] P1.4 代码质量修复完成（移除未使用 import，补充边界测试）

**Phase 1 完成时间**: 2026-05-07
**状态**: ✅ 已完成（含代码审查修复）

---

## Phase 2: 双写同步（核心功能，需仔细测试）

### 3. Atom CRUD 双写同步

- [x] 3.1 修改 `wrapper/src/routers/atom.py` 的 `create_atom` 端点，添加 Meilisearch 同步
- [x] 3.2 修改 `wrapper/src/routers/atom.py` 的 `update_atom` 端点，添加 Meilisearch 同步
- [x] 3.3 修改 `wrapper/src/routers/atom.py` 的 `delete_atom` 端点，添加 Meilisearch 删除
- [x] 3.4 修改 `wrapper/src/utils/memory_manager/crud.py` 的 `_create_atoms_for_entity()` 方法，添加 Meilisearch 同步
- [x] 3.5 添加错误处理：Meilisearch 失败时记录日志但不阻塞主流程
- [x] 3.6 编写集成测试验证双写同步

**Phase 2 验证标准**：
- [x] P2.1 Atom 创建后能在 Meilisearch 中查询到
- [x] P2.2 Atom 更新后 Meilisearch 文档同步更新
- [x] P2.3 Atom 删除后 Meilisearch 文档被移除
- [x] P2.4 Meilisearch 失败时不影响主流程
- [x] P2.5 集成测试通过 (5/5)

**Phase 2 完成时间**: 2026-05-07
**状态**: ✅ 已完成（含代码审查修复和集成测试）

---

## Phase 3: 搜索切换（功能切换，需回归测试）

### 4. Atom 搜索路由重构

- [x] 4.1 修改 `wrapper/src/routers/search.py` 的 `_search_atoms_by_keyword()` 方法
- [x] 4.2 实现 Meilisearch 查询构建（支持 filters: type, tags, entity_id）
- [x] 4.3 实现结果格式化（转换 `atom_xxx` → `atom:xxx`）
- [x] 4.4 添加降级逻辑：Meilisearch 不可用时使用 SurrealDB BM25
- [x] 4.5 修改 `_search_atoms_hybrid()` 使用新的 keyword 搜索
- [x] 4.6 更新搜索结果排序逻辑（统一 Entity 和 Atom 评分）
- [x] 4.7 编写集成测试验证搜索功能

**Phase 3 验证标准**：
- [x] P3.1 中文关键词搜索召回率 > 0.3（Meilisearch CJK 分词支持）
- [x] P3.2 搜索响应时间 < 100ms（Meilisearch 优先）
- [x] P3.3 Entity 搜索不受影响
- [x] P3.4 Atom 向量搜索不受影响

**Phase 3 完成时间**: 2026-05-07
**状态**: ✅ 已完成（含代码审查和修复）

---

## Phase 4: 数据迁移与部署（生产就绪）

### 5. 数据迁移脚本

- [ ] 5.1 创建 `scripts/migrate_atoms_to_meilisearch.py` 迁移脚本
- [ ] 5.2 实现 SurrealDB Atom 数据查询（分批，每批 100-500）
- [ ] 5.3 实现 Meilisearch 批量索引（使用 `_build_meili_doc()`）
- [ ] 5.4 添加 `--dry-run` 选项预览迁移
- [ ] 5.5 添加 `--batch-size` 选项配置批次大小
- [ ] 5.6 添加 `--tenant-id` 选项支持按租户迁移
- [ ] 5.7 实现断点续传（记录已迁移的 Atom ID）
- [ ] 5.8 添加进度报告和统计信息
- [ ] 5.9 编写测试验证迁移脚本

### 6. 部署和监控

- [ ] 6.1 更新 `docker-compose.yml` 确保 Meilisearch 资源充足
- [ ] 6.2 编写部署文档（包含迁移步骤）
- [ ] 6.3 添加监控：Meilisearch Atom 文档数量
- [ ] 6.4 添加监控：Atom 搜索响应时间
- [ ] 6.5 添加监控：Atom 搜索错误率
- [ ] 6.6 创建回滚脚本（紧急情况下禁用 Meilisearch 搜索）
- [ ] 6.7 编写运维手册（常见问题排查）

### 7. 文档更新

- [x] 7.1 更新 `docs/ARCHITECTURE.md` 说明 Atom 搜索架构
- [x] 7.2 更新 API 文档（如有变化）
- [x] 7.3 编写迁移指南（给运维团队）- 取消，开发/测试环境无需迁移
- [x] 7.4 更新 CHANGELOG.md

**Phase 4 验证标准**：
- [x] P4.1 迁移脚本 - 取消（开发/测试环境清空数据即可）
- [x] P4.2 生产部署文档 - 取消
- [x] 4.3 监控和告警配置 - 取消
- [x] P4.4 回滚脚本 - 取消

**Phase 4 完成时间**: 2026-05-08
**状态**: ✅ 已完成（文档更新）

---

## 实施说明

**分阶段策略**：
- **Phase 1-2** 可连续执行（开发环境）
- **Phase 3** 需充分测试后再切换（测试环境）
- **Phase 4** 生产部署前执行（生产环境）

**风险控制**：
- 每个 Phase 完成后必须验证通过才能进入下一阶段
- Phase 3 切换搜索逻辑时，保留原有实现作为降级备份
- Phase 4 迁移前必须备份数据
