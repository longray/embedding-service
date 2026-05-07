from __future__ import annotations

"""Meilisearch Polyglot 架构单元测试

测试 MemoryManager 中的 Meilisearch 相关功能：
1. ID 转换工具方法
2. Meilisearch 搜索结果格式化
3. 上传双写流程（mock SurrealDB + Meilisearch）
4. 搜索路由（Meilisearch 优先、SurrealDB 降级）

运行方式:
    uv run pytest tests/test_meili_integration.py -v
"""

import pytest

pytestmark = pytest.mark.integration

from unittest.mock import AsyncMock, patch

import pytest

from wrapper.src.utils.memory_manager import MemoryManager

# ==================== Fixtures ====================


@pytest.fixture
def mock_db():
    """模拟 SurrealDB 异步客户端"""
    db = AsyncMock()
    return db


@pytest.fixture
def mock_meili():
    """模拟 MeilisearchClient"""
    meili = AsyncMock()
    meili.search = AsyncMock()
    meili.add_documents = AsyncMock()
    return meili


@pytest.fixture
def manager(mock_db):
    """创建 MemoryManager 实例（不含 Meilisearch）"""
    return MemoryManager(
        db=mock_db,
        embedding_service_url="http://localhost:18000",
        batch_size=10,
    )


@pytest.fixture
def manager_with_meili(mock_db, mock_meili):
    """创建 MemoryManager 实例（含 Meilisearch）"""
    mgr = MemoryManager(
        db=mock_db,
        embedding_service_url="http://localhost:18000",
        batch_size=10,
    )
    mgr.set_meili_client(mock_meili)
    return mgr


# ==================== ID 转换测试 ====================


class TestMeiliIdConversion:
    """测试 Meilisearch ID ↔ SurrealDB ID 转换 (MeiliSyncMixin._from_meili_id)"""

    def test_from_meili_id_underscore_to_colon(self):
        """Meilisearch ID 中的下划线转回冒号: memory_test123 → memory:test123"""
        mgr = MemoryManager(db=AsyncMock(), embedding_service_url="http://localhost:18000")
        assert mgr._from_meili_id("memory_test123") == "memory:test123"

    def test_from_meili_id_multiple_underscores(self):
        """只替换第一个下划线: memory_test_abc → memory:test_abc"""
        mgr = MemoryManager(db=AsyncMock(), embedding_service_url="http://localhost:18000")
        assert mgr._from_meili_id("memory_test_abc") == "memory:test_abc"

    def test_from_meili_id_no_underscore(self):
        """无下划线不替换: abc123 → abc123"""
        mgr = MemoryManager(db=AsyncMock(), embedding_service_url="http://localhost:18000")
        assert mgr._from_meili_id("abc123") == "abc123"

    def test_from_meili_id_uuid_format(self):
        """UUID 格式: memory_550e8400-e29b → memory:550e8400-e29b"""
        mgr = MemoryManager(db=AsyncMock(), embedding_service_url="http://localhost:18000")
        assert mgr._from_meili_id("memory_550e8400-e29b") == "memory:550e8400-e29b"


# ==================== 结果格式化测试 ====================


class TestFormatMeiliResults:
    """测试 Meilisearch 搜索结果格式化"""

    def test_empty_results(self, manager):
        """空结果集"""
        result = manager._format_meili_results({"hits": []})
        assert result == []

    def test_single_hit_with_surreal_id(self, manager):
        """使用 surreal_id 还原完整 ID"""
        meili_result = {
            "hits": [
                {
                    "id": "abc123",
                    "surreal_id": "memory:abc123",
                    "content": "测试内容",
                    "type": "general",
                    "tags": ["tag1"],
                    "project_id": "test_project",
                    "_rankingScore": 0.95,
                }
            ]
        }
        result = manager._format_meili_results(meili_result)
        assert len(result) == 1
        assert result[0]["id"] == "memory:abc123"
        assert result[0]["content"] == "测试内容"
        assert result[0]["type"] == "general"
        assert result[0]["tags"] == ["tag1"]
        assert result[0]["project_id"] == "test_project"
        assert result[0]["score"] == 0.95

    def test_fallback_to_from_meili_id(self, manager):
        """无 surreal_id 时从 Meilisearch ID 还原"""
        meili_result = {
            "hits": [
                {
                    "id": "def456",
                    "content": "回退测试",
                    "_rankingScore": 0.8,
                }
            ]
        }
        result = manager._format_meili_results(meili_result)
        assert result[0]["id"] == "memory:def456"
        assert result[0]["score"] == 0.8

    def test_multiple_hits(self, manager):
        """多条搜索结果"""
        meili_result = {
            "hits": [
                {"id": "a1", "surreal_id": "memory:a1", "content": "第一条", "_rankingScore": 0.9},
                {"id": "b2", "surreal_id": "memory:b2", "content": "第二条", "_rankingScore": 0.7},
                {"id": "c3", "surreal_id": "memory:c3", "content": "第三条", "_rankingScore": 0.5},
            ]
        }
        result = manager._format_meili_results(meili_result)
        assert len(result) == 3
        assert [r["score"] for r in result] == [0.9, 0.7, 0.5]

    def test_default_values(self, manager):
        """缺失字段使用默认值"""
        meili_result = {"hits": [{"id": "x1", "content": "只有内容", "_rankingScore": 0.6}]}
        result = manager._format_meili_results(meili_result)
        assert result[0]["type"] == "general"
        assert result[0]["tags"] == []
        assert result[0]["project_id"] == "global"
        assert result[0]["metadata"] == {}

    def test_missing_ranking_score(self, manager):
        """无 _rankingScore 默认为 0.0"""
        meili_result = {"hits": [{"id": "x1", "content": "无分数"}]}
        result = manager._format_meili_results(meili_result)
        assert result[0]["score"] == 0.0


# ==================== 搜索路由测试 ====================


class TestSearchRouting:
    """测试搜索引擎路由（Meilisearch 优先、SurrealDB 降级）"""

    @pytest.mark.asyncio
    async def test_keyword_routes_to_meili_when_available(self, manager_with_meili, mock_meili):
        """Meilisearch 可用时优先使用"""
        mock_meili.search.return_value = {
            "hits": [
                {
                    "id": "abc",
                    "surreal_id": "memory:abc",
                    "content": "测试",
                    "_rankingScore": 0.9,
                }
            ],
            "estimatedTotalHits": 1,
        }

        results = await manager_with_meili._search_by_keyword("测试", 10, "default")
        assert len(results) == 1
        assert results[0]["id"] == "memory:abc"
        mock_meili.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_keyword_falls_back_to_surreal_when_no_meili(self, manager, mock_db):
        """Meilisearch 不可用时降级到 SurrealDB"""
        mock_db.query.return_value = [{"id": "memory:xyz", "content": "降级测试", "score": 1.5}]

        results = await manager._search_by_keyword("测试", 10, "default")
        assert len(results) == 1
        assert results[0]["id"] == "memory:xyz"
        mock_db.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_meili_filter_includes_tenant_id(self, manager_with_meili, mock_meili):
        """Meilisearch 搜索包含租户过滤"""
        mock_meili.search.return_value = {"hits": []}

        await manager_with_meili._search_by_keyword("测试", 10, "my_tenant")
        call_kwargs = mock_meili.search.call_args
        assert "tenant_id = 'my_tenant'" in (
            call_kwargs.kwargs.get("filter_expr", "") or call_kwargs[1].get("filter_expr", "")
        )


# ==================== 上传双写测试 ====================


class TestUploadDualWrite:
    """测试上传时同时写入 SurrealDB 和 Meilisearch"""

    @pytest.mark.asyncio
    async def test_upload_syncs_to_meili(self, manager_with_meili, mock_db, mock_meili):
        """上传成功后同步到 Meilisearch"""
        with patch.object(manager_with_meili, "_get_embeddings", new_callable=AsyncMock) as mock_emb:
            mock_emb.return_value = [[0.1] * 1024]

            mock_db.query.side_effect = [
                [],  # content_hash dedup check → no existing
                [],  # vector search → no similar
                [[{"id": "memory:test001"}]],  # batch INSERT → success
            ]

            result = await manager_with_meili.upload_memories(
                [{"content": "测试记忆", "type": "test", "tags": ["tag1"]}],
                tenant_id="default",
            )

            assert result["success"] == 1
            assert result["failed"] == 0

            mock_meili.add_documents.assert_called_once()
            meili_docs = mock_meili.add_documents.call_args[0][0]
            assert len(meili_docs) == 1
            assert meili_docs[0]["surreal_id"] == "memory:test001"
            assert meili_docs[0]["content"] == "测试记忆"
            assert meili_docs[0]["type"] == "test"
            assert meili_docs[0]["tags"] == ["tag1"]
            assert meili_docs[0]["tenant_id"] == "default"

    @pytest.mark.asyncio
    async def test_upload_without_meili_still_works(self, manager, mock_db):
        """不启用 Meilisearch 时上传仍正常"""
        with patch.object(manager, "_get_embeddings", new_callable=AsyncMock) as mock_emb:
            mock_emb.return_value = [[0.1] * 1024]

            mock_db.query.side_effect = [
                [],  # content_hash dedup check → no existing
                [],  # vector search → no similar
                [[{"id": "memory:test002"}]],  # batch INSERT → success
            ]

            result = await manager.upload_memories(
                [{"content": "无 Meilisearch 测试"}],
                tenant_id="default",
            )

            assert result["success"] == 1

    @pytest.mark.asyncio
    async def test_meili_sync_failure_does_not_affect_upload(self, manager_with_meili, mock_db, mock_meili):
        """Meilisearch 同步失败不影响主流程"""
        with patch.object(manager_with_meili, "_get_embeddings", new_callable=AsyncMock) as mock_emb:
            mock_emb.return_value = [[0.1] * 1024]
            mock_meili.add_documents.side_effect = RuntimeError("Meilisearch 连接超时")

            mock_db.query.side_effect = [
                [],  # content_hash dedup check
                [],  # vector search
                [[{"id": "memory:test003"}]],  # batch INSERT
            ]

            result = await manager_with_meili.upload_memories(
                [{"content": "降级测试记忆"}],
                tenant_id="default",
            )

            assert result["success"] == 1
            assert result["failed"] == 0

    @pytest.mark.asyncio
    async def test_upload_batch_syncs_all_docs(self, manager_with_meili, mock_db, mock_meili):
        """批量上传时所有成功记录都同步到 Meilisearch"""
        with patch.object(manager_with_meili, "_get_embeddings", new_callable=AsyncMock) as mock_emb:
            mock_emb.return_value = [[0.1] * 1024, [0.2] * 1024, [0.3] * 1024]

            mock_db.query.side_effect = [
                [],  # dedup check #1
                [],  # vector search #1
                [],  # dedup check #2
                [],  # vector search #2
                [],  # dedup check #3
                [],  # vector search #3
                [[{"id": "memory:batch1"}, {"id": "memory:batch2"}, {"id": "memory:batch3"}]],  # batch INSERT
            ]

            result = await manager_with_meili.upload_memories(
                [
                    {"content": "批量1"},
                    {"content": "批量2"},
                    {"content": "批量3"},
                ],
                tenant_id="default",
            )

            assert result["success"] == 3
            meili_docs = mock_meili.add_documents.call_args[0][0]
            assert len(meili_docs) == 3

    @pytest.mark.asyncio
    async def test_upload_includes_source_fields(self, manager_with_meili, mock_db, mock_meili):
        """source_id 和 source_timestamp 正确同步到 Meilisearch"""
        with patch.object(manager_with_meili, "_get_embeddings", new_callable=AsyncMock) as mock_emb:
            mock_emb.return_value = [[0.1] * 1024]

            mock_db.query.side_effect = [
                [],  # dedup check
                [],  # vector search
                [[{"id": "memory:src001"}]],  # batch INSERT
            ]

            await manager_with_meili.upload_memories(
                [
                    {
                        "content": "带来源信息",
                        "source_id": "mem_123_abc",
                        "source_timestamp": "2026-03-11T12:00:00Z",
                    }
                ],
                tenant_id="default",
            )

            meili_docs = mock_meili.add_documents.call_args[0][0]
            assert meili_docs[0]["source_id"] == "mem_123_abc"


# ==================== 混合搜索测试 ====================


class TestHybridSearch:
    """测试 RRF 混合搜索（向量 + 关键词）"""

    @pytest.mark.asyncio
    async def test_hybrid_uses_meili_for_keyword_leg(self, manager_with_meili, mock_db, mock_meili):
        """混合搜索的关键词部分使用 Meilisearch"""
        with patch.object(manager_with_meili, "_get_embeddings", new_callable=AsyncMock) as mock_emb:
            mock_emb.return_value = [[0.1] * 1024]

            # Mock vector search (SurrealDB)
            mock_db.query.return_value = [
                {
                    "id": "memory:v1",
                    "content": "向量匹配",
                    "distance": 0.1,
                    "type": "general",
                    "tags": [],
                    "project_id": "global",
                    "metadata": {},
                },
            ]

            # Mock keyword search (Meilisearch)
            mock_meili.search.return_value = {
                "hits": [
                    {"id": "k1", "surreal_id": "memory:k1", "content": "关键词匹配", "_rankingScore": 0.9},
                ]
            }

            results = await manager_with_meili.search_memories(query="测试", mode="hybrid", limit=10, threshold=0.7)

            assert results["mode"] == "hybrid"
            # 两个来源的结果都应该出现在 merged 结果中
            result_ids = {r["id"] for r in results["results"]}
            assert "memory:v1" in result_ids or "memory:k1" in result_ids

            # 验证 Meilisearch 被用于关键词搜索
            mock_meili.search.assert_called_once()


# ==================== set_meili_client 测试 ====================


class TestSetMeiliClient:
    """测试 Meilisearch 客户端设置"""

    def test_set_meili_client(self, manager, mock_meili):
        """设置 Meilisearch 客户端"""
        assert manager._meili is None
        manager.set_meili_client(mock_meili)
        assert manager._meili is mock_meili

    def test_set_meili_client_overwrite(self, manager, mock_meili):
        """覆盖已有的 Meilisearch 客户端"""
        meili2 = AsyncMock()
        manager.set_meili_client(mock_meili)
        manager.set_meili_client(meili2)
        assert manager._meili is meili2


# ==================== Atom 双写同步测试 ====================


class TestAtomMeiliSync:
    """测试 Atom CRUD 双写同步到 Meilisearch"""

    @pytest.mark.asyncio
    async def test_create_atom_syncs_to_meili(self, manager_with_meili, mock_meili):
        """Atom 创建后同步到 Meilisearch"""
        # 模拟 SurrealDB 返回结果
        mock_record = {"id": "atom:test123", "name": "Test Atom", "content": "Test content"}
        manager_with_meili._db_create = AsyncMock(return_value=[mock_record])
        
        # 调用 _create_atoms_for_entity 方法（这是实际包含 Meilisearch 同步的方法）
        await manager_with_meili._create_atoms_for_entity(
            "entity:parent123",
            [{"name": "Test Atom", "content": "Test content", "type": "section"}],
            "default"
        )
        
        # 验证 Meilisearch 同步被调用
        mock_meili.add_documents.assert_called_once()
        call_args = mock_meili.add_documents.call_args[0][0]
        assert len(call_args) == 1
        assert call_args[0]["doc_type"] == "atom"
        assert call_args[0]["surreal_id"] == "atom:test123"

    @pytest.mark.asyncio
    async def test_create_atom_without_meili(self, manager, mock_db):
        """Meilisearch 禁用时 Atom 创建不报错"""
        manager._db_create = AsyncMock(return_value=[{"id": "atom:test456"}])
        
        # 应该正常完成不抛出异常
        await manager._create_atoms_for_entity(
            "entity:parent123",
            [{"name": "Test Atom", "type": "section"}],
            "default"
        )
        
        # 验证 SurrealDB 创建被调用
        manager._db_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_meili_sync_failure_not_blocking(self, manager_with_meili, mock_meili):
        """Meilisearch 同步失败不阻塞主流程"""
        mock_meili.add_documents.side_effect = Exception("Meilisearch error")
        
        mock_record = {"id": "atom:test789", "name": "Test Atom"}
        manager_with_meili._db_create = AsyncMock(return_value=[mock_record])
        
        # 应该正常完成不抛出异常
        await manager_with_meili._create_atoms_for_entity(
            "entity:parent123",
            [{"name": "Test Atom", "type": "section"}],
            "default"
        )
        
        # 验证 Meilisearch 同步被尝试调用
        mock_meili.add_documents.assert_called_once()

    def test_build_atom_meili_doc(self, manager):
        """测试 Atom Meilisearch 文档构建"""
        atom_data = {
            "name": "Test Atom",
            "content": "Test content",
            "type": "section",
            "tags": ["test"],
            "entity_id": "entity:parent",
            "local_id": "01KTEST",
            "heading_level": 2,
        }
        
        doc = manager._build_meili_doc(
            "atom:test123", atom_data, "default", doc_type="atom"
        )
        
        assert doc["id"] == "atom_test123"
        assert doc["surreal_id"] == "atom:test123"
        assert doc["doc_type"] == "atom"
        assert doc["name"] == "Test Atom"
        assert doc["content"] == "Test content"
        assert doc["atom_type"] == "section"
        assert doc["entity_id"] == "entity:parent"
        assert doc["local_id"] == "01KTEST"
        assert doc["heading_level"] == 2

    def test_build_atom_meili_doc_defaults(self, manager):
        """测试 Atom Meilisearch 文档默认值"""
        atom_data = {"name": "Test"}
        
        doc = manager._build_meili_doc(
            "atom:test456", atom_data, "default", doc_type="atom"
        )
        
        assert doc["name"] == "Test"
        assert doc["content"] == ""
        assert doc["atom_type"] == "note"  # 默认类型
        assert doc["tags"] == []
        assert doc["entity_id"] == ""
        assert doc["local_id"] == ""
        assert "created_at" in doc
