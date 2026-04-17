"""代码指纹同步 API 测试 (BL-B-80)

测试 POST /api/v1/sync/code-fingerprints 端点
"""

import pytest


class TestCodeFingerprintSync:
    """代码指纹同步测试"""

    @pytest.mark.asyncio
    async def test_sync_code_fingerprints_success(self, wrapper_client):
        """测试成功的指纹同步"""
        request_data = {
            "fingerprints": [
                {"file": "src/main.py", "content_hash": "abc123", "symbols_hash": "def456"},
                {"file": "src/utils.py", "content_hash": "ghi789", "symbols_hash": "jkl012"},
            ],
            "tenant_id": "default",
            "project_id": "test-project",
        }

        response = await wrapper_client.post("/api/v1/sync/code-fingerprints", json=request_data)

        assert response.status_code == 200
        result = response.json()
        assert "changed_files" in result
        assert "unchanged_files" in result
        assert "new_files" in result
        assert "deleted_files" in result

    @pytest.mark.asyncio
    async def test_sync_code_fingerprints_empty(self, wrapper_client):
        """测试空指纹列表"""
        request_data = {
            "fingerprints": [],
            "tenant_id": "default",
            "project_id": "test-project",
        }

        response = await wrapper_client.post("/api/v1/sync/code-fingerprints", json=request_data)

        assert response.status_code == 200
        result = response.json()
        assert result["changed_files"] == []
        assert result["unchanged_files"] == []
        assert result["new_files"] == []
        assert result["deleted_files"] == []

    @pytest.mark.asyncio
    async def test_sync_code_fingerprints_missing_tenant(self, wrapper_client):
        """测试缺少 tenant_id（使用默认值）"""
        request_data = {
            "fingerprints": [
                {"file": "src/main.py", "content_hash": "abc123", "symbols_hash": "def456"},
            ],
            "project_id": "test-project",
        }

        response = await wrapper_client.post("/api/v1/sync/code-fingerprints", json=request_data)

        # 应该使用默认值 "default"
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_sync_code_fingerprints_invalid_data(self, wrapper_client):
        """测试无效数据"""
        request_data = {
            "fingerprints": "invalid",  # 应该是列表
            "tenant_id": "default",
        }

        response = await wrapper_client.post("/api/v1/sync/code-fingerprints", json=request_data)

        assert response.status_code == 422  # Validation error


class TestCodeFingerprintService:
    """CodeFingerprintService 单元测试"""

    @pytest.mark.asyncio
    async def test_compare_fingerprints_new_files(self):
        """测试比对新文件"""
        from wrapper.src.services.code_fingerprint_service import CodeFingerprintService
        from unittest.mock import MagicMock, AsyncMock

        mock_db = MagicMock()
        mock_db.query = AsyncMock(return_value=[{"result": []}])

        service = CodeFingerprintService(mock_db)

        fingerprints = [
            {"file": "src/main.py", "content_hash": "abc123", "symbols_hash": "def456"},
        ]

        result = await service.compare_fingerprints(
            fingerprints=fingerprints,
            tenant_id="default",
            project_id="test",
        )

        assert result["new_files"] == ["src/main.py"]
        assert result["changed_files"] == []
        assert result["unchanged_files"] == []

    @pytest.mark.asyncio
    async def test_compare_fingerprints_changed_files(self):
        """测试比对变更文件"""
        from wrapper.src.services.code_fingerprint_service import CodeFingerprintService
        from unittest.mock import MagicMock, AsyncMock

        mock_db = MagicMock()
        # 模拟数据库中已有相同路径但不同哈希的文件
        mock_db.query = AsyncMock(
            return_value=[
                {
                    "result": [
                        {"file_path": "src/main.py", "content_hash": "old_hash", "symbols_hash": "old_symbols"},
                    ]
                }
            ]
        )

        service = CodeFingerprintService(mock_db)

        fingerprints = [
            {"file": "src/main.py", "content_hash": "new_hash", "symbols_hash": "new_symbols"},
        ]

        result = await service.compare_fingerprints(
            fingerprints=fingerprints,
            tenant_id="default",
            project_id="test",
        )

        assert result["changed_files"] == ["src/main.py"]
        assert result["new_files"] == []
        assert result["unchanged_files"] == []

    @pytest.mark.asyncio
    async def test_compare_fingerprints_unchanged_files(self):
        """测试比对未变更文件"""
        from wrapper.src.services.code_fingerprint_service import CodeFingerprintService
        from unittest.mock import MagicMock, AsyncMock

        mock_db = MagicMock()
        # 模拟数据库中已有相同路径且相同哈希的文件
        mock_db.query = AsyncMock(
            return_value=[
                {
                    "result": [
                        {"file_path": "src/main.py", "content_hash": "abc123", "symbols_hash": "def456"},
                    ]
                }
            ]
        )

        service = CodeFingerprintService(mock_db)

        fingerprints = [
            {"file": "src/main.py", "content_hash": "abc123", "symbols_hash": "def456"},
        ]

        result = await service.compare_fingerprints(
            fingerprints=fingerprints,
            tenant_id="default",
            project_id="test",
        )

        assert result["unchanged_files"] == ["src/main.py"]
        assert result["changed_files"] == []
        assert result["new_files"] == []

    @pytest.mark.asyncio
    async def test_update_fingerprints(self):
        """测试更新指纹"""
        from wrapper.src.services.code_fingerprint_service import CodeFingerprintService
        from unittest.mock import MagicMock, AsyncMock

        mock_db = MagicMock()
        mock_db.query = AsyncMock(return_value=None)

        service = CodeFingerprintService(mock_db)

        fingerprints = [
            {"file": "src/main.py", "content_hash": "abc123", "symbols_hash": "def456"},
        ]

        count = await service.update_fingerprints(
            fingerprints=fingerprints,
            tenant_id="default",
            project_id="test",
        )

        assert count == 1
        mock_db.query.assert_called()

    @pytest.mark.asyncio
    async def test_delete_fingerprints(self):
        """测试删除指纹"""
        from wrapper.src.services.code_fingerprint_service import CodeFingerprintService
        from unittest.mock import MagicMock, AsyncMock

        mock_db = MagicMock()
        mock_db.query = AsyncMock(return_value=None)

        service = CodeFingerprintService(mock_db)

        file_paths = ["src/deleted.py"]

        count = await service.delete_fingerprints(
            file_paths=file_paths,
            tenant_id="default",
            project_id="test",
        )

        assert count == 1
        mock_db.query.assert_called()
