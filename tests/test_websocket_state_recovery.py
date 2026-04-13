"""WebSocket 状态恢复管理器测试

测试范围：
- StateRecoveryManager 基础功能
- Session ID 生成
- 状态保存和恢复
- TTL 清理

运行方式：
    uv run pytest tests/test_websocket_state_recovery.py -v
"""

import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from wrapper.src.websocket.state_recovery import StateRecoveryManager


class TestStateRecoveryManager:
    """StateRecoveryManager 单元测试"""

    @pytest.fixture
    def temp_state_file(self):
        """创建临时状态文件"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"sessions": {}}')
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)

    @pytest.fixture
    def state_manager(self, temp_state_file):
        """创建 StateRecoveryManager 实例"""
        return StateRecoveryManager(state_file=temp_state_file, ttl_days=7)

    def test_initialization(self, temp_state_file):
        """测试初始化"""
        manager = StateRecoveryManager(state_file=temp_state_file)

        assert manager.state_file == Path(temp_state_file)
        assert manager.session_count == 0

    def test_generate_session_id(self, state_manager):
        """测试 Session ID 生成"""
        session_id = state_manager.generate_session_id()

        assert session_id.startswith("sess-")
        parts = session_id.split("-")
        assert len(parts) == 3
        assert len(parts[2]) == 9

    def test_save_and_restore_state(self, state_manager):
        """测试保存和恢复状态"""
        session_id = state_manager.generate_session_id()
        offset = 100
        data = {"tenant_id": "default", "user_id": "user123"}

        state_manager.save_state(session_id, offset, data)
        restored = state_manager.restore_state(session_id)

        assert restored is not None
        assert restored["offset"] == offset
        assert restored["data"] == data
        assert "created_at" in restored
        assert "updated_at" in restored

    def test_restore_nonexistent_state(self, state_manager):
        """测试恢复不存在的状态"""
        restored = state_manager.restore_state("sess-nonexistent")

        assert restored is None

    def test_delete_state(self, state_manager):
        """测试删除状态"""
        session_id = state_manager.generate_session_id()
        state_manager.save_state(session_id, 100)

        result = state_manager.delete_state(session_id)

        assert result is True
        assert state_manager.session_count == 0
        assert state_manager.restore_state(session_id) is None

    def test_delete_nonexistent_state(self, state_manager):
        """测试删除不存在的状态"""
        result = state_manager.delete_state("sess-nonexistent")

        assert result is False

    def test_get_offset(self, state_manager):
        """测试获取 offset"""
        session_id = state_manager.generate_session_id()
        state_manager.save_state(session_id, 200)

        offset = state_manager.get_offset(session_id)

        assert offset == 200

    def test_get_offset_nonexistent(self, state_manager):
        """测试获取不存在 session 的 offset"""
        offset = state_manager.get_offset("sess-nonexistent")

        assert offset == 0

    def test_update_offset(self, state_manager):
        """测试更新 offset"""
        session_id = state_manager.generate_session_id()
        state_manager.save_state(session_id, 100)

        state_manager.update_offset(session_id, 300)
        offset = state_manager.get_offset(session_id)

        assert offset == 300

    def test_session_exists(self, state_manager):
        """测试检查 session 是否存在"""
        session_id = state_manager.generate_session_id()
        state_manager.save_state(session_id, 100)

        assert state_manager.session_exists(session_id) is True
        assert state_manager.session_exists("sess-nonexistent") is False

    def test_get_all_sessions(self, state_manager):
        """测试获取所有 sessions"""
        session1 = state_manager.generate_session_id()
        session2 = state_manager.generate_session_id()
        state_manager.save_state(session1, 100)
        state_manager.save_state(session2, 200)

        sessions = state_manager.get_all_sessions()

        assert len(sessions) == 2
        assert session1 in sessions
        assert session2 in sessions


class TestStateRecoveryTTL:
    """TTL 清理测试"""

    @pytest.fixture
    def temp_state_file(self):
        """创建临时状态文件"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"sessions": {}}')
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)

    def test_cleanup_expired(self, temp_state_file):
        """测试清理过期状态"""
        manager = StateRecoveryManager(state_file=temp_state_file, ttl_days=1)

        old_time = (datetime.utcnow() - timedelta(days=2)).isoformat()
        manager._state["sessions"]["sess-old"] = {
            "offset": 100,
            "created_at": old_time,
            "updated_at": old_time,
        }

        new_time = datetime.utcnow().isoformat()
        manager._state["sessions"]["sess-new"] = {
            "offset": 200,
            "created_at": new_time,
            "updated_at": new_time,
        }

        manager._save_state()

        cleaned = manager.cleanup_expired()

        assert cleaned == 1
        assert manager.session_count == 1
        assert "sess-old" not in manager.get_all_sessions()
        assert "sess-new" in manager.get_all_sessions()

    def test_no_cleanup_for_fresh_sessions(self, temp_state_file):
        """测试不清理新鲜状态"""
        manager = StateRecoveryManager(state_file=temp_state_file, ttl_days=7)

        session_id = manager.generate_session_id()
        manager.save_state(session_id, 100)

        cleaned = manager.cleanup_expired()

        assert cleaned == 0
        assert manager.session_count == 1


class TestStateRecoveryPersistence:
    """持久化测试"""

    def test_state_persistence(self):
        """测试状态持久化到文件"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            manager1 = StateRecoveryManager(state_file=temp_path)
            session_id = manager1.generate_session_id()
            manager1.save_state(session_id, 100, {"key": "value"})

            manager2 = StateRecoveryManager(state_file=temp_path)
            restored = manager2.restore_state(session_id)

            assert restored is not None
            assert restored["offset"] == 100
            assert restored["data"]["key"] == "value"
        finally:
            os.unlink(temp_path)

    def test_file_format(self):
        """测试文件格式"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            manager = StateRecoveryManager(state_file=temp_path)
            session_id = manager.generate_session_id()
            manager.save_state(session_id, 100)

            with open(temp_path, "r") as f:
                data = json.load(f)

            assert "sessions" in data
            assert session_id in data["sessions"]
            assert "offset" in data["sessions"][session_id]
            assert "created_at" in data["sessions"][session_id]
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
