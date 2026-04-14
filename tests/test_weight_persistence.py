"""WeightCalculator 持久化测试

测试范围：
- 权重保存到数据库
- 权重从数据库加载
- 批量持久化
- DB 连接管理

运行方式：
    uv run pytest tests/test_weight_persistence.py -v
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from wrapper.src.services.weight_calculator import WeightCalculator, WeightFactors


class TestWeightPersistence:
    """权重持久化测试"""

    @pytest.fixture
    def mock_db(self):
        """创建 mock 数据库"""
        db = AsyncMock()
        db.query = AsyncMock()
        return db

    @pytest.fixture
    def calculator_with_db(self, mock_db):
        """创建带 DB 的 WeightCalculator 实例"""
        return WeightCalculator(db=mock_db)

    @pytest.fixture
    def calculator_without_db(self):
        """创建不带 DB 的 WeightCalculator 实例"""
        return WeightCalculator(db=None)

    def test_init_with_db(self, mock_db):
        """测试带 DB 初始化"""
        calc = WeightCalculator(db=mock_db)
        assert calc._db is mock_db

    def test_init_without_db(self):
        """测试不带 DB 初始化"""
        calc = WeightCalculator(db=None)
        assert calc._db is None

    @pytest.mark.asyncio
    async def test_save_weight_to_db_success(self, calculator_with_db, mock_db):
        """测试成功保存权重到 DB"""
        mock_db.query = AsyncMock(return_value=[])

        result = await calculator_with_db.save_weight_to_db(
            caller="entity:abc",
            callee="entity:def",
            weight=0.75,
            tenant_id="default",
        )

        assert result is True
        mock_db.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_weight_to_db_no_db(self, calculator_without_db):
        """测试无 DB 时保存失败"""
        result = await calculator_without_db.save_weight_to_db(
            caller="entity:abc",
            callee="entity:def",
            weight=0.75,
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_save_weight_to_db_error(self, calculator_with_db, mock_db):
        """测试 DB 错误时保存失败"""
        mock_db.query = AsyncMock(side_effect=Exception("DB Error"))

        result = await calculator_with_db.save_weight_to_db(
            caller="entity:abc",
            callee="entity:def",
            weight=0.75,
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_persist_all_weights(self, calculator_with_db, mock_db):
        """测试批量持久化权重"""
        # 先保存一些权重到内存
        calculator_with_db.save_weight("entity:abc->entity:def", 0.75)
        calculator_with_db.save_weight("entity:def->entity:ghi", 0.80)

        mock_db.query = AsyncMock(return_value=[])

        result = await calculator_with_db.persist_all_weights(tenant_id="default")

        assert result["success"] == 2
        assert result["failed"] == 0

    @pytest.mark.asyncio
    async def test_persist_all_weights_no_db(self, calculator_without_db):
        """测试无 DB 时批量持久化失败"""
        calculator_without_db.save_weight("entity:abc->entity:def", 0.75)

        result = await calculator_without_db.persist_all_weights()

        assert result["success"] == 0
        assert result["failed"] == 0

    @pytest.mark.asyncio
    async def test_get_weight_from_db_success(self, calculator_with_db, mock_db):
        """测试从 DB 获取权重成功"""
        mock_db.query = AsyncMock(return_value=[{"weight": 0.75}])

        weight = await calculator_with_db.get_weight_from_db(
            caller="entity:abc",
            callee="entity:def",
            tenant_id="default",
        )

        assert weight == 0.75

    @pytest.mark.asyncio
    async def test_get_weight_from_db_not_found(self, calculator_with_db, mock_db):
        """测试从 DB 获取权重不存在"""
        mock_db.query = AsyncMock(return_value=[])

        weight = await calculator_with_db.get_weight_from_db(
            caller="entity:abc",
            callee="entity:def",
        )

        assert weight is None

    @pytest.mark.asyncio
    async def test_get_weight_from_db_no_db(self, calculator_without_db):
        """测试无 DB 时获取权重返回 None"""
        weight = await calculator_without_db.get_weight_from_db(
            caller="entity:abc",
            callee="entity:def",
        )

        assert weight is None

    @pytest.mark.asyncio
    async def test_load_weights_from_db(self, calculator_with_db, mock_db):
        """测试从 DB 加载权重"""
        mock_db.query = AsyncMock(
            return_value=[
                {"in": "entity:abc", "out": "entity:def", "weight": 0.75},
                {"in": "entity:def", "out": "entity:ghi", "weight": 0.80},
            ]
        )

        count = await calculator_with_db.load_weights_from_db(tenant_id="default")

        assert count == 2
        assert calculator_with_db.get_weight("entity:abc->entity:def") == 0.75
        assert calculator_with_db.get_weight("entity:def->entity:ghi") == 0.80

    @pytest.mark.asyncio
    async def test_load_weights_from_db_no_db(self, calculator_without_db):
        """测试无 DB 时加载权重返回 0"""
        count = await calculator_without_db.load_weights_from_db()

        assert count == 0

    @pytest.mark.asyncio
    async def test_load_weights_from_db_error(self, calculator_with_db, mock_db):
        """测试 DB 错误时加载权重返回 0"""
        mock_db.query = AsyncMock(side_effect=Exception("DB Error"))

        count = await calculator_with_db.load_weights_from_db()

        assert count == 0


class TestWeightPersistenceEdgeCases:
    """权重持久化边界情况测试"""

    @pytest.mark.asyncio
    async def test_persist_invalid_relation_id(self):
        """测试持久化无效的关系 ID"""
        mock_db = AsyncMock()
        calc = WeightCalculator(db=mock_db)

        # 保存无效格式的 relation_id
        calc.save_weight("invalid_id", 0.75)

        result = await calc.persist_all_weights()

        # 无效格式应该被跳过
        assert result["success"] == 0
        assert result["failed"] == 0

    @pytest.mark.asyncio
    async def test_persist_partial_failure(self):
        """测试部分持久化失败"""
        mock_db = AsyncMock()
        calc = WeightCalculator(db=mock_db)

        # 设置第一个调用成功，第二个失败
        async def side_effect(*args, **kwargs):
            if "abc" in str(args):
                return []
            raise Exception("DB Error")

        mock_db.query = AsyncMock(side_effect=side_effect)

        calc.save_weight("entity:abc->entity:def", 0.75)
        calc.save_weight("entity:def->entity:ghi", 0.80)

        result = await calc.persist_all_weights()

        assert result["success"] == 1
        assert result["failed"] == 1

    @pytest.mark.asyncio
    async def test_load_weights_with_null_weight(self):
        """测试加载权重时处理 NULL"""
        mock_db = AsyncMock()
        mock_db.query = AsyncMock(
            return_value=[
                {"in": "entity:abc", "out": "entity:def", "weight": None},
                {"in": "entity:def", "out": "entity:ghi", "weight": 0.80},
            ]
        )

        calc = WeightCalculator(db=mock_db)
        count = await calc.load_weights_from_db()

        # NULL 权重应该被跳过
        assert count == 1
        assert calc.get_weight("entity:def->entity:ghi") == 0.80

    @pytest.mark.asyncio
    async def test_save_weight_with_special_chars(self):
        """测试保存包含特殊字符的 entity ID"""
        mock_db = AsyncMock()
        mock_db.query = AsyncMock(return_value=[])

        calc = WeightCalculator(db=mock_db)

        result = await calc.save_weight_to_db(
            caller="entity:abc::123",
            callee="entity:def::456",
            weight=0.75,
        )

        assert result is True
