"""测试 db_helpers 工具函数"""

from surrealdb.data.types.record_id import RecordID

from wrapper.src.utils.db_helpers import (
    extract_record_id,
    parse_pagination_params,
    parse_record_id,
    parse_surrealdb_result,
)


class TestParseSurrealdbResult:
    """测试 parse_surrealdb_result 函数"""

    def test_parse_dict_result(self):
        """测试解析字典结果"""
        result = {"id": "atom:test123", "name": "test"}
        record = parse_surrealdb_result(result)
        assert record == result

    def test_parse_list_result(self):
        """测试解析列表结果"""
        result = [{"id": "atom:test123", "name": "test"}]
        record = parse_surrealdb_result(result)
        assert record == {"id": "atom:test123", "name": "test"}

    def test_parse_nested_list_result(self):
        """测试解析嵌套列表结果"""
        result = [[{"id": "atom:test123", "name": "test"}]]
        record = parse_surrealdb_result(result)
        assert record == {"id": "atom:test123", "name": "test"}

    def test_parse_empty_result(self):
        """测试解析空结果"""
        assert parse_surrealdb_result(None) is None
        assert parse_surrealdb_result([]) is None
        assert parse_surrealdb_result({}) == {}


class TestExtractRecordId:
    """测试 extract_record_id 函数"""

    def test_extract_from_dict(self):
        """测试从字典提取 ID"""
        record = {"id": "atom:test123"}
        assert extract_record_id(record) == "atom:test123"

    def test_extract_from_recordid_object(self):
        """测试从 RecordID 对象提取 ID"""
        record_id = RecordID("atom", "test123")
        record = {"id": record_id, "name": "test"}
        assert extract_record_id(record) == "atom:test123"

    def test_extract_direct_recordid(self):
        """测试直接传入 RecordID 对象"""
        record_id = RecordID("atom", "test123")
        assert extract_record_id(record_id) == "atom:test123"

    def test_extract_empty(self):
        """测试空值处理"""
        assert extract_record_id(None) == ""
        assert extract_record_id({}) == ""


class TestParseRecordId:
    """测试 parse_record_id 函数"""

    def test_parse_valid_id(self):
        """测试解析有效的 RecordID"""
        result = parse_record_id("atom:test123")
        assert result == ("atom", "test123")

    def test_parse_invalid_id(self):
        """测试解析无效的 RecordID"""
        assert parse_record_id("invalid") is None
        assert parse_record_id("") is None


class TestParsePaginationParams:
    """测试 parse_pagination_params 函数"""

    def test_parse_with_limit_offset(self):
        """测试使用 limit/offset 参数"""
        skip, take = parse_pagination_params(page=1, page_size=50, limit=10, offset=20)
        assert skip == 20
        assert take == 10

    def test_parse_with_page_page_size(self):
        """测试使用 page/page_size 参数"""
        skip, take = parse_pagination_params(page=3, page_size=50, limit=None, offset=None)
        assert skip == 100  # (3-1) * 50
        assert take == 50

    def test_parse_page_1(self):
        """测试第一页"""
        skip, take = parse_pagination_params(page=1, page_size=20, limit=None, offset=None)
        assert skip == 0
        assert take == 20
