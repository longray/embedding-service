# -*- coding: utf-8 -*-
"""
Meilisearch 配置文件
场景：代码/开发者工具
策略：双字段 + 零副作用
端口：18003
"""

import os
from dotenv import load_dotenv

load_dotenv()


class MeiliConfig:
    """Meilisearch 配置类"""

    # 连接配置
    HOST = os.getenv("MEILI_HOST", "http://localhost:18003")
    MASTER_KEY = os.getenv("MEILI_MASTER_KEY", "meili_master_key_2026_safe")
    INDEX_NAME = os.getenv("MEILI_INDEX_NAME", "code_search_index")

    # 监控阈值
    MAX_INDEX_SIZE_GB = int(os.getenv("MAX_INDEX_SIZE_GB", "10"))
    MAX_DOCUMENTS = int(os.getenv("MAX_DOCUMENTS", "1000000"))

    # ⭐ 核心配置：双字段策略规避副作用
    SETTINGS = {
        # ========== 可搜索字段（双字段策略）==========
        "searchableAttributes": [
            # 中文内容字段（启用 localizedAttributes）
            "title_zh",
            "description_zh",
            "content_zh",
            "tags_zh",
            # 代码搜索字段（分词优化）
            "file_name_search",
            "class_name_search",
            "method_name_search",
            "namespace_search",
            "code_content_search",
        ],
        # ========== 可过滤字段（精确匹配）==========
        "filterableAttributes": [
            "file_path",
            "version",
            "language",
            "project_name",
            "email",
            "ip_address",
            "status",
            "created_at",
            "updated_at",
            "file_size",
            "line_count",
        ],
        # ========== 可排序字段 ==========
        "sortableAttributes": [
            "created_at",
            "updated_at",
            "file_size",
            "line_count",
            "version",
        ],
        # ========== ⭐ 核心：保留特殊字符（代码场景）==========
        "nonSeparatorTokens": [
            ".",
            "-",
            "@",
            ":",
            "/",
            "_",
            "=",
            "+",
            "#",
            "::",
            "->",
            "=>",
        ],
        # ========== ⭐ 核心：中文分词配置 ==========
        "localizedAttributes": [{"locales": ["zho"], "attributePatterns": ["*_zh"]}],
        # ========== ⭐ 核心：代码词典（提升匹配率）==========
        "dictionary": [
            # 版本前缀
            "v1",
            "v2",
            "v3",
            "v4",
            "v5",
            "alpha",
            "beta",
            "rc",
            "release",
            "snapshot",
            # 编程语言
            "python",
            "java",
            "javascript",
            "typescript",
            "go",
            "rust",
            "cpp",
            "csharp",
            "ruby",
            "php",
            "swift",
            "kotlin",
            "scala",
            # 常见命名
            "http",
            "https",
            "api",
            "www",
            "localhost",
            "com",
            "cn",
            "org",
            "net",
            "io",
            "dev",
            "get",
            "post",
            "put",
            "delete",
            "patch",
            # 代码术语
            "class",
            "interface",
            "enum",
            "struct",
            "function",
            "method",
            "property",
            "attribute",
            "import",
            "export",
            "require",
            "include",
            "public",
            "private",
            "protected",
            "static",
            "async",
            "await",
            "promise",
            "callback",
            # 框架/库
            "django",
            "flask",
            "fastapi",
            "spring",
            "react",
            "vue",
            "angular",
            "next",
            "nuxt",
            "tensorflow",
            "pytorch",
            "sklearn",
            # 常见 ID 前缀
            "ID",
            "NO",
            "NUM",
            "CODE",
            "KEY",
            "ORD",
            "PAY",
            "TRK",
            "INV",
            "USR",
            # 时间
            "2025",
            "2026",
            "2027",
            "2028",
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
            # IP 段
            "192",
            "168",
            "172",
            "10",
            "127",
            "0",
            "1",
        ],
        # ========== ⭐ 核心：容错配置（代码字段禁用）==========
        "typoTolerance": {
            "enabled": True,
            "minWordSizeForTypos": {"oneTypo": 5, "twoTypos": 10},
            "disableOnAttributes": [
                "file_path",
                "version",
                "email",
                "ip_address",
                "project_name",
                "class_name_search",
                "method_name_search",
                "namespace_search",
            ],
        },
        # ========== 排序规则 ==========
        "rankingRules": [
            "words",
            "typo",
            "proximity",
            "attribute",
            "exactness",
            "sort",
        ],
        # ========== 分词优化 ==========
        "separatorTokens": [],
        "stopWords": [
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "的",
            "了",
            "是",
            "在",
            "我",
            "有",
        ],
        # ========== 分面搜索 ==========
        "faceting": {"maxValuesPerFacet": 100},
        # ========== 分页 ==========
        "pagination": {"maxTotalHits": 10000},
    }

    @classmethod
    def get_client(cls):
        """获取 Meilisearch 客户端"""
        import meilisearch

        return meilisearch.Client(cls.HOST, cls.MASTER_KEY)

    @classmethod
    def get_index(cls):
        """获取索引对象"""
        client = cls.get_client()
        return client.index(cls.INDEX_NAME)

    @classmethod
    def check_version(cls):
        """检查 Meilisearch 版本兼容性"""
        import requests
        from packaging import version

        try:
            headers = {"Authorization": f"Bearer {cls.MASTER_KEY}"}
            response = requests.get(f"{cls.HOST}/version", headers=headers, timeout=5)
            data = response.json()
            current_version = data.get("pkgVersion", "0.0.0")

            min_version = "1.7.0"

            if version.parse(current_version) >= version.parse(min_version):
                return True, current_version
            else:
                return False, f"需要 v{min_version}+, 当前 v{current_version}"
        except Exception as e:
            return False, str(e)
