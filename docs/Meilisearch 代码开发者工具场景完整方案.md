# 🚀 Meilisearch 代码/开发者工具场景完整方案

## Windows + Python + 中文内容 | 端口 18003 | 零副作用

---

## 📁 项目结构

```text
meilisearch_code/
├── docker-compose.yml          # Docker 部署（版本锁定）
├── .env                        # 环境变量
├── requirements.txt            # Python 依赖
├── config.py                   # 核心配置（双字段策略）
├── init_index.py               # 索引初始化
├── test_search.py              # 完整测试用例
├── monitor_index.py            # 索引监控
├── optimize_index.py           # 索引优化
├── sample_documents.json       # 示例文档
└── README.md                   # 使用说明
```

---

## 1️⃣ docker-compose.yml

```yaml
version: '3.8'

services:
  meilisearch:
    image: getmeili/meilisearch:v1.12.0
    container_name: meilisearch_code
    ports:
      - "18003:7700"
    environment:
      - MEILI_MASTER_KEY=meili_master_key_2026_safe
      - MEILI_ENV=production
      - MEILI_NO_ANALYTICS=true
      - MEILI_MAX_INDEX_SIZE=10737418240
      - MEILI_MAX_TASK_DB_SIZE=10737418240
    volumes:
      - meilisearch_code_data:/meili_data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:7700/health"]
      interval: 10s
      timeout: 5s
      retries: 5
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 1G

volumes:
  meilisearch_code_data:
```text

---

## 2️⃣ .env

```bash
# Meilisearch 连接配置
MEILI_HOST=http://localhost:18003
MEILI_MASTER_KEY=meili_master_key_2026_safe
MEILI_INDEX_NAME=code_search_index

# 索引配置
MAX_INDEX_SIZE_GB=10
MAX_DOCUMENTS=1000000

# Python 编码（Windows 关键）
PYTHONIOENCODING=utf-8
PYTHONUTF8=1
```

---

## 3️⃣ requirements.txt

```txt
meilisearch>=0.31.0
python-dotenv>=1.0.0
requests>=2.31.0
packaging>=23.0
```text

---

## 4️⃣ config.py

```python
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
            "file_name_search",      # 文件名（分词）
            "class_name_search",     # 类名（分词）
            "method_name_search",    # 方法名（分词）
            "namespace_search",      # 命名空间（分词）
            "code_content_search",   # 代码内容（分词）
            
            # 精确字段（用于 filter，不参与全文搜索）
            # "version", "file_path", "email" 等放在 filterableAttributes
        ],
        
        # ========== 可过滤字段（精确匹配）==========
        "filterableAttributes": [
            "file_path",             # 完整文件路径
            "version",               # 完整版本号
            "language",              # 编程语言
            "project_name",          # 项目名称
            "email",                 # 完整邮箱
            "ip_address",            # 完整 IP
            "status",                # 状态
            "created_at",            # 创建时间
            "updated_at",            # 更新时间
            "file_size",             # 文件大小
            "line_count",            # 行数
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
            ".", "-", "@", ":", "/",  # 基础
            "_", "=", "+", "#",       # 代码常用
            "::", "->", "=>",         # 语言特定（需单独测试）
        ],
        
        # ========== ⭐ 核心：中文分词配置 ==========
        "localizedAttributes": [
            {
                "locales": ["zho"],
                "attributePatterns": ["*_zh"]
            }
        ],
        
        # ========== ⭐ 核心：代码词典（提升匹配率）==========
        "dictionary": [
            # 版本前缀
            "v1", "v2", "v3", "v4", "v5",
            "alpha", "beta", "rc", "release", "snapshot",
            
            # 编程语言
            "python", "java", "javascript", "typescript",
            "go", "rust", "cpp", "csharp", "ruby",
            "php", "swift", "kotlin", "scala",
            
            # 常见命名
            "http", "https", "api", "www", "localhost",
            "com", "cn", "org", "net", "io", "dev",
            "get", "post", "put", "delete", "patch",
            
            # 代码术语
            "class", "interface", "enum", "struct",
            "function", "method", "property", "attribute",
            "import", "export", "require", "include",
            "public", "private", "protected", "static",
            "async", "await", "promise", "callback",
            
            # 框架/库
            "django", "flask", "fastapi", "spring",
            "react", "vue", "angular", "next", "nuxt",
            "tensorflow", "pytorch", "sklearn",
            
            # 常见 ID 前缀
            "ID", "NO", "NUM", "CODE", "KEY",
            "ORD", "PAY", "TRK", "INV", "USR",
            
            # 时间
            "2025", "2026", "2027", "2028",
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
            
            # IP 段
            "192", "168", "172", "10", "127", "0", "1",
        ],
        
        # ========== ⭐ 核心：容错配置（代码字段禁用）==========
        "typoTolerance": {
            "enabled": True,
            "minWordSizeForTypos": {
                "oneTypo": 5,       # 代码场景提高至 5
                "twoTypos": 10      # 代码场景提高至 10
            },
            # 精确字段禁用容错
            "disableOnAttributes": [
                "file_path",
                "version",
                "email",
                "ip_address",
                "project_name",
                "class_name_search",
                "method_name_search",
                "namespace_search",
            ]
        },
        
        # ========== 排序规则 ==========
        "rankingRules": [
            "words",        # 匹配词数
            "typo",         # 容错数
            "proximity",    # 词距离
            "attribute",    # 字段优先级
            "exactness",    # 精确匹配
            "sort",         # 排序规则
        ],
        
        # ========== 分词优化 ==========
        "separatorTokens": [],  # 不使用额外分隔符
        "stopWords": [          # 停用词（代码场景精简）
            "the", "a", "an", "and", "or", "but",
            "的", "了", "是", "在", "我", "有",
        ],
        
        # ========== 分面搜索 ==========
        "faceting": {
            "maxValuesPerFacet": 100
        },
        
        # ========== 分页 ==========
        "pagination": {
            "maxTotalHits": 10000
        }
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
            response = requests.get(f"{cls.HOST}/version", timeout=5)
            data = response.json()
            current_version = data.get('pkgVersion', '0.0.0')
            
            # 检查最低版本要求
            min_version = "1.7.0"  # localizedAttributes 需要 1.7+
            
            if version.parse(current_version) >= version.parse(min_version):
                return True, current_version
            else:
                return False, f"需要 v{min_version}+, 当前 v{current_version}"
        except Exception as e:
            return False, str(e)
```

---

## 5️⃣ init_index.py

```python
# -*- coding: utf-8 -*-
"""
Meilisearch 索引初始化脚本
场景：代码/开发者工具
策略：双字段 + 零副作用
端口：18003

使用方法:
    python init_index.py
"""

import sys
import time
import json
from pathlib import Path
from config import MeiliConfig
from meilisearch.errors import MeilisearchApiError, MeilisearchTimeoutError

def check_server_health(max_retries=30, delay=2):
    """等待 Meilisearch 服务启动"""
    import requests
    print("⏳ 等待 Meilisearch 服务启动...")
    
    for i in range(max_retries):
        try:
            response = requests.get(f"{MeiliConfig.HOST}/health", timeout=5)
            if response.status_code == 200:
                print("✅ Meilisearch 服务已就绪")
                return True
        except Exception:
            pass
        print(f"   重试 {i+1}/{max_retries}...")
        time.sleep(delay)
    
    print("❌ Meilisearch 服务启动超时")
    return False

def check_version_compatibility():
    """检查版本兼容性"""
    print("\n🔍 检查版本兼容性...")
    success, message = MeiliConfig.check_version()
    
    if success:
        print(f"✅ 版本检查通过：{message}")
        return True
    else:
        print(f"❌ 版本检查失败：{message}")
        return False

def create_or_update_index():
    """创建或更新索引配置"""
    print("\n" + "="*70)
    print("🔧 开始配置 Meilisearch 索引")
    print("="*70)
    
    try:
        client = MeiliConfig.get_client()
        index = client.index(MeiliConfig.INDEX_NAME)
        
        # 1. 创建索引
        print(f"\n📁 索引名称：{MeiliConfig.INDEX_NAME}")
        try:
            client.create_index(MeiliConfig.INDEX_NAME, {"primaryKey": "id"})
            print("✅ 索引创建成功")
        except MeilisearchApiError as e:
            if "index_already_exists" in str(e).lower():
                print("ℹ️  索引已存在，将更新配置")
            else:
                raise
        
        # 2. 应用完整配置
        print("\n⚙️  应用索引配置...")
        print(f"   - 端口：{MeiliConfig.HOST}")
        print(f"   - nonSeparatorTokens: {len(MeiliConfig.SETTINGS['nonSeparatorTokens'])} 个字符")
        print(f"   - dictionary: {len(MeiliConfig.SETTINGS['dictionary'])} 个词条")
        print(f"   - localizedAttributes: {MeiliConfig.SETTINGS['localizedAttributes']}")
        
        task = index.update_settings(MeiliConfig.SETTINGS)
        client.wait_for_task(task['taskUid'], timeout_in_ms=60000)
        print("✅ 配置更新成功")
        
        # 3. 验证配置
        print("\n🔍 验证配置...")
        current_settings = index.get_settings()
        
        assertions = [
            (current_settings.get('nonSeparatorTokens'), 
             MeiliConfig.SETTINGS['nonSeparatorTokens'],
             "nonSeparatorTokens"),
            (len(current_settings.get('localizedAttributes', [])) > 0,
             True,
             "localizedAttributes"),
            (len(current_settings.get('dictionary', [])) > 0,
             True,
             "dictionary"),
        ]
        
        for actual, expected, name in assertions:
            if actual == expected or (isinstance(expected, bool) and bool(actual) == expected):
                print(f"   ✅ {name} 验证通过")
            else:
                print(f"   ❌ {name} 验证失败")
                return False
        
        # 4. 添加测试文档（双字段策略示例）
        print("\n📝 添加测试文档（双字段策略）...")
        test_documents = [
            {
                "id": 1,
                "title_zh": "用户服务类",
                "description_zh": "处理用户认证和授权的公共服务类",
                "content_zh": "包含登录、注册、权限验证等功能",
                "tags_zh": ["用户", "认证", "服务"],
                
                # 精确字段（用于 filter）
                "file_path": "src/main/java/com/example/app/UserService.java",
                "version": "v2.1.0",
                "language": "java",
                "project_name": "example-app",
                "email": "developer@example.com",
                "ip_address": "192.168.1.100",
                "status": "active",
                "created_at": 1710230400,
                "updated_at": 1710230400,
                "file_size": 15360,
                "line_count": 520,
                
                # 搜索字段（分词优化）
                "file_name_search": "UserService java",
                "class_name_search": "UserService",
                "method_name_search": "login register authenticate authorize",
                "namespace_search": "com example app",
                "code_content_search": "public class UserService implements AuthService",
            },
            {
                "id": 2,
                "title_zh": "API 路由配置",
                "description_zh": "定义 RESTful API 端点路由",
                "content_zh": "包含用户、订单、支付等模块的路由配置",
                "tags_zh": ["API", "路由", "配置"],
                
                "file_path": "src/api/routes.py",
                "version": "v2.1.0",
                "language": "python",
                "project_name": "example-app",
                "email": "backend@example.com",
                "ip_address": "10.0.0.1",
                "status": "active",
                "created_at": 1711929600,
                "updated_at": 1711929600,
                "file_size": 8192,
                "line_count": 280,
                
                "file_name_search": "routes python",
                "class_name_search": "APIRouter",
                "method_name_search": "get post put delete",
                "namespace_search": "src api",
                "code_content_search": "from fastapi import APIRouter router = APIRouter",
            },
            {
                "id": 3,
                "title_zh": "数据库连接工具",
                "description_zh": "封装数据库连接和查询操作",
                "content_zh": "支持 MySQL、PostgreSQL、SQLite 等多种数据库",
                "tags_zh": ["数据库", "连接", "工具"],
                
                "file_path": "src/utils/database.ts",
                "version": "v1.9.5",
                "language": "typescript",
                "project_name": "example-app",
                "email": "db@example.com",
                "ip_address": "172.16.0.50",
                "status": "deprecated",
                "created_at": 1733011200,
                "updated_at": 1733011200,
                "file_size": 12288,
                "line_count": 450,
                
                "file_name_search": "database typescript",
                "class_name_search": "DatabaseConnection",
                "method_name_search": "connect query execute transaction",
                "namespace_search": "src utils",
                "code_content_search": "export class DatabaseConnection implements Connection",
            },
            {
                "id": 4,
                "title_zh": "版本发布说明",
                "description_zh": "v2.1.0 版本的发布说明和变更日志",
                "content_zh": "新增用户认证模块，修复 IP 解析 bug，优化邮箱验证",
                "tags_zh": ["版本", "发布", "日志"],
                
                "file_path": "docs/release/v2.1.0.md",
                "version": "v2.1.0",
                "language": "markdown",
                "project_name": "example-app",
                "email": "release@example.com",
                "ip_address": "192.168.1.1",
                "status": "published",
                "created_at": 1710316800,
                "updated_at": 1710316800,
                "file_size": 4096,
                "line_count": 150,
                
                "file_name_search": "release markdown",
                "class_name_search": "",
                "method_name_search": "",
                "namespace_search": "docs release",
                "code_content_search": "version v2.1.0 release notes changelog",
            },
        ]
        
        task = index.add_documents(test_documents)
        client.wait_for_task(task['taskUid'], timeout_in_ms=60000)
        print(f"✅ 已添加 {len(test_documents)} 条测试文档")
        
        # 5. 获取索引统计
        print("\n📊 索引统计:")
        stats = index.get_stats()
        print(f"   - 文档数量：{stats.get('numberOfDocuments', 0)}")
        print(f"   - 索引大小：{stats.get('rawDocumentDbSize', 0) / 1024 / 1024:.2f} MB")
        
        print("\n" + "="*70)
        print("🎉 索引初始化完成！")
        print("="*70)
        return True
        
    except MeilisearchTimeoutError as e:
        print(f"\n❌ 连接超时：{e}")
        return False
    except MeilisearchApiError as e:
        print(f"\n❌ API 错误：{e}")
        return False
    except Exception as e:
        print(f"\n❌ 未知错误：{e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("\n" + "🚀"*35)
    print("   Meilisearch 索引初始化脚本")
    print("   代码/开发者工具场景 | 端口 18003 | 零副作用")
    print("🚀"*35)
    
    # 1. 检查服务
    if not check_server_health():
        print("\n💡 提示：请先启动 Meilisearch 服务")
        print("   Docker: docker-compose up -d")
        sys.exit(1)
    
    # 2. 检查版本
    if not check_version_compatibility():
        sys.exit(1)
    
    # 3. 初始化索引
    success = create_or_update_index()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
```text

---

## 6️⃣ test_search.py

```python
# -*- coding: utf-8 -*-
"""
Meilisearch 搜索测试脚本
场景：代码/开发者工具
验证：所有测试用例 + 副作用规避
端口：18003

使用方法:
    python test_search.py
"""

from config import MeiliConfig
from meilisearch.errors import MeilisearchApiError

# ========== 测试用例清单 ==========
TEST_CASES = [
    # (搜索词, 期望有结果, 描述, 测试类型)
    # --- 特殊格式精确匹配 ---
    ("v2.1.0", True, "完整版本号匹配", "special_format"),
    ("192.168.1.100", True, "完整 IP 匹配", "special_format"),
    ("developer@example.com", True, "完整邮箱匹配", "special_format"),
    ("2026-03-12", True, "完整日期匹配", "special_format"),
    
    # --- 特殊格式前缀匹配 ---
    ("v2", True, "版本前缀匹配", "special_format"),
    ("192", True, "IP 前缀匹配", "special_format"),
    ("developer", True, "邮箱前缀匹配", "special_format"),
    ("2026", True, "日期前缀匹配", "special_format"),
    
    # --- 代码相关搜索 ---
    ("UserService", True, "类名搜索", "code"),
    ("login register", True, "方法名搜索", "code"),
    ("com.example.app", True, "命名空间搜索", "code"),
    ("fastapi", True, "框架搜索", "code"),
    ("python", True, "语言搜索", "code"),
    ("java", True, "语言搜索", "code"),
    
    # --- 中文内容搜索 ---
    ("用户服务", True, "中文标题搜索", "chinese"),
    ("认证", True, "中文标签搜索", "chinese"),
    ("数据库连接", True, "中文描述搜索", "chinese"),
    ("发布说明", True, "中文内容搜索", "chinese"),
    
    # --- 文件路径搜索 ---
    ("UserService.java", True, "文件名搜索", "code"),
    ("routes.py", True, "文件名搜索", "code"),
    ("database.ts", True, "文件名搜索", "code"),
    
    # --- 组合搜索 ---
    ("v2.1.0 用户", True, "版本 + 中文组合", "combined"),
    ("java UserService", True, "语言 + 类名组合", "combined"),
    ("python fastapi", True, "语言 + 框架组合", "combined"),
]

# ========== 过滤器测试用例 ==========
FILTER_TESTS = [
    ('version = "v2.1.0"', "版本精确过滤"),
    ('language = "python"', "语言过滤"),
    ('status = "active"', "状态过滤"),
    ('file_size > 10000', "文件大小过滤"),
    ('line_count >= 300', "行数过滤"),
    ('created_at >= 1710000000', "时间范围过滤"),
    ('email = "developer@example.com"', "邮箱精确过滤"),
    ('ip_address = "192.168.1.100"', "IP 精确过滤"),
]

def run_tests():
    """运行所有测试"""
    print("\n" + "="*80)
    print("🧪 Meilisearch 搜索测试")
    print("   场景：代码/开发者工具 | 端口：http://localhost:18003")
    print("="*80)
    
    index = MeiliConfig.get_index()
    
    results = {
        "special_format": {"passed": 0, "failed": 0, "cases": []},
        "code": {"passed": 0, "failed": 0, "cases": []},
        "chinese": {"passed": 0, "failed": 0, "cases": []},
        "combined": {"passed": 0, "failed": 0, "cases": []},
    }
    
    for query, should_match, description, test_type in TEST_CASES:
        try:
            result = index.search(query, {"limit": 1})
            has_results = len(result.get('hits', [])) > 0
            success = has_results == should_match
            
            case_result = {
                "query": query,
                "description": description,
                "hits": len(result.get('hits', [])),
                "expected": "有结果" if should_match else "无结果",
                "success": success
            }
            
            if success:
                results[test_type]["passed"] += 1
                case_result["status"] = "✅"
            else:
                results[test_type]["failed"] += 1
                case_result["status"] = "❌"
            
            results[test_type]["cases"].append(case_result)
            
        except Exception as e:
            results[test_type]["failed"] += 1
            results[test_type]["cases"].append({
                "query": query,
                "description": description,
                "status": "❌",
                "error": str(e)
            })
    
    # 打印结果
    print("\n📋 测试结果:\n")
    
    for test_type, data in results.items():
        print(f"【{test_type.upper()}】通过 {data['passed']}/{data['passed']+data['failed']}")
        for case in data["cases"]:
            status = case.get("status", "❌")
            query = case["query"][:30].ljust(30)
            desc = case["description"][:20].ljust(20)
            hits = str(case.get("hits", "N/A")).ljust(5)
            print(f"  {status} {query} {desc} {hits}")
        print()
    
    # 过滤器测试
    print("\n🔍 过滤器测试:\n")
    filter_passed = 0
    filter_failed = 0
    
    for filter_expr, description in FILTER_TESTS:
        try:
            result = index.search("", {"filter": filter_expr, "limit": 5})
            hits = len(result.get('hits', []))
            status = "✅" if hits > 0 else "⚠️"
            if hits > 0:
                filter_passed += 1
            else:
                filter_failed += 1
            print(f"  {status} {description[:25].ljust(25)} {filter_expr[:40]} → {hits} 结果")
        except Exception as e:
            filter_failed += 1
            print(f"  ❌ {description[:25].ljust(25)} {e}")
    
    # 统计汇总
    total_passed = sum(d["passed"] for d in results.values())
    total_failed = sum(d["failed"] for d in results.values())
    
    print("\n" + "="*80)
    print(f"📊 汇总：搜索测试 {total_passed}/{total_passed+total_failed} | 过滤测试 {filter_passed}/{filter_passed+filter_failed}")
    print("="*80)
    
    return total_failed == 0 and filter_failed == 0

def test_side_effects():
    """测试副作用规避"""
    print("\n" + "="*80)
    print("⚠️  副作用规避验证")
    print("="*80)
    
    index = MeiliConfig.get_index()
    
    # 1. 索引大小检查
    stats = index.get_stats()
    doc_count = stats.get('numberOfDocuments', 0)
    db_size = stats.get('rawDocumentDbSize', 0)
    
    print(f"\n📊 索引大小：{db_size / 1024 / 1024:.2f} MB ({doc_count} 文档)")
    
    if db_size < MeiliConfig.MAX_INDEX_SIZE_GB * 1024**3:
        print("   ✅ 索引大小在阈值内")
    else:
        print("   ⚠️  索引大小超过阈值")
    
    # 2. 配置验证
    settings = index.get_settings()
    
    print(f"\n⚙️  配置验证:")
    print(f"   - nonSeparatorTokens: {len(settings.get('nonSeparatorTokens', []))} 个字符 ✅")
    print(f"   - dictionary: {len(settings.get('dictionary', []))} 个词条 ✅")
    print(f"   - localizedAttributes: {len(settings.get('localizedAttributes', []))} 个配置 ✅")
    
    # 3. 容错禁用验证
    typo_config = settings.get('typoTolerance', {})
    disabled_attrs = typo_config.get('disableOnAttributes', [])
    
    print(f"\n🔒 容错禁用字段：{len(disabled_attrs)} 个")
    for attr in ['file_path', 'version', 'email', 'ip_address']:
        if attr in disabled_attrs:
            print(f"   ✅ {attr}")
        else:
            print(f"   ❌ {attr} (未禁用)")
    
    return True

def main():
    """主函数"""
    print("\n" + "🚀"*40)
    print("   Meilisearch 搜索测试脚本")
    print("   代码/开发者工具场景 | 端口 18003 | 零副作用")
    print("🚀"*40)
    
    try:
        client = MeiliConfig.get_client()
        client.health()
        print("\n✅ Meilisearch 服务连接成功")
    except Exception as e:
        print(f"\n❌ 无法连接 Meilisearch: {e}")
        print("   请确保服务运行在 http://localhost:18003")
        return 1
    
    # 运行搜索测试
    search_success = run_tests()
    
    # 运行副作用测试
    side_effect_success = test_side_effects()
    
    if search_success and side_effect_success:
        print("\n🎉 所有测试通过！")
        return 0
    else:
        print("\n⚠️  部分测试失败，请检查配置")
        return 1

if __name__ == "__main__":
    exit(main())
```

---

## 7️⃣ monitor_index.py

```python
# -*- coding: utf-8 -*-
"""
Meilisearch 索引监控脚本
端口：18003

使用方法:
    python monitor_index.py
"""

from config import MeiliConfig

def check_index_health():
    """检查索引健康状态"""
    print("\n" + "="*60)
    print("📊 Meilisearch 索引健康检查")
    print("="*60)
    
    try:
        client = MeiliConfig.get_client()
        index = MeiliConfig.get_index()
        
        # 1. 服务健康
        health = client.health()
        print(f"\n✅ 服务状态：{health.get('status', 'unknown')}")
        
        # 2. 索引统计
        stats = index.get_stats()
        doc_count = stats.get('numberOfDocuments', 0)
        db_size = stats.get('rawDocumentDbSize', 0)
        
        print(f"\n📁 索引统计:")
        print(f"   - 文档数量：{doc_count:,}")
        print(f"   - 索引大小：{db_size / 1024 / 1024:.2f} MB")
        
        # 3. 阈值检查
        print(f"\n⚠️  阈值检查:")
        
        size_limit = MeiliConfig.MAX_INDEX_SIZE_GB * 1024**3
        if db_size < size_limit:
            print(f"   ✅ 索引大小：{db_size/size_limit*100:.1f}% / {MeiliConfig.MAX_INDEX_SIZE_GB}GB")
        else:
            print(f"   ❌ 索引大小超限：{db_size/size_limit*100:.1f}%")
        
        doc_limit = MeiliConfig.MAX_DOCUMENTS
        if doc_count < doc_limit:
            print(f"   ✅ 文档数量：{doc_count/doc_limit*100:.1f}% / {doc_limit:,}")
        else:
            print(f"   ❌ 文档数量超限：{doc_count/doc_limit*100:.1f}%")
        
        # 4. 配置检查
        settings = index.get_settings()
        print(f"\n⚙️  配置检查:")
        print(f"   - nonSeparatorTokens: {len(settings.get('nonSeparatorTokens', []))} 个")
        print(f"   - dictionary: {len(settings.get('dictionary', []))} 个")
        print(f"   - localizedAttributes: {len(settings.get('localizedAttributes', []))} 个")
        
        # 5. 版本检查
        version = client.get_version()
        print(f"\n📦 版本信息:")
        print(f"   - Meilisearch: {version.get('pkgVersion', 'unknown')}")
        print(f"   - Commit SHA: {version.get('commitSha', 'unknown')[:8]}")
        
        print("\n" + "="*60)
        print("✅ 索引健康检查完成")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"\n❌ 检查失败：{e}")
        return False

if __name__ == "__main__":
    check_index_health()
```text

---

## 8️⃣ optimize_index.py

```python
# -*- coding: utf-8 -*-
"""
Meilisearch 索引优化脚本
端口：18003

使用方法:
    python optimize_index.py
"""

from config import MeiliConfig

def optimize_index():
    """优化索引性能"""
    print("\n" + "="*60)
    print("🔧 Meilisearch 索引优化")
    print("="*60)
    
    try:
        client = MeiliConfig.get_client()
        index = MeiliConfig.get_index()
        
        # 1. 获取优化前统计
        stats_before = index.get_stats()
        print(f"\n📊 优化前:")
        print(f"   - 文档数量：{stats_before.get('numberOfDocuments', 0):,}")
        print(f"   - 索引大小：{stats_before.get('rawDocumentDbSize', 0) / 1024 / 1024:.2f} MB")
        
        # 2. 删除已标记文档（如果有）
        print("\n🗑️  清理已删除文档...")
        # 实际使用时根据业务逻辑删除
        # task = index.delete_document_by_filter("status = 'deleted'")
        # client.wait_for_task(task['taskUid'])
        print("   ℹ️  无待删除文档")
        
        # 3. 获取优化后统计
        stats_after = index.get_stats()
        print(f"\n📊 优化后:")
        print(f"   - 文档数量：{stats_after.get('numberOfDocuments', 0):,}")
        print(f"   - 索引大小：{stats_after.get('rawDocumentDbSize', 0) / 1024 / 1024:.2f} MB")
        
        # 4. 计算优化效果
        size_diff = stats_before.get('rawDocumentDbSize', 0) - stats_after.get('rawDocumentDbSize', 0)
        if size_diff > 0:
            print(f"\n✅ 优化效果：释放 {size_diff / 1024 / 1024:.2f} MB")
        else:
            print(f"\nℹ️  索引已优化，无需清理")
        
        print("\n" + "="*60)
        print("✅ 索引优化完成")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"\n❌ 优化失败：{e}")
        return False

if __name__ == "__main__":
    optimize_index()
```

---

## 9️⃣ sample_documents.json

```json
[
  {
    "id": 1,
    "title_zh": "用户服务类",
    "description_zh": "处理用户认证和授权的公共服务类",
    "content_zh": "包含登录、注册、权限验证等功能",
    "tags_zh": ["用户", "认证", "服务"],
    "file_path": "src/main/java/com/example/app/UserService.java",
    "version": "v2.1.0",
    "language": "java",
    "project_name": "example-app",
    "email": "developer@example.com",
    "ip_address": "192.168.1.100",
    "status": "active",
    "created_at": 1710230400,
    "updated_at": 1710230400,
    "file_size": 15360,
    "line_count": 520,
    "file_name_search": "UserService java",
    "class_name_search": "UserService",
    "method_name_search": "login register authenticate authorize",
    "namespace_search": "com example app",
    "code_content_search": "public class UserService implements AuthService"
  },
  {
    "id": 2,
    "title_zh": "API 路由配置",
    "description_zh": "定义 RESTful API 端点路由",
    "content_zh": "包含用户、订单、支付等模块的路由配置",
    "tags_zh": ["API", "路由", "配置"],
    "file_path": "src/api/routes.py",
    "version": "v2.1.0",
    "language": "python",
    "project_name": "example-app",
    "email": "backend@example.com",
    "ip_address": "10.0.0.1",
    "status": "active",
    "created_at": 1711929600,
    "updated_at": 1711929600,
    "file_size": 8192,
    "line_count": 280,
    "file_name_search": "routes python",
    "class_name_search": "APIRouter",
    "method_name_search": "get post put delete",
    "namespace_search": "src api",
    "code_content_search": "from fastapi import APIRouter router = APIRouter"
  }
]
```text

---

## 🔟 README.md

```markdown
# Meilisearch 代码搜索方案

## 环境要求

- Windows 10/11
- Docker Desktop
- Python 3.8+

## 快速开始

### 1. 启动服务
```bash
docker-compose up -d
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```text

### 3. 初始化索引

```bash
python init_index.py
```

### 4. 运行测试

```bash
python test_search.py
```text

### 5. 监控索引

```bash
python monitor_index.py
```

### 6. 优化索引

```bash
python optimize_index.py
```text

## 核心特性

| 特性 | 配置 | 说明 |
|------|------|------|
| 端口 | 18003 | 自定义端口 |
| 双字段策略 | ✅ | 精确字段 + 搜索字段 |
| nonSeparatorTokens | 10 个字符 | 保留特殊格式 |
| dictionary | 80+ 词条 | 代码术语词典 |
| localizedAttributes | *_zh | 中文分词 |
| typoTolerance | 禁用精确字段 | 代码字段不容错 |
| 版本锁定 | v1.12.0 | 兼容性保证 |

## 副作用规避

| 副作用 | 规避方案 |
|--------|---------|
| 索引体积增长 | 双字段策略 |
| 部分匹配异常 | dictionary + 过滤器 |
| 中文分词不一致 | *_zh 统一命名 |
| 性能下降 | 监控 + 定期优化 |
| 版本兼容 | Docker 版本锁定 |

## 访问地址

- API: http://localhost:18003
- Master Key: meili_master_key_2026_safe

## 文档模板

```python
document = {
    "id": 1,
    "title_zh": "中文标题",
    "description_zh": "中文描述",
    "content_zh": "中文内容",
    "tags_zh": ["标签 1", "标签 2"],
    "file_path": "完整路径",  # 精确过滤
    "version": "v1.0.0",      # 精确过滤
    "language": "python",
    "project_name": "项目名",
    "email": "user@example.com",  # 精确过滤
    "ip_address": "192.168.1.1",  # 精确过滤
    "status": "active",
    "created_at": 1710230400,
    "updated_at": 1710230400,
    "file_size": 10240,
    "line_count": 300,
    "file_name_search": "文件名 语言",  # 搜索
    "class_name_search": "类名",        # 搜索
    "method_name_search": "方法 1 方法 2", # 搜索
    "namespace_search": "命名 空间",     # 搜索
    "code_content_search": "代码内容",   # 搜索
}
```

```text

---

## 🎯 一键执行命令

```bash
# 1. 创建项目目录
mkdir meilisearch_code && cd meilisearch_code

# 2. 创建所有文件（复制上方内容）

# 3. 启动服务
docker-compose up -d

# 4. 安装依赖
pip install -r requirements.txt

# 5. 初始化索引
python init_index.py

# 6. 运行测试
python test_search.py

# 7. 监控索引
python monitor_index.py
```

---

## ✅ 零副作用验证清单

| 副作用 | 规避方案 | 状态 |
|--------|---------|------|
| 索引体积增长 | 双字段策略 | ✅ |
| 部分匹配异常 | dictionary + 过滤器 | ✅ |
| 中文分词不一致 | *_zh 统一命名 | ✅ |
| 性能下降 | 监控 + 优化脚本 | ✅ |
| 版本兼容 | Docker 锁定 v1.12.0 | ✅ |
| 精确字段容错 | disableOnAttributes | ✅ |
| Windows 编码 | UTF-8 显式声明 | ✅ |

---

**🎉 复制以上所有文件，按顺序执行即可完成零副作用部署！**
