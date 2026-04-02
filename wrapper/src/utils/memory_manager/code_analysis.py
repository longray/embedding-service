"""代码分析桥接（analyze_memory_code, _generate_code_summary）"""

import asyncio
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class CodeAnalysisMixin:
    """代码分析相关方法"""

    async def _generate_code_summary(self, memory_id: str, tenant_id: str) -> dict[str, Any] | None:
        """调用 LLM 生成代码摘要"""
        from ..config import LLMConfig

        config = LLMConfig()
        if not config.enabled:
            return None

        try:
            # 获取记忆内容和代码分析结果
            query = "SELECT content, metadata FROM $memory_id WHERE tenant_id = $tenant_id"
            result = await self._db_query(query, {"memory_id": memory_id, "tenant_id": tenant_id})
            records = self._extract_records(result)

            if not records:
                return None

            memory = records[0]
            content = memory.get("content", "")
            metadata = memory.get("metadata", {}) or {}
            code_analysis = metadata.get("code_analysis", {})

            if not code_analysis:
                return None

            # 构建 LLM 提示
            functions = code_analysis.get("functions", [])
            classes = code_analysis.get("classes", [])
            language = code_analysis.get("language", "unknown")

            prompt = f"""分析以下 {language} 代码并提供摘要：

代码内容：
```
{content[:2000]}
```

函数列表：{", ".join(f.get("name", "") for f in functions[:10])}
类列表：{", ".join(c.get("name", "") for c in classes[:10])}

请提供：
1. 一句话摘要（描述这个模块/文件的主要功能）
2. 关键函数列表（最重要的3-5个函数及其作用）
3. 代码用途（这个代码解决什么问题）

以 JSON 格式返回：
{{
    "summary": "一句话摘要",
    "key_functions": ["函数1: 作用", "函数2: 作用"],
    "purpose": "代码用途描述"
}}"""

            # 调用 LLM API
            http_pool = await self._get_http_pool()
            headers = {"Content-Type": "application/json"}
            if config.api_key:
                headers["Authorization"] = f"Bearer {config.api_key}"

            payload = {
                "model": config.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": config.max_tokens,
                "temperature": 0.3,
            }

            response = await http_pool.post(
                f"{config.endpoint}/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=config.timeout,
            )

            if response.status_code != 200:
                logger.warning("[LLM Summary] API 调用失败: %s", response.status_code)
                return None

            result_data = response.json()
            llm_content = result_data.get("choices", [{}])[0].get("message", {}).get("content", "")

            # 解析 JSON 响应
            try:
                # 尝试提取 JSON 部分
                json_start = llm_content.find("{")
                json_end = llm_content.rfind("}")
                if json_start >= 0 and json_end > json_start:
                    json_str = llm_content[json_start : json_end + 1]
                    summary_data = json.loads(json_str)
                else:
                    # 如果不是 JSON，使用原始内容作为摘要
                    summary_data = {
                        "summary": llm_content[:200],
                        "key_functions": [],
                        "purpose": "",
                    }

                # 保存到 metadata
                code_summary = {
                    "summary": summary_data.get("summary", ""),
                    "key_functions": summary_data.get("key_functions", []),
                    "purpose": summary_data.get("purpose", ""),
                    "generated_at": asyncio.get_event_loop().time(),
                    "model": config.model_name,
                }

                metadata["code_summary"] = code_summary

                # 更新数据库
                update_query = """
                    UPDATE type::record($record_id)
                    SET metadata = $metadata
                """
                await self._db_query(update_query, {"record_id": memory_id, "metadata": metadata})

                logger.info("[LLM Summary] 摘要生成完成: %s", memory_id)
                return code_summary

            except json.JSONDecodeError as e:
                logger.warning("[LLM Summary] JSON 解析失败: %s", e)
                return None

        except Exception as e:
            logger.warning("[LLM Summary] 生成失败: %s", e)
            return None

    async def analyze_memory_code(
        self, memory_id: str, tenant_id: str = "default", persist: bool = False
    ) -> dict[str, Any]:
        effective_tenant_id = tenant_id or self._default_tenant_id
        mem_ref = self._normalize_memory_id(memory_id)

        try:
            query = "SELECT content, metadata FROM type::record($id) WHERE tenant_id = $tenant_id"
            result = await self._db_query(query, {"id": mem_ref, "tenant_id": effective_tenant_id})
            records = self._extract_records(result)

            if not records:
                logger.warning("[analyze_memory_code] 记忆不存在: %s", mem_ref)
                return {}

            content = records[0].get("content", "")
            if not content:
                logger.warning("[analyze_memory_code] 内容为空: %s", mem_ref)
                return {}

            if not self._is_code_content(content):
                logger.debug("[analyze_memory_code] 非代码内容，跳过: %s", mem_ref)
                return {}

            metadata = records[0].get("metadata") or {}
            language = metadata.get("language", "")
            if not language:
                file_path = metadata.get("file_path", "")
                if file_path and "." in file_path:
                    ext = file_path.rsplit(".", 1)[-1].lower()
                    language = {
                        "py": "python",
                        "js": "javascript",
                        "ts": "typescript",
                        "java": "java",
                        "go": "go",
                        "rs": "rust",
                        "c": "c",
                        "cpp": "cpp",
                        "h": "c",
                        "html": "html",
                        "css": "css",
                        "sql": "sql",
                    }.get(ext, "")
            if not language:
                language = "python"

            analysis_result = await self.code_analyzer.analyze_code(content, language)
            analysis_dict = analysis_result.to_metadata_dict()

            if persist:
                update_sql = "UPDATE type::record($id) SET metadata.code_analysis = $code_analysis"
                await self._db_query(update_sql, {"id": mem_ref, "code_analysis": analysis_dict})
                logger.info("[analyze_memory_code] 分析结果已持久化: %s", mem_ref)

            return analysis_dict

        except Exception as e:
            logger.warning("[analyze_memory_code] 分析失败: %s - %s", mem_ref, e)
            return {}
