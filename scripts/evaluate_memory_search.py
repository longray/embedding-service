#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
import uuid
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class MemoryItem:
    memory_id: str
    intent_id: str
    content: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class QueryCase:
    query_id: str
    mode_hint: str
    text: str
    relevant_intents: set[str]
    should_find: bool
    difficulty: str


@dataclass
class QueryEval:
    query_id: str
    mode: str
    should_find: bool
    total_returned: int
    eval_returned: int
    relevant_in_topk: int
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    hit: bool
    non_eval_in_topk: int
    topk_preview: list[dict[str, Any]]
    difficulty: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估 memory search 准确性")
    parser.add_argument("--base-url", default="http://localhost:17999", help="wrapper 服务地址")
    parser.add_argument("--topk", type=int, default=5, help="评估 top-k")
    parser.add_argument("--threshold", type=float, default=0.7, help="兼容参数：统一阈值（若未指定分模式阈值则使用）")
    parser.add_argument("--vector-threshold", type=float, default=None, help="vector 模式阈值")
    parser.add_argument("--hybrid-threshold", type=float, default=None, help="hybrid 模式阈值")
    parser.add_argument("--keyword-threshold", type=float, default=0.0, help="keyword 模式阈值（保留占位）")
    parser.add_argument("--seed", type=int, default=20260310, help="随机种子")
    parser.add_argument("--pretty", action="store_true", help="打印详细 JSON")
    parser.add_argument("--save-report", default="", help="保存报告到指定路径（json）")
    parser.add_argument("--gate-mode", default="hybrid", choices=["vector", "keyword", "hybrid"], help="门禁评估模式")
    parser.add_argument("--min-hit-rate", type=float, default=0.95, help="门禁：should_find 最低命中率")
    parser.add_argument("--max-fp-rate", type=float, default=0.01, help="门禁：should_not_find 最高误报率")
    parser.add_argument("--enforce-gate", action="store_true", help="启用门禁，不达标时返回非0")
    parser.add_argument("--enforce-layered-gate", action="store_true", help="启用分层门禁（推荐）")
    parser.add_argument("--keyword-min-hit", type=float, default=0.965)
    parser.add_argument("--vector-min-hit", type=float, default=0.94)
    parser.add_argument("--hybrid-min-hit", type=float, default=0.96)
    parser.add_argument("--keyword-max-fp", type=float, default=0.008)
    parser.add_argument("--vector-max-fp", type=float, default=0.012)
    parser.add_argument("--hybrid-max-fp", type=float, default=0.01)
    parser.add_argument("--easy-min-hit", type=float, default=0.98)
    parser.add_argument("--medium-min-hit", type=float, default=0.95)
    parser.add_argument("--hard-min-hit", type=float, default=0.90)
    parser.add_argument("--hard-negative-max-fp", type=float, default=0.005)
    parser.add_argument("--keyword-weight", type=float, default=0.20)
    parser.add_argument("--vector-weight", type=float, default=0.25)
    parser.add_argument("--hybrid-weight", type=float, default=0.55)
    parser.add_argument("--weighted-hit-min", type=float, default=0.95)
    parser.add_argument("--weighted-fp-max", type=float, default=0.01)
    return parser.parse_args()


def build_dataset(eval_id: str, seed: int) -> tuple[list[MemoryItem], list[QueryCase]]:
    rng = random.Random(seed)  # noqa: S311

    intent_specs: list[tuple[str, list[str], list[str], list[str], list[str]]] = [
        (
            "python_api",
            ["FastAPI", "Pydantic", "异步接口"],
            [
                "使用 FastAPI + Pydantic 构建异步接口，统一请求参数校验。",
                "Python 后端以 FastAPI 提供 REST API，依赖注入用于配置管理。",
                "服务端采用 async/await 处理 I/O，提升接口吞吐。",
            ],
            [
                "Python API 框架",
                "请求参数校验",
                "异步 REST 接口",
            ],
            [
                "FastAPi",
                "Pydantric",
                "异步i/o接口",
            ],
        ),
        (
            "react_ui",
            ["React", "组件状态", "hooks"],
            [
                "React 组件通过 hooks 管理状态，避免类组件复杂生命周期。",
                "前端界面使用函数组件与 useEffect 做数据同步。",
                "UI 交互通过受控组件和状态提升保证一致性。",
            ],
            ["前端组件状态", "Hook 状态管理", "函数组件副作用"],
            ["Recat hooks", "组件state提升", "useEfect"],
        ),
        (
            "surreal_vector",
            ["SurrealDB", "向量检索", "cosine"],
            [
                "SurrealDB 支持向量相似度检索，常见使用 cosine 相似度。",
                "记忆系统将文本 embedding 存入 SurrealDB 以支持语义搜索。",
                "向量检索通常配合阈值过滤，避免低相关噪声结果。",
            ],
            ["语义向量搜索", "相似度检索", "embedding 检索"],
            ["Surreal 向量", "cosin 相似度", "embeding 搜索"],
        ),
        (
            "docker_ops",
            ["Docker", "Compose", "容器编排"],
            [
                "Docker Compose 可以一键启动多服务开发环境。",
                "容器化部署通过镜像统一运行环境，减少机器差异。",
                "编排配置中应显式声明依赖顺序和健康检查。",
            ],
            ["容器编排", "镜像部署", "多服务启动"],
            ["Docer compose", "容器依赖顺序", "镜像环境一致"],
        ),
        (
            "k8s_deploy",
            ["Kubernetes", "Deployment", "HPA"],
            [
                "Kubernetes Deployment 负责副本声明与滚动更新。",
                "HPA 根据 CPU 或自定义指标自动扩缩容。",
                "服务暴露常配合 Service 与 Ingress 实现流量入口。",
            ],
            ["自动扩缩容", "滚动更新", "集群服务入口"],
            ["Kubernets 副本", "HPA扩容", "ingres 流量"],
        ),
        (
            "finance_report",
            ["财务报表", "现金流", "利润率"],
            [
                "财务分析关注利润率、毛利率和现金流健康度。",
                "季度报表中经营现金流为正通常代表主营业务稳定。",
                "成本控制和收入增长共同影响净利润变化趋势。",
            ],
            ["经营现金流", "利润率分析", "财务健康度"],
            ["现金硫", "净利润趋势", "毛利分析"],
        ),
        (
            "medical_diag",
            ["医学影像", "诊断辅助", "临床"],
            [
                "医学影像辅助诊断用于提高早期筛查效率。",
                "临床决策支持系统应提供可解释依据以辅助医生。",
                "模型在医疗场景需关注误诊成本和召回率平衡。",
            ],
            ["临床诊断辅助", "影像筛查", "医疗模型召回"],
            ["影像诊段", "临床可解释", "误诊成本"],
        ),
        (
            "nlp_rag",
            ["RAG", "检索增强", "上下文"],
            [
                "RAG 先检索再生成，用外部知识增强回答可靠性。",
                "检索阶段需要高召回，生成阶段需要上下文压缩。",
                "向量检索和关键词检索融合可提升复杂问答效果。",
            ],
            ["先检索后生成", "外部知识增强", "检索召回"],
            ["RAG问答", "上下闻压缩", "检索增強"],
        ),
        (
            "java_backend",
            ["Spring", "事务", "微服务"],
            [
                "Spring Boot 常用于构建企业级微服务后端。",
                "事务边界管理可降低并发写入时的数据不一致风险。",
                "服务治理需要注册发现、限流和链路追踪协同。",
            ],
            ["企业级后端", "事务一致性", "服务治理"],
            ["Sprng 事务", "微服务限硫", "链路追棕"],
        ),
        (
            "security_auth",
            ["JWT", "鉴权", "权限控制"],
            [
                "JWT 常用于无状态鉴权，需设置合理过期时间。",
                "权限控制应区分 read/write/admin 等细粒度能力。",
                "安全设计中应记录审计日志以追踪关键操作。",
            ],
            ["无状态鉴权", "访问控制", "审计日志"],
            ["JWt 过期", "权限颗粒度", "审记日志"],
        ),
        (
            "iot_sensor",
            ["传感器", "时序数据", "边缘计算"],
            [
                "IoT 设备持续上报时序数据，需要高吞吐写入。",
                "边缘计算可在本地完成预处理，降低云端压力。",
                "传感器数据质量需关注漂移与缺失值修复。",
            ],
            ["边缘预处理", "时序上报", "设备数据质量"],
            ["IoT 漂移修负", "时序写入", "边缘计蒜"],
        ),
        (
            "education_ai",
            ["个性化学习", "题目推荐", "学习路径"],
            [
                "教育推荐系统根据学习行为生成个性化学习路径。",
                "题目推荐需平衡难度、知识点覆盖和反馈及时性。",
                "AI 助教可基于错题模式提供针对性讲解。",
            ],
            ["学习路径推荐", "知识点覆盖", "错题讲解"],
            ["学习路经", "题目推介", "错题摸式"],
        ),
        # ========================== v2.3.0 Polyglot 搜索架构测试 ==========================
        (
            "date_event",
            ["2026-03-11", "日期搜索", "tokenizer"],
            [
                "2026-03-11 修复了 SurrealDB 日期搜索的 tokenizer 问题，确认 class tokenizer 无法正确处理连字符日期格式。",
                "2026-02-27 完成了 combat_tools.py 常量拼写错误修复，涉及 DEFAULT_WEAPON_DAMAGE_MAX 等常量。",
                "2026-03-12 上线 Polyglot 搜索架构 v2.3.0 版本，Meilisearch 接管全文搜索功能。",
            ],
            ["日期格式修复", "常量拼写修复", "搜索架构升级"],
            ["tokeniser日期", "拼写错物", "v2.3升級"],
        ),
        (
            "chinese_nlp",
            ["中文分词", "情感分析", "命名实体"],
            [
                "中文自然语言处理需要专用分词器支持，jieba 和 charabia 是常用的中文分词工具。",
                "情感分析任务中否定词和转折词对极性判断影响很大，需要特殊处理双重否定。",
                "命名实体识别在中文文本中需处理嵌套实体和边界模糊问题，常用 CRF 或 BERT 方法。",
            ],
            ["分词工具", "极性判断", "实体边界"],
            ["中文份词", "感情分柝", "命名实休"],
        ),
        (
            "code_ref",
            ["memory_manager.py", "meili_client.py", "MeilisearchConfig"],
            [
                "memory_manager.py 模块负责协调 embedding 服务和数据库操作，支持双写同步和搜索路由。",
                "meili_client.py 提供异步 Meilisearch 客户端，支持 CJK 中文分词和日期精确匹配。",
                "config.py 中的 MeilisearchConfig 类定义了 Meilisearch 连接参数、索引名称和超时配置。",
            ],
            ["记忆管理模块", "搜索客户端", "配置类"],
            ["memory_maneger", "meilli_client", "MeiliConfig"],
        ),
        (
            "version_release",
            ["v2.3.0", "SurrealDB 3.0.1", "Polyglot"],
            [
                "v2.3.0 引入了 Polyglot 搜索架构，由 Meilisearch 负责全文搜索，SurrealDB 负责向量和图操作。",
                "SurrealDB 3.0.1 存在多个 FTS bug 包括 issue 7014 和 7015，因此选择 Meilisearch 替代全文搜索。",
                "从 v2.2.1 的 class tokenizer 方案迁移到 v2.3.0 的 Polyglot 架构，彻底解决日期分词问题。",
            ],
            ["搜索架构版本", "数据库缺陷", "架构迁移"],
            ["v2.3Polyglot", "SurealDB bug", "tokenisor方案"],
        ),
        (
            "mixed_lang",
            ["FastAPI structlog", "Docker Compose", "pytest-asyncio"],
            [
                "在 FastAPI 项目中使用 structlog 进行结构化日志记录，比标准 logging 模块更清晰直观。",
                "Docker Compose 配置中需要为 Meilisearch 容器设置 MEILI_MASTER_KEY 环境变量和数据卷。",
                "pytest-asyncio 框架支持异步测试函数，适合测试 httpx.AsyncClient 发起的 HTTP 请求。",
            ],
            ["结构化日志", "容器环境变量", "异步测试框架"],
            ["Fastapi struclog", "Docker Meilisearh", "pytest异步client"],
        ),
    ]

    items: list[MemoryItem] = []
    intent_primary_content: dict[str, str] = {}
    for intent_id, keywords, base_texts, _medium_alias, _hard_alias in intent_specs:
        for idx, base in enumerate(base_texts, start=1):
            extra = keywords[(idx - 1) % len(keywords)]
            memory_id = f"{intent_id}_{idx}"
            content = f"{base} 关键线索:{extra} 评估批次:{eval_id} 样本:{memory_id}"
            metadata = {
                "eval_id": eval_id,
                "intent_id": intent_id,
                "sample_id": memory_id,
                "keywords": keywords,
            }
            items.append(MemoryItem(memory_id=memory_id, intent_id=intent_id, content=content, metadata=metadata))
            if idx == 1:
                intent_primary_content[intent_id] = content

    should_find_queries: list[QueryCase] = []
    query_counter = 0
    for intent_id, keywords, _texts, _medium_alias, hard_alias in intent_specs:
        query_counter += 1
        should_find_queries.append(
            QueryCase(
                query_id=f"q_find_{query_counter}",
                mode_hint="keyword",
                text=keywords[0],
                relevant_intents={intent_id},
                should_find=True,
                difficulty="easy",
            )
        )
        query_counter += 1
        should_find_queries.append(
            QueryCase(
                query_id=f"q_find_{query_counter}",
                mode_hint="keyword",
                text=keywords[1],
                relevant_intents={intent_id},
                should_find=True,
                difficulty="easy",
            )
        )
        query_counter += 1
        should_find_queries.append(
            QueryCase(
                query_id=f"q_find_{query_counter}",
                mode_hint="vector",
                text=intent_primary_content[intent_id],
                relevant_intents={intent_id},
                should_find=True,
                difficulty="medium",
            )
        )
        query_counter += 1
        should_find_queries.append(
            QueryCase(
                query_id=f"q_find_{query_counter}",
                mode_hint="hybrid",
                text=f"{keywords[0]} {hard_alias[0]} {hard_alias[1]}",
                relevant_intents={intent_id},
                should_find=True,
                difficulty="hard",
            )
        )

    keyword_negatives = [
        "脉冲星时钟漂移",
        "古典吉他尼龙弦",
        "海洋潮汐叶轮腐蚀",
        "考古陶片断代",
        "围棋官子手筋",
        "油画罩染修复",
        "航海六分仪校准",
        "火山岩浆黏度",
        # v2.3.0 Polyglot 负面样例
        "量子纠缠退相干",
        "2025-12-25 圣诞节",
        "nonexistent_module.py",
    ]
    vector_negatives = [
        "zzqxv kptra nmdwu",
        "qvrol xpmte aznkk",
        "nvkqe tlrpa wzmou",
        "pqlzx mvnrt aokwe",
        "xxzzq yywwv vvttm",
        "rqplm znxte avkro",
        # v2.3.0 Polyglot 负面样例
        "qwmzx ypnkr vvtba",
        "zzqqw xxpmv rrtnn",
    ]
    hybrid_negatives = [
        "zzqxv 吉他弦 航海象限",
        "pulsar qwerty magma",
        "ancient lyre niello",
        "xvnmq tide violin",
        # v2.3.0 Polyglot 负面样例
        "v99.9.9 alien architecture",
        "2099-01-01 未来事件",
    ]

    should_not_find_queries: list[QueryCase] = []
    not_counter = 0
    for text in keyword_negatives:
        not_counter += 1
        should_not_find_queries.append(
            QueryCase(
                query_id=f"q_not_{not_counter}",
                mode_hint="keyword",
                text=text,
                relevant_intents=set(),
                should_find=False,
                difficulty="hard_negative",
            )
        )
    for text in vector_negatives:
        not_counter += 1
        should_not_find_queries.append(
            QueryCase(
                query_id=f"q_not_{not_counter}",
                mode_hint="vector",
                text=text,
                relevant_intents=set(),
                should_find=False,
                difficulty="hard_negative",
            )
        )
    for text in hybrid_negatives:
        not_counter += 1
        should_not_find_queries.append(
            QueryCase(
                query_id=f"q_not_{not_counter}",
                mode_hint="hybrid",
                text=text,
                relevant_intents=set(),
                should_find=False,
                difficulty="hard_negative",
            )
        )

    all_queries = should_find_queries + should_not_find_queries
    rng.shuffle(all_queries)
    return items, all_queries


async def ensure_health(client: httpx.AsyncClient, base_url: str) -> None:
    """检查 wrapper 服务及 Meilisearch 健康状态（v2.3.0 Polyglot）。"""
    response = await client.get(f"{base_url}/health")
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") != "healthy":
        raise RuntimeError(f"wrapper 健康检查失败: {payload}")
    # v2.3.0: 检查 Meilisearch 状态
    meili_status = payload.get("meilisearch", {})
    if isinstance(meili_status, dict):
        status = meili_status.get("status", "unknown")
    else:
        status = str(meili_status) if meili_status else "unknown"
    if status not in ("available", "healthy"):
        print(f"⚠️  Meilisearch 状态: {status}（关键词搜索将使用 SurrealDB BM25 降级）")
    else:
        print(f"✅ Meilisearch 状态: {status}")

async def upload_eval_data(client: httpx.AsyncClient, base_url: str, items: list[MemoryItem]) -> dict[str, Any]:
    payload = {"memories": [{"content": item.content, "metadata": item.metadata} for item in items]}
    response = await client.post(f"{base_url}/api/v1/memories", json=payload)
    response.raise_for_status()
    data = response.json()
    return data


def _extract_eval_hit(record: dict[str, Any], eval_id: str) -> bool:
    metadata = record.get("metadata", {})
    if not isinstance(metadata, dict):
        return False
    return metadata.get("eval_id") == eval_id


def _extract_intent(record: dict[str, Any]) -> str | None:
    metadata = record.get("metadata", {})
    if not isinstance(metadata, dict):
        return None
    raw = metadata.get("intent_id")
    return str(raw) if raw is not None else None


def evaluate_one_query(
    case: QueryCase,
    mode: str,
    response_payload: dict[str, Any],
    eval_id: str,
    topk: int,
) -> QueryEval:
    results = response_payload.get("results", [])
    if not isinstance(results, list):
        results = []

    top_results = results[:topk]
    eval_top = [r for r in top_results if isinstance(r, dict) and _extract_eval_hit(r, eval_id)]
    eval_returned = len([r for r in results if isinstance(r, dict) and _extract_eval_hit(r, eval_id)])

    relevant_flags: list[bool] = []
    relevant_in_topk = 0
    reciprocal_rank = 0.0
    for idx, record in enumerate(eval_top, start=1):
        intent = _extract_intent(record)
        is_relevant = intent in case.relevant_intents
        relevant_flags.append(is_relevant)
        if is_relevant:
            relevant_in_topk += 1
            if reciprocal_rank == 0.0:
                reciprocal_rank = 1.0 / idx

    non_eval_in_topk = len(top_results) - len(eval_top)

    if case.should_find:
        precision_at_k = relevant_in_topk / topk
        denom = len(case.relevant_intents) * 3
        recall_at_k = relevant_in_topk / denom if denom > 0 else 0.0
        hit = relevant_in_topk > 0
    else:
        precision_at_k = 1.0 if relevant_in_topk == 0 else 0.0
        recall_at_k = 1.0 if relevant_in_topk == 0 else 0.0
        hit = relevant_in_topk == 0 and eval_returned == 0

    topk_preview: list[dict[str, Any]] = []
    for record in top_results[:3]:
        if not isinstance(record, dict):
            continue
        topk_preview.append(
            {
                "id": record.get("id"),
                "intent": _extract_intent(record),
                "score": record.get("score", None),
                "content": str(record.get("content", ""))[:90],
                "is_eval": _extract_eval_hit(record, eval_id),
            }
        )

    return QueryEval(
        query_id=case.query_id,
        mode=mode,
        should_find=case.should_find,
        total_returned=len(results),
        eval_returned=eval_returned,
        relevant_in_topk=relevant_in_topk,
        precision_at_k=precision_at_k,
        recall_at_k=recall_at_k,
        reciprocal_rank=reciprocal_rank,
        hit=hit,
        non_eval_in_topk=non_eval_in_topk,
        topk_preview=topk_preview,
        difficulty=case.difficulty,
    )


async def run_queries(
    client: httpx.AsyncClient,
    base_url: str,
    cases: list[QueryCase],
    modes: list[str],
    mode_thresholds: dict[str, float],
    topk: int,
    eval_id: str,
) -> tuple[dict[str, list[QueryEval]], list[dict[str, Any]]]:
    by_mode: dict[str, list[QueryEval]] = {m: [] for m in modes}
    failures: list[dict[str, Any]] = []

    for mode in modes:
        for case in cases:
            if case.mode_hint not in ("any", mode) and not (
                mode == "hybrid" and case.mode_hint in {"vector", "keyword"}
            ):
                continue
            payload = {
                "query": case.text,
                "mode": mode,
                "limit": max(topk, 10),
                "threshold": mode_thresholds[mode],
                "metadata_filters": {"eval_id": eval_id},
            }
            started = time.perf_counter()
            response = await client.post(f"{base_url}/api/v1/memories/search", json=payload)
            latency_ms = (time.perf_counter() - started) * 1000
            response.raise_for_status()
            raw = response.json()

            eval_result = evaluate_one_query(case=case, mode=mode, response_payload=raw, eval_id=eval_id, topk=topk)
            by_mode[mode].append(eval_result)

            if case.should_find and not eval_result.hit:
                failures.append(
                    {
                        "kind": "miss_should_find",
                        "mode": mode,
                        "query_id": case.query_id,
                        "query": case.text,
                        "latency_ms": round(latency_ms, 2),
                        "preview": eval_result.topk_preview,
                        "total_returned": eval_result.total_returned,
                        "eval_returned": eval_result.eval_returned,
                    }
                )

            if (not case.should_find) and (eval_result.eval_returned > 0):
                failures.append(
                    {
                        "kind": "false_positive_should_not_find",
                        "mode": mode,
                        "query_id": case.query_id,
                        "query": case.text,
                        "latency_ms": round(latency_ms, 2),
                        "preview": eval_result.topk_preview,
                        "total_returned": eval_result.total_returned,
                        "eval_returned": eval_result.eval_returned,
                    }
                )

    return by_mode, failures


def summarize(mode_results: list[QueryEval]) -> dict[str, Any]:
    should_find = [r for r in mode_results if r.should_find]
    should_not_find = [r for r in mode_results if not r.should_find]

    def avg(values: list[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    hit_rate = avg([1.0 if r.hit else 0.0 for r in should_find])
    precision_at_k = avg([r.precision_at_k for r in should_find])
    recall_at_k = avg([r.recall_at_k for r in should_find])
    mrr = avg([r.reciprocal_rank for r in should_find])

    fp_eval_rate = avg([1.0 if r.eval_returned > 0 else 0.0 for r in should_not_find])
    fp_total_rate = avg([1.0 if r.total_returned > 0 else 0.0 for r in should_not_find])
    contamination = avg([r.non_eval_in_topk / 5.0 for r in mode_results])

    eval_returned_counts = [r.eval_returned for r in mode_results]
    total_returned_counts = [r.total_returned for r in mode_results]

    by_difficulty: dict[str, dict[str, float]] = {}
    for difficulty in ["easy", "medium", "hard", "hard_negative"]:
        group = [r for r in mode_results if r.difficulty == difficulty]
        if not group:
            continue
        if difficulty == "hard_negative":
            by_difficulty[difficulty] = {
                "count": float(len(group)),
                "fp_eval_rate": round(avg([1.0 if r.eval_returned > 0 else 0.0 for r in group]), 4),
            }
        else:
            by_difficulty[difficulty] = {
                "count": float(len(group)),
                "hit_rate": round(avg([1.0 if r.hit else 0.0 for r in group]), 4),
                "mrr": round(avg([r.reciprocal_rank for r in group]), 4),
            }

    return {
        "query_count": len(mode_results),
        "should_find_count": len(should_find),
        "should_not_find_count": len(should_not_find),
        "hit_rate_should_find": round(hit_rate, 4),
        "precision_at_5_should_find": round(precision_at_k, 4),
        "recall_at_5_should_find": round(recall_at_k, 4),
        "mrr_should_find": round(mrr, 4),
        "false_positive_rate_eval_should_not_find": round(fp_eval_rate, 4),
        "false_positive_rate_total_should_not_find": round(fp_total_rate, 4),
        "non_eval_contamination_at5": round(contamination, 4),
        "avg_eval_returned": round(avg([float(x) for x in eval_returned_counts]), 3),
        "avg_total_returned": round(avg([float(x) for x in total_returned_counts]), 3),
        "median_eval_returned": float(statistics.median(eval_returned_counts)) if eval_returned_counts else 0.0,
        "median_total_returned": float(statistics.median(total_returned_counts)) if total_returned_counts else 0.0,
        "difficulty_metrics": by_difficulty,
    }


def evaluate_layered_gate(report: dict[str, Any], args: argparse.Namespace) -> tuple[bool, dict[str, Any]]:
    modes = report["modes"]
    mode_rules = {
        "keyword": {"min_hit": args.keyword_min_hit, "max_fp": args.keyword_max_fp},
        "vector": {"min_hit": args.vector_min_hit, "max_fp": args.vector_max_fp},
        "hybrid": {"min_hit": args.hybrid_min_hit, "max_fp": args.hybrid_max_fp},
    }

    mode_pass = True
    mode_checks: dict[str, Any] = {}
    for mode, rule in mode_rules.items():
        m = modes.get(mode, {})
        hit = float(m.get("hit_rate_should_find", 0.0))
        fp = float(m.get("false_positive_rate_eval_should_not_find", 1.0))
        passed = hit >= rule["min_hit"] and fp <= rule["max_fp"]
        mode_checks[mode] = {
            "hit": hit,
            "fp": fp,
            "min_hit": rule["min_hit"],
            "max_fp": rule["max_fp"],
            "pass": passed,
        }
        mode_pass = mode_pass and passed

    weighted_hit = (
        args.keyword_weight * float(modes["keyword"].get("hit_rate_should_find", 0.0))
        + args.vector_weight * float(modes["vector"].get("hit_rate_should_find", 0.0))
        + args.hybrid_weight * float(modes["hybrid"].get("hit_rate_should_find", 0.0))
    )
    weighted_fp = (
        args.keyword_weight * float(modes["keyword"].get("false_positive_rate_eval_should_not_find", 1.0))
        + args.vector_weight * float(modes["vector"].get("false_positive_rate_eval_should_not_find", 1.0))
        + args.hybrid_weight * float(modes["hybrid"].get("false_positive_rate_eval_should_not_find", 1.0))
    )

    global_pass = weighted_hit >= args.weighted_hit_min and weighted_fp <= args.weighted_fp_max

    difficulty_rules = {
        "easy": args.easy_min_hit,
        "medium": args.medium_min_hit,
        "hard": args.hard_min_hit,
    }
    difficulty_checks: dict[str, Any] = {}
    difficulty_pass = True
    for mode in ["keyword", "vector", "hybrid"]:
        dm = modes.get(mode, {}).get("difficulty_metrics", {})
        for difficulty, min_hit in difficulty_rules.items():
            count = float(dm.get(difficulty, {}).get("count", 0.0))
            if count == 0:
                continue
            val = float(dm.get(difficulty, {}).get("hit_rate", 0.0))
            passed = val >= min_hit
            difficulty_checks[f"{mode}:{difficulty}"] = {
                "hit": val,
                "min_hit": min_hit,
                "pass": passed,
            }
            difficulty_pass = difficulty_pass and passed

        hard_neg_count = float(dm.get("hard_negative", {}).get("count", 0.0))
        if hard_neg_count > 0:
            hard_neg_fp = float(dm.get("hard_negative", {}).get("fp_eval_rate", 1.0))
            hard_neg_pass = hard_neg_fp <= args.hard_negative_max_fp
            difficulty_checks[f"{mode}:hard_negative"] = {
                "fp_eval": hard_neg_fp,
                "max_fp": args.hard_negative_max_fp,
                "pass": hard_neg_pass,
            }
            difficulty_pass = difficulty_pass and hard_neg_pass

    overall_pass = mode_pass and global_pass and difficulty_pass
    details = {
        "mode_checks": mode_checks,
        "weighted": {
            "hit": weighted_hit,
            "fp": weighted_fp,
            "hit_min": args.weighted_hit_min,
            "fp_max": args.weighted_fp_max,
            "pass": global_pass,
        },
        "difficulty_checks": difficulty_checks,
        "pass": overall_pass,
    }
    return overall_pass, details


def print_summary(report: dict[str, Any], pretty: bool) -> None:
    print("\n" + "=" * 96)
    print("📊 Memory Search 评估报告")
    print("=" * 96)
    print(f"eval_id: {report['eval_id']}")
    print(f"dataset_size: {report['dataset_size']} memories")
    print(f"query_size: {report['query_size']} queries")
    print(f"topk: {report['topk']}, threshold: {report['threshold']}")
    print(f"mode_thresholds: {report.get('mode_thresholds', {})}")
    print("=" * 96)

    for mode, metrics in report["modes"].items():
        print(f"\n🔹 mode={mode}")
        print(
            "  should_find 命中率={:.2%}, P@5={:.3f}, R@5={:.3f}, MRR={:.3f}".format(
                metrics["hit_rate_should_find"],
                metrics["precision_at_5_should_find"],
                metrics["recall_at_5_should_find"],
                metrics["mrr_should_find"],
            )
        )
        print(
            "  should_not_find 误报(eval)={:.2%}, 误报(total)={:.2%}".format(
                metrics["false_positive_rate_eval_should_not_find"],
                metrics["false_positive_rate_total_should_not_find"],
            )
        )
        print(
            "  污染率(non-eval@5)={:.2%}, avg_eval_returned={}, avg_total_returned={}".format(
                metrics["non_eval_contamination_at5"],
                metrics["avg_eval_returned"],
                metrics["avg_total_returned"],
            )
        )

    if report["failures"]:
        print(f"\n⚠️ 失败样例数: {len(report['failures'])}")
        for sample in report["failures"][:15]:
            print(
                f"  - {sample['kind']} | mode={sample['mode']} | {sample['query_id']} | "
                f"latency={sample['latency_ms']}ms | eval_returned={sample['eval_returned']}"
            )
    else:
        print("\n✅ 无失败样例")

    if pretty:
        print("\n🧾 详细 JSON: ")
        print(json.dumps(report, ensure_ascii=False, indent=2))


async def main_async() -> int:
    args = parse_args()
    eval_id = f"eval_{uuid.uuid4().hex[:10]}"
    items, cases = build_dataset(eval_id=eval_id, seed=args.seed)
    mode_thresholds = {
        "vector": args.vector_threshold if args.vector_threshold is not None else args.threshold,
        "keyword": args.keyword_threshold,
        "hybrid": args.hybrid_threshold if args.hybrid_threshold is not None else args.threshold,
    }

    timeout = httpx.Timeout(60.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        await ensure_health(client, args.base_url)

        upload_started = time.perf_counter()
        upload_result = await upload_eval_data(client, args.base_url, items)
        upload_ms = (time.perf_counter() - upload_started) * 1000

        modes = ["vector", "keyword", "hybrid"]
        by_mode, failures = await run_queries(
            client=client,
            base_url=args.base_url,
            cases=cases,
            modes=modes,
            mode_thresholds=mode_thresholds,
            topk=args.topk,
            eval_id=eval_id,
        )

    report = {
        "eval_id": eval_id,
        "base_url": args.base_url,
        "dataset_size": len(items),
        "query_size": len(cases),
        "topk": args.topk,
        "threshold": args.threshold,
        "mode_thresholds": mode_thresholds,
        "upload_result": upload_result,
        "upload_elapsed_ms": round(upload_ms, 2),
        "modes": {mode: summarize(results) for mode, results in by_mode.items()},
        "failures": failures,
    }

    print_summary(report, args.pretty)

    gate_failed = False
    gate_metrics = report["modes"].get(args.gate_mode, {})
    if args.enforce_gate and gate_metrics:
        hit_rate = float(gate_metrics.get("hit_rate_should_find", 0.0))
        fp_rate = float(gate_metrics.get("false_positive_rate_eval_should_not_find", 1.0))
        gate_failed = hit_rate < args.min_hit_rate or fp_rate > args.max_fp_rate
        print("\n" + "=" * 96)
        print("🚦 回归门禁检查")
        print("=" * 96)
        print(
            f"mode={args.gate_mode}, hit_rate={hit_rate:.2%} (要求>={args.min_hit_rate:.2%}), "
            f"fp_rate={fp_rate:.2%} (要求<={args.max_fp_rate:.2%})"
        )
        print("❌ 门禁失败" if gate_failed else "✅ 门禁通过")

    if args.save_report:
        with open(args.save_report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n💾 已保存报告: {args.save_report}")

    if args.enforce_layered_gate:
        layered_pass, layered_details = evaluate_layered_gate(report, args)
        print("\n" + "=" * 96)
        print("🧱 分层门禁检查")
        print("=" * 96)
        print(json.dumps(layered_details, ensure_ascii=False, indent=2))
        gate_failed = gate_failed or (not layered_pass)

    return 2 if gate_failed else 0


def main() -> None:
    code = asyncio.run(main_async())
    raise SystemExit(code)


if __name__ == "__main__":
    main()
