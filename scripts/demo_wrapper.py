#!/usr/bin/env python3
"""包装层核心功能演示脚本（详细输出版）。

演示内容：
1) 健康检查（/health）
2) 文本嵌入 + 缓存命中对比（/v1/embeddings）
3) 批量上传记忆（/api/v1/memories）
4) 记忆搜索（vector / keyword / hybrid）

运行示例：
    uv run python scripts/demo_wrapper.py
    uv run python scripts/demo_wrapper.py --base-url http://localhost:17999 --pretty-json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class DemoConfig:
    base_url: str
    timeout: float
    pretty_json: bool
    search_limit: int
    search_threshold: float


def _section(title: str) -> None:
    print("\n" + "=" * 88)
    print(f"🧪 {title}")
    print("=" * 88)


def _pretty(data: Any, pretty_json: bool = True, max_len: int = 1600) -> str:
    text = json.dumps(data, ensure_ascii=False, indent=2) if pretty_json else json.dumps(data, ensure_ascii=False)
    if len(text) <= max_len:
        return text
    return text[:max_len] + "\n...（输出过长，已截断）"


def _mask_embedding(response_data: dict[str, Any]) -> dict[str, Any]:
    copied = dict(response_data)
    items = copied.get("data")
    if isinstance(items, list) and items:
        cloned_items: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                cloned_items.append({"raw": str(item)})
                continue
            cloned = dict(item)
            emb = cloned.get("embedding")
            if isinstance(emb, list):
                cloned["embedding"] = {
                    "dimension": len(emb),
                    "preview": emb[:8],
                }
            cloned_items.append(cloned)
        copied["data"] = cloned_items
    return copied


async def step_health(client: httpx.AsyncClient, cfg: DemoConfig) -> bool:
    _section("步骤 1/4：健康检查")
    url = f"{cfg.base_url}/health"
    print(f"➡️  请求: GET {url}")

    try:
        start = time.perf_counter()
        resp = await client.get(url)
        elapsed_ms = (time.perf_counter() - start) * 1000
    except Exception as exc:
        print(f"❌ 请求失败: {exc}")
        return False

    print(f"⬅️  响应状态: {resp.status_code} | 耗时: {elapsed_ms:.1f} ms")
    if resp.status_code != 200:
        print(f"❌ 健康检查失败，响应正文:\n{resp.text}")
        return False

    payload = resp.json()
    print("📦 响应摘要:")
    print(_pretty(payload, pretty_json=cfg.pretty_json))

    service_ok = payload.get("status") == "healthy"
    db_status = payload.get("surrealdb", {}).get("status")
    print(f"✅ 服务状态: {payload.get('status')}")
    print(f"✅ SurrealDB 状态: {db_status}")
    cache_stats = payload.get("cache_stats", {})
    if isinstance(cache_stats, dict):
        print(
            "📊 缓存统计: "
            f"size={cache_stats.get('current_size', 'N/A')}/{cache_stats.get('max_size', 'N/A')}, "
            f"hits={cache_stats.get('hits', 'N/A')}, misses={cache_stats.get('misses', 'N/A')}, "
            f"hit_rate={cache_stats.get('hit_rate', 'N/A')}"
        )

    return service_ok


async def step_embeddings_with_cache(client: httpx.AsyncClient, cfg: DemoConfig) -> bool:
    _section("步骤 2/4：文本嵌入 + 缓存命中演示")
    url = f"{cfg.base_url}/v1/embeddings"
    payload = {
        "input": "包装层服务通过缓存和连接池提升稳定性与吞吐。",
        "model": "Qwen3-Embedding-0.6B",
    }
    print(f"➡️  请求: POST {url}")
    print("📝 请求体:")
    print(_pretty(payload, pretty_json=cfg.pretty_json))

    try:
        start1 = time.perf_counter()
        resp1 = await client.post(url, json=payload)
        t1_ms = (time.perf_counter() - start1) * 1000

        start2 = time.perf_counter()
        resp2 = await client.post(url, json=payload)
        t2_ms = (time.perf_counter() - start2) * 1000
    except Exception as exc:
        print(f"❌ 请求失败: {exc}")
        return False

    print(f"⬅️  第一次状态: {resp1.status_code} | 耗时: {t1_ms:.1f} ms")
    print(f"⬅️  第二次状态: {resp2.status_code} | 耗时: {t2_ms:.1f} ms")
    if resp1.status_code != 200 or resp2.status_code != 200:
        print("❌ 嵌入请求失败")
        print(f"第一次正文: {resp1.text}")
        print(f"第二次正文: {resp2.text}")
        return False

    data1 = resp1.json()
    data2 = resp2.json()
    masked1 = _mask_embedding(data1)
    masked2 = _mask_embedding(data2)
    print("📦 第一次响应（摘要）:")
    print(_pretty(masked1, pretty_json=cfg.pretty_json))
    print("📦 第二次响应（摘要）:")
    print(_pretty(masked2, pretty_json=cfg.pretty_json))

    same = data1 == data2
    speedup = (t1_ms / t2_ms) if t2_ms > 0 else 0.0
    print(f"✅ 两次响应是否一致: {same}")
    print(f"⚡ 速度对比: 第一次/第二次 = {speedup:.2f}x")
    return True


async def step_upload_memories(client: httpx.AsyncClient, cfg: DemoConfig) -> tuple[bool, list[str]]:
    _section("步骤 3/4：批量上传记忆")
    url = f"{cfg.base_url}/api/v1/memories"
    memories: list[dict[str, Any]] = [
        {
            "content": "FastAPI 在该项目中负责包装层 HTTP API，提供统一入口。",
            "metadata": {"source": "demo", "tag": "fastapi", "priority": "high"},
        },
        {
            "content": "Wrapper 会调用 Embedding 服务将文本转换为向量，并写入 SurrealDB。",
            "metadata": {"source": "demo", "tag": "embedding", "priority": "high"},
        },
        {
            "content": "混合搜索会融合向量结果与关键词结果，兼顾语义与关键字命中。",
            "metadata": {"source": "demo", "tag": "search", "priority": "medium"},
        },
    ]
    payload = {"memories": memories}

    print(f"➡️  请求: POST {url}")
    print("📝 请求体:")
    print(_pretty(payload, pretty_json=cfg.pretty_json))

    try:
        start = time.perf_counter()
        resp = await client.post(url, json=payload)
        elapsed_ms = (time.perf_counter() - start) * 1000
    except Exception as exc:
        print(f"❌ 请求失败: {exc}")
        return False, []

    print(f"⬅️  响应状态: {resp.status_code} | 耗时: {elapsed_ms:.1f} ms")
    if resp.status_code != 200:
        print(f"❌ 上传失败:\n{resp.text}")
        return False, []

    data = resp.json()
    print("📦 响应内容:")
    print(_pretty(data, pretty_json=cfg.pretty_json))

    memory_ids = data.get("memory_ids", [])
    print(
        f"✅ 上传结果: total={data.get('total')}, success={data.get('success')}, "
        f"failed={data.get('failed')}, memory_ids={len(memory_ids)}"
    )
    return True, [str(mid) for mid in memory_ids]


async def step_search_memories(client: httpx.AsyncClient, cfg: DemoConfig) -> bool:
    _section("步骤 4/4：搜索记忆（vector / keyword / hybrid）")
    url = f"{cfg.base_url}/api/v1/memories/search"

    scenarios = [
        {
            "name": "向量搜索",
            "payload": {
                "query": "语义检索与向量匹配",
                "mode": "vector",
                "limit": cfg.search_limit,
                "threshold": cfg.search_threshold,
            },
        },
        {
            "name": "关键词搜索",
            "payload": {
                "query": "FastAPI",
                "mode": "keyword",
                "limit": cfg.search_limit,
                "threshold": cfg.search_threshold,
            },
        },
        {
            "name": "混合搜索",
            "payload": {
                "query": "embedding 搜索",
                "mode": "hybrid",
                "limit": cfg.search_limit,
                "threshold": cfg.search_threshold,
            },
        },
    ]

    all_ok = True
    for i, item in enumerate(scenarios, start=1):
        print("\n" + "-" * 88)
        print(f"🔎 子步骤 {i}/3：{item['name']}")
        print("-" * 88)
        print(f"➡️  请求: POST {url}")
        print("📝 请求体:")
        print(_pretty(item["payload"], pretty_json=cfg.pretty_json))

        try:
            start = time.perf_counter()
            resp = await client.post(url, json=item["payload"])
            elapsed_ms = (time.perf_counter() - start) * 1000
        except Exception as exc:
            print(f"❌ 请求失败: {exc}")
            all_ok = False
            continue

        print(f"⬅️  响应状态: {resp.status_code} | 耗时: {elapsed_ms:.1f} ms")
        if resp.status_code != 200:
            print(f"❌ 搜索失败:\n{resp.text}")
            all_ok = False
            continue

        data = resp.json()
        print("📦 响应内容（摘要）:")
        print(_pretty(data, pretty_json=cfg.pretty_json))
        results = data.get("results", [])
        print(f"✅ 命中数量: {data.get('total')} | 返回模式: {data.get('mode')}")
        if isinstance(results, list) and results:
            print("📌 Top 3 结果预览:")
            for idx, result in enumerate(results[:3], start=1):
                if not isinstance(result, dict):
                    print(f"  {idx}. 非结构化结果: {result}")
                    continue
                content = str(result.get("content", ""))
                preview = content[:80] + ("..." if len(content) > 80 else "")
                print(f"  {idx}. id={result.get('id')} | score={result.get('score', 'N/A')} | content={preview}")
        else:
            print("ℹ️ 当前查询没有命中结果。")

    return all_ok


def parse_args() -> DemoConfig:
    parser = argparse.ArgumentParser(description="包装层核心功能演示脚本（详细输出）")
    parser.add_argument("--base-url", default="http://localhost:17999", help="包装层服务地址")
    parser.add_argument("--timeout", type=float, default=20.0, help="HTTP 请求超时时间（秒）")
    parser.add_argument("--pretty-json", action="store_true", help="以多行缩进格式输出 JSON")
    parser.add_argument("--search-limit", type=int, default=5, help="搜索接口 limit 参数")
    parser.add_argument("--search-threshold", type=float, default=0.5, help="搜索接口 threshold 参数")
    args = parser.parse_args()

    return DemoConfig(
        base_url=args.base_url.rstrip("/"),
        timeout=args.timeout,
        pretty_json=args.pretty_json,
        search_limit=args.search_limit,
        search_threshold=args.search_threshold,
    )


async def run_demo(cfg: DemoConfig) -> int:
    print("=" * 88)
    print("🚀 包装层核心功能演示（详细输出版）")
    print("=" * 88)
    print(f"Base URL: {cfg.base_url}")
    print(f"Timeout : {cfg.timeout}s")
    print(f"JSON格式: {'pretty' if cfg.pretty_json else 'compact'}")
    print(f"搜索参数: limit={cfg.search_limit}, threshold={cfg.search_threshold}")
    print("=" * 88)

    async with httpx.AsyncClient(timeout=cfg.timeout) as client:
        ok_health = await step_health(client, cfg)
        if not ok_health:
            print("\n💥 终止：健康检查未通过，请先启动服务后重试。")
            return 1

        ok_embed = await step_embeddings_with_cache(client, cfg)
        ok_upload, _ids = await step_upload_memories(client, cfg)
        ok_search = await step_search_memories(client, cfg)

    print("\n" + "=" * 88)
    print("📋 演示结果汇总")
    print("=" * 88)
    print(f"1) 健康检查        : {'✅ 通过' if ok_health else '❌ 失败'}")
    print(f"2) 嵌入+缓存演示   : {'✅ 通过' if ok_embed else '❌ 失败'}")
    print(f"3) 批量上传记忆    : {'✅ 通过' if ok_upload else '❌ 失败'}")
    print(f"4) 三模式搜索      : {'✅ 通过' if ok_search else '❌ 失败'}")

    all_passed = ok_health and ok_embed and ok_upload and ok_search
    print("=" * 88)
    print("🎉 全部完成" if all_passed else "⚠️ 存在失败项，请根据上面的详细日志排查")
    print("=" * 88)
    return 0 if all_passed else 2


def main() -> None:
    cfg = parse_args()
    exit_code = asyncio.run(run_demo(cfg))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
