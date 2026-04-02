"""Pydantic 数据模型

从 main.py 提取的所有请求/响应模型，供各 router 模块共用。
"""

from typing import Any

from pydantic import BaseModel, Field


class EmbeddingRequest(BaseModel):
    input: str = Field(..., description="要嵌入的文本")
    model: str = Field(default="Qwen3-Embedding-0.6B", description="模型名称")


class MemoryItem(BaseModel):
    content: str = Field(..., min_length=1, description="记忆内容 (L2)")
    abstract: str | None = Field(default=None, description="摘要 (L0, ≤100字符)")
    overview: str | None = Field(default=None, description="概览 (L1, ≤500字符)")
    type: str = Field(default="general", description="记忆类型")
    tags: list[str] = Field(default_factory=list, description="标签列表")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元数据")
    project_id: str = Field(default="global", description="项目ID")
    source: str = Field(default="api", description="来源")
    source_id: str | None = Field(default=None, description="来源ID")
    local_id: str | None = Field(default=None, description="插件端本地ID (ULID)")
    source_timestamp: str | None = Field(default=None, description="来源时间戳")
    classification_confidence: float | None = Field(default=None, description="分类置信度")


class MemoryUploadRequest(BaseModel):
    memories: list[MemoryItem] = Field(..., description="记忆列表")
    tenant_id: str = Field(default="default", description="租户ID")
    auto_analyze_code: bool = Field(default=False, description="是否自动分析代码内容并持久化结果")


class MemorySearchRequest(BaseModel):
    query: str = Field(..., description="搜索查询")
    mode: str = Field(default="hybrid", description="搜索模式")
    limit: int = Field(default=10, ge=1, le=100)
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    level: int = Field(default=2, ge=0, le=2, description="返回层级: 0=abstract, 1=abstract+overview, 2=full")
    tenant_id: str = Field(default="default", description="租户ID")
    code_filter: dict[str, Any] | None = Field(
        default=None, description="代码过滤条件: {language: str, min_complexity: int}"
    )


class RelationCreateRequest(BaseModel):
    from_id: str = Field(..., description="源记忆 ID")
    to_id: str = Field(..., description="目标记忆 ID")
    relationship_type: str = Field(default="related", description="关系类型")
    weight: float = Field(default=0.5, ge=0.0, le=1.0, description="关系权重")
    tenant_id: str = Field(default="default", description="租户ID")
    description: str | None = Field(default=None, description="关系描述")


class RelationQueryRequest(BaseModel):
    direction: str = Field(default="both", description="查询方向 (outgoing/incoming/both)")
    relationship_type: str | None = Field(default=None, description="按关系类型过滤")
    tenant_id: str = Field(default="default", description="租户ID")
    limit: int = Field(default=50, ge=1, le=200)


class GraphTraversalRequest(BaseModel):
    depth: int = Field(default=1, ge=1, le=3, description="遍历深度")
    relationship_type: str | None = Field(default=None, description="按关系类型过滤")
    tenant_id: str = Field(default="default", description="租户ID")
    limit: int = Field(default=20, ge=1, le=100)


# ==================== Sync Data Models (Phase B) ====================


class SyncFingerprint(BaseModel):
    """文件指纹模型，用于增量同步"""

    path: str = Field(..., description="文件路径")
    mtime: int = Field(..., description="修改时间戳（毫秒）")
    hash: str = Field(..., description="文件内容哈希（MD5）")
    source_id: str = Field(..., description="记忆唯一标识")


class SyncPreviewRequest(BaseModel):
    """同步预览请求"""

    fingerprints: list[SyncFingerprint] = Field(..., description="本地文件指纹列表")
    tenant_id: str = Field(default="default", description="租户ID")


class SyncPreviewResponse(BaseModel):
    """同步预览响应"""

    synced: int = Field(default=0, description="成功同步数量")
    to_upload: list[dict] = Field(default_factory=list, description="需要上传的条目")
    to_delete: list[str] = Field(default_factory=list, description="需要删除的source_id列表")
    conflicts: list[dict] = Field(default_factory=list, description="冲突列表")


class SyncIncrementalRequest(BaseModel):
    """增量同步请求（已弃用，请使用 SyncPreviewRequest）"""

    fingerprints: list[SyncFingerprint] = Field(..., description="本地文件指纹列表")
    tenant_id: str = Field(default="default", description="租户ID")


class SyncIncrementalResponse(BaseModel):
    """增量同步响应（已弃用，请使用 SyncPreviewResponse）"""

    synced: int = Field(default=0, description="成功同步数量")
    to_upload: list[dict] = Field(default_factory=list, description="需要上传的条目")
    to_delete: list[str] = Field(default_factory=list, description="需要删除的source_id列表")
    conflicts: list[dict] = Field(default_factory=list, description="冲突列表")


class SyncFullRequest(BaseModel):
    """全量同步请求"""

    memories: list[MemoryItem] = Field(..., description="记忆列表")
    tenant_id: str = Field(default="default", description="租户ID")


class SyncFullResponse(BaseModel):
    total: int = Field(..., description="总数")
    success: int = Field(..., description="成功数")
    failed: int = Field(..., description="失败数")
    updated: int = Field(default=0, description="更新数")
    skipped: list[dict] = Field(default_factory=list, description="去重跳过的条目")
    errors: list[str] = Field(default_factory=list, description="错误列表")


class ConflictResolutionRequest(BaseModel):
    """冲突解决请求"""

    resolution: str = Field(..., description="解决策略: use_local | use_remote | keep_both")
    tenant_id: str = Field(default="default", description="租户ID")


# ==================== Access Log Models ====================


class AccessLogEntry(BaseModel):
    entry_id: str = Field(..., description="记忆ID")
    timestamp: str = Field(..., description="访问时间戳")
    type: str = Field(default="read", description="访问类型")


class AccessLogRequest(BaseModel):
    entries: list[AccessLogEntry] = Field(..., description="访问日志条目列表")
    tenant_id: str = Field(default="default", description="租户ID")
