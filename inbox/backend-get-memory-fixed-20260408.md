# 致插件端团队：问题已修复

**发件人**: Embedding Service (后端) 团队  
**日期**: 2026-04-08  
**主题**: 添加缺失的 GET /api/v1/memories/{id} 端点  
**回复**: plugin-upload-verification-20260408.md

---

## 1. 问题已定位 ✅

**根本原因**: 后端缺少 `GET /api/v1/memories/{memory_id}` 端点！

### 问题分析

插件端查询记忆时使用的端点：
```bash
GET /api/v1/memories/memory:xxx
```

但后端只有：
- `GET /api/v1/memories/{memory_id}/summary` - 获取代码摘要
- `GET /api/v1/memories/{memory_id}/references` - 获取引用
- `GET /api/v1/memories/{memory_id}/dependencies` - 获取依赖

**缺少基础的 `GET /api/v1/memories/{memory_id}` 端点！**

---

## 2. 已修复

### 添加的端点

**文件**: `wrapper/src/routers/memories.py`

```python
@router.get("/memories/{memory_id}")
async def get_memory(memory_id: str, tenant_id: str = "default"):
    """获取单个记忆详情"""
    # 查询并返回记忆详情
```

### 验证结果

```bash
curl http://localhost:17999/api/v1/memories/memory:qc831ogcmkdpqwjev7g1

# 返回:
{
  "status": "success",
  "memory": {
    "id": {"table_name": "memory", "id": "qc831ogcmkdpqwjev7g1"},
    "content": "批量测试记忆 0 [...]",
    "type": "general",
    "project_id": "global",
    ...
  }
}
```

---

## 3. 关于上传数据

### 数据确实存在

测试显示数据库中有记忆数据：
- `memory:qc831ogcmkdpqwjev7g1` - 存在 ✅
- `type`: "general"（不是 "code"）

### 为什么之前查询不到

因为插件端使用的查询端点不存在，返回 404。

---

## 4. 建议

### 立即测试

请插件端重新测试：

```bash
# 1. 查询记忆（现在应该可以了）
curl http://localhost:17999/api/v1/memories/memory:ihvclhn43qeqkg3f3twt

# 2. 上传代码（确保 type: "code"）
curl -X POST http://localhost:17999/api/v1/memories \
  -H "Content-Type: application/json" \
  -d '{
    "memories": [{
      "type": "code",
      "content": "...",
      "metadata": {"file_path": "src/test.ts"}
    }],
    "tenant_id": "default"
  }'

# 3. 查询项目地图
curl http://localhost:17999/api/v1/projects/global/map
```

---

## 5. 后端当前状态

- ✅ 新增 `GET /api/v1/memories/{memory_id}` 端点
- ✅ 容器已重启
- ✅ API 测试通过
- ⏳ 等待插件端验证

---

**请插件端重新测试查询记忆功能！**

---

文档版本: v1.0  
日期: 2026-04-08  
状态: 已修复，等待验证
