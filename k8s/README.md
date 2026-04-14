# Kubernetes 部署指南

OpenCode Memory Service v3.2 的 Kubernetes 部署配置。

## 目录结构

```
k8s/
├── namespace.yaml              # 命名空间
├── surrealdb-deployment.yaml   # SurrealDB 部署
├── meilisearch-deployment.yaml # Meilisearch 部署
├── wrapper-deployment.yaml     # Wrapper 服务部署
├── ingress.yaml                # Ingress 配置
├── kustomization.yaml          # Kustomize 配置
└── README.md                   # 本文档
```

## 快速开始

### 1. 安装依赖

```bash
# 安装 kubectl
# https://kubernetes.io/docs/tasks/tools/

# 安装 kustomize
# https://kubectl.docs.kubernetes.io/installation/kustomize/
```

### 2. 部署服务

```bash
# 使用 kubectl 直接部署
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/surrealdb-deployment.yaml
kubectl apply -f k8s/meilisearch-deployment.yaml
kubectl apply -f k8s/wrapper-deployment.yaml
kubectl apply -f k8s/ingress.yaml

# 或使用 kustomize
kubectl apply -k k8s/
```

### 3. 验证部署

```bash
# 查看命名空间
kubectl get namespace opencode-memory

# 查看 Pod
kubectl get pods -n opencode-memory

# 查看服务
kubectl get svc -n opencode-memory

# 查看 Ingress
kubectl get ingress -n opencode-memory
```

### 4. 访问服务

```bash
# 获取 Ingress IP
kubectl get ingress wrapper-ingress -n opencode-memory

# 测试 API
curl http://<ingress-ip>/health
```

## 配置说明

### SurrealDB

- **镜像**: `surrealdb/surrealdb:latest`
- **端口**: 8000
- **存储**: 10Gi PVC
- **资源**: 512Mi - 2Gi 内存，500m - 2000m CPU

### Meilisearch

- **镜像**: `getmeili/meilisearch:latest`
- **端口**: 7700
- **存储**: 5Gi PVC
- **资源**: 256Mi - 1Gi 内存，250m - 1000m CPU

### Wrapper

- **镜像**: `opencode-memory/wrapper:v3.2.0`
- **端口**: 18008 (HTTP), 18000 (Embedding)
- **副本**: 2
- **资源**: 1Gi - 4Gi 内存，500m - 2000m CPU

## 生产环境配置

### 1. 更新密钥

```bash
# 编辑 secret
kubectl edit secret wrapper-secret -n opencode-memory
kubectl edit secret meilisearch-secret -n opencode-memory
```

### 2. 配置 TLS

```bash
# 创建 TLS secret
kubectl create secret tls wrapper-tls \
  --cert=path/to/cert.pem \
  --key=path/to/key.pem \
  -n opencode-memory
```

### 3. 配置 Ingress

```bash
# 更新 Ingress 域名
kubectl edit ingress wrapper-ingress -n opencode-memory
```

### 4. 水平扩展

```bash
# 扩展 Wrapper 副本
kubectl scale deployment wrapper --replicas=4 -n opencode-memory
```

## 监控

### 查看日志

```bash
# 查看 Wrapper 日志
kubectl logs -f deployment/wrapper -n opencode-memory

# 查看 SurrealDB 日志
kubectl logs -f deployment/surrealdb -n opencode-memory
```

### 查看指标

```bash
# 查看资源使用
kubectl top pods -n opencode-memory

# 查看节点资源
kubectl top nodes
```

## 故障排查

### Pod 无法启动

```bash
# 查看 Pod 事件
kubectl describe pod <pod-name> -n opencode-memory

# 查看 Pod 日志
kubectl logs <pod-name> -n opencode-memory
```

### 服务无法访问

```bash
# 检查服务状态
kubectl get svc -n opencode-memory

# 检查 Endpoint
kubectl get endpoints -n opencode-memory

# 检查 Ingress
kubectl get ingress -n opencode-memory
```

### 存储问题

```bash
# 查看 PVC
kubectl get pvc -n opencode-memory

# 查看 PV
kubectl get pv
```

## 清理

```bash
# 删除所有资源
kubectl delete -k k8s/

# 或逐个删除
kubectl delete -f k8s/ingress.yaml
kubectl delete -f k8s/wrapper-deployment.yaml
kubectl delete -f k8s/meilisearch-deployment.yaml
kubectl delete -f k8s/surrealdb-deployment.yaml
kubectl delete -f k8s/namespace.yaml
```

## 参考

- [Kubernetes 文档](https://kubernetes.io/docs/)
- [Kustomize 文档](https://kubectl.docs.kubernetes.io/)
- [BACKEND-v3.2-IMPLEMENTATION.md](../docs/v3.2/BACKEND-v3.2-IMPLEMENTATION.md)
