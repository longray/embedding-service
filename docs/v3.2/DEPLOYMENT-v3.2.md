# v3.2 Docker 和部署指南

> **版本**: v3.2.0  
> **日期**: 2026-04-10  
> **状态**: 实施版  
> **服务端口**: 18008  
> **目标**: 生产环境部署和本地开发环境配置

---

## 目录

1. [部署架构](#1-部署架构)
2. [Docker 配置](#2-docker-配置)
3. [本地开发环境](#3-本地开发环境)
4. [生产环境部署](#4-生产环境部署)
5. [监控与日志](#5-监控与日志)
6. [故障排除](#6-故障排除)

---

## 1. 部署架构

### 1.1 服务组件

```
┌─────────────────────────────────────────────────────────────┐
│                      Docker Network                          │
│                                                              │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │   FastAPI App   │    │   SurrealDB     │                │
│  │   (Port 18008)  │◄──►│   (Port 8000)   │                │
│  │                 │    │                 │                │
│  │  - WebSocket    │    │  - 数据存储     │                │
│  │  - REST API     │    │  - 图数据库     │                │
│  │  - Precompute   │    │  - ChangeFeed   │                │
│  └─────────────────┘    └─────────────────┘                │
│           │                                                  │
│           ▼                                                  │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │   Meilisearch   │    │   ModelScope    │                │
│  │   (Port 7700)   │    │   API (外部)    │                │
│  │                 │    │                 │                │
│  │  - 全文搜索     │    │  - Embedding    │                │
│  │  - 索引管理     │    │  - 向量生成     │                │
│  └─────────────────┘    └─────────────────┘                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 端口映射

| 服务        | 内部端口 | 外部端口 | 说明                      |
| ----------- | -------- | -------- | ------------------------- |
| FastAPI     | 18008    | 18008    | 主服务端口（v3.2 新端口） |
| SurrealDB   | 8000     | 8000     | 数据库端口                |
| Meilisearch | 7700     | 7700     | 搜索服务端口              |

---

## 2. Docker 配置

### 2.1 Dockerfile

```dockerfile
# ============================================
# v3.2 Dockerfile - 多阶段构建
# ============================================

# -------------------------------------------------
# Stage 1: Builder
# -------------------------------------------------
FROM python:3.10-slim as builder

WORKDIR /app

# 安装构建依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY pyproject.toml poetry.lock* ./

# 安装 Poetry
RUN pip install poetry

# 配置 Poetry（不使用虚拟环境）
RUN poetry config virtualenvs.create false

# 安装依赖（仅生产依赖）
RUN poetry install --only main --no-interaction --no-ansi

# -------------------------------------------------
# Stage 2: Runtime
# -------------------------------------------------
FROM python:3.10-slim as runtime

WORKDIR /app

# 安装运行时依赖
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 从 builder 复制 Python 包
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 复制应用代码
COPY src/ ./src/

# 创建非 root 用户
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:18008/health || exit 1

# 暴露端口
EXPOSE 18008

# 启动命令
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "18008", "--workers", "4"]
```

### 2.2 docker-compose.yml

```yaml
# ============================================
# v3.2 Docker Compose 配置
# ============================================

version: "3.8"

services:
  # -----------------------------------------
  # FastAPI 服务
  # -----------------------------------------
  api:
    build:
      context: ./wrapper
      dockerfile: Dockerfile
    container_name: opencode-memory-api
    ports:
      - "18008:18008"
    environment:
      # 服务配置
      - PORT=18008
      - HOST=0.0.0.0
      - WORKERS=4

      # SurrealDB 配置
      - SURREALDB_URL=ws://surrealdb:8000
      - SURREALDB_NS=opencode
      - SURREALDB_DB=memory
      - SURREALDB_USER=root
      - SURREALDB_PASS=root

      # Meilisearch 配置
      - MEILISEARCH_URL=http://meilisearch:7700
      - MEILISEARCH_API_KEY=${MEILISEARCH_API_KEY}

      # ModelScope API
      - MODELSCOPE_API_KEY=${MODELSCOPE_API_KEY}

      # 日志级别
      - LOG_LEVEL=INFO

      # WebSocket 配置
      - WS_HEARTBEAT_INTERVAL=30
      - WS_RECONNECT_MAX_ATTEMPTS=10  # 最大重连次数（默认 10）

      # 预计算配置
      - PRECOMPUTE_BATCH_SIZE=100  # 批处理大小（默认 100，实时场景可设为 10）
      - PRECOMPUTE_INTERVAL=300
    volumes:
      - ./wrapper/src:/app/src:ro
      - api_logs:/app/logs
    depends_on:
      surrealdb:
        condition: service_healthy
      meilisearch:
        condition: service_healthy
    networks:
      - opencode-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:18008/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # -----------------------------------------
  # SurrealDB 服务
  # -----------------------------------------
  surrealdb:
    image: surrealdb/surrealdb:latest
    container_name: opencode-surrealdb
    ports:
      - "8000:8000"
    command:
      - start
      - --log
      - info
      - --user
      - root
      - --pass
      - root
      - file:/data/surrealdb.db
    volumes:
      - surrealdb_data:/data
    networks:
      - opencode-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "surreal", "is-ready", "--conn", "http://localhost:8000"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 10s

  # -----------------------------------------
  # Meilisearch 服务
  # -----------------------------------------
  meilisearch:
    image: getmeili/meilisearch:latest
    container_name: opencode-meilisearch
    ports:
      - "7700:7700"
    environment:
      - MEILI_MASTER_KEY=${MEILISEARCH_API_KEY}
      - MEILI_HTTP_PAYLOAD_SIZE_LIMIT=104857600 # 100MB
    volumes:
      - meilisearch_data:/meili_data
    networks:
      - opencode-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:7700/health"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 10s

# -----------------------------------------
# 网络配置
# -----------------------------------------
networks:
  opencode-network:
    driver: bridge

# -----------------------------------------
# 数据卷
# -----------------------------------------
volumes:
  surrealdb_data:
    driver: local
  meilisearch_data:
    driver: local
  api_logs:
    driver: local
```

### 2.3 .env 示例

```bash
# ============================================
# v3.2 环境变量配置
# ============================================

# -----------------------------------------
# API 密钥（必须设置）
# -----------------------------------------
MEILISEARCH_API_KEY=your-meilisearch-api-key-here
MODELSCOPE_API_KEY=your-modelscope-api-key-here

# -----------------------------------------
# 可选配置（使用默认值）
# -----------------------------------------
# PORT=18008
# HOST=0.0.0.0
# WORKERS=4
# LOG_LEVEL=INFO

# -----------------------------------------
# WebSocket 配置
# -----------------------------------------
# WS_HEARTBEAT_INTERVAL=30
# WS_RECONNECT_MAX_ATTEMPTS=10  # 最大重连次数（默认 10）

# -----------------------------------------
# 预计算配置
# -----------------------------------------
# PRECOMPUTE_BATCH_SIZE=100  # 批处理大小（默认 100，实时场景可设为 10）
# PRECOMPUTE_INTERVAL=300
```

---

## 3. 本地开发环境

### 3.1 快速启动

```bash
# 1. 克隆仓库
git clone https://github.com/csuwl/opencode-memory-plugin.git
cd opencode-memory-plugin/embedding_service

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 文件，设置 API 密钥

# 3. 启动服务
docker-compose up -d

# 4. 验证服务状态
docker-compose ps

# 5. 查看日志
docker-compose logs -f api
```

### 3.2 开发模式（热重载）

```bash
# 使用开发配置启动
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# docker-compose.dev.yml 内容：
version: '3.8'
services:
  api:
    volumes:
      - ./wrapper/src:/app/src  # 挂载源码（可写）
    command: ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "18008", "--reload"]
    environment:
      - LOG_LEVEL=DEBUG
```

### 3.3 本地 Python 开发

```bash
# 1. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或: venv\Scripts\activate  # Windows

# 2. 安装依赖
cd wrapper
pip install -e ".[dev]"

# 3. 启动 SurrealDB（Docker）
docker run -d --name surrealdb \
  -p 8000:8000 \
  -v surrealdb_data:/data \
  surrealdb/surrealdb:latest start \
  --user root --pass root file:/data/surrealdb.db

# 4. 启动 Meilisearch（Docker）
docker run -d --name meilisearch \
  -p 7700:7700 \
  -v meilisearch_data:/meili_data \
  -e MEILI_MASTER_KEY=your-api-key \
  getmeili/meilisearch:latest

# 5. 启动 FastAPI 服务
export MODELSCOPE_API_KEY=your-api-key
export MEILISEARCH_API_KEY=your-api-key
uvicorn src.main:app --host 0.0.0.0 --port 18008 --reload
```

---

## 4. 生产环境部署

### 4.1 生产配置清单

```yaml
# docker-compose.prod.yml
version: "3.8"

services:
  api:
    deploy:
      replicas: 2 # 多实例部署
      resources:
        limits:
          cpus: "2.0"
          memory: 4G
        reservations:
          cpus: "1.0"
          memory: 2G
    environment:
      - LOG_LEVEL=WARNING # 生产环境减少日志
      - WORKERS=8
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "3"

  surrealdb:
    deploy:
      resources:
        limits:
          cpus: "2.0"
          memory: 4G
    volumes:
      - type: bind
        source: /data/surrealdb
        target: /data

  meilisearch:
    deploy:
      resources:
        limits:
          cpus: "1.0"
          memory: 2G
```

### 4.2 反向代理配置（Nginx）

```nginx
# /etc/nginx/sites-available/opencode-memory

upstream api_backend {
    server localhost:18008;
    keepalive 32;
}

server {
    listen 80;
    server_name memory.example.com;

    # WebSocket 支持
    location /ws {
        proxy_pass http://api_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket 超时配置
        proxy_read_timeout 86400;
        proxy_send_timeout 86400;
    }

    # REST API
    location /api {
        proxy_pass http://api_backend;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # 超时配置
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # 健康检查
    location /health {
        proxy_pass http://api_backend/health;
        access_log off;
    }
}
```

### 4.3 SSL/TLS 配置

```bash
# 使用 Certbot 获取证书
certbot --nginx -d memory.example.com

# 自动续期
certbot renew --dry-run
```

### 4.5 Kubernetes 部署

#### 4.5.1 Namespace 和基础资源

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: opencode-memory
  labels:
    app: opencode-memory
---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: opencode-memory-config
  namespace: opencode-memory
data:
  PORT: "18008"
  HOST: "0.0.0.0"
  LOG_LEVEL: "INFO"
  WORKERS: "4"
  SURREALDB_URL: "ws://surrealdb:8000"
  SURREALDB_NS: "opencode"
  SURREALDB_DB: "memory"
  MEILISEARCH_URL: "http://meilisearch:7700"
  PRECOMPUTE_BATCH_SIZE: "100"
  PRECOMPUTE_INTERVAL: "300"
---
# k8s/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: opencode-memory-secret
  namespace: opencode-memory
type: Opaque
stringData:
  surrealdb-user: root
  surrealdb-pass: root
  meilisearch-api-key: ${MEILISEARCH_API_KEY}
  modelscope-api-key: ${MODELSCOPE_API_KEY}
```

#### 4.5.2 API Deployment

```yaml
# k8s/api-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: opencode-memory-api
  namespace: opencode-memory
  labels:
    app: opencode-memory-api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: opencode-memory-api
  template:
    metadata:
      labels:
        app: opencode-memory-api
    spec:
      containers:
        - name: api
          image: opencode-memory-api:v3.2.0
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: 18008
              name: http
              protocol: TCP
          env:
            - name: PORT
              valueFrom:
                configMapKeyRef:
                  name: opencode-memory-config
                  key: PORT
            - name: HOST
              valueFrom:
                configMapKeyRef:
                  name: opencode-memory-config
                  key: HOST
            - name: LOG_LEVEL
              valueFrom:
                configMapKeyRef:
                  name: opencode-memory-config
                  key: LOG_LEVEL
            - name: WORKERS
              valueFrom:
                configMapKeyRef:
                  name: opencode-memory-config
                  key: WORKERS
            - name: SURREALDB_URL
              valueFrom:
                configMapKeyRef:
                  name: opencode-memory-config
                  key: SURREALDB_URL
            - name: SURREALDB_USER
              valueFrom:
                secretKeyRef:
                  name: opencode-memory-secret
                  key: surrealdb-user
            - name: SURREALDB_PASS
              valueFrom:
                secretKeyRef:
                  name: opencode-memory-secret
                  key: surrealdb-pass
            - name: MEILISEARCH_URL
              valueFrom:
                configMapKeyRef:
                  name: opencode-memory-config
                  key: MEILISEARCH_URL
            - name: MEILISEARCH_API_KEY
              valueFrom:
                secretKeyRef:
                  name: opencode-memory-secret
                  key: meilisearch-api-key
            - name: MODELSCOPE_API_KEY
              valueFrom:
                secretKeyRef:
                  name: opencode-memory-secret
                  key: modelscope-api-key
          resources:
            requests:
              cpu: "500m"
              memory: "512Mi"
            limits:
              cpu: "2000m"
              memory: "4Gi"
          livenessProbe:
            httpGet:
              path: /health
              port: 18008
            initialDelaySeconds: 30
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 3
          readinessProbe:
            httpGet:
              path: /health
              port: 18008
            initialDelaySeconds: 10
            periodSeconds: 5
            timeoutSeconds: 3
            failureThreshold: 3
```

#### 4.5.3 SurrealDB Deployment

```yaml
# k8s/surrealdb-statefulset.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: surrealdb
  namespace: opencode-memory
spec:
  serviceName: surrealdb
  replicas: 1
  selector:
    matchLabels:
      app: surrealdb
  template:
    metadata:
      labels:
        app: surrealdb
    spec:
      containers:
        - name: surrealdb
          image: surrealdb/surrealdb:latest
          args:
            - start
            - --log
            - info
            - --user
            - root
            - --pass
            - root
            - file:/data/surrealdb.db
          ports:
            - containerPort: 8000
              name: http
          volumeMounts:
            - name: surrealdb-data
              mountPath: /data
          resources:
            requests:
              cpu: "500m"
              memory: "1Gi"
            limits:
              cpu: "2000m"
              memory: "4Gi"
```

#### 4.5.4 Meilisearch Deployment

```yaml
# k8s/meilisearch-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: meilisearch
  namespace: opencode-memory
spec:
  replicas: 1
  selector:
    matchLabels:
      app: meilisearch
  template:
    metadata:
      labels:
        app: meilisearch
    spec:
      containers:
        - name: meilisearch
          image: getmeili/meilisearch:latest
          ports:
            - containerPort: 7700
              name: http
          env:
            - name: MEILI_MASTER_KEY
              valueFrom:
                secretKeyRef:
                  name: opencode-memory-secret
                  key: meilisearch-api-key
            - name: MEILI_HTTP_PAYLOAD_SIZE_LIMIT
              value: "104857600"
          volumeMounts:
            - name: meilisearch-data
              mountPath: /meili_data
          resources:
            requests:
              cpu: "250m"
              memory: "512Mi"
            limits:
              cpu: "1000m"
              memory: "2Gi"
```

#### 4.5.5 Service 配置

```yaml
# k8s/services.yaml
---
# API Service
apiVersion: v1
kind: Service
metadata:
  name: opencode-memory-api
  namespace: opencode-memory
spec:
  type: ClusterIP
  ports:
    - port: 18008
      targetPort: 18008
      protocol: TCP
      name: http
  selector:
    app: opencode-memory-api
---
# SurrealDB Service
apiVersion: v1
kind: Service
metadata:
  name: surrealdb
  namespace: opencode-memory
spec:
  clusterIP: None
  ports:
    - port: 8000
      targetPort: 8000
      protocol: TCP
      name: http
  selector:
    app: surrealdb
---
# Meilisearch Service
apiVersion: v1
kind: Service
metadata:
  name: meilisearch
  namespace: opencode-memory
spec:
  type: ClusterIP
  ports:
    - port: 7700
      targetPort: 7700
      protocol: TCP
      name: http
  selector:
    app: meilisearch
```

#### 4.5.6 HPA 自动扩缩容

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: opencode-memory-api-hpa
  namespace: opencode-memory
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: opencode-memory-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Percent
          value: 50
          periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
        - type: Percent
          value: 100
          periodSeconds: 15
      selectPolicy: Max
```

#### 4.5.7 PVC 配置

```yaml
# k8s/pvc.yaml
---
# SurrealDB PVC
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: surrealdb-data
  namespace: opencode-memory
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
---
# Meilisearch PVC
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: meilisearch-data
  namespace: opencode-memory
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
```

#### 4.5.8 K8s 部署命令

```bash
# 创建命名空间和基础资源
kubectl apply -f k8s/namespace.yaml

# 部署所有组件
kubectl apply -f k8s/

# 检查状态
kubectl get all -n opencode-memory

# 查看日志
kubectl logs -n opencode-memory -l app=opencode-memory-api

# 扩缩容
kubectl scale deployment opencode-memory-api --replicas=5 -n opencode-memory

# 删除
kubectl delete -f k8s/ -n opencode-memory
```

---

### 4.4 系统服务配置（systemd）

```ini
# /etc/systemd/system/opencode-memory.service

[Unit]
Description=OpenCode Memory Service
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/opencode-memory-plugin/embedding_service
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
```

```bash
# 启用服务
sudo systemctl enable opencode-memory
sudo systemctl start opencode-memory
sudo systemctl status opencode-memory
```

### 4.6 SSL 自动续期

#### 4.6.1 Certbot 配置（裸机）

```bash
# 安装 Certbot
apt-get update
apt-get install certbot python3-certbot-nginx

# 获取证书
certbot --nginx -d memory.example.com --non-interactive --agree-tos \
  --email your-email@example.com

# 测试续期
certbot renew --dry-run

# 添加 cron 任务（每天凌晨 2 点检查）
crontab -e
# 0 2 * * * certbot renew --quiet --deploy-hook "systemctl reload nginx"
```

#### 4.6.2 Certbot 配置（Docker）

```dockerfile
# Dockerfile.certbot
FROM certbot/certbot as certbot

# 获取证书
certbot certonly --webroot \
  -w /var/www/html \
  -d memory.example.com \
  --non-interactive \
  --agree-tos \
  --email your-email@example.com

# 复制证书
COPY /etc/letsencrypt/live/memory.example.com/fullchain.pem /etc/ssl/certs/
COPY /etc/letsencrypt/live/memory.example.com/privkey.pem /etc/ssl/private/
```

```yaml
# docker-compose.ssl.yml
version: "3.8"

services:
  certbot:
    image: certbot/certbot
    volumes:
      - ./certbot:/etc/letsencrypt
      - ./www:/var/www/html
    command: certonly --webroot -w /var/www/html -d memory.example.com --non-interactive --agree-tos --email your-email@example.com --keep-until-expiring

  nginx:
    image: nginx:latest
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./certbot/live:/etc/ssl/certs:ro
      - ./certbot/private:/etc/ssl/private:ro
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./www:/var/www/html:ro
    depends_on:
      - certbot
```

#### 4.6.3 Certbot 配置（Kubernetes）

```yaml
# k8s/cert.yaml
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: opencode-memory-tls
  namespace: opencode-memory
spec:
  secretName: opencode-memory-tls
  issuerRef:
    name: letsencrypt-prod
    kind: ClusterIssuer
    group: cert-manager.io
  dnsNames:
    - memory.example.com
  acme:
    config:
      - http01:
          ingressClass: nginx
        selector: {}
---
# k8s/cluster-issuer.yaml
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: your-email@example.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
      - http01:
          ingress:
            class: nginx
---
# k8s/ingress.yaml (更新)
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: opencode-memory-ingress
  namespace: opencode-memory
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
    - hosts:
        - memory.example.com
      secretName: opencode-memory-tls
  rules:
    - host: memory.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: opencode-memory-api
                port:
                  number: 18008
```

#### 4.6.4 自动续期 cron 任务

```bash
# /etc/cron.d/certbot-renew
# 检查证书续期，每天凌晨 2 点

SHELL=/bin/sh
PATH=/usr/local/sbin:/usr/local/bin:/sbin:/bin:/usr/sbin:/usr/bin

# 续期检查
0 2 * * * root certbot renew --quiet --deploy-hook "systemctl reload nginx" >> /var/log/certbot-renew.log 2>&1
```

#### 4.6.5 证书更新钩子

```bash
#!/bin/bash
# /etc/letsencrypt/renewal-hooks/post/reload-nginx.sh

echo "[$(date)] 证书已更新，重载 Nginx..."

# ��载 Nginx
systemctl reload nginx

# 重启 Docker 容器（如需要）
docker-compose -f /opt/opencode-memory-plugin/docker-compose.yml restart api

# 发送通知（如使用 Slack/Webhook）
if [ -n "$WEBHOOK_URL" ]; then
  curl -X POST "$WEBHOOK_URL" \
    -H 'Content-Type: application/json' \
    -d "{\"text\": \"证书已更新 (memory.example.com)\"}"
fi

echo "[$(date)] 完成"
```

```yaml
# GitLab CI 集成示例
# .gitlab-ci.yml
certbot:
  script:
    - certbot renew --non-interactive --deploy-hook "docker-compose restart api"
  only:
    schedules:
      - cron: '0 2 * * *'
```

#### 4.6.6 续期监控

```bash
# 查看续期日志
journalctl -u certbot -f

# 查看证书状态
certbot certificates

# 测试续期流程
certbot renew --dry-run --debug

# 手动触发续期
certbot renew --force-renewal
```

---

## 5. 监控与日志

### 5.1 健康检查端点

| 端点             | 说明            | 预期响应                                                 |
| ---------------- | --------------- | -------------------------------------------------------- |
| `GET /health`    | 服务健康状态    | `{"status": "healthy"}`                                  |
| `GET /health/db` | 数据库连接状态  | `{"surrealdb": "connected", "meilisearch": "connected"}` |
| `GET /health/ws` | WebSocket 状态  | `{"connections": 5, "uptime": 3600}`                     |
| `GET /metrics`   | Prometheus 指标 | 性能指标数据                                             |

### 5.2 日志配置

```python
# src/config/logging.py
import logging
from pythonjsonlogger import jsonlogger

# 结构化日志配置
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter(
    '%(timestamp)s %(level)s %(name)s %(message)s',
    rename_fields={'levelname': 'level', 'asctime': 'timestamp'}
)
logHandler.setFormatter(formatter)

logging.getLogger().addHandler(logHandler)
logging.getLogger().setLevel(logging.INFO)
```

### 5.3 监控指标

```python
# src/config/metrics.py
from prometheus_client import Counter, Histogram, Gauge

# 请求计数
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

# 请求延迟
http_request_duration = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

# WebSocket 连接数
websocket_connections = Gauge(
    'websocket_connections',
    'Active WebSocket connections'
)

# 预计算队列长度
precompute_queue_size = Gauge(
    'precompute_queue_size',
    'Precompute task queue size'
)
```

### 5.4 日志查看命令

```bash
# 查看实时日志
docker-compose logs -f api

# 查看最近 100 行
docker-compose logs --tail=100 api

# 查看特定时间范围
docker-compose logs --since="2026-04-10T10:00:00" --until="2026-04-10T12:00:00" api

# 导出日志
docker-compose logs api > api_logs.txt
```

---

## 6. 故障排除

### 6.1 常见问题

#### 服务无法启动

```bash
# 检查端口占用
netstat -tlnp | grep 18008

# 检查环境变量
docker-compose config

# 查看详细错误
docker-compose logs api
```

#### 数据库连接失败

```bash
# 检查 SurrealDB 状态
docker-compose ps surrealdb
docker-compose logs surrealdb

# 手动测试连接
curl http://localhost:8000/health
```

#### WebSocket 连接问题

```bash
# 检查 WebSocket 端点
curl -i -N \
  -H "Connection: Upgrade" \
  -H "Upgrade: websocket" \
  -H "Host: localhost:18008" \
  -H "Origin: http://localhost:18008" \
  http://localhost:18008/ws
```

### 6.2 性能调优

#### 数据库优化

```sql
-- 分析查询性能
ANALYZE TABLE atom;
ANALYZE TABLE entity;

-- 查看慢查询（SurrealDB 日志）
-- 在 docker-compose.yml 中设置: --log debug
```

#### 内存优化

```yaml
# docker-compose.yml
services:
  api:
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G
    environment:
      - PYTHONOPTIMIZE=1 # Python 优化模式
      - UVICORN_WORKERS=4 # 根据 CPU 核心数调整
```

### 6.3 备份与恢复

```bash
# 备份 SurrealDB
docker exec opencode-surrealdb surreal dump \
  --conn http://localhost:8000 \
  --user root --pass root \
  --ns opencode --db memory \
  /backup/surrealdb-$(date +%Y%m%d).sql

# 备份 Meilisearch
docker exec opencode-meilisearch meilisearch-dump \
  --master-key $MEILISEARCH_API_KEY \
  --dump-dir /backup/meilisearch-$(date +%Y%m%d)

# 恢复 SurrealDB
docker exec opencode-surrealdb surreal import \
  --conn http://localhost:8000 \
  --user root --pass root \
  --ns opencode --db memory \
  /backup/surrealdb-20260410.sql
```

### 6.4 升级指南

```bash
# 1. 备份数据
./scripts/backup.sh

# 2. 拉取最新镜像
docker-compose pull

# 3. 停止服务
docker-compose down

# 4. 启动新版本
docker-compose up -d

# 5. 运行迁移脚本
docker-compose exec api python src/db/migrations/migrate_v2_to_v3.2.py

# 6. 验证升级
curl http://localhost:18008/health
```

---

## 附录

### A. 端口速查

| 服务        | 端口  | 用途       |
| ----------- | ----- | ---------- |
| FastAPI     | 18008 | 主服务 API |
| SurrealDB   | 8000  | 数据库     |
| Meilisearch | 7700  | 搜索服务   |

### B. 环境变量清单

| 变量                  | 必需 | 默认值 | 说明                 |
| --------------------- | ---- | ------ | -------------------- |
| `MODELSCOPE_API_KEY`  | ✅   | -      | ModelScope API 密钥  |
| `MEILISEARCH_API_KEY` | ✅   | -      | Meilisearch API 密钥 |
| `PORT`                | ❌   | 18008  | 服务端口             |
| `LOG_LEVEL`           | ❌   | INFO   | 日志级别             |
| `WORKERS`             | ❌   | 4      | Uvicorn 工作进程数   |

### C. 参考文档

- [Docker 文档](https://docs.docker.com/)
- [Docker Compose 文档](https://docs.docker.com/compose/)
- [SurrealDB Docker 部署](https://surrealdb.com/docs/installation/docker)
- [Meilisearch Docker 部署](https://www.meilisearch.com/docs/learn/getting_started/installation)
