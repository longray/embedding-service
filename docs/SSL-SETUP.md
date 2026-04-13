# SSL 配置指南

本文档介绍如何为 Embedding Service 配置 SSL 证书，实现 HTTPS 访问。

## 前置条件

- 拥有一个域名（例如：`api.example.com`）
- 域名已解析到服务器 IP
- 服务器已开放 80 和 443 端口

## 快速开始

### 1. 初始化 SSL 证书

使用提供的脚本初始化 SSL 证书：

```bash
# 进入项目目录
cd /path/to/embedding_service

# 运行初始化脚本
./scripts/init_ssl.sh api.example.com admin@example.com
```

脚本会自动：
- 创建必要的目录结构
- 使用 Certbot 申请 Let's Encrypt 证书
- 配置证书自动续期

### 2. 启动 SSL 服务

使用 docker-compose 启动带 SSL 的服务：

```bash
# 启动基础服务
docker compose up -d

# 启动 SSL 服务（Nginx + Certbot）
docker compose -f docker-compose.yml -f docker-compose.ssl.yml --profile ssl up -d
```

### 3. 验证 SSL 配置

```bash
# 检查证书
openssl s_client -connect api.example.com:443 -servername api.example.com

# 检查 HTTPS 访问
curl -I https://api.example.com/health
```

## 详细配置

### 域名配置

1. **DNS 解析**
   - 添加 A 记录：`api.example.com` → 服务器 IP
   - 等待 DNS 生效（通常几分钟到几小时）

2. **防火墙配置**
   ```bash
   # 开放 80 和 443 端口
   sudo ufw allow 80/tcp
   sudo ufw allow 443/tcp
   ```

### 证书管理

#### 查看证书信息

```bash
# 查看证书详情
docker compose -f docker-compose.ssl.yml exec certbot certbot certificates

# 查看证书过期时间
echo | openssl s_client -servername api.example.com -connect api.example.com:443 2>/dev/null | openssl x509 -noout -dates
```

#### 手动续期

证书会自动续期，但也可以手动触发：

```bash
docker compose -f docker-compose.ssl.yml run --rm certbot certbot renew
```

#### 删除证书

```bash
docker compose -f docker-compose.ssl.yml run --rm certbot certbot delete --cert-name api.example.com
```

### Nginx 配置

Nginx 配置文件位于 `nginx/nginx.conf`，主要功能：

- **SSL 终止**：处理 HTTPS 请求，解密后转发给后端
- **反向代理**：将请求转发到 wrapper 服务（端口 18008）
- **HTTP 重定向**：自动将 HTTP 请求重定向到 HTTPS
- **安全头部**：添加 HSTS、X-Frame-Options 等安全头部
- **速率限制**：防止 API 滥用

#### 自定义配置

如需自定义 Nginx 配置，可以：

1. 修改 `nginx/nginx.conf`
2. 重启 Nginx 服务：
   ```bash
   docker compose -f docker-compose.ssl.yml restart nginx
   ```

## 故障排查

### 证书申请失败

**症状**：Certbot 无法申请证书

**排查步骤**：
1. 检查域名解析是否正确：
   ```bash
   nslookup api.example.com
   ```
2. 检查 80 端口是否开放：
   ```bash
   curl -I http://api.example.com/.well-known/acme-challenge/test
   ```
3. 检查防火墙设置
4. 查看 Certbot 日志：
   ```bash
   docker compose -f docker-compose.ssl.yml logs certbot
   ```

### HTTPS 无法访问

**症状**：浏览器显示证书错误

**排查步骤**：
1. 检查证书是否存在：
   ```bash
   ls -la nginx/certbot-data/live/
   ```
2. 检查 Nginx 配置：
   ```bash
   docker compose -f docker-compose.ssl.yml exec nginx nginx -t
   ```
3. 查看 Nginx 错误日志：
   ```bash
   docker compose -f docker-compose.ssl.yml logs nginx
   ```

### 证书续期失败

**症状**：证书过期未自动续期

**排查步骤**：
1. 检查 Certbot 容器是否运行：
   ```bash
   docker compose -f docker-compose.ssl.yml ps
   ```
2. 手动测试续期：
   ```bash
   docker compose -f docker-compose.ssl.yml run --rm certbot certbot renew --dry-run
   ```
3. 检查证书过期时间：
   ```bash
   openssl x509 -in nginx/certbot-data/live/api.example.com/fullchain.pem -noout -dates
   ```

## 安全建议

1. **定期备份证书**
   ```bash
   tar -czvf ssl-backup-$(date +%Y%m%d).tar.gz nginx/certbot-data/
   ```

2. **监控证书过期**
   - 设置告警：证书过期前 7 天通知
   - 使用监控工具检查证书有效期

3. **使用强加密**
   - 默认配置已启用 TLS 1.2 和 TLS 1.3
   - 禁用弱加密算法

4. **启用 HSTS**
   - 默认已启用 HTTP Strict Transport Security
   - 强制浏览器使用 HTTPS

## 参考

- [Let's Encrypt 文档](https://letsencrypt.org/docs/)
- [Certbot 文档](https://eff-certbot.readthedocs.io/)
- [Nginx SSL 配置](https://nginx.org/en/docs/http/configuring_https_servers.html)
