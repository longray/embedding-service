# SurrealDB 备份还原指南

## 备份

```bash
bash scripts/backup.sh
```text

备份文件保存在 `backups/backup_YYYYMMDD_HHMMSS.surql`

## 还原

```bash
bash scripts/restore.sh backups/backup_20260316_195137.surql
```

⚠️ 还原会覆盖现有数据，操作前会要求确认。

## 测试还原

还原到测试数据库（不影响生产数据）：

```bash
bash scripts/test_restore.sh backups/backup_20260316_195137.surql
```text

## 验证

```bash
# 查看记录数
echo "SELECT count() FROM memory GROUP ALL;" | surreal sql \
  --endpoint http://localhost:18002 \
  --namespace memory_ns \
  --database memory_db \
  --username root \
  --password root
```

## 自动化备份

添加到 crontab：

```bash
# 每天凌晨 2 点备份
0 2 * * * cd /path/to/embedding_service && bash scripts/backup.sh
```text

## 备份文件管理

```bash
# 保留最近 30 天的备份
find backups/ -name "backup_*.surql" -mtime +30 -delete
```
