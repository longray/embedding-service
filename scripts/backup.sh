#!/bin/bash
BACKUP_DIR="backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="${BACKUP_DIR}/backup_${TIMESTAMP}.surql"

mkdir -p "$BACKUP_DIR"

surreal export \
  --endpoint http://localhost:18002 \
  --namespace memory_ns \
  --database memory_db \
  --username root \
  --password root \
  "$BACKUP_FILE"

if [ $? -eq 0 ]; then
  echo "✅ 备份成功: $BACKUP_FILE"
  ls -lh "$BACKUP_FILE"
else
  echo "❌ 备份失败"
  exit 1
fi
