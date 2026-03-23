#!/bin/bash
if [ -z "$1" ]; then
  echo "用法: ./restore.sh <备份文件>"
  echo "示例: ./restore.sh backups/backup_20260316_194943.surql"
  exit 1
fi

BACKUP_FILE="$1"

if [ ! -f "$BACKUP_FILE" ]; then
  echo "❌ 文件不存在: $BACKUP_FILE"
  exit 1
fi

echo "⚠️  警告: 此操作将覆盖现有数据"
echo "备份文件: $BACKUP_FILE"
read -p "确认继续? (y/N): " confirm

if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
  echo "已取消"
  exit 0
fi

surreal import \
  --endpoint http://localhost:18002 \
  --namespace memory_ns \
  --database memory_db \
  --username root \
  --password root \
  "$BACKUP_FILE"

if [ $? -eq 0 ]; then
  echo "✅ 还原成功"
else
  echo "❌ 还原失败"
  exit 1
fi
