#!/bin/bash
BACKUP_FILE="$1"

if [ -z "$BACKUP_FILE" ]; then
  echo "用法: ./test_restore.sh <备份文件>"
  exit 1
fi

if [ ! -f "$BACKUP_FILE" ]; then
  echo "❌ 文件不存在: $BACKUP_FILE"
  exit 1
fi

echo "测试还原到: memory_ns_test/memory_db_test"

surreal import \
  --endpoint http://localhost:18002 \
  --namespace memory_ns_test \
  --database memory_db_test \
  --username root \
  --password root \
  "$BACKUP_FILE"

if [ $? -eq 0 ]; then
  echo "✅ 测试还原成功"
  
  echo "验证数据..."
  surreal sql \
    --endpoint http://localhost:18002 \
    --namespace memory_ns_test \
    --database memory_db_test \
    --username root \
    --password root \
    --pretty \
    "SELECT count() FROM memory GROUP ALL;"
else
  echo "❌ 测试还原失败"
  exit 1
fi
