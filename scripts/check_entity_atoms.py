#!/usr/bin/env python3
import asyncio
from surrealdb import AsyncSurreal

async def main():
    db = AsyncSurreal('ws://localhost:18002/rpc')
    await db.connect()
    await db.signin({'username': 'root', 'password': 'root'})
    await db.use('memory_ns', 'memory_db')

    # 查询 entity_id 不为空的 atoms
    result = await db.query('SELECT id, local_id, name, entity_id FROM atom WHERE entity_id IS NOT NONE LIMIT 10')
    if result and len(result) > 0:
        records = result[0]
        print(f'Found {len(records)} atoms with entity_id')
        for record in records:
            print(f"ID: {record.get('id')}")
            print(f"  local_id: {record.get('local_id')}")
            print(f"  name: {record.get('name')}")
            print()
    else:
        print('No atoms with entity_id found')

    await db.close()

asyncio.run(main())
