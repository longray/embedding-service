import asyncio
from surrealdb import AsyncSurreal

async def check():
    db = AsyncSurreal('ws://localhost:18002/rpc')
    await db.connect()
    await db.signin({'username': 'root', 'password': 'root'})
    await db.use('memory_ns', 'memory_db')
    
    # 检查 atom 的 entity_id
    result = await db.query("SELECT id, local_id, name, entity_id FROM atom WHERE local_id = '01KTEST02SE00000000000001'")
    print('Atom with local_id 01KTEST02SE00000000000001:')
    if result and len(result) > 0:
        for r in result[0].get('result', []):
            print(f'  {r}')
    
    # 检查 entity 的 atoms
    result2 = await db.query("SELECT id, atoms FROM entity WHERE id = entity:oq9duyc5cn9c8qsysn6b")
    print('\nEntity entity:oq9duyc5cn9c8qsysn6b:')
    if result2 and len(result2) > 0:
        for r in result2[0].get('result', []):
            print(f'  {r}')
    
    await db.close()

asyncio.run(check())
