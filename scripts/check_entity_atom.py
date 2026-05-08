import asyncio
from surrealdb import AsyncSurreal

async def check():
    db = AsyncSurreal('ws://localhost:18002/rpc')
    await db.connect()
    await db.signin({'username': 'root', 'password': 'root'})
    await db.use('memory_ns', 'memory_db')
    
    # 检查 atom 的 entity_id
    result = await db.query("SELECT id, local_id, name, entity_id FROM atom WHERE local_id = '01KTEST01SE00000000000001'")
    print('Atom with local_id 01KTEST01SE00000000000001:')
    print(result)
    
    # 检查 entity 的 atoms
    result2 = await db.query("SELECT id, atoms FROM entity WHERE id = entity:57gltpwcg36vw7a2utt5")
    print('\nEntity entity:57gltpwcg36vw7a2utt5:')
    print(result2)
    
    await db.close()

asyncio.run(check())
