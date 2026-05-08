import asyncio
from surrealdb import AsyncSurreal

async def check():
    db = AsyncSurreal('ws://localhost:18002/rpc')
    await db.connect()
    await db.signin({'username': 'root', 'password': 'root'})
    await db.use('memory_ns', 'memory_db')
    
    # 直接测试 UPDATE
    result = await db.query("UPDATE atom SET entity_id = 'entity:test123' WHERE id = atom:ybvl8uzuj3i7a4s5vymw")
    print('UPDATE result:', result)
    
    # 检查结果
    result2 = await db.query("SELECT id, entity_id FROM atom WHERE id = atom:ybvl8uzuj3i7a4s5vymw")
    print('After UPDATE:', result2)
    
    await db.close()

asyncio.run(check())
