import asyncio
from surrealdb import AsyncSurreal, RecordID

async def check():
    db = AsyncSurreal('ws://localhost:18002/rpc')
    await db.connect()
    await db.signin({'username': 'root', 'password': 'root'})
    await db.use('memory_ns', 'memory_db')
    
    # 测试 UPDATE 语句
    entity_id = 'entity:57gltpwcg36vw7a2utt5'
    atom_id = 'atom:6hmiz2pdx2swbu93pnrp'
    
    # 方法1: 使用字符串
    result1 = await db.query(
        "UPDATE atom SET entity_id = $entity_id WHERE id = $atom_id",
        {"entity_id": entity_id, "atom_id": atom_id}
    )
    print("Method 1 (string):", result1)
    
    # 方法2: 使用 RecordID
    result2 = await db.query(
        "UPDATE atom SET entity_id = $entity_id WHERE id = $atom_id",
        {"entity_id": RecordID("entity", "57gltpwcg36vw7a2utt5"), "atom_id": RecordID("atom", "6hmiz2pdx2swbu93pnrp")}
    )
    print("Method 2 (RecordID):", result2)
    
    # 检查结果
    result3 = await db.query("SELECT id, entity_id FROM atom WHERE id = atom:6hmiz2pdx2swbu93pnrp")
    print("After update:", result3)
    
    await db.close()

asyncio.run(check())
