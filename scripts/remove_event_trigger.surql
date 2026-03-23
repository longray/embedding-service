-- 删除自动计算 content_hash 的 EVENT 触发器
-- 原因：应用层已经在插入前计算 content_hash，触发器会导致无限递归

REMOVE EVENT auto_content_hash ON TABLE memory;
