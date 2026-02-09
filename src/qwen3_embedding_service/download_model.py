import os
from modelscope import snapshot_download

print('⏳ 正在下载模型（首次约需 2-5 分钟）...')
# model_dir = snapshot_download('Qwen/Qwen3-Embedding-0.6B', cache_dir='../../models')
model_dir = snapshot_download('OpenBMB/MiniCPM3-4B-GGUF', cache_dir='models')

print(f'\n✅ 模型下载完成！')
print(f'   位置: {model_dir}')
print(f'   大小: {sum(os.path.getsize(os.path.join(dp, f)) for dp, dn, filenames in os.walk(model_dir) for f in filenames) / 1024**3:.2f} GB')
