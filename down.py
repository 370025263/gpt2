import os
from datasets import load_dataset

# 指定本地缓存目录
cache_dir = os.path.join(os.getcwd(), 'data')

# 创建目录（如果不存在）
os.makedirs(cache_dir, exist_ok=True)

# 加载数据集并指定缓存目录
dataset = load_dataset('nampdn-ai/mini-fineweb', cache_dir=cache_dir)

# 打印数据集信息
print(dataset)

