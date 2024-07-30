import os
from datasets import load_dataset, Dataset, DatasetDict

# 指定本地缓存目录
cache_dir = os.path.join(os.getcwd(), 'data')
# 创建目录（如果不存在）
os.makedirs(cache_dir, exist_ok=True)

def load_and_split_dataset(test_size=0.001, seed=42):
    # 检查是否已经存在处理好的数据集
    if os.path.exists(os.path.join(cache_dir, 'processed_dataset')):
        print("Loading pre-processed dataset from disk...")
        return DatasetDict.load_from_disk(os.path.join(cache_dir, 'processed_dataset'))

    print("Loading and processing dataset...")
    # 加载数据集
    dataset = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", cache_dir=cache_dir)
    
    # 计算分割点
    split_point = int(len(dataset) * (1 - test_size))
    
    # 分割数据集
    train_dataset = dataset.select(range(split_point))
    val_dataset = dataset.select(range(split_point, len(dataset)))
    
    # 创建 DatasetDict
    split_datasets = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    })

    # 保存处理后的数据集到本地
    split_datasets.save_to_disk(os.path.join(cache_dir, 'processed_dataset'))
    print("Processed dataset saved to disk.")

    return split_datasets

if __name__ == "__main__":
    # 加载并分割数据集
    split_datasets = load_and_split_dataset()

    # 打印数据集信息
    print("Train dataset:", split_datasets['train'])
    print("Validation dataset:", split_datasets['validation'])
