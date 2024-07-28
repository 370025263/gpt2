import os
from datasets import load_dataset
import sentencepiece as spm

def save_dataset_to_file(dataset, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for example in dataset['train']:
            f.write(example['text'] + '\n')
        for example in dataset['validation']:
            f.write(example['text'] + '\n')

def train_tokenizer(input_file, model_prefix, vocab_size=50257):
    spm.SentencePieceTrainer.Train(
        f'--input={input_file} --model_prefix={model_prefix} --vocab_size={vocab_size} '
        f'--model_type=bpe --character_coverage=1.0 --input_sentence_size=10000000 '
        f'--shuffle_input_sentence=true --num_threads=60 --pad_id=0 --unk_id=1 '
        f'--bos_id=2 --eos_id=3 --user_defined_symbols=[PAD] --hard_vocab_limit=false'
    )

def load_tokenizer(model_prefix):
    sp = spm.SentencePieceProcessor()
    sp.load(f'{model_prefix}.model')
    return sp

if __name__ == '__main__':
    # 加载数据集
    cache_dir = os.path.join(os.getcwd(), 'data')
    os.makedirs(cache_dir, exist_ok=True)
    dataset = load_dataset('deatos/fineweb-edu-mini-combined', cache_dir=cache_dir)

    # 保存数据集到文件
    input_file = os.path.join(cache_dir, 'all_texts.txt')
    save_dataset_to_file(dataset, input_file)

    # 训练 tokenizer
    train_tokenizer(input_file, 'tokenizer', 50257)

    # 加载并测试 tokenizer
    sp = load_tokenizer('tokenizer')
    print(sp.encode('hello world', out_type=str))
    print(sp.decode(sp.encode('hello world', out_type=str)))

