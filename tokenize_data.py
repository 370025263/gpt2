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
    # 注意（只加注释，没改参数，因为仓库里的 tokenizer.model 就是按这组参数训出来的，改了要重新训）：
    # 1. 原版 GPT-2 用的是字节级 BPE：先把文本转成 UTF-8 字节再合并，任何字符都能编码，没有 unk。
    #    这里是 SentencePiece 的字符级 BPE，训练语料里没出现过的字符会变成 unk（id=1）。
    #    想接近 GPT-2，可以加 --byte_fallback=true，没见过的字符退回成字节编码。
    # 2. pad_id=0 已经给 pad 留了位置，--user_defined_symbols=[PAD] 又多加了一个字面上叫 [PAD] 的普通词，是重复的。
    #    现在训练数据是拼接切块，根本不需要 pad。
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
    # 注意：分词器是在 deatos/fineweb-edu-mini-combined 上训的，而 train.py 训练用的是 down_data.py 下的 fineweb sample-10BT。
    # 两者都是 fineweb 系的英文网页，分布接近，影响不大；但严格来说分词器应该在训练语料的一个子集上训。
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

