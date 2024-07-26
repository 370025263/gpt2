import sentencepiece as spm
import os


def train_tokenizer(input_file, model_prefix, vocab_size=32000):
    ## parallel tokenizer training
    spm.SentencePieceTrainer.Train(f'--input={input_file} --model_prefix={model_prefix} --vocab_size={vocab_size} --model_type=bpe --character_coverage=1.0 --input_sentence_size=10000000 --shuffle_input_sentence=true --num_threads=8 --pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 --user_defined_symbols=[PAD] --hard_vocab_limit=false')


def load_tokenizer(model_prefix):
    sp = spm.SentencePieceProcessor()
    sp.load(f'{model_prefix}.model')
    return sp


if __name__ == '__main__':
    train_tokenizer('data/train.txt', 'tokenizer', 32000)
    sp = load_tokenizer('tokenizer')
    print(sp.encode('hello world', out_type=str))
    print(sp.decode(sp.encode('hello world', out_type=str)))
