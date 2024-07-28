import sentencepiece as spm
from transformers import GPT2Tokenizer

# 初始化 GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# 保存 tokenizer 的词汇表到一个文件
vocab_file = "gpt2_vocab.txt"
with open(vocab_file, 'w') as f:
    for token, index in tokenizer.get_vocab().items():
        f.write(f"{token}\t{index}\n")


# 训练 SentencePiece 模型
spm.SentencePieceTrainer.train(
    input=vocab_file, 
    model_prefix="gpt2_sentencepiece", 
    vocab_size=31923
)

# 加载 SentencePiece 模型
sp = spm.SentencePieceProcessor()
sp.load("gpt2_sentencepiece.model")

# 测试 SentencePiece tokenizer
test_text = "Hello, how are you?"
pieces = sp.encode_as_pieces(test_text)
ids = sp.encode_as_ids(test_text)

print("SentencePiece pieces:", pieces)
print("SentencePiece ids:", ids)

