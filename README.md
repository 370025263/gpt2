# 手撸GPT

本项目是GPT-2复现，预训练语fineweb:https://huggingface.co/datasets/HuggingFaceFW/fineweb。
训练的是简单的基座，不进行instruct tuning、RLHF。
训练目标是，GPT可以自主生成content。


## 结构细节

### GPT2-small

层数 (Layers): 12
dm大小 (Hidden Size, H): 768
前馈网络大小 (Feed Forward Size, FFN): 4H = 3072
注意力头数 (Attention Heads): 12
总参数数量: 约117 million


采用RoPE位置编码
BPE分词

## 数据
### tokenizer
使用BPE分词，词表大小为50257
block_size (i.e. input context length) = 1024
