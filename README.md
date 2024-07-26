# 手撸GPT

本项目是简单的GPT实现，预训练语料为唐诗三百首。
训练的是简单的基座，不进行instruct tuning、RLHF。
训练目标是，GPT可以自主生成唐诗。


## 结构细节
512维的词向量，6层transformer，每层有8个attention heads，每个attention head有64维的key、value、query向量。

采用RoPE位置编码
BPE分词
pre-Norm正则化

