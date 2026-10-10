import argparse
import itertools
import math
import os
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
import torch.nn.functional as F
from torchmetrics.classification import Accuracy

from decoder import GPT
from down_data import load_and_split_dataset
from tokenize_data import load_tokenizer
from pytorch_lightning import Trainer  # 确保导入 Trainer 类

# 修正：原来每篇文档截到前 100 个 token，不够的补 0，有三个问题：
# 1. 模型的上下文是 1024，却只用 100 个 token 训练，从来见不到长距离依赖；
# 2. fineweb 的文档普遍比 100 个 token 长得多，后面的内容全扔了；短文档又补了大量 pad，算力浪费在 pad 上；
# 3. 文档末尾没有 eos，模型学不到「一段话该结束了」。
# GPT 系列的做法是：每篇文档后面接一个 eos，所有文档首尾相连成一条长序列，再切成等长的块，没有 pad。
# 下面用 datasets.map 预先分词并切块（和 HuggingFace 官方 run_clm.py 的 group_texts 一样的思路），
# 结果缓存到磁盘，训练时 DataLoader 只做取数，不再在 worker 里现场分词。
_tokenizer = None

def _tokenize(batch, tokenizer_prefix):
    global _tokenizer
    if _tokenizer is None:  # 每个子进程各自加载一次
        _tokenizer = load_tokenizer(tokenizer_prefix)
    ids = _tokenizer.encode(batch['text'], out_type=int)
    return {'ids': [x + [_tokenizer.eos_id()] for x in ids]}

def _group(batch, block_size):
    concat = list(itertools.chain.from_iterable(batch['ids']))
    n = len(concat) // block_size * block_size  # 每个 map 批次末尾不足一块的部分丢掉
    return {'ids': [concat[i:i + block_size] for i in range(0, n, block_size)]}

class TextDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer_prefix, max_length=1024, num_proc=8):
        tokenized = hf_dataset.map(_tokenize, batched=True, num_proc=num_proc,
                                   remove_columns=hf_dataset.column_names,
                                   fn_kwargs={'tokenizer_prefix': tokenizer_prefix})
        self.blocks = tokenized.map(_group, batched=True, num_proc=num_proc,
                                    fn_kwargs={'block_size': max_length})
        self.blocks.set_format('torch', columns=['ids'])

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        return self.blocks[idx]['ids']

class GPTLightningModule(pl.LightningModule):
    def __init__(self, gpt_model, tokenizer, vocab_size, warmup_steps, learning_rate=6e-4):
        super(GPTLightningModule, self).__init__()
        self.gpt_model = gpt_model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.train_acc = Accuracy(task='multiclass', num_classes=vocab_size)
        self.val_acc = Accuracy(task='multiclass', num_classes=vocab_size)

    def forward(self, x, mask):
        return self.gpt_model(x, mask)

    def _shared_step(self, batch):
        x = batch.to(self.device)
        # 修正：数据已经拼接切块，没有 pad 了，只需要因果掩码（下三角）
        mask = torch.tril(torch.ones((x.size(1), x.size(1)), device=x.device)).bool()
        logits = self(x, mask)
        # 修正：这是全仓库最关键的错误。原来是 cross_entropy(logits, x)，即用位置 t 的输出预测位置 t 自己的 token。
        # 因果掩码允许位置 t 看到自己，模型只要把输入原样抄到输出就能让 loss 接近 0，什么语言知识都学不到。
        # 语言模型的目标是预测下一个 token：位置 t 的输出对应的标签是 x[t+1]。
        # 所以 logits 去掉最后一个位置（它的下一个 token 不在这一块里），标签去掉第一个 token。
        logits = logits[:, :-1, :]
        targets = x[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        preds = torch.argmax(logits, dim=-1)
        return loss, preds.reshape(-1), targets.reshape(-1)

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self._shared_step(batch)
        acc = self.train_acc(preds, targets)

        self.log('train_loss', loss, on_epoch=True)
        self.log('train_acc', acc, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self._shared_step(batch)
        acc = self.val_acc(preds, targets)

        self.log('val_loss', loss, on_epoch=True)
        self.log('val_acc', acc, on_epoch=True)
        return loss

    def configure_optimizers(self):
        # 修正：原来 lr=1e-4 固定不变、weight_decay=1e-2 作用在所有参数上。
        # GPT-2 论文没有公开优化器超参，下面用 GPT-3 论文附录 B 和表 2.1 里 125M 模型（和 GPT-2 small 同规模）的设置：
        #   Adam β1=0.9 β2=0.95，lr 6e-4，前 3.75 亿 token 线性预热，之后余弦衰减到 10%，weight decay 0.1，梯度裁剪 1.0。
        # weight decay 只加在矩阵权重上；偏置和 LayerNorm 的参数是一维的，对它们衰减没有意义，反而会伤害模型。
        decay = [p for p in self.parameters() if p.dim() >= 2]
        no_decay = [p for p in self.parameters() if p.dim() < 2]
        optimizer = torch.optim.AdamW([
            {'params': decay, 'weight_decay': 0.1},
            {'params': no_decay, 'weight_decay': 0.0},
        ], lr=self.learning_rate, betas=(0.9, 0.95))

        total_steps = self.trainer.estimated_stepping_batches
        def lr_lambda(step):
            if step < self.warmup_steps:
                return (step + 1) / self.warmup_steps
            progress = min(1.0, (step - self.warmup_steps) / max(1, total_steps - self.warmup_steps))
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}

class CustomCheckpointCallback(pl.Callback):
    def __init__(self, every_n_steps, dirpath):
        self.every_n_steps = every_n_steps
        self.dirpath = dirpath

    # 修正：原来用的是 on_epoch_end，PyTorch Lightning 2.0 已经删掉了这个钩子，函数永远不会被调用。
    # 而且原来默认每 50 个 epoch 存一次，max_epochs 只有 20，就算钩子还在也一次都存不了。
    # 现在只跑一个 epoch（见下面 max_epochs 的说明），所以改成按优化器步数存。
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step > 0 and step % self.every_n_steps == 0 and step != getattr(self, '_last_saved', None):
            self._last_saved = step  # 梯度累积时同一个 global_step 会经过多个 batch，只存一次
            checkpoint_path = f"{self.dirpath}/checkpoint-step-{step}.ckpt"
            trainer.save_checkpoint(checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GPT model with checkpointing.')
    parser.add_argument('--checkpoint_every_n_steps', type=int, default=1000, help='Save checkpoint every n optimizer steps')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Resume training from checkpoint')
    args = parser.parse_args()

    # Initialize model and tokenizer
    dm = 768
    num_heads = 12
    num_layers = 12
    vocab_size = 50257
    max_seq_len = 1024
    tokenizer = load_tokenizer('tokenizer')
    # 修正：embedding 的行数必须不少于分词器实际的词表大小，否则遇到大 id 会越界。
    # tokenize_data.py 训分词器时用了 hard_vocab_limit=false，实际词表可能比 50257 小，但不能更大。
    assert tokenizer.get_piece_size() <= vocab_size

    # Load dataset
    # 修正：原来这里加载的是 deatos/fineweb-edu-mini-combined，而 down_data.py 下载并切分好的是
    # fineweb sample-10BT（README 写的也是 fineweb）。两边不是同一份数据，down_data.py 下的东西训练时根本没用上。
    # 而且原来 dataset['train']['text'] 会把全部文本读成一个 Python 列表放进内存，10B token 的语料放不下。
    dataset = load_and_split_dataset()
    # 修正：原来 num_workers=32，matpool 这台 A100 只有 12 个 CPU 核。按机器实际核数来。
    num_workers = min(8, os.cpu_count())

    train_dataset = TextDataset(dataset['train'], 'tokenizer', max_seq_len, num_proc=os.cpu_count())
    val_dataset = TextDataset(dataset['validation'], 'tokenizer', max_seq_len, num_proc=os.cpu_count())

    # 修正：原来一步只有 24 条 × 100 个 token ≈ 2400 个 token。GPT-3 论文里 125M 模型每步是 0.5M token。
    # 显存放不下这么大的 batch，所以用梯度累积：8 条 × 1024 token × 累积 60 次 = 491520 ≈ 0.5M token 才更新一次参数。
    batch_size = 8
    accumulate_grad_batches = 60
    tokens_per_step = batch_size * max_seq_len * accumulate_grad_batches
    warmup_steps = 375_000_000 // tokens_per_step  # GPT-3 论文：前 3.75 亿 token 预热

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    gpt_model = GPT(dm, num_heads, num_layers, vocab_size, max_seq_len, tokenizer)
    gpt_lightning_module = GPTLightningModule(gpt_model, tokenizer, vocab_size, warmup_steps)

    # Initialize W&B logger
    wandb_logger = WandbLogger(project='gpt_training')

    # Custom checkpoint callback
    custom_checkpoint_callback = CustomCheckpointCallback(
        every_n_steps=args.checkpoint_every_n_steps,
        dirpath='checkpoints/'
    )

    # Initialize trainer with checkpointing, W&B logger, and resume from checkpoint if needed
    trainer = Trainer(
        # 修正：原来 max_epochs=20。fineweb sample-10BT 一个 epoch 就是 100 亿 token，
        # 对 124M 的模型已经远超 Chinchilla 建议的约 25 亿 token，没必要重复 20 遍。
        max_epochs=1,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=-1 if torch.cuda.is_available() else 1,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=1.0,  # GPT-3 论文：梯度全局范数裁剪到 1.0
        # 修正：原来是默认的 fp32。A100 支持 bf16，混合精度能省一半激活显存、算得更快；
        # bf16 和 fp32 的指数范围相同，不需要 fp16 那样的 loss scaling。
        precision='bf16-mixed' if torch.cuda.is_available() else 32,
        # 修正：只有 1 个 epoch 的话，默认要到 epoch 结束才验证一次。改成每隔一段训练 batch 验一次，只取一部分验证集。
        val_check_interval=accumulate_grad_batches * args.checkpoint_every_n_steps,
        limit_val_batches=100,
        callbacks=[custom_checkpoint_callback],
        logger=wandb_logger
    )
    trainer.fit(gpt_lightning_module, train_loader, val_loader, ckpt_path=args.resume_from_checkpoint)

    # Save final model
    torch.save(gpt_model.state_dict(), 'model.pth')

    # Inference
    # 修正：原来这里写了一个 generate_text，里面用 model.device，而普通的 nn.Module 没有 .device 属性，会直接报 AttributeError。
    # decoder.py 里 GPT.generate 现在已经实现，直接调用它。
    # 另外原来的提示词是「白日依山尽」，但分词器和模型都是用英文的 fineweb 训的，几乎没见过中文，用中文提示词看不出模型学得怎么样。
    gpt_model.eval()
    prompt = "The meaning of life is"
    generated_text = gpt_model.generate(prompt)
    print(generated_text)
