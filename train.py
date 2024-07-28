import argparse
import os
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
import torch.nn.functional as F
from torchmetrics.classification import Accuracy
from datasets import load_dataset

from decoder import GPT
from tokenize_data import load_tokenizer
from pytorch_lightning import Trainer  # 确保导入 Trainer 类

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=100):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer.encode(text, out_type=int)
        tokens = tokens[:self.max_length]
        tokens = tokens + [0] * (self.max_length - len(tokens))  # Padding
        return torch.tensor(tokens)

class GPTLightningModule(pl.LightningModule):
    def __init__(self, gpt_model, tokenizer, vocab_size, learning_rate=1e-4):
        super(GPTLightningModule, self).__init__()
        self.gpt_model = gpt_model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.train_acc = Accuracy(task='multiclass', num_classes=vocab_size)
        self.val_acc = Accuracy(task='multiclass', num_classes=vocab_size)

    def forward(self, x, mask):
        return self.gpt_model(x, mask)

    def training_step(self, batch, batch_idx):
        x = batch.to(self.device)
        mask = (x != 0).unsqueeze(1).unsqueeze(2).expand(x.size(0), 1, x.size(1), x.size(1)).to(self.device)
        mask = mask & torch.tril(torch.ones((1, x.size(1), x.size(1)), device=x.device)).bool().to(self.device)
        logits = self(x, mask)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), x.view(-1), ignore_index=0)
        
        preds = torch.argmax(logits, dim=-1)
        non_pad_elements = x != 0
        acc = self.train_acc(preds[non_pad_elements].view(-1), x[non_pad_elements].view(-1))
        
        self.log('train_loss', loss, on_epoch=True)
        self.log('train_acc', acc, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch.to(self.device)
        mask = (x != 0).unsqueeze(1).unsqueeze(2).expand(x.size(0), 1, x.size(1), x.size(1)).to(self.device)
        mask = mask & torch.tril(torch.ones((1, x.size(1), x.size(1)), device=x.device)).bool().to(self.device)
        logits = self(x, mask)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), x.view(-1), ignore_index=0)
        
        preds = torch.argmax(logits, dim=-1)
        non_pad_elements = x != 0
        acc = self.val_acc(preds[non_pad_elements].view(-1), x[non_pad_elements].view(-1))
        
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_acc', acc, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-2)
        return optimizer

class CustomCheckpointCallback(pl.Callback):
    def __init__(self, every_n_epochs, dirpath):
        self.every_n_epochs = every_n_epochs
        self.dirpath = dirpath

    def on_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.every_n_epochs == 0:
            checkpoint_path = f"{self.dirpath}/checkpoint-epoch-{epoch}.ckpt"
            trainer.save_checkpoint(checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GPT model with checkpointing.')
    parser.add_argument('--checkpoint_every_n_epochs', type=int, default=50, help='Save checkpoint every n epochs')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Resume training from checkpoint')
    args = parser.parse_args()

    # Initialize model and tokenizer
    dm = 768
    num_heads = 12
    num_layers = 12
    vocab_size = 50257
    max_seq_len = 1024
    tokenizer = load_tokenizer('tokenizer')

    # Load dataset
    cache_dir = os.path.join(os.getcwd(), 'data')
    os.makedirs(cache_dir, exist_ok=True)
    dataset = load_dataset('deatos/fineweb-edu-mini-combined', cache_dir=cache_dir)
    
    train_texts = dataset['train']['text']
    val_texts = dataset['validation']['text']

    train_dataset = TextDataset(train_texts, tokenizer)
    val_dataset = TextDataset(val_texts, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True, num_workers=32)
    val_loader = DataLoader(val_dataset, batch_size=24, shuffle=False, num_workers=32)

    gpt_model = GPT(dm, num_heads, num_layers, vocab_size, max_seq_len, tokenizer)
    gpt_lightning_module = GPTLightningModule(gpt_model, tokenizer, vocab_size)

    # Initialize W&B logger
    wandb_logger = WandbLogger(project='gpt_training')

    # Custom checkpoint callback
    custom_checkpoint_callback = CustomCheckpointCallback(
        every_n_epochs=args.checkpoint_every_n_epochs,
        dirpath='checkpoints/'
    )

    # Initialize trainer with checkpointing, W&B logger, and resume from checkpoint if needed
    trainer = Trainer(
        max_epochs=20,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=-1 if torch.cuda.is_available() else 1,
        callbacks=[custom_checkpoint_callback],
        logger=wandb_logger
    )
    trainer.fit(gpt_lightning_module, train_loader, val_loader, ckpt_path=args.resume_from_checkpoint)

    # Save final model
    torch.save(gpt_model.state_dict(), 'model.pth')

    # Inference
    def generate_text(model, tokenizer, prompt, max_length=100):
        model.eval()
        prompt = tokenizer.encode(prompt, out_type=int)
        input_ids = torch.tensor(prompt).unsqueeze(0).to(model.device)

        with torch.no_grad():
            for _ in range(max_length):
                mask = torch.tril(torch.ones((input_ids.size(1), input_ids.size(1)), device=input_ids.device)).bool()
                logits = model(input_ids, mask.unsqueeze(0))
                next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                if next_token.item() == tokenizer.eos_id():
                    break

        return tokenizer.decode(input_ids.squeeze().tolist())

    prompt = "白日依山尽"
    generated_text = generate_text(gpt_model, tokenizer, prompt)
    print(generated_text)
