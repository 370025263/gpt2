import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from decoder import GPT
from tokenize_data import load_tokenizer
from dataset import TextDataset


class GPTLightningModule(pl.LightningModule):
    def __init__(self, gpt_model, tokenizer, learning_rate=1e-4):
        super(GPTLightningModule, self).__init__()
        self.gpt_model = gpt_model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate

    def forward(self, x, mask):
        return self.gpt_model(x, mask)

    def training_step(self, batch, batch_idx):
        x = batch
        mask = (x != 0).unsqueeze(1).unsqueeze(2).expand(x.size(0), 1, x.size(1), x.size(1))
        mask = mask & torch.tril(torch.ones((1, x.size(1), x.size(1)), device=x.device)).bool()
        print(f"mask shape: {mask.shape}")
        print(f"x shape: {x.shape}")
        logits = self(x, mask)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), x.view(-1), ignore_index=0)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer


if __name__ == '__main__':
    # Initialize model and tokenizer
    dm = 512
    num_heads = 8
    num_layers = 6
    vocab_size = 50  # 320000
    max_seq_len = 30
    ## Load tokenizer
    tokenizer = load_tokenizer('tokenizer')
    ## load data
    train_dataset = TextDataset('data/train.txt', tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    ## Initialize model
    gpt_model = GPT(dm, num_heads, num_layers, vocab_size, max_seq_len, tokenizer)
    gpt_lightning_module = GPTLightningModule(gpt_model, tokenizer)
    
    # Initialize trainer
    from pytorch_lightning import Trainer

    trainer = Trainer(max_epochs=200)
    trainer.fit(gpt_lightning_module, train_loader)
    ## save model
    torch.save(gpt_model.state_dict(), 'model.pth')
    
    # Inference
    def generate_text(model, tokenizer, prompt, max_length=100):
        model.eval()
        prompt = tokenizer.encode(prompt, out_type=int)
        input_ids = torch.tensor(prompt).unsqueeze(0)

        with torch.no_grad():
            for _ in range(max_length):
                mask = torch.tril(torch.ones((input_ids.size(1), input_ids.size(1)), device=input_ids.device)).bool()
                logits = model(input_ids, mask.unsqueeze(0))
                next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                if next_token.item() == tokenizer.eos_id():
                    break

        return tokenizer.decode(input_ids.squeeze().tolist())

    prompt = "The dog said"
    generated_text = generate_text(gpt_model, tokenizer, prompt)
    print(generated_text)



