import torch
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_seq_len=128):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        with open(file_path, 'r') as f:
            self.lines = f.readlines()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        text = self.lines[idx]
        encoded = self.tokenizer.encode(text.strip(), out_type=int)
        if len(encoded) > self.max_seq_len:
            encoded = encoded[:self.max_seq_len]
        else:
            encoded += [0] * (self.max_seq_len - len(encoded))  # Padding
        return torch.tensor(encoded)


if __name__ == '__main__':
    # load tokenizer
    import sentencepiece as spm
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load('tokenizer.model')
    train_dataset = TextDataset(file_path='data/train.txt', tokenizer = tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

