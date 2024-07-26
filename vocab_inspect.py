import sentencepiece as spm

from tokenize_data import load_tokenizer


def save_vocab(tokenizer, file_path):
    vocab_size = tokenizer.get_piece_size()
    with open(file_path, 'w') as f:
        for i in range(vocab_size):
            f.write(f"{i}: {tokenizer.id_to_piece(i)}\n")

if __name__ == '__main__':
    # Load tokenizer
    tokenizer = load_tokenizer('tokenizer')

    # Save vocabulary to a file
    save_vocab(tokenizer, 'vocab.txt')

