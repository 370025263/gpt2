import torch
from decoder import GPT, RoPE
from tokenize_data import load_tokenizer

def load_model(model_path, dm, num_heads, num_layers, vocab_size, max_seq_len, tokenizer):
    model = GPT(dm, num_heads, num_layers, vocab_size, max_seq_len, tokenizer)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def generate_text(model, tokenizer, prompt, max_length=100):
    print(f"Generating text for prompt: {prompt}")
    prompt_ids = tokenizer.encode(prompt, out_type=int)
    print(f"prompt_ids: {prompt_ids}")
    input_ids = torch.tensor(prompt_ids).unsqueeze(0)
    print(f" shape of input_ids: {input_ids.shape}")

    print(prompt, end='', flush=True)

    with torch.no_grad():
        for _ in range(max_length):
            logits = model(input_ids, None)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
            next_token_id = next_token.item()
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            next_token_str = tokenizer.decode([next_token_id])
            print(next_token_str, end='', flush=True)

            if next_token_id == tokenizer.eos_id():
                break
    print()

if __name__ == '__main__':
    # Parameters
    dm = 512
    num_heads = 8
    num_layers = 6
    vocab_size = 52
    max_seq_len = 128
    model_path = 'model.pth'

    # Load tokenizer
    tokenizer = load_tokenizer('tokenizer')

    # Load model
    model = load_model(model_path, dm, num_heads, num_layers, vocab_size, max_seq_len, tokenizer)

    # Generate text
    prompt = "The dog said"
    generate_text(model, tokenizer, prompt)

