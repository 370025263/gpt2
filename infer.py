import torch
from decoder import GPT, RoPE
from tokenize_data import load_tokenizer

def load_model(model_path, dm, num_heads, num_layers, vocab_size, max_seq_len, tokenizer):
    model = GPT(dm, num_heads, num_layers, vocab_size, max_seq_len, tokenizer)
    # 修正：加 map_location，在 GPU 上训练保存的权重也能在没有 GPU 的机器上加载
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def generate_text(model, tokenizer, prompt, max_length=100):
    print(f"Generating text for prompt: {prompt}")
    prompt_ids = tokenizer.encode(prompt, out_type=int)
    print(f"prompt_ids: {prompt_ids}")
    input_ids = torch.tensor(prompt_ids).unsqueeze(0)
    print(f" shape of input_ids: {input_ids.shape}")

    printed = tokenizer.decode(prompt_ids)
    print(printed, end='', flush=True)

    with torch.no_grad():
        for _ in range(max_length):
            # 修正：原来传 mask=None，模型里就没有因果掩码，推理时的注意力是双向的，和训练对不上。
            # 现在 GPT.forward 在 mask 为空时会自动用下三角掩码，这里显式传一个，意思更清楚。
            # 同时上下文超过 max_seq_len 时只保留最后 max_seq_len 个 token。
            context = input_ids[:, -model.max_seq_len:]
            mask = torch.tril(torch.ones(context.size(1), context.size(1))).bool()
            logits = model(context, mask)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
            next_token_id = next_token.item()
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            # 修正：原来对单个 token 调 decode([next_token_id])。SentencePiece 用「▁」表示词前的空格，
            # 单独解码一个 token 时会把开头的空格去掉，输出就变成「modelspredictthenexttoken」这样词都粘在一起。
            # 改成把整段重新解码，只打印比上次多出来的部分。
            text = tokenizer.decode(input_ids[0].tolist())
            print(text[len(printed):], end='', flush=True)
            printed = text

            if next_token_id == tokenizer.eos_id():
                break
    print()

if __name__ == '__main__':
    # Parameters
    # 修正：原来是 dm=512、8 头、6 层、vocab_size=52、max_seq_len=128，和 train.py 里训练用的
    # 768、12、12、50257、1024 都不一样，load_state_dict 会因为形状不匹配直接报错。推理的结构必须和训练完全相同。
    dm = 768
    num_heads = 12
    num_layers = 12
    vocab_size = 50257
    max_seq_len = 1024
    model_path = 'model.pth'

    # Load tokenizer
    tokenizer = load_tokenizer('tokenizer')

    # Load model
    model = load_model(model_path, dm, num_heads, num_layers, vocab_size, max_seq_len, tokenizer)

    # Generate text
    prompt = "The dog said"
    generate_text(model, tokenizer, prompt)
