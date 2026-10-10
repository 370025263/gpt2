"""
torch Network of the GPT encoder
"""
import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self,dm,num_heads,max_seq_len):
            super(MultiHeadAttention,self).__init__()
            # parameters
            self.dk = dm // num_heads
            self.dm = dm
            self.num_heads = num_heads
            self.scale = math.sqrt(self.dk)
            # attention weights
            self.Wq = nn.Linear(dm,dm)
            self.Wk = nn.Linear(dm,dm)
            self.Wv = nn.Linear(dm,dm)
            self.Wo = nn.Linear(dm,dm)
            self.dropout = nn.Dropout(0.1)
            # 修正：RoPE 要作用在每一层、每个头的 q 和 k 上，维度是 dk 而不是 dm。
            # 原来在 GPT 里对 embedding 做一次，等于把它当成了加在输入上的绝对位置编码，
            # 而 RoPE 的意义在于 q·k 的点积只依赖两个位置的差，只有转 q、k 才成立。
            self.rope = RoPE(self.dk, max_seq_len)

    def forward(self,x,mask=None):
        """ forward pass of MHA, GPT-3 uses GQA,here is MHA"""
        batch_size = x.size(0)
        q = self.Wq(x)  # (batch_size,seq_len,dm)
        k = self.Wk(x)
        v = self.Wv(x)
        # 1.head split
        ## (batch_size,seq_len,dm) -> (batch_size,seq_len, h , dk)
        q = q.view(batch_size,-1, self.num_heads, self.dk)
        k = k.view(batch_size,-1, self.num_heads, self.dk)
        v = v.view(batch_size,-1, self.num_heads, self.dk)
        # 2.transpose
        q,k,v = q.transpose(1,2),k.transpose(1,2),v.transpose(1,2)
        # 修正：在 q、k 拆完头之后旋转，v 不旋转
        q,k = self.rope(q),self.rope(k)
        # 3.score
        A = torch.matmul(q,k.transpose(-2,-1))/self.scale
        # 4.mask
        if mask is not None:
            #mask = mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
            # 修正：原来填 -1e9。fp16 的最大值只有 65504，-1e9 在半精度下会直接报溢出错误。
            # 用当前 dtype 能表示的最小值，fp32、bf16、fp16 都安全。
            A = A.masked_fill(mask==0,torch.finfo(A.dtype).min)  # 确保softmax后为0，因此mask为0的地方为负无穷
        # 5.softmax
        A = torch.nn.functional.softmax(A,dim=-1)
        A = self.dropout(A)
        # 6.attention
        o = torch.matmul(A,v)  # (batch_size,h,seq_len,dk)
        # 7.transpose and head merge
        #print(f"A*v = o shape:{o.shape}")
        o = o.transpose(1,2).contiguous().view(batch_size,-1,self.dm)  # contiguous()保证内存连续
        o = self.Wo(o)
        return o


class RoPE(nn.Module):
    """ Rotary Positional Embedding（旋转位置编码，论文 RoFormer）

    修正：原来的实现不是旋转。
    1. 第一行 out 马上被第二行覆盖，第一行没有任何作用。
    2. 第二行算的是 [x偶*cos, x奇*sin] 再拼接：每一维只乘了一个三角函数，没有两维之间的混合，
       不是旋转，向量长度也变了；位置 0 时 sin=0，一半的维度直接被清零。
    3. 拼接后维度顺序变成「先偶后奇」，和输入的维度对不上。
    正确做法：把相邻两维 (x[2i], x[2i+1]) 看成一个二维向量，按位置 m 转过角度 m*θ_i：
        y[2i]   = x[2i]*cos(mθ_i) - x[2i+1]*sin(mθ_i)
        y[2i+1] = x[2i]*sin(mθ_i) + x[2i+1]*cos(mθ_i)
    这样 q_m·k_n 只依赖 m-n，这才是「相对」位置编码的来源。

    另外提一句：原版 GPT-2 用的是可学习的绝对位置编码（wpe），不是 RoPE。
    README 里选了 RoPE，这里保留 RoPE，只是把实现改对。
    """
    def __init__(self, dk, max_seq_len):
        super(RoPE, self).__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dk, 2).float() / dk))  # θ_i, (dk/2,)
        pos_seq = torch.arange(0, max_seq_len).float()
        freqs = torch.einsum("i,d->id", pos_seq, inv_freq)  # m*θ_i, (max_seq_len, dk/2)
        # 修正：原来每次 forward 都 .to(device)。注册成 buffer 后会跟着 model.to() 一起搬到 GPU。
        # persistent=False 表示不进 state_dict，它可以随时重新算出来。
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x):
        """ x: (batch_size, num_heads, seq_len, dk)"""
        seq_len = x.size(-2)
        cos, sin = self.cos[:seq_len], self.sin[:seq_len]  # (seq_len, dk/2)，自动广播到 batch 和 head
        x_even, x_odd = x[..., ::2], x[..., 1::2]
        out = torch.stack([x_even * cos - x_odd * sin,
                           x_even * sin + x_odd * cos], dim=-1)  # (..., dk/2, 2)
        return out.flatten(-2)  # 交错放回原来的位置：0,1,2,3...


class FeedForward(nn.Module):
    def __init__(self,dm,dff):
        super(FeedForward,self).__init__()
        self.linear1 = nn.Linear(dm,dff)
        self.linear2 = nn.Linear(dff,dm)
        # 修正：GPT（GPT-1 论文起）用的是 GELU，不是原始 Transformer 的 ReLU。
        # approximate="tanh" 和 OpenAI 发布的 GPT-2 代码里的 gelu 公式一致。
        self.act = nn.GELU(approximate="tanh")

    def forward(self,x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x


class DecoderLayer(nn.Module):
    """pre-norm,residual connection before layer norm"""
    def __init__(self,dm,num_heads,dff,max_seq_len):
        super(DecoderLayer,self).__init__()
        self.mha = MultiHeadAttention(dm,num_heads,max_seq_len)
        self.ffn = FeedForward(dm,dff)
        self.layernorm1 = nn.LayerNorm(dm)
        self.layernorm2 = nn.LayerNorm(dm)
        self.dropout = nn.Dropout(0.1)

    def forward(self,x,mask):
        # 修正：原来是 x = x + LN(MHA(x))，LayerNorm 放在子层的输出上，这既不是 pre-norm 也不是 post-norm。
        # GPT-2 论文 2.3 节：layer normalization 移到每个子块的输入处，即 pre-norm：
        #     x = x + Sublayer(LN(x))
        # 残差这条路上没有任何归一化，梯度可以直接从顶层流到底层，深层网络才好训。
        # MHA
        x = x + self.dropout(self.mha(self.layernorm1(x),mask))
        # FFN
        x = x + self.dropout(self.ffn(self.layernorm2(x)))
        return x


class Embedding(nn.Module):
    def __init__(self,dm,vocab_size):
        super(Embedding,self).__init__()
        self.embedding = nn.Embedding(vocab_size,dm)
        self.dm = dm
    def forward(self,x):
        # 修正：原来乘了 sqrt(dm)（原代码注释问 why multiply sqrt(dm)?）。
        # 那是原始 Transformer 的做法：它的 embedding 和输出层共享，初始化很小，
        # 乘 sqrt(dm) 是为了让它和正弦位置编码的量级相当。
        # 这里 nn.Embedding 默认按 N(0,1) 初始化，再乘 sqrt(768)≈27.7，输入的数值一开始就非常大。
        # GPT-2 不乘这个系数，embedding 按 N(0,0.02) 初始化（见 GPT._init_weights）。
        return self.embedding(x)


class GPT(nn.Module):
    def __init__(self,dm,num_heads,num_layers,vocab_size,max_seq_len,tokenizer=None):
        super(GPT,self).__init__()
        # tokenizer
        self.tokenizer = tokenizer
        # embeding
        self.embedding = Embedding(dm,vocab_size)
        # decoder
        self.dm = dm
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        # 修正：RoPE 挪到每层的 MultiHeadAttention 里了，见 MultiHeadAttention.__init__
        self.decoder_layers = nn.ModuleList([DecoderLayer(dm,num_heads,dm*4,max_seq_len) for _ in range(num_layers)])  # why dm*4?,the explanation is in the paper
        # 修正：pre-norm 结构下，最后一层的输出没有经过归一化，所以要在输出层之前补一个 LayerNorm。
        # GPT-2 论文 2.3 节：an additional layer normalization was added after the final self-attention block.
        self.layernorm_final = nn.LayerNorm(dm)
        # 修正：输出层和 embedding 共享同一个权重矩阵（OpenAI 发布的 GPT-2 代码里 logits 直接用 wte 算）。
        # 50257*768 ≈ 3860 万参数，占 124M 的三成左右；共享后两边都更新同一份词向量。
        self.linear = nn.Linear(dm,vocab_size,bias=False)
        self.linear.weight = self.embedding.embedding.weight
        # 修正：原来用的是 PyTorch 各层的默认初始化，GPT-2 不是这样初始化的
        self.apply(self._init_weights)
        # GPT-2 论文 2.3 节：残差路径上的输出投影按 1/sqrt(N) 缩放，N 是残差层数（每个 block 有两层，所以是 2*num_layers）。
        # 不缩放的话，每加一层残差，方差就累加一点，层数一多，输出的数值越来越大。
        for name, p in self.named_parameters():
            if name.endswith("mha.Wo.weight") or name.endswith("ffn.linear2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * num_layers))

    def _init_weights(self, module):
        # 和 OpenAI 的 GPT-2 代码一致：权重 N(0,0.02)，偏置为 0
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self,x,mask=None):
        """ x: (batch_size,seq_len) 的 token id"""
        # 修正：GPT 是自回归模型，任何时候都必须带因果掩码。
        # 原来 infer.py 传 mask=None，推理时注意力变成双向的，前面的位置能看到后面的 token，
        # 中间层的表示和训练时不一样，最后一个位置的输出也就跟着变了。所以 mask 为空时默认构造下三角掩码。
        if mask is None:
            seq_len = x.size(1)
            mask = torch.tril(torch.ones(seq_len,seq_len,device=x.device)).bool()
        x = self.embedding(x)
        for decoder in self.decoder_layers:
            x = decoder(x,mask)
        x = self.layernorm_final(x)
        x = self.linear(x)
        return x

    @torch.no_grad()
    def generate(self,prompt,max_new_tokens=100):
        """ generate text"""
        # 修正：原来这个函数只编码了 prompt 就结束了，什么都没生成。下面是最简单的贪心解码。
        # prompt
        device = next(self.parameters()).device
        input_ids = torch.tensor([self.tokenizer.encode(prompt,out_type=int)],device=device)
        for _ in range(max_new_tokens):
            # 超过训练长度的上下文要截掉：RoPE 只算了 max_seq_len 个位置，模型也只见过这么长
            context = input_ids[:, -self.max_seq_len:]
            logits = self(context)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if next_token.item() == self.tokenizer.eos_id():
                break
        return self.tokenizer.decode(input_ids[0].tolist())



if __name__ == "__main__":
    # param
    dm = 512
    num_heads = 8
    num_layers = 6
    vocab_size = 30000
    max_seq_len = 512
    batch_size = 1

    # model
    model = GPT(dm,num_heads,num_layers,vocab_size,max_seq_len)
    x = torch.randint(0,vocab_size,(batch_size,max_seq_len)) # (batch_size,seq_len)
    # mask (batch_size,seq_len,seq_len) 下三角矩阵
    mask = torch.tril(torch.ones(max_seq_len,max_seq_len))
    print(mask.shape)
    print(x.shape)
    out = model(x,mask)
    print(out.shape)  # (batch_size,seq_len,vocab_size)




