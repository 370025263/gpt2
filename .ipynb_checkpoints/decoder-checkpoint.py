"""
torch Network of the GPT encoder
"""
import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self,dm,num_heads):
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
        # 3.score
        A = torch.matmul(q,k.transpose(-2,-1))/self.scale
        # 4.mask
        if mask is not None:
            #mask = mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
            A = A.masked_fill(mask==0,-1e9)  # 确保softmax后为0，因此mask为0的地方为负无穷
        # 5.softmax
        A = torch.nn.functional.softmax(A,dim=-1)
        A = self.dropout(A)
        # 6.attention
        o = torch.matmul(A,v)  # (batch_size,seq_len,h,dk)
        # 7.transpose and head merge
        #print(f"A*v = o shape:{o.shape}")
        o = o.transpose(1,2).contiguous().view(batch_size,-1,self.dm)  # contiguous()保证内存连续
        o = self.Wo(o)
        return o


class RoPE(nn.Module):
    """ Relative Positional Encoding"""
    def __init__(self, dm):
        super(RoPE, self).__init__()
        self.dm = dm
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, dm, 2).float() / dm))

    def forward(self, x):
        device = x.device  # Get the device of the input tensor
        seq_len = x.size(1)
        inv_freq = self.inv_freq.to(device)  # Ensure inv_freq is on the same device as x
        pos_seq = torch.arange(0, seq_len, device=device).type_as(inv_freq)
        sinusoid_inp = torch.einsum("i,d->id", pos_seq, inv_freq)
        pos_emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        pos_emb = pos_emb.unsqueeze(0).repeat(x.size(0), 1, 1)

        x_2d = x.reshape(-1, x.size(-1))
        pos_emb_2d = pos_emb.reshape(-1, pos_emb.size(-1))

        out = torch.cat([-x_2d[:, 1::2] * pos_emb_2d[:, 1::2], x_2d[:, ::2] * pos_emb_2d[:, ::2]], dim=-1)
        out = torch.cat([x_2d[:, ::2] * pos_emb_2d[:, 1::2], x_2d[:, 1::2] * pos_emb_2d[:, ::2]], dim=-1)

        return out.reshape(x.size())


class FeedForward(nn.Module):
    def __init__(self,dm,dff):
        super(FeedForward,self).__init__()
        self.linear1 = nn.Linear(dm,dff)
        self.linear2 = nn.Linear(dff,dm)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class DecoderLayer(nn.Module):
    """pre-norm,residual connection before layer norm"""
    def __init__(self,dm,num_heads,dff):
        super(DecoderLayer,self).__init__()
        self.mha = MultiHeadAttention(dm,num_heads)
        self.ffn = FeedForward(dm,dff)
        self.layernorm1 = nn.LayerNorm(dm)
        self.layernorm2 = nn.LayerNorm(dm)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self,x,mask):
        # MHA
        x1 = self.mha(x,mask)
        x1 = self.dropout(x1)
        x1 = self.layernorm1(x1)  # pre-norm
        x = x + x1
        # FFN
        x2 = self.ffn(x)
        x2 = self.dropout(x2)
        x2 = self.layernorm2(x2)  # pre-norm
        x = x + x2
        return x


class Embedding(nn.Module):
    def __init__(self,dm,vocab_size):
        super(Embedding,self).__init__()
        self.embedding = nn.Embedding(vocab_size,dm)
        self.dm = dm
    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.dm)  # why multiply sqrt(dm)?


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
        self.rope = RoPE(dm)
        self.decoder_layers = nn.ModuleList([DecoderLayer(dm,num_heads,dm*4) for _ in range(num_layers)])  # why dm*4?,the explanation is in the paper 
        self.linear = nn.Linear(dm,vocab_size)
    
    def forward(self,x,mask):
        """ x: (batch_size,seq_len,dm)"""
        x = self.embedding(x)
        x = self.rope(x)
        for decoder in self.decoder_layers:
            x = decoder(x,mask)
        x = self.linear(x)
        return x

    def generate(self,prompt):
        """ generate text"""
        # prompt
        prompt = self.tokenizer.encode(prompt)



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





