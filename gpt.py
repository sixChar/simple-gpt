import torch
import math
import torch.nn as nn
import torch.optim as optim
import numpy as np


class MHAttn(nn.Module):
    def __init__(self, dims=512, key_dim=None, val_dim=None, num_heads=8):
        super().__init__()
        self.val_dim = val_dim or dims
        self.key_dim = key_dim or (dims // num_heads)
        self.W_q = nn.Linear(dims, self.key_dim * num_heads)
        self.W_k = nn.Linear(dims, self.key_dim * num_heads)
        self.W_v = nn.Linear(dims, self.val_dim)
        self.W_o = nn.Linear(self.val_dim * num_heads, dims)
        self.num_heads = num_heads

    def forward(self, query, key, value):
        q = self.W_q(query).reshape(query.shape[0], query.shape[1], self.key_dim, self.num_heads)
        k = self.W_k(key).reshape(key.shape[0], key.shape[1], self.key_dim, self.num_heads)
        v = self.W_v(value)

        mask = torch.triu(torch.full((query.shape[1], key.shape[1]), float('-inf')), diagonal=1)

        scores = torch.einsum("blkh,bmkh->bhlm", q, k) + mask
        attn = torch.softmax(scores / torch.sqrt(torch.full((1,), self.key_dim)), dim=-1)
        vals = torch.einsum("bhlm,bmv->blhv", attn, v).reshape(query.shape[0],query.shape[1], -1)
        return self.W_o(vals)
        



class GPTBlock(nn.Module):
    def __init__(self, model_dims, ff_dims=None, dropout=0.1):
        super().__init__()
        self.mha = MHAttn(dims=model_dims)
        self.model_dims = model_dims
        self.ff_dims = ff_dims or 4 * model_dims

        self.ff = nn.Sequential(
                    nn.Linear(model_dims, self.ff_dims),
                    nn.ReLU(),
                    nn.Linear(self.ff_dims, model_dims)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(model_dims)
        self.norm2 = nn.LayerNorm(model_dims)


    def forward(self, x):
        attn_out = self.mha(x, x, x)
        attn_out = self.dropout(attn_out)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        ff_out = self.dropout(ff_out)
        return self.norm2(x + ff_out)


def positional_encoding(dm, max_len=5000):
    position = torch.arange(max_len).unsqueeze(0)
    div_term = torch.exp(torch.arange(0, dm, 2) * (-math.log(10000.0) / dm))
    pe = torch.zeros(1, max_len, dm)
    pe[0, :, :dm//2] = torch.sin(torch.einsum("bl,d->bld", position, div_term))
    pe[0, :, dm//2:] = torch.cos(torch.einsum("bl,d->bld", position, div_term))
    return pe

class GPTModel(nn.Module):
    def __init__(self, n_tokens, dmodel, n_layers, max_len):
        super().__init__()
        self.embedding = nn.Linear(n_tokens, dmodel)
        self.pe = positional_encoding(dmodel, max_len=max_len)
        self.pe.requires_grad = False
        self.blocks = nn.ModuleList([GPTBlock(dmodel) for i in range(n_layers)])
        self.linear = nn.Linear(dmodel, n_tokens)
        self.sftm = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.pe[:, :x.size(1)]
        for block in self.blocks:
            x = block(x)
        return self.sftm(self.linear(x))




if __name__=="__main__":
    data = open("friedrich_list.html", "rb").read()
    unique_bytes = set(data)
    tkn_map = dict((a,b) for a,b in zip(unique_bytes, range(len(unique_bytes))))
    inv_tkn = dict(reversed(a) for a in tkn_map.items())
    num_tokens = len(unique_bytes)
    max_len = 5000

    def tokenize(bts):
        return list(map(lambda x: tkn_map[x], bts))
    
    def detokenize(tkns):
        return bytes(map(lambda x: inv_tkn[x], tkns))
    
    def gen_text(model, prompt, gen_len=200):
        if type(prompt) == str:
            prompt = prompt.encode()
        prompt = tokenize(prompt)
        gen = prompt

        with torch.no_grad():
            while len(gen) < gen_len:
                start = max(0, len(gen)-max_len)
                size = len(gen) - start
                x = np.zeros((size, num_tokens), dtype=np.float32)
                x[np.arange(size), gen[start:]] = 1.
                x = torch.from_numpy(x).unsqueeze(0)
                y = model(x)[0, -1]
                y = np.random.choice(np.arange(num_tokens), p=y.detach().numpy())
                gen.append(y)
        return detokenize(gen)

    model = GPTModel(len(tkn_map), 512, 6, max_len)
    opt = optim.Adam(model.parameters())

    train_steps = 10000
    batch_size = 3
    seq_len = 500
    for i in range(train_steps):
        batch_xs, batch_ys = [], []
        for j in range(batch_size):
            start = np.random.randint(0, len(data)-seq_len - 1) 
            tkns = tokenize(data[start:start+seq_len+1])

            x = np.zeros((seq_len, num_tokens), dtype=np.float32)
            x[np.arange(seq_len), tkns[:-1]] = 1.
            y = np.zeros((seq_len, num_tokens), dtype=np.float32)
            y[np.arange(seq_len), tkns[1:]] = 1.

            batch_xs.append(np.expand_dims(x, axis=0))
            batch_ys.append(np.expand_dims(y, axis=0))
        batch_xs = np.concatenate(batch_xs,axis=0)
        batch_ys = np.concatenate(batch_ys,axis=0)

        x = torch.from_numpy(batch_xs)
        y_ = torch.from_numpy(batch_ys)

        y = model(x)

        opt.zero_grad()
        loss = torch.mean(torch.sum(torch.square(y - y_), dim=-1))
        loss.backward()
        opt.step()
        
        if i % 10 == 0:
            print(f"Step: {i}")
            txt = gen_text(model, "The purpose of an American businessman ")
            print(txt)

   







    




  








