import math
import torch
import torch.nn as nn
from einops import rearrange
import ipdb


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, num_heads, resid_pdrop, block_size, group_size, casual=True):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc=nn.Linear(n_embd, 4 * n_embd),
            c_proj=nn.Linear(4 * n_embd, n_embd),
            act=NewGELU(),
            dropout=nn.Dropout(resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(
            m.c_proj(m.act(m.c_fc(x))))
        
        self.attn = nn.MultiheadAttention(
                embed_dim=n_embd, num_heads=num_heads, batch_first=True)

        # create attention mask
        self.casual = casual
        if self.casual:
            map_size = int(math.sqrt(group_size))            
            x = 1 - torch.tril(torch.ones(block_size, block_size))
            x = torch.repeat_interleave(x, map_size, dim=1)
            x = torch.repeat_interleave(x, map_size, dim=0)
            self.register_buffer("atten_mask", x.reshape(
                block_size*map_size, block_size*map_size).bool())

    def forward(self, x, return_atten=False):
        if self.casual:
            out, attention_weights = self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x),
                                           attn_mask=self.atten_mask)
        else:
            out, attention_weights = self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x))

        x = x + out
        x = x + self.mlpf(self.ln_2(x))

        if return_atten:
            return x, attention_weights
        else:
            return x


class CascadedTemporalModule(nn.Module):

    def __init__(self, group_size, block_size, n_embd, num_heads, embd_pdrop, n_layer):
        '''

        block_size: the length of sequence
        n_embd: number of dimension
        embd_pdrop: dropout rate
        n_layer: layers of model
        '''
        super().__init__()

        self.group_size = group_size
        self.block_size = block_size
        self.n_embd = n_embd
        
        self.map_size = int(math.sqrt(group_size))

        # CTT: casual temporal transformer     
        self.transformer = nn.ModuleDict(dict(
            wpe=nn.Embedding(50, n_embd),
            drop=nn.Dropout(embd_pdrop),
            h=nn.ModuleList([Block(n_embd, num_heads, embd_pdrop, block_size, group_size)
                            for _ in range(n_layer)]),
            ln_f=nn.LayerNorm(n_embd)
        ))
        
        # ST : spatial transformer
        self.transformer2 = nn.ModuleDict(dict(
            drop=nn.Dropout(embd_pdrop),
            h=nn.ModuleList([Block(n_embd, num_heads, embd_pdrop, block_size, group_size, casual=False)
                            for _ in range(n_layer)]),
            ln_f=nn.LayerNorm(n_embd)
        ))
        self.pos_embedding_img = nn.Parameter(torch.randn(1, group_size, n_embd))


        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02/math.sqrt(2 * n_layer))

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))
        self.n_layer = n_layer

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, x, study_date, return_atten=False):
        # input x: bz, T, 49, d
        # input study_date: bz, T
        b, t, g, d = x.size()
        assert d == self.n_embd, "n_embd is not equal to d"
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"

        study_date = torch.clamp(study_date.long(), 0, 49)
        pos_emb = self.transformer.wpe(study_date)
        pos_emb = pos_emb.unsqueeze(2).repeat([1, 1, g, 1])
        pos_emb = pos_emb.reshape(b,t,self.map_size,self.map_size,d)
        pos_emb = pos_emb.transpose(1,2) # ->b,h,t,w,d
        
        pos_emb = rearrange(pos_emb, "b h t w d -> (b h) (t w) d", h=self.map_size, w=self.map_size)
        
        x = x.reshape(b,t,self.map_size,self.map_size,d) # b,t,h,w,d
        x = x.transpose(1,2) # ->b,h,t,w,d
        x = rearrange(x, "b h t w d -> (b h) (t w) d", h=self.map_size, w=self.map_size)
        x = self.transformer.drop(x + pos_emb)
        for i in range(self.n_layer):
            x = self.transformer.h[i](x)
        x = self.transformer.ln_f(x)
        
        x = rearrange(x, "(b h) (t w) d -> b h t w d", b=b, h=self.map_size, t=t, w=self.map_size, d=d)
        x = x.transpose(1,2) #-> b,t,h,w,d
        x = rearrange(x, "b t h w d -> (b t) (h w) d", b=b, h=self.map_size, t=t, w=self.map_size, d=d)
        pos_emb_img = self.pos_embedding_img.unsqueeze(0).repeat([b, t, 1, 1])
        pos_emb_img = rearrange(pos_emb_img, "b t g d -> (b t) g d")
        
        x = self.transformer2.drop(x + pos_emb_img)
        for i in range(self.n_layer-1):
            x = self.transformer2.h[i](x)
        if return_atten:
            x, attention_weights = self.transformer2.h[-1](x, return_atten)
            x = self.transformer2.ln_f(x)
            return x, attention_weights
        else:
            x = self.transformer2.h[-1](x)
            x = self.transformer2.ln_f(x)
            return x


if __name__ == "__main__":
    study_date = torch.randint(0, 100, (2, 10)).float()
    study_date = torch.sort(study_date, dim=1)[0]

    model = CascadedTemporalModule(
        group_size=100,
        block_size=10,
        n_embd=512,
        num_heads=4,
        embd_pdrop=0.1,
        n_layer=2
    )

    x = torch.rand(2, 10, 100, 512)
    out = model(x, study_date, return_atten=True)
    print(out[1])
