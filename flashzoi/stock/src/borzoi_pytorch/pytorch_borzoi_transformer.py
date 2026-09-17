#Adapted from https://github.com/lucidrains/enformer-pytorch/tree/main
#
#MIT License
#
#Copyright (c) 2021 Phil Wang, 2024 Johannes Hingerl

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.
# =========================================================================

import torch.nn as nn
from torch import nn, einsum
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
import torch
import math


def get_positional_features_central_mask(positions, features, seq_len):
    pow_rate = math.exp(math.log(seq_len + 1) / features)
    center_widths = torch.pow(pow_rate, torch.arange(1, features + 1, device = positions.device)).float()
    center_widths = center_widths - 1
    return (center_widths[None, ...] > positions.abs()[..., None]).float()


def get_positional_embed(seq_len, seq_len_train, feature_size, device):
    distances = torch.arange(-seq_len + 1, seq_len, device = device)

    feature_functions = [
        get_positional_features_central_mask,
    ]

    num_components = len(feature_functions) * 2

    if (feature_size % num_components) != 0:
        raise ValueError(f'feature size is not divisible by number of components ({num_components})')

    num_basis_per_class = feature_size // num_components

    embeddings = []
    for fn in feature_functions:
        embeddings.append(fn(distances, num_basis_per_class, seq_len_train))

    embeddings = torch.cat(embeddings, dim = -1)
    embeddings = torch.cat((embeddings, torch.sign(distances)[..., None] * embeddings), dim = -1)
    return embeddings

def fast_relative_shift(a,b):
    return einsum("i d, j d -> i j", a, b).flatten().as_strided(size =(a.shape[0],a.shape[0]), stride= ((a.shape[0]-1)*2,1), storage_offset = a.shape[0] - 1)

fast_relative_shift= torch.vmap(torch.vmap(fast_relative_shift), in_dims=(0, None)) #https://johahi.github.io/blog/2024/fast-relative-shift/


class Attention(nn.Module):
    
    def __init__(
        self,
        dim=1536,
        *,
        num_rel_pos_features = 1,
        heads = 8,
        dim_key = 64,
        dim_value = 64,
        dropout = 0.,
        pos_dropout = 0.
    ):
        super().__init__()
        self.scale = dim_key ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(dim, dim_key * heads, bias = False)
        self.to_k = nn.Linear(dim, dim_key * heads, bias = False)
        self.to_v = nn.Linear(dim, dim_value * heads, bias = False)

        self.to_out = nn.Linear(dim_value * heads, dim)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

        # relative positional encoding

        self.num_rel_pos_features = num_rel_pos_features

        self.positions = None
        #self.register_buffer("positions",get_positional_embed(4096, 4096, self.num_rel_pos_features, self.to_v.weight.device), persistent = False) # 4096 as this should always be the seq len at this pos?

        self.to_rel_k = nn.Linear(num_rel_pos_features, dim_key * heads, bias = False)
        self.rel_content_bias = nn.Parameter(torch.randn(1, heads, 1, dim_key))
        self.rel_pos_bias = nn.Parameter(torch.randn(1, heads, 1, dim_key))

        # dropouts

        self.pos_dropout = nn.Dropout(pos_dropout)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        n, h, device = x.shape[-2], self.heads, x.device
        
        if self.positions is None :
            self.positions = get_positional_embed(n, 4096, self.num_rel_pos_features, self.to_v.weight.device)
        
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        q = q * self.scale

        content_logits = einsum('b h i d, b h j d -> b h i j', q + self.rel_content_bias, k)

        positions = self.pos_dropout(self.positions)
        rel_k = self.to_rel_k(positions)
        rel_k = rearrange(rel_k, 'n (h d) -> h n d', h = h)
        rel_logits = fast_relative_shift(q + self.rel_pos_bias,rel_k)
        logits = content_logits + rel_logits
        attn = logits.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out
    
class FlashAttention(nn.Module):
    def __init__(
        self,
        dim=1536,
        heads = 8,
        dropout = 0.15,
        pos_dropout = 0.15, # Not used
        rotary_emb_base = 20000.0,
        rotary_emb_scale_base = None, 
        ):
        super().__init__()

        from flash_attn.modules.mha import MHA
        self.mha = MHA(
            use_flash_attn=True,
            embed_dim=dim,
            num_heads = heads,
            num_heads_kv = (heads//2),
            qkv_proj_bias=True,#False,
            out_proj_bias=True,
            dropout=dropout,
            softmax_scale=(dim/heads) ** -0.5,
            causal=False,
            rotary_emb_dim=128,
            rotary_emb_base=rotary_emb_base,
            rotary_emb_scale_base = rotary_emb_scale_base,
            fused_bias_fc = False,
        ) 

        nn.init.kaiming_normal_(self.mha.Wqkv.weight, nonlinearity = 'relu')
        nn.init.zeros_(self.mha.out_proj.weight)
        nn.init.zeros_(self.mha.out_proj.bias)
        nn.init.ones_(self.mha.Wqkv.bias)


    def forward(self, x):
        out = self.mha(x)
        return out
