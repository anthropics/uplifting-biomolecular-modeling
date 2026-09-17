"""Exact fused rel-pos attention for enformer-pytorch: the replacement Attention forward (cuBLAS content GEMM as stock -> rel band GEMM
(custom kernel, cuBLAS's K order) -> fused add + softmax (cunn's warp-persistent order) -> cuBLAS attn.v as stock -> to_out), bitwise per layer
against the stock forward. The kernels are ``enformer_xattn.cu``; ``SO_PATH`` (set by kits.v0_2.attach) loads the prebuilt object, else nvcc
builds it from the source (load_inline)."""
import os
import time

import torch
from einops import rearrange
from enformer_pytorch.modeling_enformer import get_positional_embed, relative_shift

SO_PATH = None                     # the prebuilt extension object (kits.v0_2.attach sets it: the kit's own sm_90 object or a class build)

_EXT = None

def ext():
    global _EXT
    if _EXT is None:
        so = SO_PATH
        if so:
            import importlib.util
            spec = importlib.util.spec_from_file_location("enformer_xattn_ext", so); _EXT = importlib.util.module_from_spec(spec); spec.loader.exec_module(_EXT)
        else:
            from torch.utils.cpp_extension import load_inline
            src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "enformer_xattn.cu")).read()
            t0 = time.time()
            _EXT = load_inline(name="enformer_xattn_ext", cpp_sources="", cuda_sources=src, functions=None, extra_cuda_cflags=["-O3"], verbose=False, is_python_module=True)
            print("nvcc build s", round(time.time() - t0, 1), flush=True)
    return _EXT

ORDER = {"order": 0}

SOFTMAX = {"fn": "add2_softmax_exact"}

def relk_of(self, n, device):
    key = (n, str(device)); cache = self.__dict__.setdefault("_xattn_relk", {})
    if key not in cache:
        positions = get_positional_embed(n, self.num_rel_pos_features, device, use_tf_gamma=self.use_tf_gamma)
        positions = self.pos_dropout(positions)
        with torch.no_grad():
            cache[key] = rearrange(self.to_rel_k(positions), "n (h d) -> h n d", h=self.heads).contiguous()
    return cache[key]

KERNEL_N = 1536                    # the attention length the kernels are built for (a 196,608-bp window after the conv tower's pooling); rel_band_gemm takes 64-wide heads
KERNEL_DIM_KEY = 64


def exact_forward(self, x):
    """Stock order up to q/k/v; content GEMM = the stock's einsum; positional band + add + softmax = the exact kernels; attn.v and to_out = stock.
    A geometry the kernels are not built for (another window length, another key width) runs the stock's own attention arithmetic with the
    per-length cached positional basis (``relk_of``) — every length the stock accepts is served, exactly, and no cached tensor a graph reads
    is replaced by a call at another length."""
    n, h, device = x.shape[-2], self.heads, x.device
    q = self.to_q(x); k = self.to_k(x); v = self.to_v(x)
    if n != KERNEL_N or q.shape[-1] != h * KERNEL_DIM_KEY:              # a geometry the kernels are not built for: the stock's own arithmetic below
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
        q = q * self.scale
        content_logits = torch.einsum("b h i d, b h j d -> b h i j", q + self.rel_content_bias, k)
        rel_logits = relative_shift(torch.einsum("b h i d, h j d -> b h i j", q + self.rel_pos_bias, relk_of(self, n, device)))
        attn = self.attn_dropout((content_logits + rel_logits).softmax(dim=-1))
        out = rearrange(torch.einsum("b h i j, b h j d -> b h i d", attn, v), "b h n d -> b n (h d)")
        return self.to_out(out)
    q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
    q = q * self.scale
    content_logits = torch.einsum("b h i d, b h j d -> b h i j", q + self.rel_content_bias, k)
    rel_k = relk_of(self, n, device)
    rel_band = ext().rel_band_gemm((q + self.rel_pos_bias).contiguous(), rel_k, ORDER["order"])
    attn = getattr(ext(), SOFTMAX["fn"])(content_logits.contiguous(), rel_band)
    attn = self.attn_dropout(attn)
    out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
    out = rearrange(out, "b h n d -> b n (h d)")
    return self.to_out(out)
