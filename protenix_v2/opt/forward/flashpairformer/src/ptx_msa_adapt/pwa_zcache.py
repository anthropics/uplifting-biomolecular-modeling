"""pwa_zcache — EXACT lever 'PWA_ZCACHE': compute the pair-derived softmax weights of MSAPairWeightedAveraging ONCE per MSAStack call instead of
once per 2048-row MSA chunk (stock MSAStack.inference_forward calls the PWA ceil(n_seq/2048) times with the SAME z; each call recomputes
LayerNorm_z(z) [N,N,256] -> Linear(256->8) -> softmax over j).  Scope of the cache = one MSAStack.inference_forward call, during which z is provably
untouched (PWA/transition_m write only into m) => identical kernels on identical inputs => bitwise vs stock (checked by tests/test_pwa_zcache_gpu.py).
No effect when n_seq <= 2048 (one chunk).  Training / grad-enabled / non-eval paths untouched.
install() / uninstall(); env PTX_PWA_ZCACHE=0 disables install()."""
import os, torch
import protenix.model.modules.pairformer as PF

_ORIG = {}

def _pwa_forward_cached(self, m, z):
    w = getattr(self, "_fpf_zcache_w", None)
    if w is None or self.training or torch.is_grad_enabled():
        return _ORIG["pwa"](self, m, z)
    m = self.layernorm_m(m)                                                   # identical op sequence to stock forward minus the z path
    v = self.linear_no_bias_mv(m); v = v.reshape(*v.shape[:-1], self.n_heads, self.c)
    g = torch.sigmoid(self.linear_no_bias_mg(m)); g = g.reshape(*g.shape[:-1], self.n_heads, self.c)
    wv = torch.einsum("...ijh,...mjhc->...mihc", w, v)
    o = (g * wv).reshape(*g.shape[:-2], self.n_heads * self.c)
    return self.linear_no_bias_out(o)

def _inference_forward_cached(self, m, z, chunk_size=2048):
    pwa = self.msa_pair_weighted_averaging
    if m.shape[-3] <= chunk_size or self.training or torch.is_grad_enabled():   # single chunk: stock path verbatim
        return _ORIG["inf"](self, m, z, chunk_size)
    pwa._fpf_zcache_w = pwa.softmax_w(pwa.linear_no_bias_z(pwa.layernorm_z(z)))   # same three stock calls, once
    try:
        return _ORIG["inf"](self, m, z, chunk_size)
    finally:
        pwa._fpf_zcache_w = None

def install():
    if os.environ.get("PTX_PWA_ZCACHE", "1") == "0" or _ORIG:
        return bool(_ORIG)
    _ORIG["pwa"] = PF.MSAPairWeightedAveraging.forward; _ORIG["inf"] = PF.MSAStack.inference_forward
    PF.MSAPairWeightedAveraging.forward = _pwa_forward_cached; PF.MSAStack.inference_forward = _inference_forward_cached
    return True

def uninstall():
    if _ORIG:
        PF.MSAPairWeightedAveraging.forward = _ORIG.pop("pwa"); PF.MSAStack.inference_forward = _ORIG.pop("inf")
