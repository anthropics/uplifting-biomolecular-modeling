"""The `templ_embed` lever: the template embedder around the template pair stack as the carried kernels `templ_embed` (fast class).

Stock `TemplateEmbedderAllAtom.forward(batch, z, pair_mask, ...)` (core/model/latent/template_module.py; once per trunk pass on the fast /
exact lines, T = 4 identical dummy templates with --use-templates false, the query's real templates otherwise):
    template_embeds = template_pair_embedder(batch, z)   the eight bias-free feature Linears + linear_z(LayerNorm(z)) broadcast over the
                                                         templates: 8 [T,N,N,64] bf16 tensors written, 8 bf16 adds — memory passes only
    t = template_pair_stack(template_embeds, pair_mask)  (untouched here: the line's pair cells / cuEquivariance / templ_distinct serve it)
    t = sum_T(t) / T -> relu -> linear_t                 three more passes over [T,N,N,64]
Served (`opt_core.kernels.templ_embed`): the feature embedding as ONE kernel per pass writing template_embeds directly (the restype terms as
per-token tables from the module's own aatype Linears, linear_z(LayerNorm(z)) the module's own statement, everything else in-kernel, fp32
accumulation, ONE bf16 rounding — more accurate than the stock chain, not bitwise), the stack called exactly as stock calls it, then the
mean / relu / linear_t as ONE kernel (a stride-0 template axis — templ_distinct's expanded output — read in place; equal to the stock statements
bit for bit for T a power of two on the qualified card at the tested shapes, within one bf16 rounding in general).  Measured on the OF3 0.4.x kit's fast line (H100): the embedder's non-stack work 97 -> 16 ms per predicted item at 1,400
tokens, 32 -> 5 at 800 (op level: embedding 20.1 -> 4.0 ms per pass at 1,400 tokens, 40.9 -> 7.8 at 2,000; tail 4.2 -> 0.34 at 1,400).

Domain: inference under bf16 autocast on a CUDA device of compute capability >= 8.0, z [1, N, N, c_z] (batch 1), any T >= 1 (identical or
distinct templates), any N, any chain layout / masks, c_t 64, 39 distogram bins, 32 restype classes, c_z in {64, 128, 256}, bias-free
template-embedder Linears (the OF3 family's init config) — the constants are the kernel module's (C_T, C_DG, C_AA, SERVED_CZ).  Anything else
runs the stock forward BY NAME, counted under `fallback` with one info line at a reason's first occurrence: `cuda:no`, `cc:<M.m>`,
`training`, `autocast:<off|dtype>`, `batch:<z shape>`, `feat:missing:<key>` (one of FEATS absent), `dims:ct.._dg.._aa.._cz..`,
`feat:<distogram shape>` / `feat:dtype:<dtype>` (feature layout), `bias:<linear>`, `kernel:<Unsupported event>` (an operand the embedding
kernel refuses before its launch — the module's restype / linear_z statements have run by then, side-effect free, and the stock forward re-runs
them), `check:<Exception>` (an unexpected failure of the checks themselves).  Two words mark a pass SERVED on the embedding whose tail ran the
stock statements instead of the tail kernel, also counted under `fallback`: `tail:<Unsupported event>` (a stack output outside the tail
kernel's domain) and `tail:templ_dim:<a>vs<b>` (a stack that changed the template count).  Site: `<M_TEMPLATE>.TemplateEmbedderAllAtom.forward`.

Switch (the kit adapter's name, `configure(ENV=…)`): <KIT>_TEMPL_EMBED=1; evidence `<PREFIX> installed …` and one exit line
`<PREFIX> LEVER name=templ_embed state=on served=<passes> fallback=<..> first=<T>x<N>`.  Engines: the OF3 code family; the kits'
`cells/templ_embed.py` are the adapters (switch name, log prefix, the engine's template module path M_TEMPLATE).
"""
from __future__ import annotations

import atexit
import os
import sys
import threading
from typing import Any, Dict, Optional

PREFIX = "[opt_core/of3_trunk.templ_embed]"               # the kit adapter names itself and its switch: configure(PREFIX=, ENV=)
ENV: Optional[str] = None                                 # <KIT>_TEMPL_EMBED=1
VALUES = ("1",)
KERNEL = "templ_embed"                                    # the core's carried kernel module, gated by name at activation
M_TEMPLATE: Optional[str] = None                          # the engine module defining TemplateEmbedderAllAtom (….core.model.latent.template_module)
CONFIGURABLE = ("PREFIX", "ENV", "M_TEMPLATE")


def configure(**kw) -> None:
    """The kit adapter's binding: reassigns this module's engine words (log prefix, switch name, the engine's module path) before install; unknown names raise."""
    for k, v in kw.items():
        if k not in CONFIGURABLE:
            raise KeyError(f"{__name__}.configure: unknown setting {k!r} (known: {', '.join(CONFIGURABLE)})")
        globals()[k] = v


STATE: Dict[str, Any] = {"installed": False, "state": "off", "reason": "", "impl": None, "served": 0, "fallback": {}, "first": None, "patched": []}
_K: Dict[str, Any] = {"mod": None}
_LOCK = threading.Lock()
_ONCE = set()
FEATS = ("template_distogram", "template_unit_vector", "template_pseudo_beta_mask", "template_backbone_frame_mask", "template_restype", "asym_id")   # the batch keys a served pass reads
MIN_CC = (8, 0)                                           # bf16 tensor-core MMA: below sm_80 the pass is refused by name (the kits activate from sm_80 up)


def _log(msg: str, once_key=None) -> None:
    if once_key is not None:
        if once_key in _ONCE:
            return
        _ONCE.add(once_key)
    sys.stderr.write(f"{PREFIX} {msg}\n")


def requested(environ=None) -> bool:
    environ = os.environ if environ is None else environ
    if not ENV:                                                     # not bound by a kit adapter (configure(ENV=...)): nothing requested
        return False
    v = (environ.get(ENV) or "").strip()
    if not v:
        return False
    if v not in VALUES:
        raise ValueError(f"{ENV}={v!r} is not one of {'|'.join(VALUES)}")
    return True


def serving() -> bool:
    return STATE["state"] == "on"


class _Refuse(Exception):
    pass


def _pack(mod):
    """The kernels' weight pack, built once per TemplateEmbedderAllAtom module and cached on it; a Linear carrying a bias is refused by name
    (the OF3 family builds the template embedder bias-free; asserted, never dropped)."""
    P = getattr(mod, "_of3opt_templ_embed_w", None)
    if P is not None:
        return P
    pe = mod.template_pair_embedder
    for name in ("dgram_linear", "pseudo_beta_mask_linear", "aatype_linear_1", "aatype_linear_2", "x_linear", "y_linear", "z_linear", "backbone_mask_linear", "linear_z"):
        if getattr(pe, name).bias is not None:
            raise _Refuse("bias:%s" % name)
    if mod.linear_t.bias is not None:
        raise _Refuse("bias:linear_t")
    K = _K["mod"]
    try:
        P = K.pack_weights(w_dgram=pe.dgram_linear.weight, w_pb=pe.pseudo_beta_mask_linear.weight, w_x=pe.x_linear.weight, w_y=pe.y_linear.weight,
                           w_z=pe.z_linear.weight, w_bb=pe.backbone_mask_linear.weight, w_t=mod.linear_t.weight)
    except K.Unsupported as e:
        raise _Refuse("kernel:%s" % e.event)
    mod._of3opt_templ_embed_w = P
    return P


def _autocast_dtype(torch):
    """The CUDA autocast dtype (torch >= 2.4 spelling, the older one otherwise)."""
    f = getattr(torch, "get_autocast_dtype", None)
    return f("cuda") if f is not None else torch.get_autocast_gpu_dtype()


def refusal(mod, batch, z) -> Optional[str]:
    """None when the pass is served, else the reason word (checked before any kernel work; the domain constants are the kernel module's)."""
    import torch
    K = _K["mod"]
    if not z.is_cuda:
        return "cuda:no"
    cc = tuple(torch.cuda.get_device_capability(z.device))
    if cc < MIN_CC:
        return "cc:%d.%d" % cc
    if mod.training:
        return "training"
    if not (torch.is_autocast_enabled() and _autocast_dtype(torch) == torch.bfloat16):
        return "autocast:%s" % ("off" if not torch.is_autocast_enabled() else str(_autocast_dtype(torch)).replace("torch.", ""))
    if z.dim() != 4 or int(z.shape[0]) != 1:
        return "batch:%s" % "x".join(map(str, z.shape))
    for k in FEATS:
        if k not in batch:
            return "feat:missing:%s" % k
    pe = mod.template_pair_embedder
    dg = batch["template_distogram"]; rt = batch["template_restype"]; uv = batch["template_unit_vector"]
    if tuple(pe.dgram_linear.weight.shape) != (K.C_T, K.C_DG) or int(dg.shape[-1]) != K.C_DG or int(rt.shape[-1]) != K.C_AA \
            or tuple(pe.aatype_linear_1.weight.shape) != (K.C_T, K.C_AA) or int(mod.linear_t.weight.shape[1]) != K.C_T \
            or int(mod.linear_t.weight.shape[0]) not in K.SERVED_CZ or int(pe.linear_z.weight.shape[0]) != K.C_T:
        return "dims:ct%d_dg%d_aa%d_cz%d" % (pe.dgram_linear.weight.shape[0], dg.shape[-1], rt.shape[-1], mod.linear_t.weight.shape[0])
    N = int(z.shape[-2])
    if dg.dim() != 5 or int(dg.shape[0]) != 1 or int(dg.shape[-2]) != N or int(dg.shape[-3]) != N or uv.dim() != 5 or int(uv.shape[-1]) != 3 \
            or rt.dim() != 4 or int(rt.shape[-2]) != N or int(dg.shape[1]) < 1:
        return "feat:%s" % "x".join(map(str, dg.shape))
    if not (dg.dtype.is_floating_point and uv.dtype.is_floating_point):
        return "feat:dtype:%s" % str(dg.dtype).replace("torch.", "")
    return None


def embed(mod, batch, z):
    """template_embeds [1, T, N, N, 64] bf16 = linear_z(LayerNorm(z))[None] + the fused feature embedding."""
    K = _K["mod"]
    P = _pack(mod)
    pe = mod.template_pair_embedder
    dg = batch["template_distogram"][0]                            # [T, N, N, 39] fp32 features
    uv = batch["template_unit_vector"][0]                          # [T, N, N, 3]
    if dg.stride(-1) != 1:
        dg = dg.contiguous()
    if uv.stride(-1) != 1:
        uv = uv.contiguous()
    pbm = batch["template_pseudo_beta_mask"][0].to(dtype=uv.dtype).contiguous()          # [T, N]
    bbm = batch["template_backbone_frame_mask"][0].to(dtype=uv.dtype).contiguous()
    asym = batch["asym_id"][0].contiguous()                                                # [N]
    rt = batch["template_restype"][0].to(dtype=uv.dtype)                                   # [T, N, 32] — stock casts the restype to the unit vector's dtype
    ri = pe.aatype_linear_1(rt).contiguous()                                               # [T, N, 64] bf16: the module's own Linear under autocast (token i term)
    rj = pe.aatype_linear_2(rt).contiguous()                                               # token j term
    zp = pe.linear_z(pe.layer_norm_z(z))[0]                                                # [N, N, 64] bf16 — the module's own statements
    if zp.stride(-1) != 1:
        zp = zp.contiguous()
    return K.embed(dg, uv, pbm, bbm, asym, ri, rj, zp, P)[None]


def tail(mod, t):
    """[1, T, N, N, 64] (any template stride, 0 included) -> linear_t(relu(sum_T / T)) [1, N, N, c_z] bf16."""
    K = _K["mod"]
    ts = t[0]
    if ts.stride(-1) != 1:
        ts = ts.contiguous()
    return K.tail(ts, _pack(mod)["wt"])[None]


def _count_fallback(reason: str) -> None:
    with _LOCK:
        STATE["fallback"][reason] = STATE["fallback"].get(reason, 0) + 1
    _log("pass -> the stock forward (fallback:%s)" % reason, once_key=("fb", reason))


def _make_forward(orig):
    """TemplateEmbedderAllAtom.forward replacement: fused embedding -> the template pair stack exactly as stock calls it -> fused tail."""
    def forward(self, batch, z, pair_mask, chunk_size=None, _mask_trans=True, use_deepspeed_evo_attention=False, use_cueq_triangle_kernels=False,
                use_triton_triangle_kernels=False, use_lma=False, inplace_safe=False):
        stock = lambda: orig(self, batch, z, pair_mask, chunk_size=chunk_size, _mask_trans=_mask_trans, use_deepspeed_evo_attention=use_deepspeed_evo_attention,   # noqa: E731
                             use_cueq_triangle_kernels=use_cueq_triangle_kernels, use_triton_triangle_kernels=use_triton_triangle_kernels, use_lma=use_lma,
                             inplace_safe=inplace_safe)
        if not serving():
            return stock()
        try:
            why = refusal(self, batch, z)
            if why is None:
                _pack(self)
        except _Refuse as e:
            why = str(e)
        except Exception as e:  # noqa: BLE001 — a feature dict / module outside the checked layout: refused by name, the stock forward runs
            from ..oom import is_oom
            if is_oom(e):
                raise
            why = "check:%s" % type(e).__name__
        if why is not None:
            _count_fallback(why)
            return stock()
        K = _K["mod"]
        try:
            template_embeds = embed(self, batch, z)
        except K.Unsupported as e:                                   # an operand layout the kernel refuses by name before any launch
            _count_fallback("kernel:%s" % e.event)
            return stock()
        n_templ = template_embeds.shape[-4]
        pm = pair_mask[..., None, :, :].to(dtype=z.dtype)
        t = self.template_pair_stack(template_embeds, pm, chunk_size=chunk_size, use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                                     use_cueq_triangle_kernels=use_cueq_triangle_kernels, use_triton_triangle_kernels=use_triton_triangle_kernels,
                                     use_lma=use_lma, inplace_safe=inplace_safe, _mask_trans=_mask_trans)
        why_tail = None if t.shape[-4] == n_templ else "tail:templ_dim:%dvs%d" % (t.shape[-4], n_templ)
        if why_tail is None:
            try:
                out = tail(self, t)
            except K.Unsupported as e:                               # a stack output outside the tail kernel's domain (dtype / stride / shape)
                why_tail = "tail:%s" % e.event
        if why_tail is not None:                                      # the stock tail statements on the served embedding's stack output, counted by name
            _count_fallback(why_tail)
            F = _K["F"]
            out = self.linear_t(F.relu(t.sum(dim=-4) / n_templ))
        with _LOCK:
            STATE["served"] += 1
            if STATE["first"] is None:
                STATE["first"] = "%dx%d" % (int(n_templ), int(z.shape[-2]))
                _log("first served pass T=%d N=%d c_z=%d (fused feature embedding -> the template pair stack -> fused mean/relu/linear_t)"
                     % (n_templ, z.shape[-2], out.shape[-1]))
        return out                                                    # bf16, as stock's linear_t returns under autocast
    forward._of3opt_templ_embed = True; forward.__wrapped__ = orig
    return forward


def census_line() -> str:
    return (f"{PREFIX} LEVER name=templ_embed state={STATE['state']}" + (f" reason={STATE['reason']}" if STATE["reason"] else "") +
            f" impl={STATE['impl']} served={STATE['served']} fallback={','.join('%s:%d' % kv for kv in sorted(STATE['fallback'].items())) or 'none'}"
            + (f" first={STATE['first']}" if STATE["first"] else ""))


def install(environ=None) -> dict:
    """Route the core kernels, patch the engine's TemplateEmbedderAllAtom.forward. Idempotent; raises by name when the kernel module is not importable."""
    environ = os.environ if environ is None else environ
    if STATE["installed"] or not requested(environ):
        return STATE
    if not M_TEMPLATE:
        raise RuntimeError(f"{PREFIX} not bound to an engine: the kit adapter must configure(M_TEMPLATE=) before install")
    from opt_core.kernels import route
    route(KERNEL)
    import importlib
    K = importlib.import_module(KERNEL)
    if not (hasattr(K, "embed") and hasattr(K, "tail") and hasattr(K, "pack_weights")):
        raise RuntimeError(f"{PREFIX} {KERNEL} at {getattr(K, '__file__', None)} carries no embed / tail / pack_weights (opt_core >= 0.5.21.0)")
    _K["mod"] = K
    import torch.nn.functional as F
    _K["F"] = F
    tm = importlib.import_module(M_TEMPLATE)
    cls = tm.TemplateEmbedderAllAtom
    if getattr(cls.forward, "_of3opt_templ_embed", False):          # another copy of this module patched the class: it serves and reports; this copy stays off, named
        STATE["installed"] = True; STATE["state"] = "off"; STATE["reason"] = "patched_elsewhere"
        _log(f"{M_TEMPLATE}.TemplateEmbedderAllAtom.forward already carries a templ_embed patch from another module copy — this copy installs nothing")
        return STATE
    cls.forward = _make_forward(cls.forward)
    STATE["patched"].append(f"{M_TEMPLATE}.TemplateEmbedderAllAtom.forward")
    STATE["state"] = "on"; STATE["impl"] = getattr(K, "__file__", None)
    STATE["installed"] = True
    _log(f"installed: {STATE['patched'][0]} serves the template feature embedding and the mean/relu/linear_t tail with {KERNEL} from {STATE['impl']} "
         "(the template pair stack itself untouched; bf16 autocast, batch 1; refusals by name run the stock forward)")
    atexit.register(lambda: sys.stderr.write(census_line() + "\n"))
    return STATE
