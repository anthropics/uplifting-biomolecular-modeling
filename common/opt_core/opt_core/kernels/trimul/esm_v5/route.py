"""Row ``esm_v5_fwd``: the ESM-family engine's forward triangle-multiplication line (``ef2_trimul_v5``: Triton K1 with tensor-descriptor loads ->
cuBLAS strided-batched bmm -> Triton K3, with the engine's cc 9.0 lever composition incnt + formtab + sigmoid) served from the ten canonical weight
tensors, forward only, residual optional (``residual=False`` = the cofolding form ``update``; ``True`` = ``z + update`` as the engine runs it).

Everything here is plain torch (imported inside the functions; the module itself loads with the standard library only, so the face can
resolve cells without a framework): the weight relayout below IS the engine's pack for this line (``wgT_in`` / ``wpT_in`` = the [gate | value]
projections transposed to [c_z, 2 c_hidden] bf16, ``wz`` / ``wg_out`` = the output projection / output gate [c_z, c_hidden] / [c_z, c_z] bf16,
the four LayerNorm affine vectors rounded to bf16 and held fp32 -- the engine's fused call hands bf16-cast parameters to its pack, so a bf16
round trip of fp32 masters reproduces its bytes), and the launch cell is read from the carried ``ef2_w4_fpf_trimul_v4_cells.json`` by the
engine's own lookup order (``<cc>|<triton major.minor>`` -> ``<cc>|*`` -> ``*|*``).  No environment variable is read.

    wp  = pack(weights)                                   # once per weight set (the face caches it)
    cfg, key, kind = cell(device)                          # the (k1, k3) launch dicts this device / triton resolve to
    out = forward(z, outgoing, mask, wp, residual=False)   # z [N,N,C] or [B,N,N,C] bf16 CUDA; mask [N,N] / [B,N,N] 0/1 or None

``Unavailable(reason)`` is raised BEFORE any launch when the carried kernels cannot run in this process (triton without
``tl.make_tensor_descriptor`` / ``triton.set_allocator``, no cell for the device); the face turns it into a refusal by name.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CELLS_PATH = os.path.join(HERE, "ef2_w4_fpf_trimul_v4_cells.json")
DEFAULT_CELL = "*|*"
# the engine's cc 9.0 composition for this line without the CuTe K3 (that object is a separate row): incoming contracted in NT form on transposed
# planes, per-extent cuBLAS form table, tanh.approx gate sigmoid; stagger off
LEVERS_FWD = dict(incnt=True, formtab=True, sigmoid=True, stagger=False, k3cute=False, lnfold=True)
LEVER_NAMES = tuple(LEVERS_FWD)
N_MIN = 16                        # the engine's own eligibility floor for this line (16 <= N; measured 128 .. 2048)
SUPPORTED_C = (64, 128, 256)      # c_hidden == c_z; 256 is the engine's width, 128 the cofolding trunks', 64 the template pair stacks'
PAD = 16                          # token-axis padding of the channel-major planes (the carried forward's default)


class Unavailable(Exception):
    """The carried kernels cannot serve in this process; ``reason`` is the word for the refusal."""

    def __init__(self, reason):
        Exception.__init__(self, reason)
        self.reason = reason


_K = {}


def kernels():
    """The carried ``ef2_trimul_v5`` module (imported once).  Raises ``Unavailable`` when triton lacks the tensor-descriptor API it is written on."""
    m = _K.get("mod")
    if m is not None:
        return m
    err = _K.get("err")
    if err is not None:
        raise Unavailable(err)
    try:
        import triton
        import triton.language as tl
    except ImportError as e:                                        # a torch without triton (CPU wheels): named, not silent
        _K["err"] = "import:triton(%s)" % str(e).split("\n")[0][:60].replace(" ", "_")
        raise Unavailable(_K["err"])
    if not hasattr(tl, "make_tensor_descriptor") or not hasattr(triton, "set_allocator"):
        _K["err"] = "triton:%s(no_tl.make_tensor_descriptor/triton.set_allocator)" % getattr(triton, "__version__", "?")
        raise Unavailable(_K["err"])
    try:
        from . import ef2_trimul_v5 as m
    except ImportError as e:
        _K["err"] = "import:ef2_trimul_v5(%s)" % str(e).split("\n")[0][:60].replace(" ", "_")
        raise Unavailable(_K["err"])
    _K["mod"] = m
    return m


def triton_mm():
    """'3.7' from triton 3.7.1 (None without triton)."""
    try:
        import triton
        return ".".join(str(triton.__version__).split(".")[:2])
    except ImportError:
        return None


_CELLS = {}


def cells(path=None):
    p = path or CELLS_PATH
    t = _CELLS.get(p)
    if t is None:
        with open(p, encoding="utf-8") as fh:
            t = _CELLS[p] = json.load(fh)
    return t


def cell(device=None, cc=None, triton_version=None, path=None):
    """(cfg {'k1','k3'}, key, kind) for this device by the engine's lookup order: '<cc>|<triton M.m>' (exact) -> '<cc>|*' (generic) -> '*|*' (the
    table's default cell).  ``cc`` '9.0' / (9, 0) and ``triton_version`` may be given instead of probing.  Raises ``Unavailable`` when no row serves."""
    if cc is None:
        import torch
        dev = device if device is not None else (torch.cuda.current_device() if torch.cuda.is_available() else None)
        if dev is None:
            raise Unavailable("no_cuda_device")
        cc = "%d.%d" % tuple(torch.cuda.get_device_capability(dev))
    elif isinstance(cc, (tuple, list)):
        cc = "%d.%d" % (int(cc[0]), int(cc[1]))
    tmm = triton_version if triton_version is not None else triton_mm()
    tmm = ".".join(str(tmm).split(".")[:2]) if tmm else "none"
    tab = cells(path)
    for key, kind in (("%s|%s" % (cc, tmm), "exact"), ("%s|*" % cc, "generic"), (DEFAULT_CELL, "default")):
        c = tab.get(key)
        if c and "k1" in c and "k3" in c:
            return dict(k1=dict(c["k1"]), k3=dict(c["k3"])), key, kind
    raise Unavailable("no_cell:%s|%s" % (cc, tmm))


def pack(weights):
    """The engine's weight pack for this line from the ten canonical tensors (``ln_in_w ln_in_b w_ag w_ap w_bg w_bp ln_out_w ln_out_b w_o w_og``):
    every tensor first cast to bf16 (the engine's fused call casts its parameters so), then LN affine held fp32, [gate | value] input projections
    transposed [c_z, 2 c_hidden], output projection / gate as given [c_z, c_hidden] / [c_z, c_z].  Byte-identical to the engine's own pack of the
    same parameters."""
    import torch
    w = weights
    bf = torch.bfloat16

    def b16(t):
        return t.detach().to(bf)

    def f32(t):
        return t.detach().to(bf).float().contiguous()

    p_in = torch.cat([b16(w["w_ap"]), b16(w["w_bp"])], 0)          # value rows: a then b
    g_in = torch.cat([b16(w["w_ag"]), b16(w["w_bg"])], 0)          # gate rows: a then b
    return dict(ln_in_w=f32(w["ln_in_w"]), ln_in_b=f32(w["ln_in_b"]), ln_out_w=f32(w["ln_out_w"]), ln_out_b=f32(w["ln_out_b"]),
                wgT_in=g_in.t().contiguous(), wpT_in=p_in.t().contiguous(), wz=b16(w["w_o"]).contiguous(), wg_out=b16(w["w_og"]).contiguous(),
                C=int(w["w_og"].shape[0]), CH=int(w["w_o"].shape[1]))


def levers(overrides=None):
    """The lever dict of a call: LEVERS_FWD updated by ``overrides`` (unknown names raise; k3cute is not this row's)."""
    lv = dict(LEVERS_FWD)
    for k, v in (overrides or {}).items():
        if k not in lv:
            raise ValueError("esm_v5_fwd: unknown lever %r (levers: %s)" % (k, ", ".join(LEVER_NAMES)))
        lv[k] = bool(v)
    if lv.get("k3cute"):
        raise Unavailable("lever:k3cute(the CuTe K3 object is not part of this row)")
    return lv


def check(z, wp):
    """Raise ``Unavailable`` when (z, pack) is outside this row's envelope (before any launch)."""
    import torch
    if not z.is_cuda:
        raise Unavailable("device:%s(cuda only)" % z.device.type)
    if z.dtype != torch.bfloat16:
        raise Unavailable("dtype:%s!=bf16" % str(z.dtype).replace("torch.", ""))
    if z.dim() not in (3, 4) or z.shape[-2] != z.shape[-3]:
        raise Unavailable("shape:%s(z must be [N,N,C] or [B,N,N,C])" % (tuple(z.shape),))
    C, N = int(z.shape[-1]), int(z.shape[-2])
    if C not in SUPPORTED_C or int(wp["C"]) != C or int(wp["CH"]) != C:
        raise Unavailable("c_z=%d,c_hidden=%d(c_z == c_hidden in %s)" % (C, int(wp["CH"]), "|".join(str(c) for c in SUPPORTED_C)))
    if N < N_MIN:
        raise Unavailable("n<%d" % N_MIN)


def forward(z, outgoing, mask, wp, *, residual=False, cfg=None, lever_overrides=None, eps=1e-5, out=None, pad=PAD):
    """Serve the op: ``z`` [N,N,C] or [B,N,N,C] bf16 CUDA (pre-LayerNorm pair), ``mask`` [N,N] / [B,N,N] (0/1, float or bool) or None, ``wp`` =
    ``pack(weights)``; returns the update (``residual=False``) or ``z + update`` in bf16, same shape as z.  ``cfg`` {'k1','k3'} overrides the
    table's cell; ``lever_overrides`` updates LEVERS_FWD for this call.  A batch is served one element per launch set (the engine's own policy)."""
    import torch
    K = kernels()
    check(z, wp)
    lv = levers(lever_overrides)
    if cfg is None:
        cfg = cell(z.device)[0]
    zs = z if z.dim() == 4 else z[None]
    zs = zs if zs.is_contiguous() else zs.contiguous()
    B, N = int(zs.shape[0]), int(zs.shape[1])
    ms = None
    if mask is not None:
        ms = mask if mask.dim() == 3 else mask[None]
        if ms.dtype == torch.bool:
            ms = ms.to(torch.float32)
        if tuple(ms.shape[-2:]) != (N, N):
            raise Unavailable("mask:%s(expected [..,%d,%d])" % (tuple(mask.shape), N, N))
    o4 = out if out is not None else torch.empty_like(zs)
    o4 = o4 if o4.dim() == 4 else o4[None]
    for b in range(B):
        mb = None if ms is None else ms[min(b, ms.shape[0] - 1)].contiguous()
        K.trimul_v5_forward(zs[b], bool(outgoing), mb, wp, cfg, eps=eps, residual=bool(residual), stock_round=False, out=o4[b], levers=lv, pad=pad)
    return o4 if z.dim() == 4 else o4[0]


def describe(device=None):
    """Plain data for a LEVER / census line: the cell this process resolves to and the lever composition."""
    try:
        cfg, key, kind = cell(device)
        c = dict(cell=key, kind=kind, k1=cfg["k1"], k3=cfg["k3"])
    except Unavailable as e:
        c = dict(cell=None, reason=e.reason)
    c["levers"] = dict(LEVERS_FWD)
    c["triton"] = triton_mm()
    return c
