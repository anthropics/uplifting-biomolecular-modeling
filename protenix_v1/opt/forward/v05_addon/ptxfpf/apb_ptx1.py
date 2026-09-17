"""apb_ptx1.py — the protenix 1.1.0 adapter of the shared core's fused sampler attention-with-pair-bias kernels (opt_core.kernels.apb rows
fpf_apb / fpf_atom), lever words `ditattn`, `ditattnfp16`, `atomattn` of the kit arm (levers_ptx1.apply: `<trimul>[+lever...]`).

Kernels: `apb_triton.apb_views` (flash attention + one additive pair bias shared by the S diffusion samples) and `atom_triton.atom_apb` (the atom
transformer's 32/128 local windows addressed in-kernel: no padded / unfolded q, k, v copies, no mask tensor), installed as instance-level
`forward`s on protenix.model.modules.primitives.Attention modules of the model's DiffusionModule. The sampler runs with autocast DISABLED and
fp32 inputs in protenix 1.1.0 (configs_base skip_amp.sample_diffusion = True), so both sites are fp32 at every N under `--dtype bf16` too.
  ditattn     the 24 token-level modules diffusion_transformer.blocks[i].attention_pair_bias.attention (16 heads x 48, gating, q bias): global
              attention, pair bias [.., 1, 16, N, N] shared by the samples (AttentionPairBias.standard_multihead_attention on z with a size-1 sample
              dim), or the hoisted bias that lib/kit112_src/dit_hoist.py hands this very module (the hoist composes; its cached bias arrives here
              unchanged). Cell per card and precision state (CELLS): fp32 inputs -> 3xTF32 operands (fp32-faithful); `+ditattnfp16` -> fp16
              operands (the precision lever: the cell's `precast` form copies q/k/v to fp16 first, fp32 accumulate / softmax; rides ditattn,
              installs nothing by itself); 16-bit inputs run as loaded. 64x64 tiles, 4 warps, 2 stages, the bias tile through a host TMA
              descriptor (triton >= 3.3; without it plain loads, named).
  atomattn    the 6 atom-level modules of atom_attention_encoder / atom_attention_decoder (4 heads x 32, windows n_queries 32 / n_keys 128;
              cross_attention_mode: q from a, k/v from the separately normalised kv), local pair bias [.., 1, 4, n_trunks, 32, 128] shared by the
              samples. Replaces primitives._local_attention (pad + unfold + the -1e10 padding mask + chunked fp32 attention); operands rounded to
              nearest TF32 (the cuBLAS TF32 class), one program per (trunk, sample, head). The input embedder's atom encoder (outside the
              DiffusionModule, no sample dim) keeps stock.

Numerics class: tolerance — all three; none is bit-equal to stock (flash recurrence order, exp2 softmax, fused gate).
Composition: install AFTER bind_model and after the sampler graphs / hoist (levers_ptx1.set_sampler): an instance-level `forward` on each served
module, which the hoist's producer and the graphed denoiser step both reach through Module.__call__; CUDA-graph capture safe (no syncs, no
host-dependent shapes; the first call per (dtype, cell) compiles in the loop's eager warm-up step, before capture). Under `atomattn` the hoist's
G2 glue (rearrange_to_dense_trunk's cached padding bias) is not reached on the six modules — a subsumed site, its counter re-bases.
The row-sharded pair stack (`--n_gpu` > 1, tp.py) runs its own DiT block functions: the levers are installed and idle there (calls = 0, named).

Step aside BY NAME, never silent, never a refusal (COUNTS is the census levers_ptx1.describe() carries, kinds `dit` / `atom`):
  install time  `named`: no CUDA device | compute capability below 8.0 (`arch=sm_XY`: no bf16 / TF32 tensor-core MMA for the Triton dots) |
                a module geometry without a cell (`geometry=<H>x<D>[:nogate]`) | no DiffusionModule / an unexpected module tree | the kernel
                package or triton not importable — the lever installs on nothing it cannot serve and says so (facts()['named']).
                A card of cc >= 8.0 without its own row ENGAGES the cc 9.0 cell and is named (`cell=9.0 on sm_XY`).
  call time     `stock:<word>` counted and the module's own forward answers that call: `stock:local_call` (a windowed call on a DiT module),
                `stock:global_call` (a global call on an atom module), `stock:window=<q>x<k>`, `stock:no_bias`, `stock:bias_per_sample`
                (a bias with a real sample / batch dim: not this site's contract), `stock:bias_shape`, `stock:kv_len`, `stock:dtype=<t>`,
                `stock:device`; served calls count `apb:<operand class>` (dit: tf32x3 | fp16 | bf16in | fp16in; atom: tf32rn | bf16in | fp16in).

CPU-importable: torch, triton and protenix are imported inside the functions; the routing decisions (dit_route / atom_route / card_cell) are
pure functions of shapes, dtype names and the capability tuple (tests/test_apb_ptx1.py drives them without a GPU).
"""
import math
from collections import defaultdict

__version__ = "1.0.1"
PACKAGE = "opt_core.kernels.apb.fpf_apb"            # the shared core's carried fpf_apb package (rows fpf_apb / fpf_atom of opt_core.kernels.apb)


def _apb_pkg(row):
    """The shared core's carried package behind an attention-pair-bias row word (opt_core.kernels.apb.CARRIED_SUBPACKAGES: fpf_apb / fpf_atom ->
    kernels/apb/fpf_apb, dit_exact -> kernels/apb/dit_exact); the kit binds the rows by word."""
    import importlib
    from opt_core.kernels import apb as APB
    return importlib.import_module("opt_core.kernels.apb." + APB.CARRIED_SUBPACKAGES[row])


def _apb_mod(row):
    """The module serving a carried row (opt_core.kernels.apb.carried_module: fpf_apb -> .apb_triton, fpf_atom -> .atom_triton, dit_exact -> the package)."""
    from opt_core.kernels import apb as APB
    return APB.carried_module(row)
LEVERS = ("ditattn", "ditattnfp16", "atomattn")    # the arm words this adapter answers (levers_ptx1.LEVER_NAMES carries them)

# Geometry read from the pinned wheel (configs/configs_base.py model.diffusion_module; transformer.AtomTransformer defaults) — the geometry
# the cells cover. A module of another geometry is not installed on (named), never served with a guess.
DIT_HEADS, DIT_HEAD_DIM = 16, 48                   # c_token 768 / 16 heads
ATOM_HEADS, ATOM_HEAD_DIM = 4, 32                  # c_atom 128 / 4 heads
ATOM_WINDOW = (32, 128)                            # n_queries, n_keys
SERVED_HEAD_DIMS = (32, 48, 64)                    # head widths the DiT kernel's tile forms cover (apb_config: 48 = 32 + 16 split / padded 64)
MIN_CC = (8, 0)                                    # Ampere and newer engage (bf16 / tf32 MMA; the 64x64 tiles' shared memory is far under sm_80's 163 KB)

# The launch cell per lever per compute capability "major.minor" (ditattn: per precision state) — the same rows as
# opt_core.kernels.apb.fpf_apb install.CELLS (asserted equal at install time, cells_agree()); a card of cc >= MIN_CC without a row engages the 9.0 row, named.
CELLS = {
    "dit": {"9.0": {"fp32": {"opd": "tf32x3", "bias_tma": True},                     # H100, ditattn alone: 3xTF32 operands (fp32-faithful), bias tile via TMA descriptor
                    "fp16": {"opd": "fp16", "precast": True, "bias_tma": True}}},    # H100, ditattn+ditattnfp16: stock's fp32 q/k/v pre-cast to fp16 (3 copy kernels) -> the 16-bit kernel (same bits as casting in-kernel)
    "atom": {"9.0": {"opd": "tf32rn"}},                              # H100: operands rounded to nearest TF32, one program per (trunk, sample, head)
}
REFERENCE_CC = "9.0"

SERVED_PREFIX, ASIDE_PREFIX, ERROR_PREFIX = "apb:", "stock:", "error:"
COUNTS = {"dit": defaultdict(int), "atom": defaultdict(int)}       # per-call census words (levers_ptx1.COUNTS["dit"|"atom"] are these dicts)
STATE = {
    "dit": {"requested": False, "fp16": False, "installed_on": 0, "found": 0, "cell": None, "cell_key": None, "named": None, "calls": 0, "tma": None},
    "atom": {"requested": False, "installed_on": 0, "found": 0, "cell": None, "cell_key": None, "named": None, "calls": 0},
}
_INSTALLED = {"dit": [], "atom": []}                               # (module, had_instance_forward, previous) for uninstall


# ---------------------------------------------------------------- pure routing (no torch needed)
def card_cell(kind: str, cc, fp16: bool = False):
    """(cell dict | None, cell_key | None, named | None) for lever kind on a card of capability `cc` ((major, minor) tuple, or None = no CUDA).
    None cell = the lever installs on nothing (named says why); a card >= MIN_CC without a row engages the REFERENCE_CC row and is named."""
    if cc is None:
        return None, None, "no CUDA device"
    cc = (int(cc[0]), int(cc[1]))
    if cc < MIN_CC:
        return None, None, "arch=sm_%d%d below sm_%d%d (no bf16/tf32 tensor-core MMA for the kernel's dots)" % (cc + MIN_CC)
    key = "%d.%d" % cc
    table = CELLS[kind]
    named = None
    if key not in table:
        named = "cell=%s on sm_%d%d (unmeasured card: engaged with the cc %s cell)" % ((REFERENCE_CC,) + cc + (REFERENCE_CC,))
        key = REFERENCE_CC
    row = table[key]
    cell = dict(row["fp16" if fp16 else "fp32"]) if kind == "dit" else dict(row)
    return cell, key, named


def geometry_word(kind: str, heads: int, head_dim: int, gating: bool):
    """None when the module geometry is one the cells serve, else the `geometry=` word naming it."""
    want_h, want_d = (DIT_HEADS, DIT_HEAD_DIM) if kind == "dit" else (ATOM_HEADS, ATOM_HEAD_DIM)
    if heads == want_h and head_dim == want_d and gating:
        return None
    if kind == "dit" and gating and head_dim in SERVED_HEAD_DIMS and heads > 0:      # another head count at a served width: the kernel is per head; served, not named
        return None
    return "geometry=%dx%d%s" % (heads, head_dim, "" if gating else ":nogate")


def _squeezable_to(shape, nd: int):
    """The trailing `nd` dims of `shape` if every leading dim is 1, else None."""
    shape = tuple(int(x) for x in shape)
    if len(shape) < nd:
        return None
    lead, tail = shape[:-nd], shape[-nd:]
    return tail if all(x == 1 for x in lead) else None


def dit_route(q_shape, kv_shape, bias_shape, heads: int, *, dtype: str = "float32", is_cuda: bool = True, local: bool = False):
    """The DiT token site's decision: SERVED word None + detail, or the `stock:<word>` this call steps aside with.
    q_shape [.., N, c], kv_shape [.., M, c], bias_shape = attn_bias's shape or None; dtype = str(torch dtype) tail."""
    if local:
        return "stock:local_call"
    if not is_cuda:
        return "stock:device"
    if dtype not in ("float32", "bfloat16", "float16"):
        return "stock:dtype=%s" % dtype
    q_shape, kv_shape = tuple(q_shape), tuple(kv_shape)
    if len(q_shape) < 2 or len(kv_shape) < 2:
        return "stock:q_shape"
    N = q_shape[-2]
    if kv_shape[-2] != N or tuple(kv_shape[:-2]) != tuple(q_shape[:-2]):
        return "stock:kv_len"
    if bias_shape is None:
        return "stock:no_bias"
    b = _squeezable_to(bias_shape, 3)
    if b is None:                                     # a leading dim > 1 before [H, N, N]: a per-sample / per-batch bias — not the shared pair bias this kernel reads once
        return "stock:bias_per_sample"
    if b[0] != heads or b[1] < N or b[2] < N or (b[1] != b[2]):
        return "stock:bias_shape"
    return None


def atom_route(q_shape, kv_shape, bias_shape, local_bias_shape, heads: int, n_queries, n_keys, *, dtype: str = "float32", is_cuda: bool = True):
    """The atom site's decision (windowed call): None = served, else the `stock:<word>`."""
    if not (n_queries and n_keys):
        return "stock:global_call"
    if (int(n_queries), int(n_keys)) != ATOM_WINDOW:
        return "stock:window=%sx%s" % (n_queries, n_keys)
    if not is_cuda:
        return "stock:device"
    if dtype not in ("float32", "bfloat16", "float16"):
        return "stock:dtype=%s" % dtype
    if bias_shape is not None:
        return "stock:dense_bias"
    if local_bias_shape is None:
        return "stock:no_bias"
    q_shape, kv_shape = tuple(q_shape), tuple(kv_shape)
    if len(q_shape) < 2 or tuple(kv_shape) != q_shape:
        return "stock:kv_len"
    N = q_shape[-2]
    T = -(-N // ATOM_WINDOW[0])
    b = _squeezable_to(local_bias_shape, 4)
    if b is None:
        return "stock:bias_per_sample"
    if b != (heads, T, ATOM_WINDOW[0], ATOM_WINDOW[1]):
        return "stock:bias_shape"
    return None


def served_word(dtype: str, opd: str) -> str:
    """The census word of a served call: the operand class the kernel ran (fp32 inputs: the cell's opd; 16-bit inputs run as loaded)."""
    if dtype == "float32":
        return SERVED_PREFIX + opd
    return SERVED_PREFIX + ("bf16in" if dtype == "bfloat16" else "fp16in")


# ---------------------------------------------------------------- install / uninstall (torch + protenix + the kernel package, imported here)
def _cc():
    import torch
    if not torch.cuda.is_available():
        return None
    return tuple(torch.cuda.get_device_capability())


def cells_agree() -> bool:
    """The adapter's CELLS rows equal the carried package's rows (opt_core/kernels/apb/fpf_apb install.CELLS) for the two sampler levers."""
    import importlib
    I = importlib.import_module(_apb_pkg("fpf_apb").__name__ + ".install")
    theirs = {"dit": I.CELLS["dit_attn"], "atom": I.CELLS["atom_attn"]}
    return all(theirs[k] == CELLS[k] for k in ("dit", "atom"))


def _diffusion_module(model):
    from protenix.model.modules import diffusion as Dm
    dms = [m for m in model.modules() if isinstance(m, Dm.DiffusionModule)]
    return dms[0] if len(dms) == 1 else None, len(dms)


def _site_modules(model, kind: str):
    """[(name, Attention module)] of the kind's site inside the model's one DiffusionModule, or ([], named) when the tree is not the pinned one's."""
    from protenix.model.modules import primitives as P, transformer as T
    dm, n_dm = _diffusion_module(model)
    if dm is None:
        return [], "expected one DiffusionModule, found %d" % n_dm
    out = []
    if kind == "dit":
        for name, m in dm.named_modules():
            if isinstance(m, T.AttentionPairBias) and name.startswith("diffusion_transformer.") and isinstance(getattr(m, "attention", None), P.Attention):
                out.append((name + ".attention", m.attention))
    else:
        for name, m in dm.named_modules():
            if isinstance(m, T.AtomTransformer) and (name.startswith("atom_attention_encoder.") or name.startswith("atom_attention_decoder.")):
                if (int(getattr(m, "n_queries", 0)), int(getattr(m, "n_keys", 0))) != ATOM_WINDOW:
                    return [], "atom transformer %s windows %sx%s (cells measured for %dx%d)" % ((name, getattr(m, "n_queries", None), getattr(m, "n_keys", None)) + ATOM_WINDOW)
                for bn, blk in m.named_modules():
                    if isinstance(blk, T.AttentionPairBias) and isinstance(getattr(blk, "attention", None), P.Attention):
                        out.append((name + "." + bn + ".attention", blk.attention))
    if not out:
        return [], "no %s attention modules found under the DiffusionModule" % ("DiffusionTransformer token" if kind == "dit" else "AtomTransformer")
    return out, None


def _make_dit_forward(att, cell, orig):
    import torch
    _A = _apb_mod("fpf_apb"); apb_views, precast16 = _A.apb_views, _A.precast16
    H = int(att.num_heads); opd = cell["opd"]; precast = bool(cell.get("precast", False))
    cfg = None if cell.get("bias_tma", True) else dict(BIAS_TMA=False)
    cnt, st = COUNTS["dit"], STATE["dit"]

    def forward(q_x, kv_x, attn_bias=None, trunked_attn_bias=None, n_queries=None, n_keys=None, inf=1e10, inplace_safe=False, chunk_size=None):
        word = dit_route(tuple(q_x.shape), tuple(kv_x.shape), None if attn_bias is None else tuple(attn_bias.shape), H,
                         dtype=str(q_x.dtype).rsplit(".", 1)[-1], is_cuda=bool(q_x.is_cuda), local=bool(n_queries) or trunked_attn_bias is not None)
        if word is not None:                                                      # step aside by name: counted, the module's own forward answers this call
            cnt[word] += 1
            return orig(q_x, kv_x, attn_bias=attn_bias, trunked_attn_bias=trunked_attn_bias, n_queries=n_queries, n_keys=n_keys, inf=inf,
                        inplace_safe=inplace_safe, chunk_size=chunk_size)
        q = att.linear_q(q_x); k = att.linear_k(kv_x); v = att.linear_v(kv_x)
        g = att.linear_g(q_x) if att.gating else None
        C = q.shape[-1]; D = C // H; lead = q.shape[:-2]; N = q.shape[-2]
        q4, k4, v4 = (t.reshape(-1, N, H, D) for t in (q, k, v))
        g4 = g.reshape(-1, N, H, D) if g is not None else None
        b = attn_bias
        while b.dim() > 3:
            b = b[0]
        if b.shape[-1] != N or b.shape[-2] != N:
            b = b[:, :N, :N]
        if b.stride(-1) != 1:
            b = b.contiguous()
        if precast and q4.dtype == torch.float32:                                  # the precision cell's pre-cast form: 16-bit copies of stock's fp32 q/k/v feed the 16-bit kernel (gate logits stay fp32)
            q16, k16, v16 = precast16(q4, k4, v4, torch.float16 if opd == "fp16" else torch.bfloat16)
            o = apb_views(q16, k16, v16, b, g4, scale=1.0 / math.sqrt(D), out_dtype=torch.float32, cfg=cfg)
        else:
            o = apb_views(q4, k4, v4, b, g4, scale=1.0 / math.sqrt(D), out_dtype=q.dtype, opd=(opd if q.dtype == torch.float32 else None), cfg=cfg)
        st["calls"] += 1
        cnt[served_word(str(q.dtype).rsplit(".", 1)[-1], opd)] += 1
        return att.linear_o(o.view(*lead, N, C))

    forward.__wrapped__ = orig; forward._apb_ptx1 = "dit"
    return forward


def _make_atom_forward(att, cell, orig):
    import torch
    atom_apb = _apb_mod("fpf_atom").atom_apb
    H = int(att.num_heads); opd = cell["opd"]
    cnt, st = COUNTS["atom"], STATE["atom"]

    def forward(q_x, kv_x, attn_bias=None, trunked_attn_bias=None, n_queries=None, n_keys=None, inf=1e10, inplace_safe=False, chunk_size=None):
        word = atom_route(tuple(q_x.shape), tuple(kv_x.shape), None if attn_bias is None else tuple(attn_bias.shape),
                          None if trunked_attn_bias is None else tuple(trunked_attn_bias.shape), H, n_queries, n_keys,
                          dtype=str(q_x.dtype).rsplit(".", 1)[-1], is_cuda=bool(q_x.is_cuda))
        if word is not None:
            cnt[word] += 1
            return orig(q_x, kv_x, attn_bias=attn_bias, trunked_attn_bias=trunked_attn_bias, n_queries=n_queries, n_keys=n_keys, inf=inf,
                        inplace_safe=inplace_safe, chunk_size=chunk_size)
        q = att.linear_q(q_x); k = att.linear_k(kv_x); v = att.linear_v(kv_x)
        g = att.linear_g(q_x) if att.gating else None
        C = q.shape[-1]; D = C // H; lead = q.shape[:-2]; N = q.shape[-2]
        q4, k4, v4 = (t.reshape(-1, N, H, D) for t in (q, k, v))
        g4 = g.reshape(-1, N, H, D) if g is not None else None
        b = trunked_attn_bias
        while b.dim() > 4:
            b = b[0]
        if b.stride(-1) != 1:
            b = b.contiguous()
        o = atom_apb(q4, k4, v4, b, g4, n_queries=int(n_queries), n_keys=int(n_keys), scale=1.0 / math.sqrt(D), out_dtype=q.dtype,
                     opd=(opd if q.dtype == torch.float32 else None))
        st["calls"] += 1
        cnt[served_word(str(q.dtype).rsplit(".", 1)[-1], opd)] += 1
        return att.linear_o(o.view(*lead, N, C))

    forward.__wrapped__ = orig; forward._apb_ptx1 = "atom"
    return forward


def _install_kind(model, kind: str, fp16: bool = False) -> dict:
    st = STATE[kind]
    st["requested"] = True
    if kind == "dit":
        st["fp16"] = bool(fp16)
    if st["installed_on"]:
        return facts(kind)
    try:
        import torch  # noqa: F401
    except Exception as e:                           # pragma: no cover
        st["named"] = "torch not importable: %r" % (e,); return facts(kind)
    cell, key, named = card_cell(kind, _cc(), fp16=fp16)
    st["cell"], st["cell_key"], st["named"] = cell, key, named
    if cell is None:
        return facts(kind)
    try:
        import triton  # noqa: F401
        A = _apb_mod("fpf_apb")   # noqa: F841  (imports triton; the atom kernel imports this module too)
    except Exception as e:
        st["cell"] = None; st["named"] = "kernel package not importable: %r" % (e,); return facts(kind)
    if not cells_agree():
        st["cell"] = None; st["named"] = "CELLS differ from %s.install.CELLS" % PACKAGE; return facts(kind)
    if kind == "dit":
        st["tma"] = bool(getattr(A, "_HAS_TMA", False))
        if cell.get("bias_tma") and not st["tma"]:
            st["named"] = ((st["named"] + "; ") if st["named"] else "") + "triton without host TMA descriptors: bias tile via plain loads"
    mods, why = _site_modules(model, kind)
    if why:
        st["cell"] = None; st["named"] = why; return facts(kind)
    st["found"] = len(mods)
    skipped = defaultdict(int)
    for name, att in mods:
        gw = geometry_word(kind, int(att.num_heads), int(att.c_hidden), bool(att.gating))
        if gw is not None:
            skipped[gw] += 1
            continue
        had = "forward" in vars(att)
        prev = vars(att).get("forward")
        orig = att.forward                            # the bound stock forward (or an earlier instance-level patch: composed over, restored on uninstall)
        att.forward = (_make_dit_forward if kind == "dit" else _make_atom_forward)(att, cell, orig)
        _INSTALLED[kind].append((att, had, prev))
    st["installed_on"] = len(_INSTALLED[kind])
    if skipped:
        words = ", ".join("%s x%d" % kv for kv in sorted(skipped.items()))
        st["named"] = ((st["named"] + "; ") if st["named"] else "") + "not installed on " + words
        for w, n in skipped.items():
            COUNTS[kind]["named:" + w] += n
    return facts(kind)


def install_ditattn(model, fp16: bool = False) -> dict:
    """Install the fused kernel on the DiT token-attention modules of the model's DiffusionModule; fp16=True = ditattn + ditattnfp16
    (the precision lever rides this install: ditattnfp16 alone installs nothing and is named by the caller)."""
    return _install_kind(model, "dit", fp16=fp16)


def install_atomattn(model) -> dict:
    """Install the fused local-window kernel on the atom-attention modules of the DiffusionModule's atom encoder + decoder."""
    return _install_kind(model, "atom")


def apply(model, ditattn: bool = False, ditattnfp16: bool = False, atomattn: bool = False) -> dict:
    """The arm's three words at once (levers_ptx1.apply calls this after set_sampler): installs what is asked, uninstalls what is not,
    names `ditattnfp16` without `ditattn`. -> describe()."""
    if ditattn:
        if STATE["dit"]["installed_on"] and STATE["dit"]["fp16"] != bool(ditattnfp16):
            uninstall(model, "dit")
        install_ditattn(model, fp16=ditattnfp16)
    else:
        uninstall(model, "dit")
        STATE["dit"]["requested"] = False; STATE["dit"]["fp16"] = bool(ditattnfp16)
        if ditattnfp16:
            STATE["dit"]["named"] = "ditattnfp16 without ditattn: nothing to install (the precision lever rides ditattn)"
    if atomattn:
        install_atomattn(model)
    else:
        uninstall(model, "atom")
        STATE["atom"]["requested"] = False
    return describe()


def uninstall(model=None, kind: str = None) -> None:
    """Restore the modules' own forwards (kind = dit | atom | None = both)."""
    for k in (("dit", "atom") if kind is None else (kind,)):
        for att, had, prev in _INSTALLED[k]:
            if had:
                att.forward = prev
            else:
                try:
                    del att.forward
                except AttributeError:
                    pass
        _INSTALLED[k] = []
        STATE[k]["installed_on"] = 0


# ---------------------------------------------------------------- account
def facts(kind: str) -> dict:
    """The LEVER facts of one kind: engaged (installed on >= 1 module), installed_on / found, cell (opd, cell_key), named condition, Python-level
    calls (under the sampler graphs: the eager warm-up + capture calls; replays run the captured kernels), the census words."""
    st = STATE[kind]
    out = {"engaged": bool(st["installed_on"]), "installed_on": st["installed_on"], "found": st["found"], "cell_key": st["cell_key"],
           "opd": (st["cell"] or {}).get("opd"), "named": st["named"], "calls": st["calls"], "counts": dict(COUNTS[kind]), "package": package_version()}
    if kind == "dit":
        out["fp16"] = bool(st["fp16"]); out["bias_tma"] = st["tma"]
    return out


def package_version() -> str:
    try:
        P = _apb_pkg("fpf_apb")
        return str(getattr(P, "__version__", "unknown"))
    except Exception:
        return "absent"


def describe() -> dict:
    return {"version": __version__, "dit": facts("dit"), "atom": facts("atom")}


def lever_facts(word: str) -> dict:
    """Per arm word, what a LEVER line needs: state on | skipped (+ reason token), served / gated / fallback counters in the kit's evidence form."""
    kind = "atom" if word == "atomattn" else "dit"
    f = facts(kind)
    counts = f["counts"]
    served = sum(n for w, n in counts.items() if w.startswith(SERVED_PREFIX))
    gated = {w: n for w, n in counts.items() if w.startswith(ASIDE_PREFIX) or w.startswith("named:")}
    fallback = {w: n for w, n in counts.items() if w.startswith(ERROR_PREFIX)}
    if word == "ditattnfp16":
        on = f["engaged"] and f["fp16"]
        reason = None if on else ("rides_ditattn" if not STATE["dit"]["requested"] else (f["named"] or "ditattn_not_engaged"))
        return {"state": "on" if on else "skipped", "reason": reason, "served": served if on else 0, "gated": gated, "fallback": fallback,
                "precision": "fp16" if on else None}
    on = f["engaged"]
    return {"state": "on" if on else "skipped", "reason": None if on else (f["named"] or "not_requested"), "served": served, "gated": gated,
            "fallback": fallback, "installed_on": f["installed_on"], "cell": "%s@%s" % (f["opd"], f["cell_key"]) if on else None, "named": f["named"]}
