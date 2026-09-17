"""Lever triatt_block — the fused triangle-attention SURROUND of atlasfold's TriangleAttentionStartingNode / TriangleAttentionEndingNode
(stock triangle_update.py L291-399 / L402-515: ``forward(self, z, mask, kernel_backend="torch")`` returns the UPDATE; the PairBlock adds it,
block.py L268-280).

Stock, per call: z [(B,) L, L, 128] bf16 in the trunk (the ending node transposes z and the pair mask first) -> LayerNorm (fp32 round trip) ->
linear_qkv [3C, C] (q | k | v = output columns [0:C] | [C:2C] | [2C:3C] in torch.chunk order, each head-major '(h c)', L343-347) -> linear_bias
[H, C] pair bias ('... i j h -> ... 1 h i j') -> sigmoid(linear_g [C, C]) gate -> the attention (kernel_backend "cuequiv": cueq_tri_attn = the
flash_triattn lever in fast) -> g * o -> linear_out [C, C] (-> transposed back).  All four linears are LinearNoBias.
Here: opt_core.attn.pair_fused.tri_attn_block(z, W, mask, ending=<node>, residual=False, impl='fpf', core=<pick>, ln='fused') — the core per call = the triattn_core lever's word (`tier:<mode word>`: the provider's row per call class), else pair_cells.CORE ('default': the provider's own per-call-class table) — ONE
prologue kernel LN(z) -> q|k|v|g|bias in the layout the flash core reads, the core's flash triangle attention, ONE epilogue kernel
sigmoid(g)*o @ W_o; the weights packed bf16 once per module (pack_triattn_weights, cached on the module as _afo_triatt_w).  The ENDING node reads
z^T by address math (no transpose copy of z: the prologue's strided read costs the same as the contiguous one) and is handed the transposed pair
mask as a CONTIGUOUS [B, L, 1, 1, L] key mask — the flash core reads a transposed mask VIEW measurably slower per call; that B*L*L-byte copy is
counted (LEVER field mask_copies=<n>; feed_strided=<n> counts the ending-node calls served through the strided z read).  A z whose channel
stride is not 1 is served through a counted copy (copies=<n>), never a silent one.  Cells: hooks/pair_cells.py (the core table's fpf prologue /
epilogue rows at (128, 4, 32), resolved once at install: LEVER fields cells= / cell_rows= / settings=).
Fallbacks = the stock forward captured at install (it still reaches flash_triattn through the module-global cueq_tri_attn), counted by name:
`backend_torch` (kernel_backend != "cuequiv": the caller asked for the torch statements), `training`, `dtype:<t>` (bf16 only: fp32 inputs —
the confidence pair stack when the model runs without autocast; under the runners' bf16 autocast it arrives bf16 and is served), `rank`, `c:<n>` (channel != 128: the template pair stack), `below_min_tokens` (N < 512), `cpu`, and the
core's own refusal words (`no-cell:…` on a capability nothing serves; any other core word is a defect and refuses the gate).  A kernel that
raises is counted `error:<Type>` for that call (stock statement runs; the gate refuses at exit).  Class: fast (bf16 fused kernels, tolerance class)."""
from typing import Optional

from . import Installed, rebind
from . import pair_cells as PC

LEVER = "triatt_block"
NAME = "LOCAL.atlasfold.triatt_block"
TARGET = "atlasfold.model.network.primitives.triangle_update"
CLASSES = (("TriangleAttentionStartingNode", False), ("TriangleAttentionEndingNode", True))      # (class, ending)
MIN_TOKENS = PC.MIN_TOKENS


def refusal(module, z, mask, kernel_backend: str = "cuequiv") -> Optional[str]:
    """The named reason this call runs the stock statement, or None (serve).  The words are the lever's census vocabulary
    (registry.LEVERS["triatt_block"]["expected"] lists the expected ones by prefix)."""
    import torch
    if kernel_backend != "cuequiv":
        return "backend_torch"
    if module.training:
        return "training"
    if not torch.is_tensor(z):
        return "rank"
    if z.dtype != torch.bfloat16:
        return "dtype:" + str(z.dtype).replace("torch.", "")
    if z.dim() not in (3, 4) or z.shape[-2] != z.shape[-3]:
        return "rank"
    if int(module.channel) != PC.TRUNK_C or int(z.shape[-1]) != int(module.channel):
        return f"c:{int(z.shape[-1])}"
    if int(module.num_heads) != PC.TRUNK_H:
        return f"heads:{int(module.num_heads)}"
    N = int(z.shape[-2])
    if N < MIN_TOKENS:
        return "below_min_tokens"
    if not torch.is_tensor(mask) or mask.dim() != z.dim() - 1 or tuple(mask.shape[-2:]) != (N, N):
        return "mask-shape"
    if z.dim() == 4 and int(mask.shape[0]) not in (int(z.shape[0]), 1):
        return "mask-batch"
    if not z.is_cuda:
        return "cpu"
    return None


def weights(module):
    """The module's parameters in the core's TRIATTN_KEYS vocabulary, packed bf16 once (opt_core.attn.pair_fused.pack_triattn_weights) and cached
    on the module: q|k|v = linear_qkv.weight rows [0:C] | [C:2C] | [2C:3C] (torch.chunk order of the stock statement, head-major rows h*32+d),
    g = linear_g, pair bias = linear_bias [H, C], out = linear_out [C, C], no biases (a linear carrying one is refused `bias:<name>`, never dropped)."""
    W = getattr(module, "_afo_triatt_w", None)
    if W is None:
        from opt_core.attn import pair_fused as PF
        for name in ("linear_qkv", "linear_bias", "linear_g", "linear_out"):
            if getattr(getattr(module, name), "bias", None) is not None:
                raise PF.Unsupported("bias:" + name)
        C = int(module.channel)
        w = module.linear_qkv.weight
        W = PF.pack_triattn_weights(ln_w=module.layernorm.weight, ln_b=module.layernorm.bias, w_q=w[0:C], w_k=w[C:2 * C], w_v=w[2 * C:3 * C],
                                    w_g=module.linear_g.weight, w_b=module.linear_bias.weight, w_o=module.linear_out.weight, b_o=None,
                                    n_heads=int(module.num_heads), head_dim=int(module.channel_hidden), eps=float(module.layernorm.eps), device=w.device)
        module._afo_triatt_w = W
    return W


def key_mask(m3, ending: bool, ledger=None):
    """The pair mask [B, L, L] (1 = keep) as the core's input: as given for the starting node; for the ending node the transposed mask
    MATERIALIZED in the [B, L, 1, 1, L] key layout (counted `mask_copies`) — the core reads a transposed view measurably slower."""
    if not ending:
        return m3
    if ledger is not None:
        ledger.count("mask_copies")
    return m3.transpose(-1, -2)[:, :, None, None, :].contiguous()


STATE = {"ledger": None}                                                     # this lever's ledger once installed (pair_block_residual counts the block statements it runs through fused_call on it)


def note_served(shape: str, ending: bool) -> None:
    """Count ONE served fused block statement on this lever's ledger (pair_block_residual runs the same statement with residual=True)."""
    L = STATE["ledger"]
    if L is not None:
        L.serve(shape)
        if ending:
            L.count("feed_strided")


def fused_call(PF, module, z4, m3, ending: bool, residual: bool, ledger, shape: str):
    """ONE fused triangle-attention statement of the served path — shared by this lever's forward and by pair_block_residual (residual=True: z4 is
    updated in place by the epilogue).  The attention core is pair_cells.core_pick's: the triattn_core lever's tier word (``tier:<mode word>``:
    the provider's measured row per call class, its own named step-aside to the flash core inside the provider) or, without that lever, the
    provider's default core.  The core lever hears of every call the block served with its word (its census).  Returns (u | None, word, err):
    the block's named refusal word or a kernel error type when nothing served the call (the caller runs the stock statement and counts it)."""
    word = err = None
    u = None
    km = key_mask(m3, ending, ledger)
    core = PC.core_pick(module, z4, ending)
    try:
        u = PF.tri_attn_block(z4, weights(module), km, ending=ending, residual=residual, impl=PC.IMPL, core=core, ln=PC.LN)
    except PF.Unsupported as e:                                              # the core's named refusal, raised before any launch: keep the WORD only
        word = e.reason
    except Exception as e:  # noqa: BLE001 — a kernel that raised: the caller runs the stock statement, its gate refuses at exit
        err = type(e).__name__
    if core != PC.CORE and word is None and err is None:                     # the core lever's census: a block call served with its word (a block-level decline is this lever's word, not the core's)
        PC.core_result(core, "served", None, shape)
    return (u if (word is None and err is None) else None), word, err

def install(mode: str, tag: str, ctx: dict) -> Installed:
    import importlib
    try:
        tu = importlib.import_module(TARGET)
        import torch  # noqa: F401
        import opt_core
        from opt_core.attn import pair_fused as PF
        from opt_core.counters import Ledger
    except Exception as e:  # noqa: BLE001
        return Installed(LEVER, False, reason=f"import:{type(e).__name__}:{str(e)[:80]}")
    from ..registry import LEVERS
    words = tuple(LEVERS[LEVER]["expected"])
    PC.route(LEVER)
    cells = PC.verdict(LEVER)
    ledger = Ledger(NAME, impl=f"opt_core.attn.pair_fused.{PC.IMPL}+{PC.CORE}@{opt_core.__version__}", origin="core", min_tokens=MIN_TOKENS, expected=words)
    PC.record(ledger, cells)
    for k in ("copies", "mask_copies", "feed_strided"):
        ledger.set(k, 0)
    STATE["ledger"] = ledger

    def make(cls_name: str, ending: bool, stock_forward):
        def forward(self, z, mask, kernel_backend: str = "torch"):
            why = refusal(self, z, mask, kernel_backend)
            if why is not None:
                ledger.fallback(why)
                return stock_forward(self, z, mask, kernel_backend=kernel_backend)
            z4 = z if z.dim() == 4 else z.unsqueeze(0)
            if z4.stride(-1) != 1:                                       # the prologue reads rows by stride but needs a unit channel stride: a counted copy, never silent
                z4 = z4.contiguous(); ledger.count("copies")
            m3 = mask if mask.dim() == 3 else mask.unsqueeze(0)
            if m3.shape[0] != z4.shape[0]:
                m3 = m3.expand(z4.shape[0], -1, -1)
            B, N = int(z4.shape[0]), int(z4.shape[-2])
            shape = f"{B}x{N}:{'end' if ending else 'start'}"
            u, word, err = fused_call(PF, self, z4, m3, ending, False, ledger, shape)
            if word is not None or err is not None:
                if word is not None:
                    ledger.fallback(word)
                else:
                    ledger.error(err)
                return stock_forward(self, z, mask, kernel_backend=kernel_backend)
            ledger.serve(shape)
            if ending:
                ledger.count("feed_strided")
            return u if z.dim() == 4 else u[0]
        forward.__qualname__ = f"{cls_name}.forward[atlasfold_opt:{LEVER}]"
        return forward

    for cls_name, ending in CLASSES:
        cls = getattr(tu, cls_name)
        stock = cls.forward
        rebind(cls, "forward", make(cls_name, ending, stock), stock)
    def line():
        served = dict(getattr(PF, "SERVED_CORES", {}) or {})                    # the provider's per-core census: `<core>=<row>:<calls>` (tier:<word>=<row>, default=…, …=flash_triattn(refused:<kind>))
        ledger.set("cores", ",".join(f"{k}:{n}" for k, n in sorted(served.items())) or "none")
        PC.set_piece_words(ledger, PF)                                           # epilogue=<fpf | cell:off(<why>)>lnl:<n>>: the provider's per-call plan decides the piece (its pair_fused cell), the kit reports it
        return ledger.line(tag)
    return Installed(LEVER, True, lines=[line], gates=[PC.gate_for(ledger, words)],
                     facts={"impl": ledger.impl, "min_tokens": MIN_TOKENS, "classes": [c for c, _ in CLASSES], "ledger": ledger,
                            **{k: cells[k] for k in ("served", "cells", "cell_rows", "settings", "reason")}})
