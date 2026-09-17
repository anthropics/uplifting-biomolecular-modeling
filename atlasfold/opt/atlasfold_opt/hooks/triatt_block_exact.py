"""Lever triatt_block_exact — the EXACT-class construction of the fused triangle-attention SURROUND of atlasfold's TriangleAttentionStartingNode /
TriangleAttentionEndingNode (stock triangle_update.py L314-399 / L425-515: LN -> linear_qkv | linear_bias | sigmoid(linear_g) -> the attention
(kernel_backend "cuequiv": the module-global ``cueq_tri_attn``) -> g * o -> linear_out (-> transposed back); ``forward`` returns the UPDATE the
PairBlock adds, block.py L268-280).

Construction: the module's OWN LayerNorm runs as installed on the
attention frame (z, or z^T for the ending node — the stock statement, or ln_bf16's one-kernel statement when that exact lever rides the row),
then ``opt_core.attn.pair_fused.tri_attn_block(z, W, mask, ending=<node>, residual=False, impl='fpf', ln='stock', x_ln=<that output>,
core=<the STOCK attention call>)``: ONE prologue kernel projects x_ln -> q | k | v | g | pair bias with stock's rounding (bf16 GEMM outputs; the
bias handed on as fp32 of the bf16 projection, exactly what stock's ``cueq_tri_attn(... bias.float())`` reads), the attention core is the
module-global ``triangle_update.cueq_tri_attn`` looked up PER CALL — the stock cuEquivariance op, or the exact provider row the triattn_exact
lever binds there (bitwise the stock op wherever it serves) — and ONE epilogue kernel sigmoid(g) * o @ W_o^T with stock's
rounding points.  The outputs equal the stock statement BIT FOR BIT on the served calls (`--mode exact --det 1` is byte-identical to
`--mode off --det 1`).  Weights packed bf16 once per module (hooks/triatt_block.weights: the same pack the fast lever uses).
Served: bf16 z at c_z = 128, 4 heads x 32, kernel_backend "cuequiv", N >= AFO_TRIATT_BLOCK_EXACT_MIN_TOKENS (default 512: the fused surround's
speed floor, hooks/pair_cells.MIN_TOKENS) on PROVEN_CC cards (9.0; 8.0 inside the row rule CARD_ROWS = (min_rows, max_rows,
piece_rows) measured for these kernels at this geometry, else the stock statement by name: `below_min_rows` / `above_max_rows` / `trailing_piece`, hooks/card_rows.py).
The ending node reads z^T by address math and gets the transposed pair mask as a contiguous [B, L, 1, 1, L] key mask (counted mask_copies=,
as the fast lever does; the mask VALUES are stock's).
ORDER: after triattn_exact (whose binding of ``cueq_tri_attn`` this lever's core reaches per call, in either order) and BEFORE the fast lever
triatt_block in the fast / big rows (exact ⊂ fast: there the flash-cored block is outermost and this lever serves only what it declines).
Fallbacks, counted by name: `backend_torch`, `training`, `dtype:<t>` (bf16 only), `rank`, `c:<n>` (the template pair stack's c=64),
`heads:<n>`, `below_min_tokens`, `above_proven_rows` (B*N*N > ``AFO_TRIATT_BLOCK_EXACT_MAX_ROWS``, default 1280*1280: the
largest call the exact class covers on cc 9.0), `mask-shape`, `mask-batch`, `cpu`, `arch:sm<NN>`, `below_min_rows` / `above_max_rows` / `trailing_piece`
(cc 8.0's row rule, hooks/card_rows.py), `x_ln:<dtype>`, and the core's own refusal words (`no-cell:…`).  A kernel that raises is counted `error:<Type>` for that call (stock statement runs; the gate refuses at exit).
Individually switchable: MODEL_OPT_LEVERS_OFF=triatt_block_exact; AFO_TRIATT_BLOCK_EXACT_MIN_TOKENS=<N> moves the floor.
LEVER line: name=LOCAL.atlasfold.triatt_block_exact impl=opt_core.attn.pair_fused.fpf:ln=stock+core=stock_call@<core> origin=core
served=<calls> fallback=<n> fallback_by=<word:n> min_tokens=<floor> cells= cell_rows= copies= mask_copies= feed_strided=.  Class: exact."""
import os
from typing import Optional

from . import Installed, rebind
from . import pair_cells as PC
from . import triatt_block as FAST
from . import card_rows as CR

LEVER = "triatt_block_exact"
NAME = "LOCAL.atlasfold.triatt_block_exact"
TARGET = FAST.TARGET
CLASSES = FAST.CLASSES                                                # ((class, ending), ...)
LN = "stock"                                                          # the exact-candidate construction of opt_core.attn.pair_fused: the engine's own LayerNorm output is projected
MIN_TOKENS_ENV = "AFO_TRIATT_BLOCK_EXACT_MIN_TOKENS"
MIN_TOKENS = PC.MIN_TOKENS                                            # 512: the fused surround's speed floor at this geometry (pair_cells)
PROVEN_CC = ((9, 0), (8, 0))                                          # cards where the construction's byte identity with the stock statement is established (9.0: up to MAX_ROWS; 8.0: under the CARD_ROWS row rule)
MAX_ROWS_ENV = "AFO_TRIATT_BLOCK_EXACT_MAX_ROWS"
MAX_ROWS = 1280 * 1280                                                # the largest call (B*N*N rows) the exact class covers on cc 9.0: above it the statement BY NAME (`above_proven_rows`)
CARD_ROWS = {(8, 0): (5857, 3 * CR.PIECE_ROWS_SM80, CR.PIECE_ROWS_SM80)}        # cc -> (min_rows, max_rows, piece_rows): the row rule measured for these kernels at this geometry (c_z=128, 4x32) on cc 8.0 (hooks/card_rows.py: served iff min_rows <= B*N*N < max_rows AND a pieced call's trailing piece >= min_rows); no rule on 9.0


def min_tokens() -> int:
    """AFO_TRIATT_BLOCK_EXACT_MIN_TOKENS (blank / unparsable -> 512; never below 1)."""
    raw = (os.environ.get(MIN_TOKENS_ENV) or "").strip()
    try:
        v = int(raw) if raw else MIN_TOKENS
    except ValueError:
        v = MIN_TOKENS
    return max(v, 1)


def max_rows() -> int:
    """AFO_TRIATT_BLOCK_EXACT_MAX_ROWS (blank / unparsable -> 1280*1280)."""
    raw = (os.environ.get(MAX_ROWS_ENV) or "").strip()
    try:
        return int(raw) if raw else MAX_ROWS
    except ValueError:
        return MAX_ROWS


def refusal(module, z, mask, kernel_backend: str, floor: int, cc, ceiling: int = MAX_ROWS) -> Optional[str]:
    """The named reason this call runs the stock statement, or None (serve): the fast lever's vocabulary (hooks/triatt_block.refusal) with this
    lever's own floor, plus the exact class's card words (`arch:sm<NN>` outside PROVEN_CC; the row rule words `below_min_rows` /
    `above_max_rows` / `trailing_piece` on a card with a CARD_ROWS rule, hooks/card_rows.outside)."""
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
    if N < floor:
        return "below_min_tokens"
    rows = (int(z.shape[0]) if z.dim() == 4 else 1) * N * N
    if rows > ceiling:
        return "above_proven_rows"
    if not torch.is_tensor(mask) or mask.dim() != z.dim() - 1 or tuple(mask.shape[-2:]) != (N, N):
        return "mask-shape"
    if z.dim() == 4 and int(mask.shape[0]) not in (int(z.shape[0]), 1):
        return "mask-batch"
    if not z.is_cuda:
        return "cpu"
    if tuple(cc) not in PROVEN_CC:
        return f"arch:sm{cc[0]}{cc[1]}"
    return CR.outside(rows, CR.range_for(CARD_ROWS, cc))               # None = served (no rule on this card, or inside it)


def install(mode: str, tag: str, ctx: dict) -> Installed:
    import importlib
    try:
        tu = importlib.import_module(TARGET)
        import torch
        import opt_core
        from opt_core.attn import pair_fused as PF
        from opt_core.counters import Ledger
    except Exception as e:  # noqa: BLE001
        return Installed(LEVER, False, reason=f"import:{type(e).__name__}:{str(e)[:80]}")
    from ..registry import LEVERS
    words = tuple(LEVERS[LEVER]["expected"])
    PC.route(LEVER)
    cells = PC.verdict(LEVER)
    floor = min_tokens(); ceiling = max_rows()
    cc = tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else (0, 0)
    ledger = Ledger(NAME, impl=f"opt_core.attn.pair_fused.{PC.IMPL}:ln={LN}+core=stock_call@{opt_core.__version__}", origin="core", min_tokens=floor, expected=words)
    PC.record(ledger, cells)
    for k in ("copies", "mask_copies", "feed_strided"):
        ledger.set(k, 0)
    ledger.set("proven_rows", ceiling)
    for k_, v_ in CR.facts(CR.range_for(CARD_ROWS, cc)).items():      # the card's row rule on the line (min_rows= max_rows= piece_rows=; nothing on a card without one)
        ledger.set(k_, v_)

    def core(q, k, v, bias, mask5, scale):
        """The STOCK attention call of the cuequiv branch: the module-global cueq_tri_attn looked up per call (the stock cuEquivariance op, or the
        exact provider row triattn_exact binds there) — q/k/v [B, I, H, J, D] bf16, the pair bias fp32 [B, 1, H, I, J] (stock's `.float()` of the
        bf16 projection), the key mask [B, I, 1, 1, J] bool, scale = 1/sqrt(D) (the module's own self.scale)."""
        return tu.cueq_tri_attn(q, k, v, bias, mask5, scale)
    core.__qualname__ = f"cueq_tri_attn_core[atlasfold_opt:{LEVER}]"

    def make(cls_name: str, ending: bool, stock_forward):
        def forward(self, z, mask, kernel_backend: str = "torch"):
            why = refusal(self, z, mask, kernel_backend, floor, cc, ceiling)
            if why is not None:
                ledger.fallback(why)
                return stock_forward(self, z, mask, kernel_backend=kernel_backend)
            z4 = z if z.dim() == 4 else z.unsqueeze(0)
            if z4.stride(-1) != 1:                                       # the kernels need a unit channel stride: a counted copy, never silent
                z4 = z4.contiguous(); ledger.count("copies")
            m3 = mask if mask.dim() == 3 else mask.unsqueeze(0)
            if m3.shape[0] != z4.shape[0]:
                m3 = m3.expand(z4.shape[0], -1, -1)
            B, N = int(z4.shape[0]), int(z4.shape[-2])
            shape = f"{B}x{N}:{'end' if ending else 'start'}"
            word = err = None
            u = None
            try:
                x_ln = self.layernorm(z4.transpose(-2, -3) if ending else z4)   # the module's OWN LayerNorm on the attention frame, as installed (stock statement or ln_bf16's): bf16, contiguous
                if x_ln.dtype != torch.bfloat16:
                    raise PF.Unsupported("x_ln:" + str(x_ln.dtype).replace("torch.", ""))
                if not x_ln.is_contiguous():
                    x_ln = x_ln.contiguous(); ledger.count("copies")
                km = FAST.key_mask(m3, ending, ledger)
                u = PF.tri_attn_block(z4, FAST.weights(self), km, ending=ending, residual=False, impl=PC.IMPL, core=core, ln=LN, x_ln=x_ln)
            except PF.Unsupported as e:                                  # the core's named refusal, raised before any launch: keep the WORD only
                word = e.reason
            except Exception as e:  # noqa: BLE001 — a kernel that raised: the stock statement serves THIS call, the gate refuses at exit
                err = type(e).__name__
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
    return Installed(LEVER, True, lines=[lambda: (PC.set_piece_words(ledger, PF), ledger.line(tag))[1]], gates=[PC.gate_for(ledger, words)],   # epilogue= per the provider's cell
                     facts={"impl": ledger.impl, "min_tokens": floor, "classes": [c for c, _ in CLASSES], "ledger": ledger, "ln": LN, "cc": cc,
                            **{k: cells[k] for k in ("served", "cells", "cell_rows", "settings", "reason")}})
