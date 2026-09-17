"""Lever pair_block_residual — the residual adds of the pair-to-pair step folded into the fused kernels' epilogues (stock block.py L240-283:
``PairBlock.forward``, inference: ``z = add(z, tri_attn_start(z), inplace=True)``, ``z = add(z, tri_attn_end(z), inplace=True)``,
``z = add(z, transition_z(z), inplace=True)``, ``z = z * pair_mask[..., None]``).

Served path (one ``PairBlock.forward`` call = one census event), for the blocks the two fused pair levers serve (bf16 z [B, N, N, 128], N >= 512,
4 tri-attention heads, kernel_backend 'cuequiv', CUDA, contiguous z, a [B|1, N, N] bool pair mask):
  * single_to_pair, tri_mul_out / tri_mul_in: the stock statements verbatim (the trimul lever serves inside them; in-place adds as stock);
  * tri_attn_start / tri_attn_end: ``opt_core.attn.pair_fused.tri_attn_block(z, W, key_mask, ending, residual=True)`` through
    hooks/triatt_block.fused_call — the SAME prologue / core (pair_cells.core_pick: flash or K2B) / epilogue as `triatt_block`, the epilogue
    adding u into z in place (the ending node's u scattered transposed) instead of returning u for a separate ``z += u`` pass;
  * transition_z: ``pair_fused.transition(z, T, residual=True)`` — z + update from the kernel epilogue (stock rounding: update rounded to bf16,
    then the bf16 add) instead of ``z += t(z)``;
  * the mask: ``z.mul_(pair_mask[..., None])`` in place — the stock multiply's values (x*1 == x, x*0 == 0, NaN kept) without the new tensor;
  * pair_to_single: the stock statements verbatim.
A sub-call the core declines by name or whose kernel raises runs ITS stock statement for that call (``z = add(z, module(z, …))``, which the
fused levers serve or decline on their own ledgers) and is counted here (``degraded:<part>:<word>``; a kernel error refuses the gate at exit).
Refused per call, by name (the stock forward runs, the fused levers inside it as before): `training`, `no_pair_to_pair`, `backend_torch`,
`dtype:<t>`, `rank`, `c:<n>`, `heads:<n>`, `below_min_tokens`, `mask-shape`, `mask-batch`, `mask_dtype`, `cpu`, `noncontiguous`.
Requires `triatt_block` and `pair_transition` installed (row order; else the lever installs skipped `requires:<lever>`), so every kernel it calls
is one the row already runs — this lever changes WHERE the residual add happens, not which kernels compute the updates.
Class: fast.  Individually switchable: MODEL_OPT_LEVERS_OFF=pair_block_residual.
LEVER line: name=LOCAL.atlasfold.pair_block_residual impl=opt_core.attn.pair_fused.fpf(residual)@<core> origin=core served=<PairBlock calls folded>
fallback=<n> fallback_by=<word:n> shapes=<BxN:n> triatt_res=<tri-attn sub-calls added in the epilogue> trans_res=<transitions with the residual
in the epilogue> mask_inplace=<n> degraded=<n> errors=<n>."""
from typing import Optional

from . import Installed, rebind
from . import pair_cells as PC

LEVER = "pair_block_residual"
NAME = "LOCAL.atlasfold.pair_block_residual"
TARGET = "atlasfold.model.network.block"
MIN_TOKENS = PC.MIN_TOKENS


def refusal(block, z, pair_mask, kernel_backend: str) -> Optional[str]:
    """The named reason this PairBlock.forward call runs the stock forward, or None (serve). The words are the fused levers' vocabulary
    (hooks/triatt_block.refusal, hooks/pair_transition_fused) plus the block's own (`training`, `no_pair_to_pair`, `noncontiguous`, `mask_dtype`)."""
    import torch
    from . import triatt_block as TB
    if block.training:
        return "training"
    if not getattr(block, "pair_to_pair", False):
        return "no_pair_to_pair"
    if kernel_backend != "cuequiv":
        return "backend_torch"
    if not torch.is_tensor(z):
        return "rank"
    if z.dtype != torch.bfloat16:
        return "dtype:" + str(z.dtype).replace("torch.", "")
    if z.dim() != 4 or z.shape[-2] != z.shape[-3]:
        return "rank"
    if int(z.shape[-1]) != PC.TRUNK_C:
        return f"c:{int(z.shape[-1])}"
    N = int(z.shape[-2])
    if N < MIN_TOKENS:
        return "below_min_tokens"
    if getattr(block, "use_tri_attn", False):
        for m in (block.tri_attn_start, block.tri_attn_end):
            why = TB.refusal(m, z, pair_mask, kernel_backend)
            if why is not None:
                return why
    else:
        if not torch.is_tensor(pair_mask) or pair_mask.dim() != 3 or tuple(pair_mask.shape[-2:]) != (N, N):
            return "mask-shape"
        if int(pair_mask.shape[0]) not in (int(z.shape[0]), 1):
            return "mask-batch"
    if pair_mask.dtype != torch.bool:
        return "mask_dtype"
    from . import pair_transition_fused as HP
    why = HP.refusal(block.transition_z, z)                                   # c: / factor: / rank in the transition lever's own words
    if why is not None:
        return why
    if not z.is_cuda:
        return "cpu"
    if not z.is_contiguous():
        return "noncontiguous"
    return None


def install(mode: str, tag: str, ctx: dict) -> Installed:
    import importlib
    try:
        blk = importlib.import_module(TARGET)
        tu = importlib.import_module("atlasfold.model.network.primitives.triangle_update")
        trm = importlib.import_module("atlasfold.model.network.primitives.transition")
        import torch  # noqa: F401
        import opt_core
        from opt_core.attn import pair_fused as PF
        from opt_core.counters import Ledger
        from opt_core.oom import is_oom
    except Exception as e:  # noqa: BLE001
        return Installed(LEVER, False, reason=f"import:{type(e).__name__}:{str(e)[:80]}".replace(" ", "_"))
    from . import triatt_block as TB, pair_transition_fused as HP
    for cls_name, _e in TB.CLASSES:                                          # the fused levers' wrappers must be the current forwards: this lever calls the same kernels directly
        if "triatt_block" not in str(getattr(getattr(tu, cls_name).forward, "__qualname__", "")):
            return Installed(LEVER, False, reason="requires:triatt_block")
    if "pair_transition" not in str(getattr(trm.Transition.forward, "__qualname__", "")):
        return Installed(LEVER, False, reason="requires:pair_transition")
    from ..registry import LEVERS
    words = tuple(LEVERS[LEVER]["expected"])
    ledger = Ledger(NAME, impl=f"opt_core.attn.pair_fused.{PC.IMPL}(residual)@{opt_core.__version__}", origin="core", min_tokens=MIN_TOKENS, expected=words, max_shapes=16)
    for k in ("triatt_res", "trans_res", "mask_inplace", "degraded"):
        ledger.set(k, 0)
    add = blk.add
    einops = blk.einops

    def tri_attn_residual(block, module, z, pair_mask, ending: bool, kernel_backend: str):
        """z += tri_attn(z) with the add in the fused epilogue (in place); a decline / error runs the stock statement for this sub-call, counted."""
        m3 = pair_mask if pair_mask.dim() == 3 else pair_mask.unsqueeze(0)
        if m3.shape[0] != z.shape[0]:
            m3 = m3.expand(z.shape[0], -1, -1)
        shape = f"{int(z.shape[0])}x{int(z.shape[-2])}:{'end' if ending else 'start'}"
        try:
            u, word, err = TB.fused_call(PF, module, z, m3, ending, True, TB.STATE["ledger"], shape)   # the block lever's statement: its mask copies count on its ledger
        except Exception as e:  # noqa: BLE001
            if is_oom(e):
                raise
            u, word, err = None, None, type(e).__name__
        if u is not None:
            ledger.count("triatt_res"); TB.note_served(shape, ending); return z   # z was updated in place by the epilogue; the block lever's census counts its statement
        ledger.count("degraded")
        if err is not None:
            ledger.error(f"triatt:{err}")
        else:
            ledger.set("last_degraded", f"triatt:{word}")
        drop = block.dropout_columnwise_z if ending else block.dropout_rowwise_z
        return add(z, drop(module(z, pair_mask, kernel_backend=kernel_backend)), inplace=True)   # the stock statement (triatt_block's wrapper inside)

    def transition_residual(block, z):
        """z + transition_z(z) with the add in the kernel epilogue (stock rounding); a decline / error runs the stock statement, counted."""
        word = err = None
        try:
            y = HP.fused(block.transition_z, z, residual=True)                      # the transition lever's word (pf | v2 ...), residual folded in the epilogue
        except PF.Unsupported as e:
            word = e.reason; y = None
        except Exception as e:  # noqa: BLE001
            if is_oom(e):
                raise
            err = type(e).__name__; y = None
        if y is not None:
            ledger.count("trans_res"); HP.note_served(z); return y           # the transition lever's census counts its kernel
        ledger.count("degraded")
        if err is not None:
            ledger.error(f"transition:{err}")
        else:
            ledger.set("last_degraded", f"transition:{word}")
        return add(z, block.transition_z(z), inplace=True)                  # the stock statement (pair_transition's wrapper inside)

    stock_forward = blk.PairBlock.forward

    def forward(self, s, z, mask, pair_mask, kernel_backend: str = "torch"):
        why = refusal(self, z, pair_mask, kernel_backend)
        if why is not None:
            ledger.fallback(why)
            return stock_forward(self, s, z, mask, pair_mask, kernel_backend=kernel_backend)
        # Step 1: single_to_pair — the stock statements (inference: in-place add, dropout inert)
        if self.single_to_pair:
            z = add(z, self.dropout_z(self.pairwise_prod_diff(s)), inplace=True)
        # Step 2: pair to pair
        if self.use_tri_mul:                                                 # the stock statements: the trimul lever serves inside them
            z = add(z, self.dropout_rowwise_z(self.tri_mul_out(z, pair_mask, kernel_backend=kernel_backend)), inplace=True)
            z = add(z, self.dropout_rowwise_z(self.tri_mul_in(z, pair_mask, kernel_backend=kernel_backend)), inplace=True)
        if self.use_tri_attn:                                                # the residual adds folded into the fused epilogues (z updated in place)
            z = tri_attn_residual(self, self.tri_attn_start, z, pair_mask, False, kernel_backend)
            z = tri_attn_residual(self, self.tri_attn_end, z, pair_mask, True, kernel_backend)
        z = transition_residual(self, z)                                    # z + t(z) from the kernel epilogue
        z = z.mul_(pair_mask[..., None] if pair_mask.dim() == z.dim() - 1 else pair_mask)   # the stock multiply's values, in place
        ledger.count("mask_inplace")
        ledger.serve(f"{int(z.shape[0])}x{int(z.shape[-2])}")
        # Step 3: pair to single — the stock statements verbatim
        if self.pair_to_single:
            pair_bias = einops.rearrange(self.pair_to_single_bias(z), "... i j h -> ... h i j")
            s = add(s, self.attention(s, mask, pair_bias=pair_bias), inplace=True)
            s = add(s, self.transition_s(s), inplace=True)
            s = s * mask[..., None]
        return s, z
    forward.__qualname__ = f"PairBlock.forward[atlasfold_opt:{LEVER}]"
    rebind(blk.PairBlock, "forward", forward, stock_forward)
    return Installed(LEVER, True, lines=[lambda: ledger.line(tag)], gates=[PC.gate_for(ledger, words)],
                     facts={"impl": ledger.impl, "min_tokens": MIN_TOKENS, "ledger": ledger})
