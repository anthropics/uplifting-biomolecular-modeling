"""The pair track's fused cells from the tree's core — the `fast` mode's pair levers (`trimul_v4`, `triatt_block`, `pair_transition`).

OpenFold3's pair stack is one class, `openfold3.core.model.latent.base_blocks.PairBlock` (the 48 trunk blocks, the MSA module's 4, the
template stack's 2, the confidence head's 4), whose statements are [TriMul out, TriMul in] -> [tri-attention start, tri-attention end on zᵀ]
-> pair transition, each a residual update. This module patches that class (nothing in the stock tree is edited) so each statement runs on the
shared core's cells when the call is one the cells serve, and on the line's own path otherwise — by name, counted, never silently:

    trimul_v4        PairBlock.tri_mul_out_in -> opt_core.kernels fpf_trimul_v4 `generic.trimul_packed` (LayerNorm + the four gated projections +
                     the triangle contraction + LayerNorm-out + output gate/projection + the residual, one fused TriMul per direction; c_z = c_hidden
                     in {128, 256}, N >= generic.N_MIN, z bf16 or fp32; no size ceiling — large N may exceed GPU memory (see the core's fpf_trimul_v4 HAZARDS for the workspace formula)). The template pair stack (c = 64) and N below the floor run the
                     line's TriMul (cuEquivariance, `OF3T_TRIMUL=cueq`) — `fallback:<reason>` on the census.
                     With `trimul_provider` (OPENFOLD3_OPT_TRIMUL_PROVIDER=1, of3_trimul.py) the row per call class (GPU, precision, c_z, c_hidden,
                     N bucket, direction) is the core's ONE triangle-multiplication provider's (opt_core.kernels.trimul), word fast in this kit's
                     line's tier word (of3_trimul) or the caller's word: every row, `v4` included, through
                     opt_core.trimul.by_word (refused rows counted by word, the next one asked); the template width's rows there; its own census line.
    triatt_block     PairBlock.tri_att_start_end -> opt_core.attn.pair_fused `tri_attn_block`: prologue (LayerNorm -> q|k|v|g + pair bias in one
                     kernel) -> the core's flash triangle attention -> epilogue (sigmoid gate * o @ W_o in one kernel); the ending node reads zᵀ by
                     address math (no transpose copies). impl = OPENFOLD3_OPT_PAIR_IMPL (fpf | torch). A shape the core's cell table does not attest
                     is refused by the core BEFORE any launch and the statement runs the line's tri-attention (stock's, with the line's runner configuration) —
                     `fallback:<reason>`. The epilogue adds its bf16-rounded output into z IN PLACE (a transposed scatter for the ending node: the
                     core's residual row) — no update tensor, no separate add pass; a row the core serves only without the residual returns the
                     update for a torch add (census residual=epilogue:<n>,add:<n>).
    pair_transition  PairBlock's SwiGLUTransition -> opt_core.attn.pair_fused `transition` (LayerNorm -> W_a|W_b -> silu(a)*b -> W_out; the 4*c_z
                     hidden never reaches HBM); the transition's mask is applied to the update as stock does. Only the PairBlock's transition
                     instances are served (tagged per instance); every other SwiGLUTransition in the model is untouched.

Residual stream: under `--precision fp32` the pair tensor z is fp32 and the cells run in the core's fp32-stream form (the kernels read the fp32 z,
LayerNorm in-kernel, and return the bf16 update, which is added to the fp32 stream); under `--precision bf16` (autocast) z is bf16 and the same
calls serve it. The cells' GEMMs are bf16 with fp32 accumulation on either stream (the fast tier's numerics class).

Switches (exported by the mode table, modes.LINES fast; read once at install):
    OPENFOLD3_OPT_PAIR=<lever>[,<lever>...]   the levers to install (names above); empty/unset = install nothing
    OPENFOLD3_OPT_PAIR_IMPL=fpf|lnl|torch      the attention block's / transition's surround implementation (core name; default fpf)
    OPENFOLD3_OPT_PAIR_CORE=flash_triattn|k2b  the attention core inside the block (core name; default flash_triattn)
                           |provider[:<word>]  the `triatt_provider` lever: the core's triangle-attention provider (opt_core.kernels.triattn) picks
                                               the row per call class — `provider` = the line's tier word fast (the cell table's row per
                                               GPU / head dim / key count / call form, no kit-side row order), `provider:<word>` = any
                                               provider word (a row name, a tier, k2b@m128r2 …); refused rows serve the cell's named fallback,
                                               counted; its own `LEVER name=triatt_provider … rows=<row>:<n>` exit line
    OPENFOLD3_OPT_PAIR_LN=fused|stock          LayerNorm placement (default fused: in-kernel on either stream; stock = the module's own LayerNorm feeds the kernels)
    OPENFOLD3_OPT_PAIR_STRICT=1                a trunk-shape refusal (c_z = 128 at N >= the v4 floor) that is a defect of the line (dtype / layout /
                                               mask words) raises instead of routing to the line's path (the mode table sets it); template / small-N
                                               routes, a shape the core's cell tables do not serve on this GPU (`no-cell:…`) and a call class the
                                               core's tables give to the stock statement (`cell:<row>`: the trimul provider's STOCK_ROWS
                                               cueq / torch_math — on the cc 9.0 / 8.0 c_z-128 columns the N<=100 token bucket, fast = exact =
                                               big = cueq, the same regime `n<101` names without the provider; the tri-attention engage cells'
                                               `cell:<id>:stock_block(…)`) stay routes

Evidence: `[openfold3-opt/pairfused] installed ...` at install, one `[openfold3-opt/pairfused] LEVER name=<lever> state=on served=<n>
fallback=<n> [fallback:<reason>=<n> ...]` line per lever at exit (the census; `STATE` holds the same counters for the kit's levers record).
Installed by the package hook `opt/openfold3_opt/hooks/cells/sitecustomize.py` (first in the fast line's chain) after
`openfold3.projects.of3_all_atom.model` executes — i.e. after the chained trunk-kernels hook installed the line's own routes, so these patches are
outermost and the line's routes are their named fallbacks.
"""
from __future__ import annotations

import atexit
import os
import sys
import threading
from typing import Any, Dict, Optional

PREFIX = "[openfold3-opt/pairfused]"
LEVER_NAMES = ("trimul_v4", "triatt_block", "pair_transition")
ENV_LEVERS = "OPENFOLD3_OPT_PAIR"
ENV_IMPL = "OPENFOLD3_OPT_PAIR_IMPL"
ENV_LN = "OPENFOLD3_OPT_PAIR_LN"
ENV_CORE = "OPENFOLD3_OPT_PAIR_CORE"
ENV_STRICT = "OPENFOLD3_OPT_PAIR_STRICT"
ENV_TRIMUL_PROVIDER = "OPENFOLD3_OPT_TRIMUL_PROVIDER"                  # of3_trimul.ENV: the trimul_v4 cell asks the core's trimul provider which row serves each class (read in of3_trimul.py)
ENV_TRIMUL_WORD = "OPENFOLD3_OPT_TRIMUL_WORD"                          # of3_trimul.ENV_WORD: the caller's word knob (read in of3_trimul.py)
ENV_TRIMUL_TIER = "OPENFOLD3_OPT_TRIMUL_TIER"                          # of3_trimul.ENV_TIER: the line's tier word for the buckets PREFER binds by tier word (read in of3_trimul.py)


def trimul_weights(mod) -> dict:
    """The tensors of one OpenFold3 ``TriangleMultiplicativeUpdate`` (unfused projections) in the ``opt_core.trimul.WEIGHT_KEYS`` + ``BIAS_KEYS`` vocabulary —
    the ONE name map both the single-GPU ``trimul_v4`` cell (``pack_weights(**…)``) and the row-sharded line's fused provider
    (``opt_core.mem.rowpair.trimul_fused``) read: ``a`` = ``linear_a_*`` (the projection the contraction's left operand takes), ``b`` = ``linear_b_*``,
    ``w_o`` = ``linear_z``, ``w_og`` = ``linear_g`` (the output gate on ``LayerNorm_in(z)``)."""
    return dict(ln_in_w=mod.layer_norm_in.weight, ln_in_b=mod.layer_norm_in.bias, ln_out_w=mod.layer_norm_out.weight, ln_out_b=mod.layer_norm_out.bias,
                w_o=mod.linear_z.weight, w_og=mod.linear_g.weight, w_ag=mod.linear_a_g.weight, w_ap=mod.linear_a_p.weight,
                w_bg=mod.linear_b_g.weight, w_bp=mod.linear_b_p.weight,
                b_o=mod.linear_z.bias, b_og=mod.linear_g.bias, b_ag=mod.linear_a_g.bias, b_ap=mod.linear_a_p.bias,
                b_bg=mod.linear_b_g.bias, b_bp=mod.linear_b_p.bias)
TRUNK_C = 128                                                   # the trunk / MSA-module / confidence pair width (model_config c_z); the template stack is 64

TRUNK_H, TRUNK_D, TRUNK_N = 4, 32, 4                          # the trunk pair stack's tri-attention heads x head dim and the transition widening (c_hidden = n x c_z)

STATE: Dict[str, Any] = {"installed": False, "levers": [], "impl": None, "core": None, "provider": None, "trimul_provider": None, "ln": None, "strict": False, "errors": [],
                         "counts": {n: {"served": 0, "fallback": 0} for n in LEVER_NAMES}, "reasons": {n: {} for n in LEVER_NAMES},
                         "first": {}, "cells": {}}                        # cells: lever -> {"cells": <table key> | "safe", "settings": word | None} (the pre-flight's verdict for the trunk shape)
_LOCK = threading.Lock()
_ONCE: set = set()


def _log(msg: str) -> None:
    sys.stderr.write(f"{PREFIX} {msg}\n")


def _log_once(key: str, msg: str) -> None:
    if key not in _ONCE:
        _ONCE.add(key); _log(msg)


def _count(lever: str, served: bool, reason: str = "", C: int = 0) -> None:      # C: accepted for the call sites that name the pair width (the serve-time re-plan path passes it); unused here
    with _LOCK:
        c = STATE["counts"][lever]
        if served:
            c["served"] += 1
        else:
            c["fallback"] += 1
            r = STATE["reasons"][lever]; r[reason] = r.get(reason, 0) + 1


def levers_requested(environ=None) -> list:
    environ = os.environ if environ is None else environ
    out = []
    for w in (environ.get(ENV_LEVERS) or "").replace(" ", "").replace(",", ":").split(":"):     # lever names joined by ":" (or ",")
        if not w:
            continue
        if w not in LEVER_NAMES:
            raise ValueError(f"{ENV_LEVERS}: unknown lever {w!r} (levers: {', '.join(LEVER_NAMES)})")
        if w not in out:
            out.append(w)
    return out


def _safe_fields(name: str) -> str:
    """` cells=safe settings=safe:no_cell:<shape>` for an installed lever the core serves with its SAFE cell on this GPU (pre-flight verdict), else ``."""
    v = STATE["cells"].get(name) or {}
    return f" cells={v['cells']} settings={v['settings']}" if v.get("cells") == "safe" else ""


def census_lines() -> list:
    lines = []
    for name in LEVER_NAMES:
        if name not in STATE["levers"]:
            continue
        c = STATE["counts"][name]; rs = STATE["reasons"][name]
        extra = " ".join(f"fallback:{k}={v}" for k, v in sorted(rs.items()))
        first = STATE["first"].get(name)
        lines.append(f"{PREFIX} LEVER name={name} state=on served={c['served']} fallback={c['fallback']}" + (f" {extra}" if extra else "")
                     + (f" impl={STATE['impl']} core={STATE['core']}" + _residual_fields() if name == "triatt_block" else f" impl={STATE['impl']}" + _transition_fields() if name == "pair_transition" else " impl=fpf_trimul_v4.generic")
                     + (f" first={first}" if first else "")
                     + _safe_fields(name))
        if name == "triatt_block" and STATE.get("provider") is not None:      # the block's core is the provider: its own lever line (rows served per row, refusals by kind, cells)
            lines.append(f"{PREFIX} LEVER name=triatt_provider state=on {STATE['provider'].fields()}")
        if name == "trimul_v4" and STATE.get("trimul_provider") is not None:  # the TriMul row per class is the provider's: its own lever line (rows served, kit / line counts, refusals, witness, cells)
            lines.append(f"{PREFIX} LEVER name=trimul_provider state=on {STATE['trimul_provider'].fields()}")
    return lines


def _residual_fields() -> str:
    """` residual=epilogue:<n>,add:<n>` — served triangle-attention calls whose update the epilogue kernel added into z in place (the core's residual
    row) vs returned as a tensor and added by a torch statement (a row the core serves without the residual)."""
    r = STATE.get("residual", {"epilogue": 0, "add": 0})
    return f" residual=epilogue:{r['epilogue']},add:{r['add']}"


def _transition_fields() -> str:
    """` variants=<v2_fold|v1>:<n>,… roles=<role>:c<width>:<n>,… fold=inplace:<n>,update:<n>` — which core kernel variant served the transition
    calls, the calls per role, and how many had the residual folded in place vs returned as the update."""
    variants = ",".join(f"{k}:{v}" for k, v in sorted(STATE.get("variants", {}).items())) or "none"
    roles = ",".join(f"{k}:{v}" for k, v in sorted(STATE.get("roles", {}).items())) or "none"
    fold = STATE.get("fold", {"inplace": 0, "update": 0})
    return f" variants={variants} roles={roles} fold=inplace:{fold['inplace']},update:{fold['update']}"


def trunk_cells(lever: str, impl: str) -> list:
    """The core cell rows a lever needs to serve the TRUNK pair stack (c_z=TRUNK_C): [(impl, piece, key), ...] for opt_core.attn.pair_fused
    levers; [] for a lever that pins no pair_fused row (impl 'torch': plain torch around the attention core; trimul_v4: its own cell table)."""
    if lever == "triatt_block":
        return {"fpf": [("fpf", "prologue", (TRUNK_C, TRUNK_H, TRUNK_D)), ("fpf", "epilogue", (TRUNK_C, TRUNK_H, TRUNK_D))],
                "lnl": [("lnl", "ln_linear", (TRUNK_C,)), ("lnl", "gate_transpose", (TRUNK_H * TRUNK_D,))]}.get(impl, [])
    if lever == "pair_transition":
        return {"fpf": [("fpf", "transition", (TRUNK_C, TRUNK_N * TRUNK_C))], "lnl": [("lnl", "fused_transition", (TRUNK_C, TRUNK_N * TRUNK_C))]}.get(impl, [])
    return []


def _shape_word(lever: str) -> str:
    return {"triatt_block": f"c_z={TRUNK_C},H={TRUNK_H}", "pair_transition": f"c_z={TRUNK_C},n={TRUNK_N}", "trimul_v4": f"trimul_v4:c_z={TRUNK_C}"}[lever]


def trimul_verdict(stack, environ=None) -> dict:
    """The fpf_trimul_v4 cell verdict for a named (cc, triton) part, read from the CORE's own launch-cell table by its admission rule
    (opt_core.kernels.fpf_trimul_v4.table: exact `cc|triton` row, else `cc|*`, a row admitted by its status — the loader's own statement,
    cells.cell_for) then the core's SAFE cell (opt_core.kernels.safe_settings SAFE_ROWS pair_fused:trimul):
    {"served": bool, "cells": key | "safe" | None, "settings": "safe:no_cell:cc<NN>" | None, "reason": "" | no-cell word}."""
    from opt_core.kernels import safe_settings as _ss
    cc, mm = stack
    from opt_core.kernels.fpf_trimul_v4 import table as _v4table            # the table's ONE admission rule (the loader's: opt_core.kernels.fpf_trimul_v4.cells.cell_for)
    try:
        tab = _v4table.load_table()
    except (OSError, ValueError):
        tab = {}
    for key, _kind in _v4table.candidate_keys(cc, mm):
        row = tab.get(key)
        if isinstance(row, dict) and "k1" in row and _v4table.admitted(row):
            return {"served": True, "cells": key, "settings": None, "reason": ""}
    if _ss.safe_row("pair_fused:trimul", cc) is not None:
        return {"served": True, "cells": "safe", "settings": "safe:no_cell:cc" + str(cc).replace(".", ""), "reason": ""}
    return {"served": False, "cells": None, "settings": None, "reason": f"no-cell:fpf_trimul_v4:{cc}|{mm}"}


def preflight(levers, impl: str, *, device=None, stack=None, environ=None) -> dict:
    """The cell verdict of every requested lever for the TRUNK pair stack on this part — the core's resolution (opt_core.attn.pair_fused.preflight:
    exact row -> `cc|*` row -> the lever's SAFE cell on a capability without rows -> nothing).  Returns {lever: {"served": bool, "cells": <the table
    key that serves> | "safe" | None, "settings": "safe:no_cell:<shape>" | None, "reason": "" | <the core's no-cell word>}}; {} when no CUDA device is
    visible and no ``stack`` is named.  Informational: every lever is installed either way; a served trunk shape runs on the cells (a safe cell is an
    engaged lever: the core prints its ONE line at the first call, the census line ends ` cells=safe settings=<word>`), a shape nothing serves runs
    the line's own statement per call — `fallback:<no-cell word>=<n>` on the census, one info line, exit 0."""
    from opt_core.attn import pair_fused as PF
    if stack is None:
        import torch
        if not torch.cuda.is_available():
            return {}
        device = device if device is not None else torch.device("cuda", torch.cuda.current_device())
        stack = PF._stack(device)
    out = {}
    for lever in levers:
        if lever == "trimul_v4":
            out[lever] = trimul_verdict(stack, environ)
            continue
        ds = PF.preflight(trunk_cells(lever, impl), device, stack=stack)
        refused = [d for d in ds if not d.served]
        if refused:
            out[lever] = {"served": False, "cells": None, "settings": None, "reason": refused[0].reason}
            continue
        safe = [d for d in ds if d.safe]
        keys = "+".join(sorted({d.served_by for d in ds})) if ds else None
        out[lever] = {"served": True, "cells": "safe" if safe else keys, "settings": safe[0].word(_shape_word(lever)) if safe else None, "reason": ""}
    return out


STRICT_EXEMPT = ("n<", "N<", "chunk", "training", "no-cell:", "cell:")   # refusal words that are routes by design even on the trunk shape (below the TriMul cell's token floor; chunked / training
                                                            #  calls; a shape the core's cell tables do not serve on this GPU; a call class the core's MEASURED tables give to the stock statement by
                                                            #  name — `cell:<row>` from of3_trimul (STOCK_ROWS cueq / torch_math: the c_z-128 N<=100 token bucket on cc 9.0 / 8.0, every tier) and
                                                            #  `cell:<id>:stock_block(…)` from the core's tri-attention engage cells — those calls run the line's own statement, counted
                                                            #  `fallback:<word>=<n>`, exit 0); every other trunk-shape refusal (dtype / layout / mask words: a defect of the line) fails the run under
                                                            #  strict.  `cell:` — the provider-era spelling of the small-N route raised under strict on every input of <= 100 tokens


def _strict_refusal(lever: str, reason: str, C: int, N: int) -> None:
    """A refusal on the TRUNK shape other than a route word (STRICT_EXEMPT) is a defect of the line: under OPENFOLD3_OPT_PAIR_STRICT=1 it fails the run by name."""
    if STATE["strict"] and C == TRUNK_C and not reason.startswith(STRICT_EXEMPT):
        raise RuntimeError(f"{PREFIX} {lever}: the core refused the trunk shape (c_z={C}, N={N}): {reason} — {ENV_STRICT}=1: the fast line's trunk "
                           f"statements run on the core's cells or the run fails by name (a `no-cell:<impl>:<piece>:<key>:<cc>|<triton>` reason = no launch-cell "
                           f"row for that piece at this shape on this GPU's compute capability and Triton in the core's cell tables — opt_core "
                           f"attn/pair_fused_cells.json; the TriMul cells: opt_core kernels/fpf_trimul_v4/table.json)")


# ------------------------------------------------------------------------------------------------------------------------------ trimul_v4
def _flat(z, pair_mask):
    """`(z4, mask3, lead, why)`: z as [B, N, N, C] over its leading dims (a view for the model's contiguous [1, T|S, N, N, C] stacks), the pair
    mask as [B, N, N] (a [1, N, N] mask expanded over B), the leading shape to restore, or a refusal word (`mask-batch`)."""
    lead = tuple(z.shape[:-3])
    z4 = z.reshape(-1, *z.shape[-3:]) if z.dim() != 4 else z
    m3 = None
    if pair_mask is not None:
        m3 = pair_mask.reshape(-1, *pair_mask.shape[-2:]) if pair_mask.dim() != 3 else pair_mask
        if m3.shape[0] != z4.shape[0]:
            if m3.shape[0] != 1:
                return z4, m3, lead, "mask-batch"
            m3 = m3.expand(z4.shape[0], -1, -1)
    return z4, m3, lead, None


def _install_trimul_v4(PairBlock, TMU):
    from opt_core.kernels import route
    route("fpf_trimul_v4")                                        # the core's carried cell, checked against its SUMS before first use
    from opt_core.kernels.fpf_trimul_v4 import generic as G      # the v4 kernel's face; its launch cells are the core's own table (kernels/fpf_trimul_v4/table.json)
    import torch

    _inner = PairBlock.tri_mul_out_in                             # the line's own statement (the trunk-kernels hook's cuEquivariance route when OF3T_TRIMUL=cueq)
    router = None
    from .. import of3_trimul                                    # the `trimul_provider` lever: which TriMul row serves each call class is the core's ONE triangle-
    if of3_trimul.requested():                                    #  multiplication provider's (opt_core.kernels.trimul), asked the line's tier word for every class
        router = of3_trimul.Router(of3_trimul.spec())            #  or the caller's word (OPENFOLD3_OPT_TRIMUL_WORD); left off, the cell's v4 statement below serves as before
        STATE["trimul_provider"] = router
        how = "the line's tier word" if router.by_tier else "the caller's word"
        _log(f"trimul_provider: the TriMul row per call class from the core's provider opt_core.kernels.trimul, word={router.word} ({how}, for every class it admits; then the row v4 by name, then the line's TriMul)")

    def _weights(mod):
        w = getattr(mod, "_of3v2_trimul_w", None)
        if w is None:
            w = G.pack_weights(ln_in_w=mod.layer_norm_in.weight, ln_in_b=mod.layer_norm_in.bias, ln_out_w=mod.layer_norm_out.weight, ln_out_b=mod.layer_norm_out.bias,
                               w_o=mod.linear_z.weight, w_og=mod.linear_g.weight, w_ag=mod.linear_a_g.weight, w_ap=mod.linear_a_p.weight,
                               w_bg=mod.linear_b_g.weight, w_bp=mod.linear_b_p.weight,
                               b_o=mod.linear_z.bias, b_og=mod.linear_g.bias, b_ag=mod.linear_a_g.bias, b_ap=mod.linear_a_p.bias,
                               b_bg=mod.linear_b_g.bias, b_bp=mod.linear_b_p.bias)
            mod._of3v2_trimul_w = w
        return w

    def tri_mul_out_in(self, z, pair_mask, inplace_safe, use_cueq_triangle_kernels=False, use_triton_triangle_kernels=False):
        mods = (self.tri_mul_out, self.tri_mul_in)
        why = None
        if self.training:
            why = "training"
        elif not all(type(m) in TMU for m in mods):
            why = "module:" + type(mods[0]).__name__
        elif z.dim() < 3:
            why = f"rank{z.dim()}"
        else:
            z4, m3, lead, why = _flat(z, pair_mask)                 # [.., N, N, C] over any leading dims (the template / MSA / confidence stacks' [1, T|S, N, N, C])
            C = z.shape[-1]; N = z.shape[-2]
            kit_why = None                                          # a reason of the KIT row's envelope (width, N floor, the kernel's own supported()): the v4 statement steps aside —
            if why is not None:                                     #  for the line without the provider, for the provider's next row with it; every other reason is the line's
                pass
            elif C != mods[0].c_hidden or C not in (128, 256):
                kit_why = f"c:{C}x{mods[0].c_hidden}"
            elif N < G.N_MIN:
                kit_why = f"n<{G.N_MIN}"
            if why is None and (kit_why is None or router is not None):
                if z.dtype not in (torch.bfloat16, torch.float32) or not z.is_cuda:
                    why = f"dtype:{str(z.dtype).replace('torch.', '')}"
                elif pair_mask is not None and tuple(pair_mask.shape[-2:]) != (N, N):
                    why = "mask-shape"
            if why is None and kit_why is None:
                ok, r = G.supported(z4, m3, weights=_weights(mods[0]))
                if not ok:
                    kit_why = r
            if why is None and kit_why is not None and router is None:
                why = kit_why
        if why is not None:
            _count("trimul_v4", False, why)
            _strict_refusal("trimul_v4", why, z.shape[-1], z.shape[-2])
            _log_once("tm_fb:" + why, f"trimul_v4: z={tuple(z.shape)} {z.dtype} -> the line's TriMul (fallback:{why})")
            if router is not None:
                router.count_line(why)
            return _inner(self, z, pair_mask, inplace_safe, use_cueq_triangle_kernels=use_cueq_triangle_kernels, use_triton_triangle_kernels=use_triton_triangle_kernels)
        m = m3 if m3 is None or m3.dtype == z4.dtype else m3.to(z4.dtype)     # [B, N, N] 1 = keep, in z's dtype
        if router is None:
            for mod, outgoing in ((self.tri_mul_out, True), (self.tri_mul_in, False)):
                z4 = G.trimul_packed(z4, m, outgoing=outgoing, weights=_weights(mod), residual=True)   # z + update (dropout is the identity at inference)
        else:                                                                # trimul_provider: the row per (module, direction) class from the core's provider
            dm = ((self.tri_mul_out, "outgoing"), (self.tri_mul_in, "incoming"))
            plans = [router.plan(mod, z4, m, direction, kit_why) for mod, direction in dm]
            on_line = [p for p in plans if p.kind == "line"]
            if on_line:                                                      # a module the provider gives the line's op (or nothing serves): the block's own statement, both directions, counted
                why = on_line[0].reason or "line"
                router.count_line(why)
                _count("trimul_v4", False, why)
                _strict_refusal("trimul_v4", why, z.shape[-1], z.shape[-2])
                _log_once("tm_fb:" + why, f"trimul_v4: z={tuple(z.shape)} {z.dtype} -> the line's TriMul (fallback:{why}; trimul_provider word={router.word})")
                return _inner(self, z, pair_mask, inplace_safe, use_cueq_triangle_kernels=use_cueq_triangle_kernels, use_triton_triangle_kernels=use_triton_triangle_kernels)
            for (mod, direction), plan in zip(dm, plans):
                zin = z4
                out = None
                while plan.kind == "row":
                    ref_fn = (lambda mod=mod, zin=zin, direction=direction: G.trimul_packed(zin, m, outgoing=(direction == "outgoing"), weights=_weights(mod), residual=True)) if kit_why is None else None
                    out = router.serve_row(plan, mod, zin, m, direction, ref_fn=ref_fn)
                    if out is not None:
                        break
                    plan = router.plan(mod, zin, m, direction, kit_why)      # the row left the class (serve-time refusal / bad numerics): the next row, the kit row or the line
                if out is None:                                              # the class went to the line after its rows left it: the block's own statement on the untouched z
                    why = plan.reason or "line"                              #  (both directions; the rows write new tensors, z itself was never modified), counted
                    router.count_line(why)
                    _count("trimul_v4", False, why, C=int(z.shape[-1]))
                    _log_once("tm_fb:" + why, f"trimul_v4: z={tuple(z.shape)} {z.dtype} -> the line's TriMul (fallback:{why}; trimul_provider word={router.word})")
                    return _inner(self, z, pair_mask, inplace_safe, use_cueq_triangle_kernels=use_cueq_triangle_kernels, use_triton_triangle_kernels=use_triton_triangle_kernels)
                z4 = out
        z = z4.reshape(*lead, *z4.shape[-3:])
        _count("trimul_v4", True)
        if "trimul_v4" not in STATE["first"]:
            STATE["first"]["trimul_v4"] = f"{tuple(z.shape)}:{str(z.dtype).replace('torch.', '')}"
            _log(f"trimul_v4: first served call z={tuple(z.shape)} {z.dtype} (fpf_trimul_v4.generic, residual fused; cell {G.COUNTS.get('first_call')})")
        return z
    PairBlock.tri_mul_out_in = tri_mul_out_in


# --------------------------------------------------------------------------------------------------------------------------- triatt_block
def _ln_mode(z) -> str:
    """LayerNorm placement: 'fused' (in-kernel, on a bf16 or an fp32 stream — the core reads fp32 z directly and returns the bf16 update) unless
    OPENFOLD3_OPT_PAIR_LN=stock (the module's own LayerNorm, cast to bf16, is the kernels' input)."""
    return "stock" if STATE["ln"] == "stock" else "fused"


def _install_triatt_block(PairBlock, TriangleAttention):
    from opt_core.attn import pair_fused as PF
    import torch

    _inner = PairBlock.tri_att_start_end
    impl = STATE["impl"]; core = STATE["core"]
    if isinstance(core, str) and core.startswith("provider"):          # the `triatt_provider` lever: the attention core is the core's triangle-attention provider
        from .. import of3_triattn                                      #  (opt_core.kernels.triattn), a callable core pair_fused dispatches to per call; the row per call shape
        core = of3_triattn.pair_core(core)                              #  is the cell table's for the line's tier word fast and the call's form, refusals by name
        STATE["provider"] = core
    STATE.setdefault("residual", {"epilogue": 0, "add": 0})            # served calls whose update the epilogue kernel added into z in place vs returned for an add

    def _weights(att):
        W = getattr(att, "_of3v2_triatt_w", None)
        if W is None:
            mha = att.mha
            for lin in (att.linear_z, mha.linear_q, mha.linear_k, mha.linear_v, mha.linear_g):     # OpenFold3's pair-attention linears are bias-free (default_linear_init_config mha_init /
                assert lin.bias is None, f"{PREFIX} triatt_block: {lin} carries a bias the core pack is not given"   # mha_bias_init); a bias the pack is not handed can never be dropped silently
            W = PF.pack_triattn_weights(ln_w=att.layer_norm.weight, ln_b=att.layer_norm.bias, w_q=mha.linear_q.weight, w_k=mha.linear_k.weight,
                                        w_v=mha.linear_v.weight, w_g=mha.linear_g.weight, w_b=att.linear_z.weight, w_o=mha.linear_o.weight,
                                        b_o=mha.linear_o.bias, n_heads=att.no_heads, head_dim=att.c_hidden, eps=getattr(att.layer_norm, "eps", 1e-5),
                                        device=att.linear_z.weight.device)
            att._of3v2_triatt_w = W
        return W

    def _x_ln(att, z, ending):
        """The module's own LayerNorm output in the ATTENTION frame, bf16 contiguous (the core's ln='stock' input)."""
        x = z.transpose(-2, -3) if ending else z
        return att.layer_norm(x).to(torch.bfloat16).contiguous()

    def tri_att_start_end(self, z, _attn_chunk_size, pair_mask, use_deepspeed_evo_attention, use_cueq_triangle_kernels,
                          use_triton_triangle_kernels=False, use_lma=False, inplace_safe=False):
        atts = (self.tri_att_start, self.tri_att_end)
        why = None
        if self.training:
            why = "training"
        elif not all(type(a) is TriangleAttention and getattr(a, "starting", True) for a in atts) or atts[0].mha.linear_g is None:
            why = "module"
        elif z.dim() < 3:
            why = f"rank{z.dim()}"
        else:
            z4, m3, lead, why = _flat(z, pair_mask)                 # [.., N, N, C] over any leading dims
            ln = _ln_mode(z)
            residual = (not why and type(self) is PairBlock and z4.is_contiguous()   # the epilogue adds its output into z IN PLACE (the core's residual row) — only in
                        and all(PF.supported_triattn(z4, _weights(a), m3, impl=impl, core=core, ending=e, ln=ln, residual=True)[0] for a, e in zip(atts, (False, True))))
            # PairBlock.forward proper (pairformer, MSA-module and confidence pair stacks: z there is tri_mul_out_in's fresh output, referenced by no one else) on a dense z;
            # TemplatePairBlock (it may run the attention FIRST, on the block's input — an expanded template tensor aliasing one storage) and any strided / overlapping z
            # keep the returned update + a torch add, counted residual=add
            for a, ending in (() if why or residual else zip(atts, (False, True))):
                ok, r = PF.supported_triattn(z4, _weights(a), m3, impl=impl, core=core, ending=ending, ln=ln, residual=False)
                if not ok:
                    why = r; break
        if why is not None:
            _count("triatt_block", False, why)
            _strict_refusal("triatt_block", why, z.shape[-1], z.shape[-2])
            _log_once("ta_fb:" + why, f"triatt_block: z={tuple(z.shape)} {z.dtype} -> the line's tri-attention (fallback:{why})")
            return _inner(self, z, _attn_chunk_size, pair_mask, use_deepspeed_evo_attention, use_cueq_triangle_kernels,
                          use_triton_triangle_kernels=use_triton_triangle_kernels, use_lma=use_lma, inplace_safe=inplace_safe)
        ln = _ln_mode(z)
        for a, ending in zip(atts, (False, True)):
            x_ln = _x_ln(a, z4, ending) if ln == "stock" else None
            if residual:                                          # z4 <- z4 + update IN PLACE: the epilogue kernel adds its bf16-rounded output into z (a transposed
                PF.tri_attn_block(z4, _weights(a), m3, ending=ending, residual=True, impl=impl, core=core, ln=ln, x_ln=x_ln)   # scatter for the ending node) — the
            else:                                                 # same bf16(fp32(z) + fp32(bf16(u))) the add below computes, one full pass over z fewer per node
                u = PF.tri_attn_block(z4, _weights(a), m3, ending=ending, residual=False, impl=impl, core=core, ln=ln, x_ln=x_ln)
                z4 = z4 + u                                       # the update in z's own frame (the ending node's is a transposed view); dropout = identity
        STATE["residual"]["epilogue" if residual else "add"] += 1
        z = z4.reshape(*lead, *z4.shape[-3:])
        _count("triatt_block", True)
        if "triatt_block" not in STATE["first"]:
            STATE["first"]["triatt_block"] = f"{tuple(z.shape)}:{str(z.dtype).replace('torch.', '')}:ln={ln}:residual={'epilogue' if residual else 'add'}"
            _log(f"triatt_block: first served call z={tuple(z.shape)} {z.dtype} impl={impl} core={core} ln={ln} residual={'epilogue' if residual else 'add'} (ending node by address math)")
        return z
    PairBlock.tri_att_start_end = tri_att_start_end


# ------------------------------------------------------------------------------------------------------------------------ pair_transition
TRANSITION_VARIANTS = ("v2_fold", "v1")     # this kit's ORDERED preference among the core's transition variants: the fold kernel where its row serves, else v1 (the core serves v2_fold only to a caller that names it)
ROLE_ATTR = "_of3v2_role"          # a served SwiGLUTransition's role, set at its owner's construction: pair | template | msa; refined to pairformer | confidence | msa_pair by the model walk


def _role_from_path(path: str, coarse: str) -> str:
    p = path.lower()
    if coarse in ("msa", "template"):
        return coarse
    if "template" in p:
        return "template"
    if "msa" in p:
        return "msa_pair"
    if "head" in p or "confidence" in p:
        return "confidence"
    return "pairformer"


def tag_transition_roles(model) -> int:
    """Refine the construction-time role tags of every PairBlock / MSABlock transition under `model` from its module path (pairformer |
    confidence | msa_pair | template | msa) — the census prints served calls per role. Returns the number of instances (re)tagged."""
    from openfold3.core.model.latent.base_blocks import PairBlock, MSABlock
    from openfold3.core.model.layers.transition import SwiGLUTransition
    n = 0
    for path, m in model.named_modules():
        for cls, attr, coarse in ((PairBlock, "pair_transition", "pair"), (MSABlock, "msa_transition", "msa")):
            if isinstance(m, cls):
                tr = getattr(m, attr, None)
                if isinstance(tr, SwiGLUTransition):
                    role = _role_from_path(path + "." + attr, getattr(tr, ROLE_ATTR, None) or coarse)
                    if getattr(tr, ROLE_ATTR, None) != role:
                        setattr(tr, ROLE_ATTR, role); n += 1
    return n


def _install_pair_transition(PairBlock, SwiGLUTransition):
    """The `pair_transition` lever: every PairBlock / MSABlock-owned SwiGLUTransition (pairformer, confidence, MSA-module pair and template pair
    transitions; the MSA transition) on the core's fused transition (opt_core.attn.pair_fused `transition`). Where the core's v2 kernel serves the
    shape, the row mask AND the residual add are folded: PairBlock's tail statement `z = add(z, pair_transition(z, mask), inplace)` runs as ONE
    kernel writing z + mask*T(z) in place (eval, no grad, inplace_safe — the aliasing `z += u` produces); the template block's and the MSA
    block's transitions (their owners' own forwards) take the masked update from the kernel and their caller adds it. Elsewhere (a shape / card
    the v2 rows do not cover) the v1 kernel returns the update and the mask is multiplied in torch, as before. Refusals by name -> the stock
    transition, counted."""
    from opt_core.attn import pair_fused as PF
    import inspect
    import torch
    from openfold3.core.model.latent.base_blocks import MSABlock

    _inner_fwd = SwiGLUTransition.forward
    _inner_pb_init_done = getattr(PairBlock, "_of3v2_tagging", False)
    t_impl = STATE["impl"] if STATE["impl"] in ("fpf", "lnl") else "fpf"     # the transition has fpf | lnl kernels (impl torch names the attention surround only)
    STATE.setdefault("roles", {}); STATE.setdefault("variants", {}); STATE.setdefault("fold", {"inplace": 0, "update": 0})

    def _weights(tr):
        T = getattr(tr, "_of3v2_trans_w", None)
        if T is None:
            for lin in (tr.swiglu.linear_a, tr.swiglu.linear_b, tr.linear_out):                   # bias-free by construction (swiglu_init / swiglu_transition_init); asserted, never dropped
                assert lin.bias is None, f"{PREFIX} pair_transition: {lin} carries a bias the core pack is not given"
            T = PF.pack_transition_weights(ln_w=tr.layer_norm.weight, ln_b=tr.layer_norm.bias, w_a=tr.swiglu.linear_a.weight, w_b=tr.swiglu.linear_b.weight,
                                          w_out=tr.linear_out.weight, eps=getattr(tr.layer_norm, "eps", 1e-5), device=tr.linear_out.weight.device)
            tr._of3v2_trans_w = T
        return T

    def _tag(tr, role):
        if isinstance(tr, SwiGLUTransition) and getattr(tr, ROLE_ATTR, None) is None:
            setattr(tr, ROLE_ATTR, role)

    def _served(tr, x, fold, variant):
        role = "%s:c%d" % (getattr(tr, ROLE_ATTR, "?"), x.shape[-1])
        with _LOCK:
            STATE["roles"][role] = STATE["roles"].get(role, 0) + 1
            STATE["variants"][variant] = STATE["variants"].get(variant, 0) + 1
            STATE["fold"][fold] += 1
        _count("pair_transition", True)
        if "pair_transition" not in STATE["first"]:
            STATE["first"]["pair_transition"] = f"{tuple(x.shape)}:{str(x.dtype).replace('torch.', '')}:ln={_ln_mode(x)}:{fold}"
            _log(f"pair_transition: first served call x={tuple(x.shape)} {x.dtype} role={role} fold={fold} (the core's fused transition; hidden {tr.n}*{tr.c_in} never in HBM)")

    def _refused(tr, x, why, mask, chunk_size, ckpt_chunk_size):
        _count("pair_transition", False, why)
        _strict_refusal("pair_transition", why, x.shape[-1], x.shape[-2])
        _log_once("tr_fb:" + why, f"pair_transition: x={tuple(x.shape)} {x.dtype} -> the stock transition (fallback:{why})")
        return _inner_fwd(tr, x, mask=mask, chunk_size=chunk_size, ckpt_chunk_size=ckpt_chunk_size)

    def _plan(tr, x, mask, residual):
        """(variant | None, reason): which core kernel serves this call ('v2_fold': mask + residual folded; other names: the update)."""
        if residual and torch.is_grad_enabled():
            return None, "grad"                                                             # the in-place fold is an inference statement; the update path serves grad-enabled eval calls
        if not (torch.is_tensor(x) and x.is_cuda):
            return None, "device"
        if x.dim() < 2:
            return None, "rank"
        ln = _ln_mode(x)
        ok, word = PF.transition_plan_words(x, _weights(tr), impl=t_impl, ln=ln, residual=residual, mask=mask, variants=TRANSITION_VARIANTS)
        if not ok and mask is not None and word.startswith("mask"):                       # a kernel that does not fold the mask (or a mask that is not one value per row): the update path multiplies it in torch
            ok, word = PF.transition_plan_words(x, _weights(tr), impl=t_impl, ln=ln, residual=False, mask=None, variants=TRANSITION_VARIANTS)
            return (word if ok else None), (word if not ok else "")
        return (word if ok else None), ("" if ok else word)

    # ---- role tags at construction (instances built after install; the model walk refines them) and at first use (instances built before)
    if not _inner_pb_init_done:
        for cls, attr, coarse in ((PairBlock, "pair_transition", "pair"), (MSABlock, "msa_transition", "msa")):
            init0 = cls.__init__

            def __init__(self, *a, __init0=init0, __attr=attr, __coarse=coarse, **k):
                __init0(self, *a, **k)
                _tag(getattr(self, __attr, None), "template" if (__coarse == "pair" and type(self).__name__.startswith("Template")) else __coarse)
            __init__.__wrapped__ = init0
            cls.__init__ = __init__

    # ---- PairBlock's own forward: tag at first use; the block tail as one kernel where v2 serves
    _pb_forward = PairBlock.forward
    _pb_sig = inspect.signature(_pb_forward)

    def pb_forward(self, z, *a, **k):
        tr = self.pair_transition
        _tag(tr, "template" if type(self).__name__.startswith("Template") else "pair")
        if self.training or not (isinstance(tr, SwiGLUTransition) and torch.is_tensor(z) and z.is_cuda and z.dtype == torch.bfloat16):
            return _pb_forward(self, z, *a, **k)                                            # an unserved stream: the stock body
        try:
            bound = _pb_sig.bind(self, z, *a, **k); bound.apply_defaults(); kw = bound.arguments
        except TypeError:
            return _pb_forward(self, z, *a, **k)
        if not kw.get("inplace_safe", False):
            return _pb_forward(self, z, *a, **k)                                            # the caller asked for out-of-place semantics: the stock body (its transition call takes the update path)
        pair_mask = kw.get("pair_mask")
        pair_trans_mask = pair_mask if kw.get("_mask_trans", True) else None
        if pair_trans_mask is not None and tuple(pair_trans_mask.shape) != tuple(z.shape[:-1]):   # a mask broadcast over z's leading dims: one value per row, as a view
            try:
                pair_trans_mask = pair_trans_mask.expand(tuple(z.shape[:-1]))
            except RuntimeError:
                return _pb_forward(self, z, *a, **k)                                        # not a per-row mask: the stock body (its transition statement is counted by the class forward)
        variant, why = _plan(tr, z, pair_trans_mask, True)
        if variant != "v2_fold":
            return _pb_forward(self, z, *a, **k)                                            # v1 / refused: the stock body; its transition statement is served (or refused, counted) by the class forward below
        attn_chunk = kw.get("_attn_chunk_size")
        if attn_chunk is None:
            attn_chunk = kw.get("chunk_size")
        z = self.tri_mul_out_in(z=z, pair_mask=pair_mask, inplace_safe=True, use_cueq_triangle_kernels=kw.get("use_cueq_triangle_kernels", False),
                                use_triton_triangle_kernels=kw.get("use_triton_triangle_kernels", False))
        z = self.tri_att_start_end(z=z, _attn_chunk_size=attn_chunk, pair_mask=pair_mask, use_deepspeed_evo_attention=kw.get("use_deepspeed_evo_attention", False),
                                   use_cueq_triangle_kernels=kw.get("use_cueq_triangle_kernels", False), use_triton_triangle_kernels=kw.get("use_triton_triangle_kernels", False),
                                   use_lma=kw.get("use_lma", False), inplace_safe=True)
        try:
            z.view(-1, z.shape[-1])
        except RuntimeError:                                                                # the attention cell handed back rows that do not view as [M, c]: the stock tail statement
            from openfold3.core.utils.tensor_utils import add
            return add(z, tr(z, mask=pair_trans_mask, chunk_size=kw.get("chunk_size")), inplace=True)
        try:
            PF.transition(z, _weights(tr), residual=True, ln=_ln_mode(z), impl=t_impl, mask=pair_trans_mask, out=z, variants=TRANSITION_VARIANTS)   # z <- z + mask*T(z), one kernel, in place
        except PF.Unsupported as e:                                                         # a word raised at launch (the kernel's own domain check, a cell that fails to build): the stock tail statement, counted
            from openfold3.core.utils.tensor_utils import add
            why = e.reason if e.reason.startswith("no-cell:") else "fold:" + e.reason      # a cell that fails to build is a route word (no-cell:…); a kernel domain word on the trunk shape is a defect under strict
            _count("pair_transition", False, why)
            _strict_refusal("pair_transition", why, z.shape[-1], z.shape[-2])
            _log_once("tr_fb:" + why, f"pair_transition: z={tuple(z.shape)} {z.dtype} -> the stock tail statement (fallback:{why})")
            return add(z, _inner_fwd(tr, z, mask=pair_mask if kw.get("_mask_trans", True) else None, chunk_size=kw.get("chunk_size")), inplace=True)
        _served(tr, z, "inplace", "v2_fold")
        return z
    if not _inner_pb_init_done:
        PairBlock.forward = pb_forward; PairBlock._of3v2_tagging = True

    # ---- the class forward: every tagged instance whose caller adds the update itself (template / MSA blocks, out-of-place callers, v1 cards)
    def forward(self, x, mask=None, chunk_size=None, ckpt_chunk_size=None):
        if getattr(self, ROLE_ATTR, None) is None or not torch.is_tensor(x) or x.dim() < 2:
            return _inner_fwd(self, x, mask=mask, chunk_size=chunk_size, ckpt_chunk_size=ckpt_chunk_size)
        if self.training:
            return _refused(self, x, "training", mask, chunk_size, ckpt_chunk_size)
        if ckpt_chunk_size is not None:
            return _refused(self, x, "ckpt_chunk", mask, chunk_size, ckpt_chunk_size)
        mask_rows = mask
        if mask is not None and tuple(mask.shape) != tuple(x.shape[:-1]):                   # a mask broadcast over leading dims (the template stack's [.., 1, N, N]): expanded to the rows once
            try:
                mask_rows = mask.expand(tuple(x.shape[:-1]))
            except RuntimeError:
                return _refused(self, x, "mask-shape", mask, chunk_size, ckpt_chunk_size)
        variant, why = _plan(self, x, mask_rows, False)
        if variant is None:
            return _refused(self, x, why, mask, chunk_size, ckpt_chunk_size)
        ln = _ln_mode(x)
        if variant == "v2_fold":
            try:
                u = PF.transition(x, _weights(self), residual=False, ln=ln, impl=t_impl, mask=mask_rows, variants=TRANSITION_VARIANTS)
            except PF.Unsupported as e:                                                     # a word raised at launch: the stock statement, counted
                return _refused(self, x, e.reason if e.reason.startswith("no-cell:") else "update:" + e.reason, mask, chunk_size, ckpt_chunk_size)
            _served(self, x, "update", "v2_fold")
            return u if u.dtype == x.dtype or torch.is_autocast_enabled() else u.to(x.dtype)
        x_ln = self.layer_norm(x).to(torch.bfloat16).contiguous() if ln == "stock" else None
        xin = x if x.dtype == torch.bfloat16 and x.stride(-1) == 1 else (x.contiguous() if x.dtype == torch.bfloat16 else x)
        u = PF.transition(xin, _weights(self), residual=False, ln=ln, x_ln=x_ln, impl=t_impl, variants=TRANSITION_VARIANTS)
        if mask is not None:
            u = u * mask.unsqueeze(-1).to(u.dtype)                # stock: x = x * mask ([*, N, 1])
        _served(self, x, "update", variant)
        return u if u.dtype == x.dtype or torch.is_autocast_enabled() else u.to(x.dtype)
    SwiGLUTransition.forward = forward

    # ---- the model walk: per-role tags (pairformer | confidence | msa_pair | template | msa) once the model is built
    _arm_model_walk()


_WALK = {"armed": False, "done": False}


def _arm_model_walk():
    """Wrap OpenFold3.__init__ once (at or after the model module's import): after the stock constructor the transitions get their per-role
    tags from the module paths. Nothing else about the model changes."""
    if _WALK["armed"]:
        return
    _WALK["armed"] = True
    modname, clsname = "openfold3.projects.of3_all_atom.model", "OpenFold3"

    def arm(module):
        cls = getattr(module, clsname, None)
        if cls is None:
            return
        init0 = cls.__init__

        def __init__(self, *a, **k):
            init0(self, *a, **k)
            if not _WALK["done"]:
                _WALK["done"] = True
                n = tag_transition_roles(self)
                _log(f"pair_transition: {n} transition instances tagged by role from the model's module paths")
        __init__.__wrapped__ = init0
        cls.__init__ = __init__
    mod = sys.modules.get(modname)
    if mod is not None:
        arm(mod); return
    import importlib.abc
    import importlib.util

    class _RoleWalkFinder(importlib.abc.MetaPathFinder):            # transient: removes itself when the model module imports (not one of env.HOOK_FINDER_CLASSES)
        def find_spec(self, fullname, path=None, target=None):
            if fullname != modname:
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(fullname)
            if spec is None or spec.loader is None:
                return spec
            real_exec = spec.loader.exec_module

            def exec_module(module, _real=real_exec):
                _real(module)
                arm(module)
            spec.loader.exec_module = exec_module
            return spec
    sys.meta_path.insert(0, _RoleWalkFinder())


# --------------------------------------------------------------------------------------------------------------------------------- install
def install(environ=None) -> dict:
    """Install the requested levers on PairBlock (idempotent). Raises on an unknown lever name, a missing core module (the kit's core pin gate
    ran before this: a core without opt_core.attn.pair_fused is a pin mismatch named there), or a core cell that fails its SUMS check."""
    environ = os.environ if environ is None else environ
    if STATE["installed"]:
        return STATE
    levers = levers_requested(environ)
    if not levers:
        return STATE
    STATE["impl"] = (environ.get(ENV_IMPL) or "fpf").strip()
    if STATE["impl"] not in ("fpf", "lnl", "torch"):
        raise ValueError(f"{ENV_IMPL}={STATE['impl']!r} is not one of fpf|lnl|torch (opt_core.attn.pair_fused impl names)")
    STATE["core"] = (environ.get(ENV_CORE) or "flash_triattn").strip()
    if STATE["core"] not in ("flash_triattn", "k2b") and not (STATE["core"] == "provider" or STATE["core"].startswith("provider:")):
        raise ValueError(f"{ENV_CORE}={STATE['core']!r} is not one of flash_triattn|k2b (opt_core.attn.pair_fused attention cores) | provider[:<word>] "
                         "(the core's triangle-attention provider opt_core.kernels.triattn: the tier word fast, or <word>)")
    STATE["ln"] = (environ.get(ENV_LN) or "fused").strip()
    if STATE["ln"] not in ("fused", "stock"):
        raise ValueError(f"{ENV_LN}={STATE['ln']!r} is not one of fused|stock")
    STATE["strict"] = environ.get(ENV_STRICT, "").strip() == "1"
    verdicts = preflight(levers, STATE["impl"], environ=environ)       # what serves each lever's TRUNK shape on this GPU (a table row / the core's safe cell / nothing): the census' cells= fields;
    from openfold3.core.model.latent.base_blocks import PairBlock      #  every lever is installed either way — a shape nothing serves runs the line's own statement per call, counted
    from openfold3.core.model.layers.triangular_attention import TriangleAttention
    from openfold3.core.model.layers.triangular_multiplicative_update import TriangleMultiplicationOutgoing, TriangleMultiplicationIncoming
    from openfold3.core.model.layers.transition import SwiGLUTransition
    if "trimul_v4" in levers:
        _install_trimul_v4(PairBlock, (TriangleMultiplicationOutgoing, TriangleMultiplicationIncoming))
    if "triatt_block" in levers:
        _install_triatt_block(PairBlock, TriangleAttention)
    if "pair_transition" in levers:
        _install_pair_transition(PairBlock, SwiGLUTransition)
    STATE["installed"] = True; STATE["levers"] = levers
    STATE["cells"] = {l: {"cells": v["cells"], "settings": v["settings"]} for l, v in verdicts.items() if v.get("served")}
    _log(f"installed levers={','.join(levers)} impl={STATE['impl']} core={STATE['core']} ln={STATE['ln']} strict={int(STATE['strict'])} "
         f"(PairBlock class-wide: trunk, MSA-module, template and confidence pair stacks; refused shapes run the line's own routes by name)")
    atexit.register(lambda: [sys.stderr.write(l + "\n") for l in census_lines()])
    return STATE
