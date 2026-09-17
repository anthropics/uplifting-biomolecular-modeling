"""chai1_opt.pairtrack — the pair-track levers on the eager trunk (``LEVERS``; exact composes templ_empty / exactln / msa_pad / transition,
fast and big all seven), served by the shared core's providers and counted call by call; nothing here returns the stack's own path without a named event.

Where they attach.  Every kit mode runs trunk.pt as the structured eager trunk (``chai1_eager.trunk``: the 48 pairformer blocks, the MSA module's
4 blocks and the template embedder's 2 blocks are the same three classes ``TriangleMultiplication`` / ``TriangleAttention`` / ``Transition``),
so one attachment per class covers every pair stack of the trunk (the confidence head and the denoiser are flat transpiles: not reachable):

  ``v4trimul``  ``TriangleMultiplication`` (Chai-1's merged outgoing+incoming module: one LN_in, one [c->4d] projection a1|b1|a2|b2, [c->4d+c]
                sigmoid gates whose last c channels are the output gate, per-direction LN_out WITHOUT affine, the two directions SUMMED before
                one W_zout and the output gate) -> the shared core's ONE triangle-multiplication provider BY TIER WORD
                (``opt_core.trimul.by_word(word=<the mode's tier word>)``, ``TRIMUL_WORDS``: fast -> ``fast``, big -> ``big``; the provider's own
                cells decide the row per card, precision, c_z, crop bucket and direction, or refuse the call BY NAME), one Lever per direction:
                outgoing = rows [0:d]/[d:2d] with the pair mask, incoming = rows [2d:3d]/[3d:4d] with the TRANSPOSED pair mask (the module masks
                a2, b2 by mask[j, i]); each call packs a unit LN_out affine (ones / zeros), the shared W_zout and output-gate rows, and the two served
                updates are added (LN(x1) W^T g + LN(x2) W^T g == (LN(x1)+LN(x2)) W^T g up to rounding).  Every c_z the trunk carries asks the word
                (256: the pairformer's and the MSA module's pair stack; 64: the template pair stack when templates are present); no kit-side cell
                table, no row pinned by name.  Hook: ``CFG["trimul_impl"]`` (the module's own adapter slot; a refused call returns ``NotImplemented``
                and the module's forward runs — counted ``fallback:<row>:<reason>``, the provider's word).  ``exact`` does not compose it: no provider
                row is byte-vouched against Chai-1's bf16 merged-projection statement on this kit's stack (the provider's ``exact`` word resolves to
                its stock module-math row there), so the exact tier keeps the statement (``chai1_eager.kernels.trimul_bmm``) by name.
  ``templ_empty`` ``TemplateEmbedder``: when no template slot carries any mask (a fold without template hits:
                upstream's empty template context), the embedder's own arithmetic is z + proj_out(relu(sum_t(LN(pairformer(z_t)) * 0) / 1))
                = z + W·relu(0) = z exactly (proj_out has no bias) — the 2-block c=64 template pairformer over every empty slot computes a
                zero update.  The hook returns z without running it (one ``any()`` over the template masks per trunk call); a call WITH
                templates runs the module's forward, counted ``fallback:templates_present``.  Same output bits by the module's own algebra.

  ``exactln``   every LayerNorm STATEMENT of the structured trunk (``chai1_eager.trunk.ln_bf16`` / ``ln_f32``: the pairformer's, the MSA module's and
                the recycle projections' ``F.layer_norm(x.to(float32), (C,), w, b).to(bfloat16)`` and TriMul's affine-free fp32 output LayerNorms)
                bound to the shared core's LayerNorm provider BY TIER WORD (``opt_core.kernels.ln``: the mode's word ``exact`` | ``fast`` | ``big``;
                adapter ``opt/forward/chai1_exactln/serve.py``): every statement class (kind, C, affine, rows, layout) asks the provider once with its
                own key and the row the provider's table names serves the class — the ATen replica ``exactln`` (bitwise; the exact word's row where the
                table admits it on this stack), the Triton rows (``fastln`` …) under the fast / big words,
                or the statement itself BY NAME where the table names the stock op (the single track's N-row LayerNorms) — counted
                ``statement:<row>``.  The exact tier's rule for arithmetic replicas holds: the first call of every class an exact-class row serves is
                bit-compared against torch's own statement on its operands and a class that differs is refused by name (``serve.py``).
                Hook: ``CFG["ln_impl"]``.  The transpiled confidence head is served through its own TorchShim (``serve.bind_flat``:
                ``layer_norm`` -> the provider per class like the trunk's fp32 statements; ``to`` -> the constant-cast cache: the export casts
                its fp32 weights to bf16 at every use, a cast of the same live tensor is computed once).

  ``msa_pad``   ``MSAModule``: the export runs its row-wise MSA work (outer-product mean in 4096-row slices, pair-weighted averaging in 8192-row
                slices, the MSA transition) on all 16384 padded rows; the rows past the last one that carries any mask position are all-masked and
                never reach the module's output (exact zeros in the outer-product mean's sum, row-local elsewhere, the MSA track dropped at the
                module's end).  The hook runs the SAME statement on the leading slices only, cut on the export's own slice grid so every kernel call
                of a kept slice is the whole statement's call (``chai1_eager.msa_kernels``); the first call of every shape class runs both and
                compares the returned pair tensor (``torch.equal``) — a differing class is served whole and booked ``fallback:not_bitwise`` (refuses
                the exit gate by name); a call with nothing to cut books ``fallback:no_tail`` (declared: a deep MSA, or ``msa_rows`` sliced the rows
                first on fast / big).  Hook: ``CFG["msa_impl"]``.

  ``transition`` ``Transition`` (all four classes: pairformer pair 256 -> 512, MSA-module pair 256 -> 1024, MSA 64 -> 256, single 384 -> 768) ->
                ``opt_core.kernels.transition`` BY TIER WORD (``chai1_eager.transition_core``: every call asks the mode's tier word — exact | fast |
                big — at its own cell; exact hands the row the statement's own LayerNorm output and bit-compares every class at its first call, a
                differing class stays on the statement by name; a stock answer, a class without a cell and a refusal by name keep the statement,
                counted).  Hook: ``CFG["transition_impl"]``.
  ``trunk_n``   the trunk CALL (fast / big; tolerance-class): the ten token-indexed inputs narrowed to N' = ceil64(live tokens) and the two trunk
                representations padded back to the crop with zeros (``chai1_opt.trunk_n``: the eager trunk is shape-generic, upstream's per-crop
                components run unchanged and mask the padded positions); ``no_pad`` at a crop boundary, ``layout`` when the live tokens do not lead.

Every hook books ``served`` or ``fallback:<reason>`` (the core's refusal words) per call on its ledger; a kernel error re-raises (no retry on
the stock path).  ``lines()`` are the LEVER lines for the report / exit tally, ``census()`` the counts, ``gate()`` the fail-closed verdict
(an unexpected fallback reason or zero served calls with the lever on = refused).  ``EXPECTED_FALLBACKS`` names, per lever, the refusals the
mode expects on this engine (a fold with template features; the provider naming the stock op for a class) so they are counted, not alarming; any other word
refuses the run at exit.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from typing import Dict, Optional, Tuple

LEVERS = ("v4trimul", "templ_empty", "exactln", "triattn", "msa_pad", "transition", "trunk_n")
NAME_EXACTLN = "exactln"
TAG = "chai1-opt"
EXPECTED_FALLBACKS = {
    "v4trimul": (),                               # the provider serves every TriMul class this trunk produces on both cards (c_z 256 trunk / MSA pair stack, c_z 64
                                                  # template pair stack; a row that refuses with the tensors in hand steps aside BY NAME inside the tier word to the
                                                  # cell's next row): no refusal is expected — one that happens is counted `fallback:<row>:<reason>` and
                                                  # refuses the exit gate by name
    "templ_empty": ("templates_present",),        # a fold WITH template features runs the template pairformer (counted, not skipped)
    "exactln": ("statement", "refused"),         # the provider named the statement for a class (the single track's N-row LayerNorms: the table names the stock op); a row refused by name at serve time
    "triattn": ("stock_statement", "no_cell", "row_error"),   # triattn_core.EXPECTED_FALLBACKS: the provider names the stock op for a cell (its decision), carries no cell / row for the card, or its row fails to build / launch here at first call (traceback once): the statement serves by name
    "msa_pad": ("no_tail",),                      # chai1_eager.msa_kernels: a call with no all-masked trailing MSA slice to cut (a deep MSA; the rows already sliced by msa_rows)
    "transition": ("stock", "no_cell", "refused", "class_differs"),   # chai1_eager.transition_core: the tier word resolved to the stock op (the statement serves), a class the provider's table has no cell for yet, a refusal by name (exact vouch not on this stack / below its floor, prebuilt absent), an exact-tier row that did not bit-compare — the statement serves each, by name
                                                  # (exact: bit-compared at its first call; the statement keeps it), a stack / cell the provider refuses by name
    "trunk_n": ("no_pad", "layout"),                # chai1_opt.trunk_n: a call already at a crop boundary (N' == crop), inputs whose live tokens do not lead / dims differ
}
_STATE: Dict[str, object] = {"installed": (), "trimul": None, "pf_ledgers": {}, "orig": {}, "errors": {}, "exactln": None, "msa_pad": None}
_HERE = os.path.dirname(os.path.abspath(__file__))
TRIMUL_WORDS = {"exact": "exact", "fast": "fast", "big": "big"}                   # v4trimul: the provider TIER WORD each kit mode asks (opt_core.trimul.by_word; big never shares fast's word).
                                                                                      # `exact` is listed for completeness — the exact row does not compose v4trimul (statement by name, see above)
TRIMUL_LADDER = {"exact": "exact", "fast": "fast", "big": "fast"}                  # the core Lever's numerics class per kit mode (opt_core.trimul.MODES: stock | exact | fast — big is fast-class)
SURFACE = ("LEVERS", "TRIMUL_WORDS", "install", "trimul_word", "trimul_binding", "_trimul_hook", "_trimul_weights", "_trimul_provider", "_trimul_levers", "_templ_forward", "_exactln_module", "evidence")   # tests/test_lever_surfaces.py
EXACTLN_SERVE = os.path.normpath(os.path.join(_HERE, "..", "forward", "chai1_exactln", "serve.py"))   # the carried LayerNorm replica's serving module (kit tree)
SERVE_MODULE = "opt_core.kernels.ln"                                                  # the LayerNorm family's provider, bound BY TIER WORD (exact | fast | big) by the adapter below


class LeverUnavailable(RuntimeError):
    pass


def _trunk_module(S):
    """``chai1_eager.trunk`` — the module object of the INSTALLED eager stack (S = chai1_eager.stack)."""
    return importlib.import_module(S.__name__.rpartition(".")[0] + ".trunk")


# --------------------------------------------------------------------------------------------------------------------------- v4trimul
def _trimul_weights(m, d: int):
    """The canonical ten tensors of ONE direction of a merged TriangleMultiplication (d=0 outgoing: projection rows [0:D] | [D:2D]; d=1
    incoming: [2D:3D] | [3D:4D]); LN_out has no affine in Chai-1 (unit scale, zero shift), W_zout and the output-gate rows [4D:4D+C] are shared by both
    directions.  The core caches one pack per (provider, cache_key, dtype) on the module: the two directions are two providers."""
    import torch
    D, C = m.d, m.c
    assert m.merged_linear_p.bias is None and m.merged_linear_g.bias is None and m.linear_z_out.bias is None, "unexpected TriMul bias"
    assert tuple(m.merged_linear_p.weight.shape) == (4 * D, C) and tuple(m.merged_linear_g.weight.shape) == (4 * D + C, C), "TriMul weight shapes"
    assert m.layernorm_z_in.weight is not None and m.layernorm_z_in.bias is not None
    Wp, Wg = m.merged_linear_p.weight, m.merged_linear_g.weight
    one = torch.ones(D, device=Wp.device, dtype=torch.float32); zero = torch.zeros(D, device=Wp.device, dtype=torch.float32)
    o = 2 * D * d
    return dict(ln_in_w=m.layernorm_z_in.weight, ln_in_b=m.layernorm_z_in.bias,
                w_ap=Wp[o:o + D], w_bp=Wp[o + D:o + 2 * D], w_ag=Wg[o:o + D], w_bg=Wg[o + D:o + 2 * D],
                ln_out_w=one, ln_out_b=zero, w_o=m.linear_z_out.weight, w_og=Wg[4 * D:4 * D + C])


def trimul_word(mode: str) -> str:
    """The provider tier word `v4trimul` asks under kit mode ``mode`` (``TRIMUL_WORDS``): fast -> ``fast``, big -> ``big`` (the provider's
    memory-lean rows; never fast's word), exact -> ``exact``.  A mode outside the table is refused by name."""
    if mode not in TRIMUL_WORDS:
        raise LeverUnavailable(f"v4trimul: no provider tier word for mode {mode!r} (modes: {', '.join(TRIMUL_WORDS)})")
    return TRIMUL_WORDS[mode]


def _trimul_provider(T, d: int, word: str):
    """The Lever's provider for direction ``d`` (0 outgoing, 1 incoming): the shared core's ONE triangle-multiplication provider selected BY
    TIER WORD (``opt_core.trimul.by_word``): its cells name the row for this process's card and stack, the call's precision, c_z, crop bucket
    and direction, or refuse the call by name (``<row>:<reason>`` -> the module's forward, counted).  Nothing here names a row."""
    if not hasattr(T, "by_word"):
        raise LeverUnavailable("v4trimul refused: opt_core.trimul.by_word is missing (an older core than this kit's pin)")
    return T.by_word(lambda m: _trimul_weights(m, d), word, cache_key=f"F2.by_word.{word}.{'out' if d == 0 else 'in'}")


def _trimul_levers(mode: str):
    """The two core Levers (outgoing, incoming) of `v4trimul` for kit mode ``mode``: numerics class ``TRIMUL_LADDER[mode]``, provider word
    ``trimul_word(mode)``; each Lever carries ``chai1_word`` for the binding words."""
    from opt_core import trimul as T
    word = trimul_word(mode)
    ladder = TRIMUL_LADDER[mode]
    L_out = T.Lever(TAG + ":trimul_out", ladder, provider=_trimul_provider(T, 0, word), min_tokens=0, expected=EXPECTED_FALLBACKS["v4trimul"])
    L_in = T.Lever(TAG + ":trimul_in", ladder, provider=_trimul_provider(T, 1, word), min_tokens=0, expected=EXPECTED_FALLBACKS["v4trimul"])
    L_out.chai1_word = L_in.chai1_word = word
    for L in (L_out, L_in):                                                         # the core Lever's own line at interpreter exit too
        reg = getattr(L, "register_exit_line", None)                                  # ('LEVER name=F2.trimul impl=trimul:<row> ...', once per tag)
        if callable(reg):
            try:
                reg()
            except Exception:  # noqa: BLE001 — an exit-line registry refusal never blocks the lever
                pass
    return T, L_out, L_in


def trimul_binding() -> dict:
    """How `v4trimul` is bound in this process, for its LEVER / EXIT words: ``binding`` = ``by_word`` (the only binding: the shared provider by
    tier word); ``word`` = the tier word asked (``fast`` | ``big``); per direction ``row_<out|in>`` = the row the provider resolved for the
    calls so far (``trimul:<row>``; ``pending`` before that direction decided a call) and ``cell_<out|in>`` = ``<word>/<the provider's cell key>``
    for it.  Empty when the lever is not installed."""
    tm = _STATE["trimul"]
    if tm is None:
        return {}
    _, L_out, L_in = tm
    out = {"binding": "by_word", "word": str(getattr(L_out, "chai1_word", "?"))}
    for d, L in (("out", L_out), ("in", L_in)):
        prov = getattr(L, "provider", None)
        kern, ver = getattr(prov, "kernel", None), getattr(prov, "version", None)
        out[f"row_{d}"] = str(kern) if (kern and ver) else "pending"                 # by_word fills kernel='trimul:<row>' and version='<word>/<cell>' together at the first decided call
        out[f"cell_{d}"] = str(ver) if ver else "pending"
    return out


def _trimul_hook(T, L_out, L_in):
    import torch

    def trimul_impl(mod, z, mask):
        """CFG["trimul_impl"]: the merged module as two served directions; NotImplemented (the module's own forward) when either refuses."""
        mask_t = mask.transpose(-1, -2).contiguous() if mask is not None else None
        u_out = L_out.serve(T.Call(mod, z, mask, "outgoing", residual=False, orig=lambda: NotImplemented))
        if u_out is NotImplemented:
            L_in.serve(T.Call(mod, z, mask_t, "incoming", residual=False, orig=lambda: NotImplemented))     # count the refusal on both ledgers
            return NotImplemented
        u_in = L_in.serve(T.Call(mod, z, mask_t, "incoming", residual=False, orig=lambda: NotImplemented))
        if u_in is NotImplemented:
            return NotImplemented
        return torch.add(u_out, u_in)
    trimul_impl.chai1_opt_lever = "v4trimul"
    return trimul_impl


# --------------------------------------------------------------------------------------------------------------------------- exactln
def _exactln_module():
    """``opt/forward/chai1_exactln/serve.py`` imported once as ``chai1_exactln_serve`` from the kit tree (the adapter imports the provider ``SERVE_MODULE``)."""
    name = "chai1_exactln_serve"
    if name not in sys.modules:
        if not os.path.isfile(EXACTLN_SERVE):
            raise LeverUnavailable(f"exactln: the carried serving module is missing: {EXACTLN_SERVE}")
        spec = importlib.util.spec_from_file_location(name, EXACTLN_SERVE)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


# --------------------------------------------------------------------------------------------------------------------------- templ_empty
def _templ_forward(ledger, orig_forward):
    import torch

    def forward(self, z, template_input_feats, template_input_masks, token_pair_mask):
        if bool(torch.any(template_input_masks).item()):
            ledger.fallback("templates_present")
            return orig_forward(self, z, template_input_feats, template_input_masks, token_pair_mask)
        assert self.proj_out[1].bias is None                                          # the zero-update algebra needs a bias-free proj_out
        ledger.serve(f"{tuple(z.shape)}:slots={int(template_input_masks.shape[1])}")
        return z
    forward.chai1_opt_lever = "templ_empty"
    return forward


# --------------------------------------------------------------------------------------------------------------------------- install / evidence
def install(handle, S, levers: Tuple[str, ...], mode: str = "fast") -> dict:
    """Attach ``levers`` to the INSTALLED eager stack (``handle`` = stack.apply_eager's StackHandle, ``S`` = chai1_eager.stack).  v4trimul, exactln,
    msa_pad and transition extend the trunk wrapper's ``cfg_fn`` (the line's own CFG settings first, then the hook: ``CFG['trimul_impl']`` /
    ``['ln_impl']`` / ``['msa_impl']`` / ``['transition_impl']``); templ_empty replaces the TemplateEmbedder's class forward on
    ``chai1_eager.trunk``; triattn binds through ``triattn_core.install``; trunk_n re-classes the wrapper.  Returns report facts.  LeverUnavailable names what cannot attach."""
    unknown = [l for l in levers if l not in LEVERS]
    if unknown:
        raise LeverUnavailable(f"unknown pair-track levers {unknown}; known: {LEVERS}")
    twice = [l for l in levers if l in _STATE["installed"]]
    if twice:
        raise LeverUnavailable(f"pair-track levers {twice} are already installed in this process (one install per lever)")
    tw = (handle.parts or {}).get("trunk")
    if tw is None:
        raise LeverUnavailable("the pair-track levers attach to the eager trunk wrapper of the tier1 line; the installed handle has no trunk part")
    TR = _trunk_module(S)
    facts = {}
    if "v4trimul" in levers:                                                    # the shared provider by the mode's tier word (no kit cell table, no kernel routed by name here:
        T, L_out, L_in = _trimul_levers(mode)                                   # the provider imports and serves its own rows)
        hook = _trimul_hook(T, L_out, L_in)
        base_cfg = tw.cfg_fn

        def cfg_fn():
            if base_cfg is not None:
                base_cfg()
            TR.CFG["trimul_impl"] = hook
        cfg_fn.chai1_opt_lever = getattr(base_cfg, "chai1_opt_lever", "") + "+v4trimul"
        tw.cfg_fn = cfg_fn
        cfg_fn()
        _STATE["trimul"] = (T, L_out, L_in)
        facts["v4trimul_impl"] = L_out.provider.describe() if hasattr(L_out.provider, "describe") else "opt_core.trimul.by_word"
        facts["v4trimul_word"] = L_out.chai1_word                                # the tier word asked (the row per call is the provider's: LEVER / EXIT words name it once decided)
    if "exactln" in levers:
        SV = _exactln_module()
        try:
            facts["exactln"] = SV.install(mode)                                      # binds the mode's tier word; the provider + the replica's bindings imported now
        except RuntimeError as e:
            raise LeverUnavailable(f"exactln refused: {e}") from None
        base_cfg2 = tw.cfg_fn

        def cfg_fn2():
            if base_cfg2 is not None:
                base_cfg2()
            TR.CFG["ln_impl"] = SV.ln
        cfg_fn2.chai1_opt_lever = getattr(base_cfg2, "chai1_opt_lever", "") + "+exactln"
        tw.cfg_fn = cfg_fn2
        cfg_fn2()
        if (handle.parts or {}).get("flat_rest"):                                     # the transpiled confidence head is built by the handle's loader at its first
            inner_load = handle.loader                                                # load (chai1_eager.stack.make_loader: FlatWrapper(comps.flat(key))): bind the
                                                                                      # replica on ITS TorchShim when it is served (serve.bind_flat; once per process)
            def load_exported(comp_key, device, _inner=inner_load):
                part = _inner(comp_key, device)
                if str(comp_key) == "confidence_head.pt" and not SV.conf_fields().get("bound"):
                    flat = getattr(part, "flat", None) or getattr(getattr(part, "inner", None), "flat", None)
                    if flat is not None:
                        SV.bind_flat(flat, "confidence_head")
                return part
            load_exported.chai1_opt_lever = NAME_EXACTLN
            if getattr(handle.C1, "load_exported", None) is inner_load:
                handle.C1.load_exported = load_exported
            handle.loader = load_exported
        _STATE["exactln"] = SV
    if "templ_empty" in levers:
        from opt_core.counters import Ledger
        led = Ledger("F5.templ_empty", impl="chai1_opt.pairtrack", origin="kit", expected=EXPECTED_FALLBACKS["templ_empty"])
        _STATE["orig"]["templ_empty"] = TR.TemplateEmbedder.forward
        TR.TemplateEmbedder.forward = _templ_forward(led, TR.TemplateEmbedder.forward)
        _STATE["pf_ledgers"]["templ_empty"] = led
    if "triattn" in levers:                                                     # the triangle-attention statement on the shared core's provider (triattn_core.py: cells, binding, step-asides)
        from . import triattn_core
        led = triattn_core.new_ledger(EXPECTED_FALLBACKS["triattn"])              # its gate holds on a fold whose every call is a declared step-aside (the statement's cells)
        try:
            facts["triattn"] = triattn_core.install(tw, TR, led, mode=mode)      # by the MODE's tier word at every crop / head dim / card; wraps tw.cfg_fn after the levers above; big's trunk_chunk wraps after it
        except LookupError as e:
            raise LeverUnavailable(str(e)) from None
        _STATE["pf_ledgers"]["triattn"] = led
    if "msa_pad" in levers:                                                     # the MSA module on the leading row slices that carry a mask (chai1_eager.msa_kernels; bit-compared per class)
        MK = importlib.import_module(S.__name__.rpartition(".")[0] + ".msa_kernels")
        led = MK.new_ledger(EXPECTED_FALLBACKS["msa_pad"])                     # its gate holds on a fold with nothing to cut (declared no_tail); not_bitwise refuses it
        impl = MK.MsaPad(TR, led)
        base_cfg3 = tw.cfg_fn

        def cfg_fn3():
            if base_cfg3 is not None:
                base_cfg3()
            TR.CFG["msa_impl"] = impl
        cfg_fn3.chai1_opt_lever = getattr(base_cfg3, "chai1_opt_lever", "") + "+msa_pad"
        tw.cfg_fn = cfg_fn3
        cfg_fn3()
        _STATE["pf_ledgers"]["msa_pad"] = led
        _STATE["msa_pad"] = impl
        facts["msa_pad"] = impl.describe()
    if "transition" in levers:                                                  # the Transition statement bound by TIER word to opt_core.kernels.transition (chai1_eager.transition_core)
        TC = importlib.import_module(S.__name__.rpartition(".")[0] + ".transition_core")
        led_t = TC.new_ledger(EXPECTED_FALLBACKS["transition"])
        impl_t = TC.TransitionBinding(TR, led_t, mode)
        base_cfg4 = tw.cfg_fn

        def cfg_fn4():
            if base_cfg4 is not None:
                base_cfg4()
            TR.CFG["transition_impl"] = impl_t
        cfg_fn4.chai1_opt_lever = getattr(base_cfg4, "chai1_opt_lever", "") + "+transition"
        tw.cfg_fn = cfg_fn4
        cfg_fn4()
        _STATE["pf_ledgers"]["transition"] = led_t
        _STATE["transition"] = impl_t
        facts["transition"] = impl_t.describe()
    if "trunk_n" in levers:                                                     # the trunk call at the live-token extent (chai1_opt.trunk_n; fast / big)
        from . import trunk_n as TN
        led_n = TN.new_ledger(EXPECTED_FALLBACKS["trunk_n"])
        impl_n = TN.TrunkN(led_n)
        TN.reclass(tw, impl_n)                                                  # cooperative with hoist's ItemBoundaryTrunk below it and big's BigTrunk above it
        _STATE["pf_ledgers"]["trunk_n"] = led_n
        _STATE["trunk_n"] = impl_n
        facts["trunk_n"] = impl_n.describe()
    _STATE["installed"] = tuple(l for l in LEVERS if l in levers or l in _STATE["installed"])
    facts["pairtrack_levers"] = list(_STATE["installed"])
    return facts


def reset_for_tests() -> None:
    """Restore the hooked class forwards and forget the install (the unit tests activate several modes in one process)."""
    TR = sys.modules.get("chai1_eager.trunk")
    for name, attr in (("templ_empty", "TemplateEmbedder"),):
        orig = _STATE["orig"].pop(name, None)
        if orig is not None and TR is not None and hasattr(TR, attr):
            getattr(TR, attr).forward = orig
    if _STATE.get("exactln") is not None:
        _STATE["exactln"].reset()
        if TR is not None and isinstance(getattr(TR, "CFG", None), dict):
            TR.CFG["ln_impl"] = None
    if TR is not None and isinstance(getattr(TR, "CFG", None), dict) and "msa_impl" in TR.CFG:
        TR.CFG["msa_impl"] = None
    if TR is not None and isinstance(getattr(TR, "CFG", None), dict) and "transition_impl" in TR.CFG:
        TR.CFG["transition_impl"] = None
    _STATE.update(installed=(), trimul=None, pf_ledgers={}, orig={}, errors={}, exactln=None, msa_pad=None, transition=None, trunk_n=None)


def applied() -> Tuple[str, ...]:
    return tuple(_STATE["installed"])


def census() -> dict:
    out = {}
    tm = _STATE["trimul"]
    if tm is not None:
        _, L_out, L_in = tm
        out["v4trimul"] = {"out": L_out.census(), "in": L_in.census()}
    for name, led in _STATE["pf_ledgers"].items():
        out[name] = led.fields()
    if _STATE["exactln"] is not None:
        c = _STATE["exactln"].census()
        out["exactln"] = {"served": c["served"], "fallback": c["fallback"], "fallback_by": c["fallback_by"], "classes": c.get("classes", len(c["selftest"])), "selftest": c["selftest"],
                          "word": c.get("word"), "rows_served": c.get("rows_served", {}), "statement": c.get("statement", {}),
                          "conf": (_STATE["exactln"].conf_fields() if hasattr(_STATE["exactln"], "conf_fields") else {})}
    return out


def gate() -> Tuple[bool, str]:
    """Fail-closed: every installed lever served at least once and booked no unexpected fallback / kernel error."""
    bad = []
    tm = _STATE["trimul"]
    if tm is not None:
        for L in tm[1:]:
            g = L.gate()
            if not g.ok:
                bad.append(f"{L.tag}:{g.reason}")
    for name, led in _STATE["pf_ledgers"].items():
        g = led.gate()
        if not g.ok:
            bad.append(f"{name}:{g.reason}")
    if _STATE["exactln"] is not None:
        ok, why = _STATE["exactln"].gate()
        if not ok:
            bad.append(f"exactln:{why}")
    return (not bad), ";".join(bad)


def verdict() -> Optional[str]:
    """None when no trunk lever is installed or every installed lever's gate holds; else the refusal (the run is failed by name)."""
    if not _STATE["installed"]:
        return None
    ok, why = gate()
    return None if ok else why


def evidence(name: str) -> dict:
    """The LEVER-line evidence fields of one trunk lever in this process (empty when it is not installed)."""
    if name not in _STATE["installed"]:
        return {}
    if name == "v4trimul" and _STATE["trimul"] is not None:
        _, L_out, L_in = _STATE["trimul"]
        co, ci = L_out.census(), L_in.census()
        return {"served_out": co.get("served", 0), "served_in": ci.get("served", 0),
                "fallback_out": sum((co.get("fallback") or {}).values()), "fallback_in": sum((ci.get("fallback") or {}).values()),
                "fallback_by": dict(sorted({**(co.get("fallback") or {}), **(ci.get("fallback") or {})}.items())), **trimul_binding()}
    if name == "exactln" and _STATE["exactln"] is not None:
        c = _STATE["exactln"].census()
        return {"served": c["served"], "fallback": c["fallback"], "fallback_by": c["fallback_by"], "classes": c.get("classes", len(c["selftest"])),
                "word": c.get("word"), "rows_served": c.get("rows_served", {}), "statement": c.get("statement", {}),
                "selftest": ("all_pass" if c["selftest"] and all(v == "pass" for v in c["selftest"].values()) else (c["selftest"] or "-")), **{k: c["facts"].get(k) for k in ("torch", "cc", "stack", "provider")}}
    led = _STATE["pf_ledgers"].get(name)
    if led is None:
        return {}
    f = led.fields()
    out = {"served": f["served"], "fallback": f["fallback"], "fallback_by": dict(sorted(f["fallback_by"].items())), "shapes": f["shapes"]}
    if name == "msa_pad" and _STATE.get("msa_pad") is not None:                 # the per-class bit-compare verdicts (pass | differs) and the slice grid
        d = _STATE["msa_pad"].describe()
        out.update(classes=(d["classes"] or "none"), grid=f"{d['block_mod']}/{d['block_opm']}")
    if name == "transition" and _STATE.get("transition") is not None:           # the tier word, the rows that served, per-class verdicts, stock answers and refusals by name
        d = _STATE["transition"].describe()
        out.update(mode=d["mode"], tier=d["tier"], rows=d["rows"], classes=d["classes"], refused=d["refused"], stock=d["stock"], core=d["core"], cc=d["cc"])
    if name == "trunk_n" and _STATE.get("trunk_n") is not None:                 # grid and the last served call's crop -> N' (live)
        d = _STATE["trunk_n"].describe()
        out.update(grid=d["grid"], last=d["last"])
    return out


def lines(tag: str = TAG) -> list:
    out = []
    tm = _STATE["trimul"]
    if tm is not None:
        out += [tm[1].line(), tm[2].line()]
    for name, led in _STATE["pf_ledgers"].items():
        out.append(led.line(tag))
    if _STATE["exactln"] is not None:
        out.append(_STATE["exactln"].line(tag))
    return out


def tally_fields() -> list:
    """The trunk levers' EXIT-line fields: ``v4trimul_out_served= v4trimul_out_fallback= v4trimul_in_served= v4trimul_in_fallback=``
    (per direction, + ``v4trimul_out_fallback_by=`` / ``v4trimul_in_fallback_by=`` when that direction booked a fallback), ``exactln_word= exactln_served= exactln_fallback= exactln_classes=`` (+ rows / statement / conf / fallback_by
    when present), and ``<lever>_served= <lever>_fallback=`` (+ ``<lever>_fallback_by=``) for templ_empty, triattn, msa_pad, transition, trunk_n."""
    c = census(); f = []
    for name, v in c.items():
        if name == "v4trimul":
            for d in ("out", "in"):
                cc = v[d]; fb = cc.get("fallback") or {}                                   # the core's TriMul census books fallbacks per reason
                f.append(f"v4trimul_{d}_served={cc.get('served', 0)} v4trimul_{d}_fallback={sum(fb.values())}" +
                         (f" v4trimul_{d}_fallback_by={','.join(f'{k}:{n}' for k, n in sorted(fb.items()))}" if fb else ""))
            b = trimul_binding()                                                       # which provider row served (resolved by now): binding, word, row per direction
            if b:
                f.append(f"v4trimul_binding={b['binding']} v4trimul_word={b['word']} v4trimul_row_out={b.get('row_out')} v4trimul_row_in={b.get('row_in')}")
        elif name == "exactln":
            f.append(f"exactln_word={v.get('word')} exactln_served={v['served']} exactln_fallback={v['fallback']} exactln_classes={v['classes']}" +
                     (f" exactln_rows={','.join(f'{k}:{n}' for k, n in sorted(v['rows_served'].items()))}" if v.get("rows_served") else "") +
                     (f" exactln_statement={','.join(f'{k}:{n}' for k, n in sorted(v['statement'].items()))}" if v.get("statement") else "") +
                     (f" exactln_conf_wcast_hit={v['conf']['wcast_hit']} exactln_conf_ln_calls={v['conf']['ln_calls']}" if v.get("conf", {}).get("bound") else "") +
                     (f" exactln_fallback_by={','.join(f'{k}:{n}' for k, n in sorted(v['fallback_by'].items()))}" if v["fallback_by"] else ""))
        else:
            f.append(f"{name}_served={v.get('served', 0)} {name}_fallback={v.get('fallback', 0)}" +
                     (f" {name}_fallback_by={','.join(f'{k}:{n}' for k, n in sorted(v.get('fallback_by', {}).items()))}" if v.get("fallback_by") else ""))
    return f
