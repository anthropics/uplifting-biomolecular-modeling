"""The ``big`` memory mode of rosettafold3 — ONE composition: the
``fast`` mode + every memory lever of this kit (``LEVERS``, in order), composed and recorded through ``opt_core.mem`` under the kit's memory
policy (``mem.py``, default ``capped``: the ONE writer of ``PYTORCH_CUDA_ALLOC_CONF`` / ``FPF_RF3_TG_MAX`` — the core's allocator levers are
not in the composition). The composition is PARAMETRIC on the base (``BASE``; ``modes.big_mode(base)``): a ``big_exact`` is one line here,
never a second code path.

Fast's FPF arm (``modes.py``) is applied with the components a memory lever disengages BY LEVER PROPERTY removed (``DISENGAGED``;
``fpf_arm_for``): the trunk CUDA graph (``tg``) is a memory HOLDER (its private pool grows with the pair tensor) and is dropped at every
size. Of the components that stay, the fused triangle attention (``gflash``) is size-gated by ``TRI_GFLASH_GATE`` (0: ``triatt_chunk``'s
row-block statement takes the site at every size and ``gflash`` steps aside by name); the fused Triton transition (``ttr``) holds its site
and ``transition_chunk`` steps aside by name (``ARM_SERVES``); the MSA component is re-worded to its pair-weighted-averaging cell so the
outer-product-mean site stays ``opm_chunk``'s (``UNIT_WORD``); the TriMul and attention-pair-bias components ask their provider's big
tier word (``TIER_WORD``). The FPF-FAST trimul kernel, the Triton attention-pair-bias kernel (``apb``), the fused residual adds (``res``)
and the hoist cache stay; the sampler CUDA graph does not (``DISENGAGED_SWITCHES``: the memory row runs no CUDA graph; RF3_CUDAGRAPH=0, the
arm's lever step dropped), nor do the kit levers in ``DISENGAGED_KIT`` (``xtr`` off, ``mkdit`` replaced by ``dtk``). A fast kernel that
serves no call in a process because every call took a NAMED route away from it (the FPF trimul's counted declines from the core — below its
token floor, a card without a served cell row, a call whose dtype the cell does not serve; the fused residual adds ride the trimul and
transition kernels and are silent exactly when those are; every arm kernel under ``--n_gpu > 1``, where the row-sharded pair stack owns the
sites in each rank) is ``disengaged_by_property`` on the FPF tally, never a failure of the big row. The mode reaches larger inputs than
fast and is slower
than fast where fast fits — a stated property of the mode. The bitwise-neutral subset (``NEUTRAL_VS_FAST``: bitwise to fast under a deterministic ``scatter_mean`` on single-sequence
input) is a PROPERTY column of the levers, not a mode; a lever's engagement threshold (``confidence_offload`` >= 1000 tokens,
``feature_park`` >= 64 MB tensors, the chunked forms whole below their block) is a lever property the census names per cell — there is no
line switch and no size policy.

The levers are applied in-process right after the upstream model module ``rf3.model.RF3`` executes (``TRIGGER``; every class the levers
patch exists then, and the FPF arm, where the line has one, is already in place — it patches the pairformer trunk; these levers patch the
atom encoders, the outer product, the diffusion conditioning, the confidence path, the triangle attention, the transition and the item forward).
The line is ONE lever set: no lever switch is read from the environment (a ``ROSETTAFOLD3_BIG_<LEVER>`` word there is refused by name at start).

The levers (``LEVERS``, in order):

  atom_pair_local     [chunk]   the atom-pair conditioning ``P_LL`` in window form. Stock builds ``P_LL`` dense, ``[L, L, c_atompair]``
                                (``pairformer_layers.AtomAttentionEncoderPairformer.forward``; ``af3_diffusion_transformer.
                                AtomAttentionEncoderDiffusion._forward_hoisted``, the kit's hoisted prefix) and consumes it only through
                                the atom transformer's local windows (``AttentionPairBiasDiffusion.atom_attention``: 32 queries x 128
                                keys per window). Every op on ``P_LL`` is pointwise per (l, m) pair, so the same values are computed on
                                the window index set alone: ``[nq, 32, 128, c_atompair]`` (L x 2048 elements) in place of L^2 — the dense
                                tensor (L^2 x 16 x 2 B: 13.1 GiB at L = 20 966 atoms) is the first allocation to fail as inputs grow.
  opm_chunk           [chunk]   ``OuterProductMean_AF3.forward`` over row blocks of ``l`` (``rows``, default 256): the einsum output
                                ``[N, N, c_outer^2]`` (13.5 GB at 2565 tokens) never exists whole.
  cond_chunk          [chunk]   ``DiffusionConditioning`` pair path (``cat[Z_trunk.float(), relpos]`` -> ``to_zii`` -> two transitions)
                                over row blocks (``rows``, default 512): the ``[N, N, 2 c_z]`` fp32 transients never exist whole.
  confidence_offload  [offload] each sample's ``pae_logits`` / ``pde_logits`` (``[1, N, N, 64]`` fp32) to pinned host memory as the
                                confidence head returns them; the consumers (``compile_af3_style_confidence_outputs``, ``compute_ptm``)
                                run the stock function per sample on the device from the host copy.
  feature_park        [offload] the trunk-only input feature tensors (msa_stack + its per-recycle slice, the template conditioning, the
                                stock-cast fp32 originals the engine keeps referencing) parked on pinned host between their uses and after the trunk
                                (opt_core.mem.offload.HostPark; ONE pinned budget shared with confidence_offload).
  transition_chunk    [chunk]   ``Transition.forward`` (the pair / MSA / single transitions share the class) in row blocks of the
                                leading dim (``rows``, default 256) through ``chunk_rows``: the ``[.., N, 4c]`` hidden never whole.
  triatt_chunk        [chunk]   (lean only) ``TriangleAttention.forward`` in query-row blocks (``rows``, default 512) through the core's
                                ``triangle_attention_chunked``: the full bias once, LayerNorm + q/k/v/gate + the cuEquivariance kernel per
                                block — the ``[N, N, 4 h d]`` projections (34 GB at 4076 tokens under the FPF arm's fused form) never whole.
                                Refused by name under the FPF arm (the fast base owns that site).

Exactness: ``measured, pending a deterministic comparison`` on every lever (per-pair / per-row identical arithmetic; the GEMM kernel a
different M selects is the only open term — the label is narrowed only under a deterministic ``scatter_mean``; stock's own ``scatter_mean``
is not run-to-run reproducible). The record (``opt_core.mem.record.AppliedRecord``) names the label per lever and for the mode.

Census: one unit per model forward (``RF3WithConfidence.forward``: ``fold<n>``); every lever marks the unit when its levered path
runs; a site that ran the stock path instead is a named ``fallback`` (the unit is partial); ``exit_gate`` at interpreter exit turns a
refused line lever or a partial unit into exit 3 unless the opt-out is in force — ``pred --allow-partial`` (``warm`` too), or
``ROSETTAFOLD3_BIG_ALLOW_PARTIAL=1`` in an rf3 process activated by ``ROSETTAFOLD3_OPT=big`` (recorded either way). The parent ``pred`` process
reads the child's tally (``big`` block) and fails the run the same way (``fold.big_failures``).
"""
from __future__ import annotations

import atexit
import os
import sys
from typing import Dict, Optional

from . import _core
from . import report as _report

PREFIX = "ROSETTAFOLD3"                                    # the engine prefix the core registry records (Ctx.prefix); the environment route's opt-out word is ROSETTAFOLD3_BIG_ALLOW_PARTIAL
TRIGGER = "rf3.model.RF3"                                  # the levers apply right after this upstream module executes
BASE = "fast"                                              # big = the FAST mode + the memory lever set, PARAMETRIC on the base (a `big_exact` is one line here, never a second code path)
DISENGAGED = {                                             # fast's FPF-arm components a memory lever disengages BY LEVER PROPERTY (the census names them per cell): the arm big applies = fast's arm minus these
    "tg": "memory_holder: the 48-block pairformer trunk CUDA graph keeps its private pool for the item (+17 GB at 2000 tokens on the fast row) — disengaged at every size",
    "xatt": "site_owned:triatt_chunk — the exact base's provider-served triangle attention (arm component xatt) holds the whole-pair statement big's row-block lever takes; the exact base drops it at every size (fast's arm does not carry it)",
}
UNIT_WORD = {                                              # a component of fast's arm whose SITE a memory lever keeps in big, re-worded to its other cell(s): the word big applies, the
    "msa": ("msa.pwa", ("", "card", "both", "opm"),       # words it replaces, why (the reason string)
            "site_kept:opm_chunk -- the MSA module's outer-product-mean site stays the row-block memory lever's in big: the fused OPM cell (msa.opm) holding it "
            "instead measured peak-identical at 1200 / 1408 / 2304 tokens on the H100 (13.0 / 26.9 GiB items either way) and at 1200 on the A100 (14.2 GiB; trunk +0.77 s "
            "there), i.e. not a memory lever by number (composition #7); the pair-weighted-averaging cell rides (msa.pwa)"),
}
TIER_WORD = {                                              # a component bound through a provider's FAST tier word asks that provider's BIG tier word in the memory row:
    "fast": ("fast", "big"),                             # (fast tier sub-word, big tier sub-word) -- the TriMul head word (opt_core.kernels.trimul: the big cell keeps a
    "apb": ("fast", "big"),                              # row with no extra memory cost where the fast cell would serve a faster one) and the pair-bias
}                                                          # attention component (opt_core.kernels.apb); identical rows wherever the provider's fast and big cells agree
DISENGAGED_SWITCHES = {                                    # the base's ROW SWITCHES the memory row turns off BY RULE (modes.big_mode exports them "0"; the census: row=…, LEVER state=off reason=not_in_mode:big):
    "RF3_CUDAGRAPH": "cuda_graphs: the memory row runs no CUDA graph — the sampler roll-out graph (its private pool and static clones live for the roll-out) is the "
                     "speed rows' device (exact / fast); big denoises through the same step body eagerly (RF3_CUDAGRAPH=0: upstream's loop + the RF3_HOIST cache). "
                     "The FPF arm's lever step (@L1[.warm]) leaves with it: the step re-sets the flag in-process, and its warm sub-step is the graph's",
}
DISENGAGED_KIT = {
    "xtr": "memory_holder: the core's fused-transition serve path retains workspace per call site (measured +2.6 GB at 400 tokens) — disengaged at every size",   # the base's PACKAGE levers (modes.KIT_LEVERS) a memory lever disengages by property; dtk stays (the flash kernel holds no [I, I] logits: less memory, not more)
    "mkdit": "memory_holder: the megakernel's per-roll-out pair-bias layout [24, 16, I, I] bf16 grows as I^2 (0.77 GB at 1000 tokens, 3.1 GB at 2000; measured device peak +1.2 GB at 1000 over dtk) — replaced by dtk (REPLACED_KIT) at every size",
}
ARM_SERVES = {                                             # memory levers whose SITE a component of the arm big applies serves leaner-or-equal: off BY PROPERTY at every n_gpu, named in
    "transition_chunk": ("ttr", "site_owned:fpf_ttr — the FPF arm's fused Triton pair transition holds Transition.forward (its [.., N, 4c] hidden lives per tile inside the kernel, "
                         "never whole): measured against the row-block stock forward at 1200 tokens on the H100, -3.8 s per item for +0.02 GB allocated / -0.06 GB reserved peak "
                         "(tier audit, exact_ladder_v4 items); a run that withholds fpf_ttr (MODEL_OPT_LEVERS_OFF) has the row-block lever back"),
}                                                          # the census (big.site_owned; the LEVER line says state=off reason=site_owned:fpf_ttr) like the row-sharding adapter's sites
TRI_GFLASH_GATE = 0                                        # tokens the fused triangle attention (arm component gflash) may hold in big before the row-block path takes the site: 0 = the row-block STOCK
                                                           # triangle attention at EVERY size (the fused path's transient is ~3072*N^2 bytes and the memory row never trades peak for speed); above the
                                                           # gate triatt_chunk's row-block statement takes the site BY NAME (levers.tri_forward_chunked marks `gate:N=<n>><gate>:gflash-steps-aside`;
                                                           # the FPF tally says disengaged_by_property=fpf_gflash when no call was below)
TRI_GFLASH_GATE_ROWS = {}                                      # per-card rows by compute capability, (gate tokens, least device memory MiB) -> none: every card keeps TRI_GFLASH_GATE.
                                                           # A row here needs a peak at or below the row-block path's at every size it admits.



PACK_RESIDENCY = "lru"                                     # the shared core's transition pack-cache policy under the memory row: xtr asks the provider with the EXACT word (bytes),
                                                           # whose default residency is one packed copy per layer; big keeps at most the last layer's copy (fast/exact keep 'layer')


def set_pack_residency(TR=None) -> str:
    """Big-mode activation: ask ``opt_core.kernels.transition`` to hold its packed transition weights LRU instead of per layer (the provider's
    ``pack_residency``; a core without it -> 'absent', named).  Returns the census word printed on the BIG PACK line."""
    try:
        if TR is None:
            from opt_core.kernels import transition as TR
        fn = getattr(TR, "pack_residency", None)
        if fn is None:
            return "absent:opt_core.kernels.transition.pack_residency"
        fn(PACK_RESIDENCY)
        return PACK_RESIDENCY
    except Exception as e:                                  # never a reason to refuse the memory row
        return f"error:{type(e).__name__}"


def tri_gflash_gate(gpu: Optional[dict] = None) -> int:
    """The fused triangle attention's size gate for this device: the per-card row (``TRI_GFLASH_GATE_ROWS``: compute capability + a device
    memory floor, both from the activation report's nvidia-smi probe ``rep["gpu"]`` = {name, cc, sm, memory_mib}) or ``TRI_GFLASH_GATE``."""
    g = gpu or {}
    cc = g.get("cc")
    cc = f"{cc[0]}.{cc[1]}" if isinstance(cc, (tuple, list)) and len(cc) >= 2 else str(cc or "")
    row = TRI_GFLASH_GATE_ROWS.get(cc)
    try:
        mem = int(g.get("memory_mib") or 0)
    except (TypeError, ValueError):
        mem = 0
    if row and mem >= int(row[1]):
        return int(row[0])
    return TRI_GFLASH_GATE


REPLACED_KIT = {"mkdit": "dtk"}   # a base package lever the memory mode replaces by the lever of the same site that holds no [I, I]-class tensor (dtk's flash kernel keeps no logits)
LEVERS = ("triatt_chunk", "transition_chunk", "atom_pair_local", "opm_chunk", "cond_chunk", "confidence_offload", "feature_park")   # every memory lever of this kit, in order
NEUTRAL_VS_FAST = ("atom_pair_local", "opm_chunk", "cond_chunk", "confidence_offload")   # the property column (bitwise vs fast under a deterministic scatter_mean, single-sequence input); feature_park / transition_chunk bitwise vs the EXACT base; triatt_chunk band-class
# staged, not in the composition: samples_per_pass (the diffusion-sample chunk, B4) joins LEVERS when the lever runs on a box; until then the flag is refused by name
ROWS_OPM = 256
ROWS_COND = 512
STATE: dict = {"installed": False, "record": None, "line": None, "ctx": None, "units": 0, "trigger_fired": False,
               "reason": None, "exit": None, "module": None}


class BigError(RuntimeError):
    """The mode could not be composed or applied (the message names the lever / the site)."""


def _mem():
    try:
        return _core.load("mem")
    except ImportError as e:
        raise BigError(f"big: opt_core.mem not present in this core ({e})") from e


# ------------------------------------------------------------------------------------------------------------ the composition
def fpf_arm_for(base_arm) -> tuple:
    """``(arm, disengaged)``: the base's FPF arm with the DISENGAGED components removed (the adapter's grammar ``<trimul>[+c...][@L]``);
    ``None`` -> ``(None, {})`` (a base without an arm, e.g. exact — the parametric form of a future big_exact)."""
    if not base_arm:
        return None, {}
    head, _, lev = base_arm.partition("@")
    parts = [p for p in head.split("+") if p]
    kept = [p for p in parts if p.partition(".")[0] not in DISENGAGED]                       # a component's sub-word (gflash.k2b, apb.fast) rides its component
    kept = [UNIT_WORD[p.partition(".")[0]][0] if (p.partition(".")[0] in UNIT_WORD and p.partition(".")[2] in UNIT_WORD[p.partition(".")[0]][1]) else p for p in kept]   # a component whose site a memory lever keeps binds its other cell(s) by word (msa -> msa.pwa)
    kept = [p.partition(".")[0] + "." + TIER_WORD[p.partition(".")[0]][1] if (p.partition(".")[0] in TIER_WORD and p.partition(".")[2] == TIER_WORD[p.partition(".")[0]][0]) else p for p in kept]   # a provider's FAST tier word asks its BIG tier word in the memory row (TIER_WORD)   # a component whose site a memory lever keeps binds its other cell(s) by word (msa -> msa.pwa)
    gone = {p.partition(".")[0]: DISENGAGED[p.partition(".")[0]] for p in parts if p.partition(".")[0] in DISENGAGED}
    if "RF3_CUDAGRAPH" in DISENGAGED_SWITCHES:                                               # the lever step sets RF3_CUDAGRAPH / RF3_HOIST in-process: it leaves with the graph (RF3_HOIST rides the row's export)
        lev = ""
    return "+".join(kept) + (("@" + lev) if lev else ""), gone


def compose(environ=None):
    """``(ModeTable with big, BigLine)`` from the core's composition on this kit's mode names: base ``BASE``, levers ``LEVERS``."""
    mem = _mem()
    modes = _core.load("modes")
    from .modes import DEFAULT_MODE, MODES
    table = modes.ModeTable(tuple(m for m in MODES if m != "big"), DEFAULT_MODE)
    new_table, line = mem.compose_big(table, base=BASE, levers=LEVERS)   # base fast = the core's own fast-else-exact rule (recorded as such); a parametric exact base would be its `override`
    if tuple(new_table.modes) != tuple(MODES):                    # B16: modes.MODES is the literal an external reader regexes; the composition validates it, never defines it
        raise BigError(f"big: the composed mode table {new_table.modes} is not modes.MODES {tuple(MODES)}")
    return new_table, line


ROWPAIR_OWNS = {                                           # levers whose SITE the row-sharding adapter takes under --n_gpu > 1 (rowpair.py installs after the levers, on the same
    "triatt_chunk": "site_owned:rowpair — under --n_gpu>1 the adapter's bias-gathered, row-sharded triangle attention takes TriangleAttention.forward "
                    "(query-row blocks per rank; rowpair.py over opt_core.mem.rowpair.triatt)",           # trigger): off BY PROPERTY then, named in the record and on
    "opm_chunk": "site_owned:rowpair — under --n_gpu>1 the adapter's row-local outer product mean (opm_rows) takes OuterProductMean's pair statement",   # their LEVER lines,
    "cond_chunk": "site_owned:rowpair — under --n_gpu>1 the diffusion pair conditioning is born as this rank's rows (opt_core.mem.rowpair.diffusion.pair_cond_rows)",
    "confidence_offload": "site_owned:rowpair — under --n_gpu>1 the pae/pde logits are produced and consumed per row block on every rank (never whole): nothing to offload",
}                                                                                                          # never a silent absence from the census


ALLOW_PARTIAL_ENV = f"{PREFIX}_BIG_ALLOW_PARTIAL"       # the partial-unit opt-out on the environment-activation route (ROSETTAFOLD3_OPT=big in an rf3 process of the
                                                         # user's own): read here, kit-side; `pred` / `warm` set it for their fold processes from --allow-partial only


def allow_partial_of(environ=None) -> bool:
    """The opt-out as the fold process reads it: ``ROSETTAFOLD3_BIG_ALLOW_PARTIAL=1`` in its environment (``pred --allow-partial`` exports exactly
    that to its children and removes a caller's value otherwise: fold.kit_env)."""
    env = os.environ if environ is None else environ
    return env.get(ALLOW_PARTIAL_ENV, "0") == "1"


def selection(environ=None, n_gpu: int = 1):
    """The big line's selection: every lever of the composition ON (the line is one lever set — no switch is read from the environment;
    a ``ROSETTAFOLD3_BIG_<LEVER>`` word there is refused by name at start, ``_autoload.undeclared``), except the levers of ``ROWPAIR_OWNS``
    under ``n_gpu > 1`` (off by property: the row-sharding adapter takes their sites; ``site_owned()`` names them), plus the partial-unit
    opt-out ``allow_partial_of(environ)``. The kit's levers register on import of ``levers`` (stdlib + the core registry: no torch)."""
    mem = _mem()
    from . import levers  # noqa: F401
    _, line = compose(environ)
    return mem.selection(line.levers, switches=switches_for(n_gpu), allow_partial=allow_partial_of(environ))


def switches_for(n_gpu: int = 1) -> dict:
    """The explicit lever switches of a run: ``{lever: False}`` for each lever whose site the row-sharding adapter owns under ``n_gpu > 1``;
    empty at ``n_gpu == 1`` (the whole line)."""
    out = {lv: False for lv in site_owned(n_gpu)}
    from . import leversoff as _leversoff                                    # MODEL_OPT_LEVERS_OFF: a withheld big_<lever> is this line's own named off switch (off_by_flag)
    out.update(_leversoff.big_switches())
    return out


def opt_out_of(environ=None) -> str:
    """The opt-out the PARTIAL lines name for this process: ``--allow-partial`` in a fold ``pred`` / ``warm`` started (they set
    ``ROSETTAFOLD3_OPT_TALLY_FILE`` for their fold processes), ``ROSETTAFOLD3_BIG_ALLOW_PARTIAL=1`` in an rf3 process activated through the
    environment (``ROSETTAFOLD3_OPT=big`` in a process of the user's own)."""
    from . import stack as _stack
    env = os.environ if environ is None else environ
    return _mem().OPT_OUT if env.get(_stack.ENV_TALLY_FILE) else f"{ALLOW_PARTIAL_ENV}=1"


def _allow() -> bool:
    """The opt-out in force in this process: the installed selection's (``install``), else the environment's word."""
    sel = STATE.get("selection")
    return bool(sel.allow_partial) if sel is not None else allow_partial_of()


def arm_components(environ=None) -> list:
    """The components of the FPF arm big applies in this process: the base's arm minus DISENGAGED (``fpf_arm_for``), minus the ``fpf_<component>``
    levers ``MODEL_OPT_LEVERS_OFF`` withholds (leversoff.requested; the resolver validates the names)."""
    from . import leversoff as _leversoff
    from .modes import KIT_MODES, fpf_components
    arm, _gone = fpf_arm_for(KIT_MODES[BASE].fpf_arm)
    if not arm:
        return []
    off = {n[len("fpf_"):] for n in _leversoff.requested(environ) if n.startswith("fpf_")}
    return [c for c in fpf_components(arm)[0] if c not in off]


def arm_served(environ=None) -> dict:
    """``{lever: reason}`` for the memory levers whose site a component of the applied arm serves (``ARM_SERVES``)."""
    comps = set(arm_components(environ))
    return {lv: why for lv, (comp, why) in ARM_SERVES.items() if comp in comps}


def site_owned(n_gpu: int = 1, environ=None) -> dict:
    """``{lever: reason}`` for the levers off BY PROPERTY in this process: those the row-sharding adapter's statements replace at this ``n_gpu``
    (``ROWPAIR_OWNS``, empty at 1) and those whose site a component of the applied FPF arm serves (``ARM_SERVES``: transition_chunk under ttr)."""
    out = dict(ROWPAIR_OWNS) if int(n_gpu or 1) > 1 else {}
    for lv, why in arm_served(environ).items():
        out.setdefault(lv, why)
    return out


def expected_levers(environ=None, n_gpu: int = 1) -> list:
    return list(selection(environ, n_gpu).levers)


# ------------------------------------------------------------------------------------------------------------ install / apply
def install(rep: dict, arm: bool = True) -> dict:
    """Called by ``stack.activate`` for the ``big`` row (before the row is exported): compose the line, resolve the selection (flag
    refusals fail the activation here, by name), record the block on the activation report and, unless a dry run, arm the apply
    watch and the exit gate."""
    from . import stack as _stack
    mem = _mem()
    table, line = compose()
    n_gpu = int(rep.get("n_gpu") or 1)
    sel = selection(n_gpu=n_gpu)                                   # ROWPAIR_OWNS off by property under --n_gpu>1 (switches_for), the opt-out from the environment word
    if sel.refusals:
        raise BigError("big refused — " + "; ".join(str(r) for r in sel.refusals))
    STATE.update({"installed": True, "line": line, "selection": sel, "n_gpu": n_gpu})
    from .modes import KIT_MODES
    arm, gone = fpf_arm_for(KIT_MODES[BASE].fpf_arm)
    rep["big"] = {"composition": line.as_dict(), "base": BASE, "base_fpf_arm": KIT_MODES[BASE].fpf_arm, "fpf_arm": arm, "disengaged": gone,
                    "levers": list(LEVERS), "neutral_vs_fast": list(NEUTRAL_VS_FAST), "expected": list(sel.levers),
                    "off_by_flag": [lv for lv in sel.off_by_flag if lv not in site_owned(n_gpu)], "on_by_flag": list(sel.on_by_flag), "flags": dict(sel.flags),
                    "site_owned": site_owned(n_gpu), "n_gpu": n_gpu, "allow_partial": sel.allow_partial, "trigger": TRIGGER,
                    "applied": "deferred", "allocator_writer": f"kit:mem={rep.get('mem', {}).get('policy')}"}
    if arm:
        _stack._install_watch(TRIGGER, lambda module: apply(rep, module), rep)
        atexit.register(_at_exit, rep)
    return rep["big"]


def _mod(name: str):
    """An upstream module by name at the trigger: imported here when the upstream has not imported it yet (its own modules, in its own
    interpreter — the confidence head is imported lazily inside RF3WithConfidence.__init__ upstream; the predicted-error modules by the
    engine); an ImportError is recorded and the hook stays None (a refusal by name in the lever's applies)."""
    import importlib
    m = sys.modules.get(name)
    if m is not None:
        return m
    try:
        return importlib.import_module(name)
    except Exception as e:                                       # noqa: BLE001 — recorded, the lever refuses by name
        STATE.setdefault("import_errors", {})[name] = repr(e)
        return None


def _hooks() -> dict:
    """The engine's hook points per lever, read from the upstream modules (``_mod``)."""
    dt = _mod("rf3.model.layers.af3_diffusion_transformer")
    pl = _mod("rf3.model.layers.pairformer_layers")
    op = _mod("rf3.model.layers.outer_product")
    rs = _mod("rf3.model.RF3_structure")
    rf = _mod("rf3.model.RF3")
    gf = _mod("rf3.graph_flags")
    ah = _mod("rf3.model.layers.af3_auxiliary_heads")
    pe = _mod("rf3.utils.predicted_error")
    me = _mod("rf3.metrics.predicted_error")
    sm = _mod("rf3.diffusion_samplers.inference_sampler")
    hooks = {
        "atom_pair_local": {"encoder_diffusion": getattr(dt, "AtomAttentionEncoderDiffusion", None), "encoder_pairformer": getattr(pl, "AtomAttentionEncoderPairformer", None),
                            "attention": getattr(dt, "AttentionPairBiasDiffusion", None), "graph_flags": gf},
        "opm_chunk": {"module": getattr(op, "OuterProductMean_AF3", None)},
        "cond_chunk": {"module": getattr(rs, "DiffusionConditioning", None), "graph_flags": gf},
        "confidence_offload": {"model": getattr(rf, "RF3WithConfidence", None), "head": getattr(ah, "ConfidenceHead", None),
                               "predicted_error": pe, "metrics": me, "engine": sys.modules.get("rf3.inference_engines.rf3")},
        "samples_per_pass": {"sampler": getattr(sm, "InferenceSampler", None)},
        "triatt_chunk": {"module": getattr(_mod("rf3.model.layers.attention"), "TriangleAttention", None)},   # the stock cuEquivariance site (the lean line; the FPF arm\'s forward on the fast base is refused by name)
        "transition_chunk": {"module": getattr(_mod("rf3.model.layers.layer_utils"), "Transition", None)},   # the stock transition (the lean line; the FPF arm\'s fused ttr on the fast base is refused by name)
        "feature_park": {"model": getattr(rf, "RF3WithConfidence", None), "trunk": getattr(rf, "RF3", None)},   # the item forward + the recycling generator (the lean line; applied after confidence_offload so the pinned budget is shared)
    }
    return hooks


def apply(rep: dict, module=None) -> dict:
    """The watch's callback: apply the line (``opt_core.mem.apply``, strict: a refusal is the process's NOT ACTIVE), install the unit
    delimiters, print the BIG APPLIED line, record the block. ONCE per process: a second call with the line applied is a named no-op
    (``rep["big"]["reapply"] = "skipped:already_applied"``, one ``BIG REAPPLY skipped=…`` line) — the levers derive functions from
    the classes' SOURCE (``levers._derive``), which a second application cannot read back from the derived ones; a second call carrying a
    ``module`` object other than the one the levers were applied under (``TRIGGER`` executed again: its classes are new objects the applied
    levers do not cover) is refused by name (``reapply=refused:trigger_reexecuted``, NOT ACTIVE, :class:`BigError`)."""
    from . import levers as _levers  # noqa: F401  (registers this kit's levers with the core registry at import)
    mem = _mem()
    if STATE.get("record") is not None:                                          # applied (or refused) in this process already
        first = STATE.get("module")
        if module is not None and first is not None and module is not first:
            rep["big"]["reapply"] = "refused:trigger_reexecuted"
            line = (f"{_report.PREFIX} NOT ACTIVE: big reapply refused — {TRIGGER} executed a second time in this process; the levers are bound to "
                    f"the classes of its first execution and are not derived again (levers={','.join(STATE['record'].levers) or 'none'})")
            STATE["reason"] = line
            print(line, file=sys.stderr, flush=True)
            raise BigError(line)
        rep["big"]["reapply"] = "skipped:already_applied"
        print(f"{_report.PREFIX} BIG REAPPLY skipped=already_applied trigger={TRIGGER} levers={','.join(STATE['record'].levers) or 'none'} "
              f"(the watch's callback ran again; the levers stay as applied)", file=sys.stderr, flush=True)
        return rep["big"]
    STATE["trigger_fired"] = True
    hooks = _hooks()
    rep["big"]["import_errors"] = dict(STATE.get("import_errors") or {})
    print(f"[rosettafold3-opt] BIG PACK transition_pack_residency={set_pack_residency()}", flush=True)   # the memory row holds no per-layer packed transition copies
    ctx = mem.Ctx(prefix=PREFIX, tag=_report.TAG, framework="torch", hooks=hooks,
                  settings={"opm_chunk": {"rows": ROWS_OPM}, "cond_chunk": {"rows": ROWS_COND}, "triatt_chunk": {"gate": tri_gflash_gate(rep.get("gpu"))}}, environ=os.environ, graphs=True, opt_out=opt_out_of(),
                  extra={"base": BASE, "fpf_arm_expected": rep["big"]["fpf_arm"], "disengaged": rep["big"]["disengaged"], "allocator_writer": rep["big"]["allocator_writer"], "kit_mode": rep.get("mode"),
                         "base_switch_line": rep.get("switch_line"), "fpf_arm": (rep.get("fpf") or {}).get("arm")})
    try:
        record = mem.apply(STATE["line"], ctx, strict=True, switches=switches_for(int(STATE.get("n_gpu") or 1)),
                           allow_partial=_allow())
    except mem.Refused as e:
        rep["big"]["applied"] = "refused"
        rep["big"]["refused"] = [r.as_dict() for r in e.record.refused]
        STATE.update({"record": e.record, "ctx": ctx, "reason": str(e), "module": module})
        line = e.record.not_active_line(_report.TAG)
        print(line, file=sys.stderr, flush=True)
        raise BigError(line) from e
    STATE.update({"record": record, "ctx": ctx, "module": module})
    _levers.install_units(ctx)
    rep["big"]["applied"] = "configured"
    rep["big"]["record"] = record.manifest_block()
    print(applied_line(record), file=sys.stderr, flush=True)
    return rep["big"]


def applied_line(record) -> str:
    """``[rosettafold3-opt] BIG APPLIED mode=big:<levers> base=fast exact=<label> refused=… off=… on=… allocator=kit:… line=<name>``."""
    ex = record.extra or {}
    return record.active_line(_report.TAG, line=f"big(base={ex.get('base')})", allocator_writer=ex.get("allocator_writer"), base_row=ex.get("base_switch_line"),
                              fpf=ex.get("fpf_arm")).replace(" ACTIVE ", " BIG APPLIED ", 1)


# ------------------------------------------------------------------------------------------------------------ the exit
def state() -> Optional[dict]:
    """The big block of the exit tally (JSON): the record, the census, the exit verdict — None when the mode is not installed."""
    if not STATE["installed"]:
        return None
    sel = STATE.get("selection")
    out = {"base": BASE, "levers": list(LEVERS), "composition": STATE["line"].as_dict() if STATE["line"] else None, "trigger": TRIGGER,
           "trigger_fired": STATE["trigger_fired"], "reason": STATE["reason"], "units": STATE["units"],
           "n_gpu": int(STATE.get("n_gpu") or 1), "site_owned": site_owned(STATE.get("n_gpu") or 1)}   # the levers off by property under --n_gpu>1 (their site is rowpair's)
    rec = STATE["record"]
    if rec is not None:
        out["record"] = rec.manifest_block()
        out["census"] = rec.census()
        out["exit"] = STATE["exit"] if STATE["exit"] is not None else rec.exit_gate(0, allow_partial=_allow())
        lv = sys.modules.get(__name__.rsplit(".", 1)[0] + ".levers")
        if lv is not None:
            out["chunk"] = lv.chunk_summary()
            out["offload"] = lv.offload_counters()
            out["feature_park"] = lv.feature_park_counters()
    else:
        out["record"] = None
        out["census"] = None
        out["exit"] = {"exit_code": _report_exit_not_active(), "partial": list((STATE["selection"].levers if STATE.get("selection") else ())),
                       "allow_partial": bool(STATE.get("selection") and STATE["selection"].allow_partial),
                       "reasons": {"big": STATE["reason"] or f"{TRIGGER} never executed in this process (no levers applied)"}}
    return out


def _report_exit_not_active() -> int:
    return _core.load("report").EXIT_NOT_ACTIVE


def _at_exit(rep: dict) -> None:
    """The fail-closed gate at interpreter exit: the census verdict, one line, stored for the tally JSON (the kit's exit hook runs later)."""
    rec = STATE["record"]
    if rec is None:
        if STATE["trigger_fired"]:
            return                                                   # refused at apply: the NOT ACTIVE line was printed then
        print(f"{_report.PREFIX} BIG EXIT: {TRIGGER} never executed in this process (no levers applied; no fold ran here)", file=sys.stderr, flush=True)
        return
    v = rec.exit_gate(0, allow_partial=_allow())
    STATE["exit"] = v
    c = v["census"]
    print(f"{_report.PREFIX} BIG EXIT mode={rec.mode_line()} units={c['n_units']} ok={c['n_ok']} partial={c['n_partial']} "
          f"per_lever={_json(c['per_lever'])} exit={v['exit_code']}", file=sys.stderr, flush=True)
    line = rec.exit_line(_report.TAG, v)
    if line:
        print(line, file=sys.stderr, flush=True)


def _json(o) -> str:
    import json
    return json.dumps(o, sort_keys=True, separators=(",", ":"))
