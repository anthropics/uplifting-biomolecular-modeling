"""The ``big`` memory mode for Chai-1: the ``fast`` line plus the memory levers that make crop 2048 fit on one 80 GB card.

``big`` composes on ``modes.KIT_MODES['fast']`` BY REFERENCE (levels, eager line, DSTEP levers, stack rule) and applies the
levers below through ``opt_core.mem`` (the shared library: registry, applied record, census, exit gate); the allocator policy of the
mode (expandable segments) is the package's ``alloc`` lever, implied by the mode (``modes.KIT_MODES['big'].implied_optin``: one
allocator producer, ``alloc.py`` over ``opt_core.mem.torch_alloc``). A mode's memory levers
are fixed: no switch turns one off or adds one, and their settings are the constants below — except the documented size gates,
``CHAI1_BIG_<LEVER>_MIN_N`` and ``CHAI1_BIG_HOIST2_MAX_N`` (``GATE_WORDS``: the pair extent at or above which that lever applies / hoist2 stands down), read
here and handed to the core as settings; any other ``CHAI1_BIG_*`` name in the environment is refused by name (``gate_words``, and the
package's undeclared-variable gate). A partial unit's opt-out is the driver's ``--allow-partial`` (recorded; on the CHAI1_OPT route the
word is ``CHAI1_OPT_ALLOW_PARTIAL=1``). The line (``LINE``) is the ``lean`` line:

* ``msa_rows`` (chunk; measured) — the trunk's MSA module runs on the MSA rows that carry any mask instead of the 16384 rows upstream pads
  every MSA to (``chai_lab/data/dataset/all_atom_feature_context.py`` MAX_MSA_DEPTH; ``chai1.py`` create_empty at depth 16384 for the
  ``msa_directory=None`` form): ``msa_input_feats[:, :S_eff]`` / ``msa_mask[:, :S_eff]`` sliced at the trunk wrapper's call, before the
  stack moves them to the device, where ``S_eff`` = 1 + the index of the last row with any True (padding is appended at the end:
  ``msa_context.py`` pad). The dropped rows are all-masked: the outer-product mean adds exact zeros for them (``x.masked_fill_(…, 0)``),
  the pair-weighted averaging and the MSA transition are row-independent; what can move a bit is cuBLAS's kernel choice at a smaller M
  — hence ``measured``, not bitwise by construction. Single-sequence form: S_eff = 1 and the
  pair-weighted-averaging transient of crop 2048 (the fast line's OOM site) is gone; MSA form: S_eff = the real depth.
* ``msa_chunk`` (chunk; measured; setting ``rows``, default 1024) — the eager trunk's ``MSAPairWeightedAveraging`` instances re-classed
  onto the core's row chunker (``opt_core.mem.chunk.chunk_rows`` with ``site=msa_rows``: the carried forward on ``rows``-row blocks of the
  MSA, the mask block by the same offsets, assembled into one output plane): the per-block ``vg`` projection is rows × N × 512 × 2 B
  instead of 8192 × N × 512 × 2 B, and the carried forward's list-of-slices + ``cat`` (two output planes) never runs
  above ``rows``. Matters in the MSA form (S_eff > rows); the outer-product mean is not blocked over S (its sum over S precedes a
  LayerNorm); ``opm_chunk`` blocks it over pair rows instead.
* ``trunk_chunk`` (chunk; measured; ``rows`` 256, ``min_n`` from the --config's ``GATE_TABLE`` row, then the device's memory class) — the pair
  and template stacks' merged triangle multiplication on output-row blocks and their two-direction triangle attention on query-row blocks
  through the core's ``triangle_multiplication_chunked`` / ``triangle_attention_chunked`` (``CFG['trimul_impl']`` / ``CFG['triattn_impl']``
  behind the trunk wrapper's ``cfg_fn``); below ``min_n`` the fast line's kernels run (a named skip).
* ``opm_chunk`` (chunk; measured; ``rows`` 256, ``min_n`` as trunk_chunk's) — the MSA module's outer-product mean on pair-row blocks through
  ``chunk_rows`` (``BigOPM``: the trace's depth loop inside each block); the [N, N, 512] product and its fp32 LayerNorm plane never exist
  whole; below ``min_n`` the carried forward runs.
* ``nograph`` (allocator; bitwise) — the hoisted denoiser step runs un-graphed (``graphed=False`` into the stacks' own argument,
  ``chai1_eager.stack.build_parts`` / ``chai1_fastln.stackx.build_lever_parts``): no CUDA-graph private pool, no static input / output copies;
  a graph replay runs the kernels the eager step runs, in the same order (the stack's own DET form is the un-graphed one).

Routes. ``chai1-opt pred --mode big`` (driver.py): ``apply_line`` runs before the driver's bytes import torch, ``stack.apply_eager`` reads ``graphed_override()`` and calls ``on_adopted(parts, S)`` after ``hoist.adopt`` so the
trunk / denoiser levers land on the stack's own instances (re-classed in place, the ``hoist.adopt`` pattern); ``enable("big")`` (stack.py)
does the same in a program of its own. Units of the census are items: a new ``token_pair_trunk_initial_repr`` object at the trunk call
(upstream builds one per item and passes it at every recycle) opens ``item<k>``; each lever marks itself once per unit from its levered path,
a degraded path is a named ``fallback`` (the unit is partial), a site with nothing to do is a named ``skip``; ``exit_gate(rc)`` at the driver's
exit tally turns a refused line lever or a partial unit into exit 3 unless ``--allow-partial`` (recorded). The ACTIVE line carries
``big=<line>:<levers applied>`` after ``hoist=`` (report.py).
Refusals by name, never silence: the memory-mode library absent from the core → ``reason=producer_missing:opt_core.mem``; a hook the stack did not
provide → the lever's ``hooks.<key>``; the eager stack not installed by the first fold → every lever ``absent`` on the unit → exit 3.
"""
from __future__ import annotations

import os
import sys
import weakref
from typing import Optional

PREFIX = "CHAI1"                                          # the package's variable stem (opt_core.mem Ctx.prefix; GATE_WORDS)
TAG = "chai1-opt"
BASE = "fast"                                             # the rule: fast where the kit has it (opt_core.mem.compose.base_for)
LINE_NAME = "lean"                                        # the line's name on the ACTIVE line (big=lean:<levers>)
LINE = ("msa_rows", "msa_chunk", "trunk_chunk", "opm_chunk", "nograph")   # + the allocator policy, served by the kit's `alloc` lever (modes.KIT_MODES["big"].implied_optin)
MSA_CHUNK_ROWS = 1024
MSA_CHUNK_MIN_N = 0                                       # below this pair extent the carried forward runs (big line; exact / fast: modes.MEMORY_GATE_N; CHAI1_BIG_MSA_CHUNK_MIN_N); 0 = every size (big's line)
SURFACE = ('apply_line', 'graphed_override', '_resolve_memclass', '_device_total_gib', '_gate_hoist2', '_slice_msa_rows', '_check_msa_chunk', '_install_nograph', '_gate_nograph', '_check_nograph', '_check_opm_chunk', '_check_trunk_chunk', 'GATE_TABLE', 'config_gates')   # tests/test_lever_surfaces.py
NOGRAPH_MIN_N = 0                                         # below this pair extent the step keeps the stack's own graphed rule (CHAI1_BIG_NOGRAPH_MIN_N); 0 = every size
GATE_LINE_NAME = "gate"                                   # the gated line's name on the ACTIVE line: big=gate<min_n>:<levers> (modes.KitMode.memory_gated)
TRUNK_CHUNK_ROWS = 256                                    # output-row block of the chunked triangle multiplication
TRUNK_CHUNK_MIN_N = 1536                                  # below this pair extent the fast line's own kernel runs
OPM_CHUNK_ROWS = 256                                      # pair-row block of the chunked outer-product mean
OPM_CHUNK_MIN_N = 1536                                    # below this pair extent the carried forward runs
GATE_TABLE = {                                            # per --config (MODEL_OPT_TARGET_GPU, configs/<card>.env): the pair extent at or above which big's two pairformer chunk
    "h100": {"trunk_chunk": 2048, "opm_chunk": 2048, "hoist2_max_n": 0},      # levers apply, and the extent at or above which the row's hoist2 denoiser stands down to the base
    "a100": {"trunk_chunk": 1536, "opm_chunk": 1536, "hoist2_max_n": 1536},   # hoister (0 = never). h100 (the 80 GB class): fast + msa_rows holds crop 1536 unchunked — the chunks
}                                                         # from 2048 only; a100 serves the 40 GB card too: the chunks from 1536 and hoist2's
DEFAULT_GATES = "a100"                                    # per-item tables stood down there. No / unknown target: the 40 GB-safe table.
MEMCLASS_ROW = {"80g": "h100", "40g": "a100"}             # the device's memory class -> the GATE_TABLE row big's composed line uses: read ONCE, lazily, at adoption
MEMCLASS_MIN_GIB = 60                                     # (the stack is built, CUDA initialised; never at import / activation): total device memory >= 60 GiB = the
                                                          # 80 GB class (an A100-SXM4-80GB takes the h100 row: chunks from 2048, hoist2 never stands down), else the 40 GB
                                                          # class; the --config word is the prior (dry runs, no device), and every CHAI1_BIG_* word set by the caller wins
GATE_WORDS = {f"{PREFIX}_BIG_MSA_CHUNK_MIN_N": "msa_chunk", f"{PREFIX}_BIG_NOGRAPH_MIN_N": "nograph",
              f"{PREFIX}_BIG_TRUNK_CHUNK_MIN_N": "trunk_chunk", f"{PREFIX}_BIG_OPM_CHUNK_MIN_N": "opm_chunk", f"{PREFIX}_BIG_HOIST2_MAX_N": "hoist2"}


def config_gates(environ=None):
    """(config name, its GATE_TABLE row) for ``MODEL_OPT_TARGET_GPU`` (run.sh --config exports configs/<card>.env); unknown / unset -> DEFAULT_GATES."""
    environ = os.environ if environ is None else environ
    t = str(environ.get("MODEL_OPT_TARGET_GPU") or "").strip().lower()
    name = t if t in GATE_TABLE else DEFAULT_GATES
    return name, dict(GATE_TABLE[name])   # the --config's prior; GATE_WORDS (the ONLY CHAI1_BIG_* words this package reads; stack.DECLARED_BIG_ENV restates them) and the device's memory class may move them


def gate_words(environ) -> dict:
    """``{lever: "<value>"}`` for the documented size-gate words present in ``environ`` (``GATE_WORDS``: ``CHAI1_BIG_<LEVER>_MIN_N`` —
    the pair extent at or above which that lever applies; the core casts the value and refuses a non-integer
    by name, ``settings.<lever>.min_n`` — and ``CHAI1_BIG_HOIST2_MAX_N``, under the key ``hoist2``). Any other ``CHAI1_BIG_*`` name is a ValueError naming it: a mode's memory levers are fixed —
    nothing switches one off or on and no other setting is read."""
    head = f"{PREFIX}_BIG_"
    unknown = sorted(k for k in environ if str(k).startswith(head) and k not in GATE_WORDS)
    if unknown:
        raise ValueError(f"{', '.join(unknown)}: not a variable this package reads — a mode's memory levers are fixed; the only {head}* words are "
                         f"{', '.join(GATE_WORDS)} (the size gates)")
    return {GATE_WORDS[k]: str(environ[k]).strip() for k in GATE_WORDS if k in environ}
ITEM_ANCHOR = "token_pair_trunk_initial_repr"             # one object per item, passed at every recycle (chai1.py run_folding_on_context)
GRAPH_SITE = "hoisted_denoiser_step"                      # the one capture site of the fast line (chai1_eager/hoist.py step_graphed)
NOT_PRESENT = "reason=producer_missing:opt_core.mem"          # the shared core's memory-mode library (registry / record / compose / chunk / ckpt) is the pinned core's

_STATE = {"record": None, "ctx": None, "line": None, "levers": {}, "installers": [], "graphed": {"value": None}, "unit": None, "n_units": 0,
          "anchor": None, "marked": set(), "site_skipped": set(), "registered": False, "adopted": False, "chunk_entries": 0,
          "line_name": LINE_NAME, "min_n": {}, "gated": False, "unit_n": None, "gate_census": None, "dw": None, "graphed_rule": None, "forced_nograph": False,
          "memclass": None, "memclass_gib": None, "memclass_row": None, "min_n_override": {}, "explicit_gates": set(), "is_line": False, "hoist2_in_row": False}


def reset_for_tests() -> None:
    _STATE.update(record=None, ctx=None, line=None, levers={}, installers=[], graphed={"value": None}, unit=None, n_units=0, anchor=None,
                  marked=set(), site_skipped=set(), adopted=False, chunk_entries=0, line_name=LINE_NAME, min_n={}, gated=False, unit_n=None, gate_census=None,
                  dw=None, graphed_rule=None, forced_nograph=False)
    _STATE.update(memclass=None, memclass_gib=None, memclass_row=None, min_n_override={}, explicit_gates=set(), is_line=False, hoist2_in_row=False)


# ------------------------------------------------------------------------------------------------------------ the library
def mem():
    """``opt_core.mem`` (the core's memory-mode library), or ImportError naming the refusal (``NOT_PRESENT``)."""
    from . import _core
    _core.ensure_importable()
    try:
        from opt_core import mem as _mem
        from opt_core.mem import compose, record, registry          # noqa: F401 — the memory-mode API's modules (a core without them is refused by name)
        _mem.Ctx, _mem.apply, _mem.compose_big                    # noqa: B018 — the names this module calls on the package
    except (ImportError, AttributeError) as e:
        raise ImportError(f"{NOT_PRESENT} ({e})") from None
    return _mem


def refusal() -> Optional[str]:
    """Why the mode cannot run on this install (None = it can): the library missing."""
    try:
        mem()
    except ImportError as e:
        return str(e)
    return None


def graphed_override() -> Optional[bool]:
    """``False`` while ``nograph`` is applied in this process (stack.apply_eager reads it), else None (the stack's own rule)."""
    g = _STATE["graphed"]
    return g.get("value") if isinstance(g, dict) else None


def record():
    return _STATE["record"]


def active() -> bool:
    r = _STATE["record"]
    return r is not None and not r.refused


def mode_field() -> Optional[str]:
    """``<line>:<lever,…>`` for the ACTIVE line (report.py), None when the mode is not applied."""
    r = _STATE["record"]
    if r is None:
        return None
    return f"{_STATE['line_name']}:" + (",".join(r.levers) or "none")


# ------------------------------------------------------------------------------------------------------------ the levers
def _register() -> None:
    """Declare this engine's levers in the core's registry (once per process; the core's own levers register at ``mem.apply``)."""
    if _STATE["registered"]:
        return
    M = mem()
    R = M.registry
    Applied, refuse, register, RefusalError = R.Applied, R.refuse, R.register, R.RefusalError

    def _hooks(ctx, lever):
        return ctx.require(lever, "installers")

    def _needs(lever, *keys):
        """The precondition check of a lever: every named hook present (ctx.require raises the refusal by name)."""
        def applies(ctx):
            ctx.require(lever, *keys)
            return None
        return applies

    @register("msa_rows", family="chunk", exact="measured",
              exact_reason="the dropped MSA rows are all-masked, but dropping them changes the reduction extent of the MSA-module GEMMs and with it the "
                           "kernel's accumulation order: bitwise vs fast under the DET recipe in the SINGLE-SEQUENCE form (smoke, 25/25) and at a real MSA of "
                           "<= 904 rows (ladder 2AH1, 25/25), NOT at 5333 rows (ladder 1HXH, 0/25, PAE max|d| 1.45; msa_rows off alone restores 25/25) — "
                           "input-form-specific (B43); stock's seed-to-seed band decides the word",
              applies=_needs("msa_rows", "installers"), preconditions=("hooks.installers",),
              description="the trunk's MSA module on the rows that carry any mask (S_eff) instead of the 16384 padded rows")
    def msa_rows(ctx) -> Applied:
        h = _hooks(ctx, "msa_rows")
        h["installers"].append(("msa_rows", _install_msa_rows))
        return Applied(lever="msa_rows", settings={"anchor": ITEM_ANCHOR},
                       sites=("EagerTrunkWrapper.forward: msa_input_feats/msa_mask sliced to S_eff before the stack's move_to_device",))

    @register("msa_chunk", family="chunk", exact="measured",
              exact_reason="the pair-weighted averaging is row-local along S (the core labels the block form bitwise): bitwise vs fast under the DET recipe "
                           "in the single-sequence form (smoke, 25/25) AND at a real MSA of 5333 rows (ladder 1HXH with msa_rows off, 25/25, det 1) — the "
                           "label stays measured until stock's seed-to-seed band words it (det 0 is the shipped regime)",
              applies=_needs("msa_chunk", "installers"), preconditions=("hooks.installers", "chunk.chunk_rows"), settings=("rows", "min_n"),
              description="the MSA pair-weighted averaging on rows-row blocks through opt_core.mem.chunk.chunk_rows (one output plane) at pair "
                          "extent N >= min_n; below min_n the carried forward runs (a named skip per item)")
    def msa_chunk(ctx) -> Applied:
        h = _hooks(ctx, "msa_chunk")
        rows = ctx.setting("msa_chunk", "rows", MSA_CHUNK_ROWS, cast=int)
        if rows < 1:
            raise RefusalError(refuse("msa_chunk", "setting.rows", f"rows={rows} is not >= 1"))
        min_n = ctx.setting("msa_chunk", "min_n", MSA_CHUNK_MIN_N, cast=int)
        _STATE["min_n"]["msa_chunk"] = min_n
        try:
            from opt_core.mem import chunk as C                                  # the core's chunker (torch at its import: the box's)
            C.chunk_rows
        except (ImportError, AttributeError) as e:
            raise RefusalError(refuse("msa_chunk", "chunk.chunk_rows", f"opt_core.mem.chunk.chunk_rows not importable: {e}"))
        h["installers"].append(("msa_chunk", lambda parts, S, rec: _install_msa_chunk(parts, S, rec, rows, min_n)))
        return Applied(lever="msa_chunk", settings={"rows": rows, "min_n": min_n, "via": "opt_core.mem.chunk.chunk_rows(site=msa_rows)"},
                       sites=("chai1_eager.trunk.MSAPairWeightedAveraging.forward (instances re-classed onto chunk_rows)",))

    @register("trunk_chunk", family="chunk", exact="measured",
              exact_reason="MEASURED in situ (s11-s13, crop 2048, det 1: the line with this lever on vs off differs on 25/25 output files "
                           "while each form is run-to-run bitwise); the op-level H100 measurement under the kit's DET recipe locates it in "
                           "the ATTENTION half: the row-block SDPA vs the fast line's provider kernel differ under use_deterministic_algorithms "
                           "(max|d| 96 at N=2048; bitwise at det 0), while the row-block TriMul is bitwise vs trimul_bmm under both regimes; "
                           "stock's seed-to-seed band decides the verdict word",
              applies=_needs("trunk_chunk", "installers"), preconditions=("hooks.installers", "chunk.triangle_multiplication_chunked"),
              settings=("rows", "min_n"),
              description="the pair and template stacks' triangle ops on row blocks: the merged (outgoing+incoming) triangle "
                          "multiplication as two core triangle_multiplication_chunked calls (CFG['trimul_impl']) and the two-direction "
                          "triangle attention as two core triangle_attention_chunked calls (CFG['triattn_impl'])")
    def trunk_chunk(ctx) -> Applied:
        h = _hooks(ctx, "trunk_chunk")
        rows = ctx.setting("trunk_chunk", "rows", TRUNK_CHUNK_ROWS, cast=int)
        min_n = ctx.setting("trunk_chunk", "min_n", TRUNK_CHUNK_MIN_N, cast=int)
        _STATE["min_n"]["trunk_chunk"] = min_n
        if rows < 1:
            raise RefusalError(refuse("trunk_chunk", "setting.rows", f"rows={rows} is not >= 1"))
        try:
            from opt_core.mem import chunk as C
            C.triangle_multiplication_chunked, C.TriMulParts
        except (ImportError, AttributeError) as e:
            raise RefusalError(refuse("trunk_chunk", "chunk.triangle_multiplication_chunked", f"not importable: {e}"))
        h["installers"].append(("trunk_chunk", lambda parts, S, rec: _install_trunk_chunk(parts, S, rec, rows, min_n)))
        return Applied(lever="trunk_chunk", settings={"rows": rows, "min_n": min_n,
                                                      "via": "opt_core.mem.chunk.triangle_multiplication_chunked(mode=rows) x2 + triangle_attention_chunked x2"},
                       sites=("chai1_eager.trunk.CFG['trimul_impl'] / CFG['triattn_impl'] through EagerTrunkWrapper.cfg_fn (the fast line's kernels kept below min_n)",))

    @register("opm_chunk", family="chunk", exact="measured",
              exact_reason="the outer-product mean on pair-row blocks (every (i, j) entry is an independent sum over depth); op-level bitwise vs "
                           "the eager OPM on H100 at N=2048 under det 0 AND the DET recipe (op-level); IN SITU (s12, det 1, crop 2048) the "
                           "bisection is inconsistent — neutral beside trunk_chunk (tc_only == v6 bitwise) but opm_chunk alone vs chunks-off "
                           "differs 25/25 (max|d| 6.5) with no opm-only repeat yet — MEASURED until the repeat runs; stock's band decides",
              applies=_needs("opm_chunk", "installers"), preconditions=("hooks.installers", "chunk.chunk_rows"), settings=("rows", "min_n"),
              description="the MSA module's outer-product mean on pair-row blocks through opt_core.mem.chunk.chunk_rows (the trace's depth "
                          "loop inside each block): the [N, N, g·e·e] product and its fp32 LayerNorm plane never exist whole")
    def opm_chunk(ctx) -> Applied:
        h = _hooks(ctx, "opm_chunk")
        rows = ctx.setting("opm_chunk", "rows", OPM_CHUNK_ROWS, cast=int)
        min_n = ctx.setting("opm_chunk", "min_n", OPM_CHUNK_MIN_N, cast=int)
        _STATE["min_n"]["opm_chunk"] = min_n
        if rows < 1:
            raise RefusalError(refuse("opm_chunk", "setting.rows", f"rows={rows} is not >= 1"))
        try:
            from opt_core.mem import chunk as C
            C.chunk_rows
        except (ImportError, AttributeError) as e:
            raise RefusalError(refuse("opm_chunk", "chunk.chunk_rows", f"opt_core.mem.chunk.chunk_rows not importable: {e}"))
        h["installers"].append(("opm_chunk", lambda parts, S, rec: _install_opm_chunk(parts, S, rec, rows, min_n)))
        return Applied(lever="opm_chunk", settings={"rows": rows, "min_n": min_n, "via": "opt_core.mem.chunk.chunk_rows(site=outer_product_mean)"},
                       sites=("chai1_eager.trunk.OuterProductMean instances re-classed (BigOPM) onto chunk_rows over the pair-row axis",))

    @register("nograph", family="allocator", exact="bitwise",
              exact_reason="a CUDA-graph replay runs the kernels the eager step runs, in the same order; the stack's DET form is un-graphed",
              applies=_needs("nograph", "graphed", "installers"), preconditions=("hooks.graphed", "ckpt.graph_capture"), settings=("min_n",),
              description="the hoisted denoiser step un-graphed through the core's graph_capture policy (off): no private pool, no static copies; "
                          "min_n 0 = for the process (the stack builds un-graphed), min_n > 0 = per item at pair extent N >= min_n (the adopted "
                          "wrapper's own `graphed` switch, set at the item's trunk call; below min_n the stack's own rule — a named skip)")
    def nograph(ctx) -> Applied:
        h = ctx.require("nograph", "graphed", "installers")
        try:
            from opt_core.mem import ckpt                                     # the core's typed capture policy (DRY: the policy object is the core's)
            pol = ckpt.GraphPolicy(policy="off")                             # no record attached: its decisions are marked per item by _check_graphed
        except (ImportError, AttributeError) as e:
            raise RefusalError(refuse("nograph", "ckpt.graph_capture", f"opt_core.mem.ckpt.GraphPolicy absent: {e}"))
        min_n = ctx.setting("nograph", "min_n", NOGRAPH_MIN_N, cast=int)
        _STATE["min_n"]["nograph"] = min_n
        if bool(pol.admit(GRAPH_SITE, 0)) is not False:
            raise RefusalError(refuse("nograph", "ckpt.graph_capture", f"the policy left graphed={pol.admit(GRAPH_SITE, 0)!r}"))
        h["graphed"]["policy"] = pol
        if min_n <= 0:
            h["graphed"]["value"] = False                                     # the engine switch the policy drives: the stack's own `graphed` argument, for the process
        h["installers"].append(("nograph", lambda parts, S, rec: _install_nograph(parts, S, rec, min_n)))

        def undo():
            h["graphed"]["value"] = None
        return Applied(lever="nograph", settings={"graphed": (False if min_n <= 0 else f"False per item at N >= {min_n}"), "min_n": min_n,
                                                  "via": "opt_core.mem.ckpt.GraphPolicy", "policy": pol.describe()},
                       sites=("chai1_opt.stack.apply_eager: graphed" if min_n <= 0 else "HoistedDiffusionWrapper.graphed (per item, at the trunk call)",
                              "HoistedDiffusionWrapper.forward (marks the census)"), undo=undo)


    _STATE["registered"] = True


# ------------------------------------------------------------------------------------------------------------ apply
def apply_line(rep: Optional[dict] = None, *, out_dir: Optional[str] = None, environ=None, strict: bool = True, mode: str = "big",
               allow_partial: bool = False, opt_out: Optional[str] = None):
    """Compose the mode's memory levers and apply them through ``opt_core.mem.apply`` (every step recorded). ``big``: the line
    (``LINE``, composed on fast: modes.KIT_MODES['big'].memory), every lever at every size but the pairformer's two (min_n from the --config's ``GATE_TABLE`` row, then the device's memory class).
    ``exact`` / ``fast``: the row's GATED levers (modes.KitMode.memory_gated = msa_chunk + nograph) with ``min_n`` = the row's
    ``memory_gate`` (2048, chai1's top crop): below that pair extent every item runs the mode's own path (the levers stand down, a named
    skip per item), at or above it the pair-weighted averaging runs in row blocks and the hoisted denoiser step un-graphed — each item's
    decision printed (the MEMORY line) and counted (``gate_census``, the exit's MEMORY-GATE line). Returns the
    AppliedRecord; under ``strict`` a refused lever raises ``opt_core.mem.Refused`` (the caller prints the NOT ACTIVE line and exits 3);
    a ``CHAI1_BIG_*`` word other than the size gates (``GATE_WORDS``) raises ValueError by name. ``allow_partial`` is the route's opt-out of a partial
    ITEM as given (the driver's ``--allow-partial``), ``opt_out`` the word the core's PARTIAL lines name for it (default ``--allow-partial``;
    the CHAI1_OPT route passes ``CHAI1_OPT_ALLOW_PARTIAL=1``). ``rep`` (the activation report) gains ``big`` = the record's fields;
    ``out_dir`` rides the context's extras (the run directory)."""
    from . import modes
    M = mem()
    _register()
    if _STATE["record"] is not None:
        return _STATE["record"]
    km = modes.KIT_MODES.get(mode)
    if km is None or not (km.memory or km.memory_gated or getattr(km, "memory_free", ())):
        raise ValueError(f"mode {mode} carries no memory levers (modes.KIT_MODES[<mode>].memory / .memory_gated)")
    environ = environ if environ is not None else os.environ
    gates = gate_words(environ)                                                     # the size-gate words, if set; any other CHAI1_BIG_* name refused by name here
    _STATE.update(explicit_gates=set(gates), is_line=bool(km.memory), hoist2_in_row=bool(km.memory) and "hoist2" in tuple(km.dstep), memclass=None, min_n_override={})
    cfg_name, cfg = config_gates(environ)                                           # the --config's gate table (GATE_TABLE): the pairformer chunk levers' min_n, hoist2's stand-down extent
    hoist2_max_n = gates.pop("hoist2", cfg["hoist2_max_n"])
    if km.memory:                                                                   # big: the composed line, fast by reference — every lever, always
        table, levers, line_name, base = _line(M, modes), LINE, LINE_NAME, BASE
        settings = {"msa_chunk": {"rows": MSA_CHUNK_ROWS}, "trunk_chunk": {"min_n": cfg["trunk_chunk"]}, "opm_chunk": {"min_n": cfg["opm_chunk"]}}
        graphs = False                                                              # nograph is in the line: the hoisted step runs un-graphed
    elif km.memory_gated or getattr(km, "memory_free", ()):                         # exact / fast: the row's size-free memory levers (fast: msa_rows) + its gated levers
        gate = int(km.memory_gate)
        free = tuple(getattr(km, "memory_free", ()))
        table = levers = free + tuple(km.memory_gated); line_name, base = f"{GATE_LINE_NAME}{gate}", mode
        hoist2_max_n = gates.pop("hoist2", 0) if "hoist2" in gates else 0            # the rows' own denoiser never stands down (big's 40 GB table only)
        settings = {lv: {"min_n": gate} for lv in km.memory_gated}
        if "msa_chunk" in settings:
            settings["msa_chunk"]["rows"] = MSA_CHUNK_ROWS
        graphs = True                                                               # the mode's own graphed rule stays in force below the gate
    for lever, value in gates.items():                                              # CHAI1_BIG_<LEVER>_MIN_N moves that lever's gate (the core casts it: a non-integer refuses by name)
        settings.setdefault(lever, {})["min_n"] = value
    graphed = {"value": None}
    installers: list = []
    hooks = {lv: {"installers": installers, "graphed": graphed} for lv in levers}
    ctx = M.Ctx(prefix=PREFIX, tag=TAG, framework="torch", hooks=hooks, settings=settings, environ=environ, graphs=graphs,
                opt_out=opt_out or M.OPT_OUT,                                       # the word the PARTIAL lines name: --allow-partial (driver) | CHAI1_OPT_ALLOW_PARTIAL=1 (the CHAI1_OPT route)
                extra={"line": line_name, "base": base, "kit_mode": _kit_mode_fields(modes, mode), "out_dir": out_dir})
    try:
        hoist2_max_n = int(hoist2_max_n)
    except (TypeError, ValueError):
        raise ValueError(f"{PREFIX}_BIG_HOIST2_MAX_N={hoist2_max_n!r}: an integer pair extent (0 = hoist2 never stands down)") from None
    _STATE.update(config=cfg_name, hoist2_max_n=hoist2_max_n if (km.memory and "hoist2" in tuple(km.dstep)) else 0, hoist2_item=None, hoist2_down=0)
    _STATE.update(ctx=ctx, line=table, installers=installers, graphed=graphed, line_name=line_name, gated=bool(km.memory_gated) and not km.memory,
                  gate_census=({"units": 0, "applied": {lv: 0 for lv in levers}, "stood_down": {lv: 0 for lv in levers}} if km.memory_gated and not km.memory else None))
    try:
        rec = (M.apply(table, ctx, strict=strict, allow_partial=bool(allow_partial)) if km.memory else       # the levers as composed: no switches
               M.apply(table, ctx, base=base, strict=strict, allow_partial=bool(allow_partial)))
    except M.Refused as e:
        _STATE["record"] = e.record
        _fill(rep, e.record)
        raise
    for lv in (getattr(km, "memory_free", ()) if not km.memory else ()):
        _STATE["min_n"].setdefault(lv, 0)                                          # size-free levers of a gated row apply at every pair extent (the MEMORY line names them applied)
    _STATE["record"] = rec
    _fill(rep, rec)
    return rec


def dry_run_fields(mode: str) -> dict:
    """The ``big`` field of a DRY-RUN report: the line's name and levers the mode would apply (nothing is applied)."""
    from . import modes
    km = modes.KIT_MODES[mode]
    if km.memory:
        name, cfg = config_gates()
        return {"line": LINE_NAME, "levers": list(LINE), "gates": {"config": name, **cfg}}
    return {"line": f"{GATE_LINE_NAME}{int(km.memory_gate)}", "levers": list(getattr(km, "memory_free", ())) + list(km.memory_gated),
            "gates": {lv: int(km.memory_gate) for lv in km.memory_gated}}


def _line(M, modes):
    from opt_core import modes as core_modes
    table = core_modes.ModeTable(tuple(m for m in modes.MODES if m != "big"), modes.DEFAULT_MODE)
    _, line = M.compose_big(table, base=BASE, levers=LINE)
    return line


def _kit_mode_fields(modes, mode: str = "big") -> dict:
    km = modes.KIT_MODES[mode]
    return {"name": km.name, "levels": list(km.levels), "eager": km.eager, "dstep": list(km.dstep), "stack": km.stack}


def _fill(rep: Optional[dict], rec) -> None:
    if rep is None:
        return
    rep["big"] = {"line": _STATE["line_name"], "levers": list(rec.levers), "refused": [str(r) for r in rec.refused], "exact": rec.exact,
                    "exact_per_lever": dict(rec.exact_per_lever), "off_by_flag": list(rec.off_by_flag), "on_by_flag": list(rec.on_by_flag),
                    "allocator": dict(rec.allocator_settings)}
    if _STATE["gated"]:
        rep["big"]["gates"] = {lv: _STATE["min_n"].get(lv) for lv in rec.levers if lv in _STATE["min_n"]}
    else:
        rep["big"]["gates"] = {"config": _STATE.get("config"), "trunk_chunk": _STATE["min_n"].get("trunk_chunk"), "opm_chunk": _STATE["min_n"].get("opm_chunk"),
                                 "hoist2_max_n": _STATE.get("hoist2_max_n", 0), "memclass": _STATE.get("memclass") or "prior"}   # per lever: the pair extent at or above which it applies (modes.MEMORY_GATE_N, else CHAI1_BIG_<LEVER>_MIN_N)


CHUNK_SITE_LEVERS = ("msa_chunk", "opm_chunk", "trunk_chunk")            # the levers whose SITES chunk a plane per item (EXIT census below)


def _skip_word(reason: str) -> str:
    """One word for a chunk site's named skip: ``narrowed_extent`` (the site met an extent below its gate after trunk_n narrowed the inputs),
    ``below_gate`` (the item's pair extent is below the lever's min_n), else ``named``."""
    r = str(reason or "")
    if r.startswith("extent "):
        return "narrowed_extent"
    if r.startswith("gated:") or "< min_n" in r:
        return "below_gate"
    return "named"


def chunk_census_fields() -> list:
    """EXIT tokens ``msa_chunk=`` ``opm_chunk=`` ``trunk_chunk=`` for the chunk levers of the applied line: ``<items served>/<items>`` when the
    chunked site ran on at least one item (or nothing was skipped), else ``skipped:<word>`` (_skip_word of the last named skip) — chunk
    engagement versus a named stand-down, evidenced on the one exit line. Empty when big applied no line in this process."""
    rec = _STATE.get("record")
    if rec is None:
        return []
    applied = tuple(getattr(rec, "levers", ()) or ())
    units = getattr(rec, "units", {}) or {}
    n_items = int(_STATE.get("n_units") or 0)
    out = []
    for lv in CHUNK_SITE_LEVERS:
        if lv not in applied:
            continue
        served = sum(1 for u in units.values() if lv in (getattr(u, "ran", ()) or ()))      # items whose chunked site RAN (the record's own mark)
        reasons = [getattr(u, "skipped", {}).get(lv) for u in units.values() if lv in (getattr(u, "skipped", {}) or {})]
        if served or not reasons:
            out.append(f"{lv}={served}/{n_items}")
        else:
            out.append(f"{lv}=skipped:{_skip_word(reasons[-1])}")
    return out


def fill_exit(rep: Optional[dict]) -> None:
    """At exit: the gated line's per-item census into the activation report's ``big`` block, beside the printed MEMORY-GATE line."""
    cz = gate_census()
    if rep is not None and cz is not None and isinstance(rep.get("big"), dict):
        rep["big"]["gate_census"] = cz


def fill(rep: Optional[dict]) -> None:
    """The record's fields on an activation report (``rep['big']``), when the line has been applied."""
    if _STATE["record"] is not None:
        _fill(rep, _STATE["record"])


def not_active_line() -> str:
    r = _STATE["record"]
    return r.not_active_line(TAG) if r is not None else f"[{TAG}] NOT ACTIVE: big refused — no record"


def _device_total_gib():
    """Total memory of the current CUDA device in GiB, or None when CUDA is not initialised in this process (never initialises it)."""
    try:
        import torch
    except ImportError:
        return None
    cuda = getattr(torch, "cuda", None)
    ok = cuda is not None and all(hasattr(cuda, n) for n in ("is_available", "is_initialized", "get_device_properties", "current_device"))
    if not ok or not (cuda.is_available() and cuda.is_initialized()):
        return None
    return cuda.get_device_properties(cuda.current_device()).total_memory / float(1 << 30)


def _resolve_memclass() -> None:
    """Once per process, at adoption (the stack is built on the device): big's composed line takes its size gates from the GATE_TABLE row of
    the device's MEMORY CLASS (``MEMCLASS_ROW``: >= ``MEMCLASS_MIN_GIB`` GiB -> the 80 GB row, else the 40 GB row); the --config word chose the
    prior in ``apply_line``; a gate the caller set by word (CHAI1_BIG_*_MIN_N / _HOIST2_MAX_N) is left alone. Prints one MEMORY line naming
    the class. The gated rows (exact | fast) keep their own gate (modes.MEMORY_GATE_N): nothing to resolve there."""
    if _STATE.get("memclass") is not None or not _STATE.get("is_line"):
        return
    gib = _device_total_gib()
    if gib is None:
        _STATE["memclass"] = "prior"
        return
    cls = "80g" if gib >= MEMCLASS_MIN_GIB else "40g"
    row = MEMCLASS_ROW[cls]; tbl = GATE_TABLE[row]; explicit = _STATE.get("explicit_gates") or set()
    ov = {lv: int(tbl[lv]) for lv in ("trunk_chunk", "opm_chunk") if lv not in explicit}
    _STATE["min_n_override"] = ov
    for lv, v in ov.items():
        if lv in _STATE["min_n"]:
            _STATE["min_n"][lv] = v
    if "hoist2" not in explicit and _STATE.get("hoist2_in_row"):
        _STATE["hoist2_max_n"] = int(tbl["hoist2_max_n"])
    _STATE.update(memclass=cls, memclass_gib=round(gib, 1), memclass_row=row)
    sys.stderr.write(f"\n[{TAG}] MEMORY memclass={cls} device_gib={gib:.1f} gates={row} (config prior {_STATE.get('config')}) "
                     f"trunk_chunk>={_STATE['min_n'].get('trunk_chunk')} opm_chunk>={_STATE['min_n'].get('opm_chunk')} hoist2_max_n={_STATE.get('hoist2_max_n', 0)} "
                     f"explicit={','.join(sorted(explicit)) or 'none'}\n")
    sys.stderr.flush()


# ------------------------------------------------------------------------------------------------------------ the hooks
def on_adopted(parts: dict, S) -> dict:
    """Run every applied lever's installer on the stack's own instances (after ``hoist.adopt``); returns ``{lever: site}``."""
    rec = _STATE["record"]
    if rec is None:
        return {}
    _resolve_memclass()                                                             # the device's memory class decides the composed line's size gates (lazy, once)
    done = {}
    for name, fn in list(_STATE["installers"]):
        done[name] = fn(parts, S, rec)
    if int(_STATE.get("hoist2_max_n") or 0) > 0:                                     # the config's 40 GB table: hoist2 stands down per item at pair extent >= max_n
        _reclass(parts, "trunk", _trunk_class)
        mx = int(_STATE["hoist2_max_n"])
        _STATE["levers"].setdefault("trunk", []).append(("hoist2", lambda tw, kw, rec, _mx=mx: _gate_hoist2(tw, kw, rec, _mx)))
        done["hoist2"] = f"diffusion:hoist2 stands down per item at N >= {mx} ({_STATE.get('config')} table; chai1_opt.stack.dstep_stand_down)"
    _STATE["adopted"] = True
    return done


def _gate_hoist2(tw, kw: dict, rec, max_n: int) -> dict:
    """At the item's trunk call (big's 40 GB table): the row's hoist2 denoiser stands down to the base hoister for THIS item at pair extent
    >= max_n (``chai1_opt.stack.dstep_stand_down``; base and hoist2 are bitwise identical — memory and speed move, outputs do not), and is
    restored below it.  One line per item names the decision; never silent."""
    anchor = kw.get(ITEM_ANCHOR)
    if anchor is None or _STATE.get("hoist2_item") == id(anchor):
        return kw
    _STATE["hoist2_item"] = id(anchor)
    n = _pair_extent(anchor)
    on = n is not None and n >= max_n
    from . import stack as _stack
    used = _stack.dstep_stand_down("hoist2", on)
    if used is None:
        rec.note(f"hoist2 stand-down: no adopted hoist2 denoiser (pair_extent={n})")
        return kw
    if on:
        _STATE["hoist2_down"] = int(_STATE.get("hoist2_down") or 0) + 1
    sys.stderr.write(f"\n[{TAG}] MEMORY unit={_STATE.get('unit')} pair_extent={n if n is not None else 'unknown'} "
                     f"{'standing_down=hoist2' if on else 'applied=hoist2'} denoiser_hoister={used} ({_STATE.get('memclass_row') or _STATE.get('config')}: hoist2_max_n={max_n})\n")
    sys.stderr.flush()
    return kw


def _mark_once(rec, lever: str, detail: Optional[str] = None) -> None:
    """One mark per (unit, lever); never outside a unit — a mark with no unit open would open the record's process unit, and the
    census would then read every per-item lever as absent there."""
    if _STATE["unit"] is None:
        return
    key = (_STATE["unit"], lever)
    if key in _STATE["marked"]:
        return
    _STATE["marked"].add(key)
    rec.mark(lever, detail=detail)


def _site_skip(rec, lever: str, n: int, min_n: int, site: str) -> None:
    """A chunk SITE that takes the carried branch on the open unit names it: the trunk may run NARROWED (trunk_n: N' = ceil64(live) below
    the crop the item's gate read), so a lever the gate applied at the crop can meet an extent below its ``min_n`` at the site — a named
    skip (accounted, not partial: exit 0 with the word), once per (unit, lever), never over a mark of the same unit."""
    if rec is None or _STATE["unit"] is None:
        return
    key = (_STATE["unit"], lever)
    if key in _STATE["marked"] or key in _STATE["site_skipped"]:
        return
    _STATE["site_skipped"].add(key)
    rec.skip(lever, f"extent {n} < min_n {min_n} at the site ({site}; the trunk ran narrowed below the crop)")


def _unit_touch(rec, anchor) -> bool:
    """Open a new census unit when the item anchor is a new object (the same-object rule the hoist uses, held weakly); True on a new unit."""
    prev = _STATE["anchor"]
    if prev is not None and anchor is not None and prev() is anchor:
        return False
    unit_end()
    k = _STATE["n_units"]
    name = f"item{k}"
    rec.unit_begin(name)
    _STATE.update(unit=name, n_units=k + 1, anchor=(weakref.ref(anchor) if anchor is not None and _weakrefable(anchor) else None))
    return True


def _pair_extent(anchor) -> Optional[int]:
    """The item's pair extent N (== the crop): ``token_pair_trunk_initial_repr`` is [B, N, N, c]."""
    shp = getattr(anchor, "shape", None)
    return int(shp[-3]) if shp is not None and len(shp) >= 3 else None


def _unit_gate(rec, anchor) -> None:
    """On a new unit of a GATED line (exact / fast): read the item's pair extent, decide each lever against its ``min_n``, count the
    decision (``gate_census``) and print it — ``[chai1-opt] MEMORY unit=item<k> pair_extent=<N> applied=<levers|none>
    standing_down=<levers|none> (min_n <lever>=<n> …)``. The levers themselves act on the same numbers at their own sites (the chunked
    class's size test, the trunk checks); this is the item's one printed word. Big's line (no gate) prints nothing here."""
    _STATE["unit_n"] = _pair_extent(anchor)
    cz = _STATE.get("gate_census")
    if not _STATE.get("gated") or cz is None or rec is None:
        return
    n = _STATE["unit_n"]
    applied, down = [], []
    for lv in rec.levers:
        m = int(_STATE["min_n"].get(lv) or 0)
        (applied if (n is not None and n >= m) else down).append(lv)
    cz["units"] += 1
    for lv in applied:
        cz["applied"][lv] += 1
    for lv in down:
        cz["stood_down"][lv] += 1
    gates = " ".join(f"{lv}>={_STATE['min_n'].get(lv)}" for lv in rec.levers)
    from . import alloc as _alloc
    pol = "expandable_segments" if _alloc.applied() else "none"                    # the allocator policy this process runs under (alloc.py: the row's implied lever), named per item
    sys.stderr.write(f"\n[{TAG}] MEMORY unit={_STATE['unit']} pair_extent={n if n is not None else 'unknown'} applied={','.join(applied) or 'none'} "
                     f"standing_down={','.join(down) or 'none'} alloc={pol} (min_n {gates})\n")             # a line of its own: upstream's tqdm bar holds the current one
    sys.stderr.flush()


def gate_census() -> Optional[dict]:
    """The gated line's per-item decisions so far: ``{units, applied: {lever: n}, stood_down: {lever: n}, min_n: {lever: n}}``; None for
    big's line or when nothing is applied."""
    cz = _STATE.get("gate_census")
    if cz is None:
        return None
    return {"units": cz["units"], "applied": dict(cz["applied"]), "stood_down": dict(cz["stood_down"]), "min_n": dict(_STATE["min_n"])}


def gate_line() -> Optional[str]:
    """``[chai1-opt] MEMORY-GATE units=<n> <lever>=<applied>/<n> … (min_n <lever>=<N> …)`` — the gated line's census in one line at exit."""
    cz = gate_census()
    if cz is None:
        return None
    levers = list(cz["applied"])
    per = " ".join(f"{lv}={cz['applied'][lv]}/{cz['units']}" for lv in levers)
    gates = " ".join(f"{lv}>={cz['min_n'].get(lv)}" for lv in levers)
    return f"[{TAG}] MEMORY-GATE units={cz['units']} applied {per or 'none'} (min_n {gates})"


def _weakrefable(x) -> bool:
    try:
        weakref.ref(x)
        return True
    except TypeError:
        return False


def unit_end() -> None:
    rec = _STATE["record"]
    if rec is not None and _STATE["unit"] is not None:
        _gated_standdowns(rec, _STATE["unit"])
        rec.unit_end(_STATE["unit"])
        _STATE["unit"] = None


def _gated_standdowns(rec, unit: str) -> None:
    """At unit close: a size-gated lever (``min_n`` > 0) whose gate keeps it off for this item (pair extent < ``min_n``) and that left no event
    on the unit — its site was never entered on this item (a module another lever serves, a path the crop does not take) — is a NAMED skip
    (``gated: pair extent <N> < min_n <M> (site not entered on this item)``), accounted and not partial: the exit gate reads rc 0 with the word
    instead of ``never marked``. A lever whose gate engages (N >= min_n) and that still left no event stays unexplained (partial, exit 3)."""
    n = _STATE.get("unit_n")
    if n is None:
        return
    u = (getattr(rec, "units", None) or {}).get(unit)
    seen = set()
    if u is not None:
        seen = set(getattr(u, "ran", ()) or ()) | set(getattr(u, "fallback", {}) or {}) | set(getattr(u, "skipped", {}) or {})
    for lv, m in (_STATE.get("min_n") or {}).items():
        m = int(m or 0)
        if m > 0 and n < m and lv not in seen and (_STATE["unit"], lv) not in _STATE["marked"]:
            rec.skip(lv, f"gated: pair extent {n} < min_n {m} (site not entered on this item)", unit=unit)
            _STATE["marked"].add((unit, lv))


def _trunk_class(base):
    """``BigTrunk`` over the adopted trunk wrapper's class: the item boundary, the MSA-row slice, the msa_chunk check."""
    class BigTrunk(base):
        def forward(self, crop_size, *, return_on_cpu=False, move_to_device=None, **kw):
            rec = _STATE["record"]
            if _unit_touch(rec, kw.get(ITEM_ANCHOR)):
                _unit_gate(rec, kw.get(ITEM_ANCHOR))
            for name, fn in _STATE["levers"].get("trunk", []):
                kw = fn(self, kw, rec)
            return base.forward(self, crop_size, return_on_cpu=return_on_cpu, move_to_device=move_to_device, **kw)
    BigTrunk.__name__ = BigTrunk.__qualname__ = "BigTrunk"
    return BigTrunk


def _denoiser_class(base):
    class BigDenoiser(base):
        def forward(self, crop_size, *, return_on_cpu=False, move_to_device=None, **kw):
            rec = _STATE["record"]
            for name, fn in _STATE["levers"].get("denoiser", []):
                fn(self, rec)
            return base.forward(self, crop_size, return_on_cpu=return_on_cpu, move_to_device=move_to_device, **kw)
    BigDenoiser.__name__ = BigDenoiser.__qualname__ = "BigDenoiser"
    return BigDenoiser


def _reclass(parts: dict, which: str, maker) -> object:
    obj = parts.get(which)
    if obj is None:
        raise KeyError(f"the installed parts carry no {which!r} (keys: {sorted(parts)})")
    if type(obj).__name__ not in ("BigTrunk", "BigDenoiser"):
        obj.__class__ = maker(type(obj))
    return obj


def _install_msa_rows(parts, S, rec):
    _reclass(parts, "trunk", _trunk_class)
    _STATE["levers"].setdefault("trunk", []).append(("msa_rows", _slice_msa_rows))
    return "trunk:msa_rows"


def _slice_msa_rows(tw, kw: dict, rec) -> dict:
    feats, mask = kw.get("msa_input_feats"), kw.get("msa_mask")
    if feats is None or mask is None or getattr(feats, "ndim", 0) != 4 or getattr(mask, "ndim", 0) != 3 or tuple(feats.shape[:3]) != tuple(mask.shape):
        rec.fallback("msa_rows", f"unexpected MSA shapes: feats={getattr(feats, 'shape', None)} mask={getattr(mask, 'shape', None)}; the trunk ran on every row")
        return kw
    S = int(mask.shape[1])
    rows = mask.any(-1).any(0)                                        # [S]: the rows with any True over the batch
    idx = rows.nonzero()
    s_eff = int(idx.max().item()) + 1 if idx.numel() else 1
    if s_eff >= S:
        rec.skip("msa_rows", f"no all-masked rows to drop (S={S}, S_eff={s_eff})")
        return kw
    kw = dict(kw)
    kw["msa_input_feats"] = feats[:, :s_eff]
    kw["msa_mask"] = mask[:, :s_eff]
    _mark_once(rec, "msa_rows", detail=f"S={S}->{s_eff}")
    return kw


def _pwa_modules(tw):
    """The installed eager trunk's MSAPairWeightedAveraging instances (``Trunk.msa_module.msa_pair_weighted_averaging``), or None."""
    trunk = getattr(tw, "trunk", None)
    msa = getattr(trunk, "msa_module", None)
    if msa is None:
        return None
    return list(getattr(msa, "msa_pair_weighted_averaging", []))


def _pwa_class(base, rows: int, min_n: int = 0):
    """The pair-weighted averaging re-classed onto the core's row chunker: ``chunk_rows`` runs the carried forward on ``rows``-row blocks
    of the MSA (the mask block by the same offsets) and assembles into ONE output plane — the carried forward's own 8192-row loop with
    its list of outputs and the final ``cat`` never runs above ``rows``. Below pair extent ``min_n`` the carried forward runs as it is."""
    class BigPWA(base):
        def forward(self, msa, z, pair_mask, msa_mask):
            if min_n > 0 and int(msa.shape[-2]) < min_n:               # below the gate: the carried forward, unchanged (msa is [B, S, N, c]: N the pair extent)
                _site_skip(_STATE["record"], "msa_chunk", int(msa.shape[-2]), min_n, "pair_weighted_averaging")
                return base.forward(self, msa, z, pair_mask, msa_mask)
            from opt_core.mem import chunk as C
            rec = _STATE["record"]

            def block(blk, i0, i1):
                return base.forward(self, blk, z, pair_mask, msa_mask[:, i0:i1])

            def emit(lever, site, exact, reason, **entry):        # the core's per-call sink: (lever, site, exact, reason, **details)
                _STATE["chunk_entries"] += 1
                _mark_once(rec, "msa_chunk", detail=f"rows={rows} S={entry.get('rows')} n_chunks={entry.get('n_chunks')} (core chunk_rows, {exact})")
            return C.chunk_rows(block, msa, 1, rows, exact="bitwise", record=emit, with_offsets=True, lever="msa_chunk", site="msa_rows",
                                reason="the pair-weighted average is row-local along S: the weights come from the pair and are shared by every row")
    BigPWA.__name__ = BigPWA.__qualname__ = "BigPWA"
    return BigPWA


def _install_msa_chunk(parts, S, rec, rows: int, min_n: int = 0):
    tw = _reclass(parts, "trunk", _trunk_class)
    mods = _pwa_modules(tw)
    if not mods:
        raise LookupError("msa_chunk: the installed trunk has no msa_module.msa_pair_weighted_averaging")
    for m in mods:
        if type(m).__name__ != "BigPWA":
            m.__class__ = _pwa_class(type(m), rows, min_n)
    _STATE["levers"].setdefault("trunk", []).append(("msa_chunk", lambda tw, kw, rec: _check_msa_chunk(tw, kw, rec, rows, min_n)))
    return f"trunk.msa_module.msa_pair_weighted_averaging: {len(mods)} modules on opt_core.mem.chunk.chunk_rows(rows={rows}, min_n={min_n})"


def _check_msa_chunk(tw, kw: dict, rec, rows: int, min_n: int = 0) -> dict:
    """At the trunk call: the chunked class is in force on every PWA instance (else a named fallback); below ``min_n`` the item's pair
    extent leaves the carried forward in force — a named skip; the mark itself comes from the core chunker's record entry when the
    levered path runs."""
    mods = _pwa_modules(tw) or []
    bad = [type(m).__name__ for m in mods if type(m).__name__ != "BigPWA"]
    if bad or not mods:
        rec.fallback("msa_chunk", f"the chunked class is not in force on {','.join(bad) or 'any module'}: the MSA module ran at its own slice")
        return kw
    n = _pair_extent(kw.get(ITEM_ANCHOR))
    if min_n > 0 and n is not None and n < min_n:
        rec.skip("msa_chunk", f"pair extent {n} < min_n {min_n}: the carried pair-weighted averaging runs (the mode's own path)")
    return kw


def _install_nograph(parts, S, rec, min_n: int = 0):
    """min_n 0: the stack built the wrapper un-graphed (graphed_override); the denoiser check marks it per item. min_n > 0 (the gated
    line): the wrapper keeps the stack's own rule; at each item's trunk call ``_gate_nograph`` sets the adopted wrapper's ``graphed``
    switch — False at pair extent >= min_n, the rule's own value below (restored after an un-graphed item) — and the denoiser check marks
    or skips by the same numbers."""
    dw = _reclass(parts, "diffusion", _denoiser_class)
    if min_n > 0:
        _reclass(parts, "trunk", _trunk_class)                                   # the item boundary and its pair extent are read at the trunk call
        _STATE.update(dw=dw, graphed_rule=bool(getattr(dw, "graphed", False)), forced_nograph=False)
        _STATE["levers"].setdefault("trunk", []).append(("nograph", lambda tw, kw, rec: _gate_nograph(tw, kw, rec, min_n)))
    _STATE["levers"].setdefault("denoiser", []).append(("nograph", lambda d, rec: _check_nograph(d, rec, min_n)))
    return "diffusion:nograph" if min_n <= 0 else f"diffusion:nograph per item at N >= {min_n} (trunk gate)"


def _gate_nograph(tw, kw: dict, rec, min_n: int) -> dict:
    """At the trunk call of a gated line: the hoisted denoiser wrapper's ``graphed`` switch for THIS item — un-graphed at pair extent >=
    min_n; below it the stack's own rule (the value the wrapper was built with), put back if the previous item had forced it off. Below
    the gate the lever is a named skip for the unit."""
    dw = _STATE.get("dw")
    if dw is None:
        rec.fallback("nograph", "no adopted denoiser wrapper to switch: the step keeps whatever graphed rule it has")
        return kw
    n = _pair_extent(kw.get(ITEM_ANCHOR))
    if n is not None and n >= min_n:
        if getattr(dw, "graphed", None):
            dw.graphed = False
            _STATE["forced_nograph"] = True
        return kw
    if _STATE.get("forced_nograph"):                                             # back below the gate after an un-graphed item: the rule's own value again
        dw.graphed = bool(_STATE.get("graphed_rule"))
        _STATE["forced_nograph"] = False
    if _STATE["unit"] is not None and (_STATE["unit"], "nograph") not in _STATE["marked"]:
        _STATE["marked"].add((_STATE["unit"], "nograph"))
        rec.skip("nograph", f"pair extent {n} < min_n {min_n}: the step keeps the stack's own rule (graphed={bool(getattr(dw, 'graphed', None))})")
    return kw


def _check_nograph(dw, rec, min_n: int = 0) -> None:
    if min_n > 0 and (_STATE.get("unit_n") is None or _STATE["unit_n"] < min_n):
        return                                                                   # below the gate: the trunk gate recorded the named skip for this unit
    if getattr(dw, "graphed", None):
        rec.fallback("nograph", "the denoiser wrapper has graphed=True: the step ran from a CUDA graph")
        return
    _mark_once(rec, "nograph", detail="graphed=False")


# ------------------------------------------------------------------------------------------------------------ the installers
def _install_trunk_chunk(parts, S, rec, rows: int, min_n: int):
    """The trunk wrapper's ``cfg_fn`` (the stack's own per-call fixer of ``chai1_eager.trunk.CFG``: ``Components.cfg_lc`` sets the fast
    line's kernels before EVERY trunk call) is wrapped: after it runs, ``CFG['trimul_impl']`` and ``CFG['triattn_impl']`` become the chunked plugs, which hand every
    call below ``min_n`` back to what cfg_fn set (the fast line's kernels) and serve every batch at or above it."""
    min_n = int(_STATE.get("min_n_override", {}).get("trunk_chunk", min_n))       # the device memory class's gate (_resolve_memclass), else the prior
    tw = _reclass(parts, "trunk", _trunk_class)
    TR = getattr(S, "trunk_module", None)                              # the stack's trunk module (chai1_eager.trunk; a stub names its own)
    if TR is None:
        import importlib
        TR = importlib.import_module("chai1_eager.trunk")
    if not isinstance(getattr(TR, "CFG", None), dict) or "trimul_impl" not in TR.CFG or "triattn_impl" not in TR.CFG:
        raise LookupError("trunk_chunk: the eager trunk module carries no CFG['trimul_impl'] / CFG['triattn_impl'] plug points")
    if not hasattr(tw, "cfg_fn"):
        raise LookupError("trunk_chunk: the installed trunk wrapper carries no cfg_fn (the stack's per-call CFG fixer)")
    inner = tw.cfg_fn
    state = {"prev": None}

    def impl(mod, z, mask):
        if z.shape[-3] < min_n:                                        # every batch (the template stack runs its 2-block pairformer at B = n_templates)
            _site_skip(rec, "trunk_chunk", int(z.shape[-3]), min_n, "triangle_multiplication")
            prev = state["prev"]
            return prev(mod, z, mask) if prev is not None else NotImplemented
        return _trimul_chunked(mod, z, mask, rows, _STATE["record"])
    impl.__name__ = "trimul_chunked"

    def impl_attn(mod, zn, mask):
        if zn.shape[-3] < min_n:
            _site_skip(rec, "trunk_chunk", int(zn.shape[-3]), min_n, "triangle_attention")
            prev = state["prev_attn"]
            return prev(mod, zn, mask) if prev is not None else NotImplemented
        return _triattn_chunked(mod, zn, mask, rows, _STATE["record"])
    impl_attn.__name__ = "triattn_chunked"

    def cfg_fn():
        if inner:
            inner()
        cur = TR.CFG.get("trimul_impl")
        if cur is not impl:
            state["prev"] = cur                                        # what the stack set for this call (the fast line's kernel)
        TR.CFG["trimul_impl"] = impl
        cur = TR.CFG.get("triattn_impl")
        if cur is not impl_attn:
            state["prev_attn"] = cur
        TR.CFG["triattn_impl"] = impl_attn
    cfg_fn.big_trunk_chunk = impl
    cfg_fn.big_trunk_chunk_attn = impl_attn
    tw.cfg_fn = cfg_fn
    _STATE["levers"].setdefault("trunk", []).append(("trunk_chunk", lambda tw_, kw, rec_: _check_trunk_chunk(tw_, kw, rec_, cfg_fn, min_n)))
    return f"EagerTrunkWrapper.cfg_fn wrapped -> CFG[trimul_impl]=trimul_chunked, CFG[triattn_impl]=triattn_chunked (rows={rows}, min_n={min_n}) after the stack's own fixer"


def _install_opm_chunk(parts, S, rec, rows: int, min_n: int):
    """At the trunk call: the chunked class is in force on every OPM instance (else a named fallback); a crop below ``min_n`` is a named
    skip; the mark itself comes from the core chunker's record entry when the levered path runs."""
    min_n = int(_STATE.get("min_n_override", {}).get("opm_chunk", min_n))         # as above
    tw = _reclass(parts, "trunk", _trunk_class)
    mods = _opm_modules(tw)
    if not mods:
        raise LookupError("opm_chunk: the installed trunk has no msa_module.outer_product_mean")
    for m in mods:
        if type(m).__name__ != "BigOPM":
            m.__class__ = _opm_class(type(m), rows, min_n)
    _STATE["levers"].setdefault("trunk", []).append(("opm_chunk", lambda tw_, kw, rec_: _check_opm_chunk(tw_, kw, rec_, min_n)))
    return f"trunk.msa_module.outer_product_mean: {len(mods)} modules on opt_core.mem.chunk.chunk_rows(rows={rows}, min_n={min_n})"


def _check_opm_chunk(tw, kw: dict, rec, min_n: int) -> dict:
    mods = _opm_modules(tw) or []
    bad = [type(m).__name__ for m in mods if type(m).__name__ != "BigOPM"]
    if bad or not mods:
        rec.fallback("opm_chunk", f"outer_product_mean instances not re-classed: {bad or 'none found'}")
        return kw
    z = kw.get(ITEM_ANCHOR)
    n = int(z.shape[-3]) if z is not None and len(getattr(z, "shape", ())) >= 3 else None
    if n is not None and n < min_n:
        rec.skip("opm_chunk", f"pair extent {n} < min_n {min_n}: the carried forward runs")
    return kw


def _check_trunk_chunk(tw, kw: dict, rec, cfg_fn, min_n: int) -> dict:
    """At the trunk call: the wrapper's cfg_fn is still ours (else a named fallback); a crop below ``min_n`` is a named skip (the fast
    kernel runs); the mark itself comes from the core chunker's record entry when the levered path runs."""
    if getattr(tw, "cfg_fn", None) is not cfg_fn:
        rec.fallback("trunk_chunk", f"the trunk wrapper's cfg_fn is {getattr(tw, 'cfg_fn', None)!r}, not the chunked plug's")
        return kw
    z = kw.get(ITEM_ANCHOR)                                            # the pair repr [B, N, N, C]: its extent is the crop
    n = int(z.shape[-3]) if z is not None and len(getattr(z, "shape", ())) >= 3 else None
    if n is not None and n < min_n:
        rec.skip("trunk_chunk", f"pair extent {n} < min_n {min_n}: the fast line's kernel runs")
    return kw


def _trimul_chunked(mod, z, mask, rows: int, rec):
    """chai's merged TriangleMultiplication (``x = LN(x1_out) + LN(x2_in)`` → linear_out × out_gate) on output-row blocks: two core
    ``triangle_multiplication_chunked`` calls — outgoing first, its ``finish`` returning the fp32 ``LN(x1)`` block (assembled into one
    fp32 plane), then incoming, whose ``finish`` adds that plane's rows to ``LN(x2)`` and applies the output projection and gate.
    Every projection is the block's FULL-width GEMM (stock's extents) sliced to its channels afterwards; the masks are stock's
    (``mask`` for a1/b1, ``mask^T`` for a2/b2 — the core receives the transposed view for the incoming call)."""
    import torch
    import torch.nn.functional as F
    from opt_core.mem import chunk as C
    from chai1_eager.trunk import BF, F32, bfw
    Cc, D = mod.c, mod.d
    Wp, Wg, Wo = bfw(mod.merged_linear_p.weight), bfw(mod.merged_linear_g.weight), bfw(mod.linear_z_out.weight)
    lnw, lnb = mod.layernorm_z_in.weight, mod.layernorm_z_in.bias

    def ln_in(xr):
        return F.layer_norm(xr.to(F32), (Cc,), lnw, lnb)

    def gated(x_ln):
        xb = x_ln.to(BF)
        pj = F.linear(xb, Wp)
        g = torch.sigmoid(F.linear(xb, Wg))
        return torch.mul(pj, g[..., :-Cc]), g

    def operand(slot):
        def f(x_ln, m):
            ab, _ = gated(x_ln)
            t = ab[..., slot * D:(slot + 1) * D]
            return t if m is None else t.masked_fill(torch.bitwise_not(m.unsqueeze(-1)), 0)
        return f

    def emit(lever, site, exact, reason, **entry):                # the core's per-call sink
        _STATE["chunk_entries"] += 1
        _mark_once(rec, "trunk_chunk", detail=f"rows={rows} N={entry.get('rows')} n_chunks={entry.get('n_chunks')} (core triangle_multiplication_chunked, {exact})")

    bT_cache = {}

    def channel_major(b):
        """the fixed operand [B, N, N, D] -> [B, D, N, N] ONCE per operand (torch.einsum would re-copy it for every block, and those
        copies dominate the chunked call); the batched matmuls below are trimul_bmm's own, with M = the block.
        Keyed by the operand OBJECT (a strong reference held in the entry): a data_ptr key can serve the freed b1 plane's stale copy to
        the incoming direction once the allocator reuses its block (wrong products, silently)."""
        e = bT_cache.get(id(b))
        if e is None or e[0] is not b:
            bT_cache.clear()
            e = bT_cache[id(b)] = (b, b.permute(0, 3, 1, 2).contiguous())
        return e[1]

    def product_out(a, b):                                            # x1[i,j,d] = sum_k a[i,k,d] b[j,k,d]  (a: rows i of the block)
        aT = a.permute(0, 3, 1, 2)                                     # [B, D, R, N]
        return torch.matmul(aT, channel_major(b).transpose(-1, -2)).permute(0, 2, 3, 1)   # [B, R, N, D] (a strided view; LN below reads it)

    def product_in(a, b):                                             # x2[i,j,d] = sum_k a[k,i,d] b[k,j,d]  (a: columns i of the block, [B, N, R, D])
        aT = a.permute(0, 3, 2, 1)                                     # [B, D, R, N] = a[k,i,d] transposed on (k, i)
        return torch.matmul(aT, channel_major(b)).permute(0, 2, 3, 1)  # [B, R, N, D]

    out_parts = C.TriMulParts(layer_norm_in=ln_in, operand_a=operand(0), operand_b=operand(1), product=product_out,
                              finish=lambda p1, x_ln_rows: F.layer_norm(p1.to(F32, memory_format=torch.contiguous_format), (D,)), outgoing=True)
    ln1 = C.triangle_multiplication_chunked(z, mask, out_parts, chunk=rows, mode="rows", record=emit, arrange=False, lever="trunk_chunk")   # fp32 [B,N,N,D]
    bT_cache.clear()                                                   # the outgoing direction's fixed operand is gone with its plane
    cursor = {"i": 0}

    def finish_in(p2, x_ln_rows):
        r = p2.shape[-3]
        i0 = cursor["i"]; cursor["i"] = i0 + r
        x = torch.add(ln1.narrow(-3, i0, r), F.layer_norm(p2.to(F32, memory_format=torch.contiguous_format), (D,)))
        _, g = gated(x_ln_rows)
        y = F.linear(x.to(BF), Wo)
        return torch.mul(y, g[..., -Cc:])
    in_parts = C.TriMulParts(layer_norm_in=ln_in, operand_a=operand(2), operand_b=operand(3), product=product_in, finish=finish_in, outgoing=False)
    y = C.triangle_multiplication_chunked(z, mask.transpose(-1, -2), in_parts, chunk=rows, mode="rows", record=emit, arrange=False, lever="trunk_chunk")
    if cursor["i"] != z.shape[-3]:
        raise RuntimeError(f"trunk_chunk: the incoming pass consumed {cursor['i']} rows of {z.shape[-3]}")
    del ln1
    return y


def _opm_class(base, rows: int, min_n: int):
    """The eager trunk's OuterProductMean re-classed onto the core's ``chunk_rows`` over the pair-ROW axis (the MSA's N axis): for a
    block of rows i, the trace's own depth loop runs with A from the block's columns and B from the whole depth slice (cached per
    slice for the call), the outer product [B, R, N, g·e·e] accumulated over depth, then the output LayerNorm and projection on the
    block — the [B, N, N, 512] product / fp32 LayerNorm planes never exist whole. Row-local by construction
    (every (i, j) entry is an independent sum over depth); below ``min_n`` the carried forward runs."""
    class BigOPM(base):
        def forward(self, msa, msa_mask):
            import torch
            import torch.nn.functional as F
            from opt_core.mem import chunk as C
            from chai1_eager.trunk import BF, F32, bfw
            B, S, N = msa.shape[0], msa.shape[1], msa.shape[2]
            if N < min_n:
                _site_skip(_STATE["record"], "opm_chunk", int(N), min_n, "outer_product_mean")
                return base.forward(self, msa, msa_mask)
            rec = _STATE["record"]
            wa, wb = self._weights_bf16()
            CH = self.CH
            bm_cache = {}

            def slice_xb(s0):
                ch, mk = msa[:, s0:s0 + CH], msa_mask[:, s0:s0 + CH]
                x = F.layer_norm(ch.to(F32), (self.c_m,))
                x.masked_fill_(torch.bitwise_not(mk.reshape(B, mk.shape[1], N, 1)), 0)
                return x.to(BF)

            def bm_full(s0):
                t = bm_cache.get(s0)
                if t is None:
                    t = bm_cache[s0] = torch.einsum("abc,defc->abdef", wb, slice_xb(s0))
                return t

            def block(blk, i0, i1):                                    # blk: [B, R, S, c_m], the MSA's pair-row block (transposed view)
                msa_rows = blk.transpose(1, 2)                         # [B, S, R, c_m]
                acc = None
                for s0 in range(0, S, CH):
                    ch, mk = msa_rows[:, s0:s0 + CH], msa_mask[:, s0:s0 + CH, i0:i1]
                    x = F.layer_norm(ch.to(F32), (self.c_m,))
                    x.masked_fill_(torch.bitwise_not(mk.reshape(B, mk.shape[1], i1 - i0, 1)), 0)
                    xb = x.to(BF)
                    del x
                    A = torch.einsum("abc,defc->abdef", wa, xb)
                    del xb
                    op = torch.einsum("abcde,afcdg->cegabf", A, bm_full(s0))
                    del A
                    op = op.reshape(B, i1 - i0, N, -1)
                    acc = torch.add(op, 0) if acc is None else torch.add(acc, op)
                    del op
                x = F.layer_norm(acc.to(F32), (acc.shape[-1],), self.ln_out.weight, self.ln_out.bias, 0.1)
                return F.linear(x.to(BF), bfw(self.linear_out.weight), bfw(self.linear_out.bias))

            def emit(lever, site, exact, reason, **entry):        # the core's per-call sink
                _STATE["chunk_entries"] += 1
                _mark_once(rec, "opm_chunk", detail=f"rows={rows} N={entry.get('rows')} n_chunks={entry.get('n_chunks')} (core chunk_rows site=outer_product_mean, {exact})")
            # chunk_rows assembles along the SAME axis it slices: the MSA's pair-row axis (N, dim 2) is presented at dim 1 (a transposed
            # view, no copy) so the block output [B, R, N, c_z] assembles into the pair-shaped [B, N, N, c_z]
            y = C.chunk_rows(block, msa.transpose(1, 2), 1, rows, exact="bitwise", record=emit, with_offsets=True, lever="opm_chunk",
                             site="outer_product_mean",
                             reason="the outer-product mean is row-local along the pair's i axis: each (i, j) entry is an independent sum over depth")
            bm_cache.clear()
            return y
    BigOPM.__name__ = BigOPM.__qualname__ = "BigOPM"
    return BigOPM


def _opm_modules(tw):
    trunk = getattr(tw, "trunk", None)
    msa = getattr(trunk, "msa_module", None) if trunk is not None else None
    if msa is None:
        return None
    return list(getattr(msa, "outer_product_mean", []))


def _triattn_chunked(mod, zn, mask, rows: int, rec):
    """chai's TriangleAttention between its input LayerNorm and ``linear_out`` (the ``CFG['triattn_impl']`` contract: ``zn`` is the
    LayerNorm'd pair cast to bf16 (trunk.py ``ln_bf16``; ``.to(BF)`` below is then the same tensor), the return is ``cat(starting, ending)`` gated outputs) on QUERY-ROW blocks through the core's
    ``triangle_attention_chunked``: per direction, pass 1 assembles the full triangle bias from row blocks (the bias of pair (j, k)
    serves every query row), pass 2 projects q/k/v/g for one block of query rows and runs the module's own SDPA statement with the full bias —
    what is held is the full bias (2·H channels) and one block's projections, never the [B, N, N, 4·H·dh] plane. The ending
    direction runs on the transposed view (stock's own statement, whose output stays in that orientation, as traced); its bias
    is stock's un-transposed one (transposed back from the pass-1 assembly on the view). ``layer_norm`` changes nothing here: ``zn`` is
    already normalised at this plug point."""
    import torch
    import torch.nn.functional as F
    from opt_core.mem import chunk as C
    from chai1_eager.trunk import BF, bfw
    B, N = mask.shape[0], mask.shape[1]
    H, dh = mod.H, mod.dh

    def sdpa(q, k, v, bias, B, H):                                     # the module's own statement per query-row block (chai1_eager.trunk
        return F.scaled_dot_product_attention(q, k, v, bias)           # TriangleAttention.forward: F.scaled_dot_product_attention(q, k, v, b[d]));
                                                                       # the full bias [B*H, 1, N, N] broadcasts over the block's R rows
    Wb = bfw(mod.pair2b.weight)
    not_mask = torch.bitwise_not(mask.reshape(B, 1, 1, 1, N, N))

    def emit(lever, site, exact, reason, **entry):                # the core's per-call sink
        _STATE["chunk_entries"] += 1
        _mark_once(rec, "trunk_chunk", detail=f"rows={rows} N={entry.get('rows')} n_chunks={entry.get('n_chunks')} (core triangle_attention_chunked {site}, {exact})")

    def bias_rows(x_ln_rows):                                          # [B, R, N, 2H] bf16: the projection stock takes once on the whole plane
        return F.linear(x_ln_rows.to(BF), Wb)

    outs = []
    for d, W in enumerate((mod.pair2qkvg1.weight, mod.pair2qkvg2.weight)):
        Wd = bfw(W)
        x_d = zn if d == 0 else zn.transpose(1, 2)

        def attention(x_ln_rows, bias_full, mask_rows, d=d, Wd=Wd):
            b = bias_full if d == 0 else bias_full.transpose(1, 2)     # stock's bias is un-transposed for both directions
            b = b.reshape(B, N, N, 2, H).permute(0, 3, 4, 1, 2).reshape(B, 2, H, 1, N, N)
            b = b.masked_fill(not_mask, -10000)
            b = b.reshape(B, 2, H, N, N).permute(1, 0, 2, 3, 4).reshape(2, B * H, 1, N, N)[d]
            R = x_ln_rows.shape[-3]
            x = F.linear(x_ln_rows.to(BF), Wd).view(B, R, N, H, 4, dh)
            q, k, v, g = (x[:, :, :, :, i, :].permute(0, 3, 1, 2, 4).reshape(B * H, R, N, dh) for i in range(4))
            o = sdpa(q, k, v, b, B, H)                                  # [B*H, R, N, dh]
            o = torch.mul(o, torch.sigmoid(g))
            return o.reshape(B, H, R, N, dh).permute(0, 2, 3, 1, 4).reshape(B, R, N, H * dh)
        parts = C.TriAttnParts(layer_norm=lambda xr: xr, bias=bias_rows, attention=attention)
        y = C.triangle_attention_chunked(x_d, mask if d == 0 else mask.transpose(1, 2), parts, chunk=rows, starting=True, record=emit,
                                         lever="trunk_chunk")
        outs.append(y)                                                 # stock keeps the ending direction in the transposed orientation (as traced)
    out = torch.cat(outs, -1)
    del outs
    return out


# ------------------------------------------------------------------------------------------------------------ the exit
def exit_gate(rc: int, allow_partial: bool = False) -> dict:
    """Close the open unit, run the census and the fail-closed gate on a partial ITEM (a lever that fell back or never marked inside a fold —
    a runtime event, fail-loud: exit 3) with its recorded opt-out (``allow_partial``: the driver's ``--allow-partial``, the one opt-out on
    that route — the PARTIAL line names the route's word, ``Ctx.opt_out``); prints the partial line when there is one. Returns the verdict (``exit_code`` is the code the
    caller exits with). A lever switched off at activation is not an item event (the activation's NOTE line named it)."""
    rec = _STATE["record"]
    if rec is None:
        return {"exit_code": rc, "partial": [], "skipped": "big not applied in this process"}
    unit_end()
    v = rec.exit_gate(rc, expect_units=True, allow_partial=bool(allow_partial))   # the flag's value, stored on the record (the verdict and the census block agree)
    gl = gate_line()
    if gl:
        sys.stderr.write(gl + "\n"); sys.stderr.flush()                        # the gated line's per-item decisions, counted (exact / fast)
        v["gate_census"] = gate_census()
    line = rec.exit_line(TAG, v)
    if line:
        sys.stderr.write(line + "\n"); sys.stderr.flush()
    return v


def manifest_block() -> Optional[dict]:
    rec = _STATE["record"]
    if rec is None:
        return None
    blk = rec.manifest_block()
    cz = gate_census()
    return dict(blk, line_name=_STATE["line_name"], gate_census=cz) if cz is not None else blk
