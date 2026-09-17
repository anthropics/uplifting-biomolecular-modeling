"""The mode table — the ONE place a kit mode is defined for Chai-1 (every command, the resolver, the CLI, run.sh and the tests
read it from here; configs carry no mode and no lever switch).

Three kits compose a mode. The driver kit's switch is ``--levels`` (``kit/chai_worker.py`` argparse, default ``W1,W2,W5``): a comma list
of lever names the driver installs at import (W1, W5: three monkeypatched attributes) or applies in its loop (W2: the feature context built
once per input); the fold call is upstream's ``run_folding_on_context``, once per seed. The eager stack's switch is the
``levers`` argument of ``chai1_eager.stack.install`` (``opt/forward/eager_trunk/chai1_eager/stack.py``, ``LEVERS`` = stock | tier1):
it replaces ``chai1.load_exported`` so that upstream's own fold function receives the eager trunk, the hoisted (CUDA-graphed)
diffusion module and flat eager embedders / confidence head instead of the six TorchScript modules; the package installs it on top of
the driver's levers, after the driver's own W1 memo (which stays the weight source underneath: ``stack.py`` ``Components``). The DSTEP
add-on's switch is the lever tuple of ``chai1_fastln.stackx.build_lever_parts`` (``opt/forward/dstep_megakernel/chai1_fastln/stackx.py``,
``ALL_LEVERS`` = hoist2, compiled, dit_attn): it builds a private denoiser for the eager line with the lever's hoister / compiled step / the add-on's attention router bound
inside that transpiled component's namespace only; the package serves it through the eager stack's own ``make_loader``. A kit mode is
therefore a literal ``--levels`` list plus one eager lever name plus a DSTEP lever tuple, nothing else. Rows:

  off     stock — the upstream API in a clean process (stock_fold.py), no kit module on the path, no ``CHAI_*`` switch set.
  exact   ``--levels W1,W2,W5`` (the driver's own default) + ``tier1`` + ``hoist2`` + the pair-track levers ``templ_empty`` ``exactln`` ``msa_pad``
          ``transition`` + the host-side levers ``rankcc`` ``tailasync`` ``confmemo`` ``prefetch`` (+ the implied ``alloc``): W1 resident modules + resident ESM, W2
          per-input feature context, W5 cross-input ESM memo, the eager stack's tier1 line (the exported model re-expressed as eager
          PyTorch: copy-free trunk layouts, the hoisted denoiser step — a CUDA graph at default numerics, un-graphed under the
          deterministic recipe by ``install``'s own rule — flat embedders and confidence head), the denoiser step's value-taint hoister
          (``hoist2``, chai1_eager/hoist.py HoistedForward2: the step-invariant atom-pair block, pair biases and AdaLN conditioning run once
          per sample instead of once per step — same ops, same operands, bitwise to the base hoister), the pair-track levers
          (pairtrack.py: each bitwise by the module's own algebra or bit-compared at its first call) and the host-side levers (postproc.py,
          featfast.py: exact by construction); the fold call is upstream's own ``run_folding_on_context``. Byte-identical to stock under the
          deterministic recipe (det.py) on one and the same torch / CUDA stack: the pinned stack
          (``PINNED_STACK``, torch 2.13.0+cu130); on any other stack the run carries the
          record ``STACK not pinned: torch <v>+<cuda> (pinned: torch 2.13.0+cu130)`` (a NOTE line; ``stack_rule`` bitwise: nothing is
          required of the box).
  fast    exact's composition + ``compiled`` + ``dit_attn`` + ``v4trimul`` + ``triattn`` + ``trunk_n`` + ``msa_rows`` (``memory_free``) + the implied opt-in ``tf32``: the hoisted denoiser's per-step function through
          torch.compile / Inductor inside the kit's CUDA graph (chai1_eager/hoist.py ``compile``: fused pointwise / layout glue, not bitwise; the
          first item of a process at each crop pays the compile), the shared core's fused triangle
          multiplication on the pairformer's and the MSA module's TriMul (pairtrack.py: ``opt_core.trimul.by_word`` with the provider's tier
          word ``fast``; ``big`` asks the word ``big``), and TF32 tensor-core
          products for the process's fp32 GEMMs (precision.py; stock enables none); ``dit_attn`` is the add-on's attention router on the
          denoiser's attention, ``trunk_n`` the trunk call at the live-token extent (trunk_n.py), ``msa_rows`` the MSA module on the rows
          that carry a mask (big.py). Tier 2 (inside stock's own seed-to-seed band; same
          steps, recycles, schedule and outputs). Runs at any fold settings — nothing in the composition depends on the number of diffusion
          samples. Its Triton kernels build their launchers with a C compiler at first use: refused by name on a box without one
          (``stack_rule`` triton); pinned stack: ``KitMode.stack``.
  big   fast's composition by reference (its trunk levers and implied ``tf32`` included) plus the memory line of ``big.py`` applied
          through ``opt_core.mem`` (msa_rows, msa_chunk, trunk_chunk, opm_chunk, nograph) and the allocator's expandable segments (the
          ``alloc`` lever, implied): fast-class numerics, folds the top crop (2048 tokens) on one 80 GB card; ``--n_gpu`` P ∈ {1} (ngpu.py).

Implied levers (``OPTIN_LEVERS``): levers the package itself applies beside a mode's composition because the mode's row names them
(``KitMode.implied_optin``: ``tf32`` on fast / big, ``alloc`` on exact / fast / big); there is no request for a lever beside a mode. Each
names the modes it may join (``OptIn.modes``); one outside them is refused by name (``optin_refusal``), never dropped. ``tf32``: TF32 tensor-core products for every fp32 GEMM of the process (``torch.backends.cuda.matmul.allow_tf32``; precision.py) —
the transpiled denoiser's GEMMs (fp32 as traced), the flat embedders' and the confidence head's; the eager trunk's bf16 linears are
unaffected. It changes fp32 rounding, so it joins ``fast`` and ``big``, never ``exact`` (tier 2; implied by both rows — big carries every fast lever); applied
after the deterministic recipe and before the eager stack installs, so the hoisted step's CUDA graph captures TF32 kernels. ``alloc``: the CUDA caching allocator with expandable segments (``PYTORCH_CUDA_ALLOC_CONF`` through ``opt_core.mem.torch_alloc``;
alloc.py) — placement only, refused by name when the variable names another configuration or CUDA is already initialised;
implied on every kit row (exact, fast, big: the memory gate at crop 2048 needs it and it is read at CUDA init). ``tf32`` is implied on
fast / big.

DEFAULT_MODE — the mode a command runs at when ``--mode`` is omitted and ``CHAI1_OPT`` is unset — is ``fast`` (the package default wherever
the engine ships a fast mode); a value, never a rule computed from the table. ``KitMode.stack`` is documentation: the
torch / CUDA stack the row is pinned to (``PINNED_STACK``, torch 2.13.0+cu130 — stock/PINS.json ``stack``; no code compares it
with the box). ``KitMode.stack_rule`` is what the box must provide (``stack.mode_stack_check``): ``bitwise`` (exact) — nothing on the box; the
row is bitwise vs stock on one and the same stack, so a box on another torch / CUDA build runs it with the STACK record on its lines and in
its report (``stack.stack_pinning``: ``STACK not pinned: torch <v>+<cuda> (pinned: torch 2.13.0+cu130)``); ``triton`` (fast, big) — a
C compiler for the
Triton kernels' launcher build: without one the mode is refused by name at activation (``NOT ACTIVE: mode fast needs a C compiler for its
Triton kernels ($CC / cc / gcc / clang: none on PATH) — --mode exact and --mode off need none``, exit 3), never left to fail inside Triton.
``CHAI1_OPT_STRICT_STACK=1`` (run.sh ``--strict-stack``) turns the STACK record into a refusal by name on every mode, for a caller that
requires the pinned stack. ``CHAI1_OPT`` unset means off for the environment route (the .pth installs no finder).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Tuple

MODES: Tuple[str, ...] = ("off", "exact", "fast", "big")
KIT_MODE_NAMES: Tuple[str, ...] = ("exact", "fast", "big")

ENV = "CHAI1_OPT"                                       # the environment route: CHAI1_OPT=<mode>; unset = off (nothing fires)
WORKER_RELPATH = "chai_worker.py"             # the worker the tree RUNS (the kit driver, ``kit/chai_worker.py`` in the lever citations), relative to
                                                        # its directory (opt/forward/errata_02); it imports the kit's own module chai_proto
                                                        # from the kit directory (its $KIT switch / the package puts kit/ first on sys.path)
PROTO_RELPATH = os.path.join("kit", "chai_proto.py")    # relative to the kit directory (opt/forward/fast_inference)
EAGER_STACK_RELPATH = os.path.join("chai1_eager", "stack.py")   # the eager stack's entry point, relative to the eager directory (opt/forward/eager_trunk)
EAGER_PACKAGE = "chai1_eager"
DSTEP_STACKX_RELPATH = os.path.join("chai1_fastln", "stackx.py")   # the DSTEP add-on's lever builder, relative to its directory (opt/forward/dstep_megakernel)
DSTEP_PACKAGE = "chai1_fastln"

PINNED_STACK = "torch 2.13.0+cu130"                  # the torch / CUDA stack every row is pinned to (stock/PINS.json "stack": torch 2.13.0+cu130,
                                                        # CUDA 13.0, triton 3.7.1, gcc 11.4.0, H100) — documentation carried on KitMode.stack; nothing compares
                                                        # it with the box (the box's own record is stack.stack_pinning, from stock/check_pins.py)


@dataclass(frozen=True)
class KitMode:
    name: str
    levels: Tuple[str, ...]                 # the literal --levels list handed to the driver
    eager: Optional[str]                    # the eager stack lever installed on top (chai1_eager.stack.LEVERS: tier1); None = none
    tier: str                               # the kits' own numerics reading of the composition
    driver_path: str                        # which fold loop runs
    dstep: Tuple[str, ...] = ()             # the DSTEP add-on's levers on the eager line (chai1_fastln.stackx.ALL_LEVERS: hoist2 | compiled | dit_attn)
    in_process: Tuple[str, ...] = ()        # levers enable() can install in a program of its own (chai_lab imported, before any load)
    stack: Optional[str] = None             # documentation: the torch / CUDA stack the row is pinned to (PINNED_STACK); compared with nothing
    notes: Tuple[str, ...] = field(default_factory=tuple)
    memory: Tuple[str, ...] = ()            # the memory levers of the row (big.LINE), applied through opt_core.mem and probed on its applied record
    memory_gated: Tuple[str, ...] = ()      # memory levers the row applies ABOVE a size only (GATED_MEMORY: msa_chunk + nograph on exact / fast), through the
                                            # same opt_core.mem machinery (big.apply_line(mode=)); not in lever_names / levers_applied= (conditional per
                                            # item): the ACTIVE line names them as big=gate<memory_gate>:<levers>, each item prints its MEMORY line
    memory_free: Tuple[str, ...] = ()       # memory levers the row applies at EVERY size through big.py's gated line (fast: msa_rows — the MSA module on the
                                            # rows that carry a mask; tolerance-class at deep MSAs, inside fast's band; exact does not carry it)
    memory_gate: int = 0                    # the pair extent (== crop) at or above which memory_gated applies: MEMORY_GATE_N = 2048, chai1's top crop —
                                            # every crop <= 1536 runs the row's own path unchanged
    implied_optin: Tuple[str, ...] = ()     # package opt-in levers the row always carries (exact: `alloc`; fast / big: `tf32` + `alloc` — the allocator policy rides every
                                            # row whose memory gate can engage: expandable segments are read at CUDA init, before any crop is known) — merged into every request
    pairtrack: Tuple[str, ...] = ()         # the trunk levers on the eager line's structured trunk (pairtrack.LEVERS), installed after the eager stack
    postproc: Tuple[str, ...] = ()                     # the fold's host-side levers that ride the row, by name (postproc.LEVERS rankcc / tailasync + featfast.LEVERS confmemo / prefetch): exact-class, every kit row
    stack_rule: Optional[str] = None        # what the box must provide (stack.mode_stack_check): "bitwise" — nothing; the row is bitwise vs stock on one
                                            # and the same stack (another stack carries the STACK record);
                                            # "triton" — a C compiler for its Triton kernels' launcher build: refused by name without one

    @property
    def levels_arg(self) -> str:
        return ",".join(self.levels)

    @property
    def lever_names(self) -> Tuple[str, ...]:
        """Every lever of the row — the driver's, the eager stack's, the DSTEP add-on's, the memory line's — in the activation line's order."""
        return self.levels + ((self.eager,) if self.eager else ()) + self.dstep + self.pairtrack + self.postproc + self.memory

    @property
    def dstep_arg(self) -> str:
        return ",".join(self.dstep) if self.dstep else "none"


MEMORY_GATE_N: int = 2048                                        # pair extent (== crop, chai1's AVAILABLE_MODEL_SIZES 256…1536, 2048) at or above which exact and
                                                                 # fast apply GATED_MEMORY: the top crop only (> 1536 tokens); CHAI1_BIG_<LEVER>_MIN_N overrides per lever
GATED_MEMORY: Tuple[str, ...] = ("msa_chunk", "nograph")         # big.py levers: the MSA pair-weighted averaging in 1024-row blocks (row-local: the same
                                                                 # arithmetic per row) and the hoisted denoiser step un-graphed (no graph pool) — what makes
                                                                 # crop 2048 with a deep MSA fit one 80 GB card (big.py)

KIT_MODES: Dict[str, KitMode] = {
    "exact": KitMode(
        name="exact", levels=("W1", "W2", "W5"), eager="tier1", dstep=("hoist2",), pairtrack=("templ_empty", "exactln", "msa_pad", "transition"), postproc=("rankcc", "tailasync", "confmemo", "prefetch"), implied_optin=("alloc",),
        memory_gated=GATED_MEMORY, memory_gate=MEMORY_GATE_N,
        tier="bitwise under the deterministic recipe: the driver kit at its settings with the exported model re-expressed as eager PyTorch "
             "(the eager stack's tier1 line: copy-free trunk layouts, the hoisted denoiser step — its invariant work hoisted by value, hoist2 — flat embedders and confidence head — "
             "byte-identical CIFs and scores vs stock at 400 and 1000 tokens, 5/5 samples each, H100) and the template embedder's "
             "empty-template short-circuit (templ_empty: z returned when no template slot carries a mask, the module's own algebra), the MSA "
             "module's statement cut to the leading 8192 / 4096-row slices that carry any MSA mask position (msa_pad: the padded rows past them are "
             "all-masked — exact zeros in the outer-product mean, row-local elsewhere; the export's slice grid kept, bit-compared per shape class) and "
             "the trunk's transitions bound by tier word to the shared core's transition provider (transition, opt_core.kernels.transition asked "
             "'exact' at every call's cell: the row it resolves is fed the statement's own LayerNorm output and bit-compared per class at the first "
             "call, a differing class kept on the statement by name; a stock answer or a class without a cell keeps the statement)",
        driver_path="upstream chai1.run_folding_on_context (chai_worker.py, once per seed) over the eager stack's modules",
        in_process=("W1", "W5", "tier1", "hoist2", "templ_empty", "exactln", "msa_pad", "transition", "rankcc", "tailasync", "confmemo"),   # (+ the gated memory levers, big.apply_line(mode=)) W2 is the driver loop's feature-context reuse (chai_worker.py): a program
                                            # calling run_inference builds its own context per call, so W2 is not applicable in-process; prefetch reads the worker's uid list: not either
        stack=PINNED_STACK, stack_rule="bitwise",    # bitwise vs stock on one and the same torch / CUDA stack (the pinned stack);
                                            # another stack carries the STACK record
        notes=("under the deterministic recipe the eager stack's install rule runs the hoisted denoiser un-graphed (graph capture is "
               "refused under torch.use_deterministic_algorithms; eager README.md); at default numerics it is CUDA-graphed",
               "at default numerics the driver kit matches stock's seeds 1..n but not seed 0 bit-for-bit (traced ESM first-call effect, "
               "kit KNOWN_ISSUES.md §2); inside the stock run-to-run floor",
               "at crop 2048 (an input over 1536 tokens) the MSA pair-weighted averaging runs in 1024-row blocks and the hoisted denoiser "
               "step un-graphed (memory_gated, big.py: each item's MEMORY line); every smaller crop runs the row unchanged",),
    ),
    "fast": KitMode(
        name="fast", levels=("W1", "W2", "W5"), eager="tier1", dstep=("hoist2", "compiled", "dit_attn"), pairtrack=("templ_empty", "v4trimul", "exactln", "triattn", "msa_pad", "transition", "trunk_n"), postproc=("rankcc", "tailasync", "confmemo", "prefetch"), implied_optin=("tf32", "alloc"),
        memory_free=("msa_rows",), memory_gated=GATED_MEMORY, memory_gate=MEMORY_GATE_N,
        tier="tier 2 (not bitwise; inside stock's own seed-to-seed band): exact's composition plus TF32 tensor-core products for "
             "the process's fp32 GEMMs (the fp32 denoiser, confidence head and embedders: tf32, torch.backends.cuda.matmul.allow_tf32) and "
             "the shared core's fused triangle multiplication on the pairformer's and the MSA module's TriMul (v4trimul: opt_core.trimul.by_word "
             "with the provider's tier word `fast` — its measured row per card, c_z, crop bucket and direction) and its triangle-attention provider on their triangle attention (triattn, "
             "opt_core.kernels.triattn by the mode's tier word at every crop and card), the trunk call at the live-token extent instead of the padded crop (trunk_n: inputs narrowed to ceil64(live tokens), the trunk representations padded back with zeros for the per-crop components; tolerance-class) and the shared core's transition rows asked by the tier word 'fast' at every call's cell (transition: fused-LayerNorm rows, first call per class checked against the statement; 'big' asks its own word); "
             "same steps, recycles, schedule and outputs",
        driver_path="upstream chai1.run_folding_on_context (chai_worker.py, once per seed) over the eager stack's modules",
        in_process=("W1", "W5", "tier1", "hoist2", "compiled", "dit_attn", "templ_empty", "v4trimul", "exactln", "triattn", "msa_pad", "transition", "trunk_n", "rankcc", "tailasync", "confmemo"),
        stack=PINNED_STACK, stack_rule="triton",     # the pinned stack; its Triton kernels need a C compiler on the box (refused by name without one)
        notes=("under the deterministic recipe the eager stack's install rule runs the hoisted denoiser un-graphed (graph capture is "
               "refused under torch.use_deterministic_algorithms; eager README.md)",
               "at crop 2048 (an input over 1536 tokens) the MSA pair-weighted averaging runs in 1024-row blocks and the hoisted denoiser "
               "step un-graphed (memory_gated, big.py: each item's MEMORY line) — the crop that does not fit one 80 GB card otherwise; "
               "every smaller crop runs the row unchanged",),
    ),
}
KIT_MODES["big"] = KitMode(                                    # composes on fast BY REFERENCE: the same driver levers, eager line, DSTEP levers and stack rule
    name="big", levels=KIT_MODES["fast"].levels, eager=KIT_MODES["fast"].eager, dstep=KIT_MODES["fast"].dstep,
    pairtrack=KIT_MODES["fast"].pairtrack,                          # fast's trunk levers, by reference (big inherits fast's numerics)
    postproc=KIT_MODES["fast"].postproc,                            # fast's host-side tail / feature-build levers, by reference (no memory cost: host work only)
    memory=("msa_rows", "msa_chunk", "trunk_chunk", "opm_chunk", "nograph"),   # == big.LINE (locked by test)
    implied_optin=KIT_MODES["fast"].implied_optin,                  # fast's implied levers: tf32 + the allocator policy (alloc)
    tier="fast-class (tier 2) by guarantee — runs bigger, within the fast band: fast's composition (its triangle multiplication asking the "
         "provider's `big` word: the rows measured for memory as well as speed on the card) plus the memory line of big.py "
         "(msa_rows: the MSA module on the rows that carry a mask; msa_chunk: pair-weighted averaging on MSA-row blocks; trunk_chunk: "
         "the pairformer's triangle multiplication on output-row blocks and its triangle attention on query-row blocks at N >= 1536; "
         "opm_chunk: the outer-product mean on pair-row blocks at N >= 1536; nograph: the hoisted denoiser step un-graphed) and the "
         "allocator's expandable segments (the `alloc` lever, implied beside fast's `tf32`); each memory lever labelled bitwise | measured on its LEVER line; "
         "folds chai1's top crop (2048 tokens) on one 80 GB card where fast does not",
    driver_path=KIT_MODES["fast"].driver_path + "; the trunk's MSA rows sliced to the rows that carry any mask (big.py)",
    in_process=KIT_MODES["fast"].in_process + ("msa_rows", "msa_chunk", "trunk_chunk", "opm_chunk", "nograph", "alloc"),
    stack=KIT_MODES["fast"].stack, stack_rule=KIT_MODES["fast"].stack_rule,
    notes=KIT_MODES["fast"].notes + ("the memory levers are fixed (opt_core.mem; no switch removes one); the ACTIVE line names the line "
                                     "and the levers in force (big=) and the GPUs per fold (n_gpu=1 sharding=none)",),
)

DEFAULT_MODE: str = "fast"                                       # the package default — README.md, Modes; a value, never a rule computed from the table
DRIVER_LEVERS: Tuple[str, ...] = ("W1", "W2", "W5")               # every name the driver accepts on --levels
EAGER_LEVERS: Tuple[str, ...] = ("tier1",)                        # the eager stack's levers (chai1_eager.stack.LEVERS minus its stock arm)
DSTEP_LEVERS: Tuple[str, ...] = ("hoist2", "compiled", "dit_attn")                        # the DSTEP add-on's levers (chai1_fastln.stackx.ALL_LEVERS): hoist2 in every row; compiled, dit_attn in fast / big
PAIRTRACK_LEVERS: Tuple[str, ...] = ("v4trimul", "templ_empty", "exactln", "triattn", "msa_pad", "transition", "trunk_n")   # pairtrack.LEVERS (locked by test): the trunk levers on the eager line
POSTPROC_LEVERS = ("rankcc", "tailasync", "confmemo", "prefetch")            # the fold's host-side tail (postproc.py: rankcc, tailasync) + the feature build's conformer-library memo and next-item prefetch (featfast.py: confmemo, prefetch): exact-class by construction, every kit row


@dataclass(frozen=True)
class OptIn:
    name: str
    modes: Tuple[str, ...]                  # the kit modes the lever may join; a request under any other mode is refused by name
    tier: str                               # the numerics class the lever is carried under on this tree


OPTIN_LEVERS: Dict[str, OptIn] = {          # the package's implied levers (precision.py / alloc.py apply and probe them); a row carries them as KitMode.implied_optin
    "tf32": OptIn(name="tf32", modes=("fast", "big"),
                  tier="tier 2 (inside stock's seed-to-seed band; implied by the fast and big rows): TF32 products change the rounding of every fp32 "
                       "GEMM — never bitwise to stock, so never under exact"),
    "alloc": OptIn(name="alloc", modes=("exact", "fast", "big"),
                   tier="placement only — the CUDA caching allocator's segment policy changes no kernel input, outputs unchanged by construction; "
                        "part of exact, fast and big (implied: the memory gate at crop 2048 needs it and it is read at CUDA init); never under off (stock sets no environment)"),
}


def with_implied(mode: str, optin: Tuple[str, ...]) -> Tuple[str, ...]:
    """The opt-in list a run of ``mode`` carries: the row's implied levers (``KitMode.implied_optin``: ``alloc`` on every kit row, ``tf32``
    on fast / big) first, then the requested ones, duplicates dropped."""
    implied = KIT_MODES[mode].implied_optin if mode in KIT_MODES else ()
    return parse_optin(tuple(implied) + tuple(optin))


def parse_optin(spec) -> Tuple[str, ...]:
    """``'tf32,x'`` / ``['tf32']`` / None → the requested opt-in names in request order, duplicates dropped; unknown names kept (the
    refusal names them)."""
    if spec is None:
        return ()
    names = spec.split(",") if isinstance(spec, str) else list(spec)
    out = []
    for n in (x.strip() for x in names):
        if n and n not in out:
            out.append(n)
    return tuple(out)


def optin_refusal(mode: str, optin: Tuple[str, ...]) -> Optional[str]:
    """Why the requested opt-in levers cannot join ``mode`` (None = they can, or none requested): an unknown name, or a lever the mode
    does not admit (``OptIn.modes``) — refused by name, never dropped."""
    if not optin:
        return None
    unknown = [n for n in optin if n not in OPTIN_LEVERS]
    if unknown:
        return f"unknown opt-in lever(s) {','.join(unknown)} (the package's opt-in levers: {','.join(OPTIN_LEVERS)})"
    barred = [n for n in optin if mode not in OPTIN_LEVERS[n].modes]
    if barred:
        return "; ".join(f"opt-in lever {n} does not join mode {mode}: it runs under {'|'.join(OPTIN_LEVERS[n].modes)} only ({OPTIN_LEVERS[n].tier})" for n in barred)
    return None
# The eager stack's tier1 line ALONE (fast without the add-on's kernels) is likewise opt-in — reachable through the stack's own
# install(levers="tier1"), in no kit row.


def kit_mode(name: str) -> KitMode:
    if name not in KIT_MODES:
        raise KeyError(f"{name!r} is not a kit mode ({'|'.join(KIT_MODE_NAMES)})")
    return KIT_MODES[name]


def driver_levels(mode: str) -> str:
    """The literal ``--levels`` value for a kit mode."""
    return kit_mode(mode).levels_arg


def eager_lever(mode: str) -> Optional[str]:
    """The eager stack lever of a kit mode (``chai1_eager.stack.install(levers=...)``), None for a row without one."""
    return kit_mode(mode).eager


def refusal(mode: str, *, in_process: bool = False) -> Optional[str]:
    """Why a mode cannot run here (None = it can): in a program of its own (enable()), only the levers installable before the first load
    are reachable — a mode without an in-process form is refused. Every mode folds at any fold settings (stock's run_inference flags,
    settings.py), every one passed through."""
    if mode == "off":
        return None
    why = levers_off_refusal(mode)                                  # MODEL_OPT_LEVERS_OFF: a name the mode does not compose / cannot switch off — by name
    if why:
        return why
    km = kit_mode(mode)
    if in_process and not km.in_process:
        return (f"mode {mode} has no in-process form: its levers ({km.levels_arg}) run inside the kit driver's own loop "
                f"({km.driver_path}); use `chai1-opt pred --mode {mode}`")
    return None


# ------------------------------------------------------------------------------------------------ MODEL_OPT_LEVERS_OFF (ablation)
ENV_LEVERS_OFF = "MODEL_OPT_LEVERS_OFF"        # one word across the model-opt kits: a comma list of lever names to leave OFF for one run — an ablation, never a
                                               # production configuration. Resolved HERE, once, for every route: each named lever the mode composes is removed from the row's
                                               # composition BEFORE activation (the table below is rewritten; every consumer — the driver's --levels list, the eager
                                               # / DSTEP / pair-track installs, the gated memory levers, the implied opt-ins — reads the reduced row); the run proceeds
                                               # as a NAMED partial activation (ACTIVE … PARTIAL off=<levers> (MODEL_OPT_LEVERS_OFF) + one NOTE line; the report's
                                               # levers_off / levers_off_env) with no further opt-out (the variable is the explicit request). A name the mode does not
                                               # compose is refused by name (exit 3) listing the mode's levers; a lever the kit cannot remove on its own is refused by
                                               # name with the reason; `off` ignores the variable (one NOTE line).
NOT_SWITCHABLE: Dict[str, str] = {
    "tier1": "the eager line every other model lever rides on (hoist2, the pair-track levers, the memory levers are installed on its parts) — "
             "`--mode off` is the run without it",
    "W1": "the driver's resident-module loader: the eager line's loader and the resident ESM are installed over it in this kit (a W1-less loader "
          "topology is not one the kit builds)",
}
BIG_LINE_REASON = ("big's memory line is applied as composed (big.py over opt_core.mem: every lever of the line, no per-lever switch in this kit); "
                     "exact / fast switch their gated msa_chunk / nograph individually")
_ROWS: Dict[str, KitMode] = dict(KIT_MODES)      # the rows as composed (pristine); KIT_MODES is rewritten per MODEL_OPT_LEVERS_OFF by apply_levers_off()


def levers_off_names(environ=None) -> Tuple[str, ...]:
    """``MODEL_OPT_LEVERS_OFF`` as names (comma list, blanks dropped, order kept, duplicates dropped); () when unset / empty."""
    environ = os.environ if environ is None else environ
    out = []
    for w in (environ.get(ENV_LEVERS_OFF) or "").split(","):
        w = w.strip()
        if w and w not in out:
            out.append(w)
    return tuple(out)


def row_levers(km: KitMode) -> Tuple[str, ...]:
    """Every lever name a row composes, in the activation line's order: the driver's, the eager line, the DSTEP levers, the pair-track levers,
    the memory line, the gated memory levers, the implied opt-ins."""
    out = []
    for n in tuple(km.lever_names) + tuple(km.memory_free) + tuple(km.memory_gated) + tuple(km.implied_optin):
        if n not in out:
            out.append(n)
    return tuple(out)


def not_switchable(km: KitMode) -> Dict[str, str]:
    """lever -> why this row cannot leave it off on its own."""
    out = {n: why for n, why in NOT_SWITCHABLE.items() if n in row_levers(km)}
    for n in km.memory:                                                   # big's line: as composed
        out.setdefault(n, BIG_LINE_REASON)
    return out


def without(km: KitMode, names) -> KitMode:
    """The row minus the named levers (composition fields only; big's memory line is not reducible here — not_switchable)."""
    names = set(names)
    keep = lambda t: tuple(n for n in t if n not in names)               # noqa: E731
    return replace(km, levels=keep(km.levels), dstep=keep(km.dstep), pairtrack=keep(km.pairtrack), postproc=keep(km.postproc), memory_free=keep(km.memory_free), memory_gated=keep(km.memory_gated),
                   implied_optin=keep(km.implied_optin), in_process=keep(km.in_process))


def levers_off_refusal(mode: str, environ=None) -> Optional[str]:
    """Why ``MODEL_OPT_LEVERS_OFF`` cannot be honoured for ``mode`` (None = it can, or nothing asked, or mode off): a name the mode does not
    compose (listing the mode's lever names), or a lever the kit cannot switch off on its own (its reason)."""
    names = levers_off_names(environ)
    if not names or mode not in _ROWS:
        return None
    km = _ROWS[mode]
    known = row_levers(km)
    raw = (os.environ if environ is None else environ).get(ENV_LEVERS_OFF)
    unknown = [n for n in names if n not in known]
    if unknown:
        return (f"{ENV_LEVERS_OFF}={raw!r} names lever(s) {','.join(unknown)} that mode {mode} does not compose "
                f"(its levers: {','.join(known)})")
    cannot = not_switchable(km)
    stuck = [n for n in names if n in cannot]
    if stuck:
        return "; ".join(f"{ENV_LEVERS_OFF}={raw!r}: lever {n} of mode {mode} cannot be switched off on its own — {cannot[n]}" for n in stuck)
    return None


def levers_off_for(mode: str, environ=None) -> Tuple[str, ...]:
    """The levers ``MODEL_OPT_LEVERS_OFF`` leaves off for ``mode``: the names the pristine row composes and can switch, in the row's order."""
    names = levers_off_names(environ)
    if not names or mode not in _ROWS:
        return ()
    km = _ROWS[mode]
    cannot = not_switchable(km)
    return tuple(n for n in row_levers(km) if n in names and n not in cannot)


def no_compile_to_env(environ=None) -> str:
    """``--no-compile`` (pred / the driver; the sanctioned opt-out of the compiled denoiser step) is an alias of ``MODEL_OPT_LEVERS_OFF=compiled``:
    the one mechanism — the variable gains the name (idempotent) before the rows are reduced; returns the variable's new value."""
    env = os.environ if environ is None else environ
    names = [n for n in (env.get(ENV_LEVERS_OFF) or "").replace(" ", "").split(",") if n]
    if "compiled" not in names:
        names.append("compiled")
    env[ENV_LEVERS_OFF] = ",".join(names)
    return env[ENV_LEVERS_OFF]


def aoti_packages_present(environ=None) -> int:
    """How many ahead-of-time packages of the compiled step (``chai1_fastln.aoti``: ``$MODEL_OPT_JIT_ROOT/<jit.cache_key>/aoti/dstep_c*.pt2``,
    built by ``run.sh warm`` / ``python -m chai1_opt.warm_aoti``) the JIT root in force holds for this stack key — read WITHOUT importing torch
    (jit.cache_key reads the installed distribution), before any model exists. 0 when the root is unset or holds none."""
    import glob
    from . import jit
    env = os.environ if environ is None else environ
    root = env.get(jit.ENV_ROOT)
    if not root:
        return 0
    key = env.get(jit.ENV_KEY) or jit.cache_key(None, env)
    return len(glob.glob(os.path.join(root, key, "aoti", "dstep_c*_s*_*.pt2")))


COMPILE_NO_LAYOUT_META = "no_layout_meta"                                       # == chai1_fastln.aoti's refusal word for a package built before input layouts were recorded


def aoti_packages_layout_state(environ=None):
    """Probe the ahead-of-time packages this stack key holds — their ``*.meta.json`` only, nothing loaded, no torch: ``"meta"`` when at least one
    package records its input layouts (``inputs`` + ``input_strides``: served at its crop), ``"no_layout_meta"`` when packages are present but
    none records them (each is refused by name at its crop and the per-process compile serves), None when the JIT root
    is unset or holds no package for this key."""
    import glob, json
    from . import jit
    env = os.environ if environ is None else environ
    root = env.get(jit.ENV_ROOT)
    if not root:
        return None
    key = env.get(jit.ENV_KEY) or jit.cache_key(None, env)
    d = os.path.join(root, key, "aoti")
    pkgs = glob.glob(os.path.join(d, "dstep_c*_s*_*.pt2"))
    if not pkgs:
        return None
    for p in sorted(pkgs):
        try:
            with open(p[:-4] + ".meta.json") as fh:
                meta = json.load(fh)
        except Exception:  # noqa: BLE001 — a package without a readable meta is one the loader refuses too
            continue
        if meta.get("inputs") and meta.get("input_strides"):
            return "meta"
    return COMPILE_NO_LAYOUT_META


COMPILE_CPU_ISA_MISMATCH = "cpu_isa_mismatch"                                    # == chai1_fastln.cpu_isa.WORD (test_aoti pins the pair): a package whose launcher was compiled for CPU
CPU_ISA_RELPATH = os.path.join("chai1_fastln", "cpu_isa.py")                    # instructions this host lacks is refused by name at load; the probe module (stdlib only) beside aoti.py


def _cpu_isa_probe():
    """chai1_fastln/cpu_isa.py loaded from its file (stdlib only: no package __init__, no torch) — the words the loader itself uses."""
    import importlib.util, sys
    m = sys.modules.get("chai1_fastln.cpu_isa")
    if m is not None:
        return m
    from . import stack                                                        # lazy: stack imports this module at its top
    path = os.path.join(stack.dstep_home(), CPU_ISA_RELPATH)
    spec = importlib.util.spec_from_file_location("_chai1_fastln_cpu_isa_probe", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"no probe module at {path}")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def aoti_packages_isa_state(environ=None):
    """Probe the ahead-of-time packages this stack key holds against the HOST CPU — archive metadata + /proc/cpuinfo (or
    ``CHAI1_OPT_AOTI_HOST_ISA``), nothing loaded, no torch: ``"cpu_isa_mismatch"`` when every package with recorded input layouts records an
    ``AOTI_CPU_ISA`` word naming CPU instructions this host lacks (each is refused by name at its crop and the EAGER hoisted step serves),
    ``"ok"`` when at least one is loadable here (or records no word: not judged), None when the JIT root is unset, holds
    no such package, or the probe module is not in the tree."""
    import glob, json
    from . import jit
    env = os.environ if environ is None else environ
    root = env.get(jit.ENV_ROOT)
    if not root:
        return None
    key = env.get(jit.ENV_KEY) or jit.cache_key(None, env)
    pkgs = sorted(glob.glob(os.path.join(root, key, "aoti", "dstep_c*_s*_*.pt2")))
    if not pkgs:
        return None
    try:
        P = _cpu_isa_probe()
    except Exception:  # noqa: BLE001 — a tree without the probe: no ISA word from here (the loader's own gate still acts at the crop)
        return None
    verdicts = []
    for p in pkgs:
        try:
            with open(p[:-4] + ".meta.json") as fh:
                meta = json.load(fh)
        except Exception:  # noqa: BLE001
            continue
        if not (meta.get("inputs") and meta.get("input_strides")):
            continue                                                           # a no_layout_meta package is refused under that word first
        verdicts.append(P.verdict(p, env).get("ok"))
    if not verdicts:
        return None
    return "ok" if any(v is None or v for v in verdicts) else COMPILE_CPU_ISA_MISMATCH


COMPILE_ASIDE_CC80 = "no_aoti_packages_cc80"                                     # == chai1_fastln.aoti.ASIDE_CC80 (test_aoti pins the pair): the word both the token and the step-aside carry
_CC80_TARGET_WORDS = ("a100", "sm80", "cc80", "cc8.0", "8.0")


def compile_card_aside_word(environ=None):
    """``COMPILE_ASIDE_CC80`` when the configuration targets a cc-8.0 card (``MODEL_OPT_TARGET_GPU`` from configs/<card>.env starts with a100 /
    sm80 / cc80), else None — read from the environment only (no CUDA touch: the allocator lever owns the first one). On that card the compiled
    step serves only through an ahead-of-time package; a crop without one runs the eager hoisted step (the numerics are then the eager
    step's) instead of a per-process compile at that crop."""
    env = os.environ if environ is None else environ
    w = str(env.get("MODEL_OPT_TARGET_GPU") or "").strip().lower()
    return COMPILE_ASIDE_CC80 if w and any(w.startswith(x) for x in _CC80_TARGET_WORDS) else None


def compile_word(rep: dict, environ=None):
    """The ACTIVE line's ``compile=`` token: ``on:aoti`` when the compiled step is among the applied levers AND the JIT root holds ahead-of-time
    packages WITH recorded input layouts for this stack (aoti_packages_layout_state; ``off:no_layout_meta`` when the packages present were
    built before layouts were recorded — refused by name per crop, the eager hoisted step serves; ``off:cpu_isa_mismatch`` when the packages' launchers
    were compiled for CPU instructions this host lacks (aoti_packages_isa_state) — refused by name per crop alike; a crop with a package loads its compiled step without compiling; a crop without one compiles through
    Dynamo at its first step — each crop says which on stderr, the EXIT line counts them ``aoti=<served>/<entered>``), ``on:dynamo`` when it is
    applied and no package is there (every crop compiles at its first use in each process) — except on cc 8.0, where that reads
    ``off:card:no_aoti_packages_cc80`` (compile_card_aside_word: the compile steps aside per crop, the eager hoisted step serves), ``off:user`` when the run switched it
    off (``--no-compile`` / MODEL_OPT_LEVERS_OFF), None for a mode that never compiles (exact: bit-exactness; off). A compile that cannot engage
    on the box is only known inside the first fold: the EXIT line names it (``compiled=stepped_aside:<reason>``)."""
    if "compiled" in (rep.get("levers_applied") or []):
        state = None
        try:
            state = aoti_packages_layout_state(environ)                          # the packages' meta only: present + layouts recorded | present without | absent
        except Exception:  # noqa: BLE001 — the word never fails the activation line
            state = None
        if state == "meta":                                                     # loadable packages: served at their crops — unless their launchers were built for CPU
            try:                                                                # instructions this host lacks (aoti_packages_isa_state: the archive's AOTI_CPU_ISA vs the host):
                isa = aoti_packages_isa_state(environ)                          # then each is refused by name at its crop and the EAGER hoisted step serves
            except Exception:  # noqa: BLE001                                   # (EXIT aoti=0/n(refused:cpu_isa_mismatch:n) compiled=stepped_aside:cpu_isa_mismatch)
                isa = None
            if isa == COMPILE_CPU_ISA_MISMATCH:
                return f"off:{COMPILE_CPU_ISA_MISMATCH}"
            return "on:aoti"
        if state == COMPILE_NO_LAYOUT_META:                                     # packages present but built before layouts were recorded: each is refused by name at its crop
            return f"off:{COMPILE_NO_LAYOUT_META}"                             # and the EAGER hoisted step serves on every card (EXIT aoti=0/n(refused:no_layout_meta:n)
        why = compile_card_aside_word(environ)                                  # compiled=stepped_aside:no_layout_meta) — never a per-process build
        if why:                                                                 # cc 8.0 (configs/a100.env's MODEL_OPT_TARGET_GPU word) without a package: the compile steps aside
            return f"off:card:{why}"                                            # by name per crop (the eager hoisted step; EXIT compiled=stepped_aside:<why>)
        return "on:dynamo"
    if "compiled" in (rep.get("levers_off") or []) or ("compiled" in levers_off_names() and rep.get("mode") in ("fast", "big")):
        return "off:user"
    return None


def apply_levers_off(environ=None) -> Tuple[str, ...]:
    """Rewrite KIT_MODES from the pristine rows minus what ``MODEL_OPT_LEVERS_OFF`` leaves off per row (idempotent; unset / empty restores the
    rows as composed). Every consumer of the table (kit_mode, driver_levels, with_implied, big's gated levers, the reports) then reads the
    reduced composition. Returns the names read. A refusal (levers_off_refusal) is the caller's gate: rows are reduced by their switchable
    names only."""
    names = levers_off_names(environ)
    for mode, km in _ROWS.items():
        KIT_MODES[mode] = without(km, levers_off_for(mode, environ)) if names else km
    return names


apply_levers_off()                              # the process's own environment at import; activate() / the driver / the CLI re-apply at their start

