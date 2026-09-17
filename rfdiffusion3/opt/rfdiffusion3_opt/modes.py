"""The mode table — the one place a mode, its switches and its tier are named in this tree (configs/ carry no lever switch).

A mode resolves to an ENVIRONMENT DELTA that a caller exports before the process imports `rfd3.model`: the kit's levers are read once at
import / construction time — `RFD3_HOIST` when `rfd3.model.hoist` is imported (kit `patched/rfd3/model/hoist.py:18`),
`RFD3_FZT` when `rfd3.model.layers.layer_utils` is imported (kit `patched/rfd3/model/layers/layer_utils.py`, the `[RFD3_FZT]`
block after `Transition`), `RFD3_CUDAGRAPH` when the sampler is constructed (`patched/rfd3/model/inference_sampler.py:610-611`), `RFD3_INIT_CHUNK` when
`rfd3.model.layers.layer_utils` is imported (its `[RFD3_INIT_CHUNK]` block, called by `layers/blocks.py` `SinusoidalDistEmbed.forward`: the token initialiser's two all-atom-pair
embeddings over row blocks of at most INIT_CHUNK_PAIRS pairs — stock's tensor bit for bit, its fp32 intermediates bounded instead of L_atom²-sized). There is no in-process
activation point and no device table: the kit reads no capability (`hoist.py` and `cudagraph_sampler.py` have no compute-capability
branch), so a mode is the same delta on every GPU class the tree supports. The names, the default and the unknown-mode refusal are an
`opt_core.modes.ModeTable` (TABLE); the deltas are this kit's (KIT_MODES).

  off    stock — nothing exported; `design --mode off` runs the upstream command line in the pristine interpreter (stock_design.py)
  exact  {RFD3_HOIST: "1", RFD3_FZT: "1", RFD3_CUDAGRAPH: "1", RFD3_INIT_CHUNK: "1"} — the hoist with its five parts (pll, downcast, valid, where,
         dedup: hoist.py PARTS), the fused SwiGLU transition (`Transition._forward_impl` served by `opt_core.attn.pair_fused.transition`, impl
         fpf, ln='stock': the module's own RMSNorm output is the kernel's input, the stock bf16 rounding points kept, the hidden never written
         to memory; the core's cell table serves the route's four cells), the CUDA-graph replay of the eager denoiser step, and the row-blocked
         initialiser embedding; the kit's class "EXACT (type-A: identical designs)" (MANIFEST.json "class"): byte-identical final coordinates
         and sequence indices vs stock under the kit's deterministic recipe (`rfdiffusion3/CHANGES.md`).
  fast   the exact delta plus {RFD3_COMPILE: "1", RFD3_TOKEN_SDPA: "1", RFD3_GATHER_ATTN: "1"} — torch.compile of upstream's four compile targets
         inside the roll-out cache and the graph replay (hoist.py [RFD3_COMPILE], automatic dynamic shapes, the cache accessors as dynamo
         boundaries, over opt_core.capture.compile), the token transformer's pair-bias attention through upstream's dense SDPA function
         (attention.py [RFD3_TOKEN_SDPA]), and the atom-level index-set attention through the shared core's gather kernel (gather.py,
         RFD3_GATHER_ATTN); tolerance class: inductor's, SDPA's and the gather kernel's summation orders, not eager's — small numeric
         differences within stock's seed-to-seed spread, faster than exact. The DEFAULT mode on this engine (DEFAULT_MODE below: `--mode`
         omitted and RFDIFFUSION3_OPT unset select it; `--mode exact` selects stock's identical designs by name).
         Both kit modes carry the graph sampler, which drives every option of upstream's default sampler that lives outside the denoiser
         call — realignment (`inference_sampler.allow_realignment`), origin jitter (`s_jitter_origin`), the motif-fix schedule
         (`fraction_of_steps_to_fix_motif`) and partial diffusion (a spec's `partial_t`) — with upstream's own statements at upstream's
         points (cudagraph_sampler.py `sample_diffusion_like_af3`), and falls back to the eager denoiser by name when a capture fails
         (`fallbacks` in its tally): a run with fallbacks > 0 is a PARTIAL activation (exit 3 unless --allow-partial records it). The three
         computations the levers do not drive — classifier-free guidance, upstream's low-memory tokenization, the symmetry sampler — are
         refused by name before a run (UNSERVED below), never run under a mode's name.

A lever lives inside a mode and never names one: `--mode` lists MODES only and any other word is refused as not a mode (a lever subset is
not a mode). VARIANTS has one entry, `design`: upstream's `rfd3 design` verb on whatever the spec asks (unconditional, binder, motif
scaffolding …), under the refusals of UNSERVED below.

ROUTES is the second table: the ROUTES a pass runs under and the accelerator words each one expects (census.py, the KERNELS
census and its REQUIRE guard read it; nothing else names an expectation). A route is a mode plus what the design line asks of upstream's
own accelerator switch `compile_model` (settings.compile_requested); a kit mode with that switch on its line has no route and is
refused by name (route_refusal: COMPILE_REFUSED):

  stock         mode off + `compile_model=true`  — upstream's fastest documented environment: torch.compile of the four diffusion submodules
                (rfdiffusion3.yaml:70-71, engine.py:212-269) on top of the shipped dense-SDPA atom attention; expects compile engaged
  default       mode off, the line as shipped      — upstream exactly as shipped (compile_model unset = False); expects compile off-by-route
  exact         mode exact                          — hoist + fused transition + graph replay of the eager step; the kit's bit-identical class is stated against
                EAGER upstream (inductor reorders floating-point reductions): compile_model=true under exact is refused by name (route_refusal);
                expects compile off-by-route
  fast          mode fast                           — the exact delta with the four compile targets compiled by the kit inside the graph replay + token SDPA;
                expects compile engaged in the kit form (`engaged:kit:…`);
                compile_model=true under fast is refused by name (COMPILE_UNDER_FAST: upstream's switch would wrap the targets without the kit's boundaries)
Every route expects atom_attn engaged (dense-SDPA; upstream's default at inference, attention.py:409-454) and names cueq / ds4sci
n/a-upstream (dead code at the pin: attention.py:282, pairformer_layers.py:38). The kit ships no big mode: there is no big route.

UNSERVED is the third table: the three upstream computations a kit mode's levers do not drive, each refused BY NAME under exact | fast
(`NOT ACTIVE: mode=<m> cannot serve …`, exit 3, nothing of the run starts; `--mode off` — RFDIFFUSION3_OPT=off on upstream's own command
line — is the stock path that serves them):
  classifier-free guidance  `inference_sampler.use_classifier_free_guidance=true` (active with `cfg_scale != 1.0`: upstream model/RFD3.py:65-67;
                            rfdiffusion3.yaml:31-33, default off). Upstream then runs a SECOND, unconditioned denoiser pass per step on stripped
                            features (RFD3.py:95-97; patched inference_sampler.py, the `use_classifier_free_guidance` branch) — the roll-out
                            cache keys the pair bias `to_b(P_LL)` and the pooled features `downcast_c` on the module, one set per roll-out
                            (patched layers/attention.py `hoist_get`, RFD3_diffusion_module.py), and the captured step holds one feature set per
                            graph: the reference pass would be served the conditional tensors. `cfg_scale` / `cfg_t_max` / `cfg_features`
                            alone leave CFG off (upstream reads them only under the switch) and are served like any other token.
  low-memory tokenization   `low_memory_mode=true` (rfdiffusion3.yaml:67, default off; the engine exports RFD3_LOW_MEMORY_MODE=1 itself,
                            engine.py:202-205, and the model drops the resident all-atom pair track for a chunked embedder recomputed inside
                            every denoiser step, RFD3.py:43-56, inference_sampler.py `chunked_pairwise_embedder`, P_LL=None) — the pair-bias
                            hoist, the gather kernel and the captured step all read the resident P_LL.
  symmetry sampler          `inference_sampler.kind=symmetry` (SampleDiffusionWithSymmetry: its own per-step loop symmetrises the coordinates,
                            apply_symmetry_to_X_L) — the CUDA-graph roll-out drives the default sampler SampleDiffusionWithMotif only.
The design line is read for all three before activation (line_unserved: `design` on its composed line, the autoload finder on the process's own
argv); a spelling the line reader cannot see (hydra's dict / group syntax, a checkpoint's own config) is caught at the engine's initialize
from the built model's own decision (census.Observer.after_initialize → unserved_by_engine), before the first roll-out, with the same words
and the same exit code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from opt_core.modes import ModeError, ModeTable, mode_argument

MODES: Tuple[str, ...] = ("exact", "fast", "off")  # every command reads this list; _autoload.MODES restates it import-free (test_merge_locks)
DEFAULT_MODE = "fast"                                # `--mode` omitted and RFDIFFUSION3_OPT unset = fast for the package commands; exact (identical designs) and off (stock) are selected by name; unset on upstream's own command line = stock (the .pth installs nothing)
TABLE = ModeTable(modes=MODES, default=DEFAULT_MODE, unknown_message=lambda v: f"'{str(v).strip().lower()}' is not a mode ({'|'.join(MODES)})")
VARIANTS: Tuple[str, ...] = ("design",)
KIT_SWITCH = "RFD3_HOIST"                            # the kit's hoist lever switch (hoist.py:18)
KIT_FZT_SWITCH = "RFD3_FZT"                          # the fused SwiGLU transition switch (layers/layer_utils.py, the [RFD3_FZT] block: read once at that module's import)
KIT_GRAPH_SWITCH = "RFD3_CUDAGRAPH"                  # the CUDA-graph sampler switch (inference_sampler.py:610-611)
KIT_COMPILE_SWITCH = "RFD3_COMPILE"                  # [RFD3_COMPILE] inductor compile of upstream's four compile targets under the roll-out cache (hoist.py, the [RFD3_COMPILE] block; applied at the first roll-out)
KIT_GATHER_SWITCH = "RFD3_GATHER_ATTN"               # [RFD3_GATHER_ATTN] the atom-level index-set attention served by the shared core's gather kernel (gather.py; opt_core.kernels.gather_attn, strategy F5.gather_attn), installed by the kernels observer at the engine's initialize
KIT_TOKEN_SDPA_SWITCH = "RFD3_TOKEN_SDPA"            # [RFD3_TOKEN_SDPA] the token transformer's pair-bias attention through upstream's dense SDPA function (layers/attention.py use_dense_sdpa_pairbias)
KIT_INIT_CHUNK_SWITCH = "RFD3_INIT_CHUNK"            # [RFD3_INIT_CHUNK] the token initialiser's two all-atom-pair sinusoidal embeddings run over row blocks (layers/blocks.py, the [RFD3_INIT_CHUNK] block: read once at import): stock's bits, fp32 intermediates bounded
KIT_INIT_CHUNK_MODULE = "rfd3.model.layers.layer_utils"   # the patched module that reads KIT_INIT_CHUNK_SWITCH and carries init_chunk_describe() (blocks.py SinusoidalDistEmbed calls into it)
KIT_PARTS = ("pll", "downcast", "valid", "where", "dedup")   # the hoist's five parts (hoist.py PARTS), in the kit's order — no selector: the lever is all five
KIT_HOIST_MODULE = "rfd3.model.hoist"
KIT_FZT_MODULE = "rfd3.model.layers.layer_utils"      # reads RFD3_FZT at import; FZT / FZT_STATS / fzt_describe() are its evidence
KIT_GRAPH_MODULE = "rfd3.model.cudagraph_sampler"


@dataclass(frozen=True)
class KitMode:
    name: str
    env: Dict[str, str]                              # the delta exported before `import rfd3.model`
    tier: str                                        # exact | tolerance
    kit_class: str                                   # the kit's own label for the line
    default: bool = False
    doc: str = ""


KIT_MODES: Dict[str, KitMode] = {
    "exact": KitMode("exact", {KIT_SWITCH: "1", KIT_FZT_SWITCH: "1", KIT_GRAPH_SWITCH: "1", KIT_INIT_CHUNK_SWITCH: "1"}, "exact", "EXACT (type-A: identical designs)", default=DEFAULT_MODE == "exact",
                     doc="the hoist (its five parts, hoist.py PARTS), the fused SwiGLU transition served by opt_core (layers/layer_utils.py [RFD3_FZT]), and the CUDA-graph sampler (the eager denoiser step captured once per item shape and replayed: the same kernels on the same bytes)"),
    "fast": KitMode("fast", {KIT_SWITCH: "1", KIT_FZT_SWITCH: "1", KIT_GRAPH_SWITCH: "1", KIT_INIT_CHUNK_SWITCH: "1", KIT_COMPILE_SWITCH: "1", KIT_TOKEN_SDPA_SWITCH: "1", KIT_GATHER_SWITCH: "1"}, "tolerance",
                    "Tier 2 (deltas reported): CUDA-graph sampler + hoist + fused transition + inductor-compiled submodules + token attention on dense SDPA", default=DEFAULT_MODE == "fast",
                    doc="forward-pass band (tier-2) — faster than exact; the exact delta (hoist, fused transition, graph sampler) plus torch.compile (automatic dynamic shapes) of upstream's four compile targets inside the captured step (RFD3_COMPILE) and the token transformer's pair-bias attention through upstream's dense SDPA function (RFD3_TOKEN_SDPA) and the atom-level index-set attention through the shared core's fused gather kernel (RFD3_GATHER_ATTN, strategy F5.gather_attn: no dense (D,H,L,L) bias is built, the batch sizes upstream's dense-memory rule sends sparse stay served); its envelope and speed rows are measured outside this tree"),
}

from .routes import (COMPILE_REFUSED, COMPILE_UNDER_EXACT, COMPILE_UNDER_FAST, KERNEL_ACCELERATORS, ROUTES, Route,   # noqa: E402,F401 — the routes table lives in the core-free `routes`; served here for the kit routes' callers
                     route_of, route_refusal)


CFG_KEY = "inference_sampler.use_classifier_free_guidance"   # rfdiffusion3.yaml:31 (default False); CFG is active when true AND cfg_scale != 1.0 (upstream model/RFD3.py:65-67)
CFG_SCALE_KEY = "inference_sampler.cfg_scale"                # rfdiffusion3.yaml:33 (default 1.5): read only under the switch
LOWMEM_KEY = "low_memory_mode"                               # rfdiffusion3.yaml:67 (default False): the engine exports RFD3_LOW_MEMORY_MODE=1 itself (engine.py:202-205)
SAMPLER_KIND_KEY = "inference_sampler.kind"                  # rfdiffusion3.yaml (default "default"): "symmetry" selects SampleDiffusionWithSymmetry (inference_sampler.py ConditionalDiffusionSampler._registry)
SYMMETRY_KIND = "symmetry"
SYMMETRY_SAMPLER = "SampleDiffusionWithSymmetry"             # the class the symmetry kind builds; the graph roll-out drives SampleDiffusionWithMotif only (cudagraph_sampler.install)
STOCK_PATH = "run --mode off (RFDIFFUSION3_OPT=off on upstream's own command line) for the stock path"   # the one remedy every UNSERVED refusal names
UNSERVED: Dict[str, str] = {                                  # feature -> why the levers do not drive it (the NOT ACTIVE line's mechanism words; unserved_reason composes the line)
    "classifier-free guidance": ("upstream runs a second, unconditioned denoiser pass per step on stripped features and the kit's roll-out cache (RFD3_HOIST: the pair bias "
                                 "and pooled atom features, one set per roll-out) and captured step (RFD3_CUDAGRAPH: one feature set per graph) hold one conditioning per roll-out"),
    "low_memory_mode": ("upstream's memory-efficient tokenization drops the resident all-atom pair track (P_LL=None) and recomputes it in row blocks inside every denoiser step "
                        "(chunked_pairwise_embedder), which the pair-bias hoist, the gather kernel and the captured step all read"),
    "symmetry sampler": ("the CUDA-graph roll-out (RFD3_CUDAGRAPH) drives upstream's default sampler SampleDiffusionWithMotif; the symmetry sampler SampleDiffusionWithSymmetry "
                         "symmetrises the coordinates inside its own per-step loop (apply_symmetry_to_X_L), which the driver does not restate"),
}


def unserved_reason(mode: str, feature: str, asked: str) -> str:
    """The NOT ACTIVE reason for an UNSERVED feature: `mode=<m> cannot serve <feature> (<how it was asked>): <mechanism> — refused (exit 3); <STOCK_PATH>`."""
    return f"mode={mode} cannot serve {feature} ({asked}): {UNSERVED[feature]} — refused (exit 3); {STOCK_PATH}"


def cfg_active(switch, scale=None) -> bool:
    """Upstream's own classifier-free-guidance decision (RFD3.py:65-67): the switch true AND `cfg_scale`, when given, not 1.0. The ONE rule —
    the design line's tokens (cfg_requested) and the engine's composed config (census.unserved_by_engine) both ask here."""
    from .settings import is_true
    if not is_true(switch):
        return False
    if scale is None:
        return True
    try:
        return float(scale) != 1.0
    except (TypeError, ValueError):
        return True                                       # an unparsable scale is upstream's to reject; the switch is on


def cfg_requested(tokens_or_kv) -> bool:
    """True when the design line turns classifier-free guidance on (cfg_active over CFG_KEY / CFG_SCALE_KEY). Hydra's dict / group spellings are
    read from the engine instead (census.unserved_by_engine)."""
    from .settings import kv_of
    kv = kv_of(tokens_or_kv)
    return cfg_active(kv.get(CFG_KEY), kv.get(CFG_SCALE_KEY))


def lowmem_requested(tokens_or_kv) -> bool:
    """True when the design line turns upstream's low-memory tokenization on (LOWMEM_KEY true)."""
    from .settings import is_true, kv_of
    return is_true(kv_of(tokens_or_kv).get(LOWMEM_KEY))


def symmetry_requested(tokens_or_kv) -> bool:
    """True when the design line selects the symmetry sampler (SAMPLER_KIND_KEY = symmetry)."""
    from .settings import kv_of
    v = kv_of(tokens_or_kv).get(SAMPLER_KIND_KEY)
    return v is not None and v.strip().strip("'\"").lower() == SYMMETRY_KIND


def line_unserved(tokens, mode: str) -> Optional[str]:
    """The NOT ACTIVE reason when the design line asks upstream for a computation the kit mode's levers do not drive (UNSERVED: classifier-free
    guidance on, low_memory_mode on, the symmetry sampler), else None. `off` serves every line."""
    if mode == "off":
        return None
    if cfg_requested(tokens):
        return unserved_reason(mode, "classifier-free guidance", f"{CFG_KEY}=true")
    if lowmem_requested(tokens):
        return unserved_reason(mode, "low_memory_mode", f"{LOWMEM_KEY}=true")
    if symmetry_requested(tokens):
        return unserved_reason(mode, "symmetry sampler", f"{SAMPLER_KIND_KEY}={SYMMETRY_KIND}")
    return None


@dataclass(frozen=True)
class Resolution:
    mode: str
    variant: str
    env: Dict[str, str]                              # {} for off
    tier: Optional[str]
    kit_class: Optional[str]
    activation_point: str
    default: bool = False
    notes: Tuple[str, ...] = field(default_factory=tuple)


ACTIVATION_POINT = (f"the environment before `import rfd3.model` ({KIT_SWITCH} is read at the import of {KIT_HOIST_MODULE}, hoist.py:18; "
                    f"{KIT_FZT_SWITCH} at the import of {KIT_FZT_MODULE}; {KIT_GRAPH_SWITCH} at the sampler's construction, inference_sampler.py:610-611); "
                    "late activation is impossible by construction")


def normalize(mode: Optional[str]) -> str:
    m = (mode or "").strip().lower()
    return m or DEFAULT_MODE


def mode_of(cli: Optional[str], environ_value: Optional[str]) -> str:
    """The mode of a run: `--mode` when given, else RFDIFFUSION3_OPT, else the default (opt_core.modes.mode_argument over TABLE); an
    unknown word raises ModeError."""
    return mode_argument(cli, environ_value, TABLE)


def resolve(mode: Optional[str], variant: Optional[str] = None) -> Resolution:
    """A package mode -> its environment delta. Raises ValueError for anything that is not a mode.
    No GPU-memory-class `key` parameter, on purpose: the kit has no per-class table and no device gate, so a key could never change the
    delta — a mode is one delta on every GPU class the tree supports."""
    m = normalize(mode)
    v = (variant or VARIANTS[0]).strip().lower()
    if v not in VARIANTS:
        raise ValueError(f"'{v}' is not a variant ({'|'.join(VARIANTS)})")
    try:
        m = TABLE.check(m)
    except ModeError as e:
        raise ValueError(str(e)) from None
    if m == "off":
        return Resolution("off", v, {}, None, None, "none: stock, nothing exported (the pristine interpreter, stock_design.py)")
    km = KIT_MODES[m]
    return Resolution(m, v, dict(km.env), km.tier, km.kit_class, ACTIVATION_POINT, km.default)


def describe_env(env: Dict[str, str]) -> str:
    """The delta as the activation line spells it: `RFD3_HOIST=1,RFD3_FZT=1,RFD3_CUDAGRAPH=1,RFD3_INIT_CHUNK=1` (exact), the same plus
    `,RFD3_COMPILE=1,RFD3_TOKEN_SDPA=1,RFD3_GATHER_ATTN=1` (fast), or `none`."""
    return ",".join(f"{k}={v}" for k, v in env.items()) or "none"


def jit_cache_key() -> str:
    """torch<version sans local tag>-cu<CUDA version sans dot>-sm<compute capability digits>, e.g. torch2.13.0-cu130-sm90 — the JIT
    cache key the configs derive, in the core's spelling (opt_core.jit_cache.key) from parts read here. Raises (never a bare 'unknown'
    part) when torch is not importable, has no CUDA build, or the GPU capability query fails: a wrong-but-plausible key would silently
    blend JIT caches across incompatible stacks."""
    import torch
    from opt_core.jit_cache import key
    tv = torch.__version__.split("+")[0]
    if not torch.version.cuda:
        raise RuntimeError("jit_cache_key: torch has no CUDA build (torch.version.cuda is unset)")
    cc = torch.cuda.get_device_capability(0)          # raises when no GPU is visible; the cause is the driver's own message
    return key(version=tv, cuda=torch.version.cuda, cc=f"{cc[0]}.{cc[1]}")
