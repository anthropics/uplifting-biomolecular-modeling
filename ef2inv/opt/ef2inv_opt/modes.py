"""The mode table: the one place a mode is defined (README.md §Modes and CHANGES.md describe these rows).

A mode names exactly one composition of the kit: the value of ``EF2_FAST_KIT``, the design kit's switch. Every mode runs the one stock
cookbook file (stock/PINS.json "upstream"). On a kit arm the package sets the variable, puts the design kit's ``k/`` directory on the path and
installs the kit on the imported module from outside (fastkit.py: the featurisation cache, the kit-enable step ``ESMFold2Design._enable_fast_kit``
with the lever set behind each value, the state guard checked at every design step), then reads the kit's own printed line
(``fast kit: <switch>, checkpoint policy <ck> for <N> tokens``) back as evidence.

  off    stock: the upstream cookbook file in a subprocess proven clean of every kit variable, kit module and kit directory
         (stock_design.py env_proof). Numerics class: stock (reference).
  exact  EF2_FAST_KIT=exact — the exact lever set (fastkit.COMPOSITION["exact"]; MODES["exact"].levers names each lever and its k/ module):
         memory-planned checkpointing of the pair trunk over the stock compiled blocks, the cuEquivariance tile table, the skipped confidence
         head and deferred structure sample, the sampler's pair-bias hoist and denoise-step graph, the exact fused ESM-C rotary, the loop levers,
         the ESMC / pPPL / trunk CUDA graphs (the trunk pool only for complexes of at most fastkit.POOL_MAX_TOKENS tokens) and the featurisation
         cache. Pins the stock arm's two switches (pair stack unchunked, cuEquivariance backend). Numerics class: exact — bit-identical to stock
         under the det recipe on both arms (det.py); tier 2 at production numerics (stock's own scatter-add atomics are not run-to-run
         reproducible).
  fast   EF2_FAST_KIT=agk3 — the exact set plus the tier-2 kernel levers (TIER2_KERNEL_LEVERS: fused triangle multiplication fwd + frozen-weight
         bwd, the K-D3 transition fwd+bwd with its one-kernel forwards where the card serves them, the row-major frozen-LayerNorm backward, the
         no-save first pass of the checkpointed pair blocks, the bf16 confidence head, the inversion models' pair stack chunked at the fork's 64 —
         read on the no-grad confidence path only), the memory plan FILLING the card (ef2_bwd_ckpt rule ``budget``: pair blocks whose kept
         activations fit are kept, the rest checkpointed) and the trunk fwd+bwd CUDA-graph pool for complexes of at most fastkit.POOL_MAX_TOKENS
         tokens (ef2_stepgraph; it steps aside by name above that). The four HERO CRITICS (``app.hf_critic_models``, stock models the grad-mode
         kernels never touch) fold on the stock arm's two switches (Mode.critic_chunk_size / critic_kernel_backend) unless a --chunk-size /
         --kernel-backend flag covers every model. Numerics class: tier 2 (TIER2_NUMERICS); never bit-identical to stock. DEFAULT_MODE.
  big  EF2_FAST_KIT=big — fast's lever set at the memory floor: the memory plan PINNED to policy ``block`` (ef2_bwd_ckpt rule ``memory_floor``:
         every pair block of every grad pass checkpointed and recomputed in the backward; the plan's estimate and budget are still computed and
         printed), NO trunk fwd+bwd CUDA-graph pool at any complex size (its private pools hold the trunk's activations resident), and the levers of
         fastkit.BIG_MODE_OFF switched OFF BY NAME (``LEVER name=<lever> state=off … reason=mode:big``) because each keeps memory resident:
         the pPPL fwd+bwd graph, the ESMC-forward graph and the target-chain hoist over it, the sampler's denoise-step graph and pair-bias hoist
         (private pools per fold model), the fold/pLM overlap (the pLM term's working set alive through the trunk's backward). Everything else is
         fast's. Numerics class: tier 2, fast's digits (checkpointing, graph replay and the overlap are exact levers). The mode for the largest
         complexes and the smaller cards.

Refused by name (never a silent fallback): any other ``--mode`` (there is no lever axis: a lever lives inside one of the modes); ``--batch-size`` > 1
on a kit arm (settings.batch_size_problem).
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

DEFAULT_MODE = "fast"                      # the package default (tier 2, the recommended composition); `off`, `exact` and `big` are selected by name
ENV_MODE = "EF2INV_OPT"                    # the mode from the environment (run.sh: a --mode that disagrees with it is refused)
KIT_SWITCH = "EF2_FAST_KIT"                # the design kit's switch: set in a kit arm's environment, installed on the stock module by fastkit.install, read back by its k/ modules
DESIGN_KIT_DIRNAME = "ESMFOLD2_DESIGN_FAST_INFERENCE_KIT_v1"   # the carried design kit (opt/forward/): the lever modules k/ef2_*.py, their unit tests, the lever notes
KIT_MODULE_PREFIXES = ("ef2_",)            # the design kit's lever modules on the path (k/ef2_*.py)
KIT_PATH_MARKER = "ef2_autograd_kernels.py"   # a sys.path entry holding this file is the kit's k/ directory
STUBS_MARKER = os.path.join("modal", "__init__.py")   # a sys.path entry holding this file is a `modal` stubs directory: none may be on an arm's path (envproof)
SDK_STANDIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_absent_sdk_stub.py")   # the no-op stand-in the cookbook's `import modal` (its cloud SDK, absent here) resolves to in an arm
                                                                                            # process (registered in sys.modules by stock_design.prepare_arm_process; carried once here
                                                                                            # and nowhere as a package)


@dataclass(frozen=True)
class Mode:
    name: str
    kit_switch: Optional[str]              # EF2_FAST_KIT value; None = stock (the variable must be absent, nothing installed on the module)
    levers: List[str]                      # the kit's lever names, for reports only
    numerics_class: str
    tier: str
    what: str
    chunk_size: object = "shipped"          # the mode's set_chunk_size(v) on every loaded ESMFold2 model right after load: None = unchunked, N; "shipped" (internal word) = no call, the fork's default (64) untouched
    kernel_backend: object = "shipped"      # the mode's set_kernel_backend(v) on every loaded ESMFold2 model right after load: "cuequivariance" | "fused" | None; "shipped" (internal word) = no call, the cookbook's own call untouched
    critic_chunk_size: object = "shipped"   # the mode's set_chunk_size(v) on the HERO CRITICS only (app.hf_critic_models), where its every-model switch above is "shipped": fast / big = stock's None; "shipped" = no critic-scope call
    critic_kernel_backend: object = "shipped"   # the mode's set_kernel_backend(v) on the hero critics only, same rule: fast / big = stock's "cuequivariance"; an every-model value (exact's pins, a user flag) covers the critics and this yields to it, by name

    @property
    def is_kit(self) -> bool:
        """A kit arm (exact | fast | big): EF2_FAST_KIT set, the design kit's k/ on the path, fastkit.install on the module; False = off (stock as shipped)."""
        return self.kit_switch is not None

    def critic_switches(self, chunk_size: object = "shipped", kernel_backend: object = "shipped") -> Dict[str, object]:
        """The two switches the hero critics fold with, per key: the every-model value in force — the arm's ``--chunk-size`` / ``--kernel-backend``
        (the stock arm's settings, exact's pins, a user flag on fast / big: it covers the critics; ``scope`` "all") — else the mode's critic-scope
        value (fast / big: stock's none + cuequivariance; ``scope`` "critics"), else "shipped" (no call). ``scope`` names, per key, who set it."""
        out: Dict[str, object] = {}; scope: Dict[str, str] = {}
        for key, every, critic in (("chunk_size", chunk_size, self.critic_chunk_size), ("kernel_backend", kernel_backend, self.critic_kernel_backend)):
            if every != "shipped":
                out[key], scope[key] = every, "all"
            elif critic != "shipped":
                out[key], scope[key] = critic, "critics"
            else:
                out[key], scope[key] = "shipped", "shipped"
        out["scope"] = scope
        return out


# Stock = the cookbook with the pair stack UNCHUNKED and the cuEquivariance kernel backend on every loaded model (STOCK.md §How stock is run):
# The stock arm's two documented speed settings of upstream's API — `design --mode off --chunk-size none --kernel-backend cuequivariance` =
# set_chunk_size(None) + set_kernel_backend('cuequivariance') on every loaded ESMFold2 model right after load; `off` itself makes NO setter call
# (the cookbook exactly as shipped: the fork's pair-stack chunk 64, the loader's own fused backend call); `exact`'s own values are these two (it is
# bit-identical to that stock configuration under the det recipe) and a differing flag is applied as the stock arm applies it, the one exact lever with
# nothing to attach to stepping aside by name (settings.EXACT_SWITCH_NOTES); `fast` / `big` keep the cookbook's
# own choices on the INVERSION models under their kernels (no every-model call) and put the HERO CRITICS on the stock arm's pair (critic_chunk_size /
# critic_kernel_backend: the critics are stock models folding under grad, where the loader's `fused` choice dispatches the reference pair stack —
# the stock arm's unchunked cuEquivariance kernels are the fastest correct critic fold); a flag there is a recorded user override on EVERY model,
# critics included (a NOTE line, opt_manifest.json `overrides`; the critics' MODEL-SWITCH line then says scope=all).
STOCK_CHUNK_SIZE = None
STOCK_KERNEL_BACKEND = "cuequivariance"
CRITIC_MODELS_ATTR = "hf_critic_models"      # the cookbook app's dict of HERO CRITIC models (ESMFold2Design.load, stock file l.1304-1309; four models loaded apart from the inversion models): the critic scope
CRITIC_SWITCH_WORDS = "hero critics on the stock arm's switches (set_chunk_size(None) + set_kernel_backend('cuequivariance') on app.hf_critic_models right after load; a --chunk-size / --kernel-backend flag covers them instead)"

TIER2_KERNEL_LEVERS: List[str] = ["fused triangle multiplication fwd + frozen-weight bwd (ef2_trimul fused)", "K-D3 transition fwd+bwd (agk transition=refround_lean)",
                   "K-D3's forward on the carried one-kernel sm_90a transition where the card serves it, K-D3's own forward by name elsewhere (ef2_t16_transition)",
                   "K-D3's own forward, where it serves, with the W12 projection + SwiGLU as one Triton kernel on a card with an entry (ef2_kd3_gemmswiglu: sm_80; not asked where t16 serves)", "row-major frozen-LayerNorm backward (ef2_fused_ln)",
                   "no-save first pass of the checkpointed pair blocks on the opt_core forward-only triangle-multiplication row (ef2_trimul_nosave: tx_sm90a > esm_v61 > esm_v5_fwd > v4, the first row the stack serves, asked by name; sm_90)",
                   "bf16 confidence head on confidence steps (ef2_bf16_confidence)", "pair stack chunked at the fork's 64 (set_chunk_size(64) on the inversion models: read on the no-grad confidence path only, the grad-mode kernels never chunk)", CRITIC_SWITCH_WORDS]   # the tier-2 kernel levers fast and big share (fastkit.COMPOSITION agk3 / big: the same words)
TIER2_NUMERICS = "tier 2: rounding points changed (bf16 contraction) + reordered accumulation (K-D3) + bf16 confidence head; checkpointing is an exact lever; never bit-identical to stock"

MODES: Dict[str, Mode] = {
    "off": Mode("off", None, [], "stock (reference)", "stock",
                "the upstream cookbook file in a subprocess proven clean of every kit variable, module and directory, exactly as shipped; --chunk-size / --kernel-backend call upstream's setters (the stock arm's documented speed settings: none + cuequivariance)"),
    "exact": Mode("exact", "exact",
                  ["memory-planned activation checkpointing (ef2_bwd_ckpt: agk checkpoint policy from free memory)", "cuEquivariance triangle-multiplication tile table (ef2_trimul cueq_tiles)",
                   "skip-unused-confidence (agk.enable_skip_unused_confidence)", "deferred structure sample (ef2_lazy_structure)",
                   "diffusion sampler: pair bias hoisted + one denoise step as a CUDA graph, on every fold model (ef2_pairbias_attn hoist, ef2_sampler_graph)",
                   "exact fused rotary on the ESMC trunks (ef2_esmc_rope)", "fold entry re-plumbed: one host transfer, ESMC feature pass before featurisation, target featurisation spliced (ef2_loop_prep)",
                   "pseudo-perplexity body without host syncs (ef2_loop_pppl)", "pseudo-perplexity term overlapped with the fold on a side stream (ef2_esmc_overlap)",
                   "ESMC-6B forward CUDA graph (ef2_esmc_graph)", "ESMC target-chain hidden-state hoist: the feature pass for the binder's rows only (ef2_esmc_hoist)", "pPPL loss fwd+bwd CUDA graph (ef2_pppl_graph)",
                   "trunk fwd+bwd CUDA-graph pool for complexes <= fastkit.POOL_MAX_TOKENS tokens (ef2_stepgraph, 2 slots)", "featurisation cache (fastkit.py, around prepare_esmfold2_tensors)"],
                  "exact: bit-identical to stock under the det recipe on both arms (det.py); tier 2 at production numerics", "exact / 2",
                  "the kit's exact lever set (EF2_FAST_KIT=exact) installed on the stock cookbook module; pinned pair stack unchunked + cuEquivariance kernel backend (the stock arm's settings)",
                  STOCK_CHUNK_SIZE, STOCK_KERNEL_BACKEND),
    "fast": Mode("fast", "agk3",
                 [*TIER2_KERNEL_LEVERS,
                  "memory-planned activation checkpointing filling the card (ef2_bwd_ckpt rule budget: the pair blocks whose kept activations fit are kept, the rest checkpointed)",
                  "trunk fwd+bwd CUDA-graph pool for complexes <= fastkit.POOL_MAX_TOKENS tokens (ef2_stepgraph, 2 slots), stepped aside by name above (the memory plan keeps what fits: the reach regime)",
                  "+ the exact set (ef2_trimul cueq_tiles process-wide, serving the hero critics' folds: the fused kernel replaces the inversion models' calls)"],
                 TIER2_NUMERICS, "2",
                 "the kit's recommended tier-2 composition (EF2_FAST_KIT=agk3): the fused kernels + chunked pair stack + the memory plan filling the card (pair blocks kept where they fit) + the trunk graph pool at or below 256 tokens (above it the plan keeps what fits); hero critics on the stock arm's switches (chunk none + cuequivariance) unless --chunk-size / --kernel-backend is passed (then every model takes the flag)",
                 critic_chunk_size=STOCK_CHUNK_SIZE, critic_kernel_backend=STOCK_KERNEL_BACKEND),
    "big": Mode("big", "big",
                  [*TIER2_KERNEL_LEVERS,
                   "memory plan pinned to its floor: every pair block of every grad pass checkpointed (ef2_bwd_ckpt rule memory_floor: policy block, kept 0; estimate and budget still printed)",
                   "NO trunk fwd+bwd CUDA-graph pool at any complex size (ef2_stepgraph off by the mode's rule: its private pools hold the trunk's activations resident)",
                   "switched OFF by name for their measured memory (fastkit.BIG_MODE_OFF; LEVER state=off reason=mode:big): the pPPL fwd+bwd graph (ef2_pppl_graph), the ESMC-forward graph (ef2_esmc_graph) and the target-chain hoist (ef2_esmc_hoist), the sampler's denoise-step graph (ef2_sampler_graph) and pair-bias hoist (ef2_pairbias_attn), the fold/pLM overlap (ef2_esmc_overlap)",
                   "+ the rest of the exact set (loop levers, fused ESM-C rotary, deferred structure sample, skipped confidence head, featurisation cache; ef2_trimul cueq_tiles process-wide, serving the hero critics' folds: the fused kernel replaces the inversion models' calls)"],
                  TIER2_NUMERICS + "; fast's digits (checkpointing and graph replay are exact levers)", "2",
                  "fast's lever set minus the levers with a measured memory cost (EF2_FAST_KIT=big): the memory plan pinned to its floor (policy block: every pair block checkpointed, reason=memory_floor), no trunk graph pool at any size, and the pPPL / ESMC / sampler graphs, the ESMC and pair-bias hoists and the fold/pLM overlap off by name (reason=mode:big) — peak memory at or below the earlier releases' big and, from 500 tokens, the stock arm's; the fused kernels, the no-save first pass, chunked pair stack and the loop levers as fast; hero critics on the stock arm's switches (chunk none + cuequivariance) unless --chunk-size / --kernel-backend is passed (then every model takes the flag)",
                  critic_chunk_size=STOCK_CHUNK_SIZE, critic_kernel_backend=STOCK_KERNEL_BACKEND),
}
ALIASES: Dict[str, str] = {}                # no mode has an alias as shipped; the mechanism stays: an alias resolves BY NAME to its row and the ACTIVE line names the word given (alias=<word>)
MODES.update({alias: MODES[target] for alias, target in ALIASES.items()})
MODE_NAMES: List[str] = sorted({m.name for m in MODES.values()}, key=["off", "exact", "fast", "big"].index)   # the modes proper, in tier order (an alias is a word for one of these, never a row of its own)


def mode_word(name: Optional[str]) -> str:
    """The mode word as given: ``name``, else the environment's ``EF2INV_OPT``, else DEFAULT_MODE (not yet resolved: it may be an alias)."""
    return name or os.environ.get(ENV_MODE) or DEFAULT_MODE


def alias_word(name: Optional[str]) -> Optional[str]:
    """The alias a run was invoked by — the given word (mode_word) when it is an ALIASES key (ALIASES is empty as shipped), else None (the mode's own
    name, or an unknown word resolve refuses). The ACTIVE line's ``alias=`` token; absent when None."""
    w = mode_word(name)
    return w if w in ALIASES else None


def resolve(name: Optional[str]) -> Mode:
    """The mode for a name (``None`` = the environment's ``EF2INV_OPT``, else DEFAULT_MODE); an alias (ALIASES, empty as shipped) resolves to
    its mode. Unknown names are refused by name."""
    n = mode_word(name)
    if n in MODES:
        return MODES[n]
    raise ValueError(f"mode {n!r} refused: unknown; modes are {sorted(MODES)}")


def kit_root(model_opt: Optional[str] = None) -> str:
    """``opt/forward/`` of this tree (``MODEL_OPT`` = the ef2inv/ directory, configs/h100.env)."""
    root = model_opt or os.environ.get("MODEL_OPT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(root, "opt", "forward")


def kit_paths(model_opt: Optional[str] = None) -> Dict[str, str]:
    fwd = kit_root(model_opt)
    return {"forward": fwd,
            "design_kit": os.path.join(fwd, DESIGN_KIT_DIRNAME),
            "k_dir": os.path.join(fwd, DESIGN_KIT_DIRNAME, "k")}


def kit_presence(model_opt: Optional[str] = None) -> Dict[str, object]:
    """The design kit's presence -- a cheap pre-flight fact, not a bytes check (nothing in this package
    re-hashes the kit's files). ``root_present``: the design kit directory (kit_paths() ``design_kit``) exists.
    ``entry_present``: its entry module (``k/`` + KIT_PATH_MARKER, the same file each sys.path check of a kit mode keys on) is there -- a
    half-copied or wrong tree fails this before anything imports it."""
    kp = kit_paths(model_opt)
    root, entry = kp["design_kit"], os.path.join(kp["k_dir"], KIT_PATH_MARKER)
    return {"root": root, "root_present": os.path.isdir(root), "entry": entry, "entry_present": os.path.isfile(entry)}


JIT_CACHE_VAR = "EF2INV_JIT_CACHE"                 # the deployment switch (configs/h100.env; read by the launcher BEFORE the arm's environment is stripped)
JIT_LOCAL_ROOT_VAR = "EF2INV_JIT_LOCAL_ROOT"      # per-box form: the box-local root (default: a fresh temporary directory per arm process)
JIT_CACHE_FORMS = ("shared", "per-box")           # frozen:<sha256> (a read-only cache artifact cited by sha) is a named form this tree does not ship: refused
JIT_SHARED_ROOT_VAR = "MODEL_OPT_JIT_ROOT"    # the optional shared JIT-cache root, <root>/<stack key>/{triton,inductor} — the operator's variable, no default: unset, configs/h100.env exports no cache dir (an explicit EF2INV_JIT_CACHE=shared without it is refused by name)
JIT_DIR_VARS = ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR")


def jit_cache_form(value: Optional[str]) -> str:
    """``shared`` | ``per-box``; anything else (``frozen:<sha>`` included) raises by name."""
    v = (value or "shared").strip()
    if v not in JIT_CACHE_FORMS:
        raise ValueError(f"{JIT_CACHE_VAR}={v!r}: the shipped forms are {' | '.join(JIT_CACHE_FORMS)} "
                         f"(frozen:<sha256>, a read-only cache artifact cited by sha, is named but not shipped: no frozen cache is shipped)")
    return v


def jit_cache_dirs(form: str, *, key: Optional[str] = None, local_root: Optional[str] = None, shared_root: Optional[str] = None) -> Dict[str, str]:
    """The two cache directories of a form. shared: ``<shared_root>/<key>/{triton,inductor}`` (``shared_root`` = the operator's
    ``MODEL_OPT_JIT_ROOT``, a directory kept across runs and shareable between machines; keyed by the running stack — a speed lever, and a state
    channel between runs: kernel/autotune choices written by one run are read by the next). per-box: ``<local_root>/{triton,inductor}`` under a
    box-local root (a fresh temporary directory when none is given) — nothing read from or written to the shared root; the det-arm
    form (every ``--det >= 1`` arm)."""
    form = jit_cache_form(form)
    if form == "shared":
        if not key:
            raise ValueError("the shared JIT cache is keyed by the running stack: key required (modes.jit_cache_key)")
        if not shared_root:
            raise ValueError(f"{JIT_SHARED_ROOT_VAR} is not set — see README.md §Variables (the shared JIT cache's root, a directory kept across runs; <root>/<key>/{{triton,inductor}})")
        root = f"{shared_root.rstrip('/')}/{key}"
    else:
        import tempfile
        root = local_root or tempfile.mkdtemp(prefix="ef2inv-jitcache-")
    return {"TRITON_CACHE_DIR": f"{root}/triton", "TORCHINDUCTOR_CACHE_DIR": f"{root}/inductor"}


def jit_cache_state(environ) -> Dict[str, object]:
    """What a process's environment says about its JIT cache: the switch, the shared root (``MODEL_OPT_JIT_ROOT``, None when unset), the
    two dirs, and the form the dirs imply (``shared`` when a dir lies under the shared root, ``per-box`` when both lie elsewhere or no
    shared root is named, ``unset`` when neither dir is set)."""
    dirs = {v: environ.get(v) for v in JIT_DIR_VARS if environ.get(v)}
    shared_root = (environ.get(JIT_SHARED_ROOT_VAR) or "").rstrip("/") or None
    under = [d for d in dirs.values() if shared_root and (d.rstrip("/") + "/").startswith(shared_root + "/")]
    implied = "unset" if not dirs else ("shared" if under else "per-box")
    return {"switch": environ.get(JIT_CACHE_VAR), "shared_root": shared_root, "dirs": dirs, "form": implied}


def jit_cache_key() -> str:
    """The JIT cache key of the running stack, ``torch<version sans local tag>-cu<CUDA sans dot>-sm<capability digits>`` — the shared core's
    rule (``opt_core.jit_cache.key``: resolved without importing torch; a part that cannot be established raises ``StackKeyUnknown`` by name)."""
    from opt_core import jit_cache
    return jit_cache.key()
