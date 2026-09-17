"""kernels — the KERNELS census: which accelerator path a pass ACTUALLY ran, read from the objects bound at upstream's call sites and
from the accelerators' own runtime signals, printed once per pass on every route; enforced at the pass's end on the kit routes (exact, fast),
reported on the stock routes (stock, default: routes.Route.enforce).

rfd3 at the pin has three live accelerator surfaces (guarded: compile, the dense atom attention, apex's FusedRMSNorm binding) and two dead ones (named `n/a-upstream` on the line):

  compile    torch.compile (inductor; ``dynamic=False`` on the stock route, automatic dynamic shapes under the kit's lever on fast) of the diffusion module's ``encoder``, ``diffusion_token_encoder``,
             ``diffusion_transformer``, ``decoder`` — upstream's ``compile_model`` key (rfd3/configs/inference_engine/rfdiffusion3.yaml:70-71,
             default False; rfd3/engine.py:212-269, applied in ``RFD3InferenceEngine.initialize()``). Read: each target attribute of the
             diffusion module located the way engine.py:236-253 locates it IS a ``torch._dynamo.eval_frame.OptimizedModule``; dynamo not
             disabled (``torch._dynamo.config.disable``); after the run, dynamo's own counters (``torch._dynamo.utils.counters``: graphs
             compiled, frames ok/total) and its compile seconds.
  atom_attn  the atom pair-bias attention: under the kit's ``gather_attn`` lever (RFD3_GATHER_ATTN, mode fast) the shared core's fused gather
             kernel — word ``engaged:gather_attn(<kernel>;F5.gather_attn)[served=<n>,stock=0]`` when every atom call was served (gather.py;
             some calls refused by name reads ``partial:gather_attn(stock=<m>,stock_by=<reason:n+…>)[served=<n>]`` — kind partial —, a switch that
             could not install ``fallback:gather-unavailable(<reason>)``, one that never served ``fallback:gather-never-served(stock=<m>)``: all three
             refused on fast, which expects atom_attn engaged);
             otherwise the dense-SDPA atom pair-bias attention (``F.scaled_dot_product_attention``, rfd3/model/layers/attention.py:457-505; default ON
             at inference, decided per process by ``use_dense_sdpa_pairbias``, attention.py:409-454, and logged once, :397-406,
             ``Atom attention path: DENSE-SDPA|SPARSE (<reason>) [<shape>]``). Read: the static part of the decision after ``initialize()``
             (``RFD3_DENSE_SDPA_ATTENTION``, ``RFD3_LOW_MEMORY_MODE``: a SPARSE decided there is refused before the first step) and
             upstream's own path record, trapped from its logger at the first attention call (the data-dependent part: device, L vs k, free
             memory); the bound ``dense_sdpa_pairbias_attention`` names the implementation.
  cueq       cuEquivariance fused pair-bias attention: n/a by upstream source — ``use_kernel = False`` hardcoded (attention.py:282; the
             import guard :27-37 and the branch :290-312 are dead code).
  ds4sci     DeepSpeed DS4Sci evoformer attention: n/a by upstream source — ``self.use_deepspeed_evo = False`` hardcoded
             (rfd3/model/layers/pairformer_layers.py:38; the branch :75-76 raises NotImplementedError).

The fifth word, ``rmsnorm=``, is which RMSNorm class upstream bound: upstream imports NVIDIA apex's ``FusedRMSNorm`` for every ``RMSNorm``
of the network when ``apex.normalization.fused_layer_norm`` is importable and binds ``torch.nn.RMSNorm`` otherwise, with its own warning
(rfd3/model/layers/layer_utils.py:13-23, the module attribute ``RMSNorm_``). Read: the class of every RMSNorm module of the built network
(the net upstream's walk reaches, engine.py:236-253) and the module attribute; ``<word_r>`` is one of
``engaged:apex.FusedRMSNorm(<n>-modules[+ext])@apex<version>`` (``+ext``: the ``fused_layer_norm_cuda`` extension is loaded in the process) |
``fallback:torch.nn.RMSNorm(<n>-modules)@torch<version>`` | ``engaged:apex.FusedRMSNorm(bound;modules-unread[+ext])@apex<version>`` /
``fallback:torch.nn.RMSNorm(bound;modules-unread)@torch<version>`` (no RMSNorm module on the walk: the module binding alone names the class) |
``fallback:mixed(apex=<a>,torch=<t>)@apex<version>,torch<version>`` | ``unread`` (no network and no binding to read) | ``unread:<Exception>``
(the read itself raised). The pinned stack carries apex (STOCK.md "Pinned stack"), so every route expects ``engaged`` (routes.ROUTES):
a torch-bound pass on a kit route is refused by the REQUIRE guard like any other fallback — ``rmsnorm=fallback:…, expected engaged; exit 5``
right after ``initialize()`` — and a word still unread at the pass's end is refused there; on the stock routes the same fact is a
``KERNELS report-only …`` line and the run proceeds, as upstream's does.

The line (stderr, once per pass; ``route=`` is the cross-engine grep token)::

  [rfdiffusion3-opt] KERNELS route=<stock|default|exact|fast> mode=<off|exact|fast> B=<n>
                     compile=<word> atom_attn=<word> cueq=<word> ds4sci=<word> rmsnorm=<word_r>
                     [compile_s=<sec> inductor_cache=<cold|warm(dir)> graphs=<n> frames=<ok>/<total> graph_breaks=<n>]
                     torch=<torch.__version__> cuda=<torch.version.cuda> tf32_matmul=<True|False> tf32_cudnn=<True|False> alloc_conf=<$PYTORCH_CUDA_ALLOC_CONF|unset>
                     apex=<absent|present:<version>> ckpt=<basename>@<sha256[:12]>

with ``<word>`` ∈ ``engaged:<impl>@<version>`` | ``off-by-route:<reason>`` | ``absent`` | ``fallback:<reason>`` | ``n/a-upstream:<reason>``
(``unread`` only on a refusal printed before the fact could be read). ``B`` is the engine's live ``diffusion_batch_size``
(``engine.transform_overrides``, foundry inference_engines/base.py:106, read at ``initialize()``; the line's token before that). The bracketed
compile fields print on a compiled pass only; the stack facts after them print on every pass, read at the pass's end and never guarded:
``torch``/``cuda`` from the imported torch (``absent`` when the process imported none), ``tf32_matmul`` = ``torch.backends.cuda.matmul.allow_tf32``
and ``tf32_cudnn`` = ``torch.backends.cudnn.allow_tf32`` as found then (read-backs from inside the process, after upstream's own setup ran:
what the sampler computed with), ``alloc_conf`` the allocator variable's value, ``apex`` by ``importlib.util.find_spec`` (the
distribution's presence — never imported here; which class the network's norms ARE is ``rmsnorm=`` above), ``ckpt`` the checkpoint the engine actually loaded (``engine.ckpt_path`` after upstream's
registry resolution, base.py:76-95) with the first 12 hex digits of its sha256 (``registry.checkpoint_sha256``, the digest cache; ``unread``
when no engine initialized). The expected word KIND per route is the routes table
(``routes.ROUTES``, the single source of truth: stock expects compile engaged, default / exact / fast expect it off-by-route, every route
expects atom_attn engaged). On the kit routes (routes.Route.enforce) the REQUIRE guard compares kinds and REFUSES the pass by name —
``NOT ACTIVE: KERNELS refused on route <r> — <accel>=<word>, expected <kind>; exit 5`` — as early as the fact is known: a compile the route expects that did not wrap the four bound
submodules (or a dynamo that is disabled), and an atom attention statically bound for SPARSE, right after ``initialize()`` (before the first
diffusion step); a SPARSE decided at the first attention call, there (the pass unwinds with exit 5 out of upstream's own logging call);
a SPARSE pre-flight decision on a gated route (stock, fast), at the first denoising step of that shape (ATTN_PREFLIGHT below); anything
still unread or degraded after the run (no path record, no graph compiled, no pre-flight read on a gated route), at the pass's end with exit 5. Graph breaks and
frames dynamo skips are inductor's partial-graph operation on upstream's own modules at the pin: named on the line, not refused. There is
no allowance switch on a kit route: it runs the accelerators it names or does not run. On the stock routes (stock, default) every such miss
is a ``KERNELS report-only on route <r>: <accel>=<word>, expected <kind> — upstream's run proceeds`` line and the run keeps its own exit
code; nothing stops there, and a sparse path — upstream's own rule at that batch × length, or the caller's own switch
(``RFD3_DENSE_SDPA_ATTENTION=0``, ``low_memory_mode=True``) — is named there, not judged. THE STOCK ROUTES INSTALL NOTHING BUT LOGGING
HANDLERS (``Observer.install``: routes stock | default = upstream in the pristine interpreter): no upstream callable is wrapped, shimmed or
re-bound and no import hook is set in a stock process; every census word there is READ — off upstream's own log records (the compile record
engine.py:265-269 / :247-251, the RMSNorm binding record layer_utils.py:16 / :19-21, the first-decision path record attention.py:397-406, the
per-roll-out clock engine.py:402) or off the process's environment and torch's globals before the hand-over (stack facts) —, the batch and the
checkpoint are the line's own tokens, and the ATTN_PATH / ATTN_PREFLIGHT lines below are the KIT routes' (nothing counts calls or probes a
step in a stock process).

Beside the census, the same observer prints on the KIT routes, all on stderr and flushed at once:

* per item — one pair of lines per diffusion batch of the pass, inside the handler of upstream's own batch clock record
  (``Finished inference batch in <s> seconds.``, engine.py:402), ``k`` counting from 0::

    [rfdiffusion3-opt] PEAK item=batch:<k> alloc_gib=<torch.cuda.max_memory_allocated()/2**30:.2f> reserved_gib=<torch.cuda.max_memory_reserved()/2**30:.2f>

  ``clock_s`` is upstream's own digits, verbatim (the text the record carries, not a re-formatted float); ``B`` the line's ``B``. The PEAK line
  is the allocator peak over the item (reset after ``initialize()`` for batch 0, at each clock record for the next; nothing without a CUDA
  torch). Recorded under ``items`` and ``peaks`` in the census record.
* per pass, right after the KERNELS line — the atom-attention path census and the scope of its counts::

    [rfdiffusion3-opt] ATTN_PATH atom_dense=<n> atom_sparse=<m> token_full=<k> compiled=<0|1> calls_uncounted=<0|1>

  counted by a transparent shim on ``rfd3.model.layers.attention.use_dense_sdpa_pairbias`` (the module attribute upstream's own call site
  looks up, attention.py:340-342; installed when the module is imported, arguments and return value passed through unchanged, exceptions
  propagate): each interpreter-executed call is one of ``token_full`` (called with ``full`` truthy — the token-level transformer,
  RFD3_diffusion_module.py:339 → blocks.py:632, which runs the dense masked einsum ``pairbias_attention_``; upstream's own record labels it
  ``SPARSE (full=True)``), ``atom_dense`` (returned True: ``F.scaled_dot_product_attention``) or ``atom_sparse`` (returned False with ``full``
  falsy: the gather path — upstream decides it per call from ``torch.cuda.mem_get_info``, attention.py:442-452, and logs only the first
  decision of the process). ``compiled=1`` when upstream's torch.compile is engaged on the pass (route stock): dynamo takes the decision
  while it traces the compiled children and freezes it into each graph — those calls execute no interpreter code afterwards and are not
  counted (the shim steps aside while dynamo traces, so it adds no guard and no graph break). ``calls_uncounted=1`` when calls ran outside
  the interpreter: compiled, or the graph sampler (mode fast) replayed captured steps — the counts are then the eager warm-up's and the
  capture's. Written, atomically, with the pre-flight
  decisions in the census record ``attn_path`` (``opt_manifest.json`` ``census.attn_path``: ``{"atom_dense", "atom_sparse", "token_full", "compiled", "calls_uncounted", "pin_hits", "pin_misses", "preflight": [...]}``,
  ``out_dir`` from the design line) and recorded under ``attn_path``. Report only: the guard reads none of the counts.
* per distinct roll-out shape — the PRE-FLIGHT atom-attention decision, at the first denoising step of every (D, L) the pass runs (D the
  diffusion batch, L the design's atoms), before any attention call of that shape — so before dynamo traces (stock) and before the graph
  sampler captures (fast)::

    [rfdiffusion3-opt] ATTN_PREFLIGHT route=<r> B=<D> n_atoms=<L> L=<tokens|unread> atom_sparse=<0|1> free_gib=<torch.cuda.mem_get_info free/2**30:.3f> est_dense_gib=<2·D·H·L²·2/2**30:.3f> pinned=<0|1>

  (``n_atoms`` the atom count the decision is taken at — X_noisy_L's L —, ``L`` the token count, residues for a protein: upstream's own
  ``f["atom_to_token_map"].max() + 1``, RFD3_diffusion_module.py:193-195, ``unread`` when the step's features do not carry the map.)

  On the kit routes (routes.ROUTES ``attn_pin``: exact, fast) that decision is AUTHORITATIVE — ``pinned=1``: every (D, L, k, H) the pre-flight
  evaluated (k the diffusion module's ``n_attn_keys``, H each atom-attention head count) is written to a pin table the decision shim answers
  from, so the model's own queries of ``use_dense_sdpa_pairbias`` for those keys — the graph sampler's warm-up and capture, eager roll-outs,
  compiled frames — return the pre-flight's value without re-reading free memory (upstream's rule at the call would see the step's activations
  live and can flip to sparse at a shape compiled upstream, which evaluates the same rule at trace time before activations, runs dense). A pinned
  dense path that does not fit raises torch's out-of-memory error out of the attention call: a named failure, never a fallback to sparse.
  Upstream's force-sparse switches (RFD3_DENSE_SDPA_ATTENTION=0, RFD3_LOW_MEMORY_MODE=1) make the pre-flight itself say sparse (the kit's modes refuse
  them before that). The ``attn_path`` record carries ``pin_hits`` / ``pin_misses`` (atom calls answered from the
  table / evaluated per call while the table held other keys) and each pre-flight entry its ``decisions`` per head count and ``pinned``.

  The observer wraps the diffusion module's ``forward`` on the instance (the module the samplers call per step; upstream compiles its four
  children, never it) and there calls upstream's OWN decision rule once, eagerly, unshimmed, on zero-byte tensors of shape (D, L, ·) /
  (·, ·, k) on the step's device with H the largest ``n_head`` among the atom-level attention modules under ``encoder`` / ``decoder`` and k the
  module's ``n_attn_keys`` — the rule reads shapes, the device, the environment and the free memory of that moment, nothing else, and holds
  no state (on the kit's patched file it is ``_use_dense_sdpa_pairbias_stock``; the public entry there memoises per shape under the graph
  sampler and is never what the probe calls, so the model's own calls decide exactly as without it); upstream's once-per-process record is
  left to the model's first real call. ``est_dense_gib`` is upstream's own estimate (attention.py:445) that its
  0.35 × free rule compares. On the KIT routes (``routes.ROUTES[r].preflight_gate``: exact, fast — the captured / compiled dense step cannot serve
  upstream's sparse path) a sparse pre-flight REFUSES the pass there — ``NOT ACTIVE: KERNELS refused on route <r> — attn_preflight=sparse(B=…,L=…,
  free_gib=…,est_dense_gib=…), expected dense; exit 5`` (``sparse(B=…,n_atoms=…,…)``) — and a gated pass whose roll-outs ran with no pre-flight read is refused at its end
  (``attn_preflight=unread(<why>)``); the stock routes probe no step and print no such line (upstream takes the path its own rule picks; its path
  record names it in the ``atom_attn`` word). Recorded under
  ``attn_path.preflight`` (route, B, L, k, H, atom_sparse, decisions per head count, pinned, free_gib, est_dense_gib, device) and ``attn_preflight_lines``.

Every line goes through the package's one printer (``_emit.emit``): at a line start whatever upstream's progress bars left on the stream,
flushed at once.

One observer per process (``observe(route, mode, tokens)``), installed by ``stack.activate`` on the kit routes (``design --mode exact|fast``,
``RFDIFFUSION3_OPT=<mode> rfd3 design …``) and by ``stock_design.main`` in the pristine interpreter's child on the stock routes
(``design --mode off [compile_model=true]``) after its environment proof. On the STOCK routes the observer is a logging handler on upstream's
three loggers (``rfd3.engine``, ``rfd3.model.layers.attention``, ``rfd3.model.layers.layer_utils``; their level is set to INFO so the records
exist whatever the root configuration) and nothing else. On the KIT routes a meta-path hook lets ``rfd3.engine`` execute and wraps
``RFD3InferenceEngine.initialize`` with a read-only post-call (the engine object is otherwise unreachable: upstream builds it inside its own
CLI, rfd3/run_inference.py:38-40), lets ``rfd3.model.layers.attention`` execute and puts the counting shim on its decision function, and the
handler on upstream's loggers (``rfd3.engine``, ``rfd3.model.layers.attention``) traps the path record, the compile-skipped warning (engine.py:247-251) and the per-roll-out clock (``Finished
inference batch in <s> seconds.``, engine.py:402). A runner outside this package makes the same two calls: ``observe(...)`` before upstream's
entry point, ``finish(rc)`` after it. Standard library at import; torch is read through ``sys.modules`` only after upstream imported it.
"""
from __future__ import annotations

import logging
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

from ._autoload import EXIT_NOT_ACTIVE, TAG                # the kit's tag and NOT ACTIVE code, import-free (this module loads in the pristine interpreter too)
from ._emit import emit as _emit

PREFIX = f"[{TAG}]"
EXIT_KERNELS = 5                                            # the KERNELS refusal: an expected accelerator absent / fell back on a kit route, or the pre-flight dense bound (report.EXIT_KERNELS re-exports it)
ACCELERATORS: Tuple[str, ...] = ("compile", "atom_attn", "cueq", "ds4sci", "rmsnorm")   # = routes.KERNEL_ACCELERATORS (test_census locks the two)
STOCK_OBSERVATION = "records-only(rfd3.engine, rfd3.model.layers.attention, rfd3.model.layers.layer_utils loggers; no upstream callable wrapped)"   # what the census installs in a STOCK process (routes stock | default): Observer.install
KINDS: Tuple[str, ...] = ("engaged", "off-by-route", "absent", "fallback", "n/a-upstream", "partial")   # partial: an accelerator that served some calls and refused others by name (the gather lever: atom_attn=partial:gather_attn(stock=<m>,…)[served=<n>]) — never `engaged`, so a route expecting engaged refuses it
UNREAD = "unread"                                           # a word not yet read when a refusal prints the line early; never on a completed pass (a word still unread at the end is refused)
ENGINE_MODULE = "rfd3.engine"
ENGINE_CLASS = "RFD3InferenceEngine"
ATTENTION_MODULE = "rfd3.model.layers.attention"
ATTENTION_LOGGER = ATTENTION_MODULE                         # RankedLogger(__name__) (attention.py:25)
ENGINE_LOGGER = ENGINE_MODULE                               # RankedLogger(__name__) (engine.py)
LAYER_UTILS_LOGGER = "rfd3.model.layers.layer_utils"        # RankedLogger(__name__) (layer_utils.py:12): the RMSNorm binding record, read on the stock routes
COMPILE_TARGETS_ATTR = "_COMPILE_TARGETS"                   # engine.py:212-217 — read from the class, never restated
UNWRAP_ATTRS = ("module", "shadow", "model")                # engine.py:243 — the walk from trainer.state["model"] to the net with `diffusion_module`
UNWRAP_DEPTH = 5                                            # engine.py:240
ATTENTION_PATH_RE = re.compile(r"Atom attention path: (DENSE-SDPA|SPARSE|GATHER) \((.*?)\) \[(.*?)\]")   # attention.py:403-404; GATHER = the gather_attn lever's answer (gather.PATH_WORD)
COMPILE_SKIPPED_TEXT = "Could not locate the diffusion module; skipping torch.compile"        # engine.py:248-250 (upstream's silent-fallback branch)
COMPILE_ENABLED_TEXT = "torch.compile enabled for diffusion submodules"                       # engine.py:265-269
RMSNORM_APEX_TEXT = "Fused RMSNorm enabled!"                                                    # layer_utils.py:16 — upstream's import guard bound apex's FusedRMSNorm
RMSNORM_TORCH_TEXT = "Using nn.RMSNorm instead of apex"                                          # layer_utils.py:19-21 — the guard fell to torch.nn.RMSNorm (no apex)
BATCH_CLOCK_RE = re.compile(r"Finished inference batch in ([0-9.]+) seconds")                # engine.py:402 — upstream's own per-roll-out clock
DECISION_FUNCTION = "use_dense_sdpa_pairbias"               # attention.py:409 — the per-call dense/sparse decision, looked up by name at the call site (:340)
DECISION_RULE_KIT = "_use_dense_sdpa_pairbias_stock"           # the kit's patched attention.py keeps upstream's rule under this name and routes use_dense_sdpa_pairbias through the graph sampler's per-shape memo under RFD3_CUDAGRAPH (patched attention.py:431-436): the pre-flight calls the RULE, never the memo
DECISION_FULL_PARAM = "full"                                # its `full` parameter (attention.py:409: Q, indices, full, H): truthy = the token-level transformer's call
DECISION_PARAMS = ("Q", "indices", "full", "H")              # upstream's signature, attention.py:409: the pin key (D, L, k, H) is read from Q.shape[:2], indices.shape[-1], H — the key the graph sampler's memo uses (cudagraph_sampler.cg_dense_decision: ("dense", D, L, k, H, full))
PATH_REPORT_FUNCTION = "_report_attention_path"           # attention.py:397-406 — upstream's once-per-process path record; a pinned answer reports through it in upstream's own words (the census traps that record like any other)
PIN_REASON = "pinned at pre-flight"                        # the reason suffix of a pinned answer's path record: `DENSE-SDPA (<est> GiB bias; pinned at pre-flight) [D=… L=… k=… H=…]`
ATTN_PATH_KEYS: Tuple[str, ...] = ("atom_dense", "atom_sparse", "token_full")   # the ATTN_PATH line's count fields, in order (the attn_path record carries the same keys); under the gather_attn lever atom_dense counts the dense BRANCH, served by the kernel (the KERNELS atom_attn word carries served/stock)
ATTN_SCOPE_KEYS: Tuple[str, ...] = ("compiled", "calls_uncounted")                # ... then its scope fields: 0|1 on the line, booleans in the file
GRAPH_MODULE = "rfd3.model.cudagraph_sampler"               # the kit's graph sampler (mode fast); its process tally GRAPH_STATS {installs, captures, hits, fallbacks}
GRAPH_STATS_ATTR = "GRAPH_STATS"
DM_X_PARAM = "X_noisy_L"                                    # RFD3_diffusion_module.py:171-173 — forward(self, X_noisy_L, t, f, …): the (D, L, 3) noisy coordinates of the step
DM_F_PARAM = "f"                                            # ... and `f`, the features dict; upstream's own token count is `f["atom_to_token_map"].max() + 1` (RFD3_diffusion_module.py:193-195)
DM_TOKEN_MAP_KEY = "atom_to_token_map"
DM_KEYS_ATTR = "n_attn_keys"                                # RFD3_diffusion_module.py:196-201 — the k of the atom attention indices the module builds for its children
ATOM_ATTENTION_CLASS = "LocalAttentionPairBias"             # attention.py:189 — the pair-bias attention; under `encoder` / `decoder` it is the atom level (full=False), its `n_head` the decision's H
ATOM_ATTENTION_PARENTS: Tuple[str, ...] = ("encoder", "decoder")
GIB = float(1 << 30)
LAYER_UTILS_MODULE = "rfd3.model.layers.layer_utils"           # (also LAYER_UTILS_LOGGER: RankedLogger(__name__), layer_utils.py:12) upstream's RMSNorm binding: `RMSNorm_` = apex FusedRMSNorm when importable, else torch.nn.RMSNorm (layer_utils.py:13-23)
APEX_NORM_MODULE = "apex.normalization.fused_layer_norm"      # the module upstream's import guard tries (layer_utils.py:14)
APEX_CUDA_EXT = "fused_layer_norm_cuda"                       # apex's extension module FusedRMSNorm computes with on CUDA tensors (imported in its constructor)
ALLOC_CONF_ENV = "PYTORCH_CUDA_ALLOC_CONF"
BATCH_KEY = "diffusion_batch_size"                          # the line token (settings.BATCH_KEY) and the engine's transform_overrides key (rfd3/engine.py:168)
DENSE_ENV = "RFD3_DENSE_SDPA_ATTENTION"                     # attention.py:427 (default "1" = dense on)
LOWMEM_ENV = "RFD3_LOW_MEMORY_MODE"                         # attention.py:430; the engine sets it itself under low_memory_mode=True (engine.py:202-205)
CUEQ_WORD = "n/a-upstream:dead-code(attention.py:282,use_kernel=False)"
DS4SCI_WORD = "n/a-upstream:hard-off(pairformer_layers.py:38;branch-raises:75-76)"
_TRUE = ("true", "1", "yes", "on", "y", "t")                # hydra/OmegaConf spellings that make `if self.compile_model:` true on the design line


def _gather():
    """The [RFD3_GATHER_ATTN] lever's switch / tally / words module (gather.py — standard library at import; torch and the kernel are touched only
    when a kit route engages it), imported on first use so a STOCK child that runs only the logging observers never loads a lever module."""
    from . import gather
    return gather


class KernelsRefused(SystemExit):
    """The guard's refusal: SystemExit(EXIT_KERNELS), raised out of the observation point so the pass unwinds at once."""

    def __init__(self, reason: str):
        super().__init__(EXIT_KERNELS)
        self.reason = reason


class NotServed(SystemExit):
    """A kit mode meeting, at the engine's initialize, a computation its levers do not drive (modes.UNSERVED: classifier-free guidance, the
    low-memory tokenization, the symmetry sampler) that the design line's reader could not see (hydra's dict / group syntax, a checkpoint's own
    config): SystemExit(EXIT_NOT_ACTIVE) raised out of the observation point — the NOT ACTIVE line is printed first, no roll-out starts."""

    def __init__(self, reason: str):
        super().__init__(EXIT_NOT_ACTIVE)
        self.reason = reason


def unserved_by_engine(engine, mode: str) -> Optional[str]:
    """The NOT ACTIVE reason when the initialized engine asks upstream for an UNSERVED computation — classifier-free guidance on (the composed
    `inference_sampler` config the engine holds, engine.py:180, under upstream's own rule modes.cfg_active, or the built model's own decision,
    RFD3.py:65-67), the low-memory tokenization (the switch the engine exported itself, engine.py:202-205), or the symmetry sampler (the composed
    config's `kind`, or the class of the sampler the model built) —, else None. The kit routes only (the caller checks the route)."""
    from . import modes                                  # the kit routes run in the patched interpreter, where the package's tables are importable
    overrides = getattr(engine, "inference_sampler_overrides", None)
    overrides = overrides if isinstance(overrides, dict) else {}
    asked = modes.cfg_active(overrides.get("use_classifier_free_guidance"), overrides.get("cfg_scale"))
    net = locate_net(engine)
    decided = bool(getattr(net, "use_classifier_free_guidance", False)) if net is not None else False
    if asked or decided:
        return modes.unserved_reason(mode, "classifier-free guidance", "inference_sampler.use_classifier_free_guidance on in the engine's composed config" if asked else "the built model's use_classifier_free_guidance")
    if os.environ.get(LOWMEM_ENV) == "1":
        return modes.unserved_reason(mode, "low_memory_mode", f"{LOWMEM_ENV}=1 exported by the engine (low_memory_mode on in its config)")
    kind = str(overrides.get("kind") or "").strip().lower()
    built = type(getattr(getattr(net, "inference_sampler", None), "sampler", None)).__name__ if net is not None else None
    if kind == modes.SYMMETRY_KIND or built == modes.SYMMETRY_SAMPLER:
        return modes.unserved_reason(mode, "symmetry sampler", f"{modes.SAMPLER_KIND_KEY}={modes.SYMMETRY_KIND} in the engine's composed config" if kind == modes.SYMMETRY_KIND else f"the built model's sampler is {built}")
    return None


def kind_of(word: str) -> str:
    """``engaged:inductor@2.13`` -> ``engaged``; ``absent`` -> ``absent``; ``unread`` -> ``unread``."""
    return (word or UNREAD).split(":", 1)[0]


def _token(text: str) -> str:
    """A word is one whitespace-free token."""
    return re.sub(r"\s+", "-", str(text).strip())


def compile_requested(tokens_or_kv) -> bool:
    """True when the design line turns upstream's ``compile_model`` on (last token wins; hydra's ``+``/``++``/``~`` prefixes stripped)."""
    from .settings import compile_requested as _cr
    return _cr(tokens_or_kv)


# --- the readers ------------------------------------------------------------------------------------------------------------

def torch_version() -> str:
    t = sys.modules.get("torch")
    return getattr(t, "__version__", None) or "unknown"


def locate_net(engine):
    """The network object that holds `diffusion_module` — reached exactly as engine.py:236-253 reaches it (trainer.state["model"],
    `_forward_module`, then `module` / `shadow` / `model` until an object with `diffusion_module`). None when the walk finds none (upstream's
    own 'Could not locate the diffusion module' case)."""
    try:
        model = engine.trainer.state["model"]
    except (AttributeError, KeyError, TypeError):        # no trainer / no state / no model: nothing to read
        return None
    net = getattr(model, "_forward_module", model)
    for _ in range(UNWRAP_DEPTH):
        if hasattr(net, "diffusion_module"):
            return net
        for attr in UNWRAP_ATTRS:
            if hasattr(net, attr):
                net = getattr(net, attr)
                break
    return net if hasattr(net, "diffusion_module") else None


def locate_diffusion_module(engine):
    """The diffusion module whose four children upstream wraps (`locate_net(engine).diffusion_module`); None when the walk finds none."""
    net = locate_net(engine)
    return net.diffusion_module if net is not None else None


def _qualname(obj) -> str:
    return f"{getattr(obj, '__module__', '?')}.{getattr(obj, '__qualname__', '?')}"


def census_rmsnorm(engine, net=None) -> Tuple[str, dict]:
    """(word, detail) for `rmsnorm`: which RMSNorm class upstream's import guard bound (layer_utils.py:13-23), read from the RMSNorm modules
    of the built network (`net`, else `locate_net(engine)`; a compiled submodule's `_orig_mod` is a registered child, so `modules()` walks it and
    the `OptimizedModule` wrapper itself is no RMSNorm instance) and from the module attribute `RMSNorm_` — never from the environment. The forms
    are the module docstring's `<word_r>` list."""
    lu = sys.modules.get(LAYER_UTILS_MODULE)
    bound = getattr(lu, "RMSNorm_", None) if lu is not None else None
    apex_cls = getattr(sys.modules.get(APEX_NORM_MODULE), "FusedRMSNorm", None)
    torch_cls = getattr(getattr(sys.modules.get("torch"), "nn", None), "RMSNorm", None)
    ext = "+ext" if APEX_CUDA_EXT in sys.modules else ""            # the extension FusedRMSNorm's forward calls on CUDA tensors (imported by its constructor)
    detail: Dict[str, object] = {"bound": _qualname(bound) if bound is not None else None, "apex_imported": APEX_NORM_MODULE in sys.modules,
                                 "cuda_ext_loaded": bool(ext)}
    n_apex = n_torch = 0
    if net is None:
        net = locate_net(engine)
    walk = getattr(net, "modules", None)
    if callable(walk):
        try:
            for m in walk():
                if apex_cls is not None and isinstance(m, apex_cls):
                    n_apex += 1
                elif torch_cls is not None and isinstance(m, torch_cls):
                    n_torch += 1
        except (TypeError, AttributeError, RuntimeError):
            pass
    detail["apex_modules"], detail["torch_modules"] = n_apex, n_torch
    apex_ver = None
    if apex_cls is not None:
        try:
            from importlib import metadata
            apex_ver = metadata.version("apex")
        except Exception:  # noqa: BLE001 — a source tree on the path without distribution metadata: version unknown, the class still names the path
            apex_ver = "unknown"
    detail["apex_version"] = apex_ver
    tv = torch_version()
    if n_apex and n_torch:
        return f"fallback:mixed(apex={n_apex},torch={n_torch})@apex{apex_ver},torch{tv}", detail
    if n_apex:
        return f"engaged:apex.FusedRMSNorm({n_apex}-modules{ext})@apex{apex_ver}", detail
    if n_torch:
        return f"fallback:torch.nn.RMSNorm({n_torch}-modules)@torch{tv}", detail
    if apex_cls is not None and bound is apex_cls:
        return f"engaged:apex.FusedRMSNorm(bound;modules-unread{ext})@apex{apex_ver}", detail
    if torch_cls is not None and bound is torch_cls:
        return f"fallback:torch.nn.RMSNorm(bound;modules-unread)@torch{tv}", detail
    return UNREAD, detail


def _optimized_module_class():
    ef = sys.modules.get("torch._dynamo.eval_frame")
    if ef is None and "torch" in sys.modules:
        try:
            import importlib
            ef = importlib.import_module("torch._dynamo.eval_frame")
        except ImportError:
            ef = None
    return getattr(ef, "OptimizedModule", None) if ef is not None else None


def _dynamo_config():
    cfg = sys.modules.get("torch._dynamo.config")
    if cfg is None and "torch" in sys.modules:
        try:
            import importlib
            cfg = importlib.import_module("torch._dynamo.config")
        except ImportError:
            cfg = None
    return cfg


def census_compile(engine) -> Tuple[str, dict]:
    """(word, detail) for `compile`, from the bound submodules of `engine` after its initialize()."""
    detail: Dict[str, object] = {"compile_model": bool(getattr(engine, "compile_model", False))}
    targets = tuple(getattr(type(engine), COMPILE_TARGETS_ATTR, None) or getattr(engine, COMPILE_TARGETS_ATTR, None) or ())
    detail["targets"] = list(targets)
    dm = locate_diffusion_module(engine)
    detail["diffusion_module"] = type(dm).__name__ if dm is not None else None
    om = _optimized_module_class()
    present = [n for n in targets if dm is not None and getattr(dm, n, None) is not None]
    wrapped = [n for n in present if om is not None and isinstance(getattr(dm, n), om)]
    detail["present"], detail["wrapped"] = present, wrapped
    cfg = _dynamo_config()
    disabled = bool(getattr(cfg, "disable", False)) if cfg is not None else False
    detail["dynamo_disabled"] = disabled
    ver = torch_version()
    kit = kit_compile_state()
    detail["kit_compile"] = kit
    if not detail["compile_model"]:
        if kit is not None and kit.get("RFD3_COMPILE"):
            # [RFD3_COMPILE] the kit's own compile (patched hoist.compile_install): armed at the module's import, applied to upstream's targets at
            # the first roll-out — after initialize(), so right after it nothing is wrapped yet (`armed`); Observer.finish re-reads it (wrapped
            # count and dynamo's graphs) and names a lever that never applied as a fallback
            if disabled:
                return "fallback:dynamo-disabled(torch._dynamo.config.disable)", detail
            if kit.get("installs"):
                if kit.get("gate"):                      # the core record's fail-closed sentences: fewer wrapped than declared, a recompile-limit hit, never ran compiled
                    return f"fallback:kit-compile-gate({kit['gate']})", detail
                if int(kit.get("wrapped") or 0) != len(targets):
                    return f"fallback:kit-compile-not-wrapped({kit.get('wrapped')}/{len(targets)})", detail
                return f"engaged:kit:torch.compile(inductor,dynamic={_dyn(kit)};{kit.get('wrapped')}/{len(targets)};cache-accessors=dynamo-boundaries)@torch{ver}", detail
            return f"engaged:kit:torch.compile(inductor,dynamic={_dyn(kit)};armed;{len(targets)}-targets)@torch{ver}", detail
        if wrapped:
            return f"fallback:wrapped-without-compile_model({len(wrapped)}/{len(targets)})", detail
        return "off-by-route:compile_model=false", detail
    if not targets:
        return f"absent:no-{COMPILE_TARGETS_ATTR}-on-{ENGINE_CLASS}", detail
    if dm is None:
        return "fallback:diffusion-module-not-located(engine.py:247-251)", detail
    if om is None:
        return "absent:torch._dynamo-not-importable", detail
    if len(wrapped) != len(targets):
        missing = [n for n in targets if n not in wrapped]
        return f"fallback:not-wrapped({len(wrapped)}/{len(targets)}:missing={','.join(missing)})", detail
    if disabled:
        return "fallback:dynamo-disabled(torch._dynamo.config.disable)", detail
    return f"engaged:torch.compile(inductor,dynamic=False;{len(wrapped)}/{len(targets)})@torch{ver}", detail


def _dyn(kit: dict) -> str:
    """The kit record's shape word for the census: `auto` (torch's automatic dynamic shapes) | `1` (symbolic) | `0` (static, upstream's) — the core
    record's own word (opt_core.capture.compile.dynamic_word), read from the kit's report."""
    v = kit.get("dynamic")
    return v if isinstance(v, str) else ("auto" if v is None else str(int(bool(v))))


def kit_compile_state() -> Optional[dict]:
    """[RFD3_COMPILE] the kit compile lever's facts as its module reports them (patched rfd3.model.hoist COMPILE_STATS via compile_describe():
    RFD3_COMPILE, installs, wrapped, declared, targets, dynamo graphs/frames/breaks/recompile-limit hits, the record's gate sentences), or None when that
    module is not imported in this process (the stock child)."""
    mod = sys.modules.get("rfd3.model.hoist")
    fn = getattr(mod, "compile_describe", None) if mod is not None else None
    if not callable(fn):
        return None
    try:
        return dict(fn())
    except Exception:  # noqa: BLE001 — an unreadable lever report reads as absent; the census then names what it sees on the modules
        return None


def inductor_cache_dir(environ=None) -> str:
    """torch inductor's on-disk cache directory: TORCHINDUCTOR_CACHE_DIR (configs/h100.env keys it per stack), else torch's default
    `<tmp>/torchinductor_<user>` (torch/_inductor/runtime/cache_dir_utils.py)."""
    environ = os.environ if environ is None else environ
    d = environ.get("TORCHINDUCTOR_CACHE_DIR")
    if d:
        return d
    import getpass
    import tempfile
    try:
        user = getpass.getuser()
    except (KeyError, OSError, ImportError):
        user = "unknown"
    return os.path.join(tempfile.gettempdir(), f"torchinductor_{user}")


def inductor_cache_state(environ=None) -> str:
    """`warm(<dir>)` when the inductor cache directory holds anything before the pass, else `cold(<dir>)` — the token that says which compile
    cost a compiled pass paid (the KERNELS line's `inductor_cache=`)."""
    d = inductor_cache_dir(environ)
    try:
        warm = os.path.isdir(d) and any(True for _ in os.scandir(d))
    except OSError:
        warm = False
    return f"{'warm' if warm else 'cold'}({_token(d)})"


def dynamo_counters() -> dict:
    """dynamo's own tallies after a run: graphs compiled, frames ok/total, graph breaks, compile seconds (None where unreadable)."""
    out = {"unique_graphs": None, "frames_ok": None, "frames_total": None, "graph_breaks": None, "compile_s": None}
    utils = sys.modules.get("torch._dynamo.utils")
    if utils is None:
        return out
    try:
        c = utils.counters
        out["unique_graphs"] = int(c["stats"].get("unique_graphs", 0))
        out["frames_ok"] = int(c["frames"].get("ok", 0))
        out["frames_total"] = int(c["frames"].get("total", 0))
        out["graph_breaks"] = int(sum(c["graph_break"].values()))
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    try:
        metrics = getattr(utils, "compilation_time_metrics", None) or {}
        key = next((k for k in metrics if str(k).endswith("_compile.compile_inner") or str(k).endswith("compile_inner")), None)
        if key is not None:
            out["compile_s"] = round(float(sum(metrics[key])), 1)
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    return out


def census_atom_attn_static(environ=None, device_type: Optional[str] = None) -> Optional[str]:
    """The part of upstream's dense-SDPA decision that is bound before any data flows (attention.py:427-435): a SPARSE forced by the two
    environment names or by a model that is not on CUDA (`device_type`: the diffusion module's parameter device after initialize()). None =
    not statically sparse; the data-dependent part (L <= k, free memory: attention.py:436-452) is read from upstream's path record at the
    first attention call."""
    environ = os.environ if environ is None else environ
    if environ.get(DENSE_ENV, "1") != "1":
        return f"fallback:sparse(disabled-via-{DENSE_ENV}={_token(environ.get(DENSE_ENV))})"
    if environ.get(LOWMEM_ENV, "0") == "1":
        return f"fallback:sparse(low-memory-mode,{LOWMEM_ENV}=1)"
    if device_type is not None and device_type != "cuda":
        return f"fallback:sparse(not-on-CUDA,device={_token(device_type)})"
    return None


def module_device_type(module) -> Optional[str]:
    """The device type of a module's first parameter (None when it has none or is not a torch module)."""
    try:
        return next(module.parameters()).device.type
    except (StopIteration, AttributeError, TypeError):
        return None


def word_of_path_record(message: str) -> Optional[Tuple[str, dict]]:
    """upstream's one path record -> (word, detail); None when `message` is not that record."""
    m = ATTENTION_PATH_RE.search(message or "")
    if not m:
        return None
    chosen, reason, shape = m.group(1), m.group(2), m.group(3)
    detail = {"chosen": chosen, "reason": reason, "shape": shape}
    if chosen == "DENSE-SDPA":
        impl = "F.scaled_dot_product_attention"
        mod = sys.modules.get(ATTENTION_MODULE)
        fn = getattr(mod, "dense_sdpa_pairbias_attention", None) if mod is not None else None
        if fn is not None:
            detail["bound"] = f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__qualname__', '?')}"
        return f"engaged:dense-sdpa({impl})@torch{torch_version()}", detail
    if chosen == "GATHER":                                   # [RFD3_GATHER_ATTN] the lever's rule answered: the atom calls take the shared core's gather kernel (gather.py)
        return gather_word(), dict(detail, **{"gather": _gather().describe()})
    return f"fallback:sparse({_token(reason)})", detail


def gather_word() -> str:
    """The atom_attn word of the gather_attn lever while the pass runs: `engaged:gather_attn(<kernel version>;<strategy>)` (Observer.finish replaces
    it with the final word from the tally, gather.final_word: engaged `[served=<n>,stock=0]` | `partial:gather_attn(stock=<m>,…)[served=<n>]` |
    `fallback:gather-never-served(…)`), or `fallback:gather-unavailable(<reason>)` when the switch is on and the kernel could not be installed (a fast
    pass is then refused by name: atom_attn expects engaged)."""
    bad = _gather().unavailable_word()
    if bad is not None:
        return bad
    s = _gather().STATS
    return f"engaged:gather_attn({_token(s.get('kernel') or 'unknown')};{_gather().STRATEGY})"


def classify_decision(full, result) -> str:
    """One call of the decision function -> its ATTN_PATH field: `full` truthy -> token_full; else the returned value: True -> atom_dense,
    False -> atom_sparse."""
    if full:
        return "token_full"
    return "atom_dense" if result else "atom_sparse"         # [RFD3_GATHER_ATTN] under the gather_attn lever the dense branch is served by the kernel: still atom_dense here (the line's grammar is a consumer contract); the KERNELS atom_attn word says who served


def _is_compiling_predicate():
    """torch's own `is this code being traced` predicate (torch.compiler.is_compiling), resolved once from the imported torch; a constant
    False without one. Inside a dynamo trace it folds to True, so the shim's counting branch is dead code there: an inlined call adds nothing
    to the traced graph and no guard on the counters."""
    torch = sys.modules.get("torch")
    for path in (("compiler", "is_compiling"), ("_dynamo", "is_compiling")):
        obj = torch
        for name in path:
            obj = getattr(obj, name, None) if obj is not None else None
        if callable(obj):
            return obj
    return lambda: False


def decision_key(args: tuple, kwargs: dict, positions: Dict[str, Optional[int]]) -> Tuple[Optional[tuple], object]:
    """((D, L, k, H) | None, full) of one call of upstream's decision function: the atom-attention pin key — Q.shape[:2], indices.shape[-1], H,
    the key the graph sampler's per-shape memo uses — and its `full` argument; None when an argument is unreadable (the call is then not pinnable
    and runs upstream's rule)."""
    def arg(name):
        if name in kwargs:
            return kwargs[name]
        i = positions.get(name)
        return args[i] if i is not None and len(args) > i else None
    full = arg(DECISION_FULL_PARAM)
    try:
        Q, indices, H = arg("Q"), arg("indices"), arg("H")
        D, L = int(Q.shape[0]), int(Q.shape[1])
        return (D, L, int(indices.shape[-1]), int(H)), full
    except (AttributeError, TypeError, ValueError, IndexError):
        return None, full


def counting_shim(fn, counts: Dict[str, int], full_index: Optional[int] = None, pins: Optional[Dict[tuple, bool]] = None, report=None):
    """A wrapper of upstream's decision function on the module attribute its call site looks up. Unpinned calls are transparent: `fn` with the
    arguments unchanged, its value returned unchanged, its exceptions propagating. An ATOM call (`full` falsy) whose key (D, L, k, H) is in `pins`
    — written by the pre-flight decision (Observer._preflight_decision) and by nothing else — answers from the pin WITHOUT calling `fn`: the
    pre-flight evaluation of upstream's rule is authoritative for the process (every later query for that key — the graph sampler's warm-up and
    capture, eager roll-outs, compiled frames, all of which reach the rule through this attribute — returns it without re-reading free memory), and
    upstream's once-per-process path record is made through `report` (the module's own `_report_attention_path`) in upstream's words with the reason
    suffixed `pinned at pre-flight`. Every interpreter-executed call is counted under classify_decision(full, result) in `counts`; pinned answers
    also under `pin_hits`, atom calls that found pins for other keys only under `pin_misses`. `full` is read from the keyword or from position
    `full_index` (upstream's signature: Q, indices, full, H). Counting and pins are skipped while dynamo traces (see _is_compiling_predicate): a
    traced caller inlines upstream's rule as on the stock route (the kit routes make this shim a dynamo boundary, Observer._wrap_attention)."""
    import functools
    import inspect
    try:
        names = list(inspect.signature(fn).parameters)
    except (TypeError, ValueError):                         # no signature: keyword reads only
        names = []
    positions: Dict[str, Optional[int]] = {p: (names.index(p) if p in names else None) for p in DECISION_PARAMS}
    if full_index is not None:
        positions[DECISION_FULL_PARAM] = full_index
    is_compiling = _is_compiling_predicate()
    pins = pins if pins is not None else {}

    @functools.wraps(fn)
    def use_dense_sdpa_pairbias(*args, **kwargs):
        if is_compiling():
            return fn(*args, **kwargs)
        key, full = decision_key(args, kwargs, positions)
        if pins and not full:
            if key in pins:
                result = bool(pins[key])
                counts["pin_hits"] = counts.get("pin_hits", 0) + 1
                if callable(report):
                    D, L, k, H = key
                    est = dense_bias_bytes(D, H, L) / GIB
                    if result and _gather().active():      # [RFD3_GATHER_ATTN] pinned served: upstream's record names the gather path (no dense bias is built)
                        report(_gather().PATH_WORD, f"{_gather().PATH_REASON}; {PIN_REASON}", f"D={D} L={L} k={k} H={H}")
                    else:
                        report("DENSE-SDPA" if result else "SPARSE", f"{est:.1f} GiB bias; {PIN_REASON}" if result else f"needs {est:.1f} GiB; {PIN_REASON}", f"D={D} L={L} k={k} H={H}")
                counts[classify_decision(full, result)] = counts.get(classify_decision(full, result), 0) + 1
                return result
            counts["pin_misses"] = counts.get("pin_misses", 0) + 1
        result = fn(*args, **kwargs)
        ckey = classify_decision(full, result)
        counts[ckey] = counts.get(ckey, 0) + 1
        return result

    use_dense_sdpa_pairbias._kernels_observed = True
    use_dense_sdpa_pairbias._kernels_counts = counts
    use_dense_sdpa_pairbias._kernels_pins = pins
    return use_dense_sdpa_pairbias


def apex_word() -> str:
    """`absent` | `present:<version>` — by importlib.util.find_spec (apex is never imported here; its version from the distribution
    metadata, else from an apex upstream already imported, else `unknown`)."""
    import importlib.util
    try:
        spec = importlib.util.find_spec("apex")
    except (ImportError, ValueError):                   # a broken parent spec / an apex in sys.modules without __spec__
        spec = None
    if spec is None and "apex" not in sys.modules:
        return "absent"
    import importlib.metadata as md
    try:
        version = md.version("apex")
    except (md.PackageNotFoundError, ValueError):        # importable without distribution metadata (a source tree on the path): the version comes from the module or is `unknown`
        version = None
    if not version:
        version = getattr(sys.modules.get("apex"), "__version__", None)
    return f"present:{_token(version or 'unknown')}"


def stack_facts(environ=None) -> Dict[str, str]:
    """The process facts the KERNELS line carries after the compile fields, read now: torch, cuda, tf32_matmul, tf32_cudnn, alloc_conf, apex."""
    environ = os.environ if environ is None else environ
    torch = sys.modules.get("torch")
    if torch is None:
        tv, cuda, tf32, tf32_cudnn = "absent", "absent", "absent", "absent"
    else:
        tv = _token(getattr(torch, "__version__", None) or "unknown")
        cuda = _token(getattr(getattr(torch, "version", None), "cuda", None) or "none")
        try:
            tf32 = str(bool(torch.backends.cuda.matmul.allow_tf32))          # the read-back of what the matmuls compute with (TORCH_ALLOW_TF32_CUBLAS_OVERRIDE, torch.set_float32_matmul_precision, upstream's own setup all land here)
        except AttributeError:                           # a torch without the backends surface (CPU-only builds, stubs): the fact is absent, not guessed
            tf32 = "absent"
        try:
            tf32_cudnn = str(bool(torch.backends.cudnn.allow_tf32))
        except AttributeError:
            tf32_cudnn = "absent"
    return {"torch": tv, "cuda": cuda, "tf32_matmul": tf32, "tf32_cudnn": tf32_cudnn, "alloc_conf": _token(environ.get(ALLOC_CONF_ENV) or "unset"),
            "apex": apex_word()}


def ckpt_word(path: Optional[str]) -> str:
    """`<basename>@<sha256[:12]>` of the checkpoint file the engine loaded (the digest through registry.checkpoint_sha256: hashed once per
    file, then from the cache); `unread` without a path (no engine initialized), `<basename>@absent` when the file is not there to hash."""
    if not path:
        return UNREAD
    base = _token(os.path.basename(str(path)))
    from .registry import checkpoint_sha256
    try:
        sha, _ = checkpoint_sha256(str(path))
    except OSError:                                      # the file is not there (or not readable) to hash: named, not guessed
        return f"{base}@absent"
    return f"{base}@{sha[:12]}"


# --- the line and the guard -------------------------------------------------------------------------------------------------

def kernels_line(route: str, mode: str, batch: Optional[str], words: Dict[str, str], extras: Optional[Dict[str, object]] = None) -> str:
    """The one KERNELS line. Every accelerator of ACCELERATORS appears, in order; an unread one prints `unread`."""
    fields = [f"{PREFIX} KERNELS route={route} mode={mode}"]
    if batch is not None:
        fields.append(f"B={batch}")
    fields += [f"{a}={words.get(a) or UNREAD}" for a in ACCELERATORS]
    for k, v in (extras or {}).items():
        if v is not None:
            fields.append(f"{k}={v}")
    return " ".join(fields)


def attn_path_counts(counts: Optional[Dict[str, int]] = None) -> Dict[str, int]:
    """The ATTN_PATH counts as ints, every key present (0 when never counted)."""
    counts = counts or {}
    return {k: int(counts.get(k, 0)) for k in ATTN_PATH_KEYS}


def attn_path_record(counts: Optional[Dict[str, int]] = None, compiled: bool = False, calls_uncounted: bool = False, preflight: Optional[List[dict]] = None) -> dict:
    """The census record's `attn_path` (opt_manifest.json `census.attn_path`): the three counts, the two scope flags, the pre-flight decisions, the pin table's hits / misses."""
    rec: Dict[str, object] = dict(attn_path_counts(counts))
    rec.update({"compiled": bool(compiled), "calls_uncounted": bool(calls_uncounted), "preflight": list(preflight or []),
                "pin_hits": int((counts or {}).get("pin_hits", 0)), "pin_misses": int((counts or {}).get("pin_misses", 0))})
    return rec


def attn_path_line(counts: Optional[Dict[str, int]] = None, compiled: bool = False, calls_uncounted: bool = False) -> str:
    """The one ATTN_PATH line: `[rfdiffusion3-opt] ATTN_PATH atom_dense=<n> atom_sparse=<m> token_full=<k> compiled=<0|1> calls_uncounted=<0|1>`."""
    c = attn_path_counts(counts)
    return (f"{PREFIX} ATTN_PATH " + " ".join(f"{k}={c[k]}" for k in ATTN_PATH_KEYS)
            + f" compiled={int(bool(compiled))} calls_uncounted={int(bool(calls_uncounted))}")


def preflight_line(route: str, pre: dict) -> str:
    """The pre-flight decision line: `[rfdiffusion3-opt] ATTN_PREFLIGHT route=<r> B=<D> n_atoms=<atoms> L=<tokens|unread> atom_sparse=<0|1> free_gib=<x>
    est_dense_gib=<y> pinned=<0|1>` — n_atoms is the atom count the atom-level attention runs over (the decision's L), L the token count (residues, for a
    protein), est_dense_gib upstream's estimate at the LARGEST atom-attention head count, pinned whether the decision was pinned for the process (1 on the
    kit routes, 0 on the stock routes: routes.Route.attn_pin)."""
    tokens = pre.get("L")
    return (f"{PREFIX} ATTN_PREFLIGHT route={route} B={pre['B']} n_atoms={pre['n_atoms']} L={tokens if tokens is not None else UNREAD} atom_sparse={pre['atom_sparse']} "
            f"free_gib={pre['free_gib']:.3f} est_dense_gib={pre['est_dense_gib']:.3f} pinned={int(bool(pre.get('pinned', 0)))}")


def token_count(f) -> Optional[int]:
    """Upstream's own token count of a step's features, `f["atom_to_token_map"].max() + 1` (RFD3_diffusion_module.py:193-195); None when unreadable."""
    try:
        tok_idx = f[DM_TOKEN_MAP_KEY]
        return int(tok_idx.max()) + 1
    except (TypeError, KeyError, IndexError, AttributeError, ValueError, RuntimeError):
        return None


def dense_bias_bytes(D: int, H: int, L: int) -> int:
    """Upstream's own estimate of the dense path's extra allocation — attention.py:445, `dense_bytes = 2 * D * H * L * L * 2` (the (D, H, L, L)
    bf16 bias plus one transient of the same size), the quantity its `0.35 × free bytes` rule compares."""
    return 2 * D * H * L * L * 2


def graph_replays() -> int:
    """Captures + hits of the kit's graph sampler in this process (GRAPH_STATS), 0 when the module is absent: a captured roll-out replays its
    remaining steps without executing the attention code in the interpreter."""
    stats = getattr(sys.modules.get(GRAPH_MODULE), GRAPH_STATS_ATTR, None)
    if not isinstance(stats, dict):
        return 0
    return int(stats.get("captures", 0) or 0) + int(stats.get("hits", 0) or 0)


def atom_attention_heads(dm) -> List[int]:
    """The `n_head` of every atom-level pair-bias attention module under the diffusion module's `encoder` / `decoder` (class
    LocalAttentionPairBias), by walking `.modules()` where the children have it, else their attributes one level down."""
    heads: List[int] = []
    for parent in ATOM_ATTENTION_PARENTS:
        sub = getattr(dm, parent, None)
        sub = getattr(sub, "_orig_mod", sub)                 # a compiled child (torch._dynamo OptimizedModule): the module it wraps
        if sub is None:
            continue
        if callable(getattr(sub, "modules", None)):         # an nn.Module: its own recursive walk
            walk = list(sub.modules())
        else:                                                # a plain object: itself and its attributes
            walk = [sub] + list(vars(sub).values()) if hasattr(sub, "__dict__") else [sub]
        for m in walk:
            if type(m).__name__ == ATOM_ATTENTION_CLASS and isinstance(getattr(m, "n_head", None), int):
                heads.append(int(m.n_head))
    return heads


def refusal_line(route: str, accel: str, word: str, expected_kind: str) -> str:
    """The guard's refusal (the NOT ACTIVE family). It never carries the `KERNELS route=` token: that grep counts passes, one line each."""
    return f"{PREFIX} NOT ACTIVE: KERNELS refused on route {route} — {accel}={word}, expected {expected_kind}; exit {EXIT_KERNELS}"


def expected_kinds(route: str) -> Dict[str, str]:
    """The expected word kind per accelerator for `route` — the routes table's row (routes.ROUTES), nothing restated here."""
    from .routes import ROUTES
    if route not in ROUTES:
        raise ValueError(f"'{route}' is not a route ({'|'.join(ROUTES)})")
    return dict(ROUTES[route].expected)


def preflight_gated(route: str) -> bool:
    """Whether the pre-flight atom-attention decision refuses `route` when sparse — the routes table's `preflight_gate` (routes.ROUTES)."""
    from .routes import ROUTES
    return bool(getattr(ROUTES.get(route), "preflight_gate", False))


def route_enforced(route: str) -> bool:
    """Whether an accelerator word that misses its expected kind REFUSES the pass on `route` (routes.ROUTES `enforce`: the kit routes) or is
    reported on a `KERNELS report-only` line while the pass proceeds (the stock routes: upstream completes such a pass itself). The pre-flight
    dense bound (`preflight_gate`) is not this switch's: it stops the kit routes."""
    from .routes import ROUTES
    return bool(getattr(ROUTES.get(route), "enforce", True))


def report_only_line(route: str, accel: str, word: str, expected_kind: str) -> str:
    """``[rfdiffusion3-opt] KERNELS report-only on route stock: rmsnorm=fallback:torch, expected engaged — upstream's run proceeds`` — the stock
    routes' record of an accelerator that missed its expected kind (never a refusal there: routes.Route.enforce)."""
    return f"{PREFIX} KERNELS report-only on route {route}: {accel}={word}, expected {expected_kind} — upstream's run proceeds"


def attn_pinned(route: str) -> bool:
    """Whether the pre-flight atom-attention decision is pinned for the process on `route`: the routes table's `attn_pin` (routes.ROUTES: the kit
    routes pin, the stock routes do not)."""
    from .routes import ROUTES
    return bool(getattr(ROUTES.get(route), "attn_pin", False))


def mismatches(route: str, words: Dict[str, str], final: bool) -> List[Tuple[str, str, str]]:
    """[(accel, word, expected_kind)] for every accelerator whose word's kind is not the route's expected kind. Before the pass ends
    (`final` False) an unread word is not yet a mismatch; at the end it is."""
    out = []
    for accel, kind in expected_kinds(route).items():
        word = words.get(accel)
        if word is None and not final:
            continue
        if kind_of(word or UNREAD) != kind:
            out.append((accel, word or UNREAD, kind))
    return out


# --- the observer -----------------------------------------------------------------------------------------------------------

class Observer:
    """One pass's census state: installs the two observation points, accumulates the words, prints the line once, applies the guard."""

    def __init__(self, route: str, mode: str, tokens: Optional[List[str]] = None, emit=None):
        from . import settings as _settings
        self.route, self.mode = route, mode
        kv = _settings.kv_of(tokens or [])
        self.kv: Dict[str, str] = dict(kv)                   # the line's own key=value tokens (read-only: the census names what the line asked)
        self.batch: Optional[str] = kv.get(BATCH_KEY)          # the line's token until initialize(); then the engine's live value (after_initialize)
        self.ckpt_path: Optional[str] = kv.get(_settings.CKPT_KEY) if not route_enforced(route) else None   # the kit routes: the checkpoint the engine resolved and loaded (after_initialize); the stock routes: the line's own `ckpt_path=` token (no engine object is touched there)
        self.words: Dict[str, str] = {"cueq": CUEQ_WORD, "ds4sci": DS4SCI_WORD}
        self.detail: Dict[str, object] = {}
        self.batch_seconds: List[float] = []            # upstream's own per-roll-out clock (engine.py:402), in order
        self.items: List[dict] = []                     # one record per upstream clock line: {"item", "clock_s" (upstream's text), "B"} — the census keeps them; nothing is printed per item (the kit prints no timing line)
        self.peaks: List[dict] = []                     # the allocator peak per item (= diffusion batch), printed as the PEAK line at the same record
        self.attn_counts: Dict[str, int] = {k: 0 for k in ATTN_PATH_KEYS}   # the counting shim's tallies (counting_shim mutates this dict)
        self.attn_pins: Dict[tuple, bool] = {}          # (D, L, k, H) -> dense: the pre-flight decisions pinned for the process (written by _preflight_decision only; the shim answers from it)
        self.pin = attn_pinned(route)                   # whether this pass pins (routes attn_pin: the kit routes)
        self.attn_printed: Optional[str] = None
        self.preflight: List[dict] = []                    # the pre-flight decision per distinct (D, L) of the pass, in order (ATTN_PREFLIGHT lines)
        self.preflight_lines: List[str] = []
        self._preflight_seen: set = set()
        self.records: List[str] = []                    # the trapped upstream records, verbatim
        self.rmsnorm: str = UNREAD                      # which RMSNorm class upstream bound (census_rmsnorm): the `rmsnorm` word, guarded like the others once read
        self.refused: Optional[dict] = None
        self.not_served: Optional[dict] = None           # a NOT ACTIVE met at initialize (unserved_by_engine): {"reason", "when"} — the pass never started
        self.report_only: Dict[str, dict] = {}          # accel -> {word, expected, when}: misses REPORTED on a stock route (routes.Route.enforce False), never refused there
        self.printed: Optional[str] = None
        self.engine_seen = False
        self._engine = None
        self._emit = emit or _default_emit
        self._handler = None
        self._finder = None

    # installation
    def install(self) -> "Observer":
        self.detail["inductor_cache"] = inductor_cache_state()     # cold | warm: read before anything of the pass compiles (compile_s differs by an order of magnitude)
        self._handler = _RecordHandler(self)
        for name in self._loggers():
            lg = logging.getLogger(name)
            if lg.level == logging.NOTSET or lg.level > logging.INFO:
                lg.setLevel(logging.INFO)                # the records must exist whatever the root configuration (upstream prints them at INFO)
            lg.addHandler(self._handler)
        if not route_enforced(self.route):               # THE STOCK ROUTES (stock | default: upstream in the pristine interpreter): logging handlers on upstream's own
            self.detail["observation"] = STOCK_OBSERVATION   # loggers and NOTHING else — no upstream callable is wrapped, shimmed or re-bound, no import hook; every census
            self.detail["batch_source"] = "line" if self.batch is not None else "upstream-default(rfdiffusion3.yaml:20)"   # word there is read from upstream's own log records (on_record) or
            if self.batch is None:                       # from the process's environment before the hand-over (stack_facts); the batch and checkpoint are the line's tokens — a line without
                from . import settings as _settings      # the batch token runs upstream's shipped value
                self.batch = _settings.BATCH_DEFAULT
            return self
        pending = {}
        for name, wrap in ((ENGINE_MODULE, self._wrap_engine), (ATTENTION_MODULE, self._wrap_attention)):
            mod = sys.modules.get(name)
            if mod is not None:
                wrap(mod)
            else:
                pending[name] = wrap
        if pending:
            self._finder = _ModuleFinder(pending)
            sys.meta_path.insert(0, self._finder)
        return self

    def _loggers(self) -> Tuple[str, ...]:
        """The upstream loggers this pass reads: the attention path record and the engine's records on every route; the RMSNorm binding record
        (layer_utils) on the stock routes, where no engine object is inspected."""
        return (ATTENTION_LOGGER, ENGINE_LOGGER) + (() if route_enforced(self.route) else (LAYER_UTILS_LOGGER,))

    def uninstall(self) -> None:
        if self._handler is not None:
            for name in self._loggers():
                try:
                    logging.getLogger(name).removeHandler(self._handler)
                except ValueError:
                    pass
        if self._finder is not None:
            try:
                sys.meta_path.remove(self._finder)
            except ValueError:
                pass

    def _wrap_engine(self, module) -> None:
        cls = getattr(module, ENGINE_CLASS, None)
        if cls is None or getattr(cls.initialize, "_kernels_observed", False):
            return
        orig = cls.initialize
        observer = self

        def initialize(engine, *a, **k):
            out = orig(engine, *a, **k)                  # upstream's own initialize (the compile happens inside it, engine.py:219-224) ...
            observer.after_initialize(engine)           # ... then the read-only census of what it bound
            return out
        initialize._kernels_observed = True
        initialize.__wrapped__ = orig
        cls.initialize = initialize

    def _wrap_attention(self, module) -> None:
        """Put the counting shim on the module attribute upstream's call site looks up (idempotent; a module without the function is left alone)."""
        fn = getattr(module, DECISION_FUNCTION, None)
        if fn is None:
            return
        if getattr(fn, "_kernels_observed", False):          # already shimmed (one shim per process, never stacked): this observer reads its tally and writes its pin table
            self.attn_counts = fn._kernels_counts
            self.attn_pins = getattr(fn, "_kernels_pins", self.attn_pins)
            return
        shim = counting_shim(fn, self.attn_counts, pins=self.attn_pins, report=getattr(module, PATH_REPORT_FUNCTION, None))
        hoist = sys.modules.get("rfd3.model.hoist")
        if hoist is not None and getattr(hoist, "COMPILE", False):
            # [RFD3_COMPILE] the shim sits between compiled callers and the kit's (already dynamo-disabled) decision function: it is a dynamo
            # boundary itself, so every call runs in the interpreter and is counted, and no compiled frame inlines it (opt_core.capture.compile)
            from opt_core.capture.compile import nodynamo
            marked = nodynamo(shim, active=True)
            for attr in ("_kernels_observed", "_kernels_counts", "_kernels_pins"):
                setattr(marked, attr, getattr(shim, attr))
            shim = marked
        setattr(module, DECISION_FUNCTION, shim)
        self.detail["attn_shim"] = f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__qualname__', '?')}"

    # observation points
    def after_initialize(self, engine) -> None:
        self.engine_seen = True
        self._engine = engine                            # [RFD3_COMPILE] re-read at finish: the kit's compile applies at the first roll-out, after this point
        if route_enforced(self.route):                   # the kit routes: a computation the mode's levers do not drive (modes.UNSERVED), spelled where the line reader could not see it, is refused by name here — before the first roll-out, nothing attached
            why = unserved_by_engine(engine, self.mode)
            if why is not None:
                self._refuse_unserved(why)
        overrides = getattr(engine, "transform_overrides", None)  # base.py:106 — {"diffusion_batch_size": n} as the rfd3 engine passed it (engine.py:168)
        live = overrides.get(BATCH_KEY) if isinstance(overrides, dict) else None
        if live is not None:                             # the engine's live diffusion_batch_size is the B that counts (a line without the token runs upstream's configured value)
            self.batch = str(live)
        self.detail["batch_source"] = "engine" if live is not None else ("line" if self.batch is not None else None)
        ck = getattr(engine, "ckpt_path", None)
        self.ckpt_path = os.fspath(ck) if ck else None    # base.py:95 — the registry-resolved file upstream loads
        mod = sys.modules.get(ATTENTION_MODULE)          # the model is built: the attention module is imported by now whatever route brought it in
        if mod is not None:
            if route_enforced(self.route) and _gather().wanted():   # [RFD3_GATHER_ATTN] (kit routes only: a stock child never loads the lever module) the lever wraps the module's decision rule and dense function first; the counting shim then sits on the public entry as on every route
                _gather().install(mod)
                self.detail["gather"] = _gather().describe()
            self._wrap_attention(mod)
        self._peak_reset()                               # item 0 starts here: the allocator peak is per item
        word, detail = census_compile(engine)
        self.words["compile"] = word
        self.detail["compile"] = detail
        self._read_rmsnorm(engine)
        dm = locate_diffusion_module(engine)
        static = census_atom_attn_static(device_type=module_device_type(dm) if dm is not None else None)
        if static is not None:
            self.words["atom_attn"] = static
            self.detail["atom_attn"] = {"static": True}
        if dm is not None:
            self._install_preflight(dm)
        else:
            self.detail["preflight"] = "not-installed(diffusion module not located)"
        self._guard(final=False)

    # the pre-flight atom-attention decision
    def _install_preflight(self, dm) -> None:
        """Wrap the diffusion module's forward on the instance (the module upstream's samplers call once per denoising step; never compiled
        itself — upstream compiles its four children — and called in the interpreter by the graph sampler's warm-up before its capture), so
        the first step of every distinct (D, L) runs the pre-flight decision before any attention call of that shape."""
        fwd = getattr(dm, "forward", None)
        if fwd is None:
            self.detail["preflight"] = "not-installed(diffusion module has no forward)"
            return
        if getattr(fwd, "_kernels_observed", False):
            return
        import functools
        obs = self

        @functools.wraps(fwd)
        def forward(*args, **kwargs):
            obs._preflight_step(dm, args, kwargs)
            return fwd(*args, **kwargs)

        forward._kernels_observed = True
        forward.__wrapped__ = fwd
        try:
            dm.forward = forward                             # instance attribute: nn.Module.__call__ reads self.forward
        except (AttributeError, TypeError) as e:
            self.detail["preflight"] = f"not-installed({e!r})"
            return
        self.detail["preflight"] = "installed"

    def _preflight_step(self, dm, args, kwargs) -> None:
        """One denoising step is about to run: on the first step of a (D, L) not seen before, evaluate upstream's own decision function at
        that shape — eagerly, with the free memory of this moment — print ATTN_PREFLIGHT, record it, and refuse the pass on a gated route
        when the decision is sparse. Never raises for a fact it cannot read (recorded under detail.preflight_unread); KernelsRefused propagates."""
        x = kwargs.get(DM_X_PARAM, args[0] if args else None)
        shape = getattr(x, "shape", None)
        if shape is None or len(shape) < 2:
            self.detail.setdefault("preflight_unread", f"no {DM_X_PARAM} of shape (D, L, 3) on the step call")
            return
        D, L = int(shape[0]), int(shape[-2])             # L: the ATOM count (X_noisy_L is (D, L_atoms, 3)) — the decision's L
        if (D, L) in self._preflight_seen:
            return
        torch = sys.modules.get("torch")
        cuda = getattr(torch, "cuda", None)
        if cuda is not None and callable(getattr(cuda, "is_current_stream_capturing", None)) and cuda.is_current_stream_capturing():
            return                                           # never inside a graph capture: the next interpreter-executed step of this shape reads it
        self._preflight_seen.add((D, L))
        pre = self._preflight_decision(dm, x, D, L)
        if pre is None:
            return
        pre["L"] = token_count(kwargs.get(DM_F_PARAM, args[2] if len(args) > 2 else None))   # the token count (residues) beside the atom count, upstream's own expression
        self.preflight.append(pre)
        line = preflight_line(self.route, pre)
        self.preflight_lines.append(line)
        self._emit(line)
        by_switch = bool((self.detail.get("atom_attn") or {}).get("static")) and not route_enforced(self.route)   # the stock routes under upstream's own force-sparse / low-memory switch: the caller's sparse path, reported (KERNELS report-only), never the dense bound's stop
        if pre["atom_sparse"] and preflight_gated(self.route) and self.refused is None and not by_switch:
            self._refuse("attn_preflight", f"sparse(B={D},n_atoms={L},free_gib={pre['free_gib']:.3f},est_dense_gib={pre['est_dense_gib']:.3f})", "dense")

    def _preflight_decision(self, dm, x, D: int, L: int) -> Optional[dict]:
        """Upstream's decision RULE called ONCE per atom-attention head count, eagerly, at (D, L, k, H): k the diffusion module's `n_attn_keys`, H
        every distinct `n_head` among its atom-level attention modules (the line reports the largest — the call that goes sparse first), on
        zero-byte tensors of the step's device — the rule reads shapes, the device, the environment and torch.cuda.mem_get_info, nothing else
        (attention.py:417-454), and holds no state (on the kit's patched file the rule is `_use_dense_sdpa_pairbias_stock`; its public entry
        memoises per shape under the graph sampler and is NOT what the probe calls). This runs at the first denoising step of the shape, BEFORE the
        step's pair / activation tensors exist: the free memory compiled upstream's trace-time evaluation of the same rule sees. When the pass pins
        (self.pin: the kit routes), every (D, L, k, H) decision is written to the pin table the decision shim answers from —
        the model's own calls for those keys never re-evaluate the rule, and a pinned dense path that does not fit raises torch's out-of-memory error
        out of the attention call (never a fallback to sparse); upstream's force-sparse switches make the rule itself say sparse here, so a gated
        route refuses before the first step as without the pin. Unpinned (the stock routes), the model's own calls compute
        their own decision exactly as without the probe and upstream's once-per-process path record is left to the model's first call (the module
        flag is held across the probe either way; a pinned first call reports through upstream's record function with the reason `pinned at pre-flight`)."""
        mod = sys.modules.get(ATTENTION_MODULE)
        fn = getattr(mod, DECISION_RULE_KIT, None) if mod is not None else None   # the kit's patched file: upstream's rule itself — never its graph-mode entry, whose per-shape memo would cache (and silence the record of) this probe's evaluation for the model's own calls
        if fn is None and mod is not None:
            fn = getattr(mod, DECISION_FUNCTION, None)       # the pristine file: the module attribute IS the rule ...
            fn = getattr(fn, "__wrapped__", fn)             # ... minus the counting shim (the pre-flight is not a counted call)
        heads = atom_attention_heads(dm)
        k = getattr(dm, DM_KEYS_ATTR, None)
        torch = sys.modules.get("torch")
        why = None
        if fn is None:
            why = f"{ATTENTION_MODULE}.{DECISION_FUNCTION} absent"
        elif not heads:
            why = f"no {ATOM_ATTENTION_CLASS} with n_head under {'/'.join(ATOM_ATTENTION_PARENTS)}"
        elif not isinstance(k, int):
            why = f"diffusion module has no integer {DM_KEYS_ATTR}"
        elif torch is None or not callable(getattr(torch, "empty", None)):
            why = "no torch to build the probe tensors"
        if why:
            self.detail["preflight_unread"] = why
            return None
        distinct = sorted(set(heads), reverse=True)          # largest first: the line's H and estimate
        H = distinct[0]
        device = getattr(x, "device", None)
        q = torch.empty((D, L, 0), device=device)           # zero bytes: the function reads Q.shape, Q.is_cuda, Q.device
        idx = torch.empty((0, 0, k), device=device)         # ... and indices.shape[-1]
        free = None
        cuda = getattr(torch, "cuda", None)
        if getattr(q, "is_cuda", False) and cuda is not None and callable(getattr(cuda, "mem_get_info", None)):
            free = cuda.mem_get_info(device)[0]
        held = getattr(mod, "_DENSE_PATH_REPORTED", None)
        decisions: Dict[int, bool] = {}
        try:
            if held is not None:
                mod._DENSE_PATH_REPORTED = True              # upstream logs its first decision once per process: that record stays the model's own first call's, not this probe's
            for h in distinct:
                decisions[h] = bool(fn(Q=q, indices=idx, full=False, H=h))
        finally:
            if held is not None:
                mod._DENSE_PATH_REPORTED = held
        if self.pin:
            for h, dense in decisions.items():
                self.attn_pins[(D, L, int(k), h)] = dense      # the AUTHORITATIVE decision for this key, for the process (counting_shim answers from it)
        dense_all = all(decisions.values())
        est = dense_bias_bytes(D, H, L)
        if route_enforced(self.route) and _gather().active():  # [RFD3_GATHER_ATTN] (kit routes only) under the lever the rule answers for the kernel (dense branch, nothing (D,H,L,L) allocated): est_dense_gib stays upstream's formula, informational
            self.detail["gather_preflight"] = {"B": D, "n_atoms": L, "served_branch": bool(all(decisions.values()))}
        return {"route": self.route, "B": D, "n_atoms": L, "L": None, "k": int(k), "H": H, "heads": sorted(set(heads)), "atom_sparse": 0 if dense_all else 1, "dense": dense_all,
                "decisions": {str(h): d for h, d in sorted(decisions.items())}, "pinned": 1 if self.pin else 0,
                "free_gib": (free / GIB) if free is not None else -1.0, "est_dense_gib": est / GIB, "device": str(device), "after_items": len(self.items)}

    def on_record(self, name: str, message: str) -> None:
        if name == ATTENTION_LOGGER:
            got = word_of_path_record(message)
            if got is None:
                return
            self.records.append(message)
            word, detail = got
            current = kind_of(self.words.get("atom_attn") or UNREAD)
            if current != "fallback":                            # a statically-read SPARSE stays the final word
                self.words["atom_attn"] = word
                self.detail["atom_attn"] = detail
            self._guard(final=False)
        elif name == ENGINE_LOGGER:
            m = BATCH_CLOCK_RE.search(message or "")
            if m:
                k = len(self.batch_seconds)
                self.batch_seconds.append(float(m.group(1)))
                item = {"item": f"batch:{k}", "clock_s": m.group(1), "B": self.batch}
                self.items.append(item)
                self._peak_line(k)                              # the PEAK line of the item that just finished, then the reset for the next
                return
            if COMPILE_SKIPPED_TEXT in (message or ""):
                self.records.append(message)
                self.words["compile"] = "fallback:compile-skipped(engine.py:247-251)"
                self._guard(final=False)
            elif COMPILE_ENABLED_TEXT in (message or ""):
                self.records.append(message)
                if not route_enforced(self.route):               # the stock routes read upstream's compile off its own record (engine.py:265-269); the kit routes read the engine's modules (census_compile)
                    self.words["compile"] = "engaged:torch.compile(record:engine.py:265-269)"
                    self._guard(final=False)
        elif name == LAYER_UTILS_LOGGER:                          # the stock routes: which RMSNorm upstream's import guard bound, off its own record (layer_utils.py:13-23)
            text = message or ""
            if RMSNORM_APEX_TEXT in text or RMSNORM_TORCH_TEXT in text:
                self.records.append(message)
                self.rmsnorm = ("engaged:apex.FusedRMSNorm(record:layer_utils.py:16)" if RMSNORM_APEX_TEXT in text else
                                "fallback:torch.nn.RMSNorm(record:layer_utils.py:19-21)")
                self.words["rmsnorm"] = self.rmsnorm
                self.detail["rmsnorm"] = {"source": "record", "record": text.strip()[:120]}
                self._guard(final=False)

    @staticmethod
    def _cuda():
        """torch.cuda when this process has CUDA peak statistics to read (a torch imported by upstream, CUDA available), else None."""
        cuda = getattr(sys.modules.get("torch"), "cuda", None)
        if cuda is None or not callable(getattr(cuda, "is_available", None)) or not cuda.is_available():
            return None
        return cuda

    def _peak_reset(self) -> None:
        cuda = self._cuda()
        if cuda is not None:
            cuda.reset_peak_memory_stats()

    def _peak_line(self, k: int) -> None:
        """`PEAK item=batch:<k> alloc_gib=<max_memory_allocated> reserved_gib=<max_memory_reserved>` — the torch allocator's peak over the item
        (upstream's k-th diffusion batch of the pass, reset at its start), read from torch.cuda's peak statistics (`max_memory_allocated` /
        `max_memory_reserved` of the current device); the device-level cross-check is an nvidia-smi sampler outside the process. Nothing
        without torch / CUDA."""
        cuda = self._cuda()
        if cuda is None:
            return
        dev = cuda.current_device()
        peak = {"item": f"batch:{k}", "alloc_gib": cuda.max_memory_allocated(dev) / GIB, "reserved_gib": cuda.max_memory_reserved(dev) / GIB, "device": str(dev)}
        self.peaks.append(peak)
        self._emit(f"{PREFIX} PEAK item={peak['item']} alloc_gib={peak['alloc_gib']:.2f} reserved_gib={peak['reserved_gib']:.2f}")
        cuda.reset_peak_memory_stats(dev)

    # the end of the pass
    def finish(self, rc: Optional[int] = None) -> int:
        """Complete the words, print the line once, apply the guard. Returns the pass's exit code: EXIT_KERNELS when the guard refuses a
        pass that otherwise succeeded (rc 0 / None), else rc unchanged (a failed run keeps its own code; the line still prints). A pass refused
        at initialize for a computation the mode does not drive (not_served) printed its NOT ACTIVE line there and ran nothing: no KERNELS line,
        the kit's NOT ACTIVE code."""
        if self.not_served is not None:
            return rc if rc not in (0, None) else EXIT_NOT_ACTIVE
        counters = dynamo_counters()
        self.detail["dynamo"] = counters
        cw = self.words.get("compile")
        if cw is None:
            if not route_enforced(self.route):               # the stock routes (no engine object read): no compile record = upstream compiled nothing — off when the line did not ask (default), unread when it did (stock)
                self.words["compile"] = (f"{UNREAD}:no-compile-record(engine.py:265-269)" if expected_kinds(self.route).get("compile") == "engaged" else
                                         "off-by-route:compile_model=false")
            else:
                self.words["compile"] = "absent:engine-never-initialized" if not self.engine_seen else UNREAD
        elif kind_of(cw) == "engaged" and (rc in (0, None)):
            if cw.startswith("engaged:kit:") and self._engine is not None:
                self.words["compile"], self.detail["compile"] = census_compile(self._engine)   # [RFD3_COMPILE] armed at initialize -> applied (or not) by now
                if ";armed;" in self.words["compile"]:
                    self.words["compile"] = "fallback:kit-compile-never-applied"      # the lever read on at import, yet no roll-out ever installed it: not a fast pass
            if kind_of(self.words["compile"]) == "engaged" and counters["unique_graphs"] == 0:
                self.words["compile"] = "fallback:no-graph-compiled(dynamo-counters)"     # wrapped, yet dynamo compiled nothing: every frame ran eager
        # graph breaks and frames dynamo skipped are inductor's normal partial-graph operation at the pin (upstream's own modules break the
        # graph at data-dependent scalars): they are NAMED on the line (graphs= frames= graph_breaks=), not a refusal — the compiled graphs run.
        if self.words.get("atom_attn") is None:
            mod = sys.modules.get(ATTENTION_MODULE)
            reported = getattr(mod, "_DENSE_PATH_REPORTED", None) if mod is not None else None
            lowmem = (not route_enforced(self.route)) and (str(self.kv.get("low_memory_mode", "")).lower() in ("true", "1") or os.environ.get(LOWMEM_ENV) == "1")
            self.words["atom_attn"] = ("fallback:path-record-unread(logging)" if reported else
                                       ("fallback:sparse(low-memory-mode;line)" if lowmem else        # upstream's low-memory path never reaches the dense/sparse decision (no record): the line's own token / upstream's own switch names it
                                        ("absent:attention-never-ran" if rc in (0, None) else UNREAD)))
        if route_enforced(self.route) and _gather().wanted():  # [RFD3_GATHER_ATTN] (kit routes only: the stock routes report upstream's own path record and never load the lever module) the lever's final word: engaged with its served/stock census, or the named fallback (unavailable / never served)
            self.detail["gather"] = _gather().describe()
            bad = _gather().unavailable_word()
            if bad is not None:
                self.words["atom_attn"] = bad
            elif kind_of(self.words.get("atom_attn") or UNREAD) == "engaged":   # the final word from the tally: engaged (stock=0) | partial (stock>0) | fallback (never served) — gather.final_word
                self.words["atom_attn"] = _gather().final_word(gather_word(), calls_seen=bool(self.items or sum(attn_path_counts(self.attn_counts).values()) > 0))
        extras = self._extras()
        if self.printed is None:
            self.printed = kernels_line(self.route, self.mode, self.batch, self.words, extras)
            self._emit(self.printed)
        self._attn_path_report()
        if rc not in (0, None):
            return rc
        bad = mismatches(self.route, self.words, final=True)
        if bad and not route_enforced(self.route):          # the stock routes: an accelerator miss is a record (KERNELS report-only line), never a refusal — upstream completes such a pass itself
            for accel, word, kind in bad:
                if accel not in self.report_only:
                    self.report_only[accel] = {"word": word, "expected": kind, "when": "finish"}
                    self._emit(report_only_line(self.route, accel, word, kind))
            bad = []
        if not bad and preflight_gated(self.route) and not self.preflight and self.items:   # a gated route whose roll-outs ran with no pre-flight decision read: the gate cannot pass on nothing
            why = self.detail.get("preflight_unread") or ("installed(no step observed)" if self.detail.get("preflight") == "installed" else self.detail.get("preflight")) or "no engine initialized"
            bad = [("attn_preflight", f"unread({_token(str(why))})", "dense")]
        if bad and self.refused is None:
            accel, word, kind = bad[0]
            self.refused = {"accel": accel, "word": word, "expected": kind, "when": "finish"}
            self._emit(refusal_line(self.route, accel, word, kind))
        self._compile_drift_note()
        return EXIT_KERNELS if self.refused else (0 if rc is None else rc)

    def _compile_drift_note(self) -> None:
        """One line when dynamo skipped more frames than the kit measured on its pinned stack (opt_core CompileRecord: frames it attempted and
        did not convert run eager): ``COMPILE frames_skipped=<n> above the <allowed> measured on the pinned stack (torch <version>): named — those
        frames run eager, the compiled graphs serve the rest``. A torch / dynamo other than the pinned one is NAMED here — never a reason to call
        the lever off or to refuse the pass: the compile word stays what the graphs say (engaged when every target wrapped, compiled frames ran
        and no recompile-limit was hit; the kit gate's sentences otherwise)."""
        kit = (self.detail.get("compile") or {}).get("kit_compile") or {}
        n, allowed = kit.get("frames_skipped"), kit.get("frames_skipped_allowed")
        if isinstance(n, int) and isinstance(allowed, int) and n > allowed and not self.detail.get("compile_drift_note"):
            self.detail["compile_drift_note"] = {"frames_skipped": n, "frames_skipped_allowed": allowed}
            self._emit(f"{PREFIX} COMPILE frames_skipped={n} above the {allowed} measured on the pinned stack (torch {torch_version()}): named — those frames run eager, the compiled graphs serve the rest")

    def _attn_scope(self) -> Tuple[bool, bool]:
        """(compiled, calls_uncounted): compiled = upstream's torch.compile engaged on this pass (the attention calls of the compiled children
        run inside dynamo's graphs: the dense/sparse decision is taken while dynamo traces and frozen into each graph, so those calls are not
        counted one by one); calls_uncounted = compiled, or the graph sampler replayed captured steps (their attention runs without the
        interpreter). The pre-flight decision is the census of those routes."""
        compiled = kind_of(self.words.get("compile") or UNREAD) == "engaged"
        return compiled, compiled or graph_replays() > 0

    def _attn_record(self) -> dict:
        compiled, uncounted = self._attn_scope()
        return attn_path_record(self.attn_counts, compiled, uncounted, self.preflight)

    def _attn_path_report(self) -> None:
        """The ATTN_PATH line (once) on the kit routes; its record travels in the census record (`attn_path`). Report only: nothing here refuses.
        The stock routes count no attention calls (no shim in a stock process) and print no ATTN_PATH line: upstream's own path record names
        the path there (the `atom_attn` word)."""
        if self.attn_printed is not None or not route_enforced(self.route):
            return
        compiled, uncounted = self._attn_scope()
        self.attn_printed = attn_path_line(self.attn_counts, compiled, uncounted)
        self._emit(self.attn_printed)

    def _read_rmsnorm(self, engine) -> None:
        """Read the `rmsnorm` word (census_rmsnorm). Never fatal here: a read that raises is worded `unread:<Exception>`."""
        try:
            self.rmsnorm, self.detail["rmsnorm"] = census_rmsnorm(engine)
        except Exception as e:  # noqa: BLE001 — a report-only word cannot abort a pass; the failure is named on the line instead
            self.rmsnorm, self.detail["rmsnorm"] = f"{UNREAD}:{_token(type(e).__name__)}", {"error": repr(e)}
        if self.rmsnorm != UNREAD:                       # a read word joins the guarded words (routes.ROUTES expects `engaged` on every route); unread stays unread until the pass's end
            self.words["rmsnorm"] = self.rmsnorm

    def _extras(self) -> Dict[str, object]:
        """The KERNELS line's k=v tail, in order: the compile fields (a compiled pass only), then the stack facts and the checkpoint (every pass).
        (`rmsnorm` is a word of the line, ACCELERATORS: a line printed before after_initialize ran — an early refusal — reads the module
        binding alone first, if bound.)"""
        if self.rmsnorm == UNREAD:
            self._read_rmsnorm(None)
        out: Dict[str, object] = {}
        c = self.detail.get("dynamo") or {}
        if kind_of(self.words.get("compile") or UNREAD) in ("engaged", "fallback") and c.get("unique_graphs") is not None:
            frames = f"{c['frames_ok']}/{c['frames_total']}" if c.get("frames_total") is not None else None
            out.update({"compile_s": c.get("compile_s"), "inductor_cache": self.detail.get("inductor_cache"), "graphs": c.get("unique_graphs"), "frames": frames,
                        "graph_breaks": c.get("graph_breaks")})
        facts = stack_facts()
        facts["ckpt"] = ckpt_word(self.ckpt_path)
        self.detail["stack"] = dict(facts)
        out.update(facts)
        return out

    def _guard(self, final: bool) -> None:
        bad = mismatches(self.route, self.words, final=final)
        if not bad or self.refused is not None:
            return
        if not route_enforced(self.route):               # the stock routes: the miss is a record, the pass proceeds (upstream completes it itself)
            for accel, word, kind in bad:
                if accel not in self.report_only:
                    self.report_only[accel] = {"word": word, "expected": kind, "when": "before-timing" if not self.batch_seconds else ("final" if final else "mid-run")}
                    self._emit(report_only_line(self.route, accel, word, kind))
            return
        self._refuse(*bad[0])

    def _refuse(self, accel: str, word: str, kind: str) -> None:
        """Refuse the pass NOW: record the refusal, print the KERNELS line (once) and the verdict, raise KernelsRefused (SystemExit 5)."""
        self.refused = {"accel": accel, "word": word, "expected": kind, "when": "before-timing" if not self.batch_seconds else "mid-run"}
        if self.printed is None:
            self.printed = kernels_line(self.route, self.mode, self.batch, self.words, self._extras())
            self._emit(self.printed)
        line = refusal_line(self.route, accel, word, kind)
        self._emit(line)
        raise KernelsRefused(line)

    def _refuse_unserved(self, reason: str) -> None:
        """Refuse the pass NOW for a computation the mode does not drive: record it, print the kit's NOT ACTIVE line, raise NotServed
        (SystemExit 3). No KERNELS line: no roll-out ran, there is no pass to hold to the census."""
        self.not_served = {"reason": reason, "when": "initialize"}
        self._emit(f"{PREFIX} NOT ACTIVE: {reason}")
        raise NotServed(reason)

    def record(self) -> dict:
        """The census as data (opt_manifest.json `kernels`; under --mode off the stock child's copy arrives as `stock_env_proof.after.kernels`)."""
        return {"route": self.route, "mode": self.mode, "batch": self.batch, "words": dict(self.words),
                "line": self.printed, "refused": self.refused, "not_served": self.not_served, "report_only": dict(self.report_only), "detail": self.detail, "batch_seconds": list(self.batch_seconds), "items": list(self.items),
                "peaks": list(self.peaks), "attn_path": self._attn_record(), "attn_path_line": self.attn_printed, "attn_preflight_lines": list(self.preflight_lines),
                "ckpt_path": self.ckpt_path, "rmsnorm": self.rmsnorm, "records": list(self.records), "expected": expected_kinds(self.route)}


class _RecordHandler(logging.Handler):
    """Traps upstream's records on the two loggers; a refusal raised in on_record unwinds out of upstream's own logging call."""

    def __init__(self, observer: Observer):
        super().__init__(level=logging.DEBUG)
        self.observer = observer

    def emit(self, record):                             # noqa: D401 — logging API
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 — the logging contract: a handler never raises on a record it cannot format; the record is skipped, the census stays unread
            return
        self.observer.on_record(record.name, message)   # KernelsRefused (SystemExit) propagates by design


class _ModuleFinder:
    """Duck-typed meta-path finder: lets each named upstream module execute, then applies its read-only wrap — `rfd3.engine` (the engine
    class's initialize) and `rfd3.model.layers.attention` (the counting shim) — one shot per name; removes itself when none is pending."""

    def __init__(self, pending: Dict[str, object]):
        self.pending = dict(pending)                     # {module name: wrap(module)}

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self.pending:
            return None
        spec = None
        for finder in sys.meta_path:
            if finder is self:
                continue
            try:
                spec = finder.find_spec(fullname, path, target)
            except Exception:  # noqa: BLE001 — a foreign meta_path finder raising in find_spec is skipped (as stack.AppliedHook does)
                spec = None
            if spec is not None:
                break
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return None
        orig = spec.loader.exec_module

        def exec_module(module, _orig=orig, _name=fullname):
            _orig(module)                                # upstream's module body first ...
            wrap = self.pending.pop(_name, None)
            if not self.pending:
                try:
                    sys.meta_path.remove(self)
                except ValueError:
                    pass
            if wrap is not None:
                wrap(module)                             # ... then the read-only wrap
        spec.loader.exec_module = exec_module
        return spec


_default_emit = _emit                                    # the package's one printer (_emit.emit): fresh line, flushed — a census line printed inside a handler is on the wire when it returns


# --- the process-level API --------------------------------------------------------------------------------------------------

_STATE: Dict[str, Optional[Observer]] = {"observer": None}


def observe(route: str, mode: str, tokens: Optional[List[str]] = None, emit=None) -> Observer:
    """Install the census for this process's pass (idempotent: the first observer stands). `tokens` = the design line's key=value
    overrides (settings label, batch size)."""
    if _STATE["observer"] is not None:
        return _STATE["observer"]
    expected_kinds(route)                                # an unknown route is refused by name here, before anything is installed
    obs = Observer(route, mode, tokens, emit=emit).install()
    _STATE["observer"] = obs
    return obs


def not_served() -> Optional[dict]:
    """The pass's refusal at initialize for a computation the mode does not drive ({"reason", "when"}), or None: the ONE fact the verb and the
    exit hooks read to print nothing further about levers — no roll-out ran, there is no evidence to report and no partial verdict to reach."""
    obs = _STATE["observer"]
    return obs.not_served if obs is not None else None


def observer() -> Optional[Observer]:
    return _STATE["observer"]


def finish(rc: Optional[int] = None) -> int:
    """The pass's end: print the line (once), apply the guard, return the exit code (see Observer.finish). No observer: rc unchanged."""
    obs = _STATE["observer"]
    if obs is None:
        return 0 if rc is None else rc
    return obs.finish(rc)


def finish_at_exit() -> None:
    """The exit-hook form (upstream's own command line has no verb after the run): the line prints and the guard applies; a refusal found
    here (the census still unread or degraded at the pass's end) terminates the interpreter with EXIT_KERNELS from the hook itself
    (``os._exit``: an exit hook cannot return a code, and a refusal that prints `exit 5` while the process exits 0 would be a fail-open). A
    refusal reached during the run already exited 5; a process that activated a mode but never initialized an engine ran no pass: nothing
    to print."""
    obs = _STATE["observer"]
    if obs is not None and obs.printed is None and obs.engine_seen:
        obs.finish(None)
        if obs.refused is not None:
            sys.stderr.flush(); sys.stdout.flush()
            os._exit(EXIT_KERNELS)


def record() -> Optional[dict]:
    obs = _STATE["observer"]
    return obs.record() if obs is not None else None


def reset() -> None:
    """Tests only: drop the process observer."""
    obs = _STATE["observer"]
    if obs is not None:
        obs.uninstall()
    _STATE["observer"] = None
