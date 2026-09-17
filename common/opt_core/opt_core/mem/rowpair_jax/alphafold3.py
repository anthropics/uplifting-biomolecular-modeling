"""The row-sharded pair stack of the AlphaFold-3 Haiku model library (``alphafold3.model``, the sokrypton JAX tree the AF3 JAX kit runs): ONE
producer of the patched module bodies; the kit's adapters are thin (mesh, placement, evidence, launcher) and import this recipe.

:func:`install(rmesh, patches, ...)` rebinds, through :func:`haiku.rebind` (Haiku's own method wrapper: name scopes and the parameter tree are the
stock's) and the adapter's :class:`PatchSet`, the bodies below — each the STOCK body on this device's row block ``[N/P, N, C]`` with the plan's
collectives inserted (:mod:`shard` / :mod:`trimul` / :mod:`triatt` / :mod:`transition`), and each dispatching to the untouched stock body when it
runs outside a row-sharded region (the replicated call sites keep the stock program):

  trunk   ``modules.TriangleMultiplication.__call__`` (outgoing: all_gather of the partner plane; incoming: all_gather + all_to_all — the
          contraction complete per device, :func:`trimul.contract`), ``modules.GridSelfAttention.__call__`` (row attention row-local behind the
          gathered bias; column attention bracketed by the all_to_all transpose), ``modules.PairFormerIteration.__call__`` and
          ``modules.EvoformerIteration.__call__`` (ONE region per block: the five pair sub-layers + single attention's gathered pair logits /
          OuterProductMean with the left operand sliced to the block / MSAAttention's gathered pair bias), ``modules.OuterProductMean.__call__``,
          ``modules.MSAAttention.__call__``, the ``evoformer.Evoformer`` prologue methods and ``__call__`` (row constraints on their pair outputs);
  model   ``model.Model.__call__`` (the recycle carry constrained row-sharded through ``hk.fori_loop``; the heads ONE manual region whose Haiku
          frame carries the heads key — the stock's draw position) ;
  heads   ``diffusion_head.DiffusionHead._conditioning`` on the row block through the kit-supplied pair-conditioning transcription (``b21``: the
          engine's own ``pair_conditioning_rows`` / ``conditioning_factory`` / ``REL_FIELDS`` module, passed in — the core imports no kit) with the
          atom encoder's 16-channel pair projection all-gathered, ``diffusion_transformer.self_attention`` with queries on the row block;
          ``confidence_head.ConfidenceHead.__call__`` / ``._embed_features`` and ``distogram_head.DistogramHead.__call__`` on the row block
          (symmetrisations as all_to_all transposes, the global PDE mean as psums, the ``[N, N]`` outputs all-gathered).

Options are NAMED states on the record :func:`install` returns (and on the adapter's lever line): ``heads=sharded|replicated`` (replicated: the
f32 pair is all-gathered at the trunk → heads boundary — per-card relief is then the trunk's only), ``conf=sharded|replicated``,
``diffusion=sharded|replicated|absent`` (``absent`` when no ``b21`` module is given: the diffusion head reads the gathered pair), ``schedule=`` of
:func:`trimul.contract`. Numerics: the region bodies are the stock arithmetic on row blocks (classes ``row_local`` / ``complete_contraction`` /
``moves_bytes``; the PDE mean ``reordered``) — a P>1 run is tested as a BAND against the kit's P=1 run. Haiku facts the recipe relies on are
:mod:`haiku`'s (rng-less body frames at apply; ``~_embed_features`` scope kept by rebinding the method under its own name; no ordered callbacks).

PIN. The bodies transcribe ONE stock tree: sokrypton/alphafold3 v3.1.4 @ bc32b22f (:data:`PIN`: the sha256 of each patched module's source file
as installed in the kit's pinned stack). :func:`install` hashes the library it finds and REFUSES BY NAME on any other bytes (a kit on another
tree passes its own ``pin`` mapping only together with its own transcription review). ``n_gpu=1`` never routes through the recipe (refused).

HAZARDS carried from the canary, named (:data:`HAZARDS`): the column attention and the distogram/confidence masks assume the pair mask is the
outer product of the sequence mask (symmetric) — true for this tree's callers; the region bodies run under an rng-less Haiku frame (no sub-layer
draws keys at inference); ``_embed_features`` is rebound under its own name so its Linears keep the ``~_embed_features`` scope; the trunk marker is
an UNORDERED callback (ordered effects are refused on >1 device); the atom-encoder interceptor answers two named modules with precomputed tensors.

Transcribed from a measured multi-device program (:data:`CANARY`; the ``b21`` diffusion extraction is the kit's own transcription) onto
this package's primitives; per-body
source lines and numerics classes in :data:`BODIES`. The model library is imported inside :func:`install`; a process without it gets a refusal
naming the lever. Standard library at import.
"""
from __future__ import annotations

import contextlib
import functools
import importlib
from typing import Any, Callable, Dict, List, Optional

from .. import MemLeverRefused
from . import LEVER, _lazy
from . import haiku as _hk
from . import shard as _shard
from . import transition as _tr
from . import triatt as _ta
from . import trimul as _tm

LIBRARY = {"modules": "alphafold3.model.network.modules", "evoformer": "alphafold3.model.network.evoformer",
           "diffusion_transformer": "alphafold3.model.network.diffusion_transformer", "diffusion_head": "alphafold3.model.network.diffusion_head",
           "confidence_head": "alphafold3.model.network.confidence_head", "distogram_head": "alphafold3.model.network.distogram_head",
           "atom_cross_attention": "alphafold3.model.network.atom_cross_attention", "mapping": "alphafold3.model.components.mapping",
           "haiku_modules": "alphafold3.model.components.haiku_modules", "model": "alphafold3.model.model", "feat_batch": "alphafold3.model.feat_batch"}
B21_NAMES = ("pair_conditioning_rows", "conditioning_factory", "REL_FIELDS")     # the kit's diffusion pair-conditioning transcription, passed in
SITES_TRUNK = ("TriangleMultiplication.__call__", "GridSelfAttention.__call__", "PairFormerIteration.__call__", "EvoformerIteration.__call__",
               "OuterProductMean.__call__", "MSAAttention.__call__", "Evoformer._seq_pair_embedding", "Evoformer._embed_bonds",
               "Evoformer._embed_template_pair", "Evoformer._embed_process_msa", "Evoformer.__call__")
SITES_MODEL = ("model.hk(fori_loop carry)", "Model.__call__")
PROLOGUE_PAIR_OUTPUT = (("_seq_pair_embedding", 0), ("_embed_bonds", None), ("_embed_template_pair", 0), ("_embed_process_msa", 0))
# Evoformer prologue methods → where the pair array sits in their return value (evoformer.py @ the pin: _seq_pair_embedding → (pair_activations, pair_mask);
# _embed_bonds → pair_activations; _embed_template_pair → (pair_activations, key); _embed_process_msa → (pair_activations, key)); checked against the
# method's source at install (_return_arity) and again at trace
SITES_HEADS = ("DiffusionHead._conditioning", "diffusion_transformer.self_attention", "ConfidenceHead.__call__", "ConfidenceHead._embed_features",
               "DistogramHead.__call__")

PIN: Dict[str, Any] = {"tree": "sokrypton/alphafold3 v3.1.4 @ bc32b22ff5902e3daffd5d1f7203d7f2ab6cb997",
                       "image": "the kit's pinned image (/alphafold3_venv site-packages)",
                       "files": {      # {LIBRARY short name: sha256 of the module source} as installed in the kit's pinned stack; other bytes = install refuses unless the adapter passes pin=
                           "atom_cross_attention": "469728f8a955108b1701d2c42e1ab739c23e8ff7e664bbf9a5608e7f379ed384",
                           "confidence_head": "923dc7d5a07b264fd6d2341425048b08174b1826198a1680a4c1a84bbb07c4d0",
                           "diffusion_head": "7f86f48a38b9be527ce66a9affe315a05b0f14130e6c47a81d29eb7a91a82391",
                           "diffusion_transformer": "4b8a17b31e117549299e14b5a5fb990bc0d67384b8dbe47746b3c0a1a99dfe05",
                           "distogram_head": "538ff1f6096f1f3d2bc58b628d80f2f900f1bd8a84c8dedca0b8824c242d3669",
                           "evoformer": "e492da0ba999203cae1c4dfcce3f7de89c826ce66ba46ca3fc1698991e1c0be7",
                           "feat_batch": "3bb1c61a1c4f50753ab96f01f8cc0903f6840ee0ec4d8b040f982e5962103a70",
                           "haiku_modules": "22b4993e8574289a4b77a302c489e1e2dc27ea959b04d1262543f19adaaabf15",
                           "mapping": "216ab9188061de38b8a2a15a95c3472ca5b9f414a9779fb4b7fe43bf22b07d80",
                           "model": "179870b4b676b427a72efb6be242aeec2670bf7cc9555c570ec4c485001da175",
                           "modules": "52e4f658657e8081cbc6d7d651cb3eb6af4e4aa3a86adf9fbf212e25d2ddd60d",
                       }}
HAZARDS = ("symmetric_pair_mask(col_attention,distogram,confidence)", "rngless_body_frame(apply)", "embed_features_scope(~_embed_features)",
           "unordered_trunk_marker", "atom_encoder_interceptor(diffusion_lnorm_trunk_pair_cond,diffusion_embed_trunk_pair_cond)",
           "module_state_trace_flags(thread_local;pc16_cleared_at_region_exit)")
CANARY = "af3_shard.py@572e20cc35df"
BODIES = {                                   # site -> (numerics class vs the stock body on the same rows, canary source lines)
    "TriangleMultiplication.__call__": ("complete_contraction", CANARY + ":98-141"),
    "GridSelfAttention.__call__": ("row_local", CANARY + ":143-168"),
    "PairFormerIteration.__call__": ("row_local", CANARY + ":170-221"),
    "EvoformerIteration.__call__": ("row_local", CANARY + ":223-244"),
    "OuterProductMean.__call__": ("row_local", CANARY + ":246-270"),
    "MSAAttention.__call__": ("row_local", CANARY + ":272-291"),
    "Evoformer.(prologue,__call__)": ("moves_bytes", CANARY + ":293-312"),
    "model.hk(fori_loop carry)": ("moves_bytes", CANARY + ":314-331"),
    "Model.__call__": ("moves_bytes", CANARY + ":333-404"),
    "DiffusionHead._conditioning": ("row_local", CANARY + ":406-447 (b21 = the kit's transcription)"),
    "diffusion_transformer.self_attention": ("row_local", CANARY + ":598-634"),
    "ConfidenceHead.__call__": ("row_local+reordered(average_pde psum)", CANARY + ":449-554"),
    "ConfidenceHead._embed_features": ("row_local", CANARY + ":556-572"),
    "DistogramHead.__call__": ("row_local", CANARY + ":574-596"),
}

import threading as _threading


class _TraceState(_threading.local):
    """The recipe's trace-time flags (HAZARD ``module_state_trace_flags``): region depth counters and the heads-mode switches the patched bodies read
    while ONE thread traces the model, plus the precomputed 16-channel pair projection handed from the diffusion conditioning to the atom encoder within
    the same heads region (set and cleared inside that region; reading it outside is refused). Thread-local: a second thread tracing concurrently sees
    its own state; jax traces a jitted function on the calling thread."""
    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.sharded, self.manual = 0, 0
        self.heads_rows, self.heads_rows_conf, self.pc16_computing = False, False, False
        self.pc16_full = None

    def __getitem__(self, k):
        return getattr(self, k)

    def __setitem__(self, k, v):
        setattr(self, k, v)

    def update(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


_S = _TraceState()


def library(lever: str = LEVER) -> Dict[str, Any]:
    """The AlphaFold-3 model library modules of this process (``{short: module}``) or a refusal naming the lever."""
    out = {}
    for short, name in LIBRARY.items():
        try:
            out[short] = importlib.import_module(name)
        except ImportError as e:
            raise MemLeverRefused(lever, f"{name} not importable ({e})") from None
    return out


def library_shas(L: Optional[Dict[str, Any]] = None, lever: str = LEVER) -> Dict[str, str]:
    """``{short: sha256 of the module's source file}`` for the modules of :data:`LIBRARY` in this process."""
    import hashlib  # noqa: PLC0415
    L = L if L is not None else library(lever)
    out = {}
    for short, mod in L.items():
        path = getattr(mod, "__file__", None)
        if not path:
            raise MemLeverRefused(lever, f"{LIBRARY[short]} has no __file__ to pin")
        with open(path, "rb") as fh:
            out[short] = hashlib.sha256(fh.read()).hexdigest()
    return out


def check_pin(L: Dict[str, Any], pin: Optional[Dict[str, str]] = None, lever: str = LEVER) -> Dict[str, str]:
    """Refuse BY NAME unless every module of :data:`LIBRARY` hashes to ``pin`` (default :data:`PIN` ``files``): the bodies transcribe one tree. Returns the shas."""
    want = dict(PIN["files"]) if pin is None else dict(pin)
    if not want:
        raise MemLeverRefused(lever, "alphafold3 recipe: no pin of record for the model library (PIN['files'] empty and no pin= given)")
    have = library_shas(L, lever)
    bad = [f"{s}:{have.get(s, 'absent')[:12]}!={want.get(s, 'unpinned')[:12]}" for s in sorted(set(want) | set(have)) if have.get(s) != want.get(s)]
    if bad:
        raise MemLeverRefused(lever, "alphafold3 recipe: the model library is not the transcribed tree (%s): %s" % (PIN["tree"], ",".join(bad)))
    return have


def in_sharded_region() -> bool:
    """True inside a row-sharded region of this recipe (the patched bodies then run on local rows; else they call the stock body)."""
    return int(_S["sharded"]) > 0


def in_manual_region() -> bool:
    """True inside any manual (shard_map) region of this recipe — the row-sharded blocks or the heads region."""
    return int(_S["manual"]) > 0


@contextlib.contextmanager
def _sharded_body(lever: str):
    _S["sharded"] += 1
    _S["manual"] += 1
    try:
        with _hk.body_frame(None, lever):                                  # rng-less: hk.scan in the sub-batch loops threads no key (haiku fact 2)
            yield
    finally:
        _S["sharded"] -= 1
        _S["manual"] -= 1


@contextlib.contextmanager
def _manual_body(rng: Any, lever: str):
    _S["manual"] += 1
    try:
        with _hk.body_frame(rng, lever):                                   # the heads draw keys: the region's frame carries the explicit key
            yield
    finally:
        _S["manual"] -= 1


def install(rmesh: Any, patches: Any, *, heads: str = "sharded", conf: str = "sharded", b21: Any = None, schedule: str = _tm.DEFAULT,
            sites: tuple = ("trunk", "model", "heads"), mark: Optional[Callable[[str], None]] = None, pin: Optional[Dict[str, str]] = None,
            lever: str = LEVER, _test_single_device_mesh: bool = False) -> Dict[str, Any]:
    """Rebind the recipe's bodies (see the module docstring) for mesh ``rmesh`` into ``patches`` (a :class:`PatchSet`; ``patches.restore()`` undoes
    everything). ``sites``: which groups to install (``trunk`` — the Evoformer/Pairformer pair stack; ``model`` — ``Model.__call__`` with the sharded
    recycle carry and the heads region; ``heads`` — the diffusion / confidence / distogram bodies on row blocks). ``heads`` / ``conf``: ``sharded`` or
    ``replicated``; ``b21``: the kit's pair-conditioning transcription module (attributes :data:`B21_NAMES`) — without it the diffusion head runs on
    the gathered pair (``diffusion=absent``). ``mark(text)``: an optional host-side phase marker called from an UNORDERED debug callback after the
    trunk. Returns the install record ``{"sites": [...], "heads": …, "conf": …, "diffusion": …, "schedule": …, "axis": …, "n_gpu": …}`` for the
    adapter's lever line (``library=<sha prefixes>`` included). Refusals by name: ``n_gpu=1`` (P=1 never routes through the recipe), the library not the
    pinned tree (:func:`check_pin`; ``pin`` overrides :data:`PIN` for a kit that reviewed another tree), unknown option words, a ``b21`` lacking a name, a
    stock attribute that moved (PatchSet), the library absent."""
    if int(getattr(rmesh, "n_gpu", 0)) < 2 and not _test_single_device_mesh:      # the unit tests' fidelity test installs on a 1-device mesh (≡ stock); no kit path passes the hook
        raise MemLeverRefused(lever, "alphafold3 recipe: n_gpu=%s — P=1 installs nothing (the recipe is the P>1 program)" % (getattr(rmesh, "n_gpu", None),))
    for word, val, allowed in (("heads", heads, ("sharded", "replicated")), ("conf", conf, ("sharded", "replicated")), ("schedule", schedule, _tm.SCHEDULES)):
        if val not in allowed:
            raise MemLeverRefused(lever, f"install: {word}={val!r} is not one of {','.join(allowed)}")
    for g in sites:
        if g not in ("trunk", "model", "heads"):
            raise MemLeverRefused(lever, f"install: site group {g!r} is not one of trunk,model,heads")
    if b21 is not None:
        missing = [n for n in B21_NAMES if not hasattr(b21, n)]
        if missing:
            raise MemLeverRefused(lever, f"install: the b21 module lacks {','.join(missing)}")
    L = library(lever)
    shas = check_pin(L, pin, lever)
    hk = _lazy.haiku(lever)
    jax = _lazy.jax(lever)
    jnp = _lazy.jnp(lever)
    modules, evoformer_mod, DT, DH, CH, DG = L["modules"], L["evoformer"], L["diffusion_transformer"], L["diffusion_head"], L["confidence_head"], L["distogram_head"]
    mapping, hm, model_mod, feat_batch = L["mapping"], L["haiku_modules"], L["model"], L["feat_batch"]
    AXIS = rmesh.axis
    P = int(rmesh.n_gpu)
    rows, rep = _shard.rows_spec(AXIS), _shard.replicated_spec()
    rows2, cols2 = _shard.rows_spec(AXIS, 2, 0), _shard.cols_spec(AXIS, 2, 1)      # the [N, N] pair mask: this device's rows / columns
    _S.reset()
    shard_heads = "heads" in sites and heads == "sharded" and b21 is not None
    shard_conf = "heads" in sites and conf == "sharded"
    # the record reports the EFFECTIVE state, never the request: without b21 the diffusion head reads the gathered pair whatever `heads` asked for
    diffusion_word = "not_installed" if "heads" not in sites else ("sharded" if shard_heads else ("absent" if b21 is None else "replicated"))
    heads_word = "not_installed" if "heads" not in sites else ("sharded" if shard_heads else "replicated")
    record: Dict[str, Any] = {"sites": [], "heads": heads_word, "heads_requested": heads, "conf": ("sharded" if shard_conf else "replicated") if "heads" in sites else "not_installed",
                              "diffusion": diffusion_word, "schedule": schedule, "axis": AXIS, "n_gpu": P,
                              "library": ",".join(f"{k}:{v[:8]}" for k, v in sorted(shas.items())), "hazards": ",".join(HAZARDS)}

    def _rebind(cls, name, fn, site):
        orig = _hk.rebind(patches, cls, name, fn, lever)
        record["sites"].append(site)
        return orig

    def _rows_of(x, start, n_loc, axis=0):
        return jax.lax.dynamic_slice_in_dim(x, start, n_loc, axis=axis)

    # ================================================================================================ trunk
    if "trunk" in sites:
        # ---- modules.TriangleMultiplication.__call__ (modules.py:272-344) on the local row block
        def trimul_call(self, act, mask):
            """act: [N/P, N, C] local rows; mask: [N/P, N] local rows. The stock body + the partner collective before the einsum (:326)."""
            if not in_sharded_region():
                return stock_trimul(self, act, mask)
            mask = mask[None, ...]
            num_channels = act.shape[-1]
            equation = {"ikc,jkc->ijc": "cik,cjk->cij", "kjc,kic->ijc": "ckj,cki->cij"}[self.config.equation]
            act = hm.LayerNorm(name="left_norm_input")(act)
            input_act = act
            if self.config.use_glu_kernel:
                weights_projection, _ = hm.haiku_linear_get_params(act, num_output=num_channels * 2, name="projection")
                weights_gate, _ = hm.haiku_linear_get_params(act, num_output=num_channels * 2, initializer=self.global_config.final_init, name="gate")
                weights_glu = jnp.stack([weights_gate, weights_projection], axis=1)
                projection = modules.tokamax.gated_linear_unit(act, weights_glu, activation=jax.nn.sigmoid)
                projection = jnp.transpose(projection, (2, 0, 1))
                projection *= mask
            else:
                projection = hm.Linear(num_channels * 2, name="projection")(act)
                projection = jnp.transpose(projection, (2, 0, 1))
                projection *= mask
                gate = hm.Linear(num_channels * 2, name="gate", bias_init=1.0, initializer=self.global_config.final_init)(act)
                gate = jnp.transpose(gate, (2, 0, 1))
                projection *= jax.nn.sigmoid(gate)
            projection = projection.reshape(num_channels, 2, *projection.shape[1:])
            a, b = jnp.split(projection, 2, axis=1)
            a, b = jnp.squeeze(a, axis=1), jnp.squeeze(b, axis=1)          # [C, N/P, N] each
            act = _tm.contract(equation, a, b, AXIS, schedule=schedule, lever=lever)   # outgoing: gather(b); incoming: gather(a) + all_to_all(b) — complete per device
            act = hm.LayerNorm(name="center_norm", axis=0, param_axis=0)(act)
            act = jnp.transpose(act, (1, 2, 0))
            act = hm.Linear(num_channels, initializer=self.global_config.final_init, name="output_projection")(act)
            gate_out = hm.Linear(num_channels, name="gating_linear", bias_init=1.0, initializer=self.global_config.final_init)(input_act)
            act *= jax.nn.sigmoid(gate_out)
            return act
        stock_trimul = _rebind(modules.TriangleMultiplication, "__call__", trimul_call, "TriangleMultiplication.__call__")

        # ---- modules.GridSelfAttention.__call__ (modules.py:208-255)
        def gsa_call(self, act, pair_mask_rows, pair_mask_cols=None, num_residues_global=None):
            """act: [N/P, N, C] local rows; pair_mask_rows: [N/P, N]; pair_mask_cols: [N, N/P]; the chunk policy reads the GLOBAL N (stock's choice)."""
            if not in_sharded_region():
                return stock_gsa(self, act, pair_mask_rows)
            act = hm.LayerNorm(name="act_norm")(act)
            nonbatched_bias = hm.Linear(self.config.num_head, use_bias=False, name="pair_bias_projection")(act)   # [N/P, N, H]
            nonbatched_bias = jnp.transpose(nonbatched_bias, [2, 0, 1])                                           # [H, N/P, N]
            nonbatched_bias = _ta.bias_full(nonbatched_bias, AXIS, row_dim=1)                                     # [H, N, N] (stock :225-228)
            if self.global_config.of3_weights and self.transpose:
                nonbatched_bias = jnp.swapaxes(nonbatched_bias, -1, -2)                                          # stock :231-232
            chunk_size = modules.get_shard_size(num_residues_global, self.global_config.pair_attention_chunk_size)
            if self.transpose:
                act = _ta.enter_transposed(act, AXIS)                                                             # [N/P(j), N(i), C]: my rows of x^T
                pm = jnp.swapaxes(pair_mask_cols, -1, -2)                                                         # [N/P(j), N(i)]
            else:
                pm = pair_mask_rows                                                                               # the stock's swapaxes of a symmetric mask
            pm = pm[:, None, None, :].astype(jnp.bool_)
            act = mapping.inference_subbatch(self._attention, chunk_size, batched_args=[act, pm], nonbatched_args=[nonbatched_bias])
            if self.transpose:
                act = _ta.exit_transposed(act, AXIS)                                                              # back to [N/P(i), N(j), C]
            return act
        stock_gsa = _rebind(modules.GridSelfAttention, "__call__", gsa_call, "GridSelfAttention.__call__")

        # ---- shared: the five pair sub-layers on the local block (PairFormerIteration :479-515 / EvoformerIteration :599-635)
        def pair_sublayers(self, pair_act, pm_rows, pm_cols, num_residues):
            pair_act += modules.TriangleMultiplication(self.config.triangle_multiplication_outgoing, self.global_config, name="triangle_multiplication_outgoing")(pair_act, pm_rows)
            pair_act += modules.TriangleMultiplication(self.config.triangle_multiplication_incoming, self.global_config, name="triangle_multiplication_incoming")(pair_act, pm_rows)
            pair_act += modules.GridSelfAttention(self.config.pair_attention, self.global_config, name="pair_attention1", transpose=False)(pair_act, pm_rows, pm_cols, num_residues)
            pair_act += modules.GridSelfAttention(self.config.pair_attention, self.global_config, name="pair_attention2", transpose=True)(pair_act, pm_rows, pm_cols, num_residues)
            transition_block = modules.TransitionBlock(self.config.pair_transition, self.global_config, name="pair_transition")
            if self.config.shard_transition_blocks:
                transition_block = mapping.sharded_apply(transition_block, modules.get_shard_size(num_residues, self.global_config.pair_transition_shard_spec))
            pair_act += transition_block(pair_act)
            return pair_act

        # ---- modules.PairFormerIteration.__call__ (modules.py:458-543)
        def pfi_call(self, act, pair_mask, single_act=None, seq_mask=None):
            if "confidence" in self.module_name and not shard_conf:
                return stock_pfi(self, act, pair_mask, single_act, seq_mask)  # the confidence head's pairformer stays the replicated stock body under conf=replicated
            num_residues = int(pair_mask.shape[0])                             # the FULL token count (chunk sizes); act may be this device's rows inside a manual region
            if not in_manual_region():
                _shard.local_extent(num_residues, P, lever)                    # an unpadded N is refused by name (the kit pads: shard.pad_plan)
            with_single = self.with_single
            def body(act_l, pm_rows, pm_cols, pair_mask_full, single_act, seq_mask):
                with _sharded_body(lever):
                    act_l = pair_sublayers(self, act_l, pm_rows, pm_cols, num_residues)
                    if with_single:
                        pair_logits = hm.Linear(self.config.single_attention.num_head, name="single_pair_logits_projection")(hm.LayerNorm(name="single_pair_logits_norm")(act_l))  # [N/P, N, H]
                        pair_logits = _tr.pair_logits_full(pair_logits, AXIS)                # [N, N, H] (stock :519-523)
                        pair_logits = jnp.transpose(pair_logits, [2, 0, 1])
                        single_act = single_act + DT.self_attention(single_act, seq_mask, pair_logits, self.config.single_attention, self.global_config, name="single_attention_")
                        single_act = single_act + modules.TransitionBlock(self.config.single_transition, self.global_config, name="single_transition")(single_act, broadcast_dim=None)
                    return act_l, single_act
            if single_act is None:
                single_act = jnp.zeros((1,), act.dtype)
                seq_mask = jnp.zeros((1,), act.dtype)
            if in_manual_region():
                # already inside a manual region (the heads region): act = this device's rows [N/P, N, C]; the body runs directly
                n_loc = int(act.shape[0])
                start = _shard.axis_index(AXIS) * n_loc
                pm_rows = _rows_of(pair_mask, start, n_loc, 0)
                pm_cols = _rows_of(pair_mask, start, n_loc, 1)
                act, single_out = body(act, pm_rows, pm_cols, pair_mask, single_act, seq_mask)
                return (act, single_out) if with_single else act
            f = _shard.shard_map(body, rmesh, (rows, rows2, cols2, rep, rep, rep), (rows, rep))
            act, single_out = f(act, pair_mask, pair_mask, pair_mask, single_act, seq_mask)
            return (act, single_out) if with_single else act
        stock_pfi = _rebind(modules.PairFormerIteration, "__call__", pfi_call, "PairFormerIteration.__call__")

        # ---- modules.EvoformerIteration.__call__ (modules.py:576-637)
        def evi_call(self, activations, masks):
            msa_act, pair_act = activations["msa"], activations["pair"]
            msa_mask, pair_mask = masks["msa"], masks["pair"]
            num_residues = pair_act.shape[0]
            n_loc = _shard.local_extent(num_residues, P, lever)
            def body(pair_l, pm_rows, pm_cols, msa_act, msa_mask):
                with _sharded_body(lever):
                    idx = _shard.axis_index(AXIS)
                    pair_l += modules.OuterProductMean(config=self.config.outer_product_mean, global_config=self.global_config, num_output_channel=int(pair_l.shape[-1]), name="outer_product_mean")(msa_act, msa_mask, idx, n_loc)
                    msa_act += modules.MSAAttention(self.config.msa_attention, self.global_config, name="msa_attention1")(msa_act, msa_mask, pair_act=pair_l)
                    msa_act += modules.TransitionBlock(self.config.msa_transition, self.global_config, name="msa_transition")(msa_act)
                    pair_l = pair_sublayers(self, pair_l, pm_rows, pm_cols, num_residues)
                    return pair_l, msa_act
            f = _shard.shard_map(body, rmesh, (rows, rows2, cols2, rep, rep), (rows, rep))
            pair_act, msa_act = f(pair_act, pair_mask, pair_mask, msa_act, msa_mask)
            return {"msa": msa_act, "pair": pair_act}
        _rebind(modules.EvoformerIteration, "__call__", evi_call, "EvoformerIteration.__call__")

        # ---- modules.OuterProductMean.__call__ (modules.py:367-423): the local output rows (left operand sliced to the block)
        def opm_local(self, act, mask, idx=None, n_loc=None):
            if not in_sharded_region():
                return stock_opm(self, act, mask)
            mask = mask[..., None]
            act = hm.LayerNorm(name="layer_norm_input")(act)
            left_act = mask * hm.Linear(self.config.num_outer_channel, initializer="linear", name="left_projection")(act)
            right_act = mask * hm.Linear(self.config.num_outer_channel, initializer="linear", name="right_projection")(act)
            if self.global_config.final_init == "zeros":
                w_init = hk.initializers.Constant(0.0)
            else:
                w_init = hk.initializers.VarianceScaling(scale=2.0, mode="fan_in")
            output_w = hk.get_parameter("output_w", shape=(self.config.num_outer_channel, self.config.num_outer_channel, self.num_output_channel), dtype=act.dtype, init=w_init)
            output_b = hk.get_parameter("output_b", shape=(self.num_output_channel,), dtype=act.dtype, init=hk.initializers.Constant(0.0))
            left_loc, mask_loc = _tr.opm_operands(left_act, mask, AXIS, n_loc, res_dim=1)         # [S, N/P, c], [S, N/P, 1]
            def compute_chunk(left_act):
                out = jnp.einsum("abc,ade,cef->bdf", left_act, right_act, output_w)
                return out + output_b
            act = mapping.inference_subbatch(compute_chunk, self.config.chunk_size, batched_args=[left_loc], nonbatched_args=[], input_subbatch_dim=1, output_subbatch_dim=0)
            epsilon = 1e-3
            norm = jnp.einsum("abc,adc->bdc", mask_loc, mask)
            return act / (epsilon + norm)
        stock_opm = _rebind(modules.OuterProductMean, "__call__", opm_local, "OuterProductMean.__call__")

        # ---- modules.MSAAttention.__call__ (modules.py:109-137): pair_act = the local rows; the logits all-gathered to [H, N, N]
        def msa_attention_local(self, act, mask, pair_act):
            if not in_sharded_region():
                return stock_msaatt(self, act, mask, pair_act)
            act = hm.LayerNorm(name="act_norm")(act)
            pair_act = hm.LayerNorm(name="pair_norm")(pair_act)
            logits = hm.Linear(self.config.num_head, use_bias=False, name="pair_logits")(pair_act)   # [N/P, N, H]
            logits = _tr.pair_logits_full(logits, AXIS)                                                # [N, N, H]
            logits = jnp.transpose(logits, [2, 0, 1])
            logits += 1e9 * (jnp.max(mask, axis=0) - 1.0)
            weights = jax.nn.softmax(logits, axis=-1)
            num_channels = act.shape[-1]
            value_dim = num_channels // self.config.num_head
            v = hm.Linear([self.config.num_head, value_dim], use_bias=False, name="v_projection")(act)
            v_avg = jnp.einsum("hqk, bkhc -> bqhc", weights, v)
            v_avg = jnp.reshape(v_avg, v_avg.shape[:-2] + (-1,))
            gate_values = hm.Linear(self.config.num_head * value_dim, bias_init=1.0, initializer="zeros", name="gating_query")(act)
            v_avg *= jax.nn.sigmoid(gate_values)
            return hm.Linear(num_channels, initializer=self.global_config.final_init, name="output_projection")(v_avg)
        stock_msaatt = _rebind(modules.MSAAttention, "__call__", msa_attention_local, "MSAAttention.__call__")

        # ---- the Evoformer prologue's pair outputs and the trunk output stay row-sharded (evoformer.py:278-310, :358-362): plain attributes, not Haiku methods
        Ev = evoformer_mod.Evoformer
        def _constrain_pair(method, out_index, label):
            @functools.wraps(method)
            def wrapped(self, *a, **k):
                out = method(self, *a, **k)
                if _hk.running_init():
                    return out
                if out_index is None:
                    if isinstance(out, (tuple, list)):
                        raise MemLeverRefused(lever, f"alphafold3 recipe: {label} returned a {len(out)}-tuple where the transcription expects the pair array")
                    return _shard.constrain(out, rmesh)
                if not isinstance(out, (tuple, list)) or len(out) <= out_index:
                    raise MemLeverRefused(lever, f"alphafold3 recipe: {label} did not return a tuple with the pair at index {out_index}")
                out = list(out)
                out[out_index] = _shard.constrain(out[out_index], rmesh)
                return tuple(out)
            return wrapped
        # the Evoformer prologue's pair producers (evoformer.py, pinned): the pair activations are output 0 of a (pair, mask|key) tuple, or the only output
        for meth, oi in PROLOGUE_PAIR_OUTPUT:
            fn = getattr(Ev, meth)
            arity = _return_arity(fn)
            if (oi is None) != (arity == 1):
                raise MemLeverRefused(lever, f"alphafold3 recipe: Evoformer.{meth} returns {arity} value(s); the transcription table says pair index {oi!r} — the tree differs from the pin's")
            patches.replace(Ev, meth, _constrain_pair(fn, oi, "Evoformer." + meth))
            record["sites"].append("Evoformer." + meth)
        _ev_call = Ev.__call__
        @functools.wraps(_ev_call)
        def ev_call(self, *a, **k):
            out = _ev_call(self, *a, **k)
            if _hk.running_init():
                return out
            out = dict(out)
            out["pair"] = _shard.constrain(out["pair"], rmesh)
            return out
        patches.replace(Ev, "__call__", ev_call)
        record["sites"].append("Evoformer.__call__")

    # ================================================================================================ model
    if "model" in sites:
        evoformer_network, confidence_head, distogram_head, diffusion_head = model_mod.evoformer_network, model_mod.confidence_head, model_mod.distogram_head, model_mod.diffusion_head

        class _HKModelProxy:                                                # model.py's `hk` name: fori_loop with the pair carry constrained (model.py:311-326)
            def __init__(self, hk_mod):
                self._hk = hk_mod
            def __getattr__(self, k):
                return getattr(self._hk, k)
            def fori_loop(self, lower, upper, body_fun, init_val):
                embeddings, key = init_val
                embeddings = _tr.constrain_carry(dict(embeddings), rmesh)
                embeddings, key = self._hk.fori_loop(lower, upper, body_fun, (embeddings, key))
                embeddings = _tr.constrain_carry(dict(embeddings), rmesh)  # the trunk → heads boundary: the pair stays row-sharded into the heads region
                if mark is not None:
                    jax.debug.callback(lambda v: mark("trunk_done"), embeddings["pair"][0, 0, 0])   # data-dependent, UNORDERED (ordered effects are refused on >1 device)
                return embeddings, key
        patches.replace(model_mod, "hk", _HKModelProxy(model_mod.hk))
        record["sites"].append("model.hk(fori_loop carry)")

        def model_call(self, batch, key=None):
            """model.py:272-360 verbatim up to the trunk; the heads (:328-360) run inside ONE manual region whose frame carries the heads key."""
            if key is None:
                key = hk.next_rng_key()
            if self.global_config.of3_weights:
                batch = dict(batch)
                batch["ref_element"] = jnp.maximum(0, batch["ref_element"] - 1)
            batch_dict = batch
            batch = feat_batch.Batch.from_data_dict(batch)
            embedding_module = evoformer_network.Evoformer(self.config.evoformer, self.global_config)
            target_feat = model_mod.create_target_feat_embedding(batch=batch, config=embedding_module.config, global_config=self.global_config)

            def recycle_body(_, args):
                prev, key = args
                key, subkey = jax.random.split(key)
                embeddings = embedding_module(batch=batch, prev=prev, target_feat=target_feat, key=subkey)
                embeddings["pair"] = embeddings["pair"].astype(jnp.float32)
                embeddings["single"] = embeddings["single"].astype(jnp.float32)
                return embeddings, key

            num_res = batch.num_res
            embeddings = {"pair": jnp.zeros([num_res, num_res, self.config.evoformer.pair_channel], dtype=jnp.float32),
                          "single": jnp.zeros([num_res, self.config.evoformer.seq_channel], dtype=jnp.float32), "target_feat": target_feat}
            if hk.running_init():
                embeddings, _ = recycle_body(None, (embeddings, key))
            else:
                num_iter = self.config.num_recycles + 1
                embeddings, _ = model_mod.hk.fori_loop(0, num_iter, recycle_body, (embeddings, key))

            heads_key = hk.next_rng_key()                                    # = the stock's draw at model.py:267 (the same position in the outer sequence)

            def heads_body(batch_dict_, embeddings_, key_):
                with _manual_body(key_, lever):
                    batch_ = feat_batch.Batch.from_data_dict(batch_dict_)
                    if shard_heads:
                        _S["heads_rows"] = True                              # the diffusion head sees the LOCAL pair rows
                        try:
                            with hk.intercept_methods(_atom_encoder_interceptor):
                                denoising_step = functools.partial(self.diffusion_module, batch=batch_, embeddings=embeddings_, use_conditioning=True)
                                samples = diffusion_head.sample(denoising_step=denoising_step, batch=batch_, key=key_, config=self.config.heads.diffusion.eval)
                        finally:
                            _S["heads_rows"] = False
                    if shard_conf:
                        emb_full = embeddings_                               # the confidence + distogram heads on the local rows
                        _S["heads_rows_conf"] = True
                    else:
                        emb_full = dict(embeddings_)
                        emb_full["pair"] = _tr.rows_full(embeddings_["pair"], AXIS)   # replicated heads: one all_gather of the f32 pair
                    if not shard_heads:
                        emb_diff = emb_full if not shard_conf else dict(embeddings_, pair=_tr.rows_full(embeddings_["pair"], AXIS))
                        denoising_step = functools.partial(self.diffusion_module, batch=batch_, embeddings=emb_diff, use_conditioning=True)
                        samples = diffusion_head.sample(denoising_step=denoising_step, batch=batch_, key=key_, config=self.config.heads.diffusion.eval)
                    try:
                        confidence_output = mapping.sharded_map(
                            lambda dense_atom_positions: confidence_head.ConfidenceHead(self.config.heads.confidence, self.global_config)(
                                dense_atom_positions=dense_atom_positions, embeddings=emb_full, seq_mask=batch_.token_features.mask,
                                token_atoms_to_pseudo_beta=batch_.pseudo_beta_info.token_atoms_to_pseudo_beta, asym_id=batch_.token_features.asym_id),
                            in_axes=0)(samples["atom_positions"])
                        distogram = distogram_head.DistogramHead(self.config.heads.distogram, self.global_config)(batch_, emb_full, return_distogram=self.config.return_distogram)
                    finally:
                        _S["heads_rows_conf"] = False
                    output = {"diffusion_samples": samples, "distogram": distogram, **confidence_output}
                    if self.config.return_embeddings:
                        output["single_embeddings"] = emb_full["single"]
                        output["pair_embeddings"] = _tr.rows_full(emb_full["pair"], AXIS) if shard_conf else emb_full["pair"]
                    _S["pc16_full"] = None                                   # the precomputed pair projection lives only inside this region's trace
                    return output
            if hk.running_init():
                return heads_body(batch_dict, embeddings, heads_key)
            emb_specs = {"pair": rows, "single": rep, "target_feat": rep}
            return _shard.shard_map(heads_body, rmesh, (rep, emb_specs, rep), rep)(batch_dict, embeddings, heads_key)
        _rebind(model_mod.Model, "__call__", model_call, "Model.__call__")

    # ================================================================================================ heads
    if "heads" in sites:
        ACA = L["atom_cross_attention"]
        # ---- diffusion head on the local pair rows (needs the kit's b21 transcription)
        if shard_heads:
            def pair_half_rows(self_, batch, embeddings, use_conditioning):
                """b21.pair_conditioning_rows for this device's row block (rid = the global row ids of the block); then the atom encoder's 16-channel
                projection of the block (atom_cross_attention.py:233-244, the stock names) all-gathered to [N, N, 16] for its flattened gather (:246-267)."""
                pair = embeddings["pair"]                                    # [N/P, N, C] local rows
                n_loc = int(pair.shape[0])
                rid = _shard.axis_index(AXIS) * n_loc + jnp.arange(n_loc)
                tf = batch.token_features
                cols = {k: getattr(tf, k) for k in b21.REL_FIELDS}
                pair_cond = b21.pair_conditioning_rows(pair, rid, cols, use_conditioning, pair_channel=self_.config.conditioning.pair_channel,
                                                       global_config=self_.global_config, hm=DH.hm, DT=DT)
                c = self_.config
                _S["pc16_computing"] = True                                  # the interceptor passes these two calls through
                try:
                    pc16 = DH.hm.Linear(c.per_atom_pair_channels, precision="highest", initializer=self_.global_config.final_init, name="diffusion_embed_trunk_pair_cond")(
                        DH.hm.LayerNorm(use_fast_variance=False, create_offset=False, name="diffusion_lnorm_trunk_pair_cond")(jnp.asarray(pair_cond, dtype=jnp.float32)))
                finally:
                    _S["pc16_computing"] = False
                _S["pc16_full"] = _tr.rows_full(pc16, AXIS)                 # [N, N, 16] f32 — the only pair-shaped replicated tensor of the head
                return pair_cond
            _cond_rows = b21.conditioning_factory(pair_half_rows, DH, DT)
            stock_cond = _hk.stock_body(DH.DiffusionHead, "_conditioning")
            def cond_call(self, batch, embeddings, noise_level, use_conditioning):
                if not _S["heads_rows"]:
                    return stock_cond(self, batch, embeddings, noise_level, use_conditioning)
                return _cond_rows(self, batch, embeddings, noise_level, use_conditioning)
            patches.replace(DH.DiffusionHead, "_conditioning", cond_call)
            record["sites"].append("DiffusionHead._conditioning")

            stock_self_attention = DT.self_attention
            def self_attention_rows(x, mask, pair_logits, config, global_config, single_cond=None, name=""):
                """diffusion_transformer.self_attention (:129-186) with the queries restricted to this device's rows when pair_logits carries N/P rows
                ([H, N/P, N]): k/v over all tokens (replicated act), the gate and the adaptive zero-init on the row block, the output rows all-gathered."""
                if pair_logits is None or pair_logits.shape[-2] == x.shape[-2]:
                    return stock_self_attention(x, mask, pair_logits, config, global_config, single_cond, name)
                hm2 = DT.hm
                n_loc = int(pair_logits.shape[-2])
                start = _shard.axis_index(AXIS) * n_loc
                assert len(mask.shape) == len(x.shape) - 1, f"{mask.shape}, {x.shape}"
                bias = (1e9 * (mask - 1.0))[..., None, None, :]
                x = DT.adaptive_layernorm(x, single_cond, name=name)
                num_channels = x.shape[-1]
                key_dim = config.key_dim if config.key_dim is not None else num_channels
                value_dim = config.value_dim if config.value_dim is not None else num_channels
                num_head = config.num_head
                assert key_dim % num_head == 0 and value_dim % num_head == 0
                key_dim = key_dim // num_head
                value_dim = value_dim // num_head
                qk_shape = (num_head, key_dim)
                x_rows = jax.lax.dynamic_slice_in_dim(x, start, n_loc, axis=-2)
                sc_rows = None if single_cond is None else jax.lax.dynamic_slice_in_dim(single_cond, start, n_loc, axis=-2)
                q = hm2.Linear(qk_shape, use_bias=True, name=f"{name}q_projection")(x_rows)
                k = hm2.Linear(qk_shape, use_bias=False, name=f"{name}k_projection")(x)
                q = q.astype(jnp.float32)
                k = k.astype(jnp.float32)
                bias = bias.astype(jnp.float32)
                logits = jnp.einsum("...qhc,...khc->...hqk", q * key_dim ** (-0.5), k) + bias
                logits += pair_logits
                weights = jax.nn.softmax(logits, axis=-1)
                weights = jnp.asarray(weights, dtype=x.dtype)
                v_shape = (num_head, value_dim)
                v = hm2.Linear(v_shape, use_bias=False, name=f"{name}v_projection")(x)
                weighted_avg = jnp.einsum("...hqk,...khc->...qhc", weights, v)
                weighted_avg = jnp.reshape(weighted_avg, weighted_avg.shape[:-2] + (-1,))
                gate_logits = hm2.Linear(num_head * value_dim, bias_init=1.0, initializer="zeros", name=f"{name}gating_query")(x_rows)
                weighted_avg *= jax.nn.sigmoid(gate_logits)
                output = DT.adaptive_zero_init(weighted_avg, num_channels, sc_rows, global_config, name)
                return _shard.gather(output, AXIS, dim=-2)
            patches.replace(DT, "self_attention", self_attention_rows)
            record["sites"].append("diffusion_transformer.self_attention")

        def _atom_encoder_interceptor(next_f, args, kwargs, context):
            """atom_cross_attention.py:233-244: the encoder's LayerNorm+Linear on the (local) trunk_pair_cond are answered with the precomputed
            all-gathered [N, N, 16] so that its flattened gather (:246-267, num_tokens = shape[0]) sees the full token count."""
            if context.method_name != "__call__" or _S["pc16_computing"]:
                return next_f(*args, **kwargs)
            name = getattr(context.module, "name", "") or ""
            if _S["heads_rows"] and name.startswith("diffusion_lnorm_trunk_pair_cond"):
                return args[0]
            if _S["heads_rows"] and name.startswith("diffusion_embed_trunk_pair_cond"):
                if _S["pc16_full"] is None:
                    raise MemLeverRefused(lever, "alphafold3 recipe: the atom encoder asked for the trunk pair projection before the diffusion conditioning computed it (call order changed)")
                return _S["pc16_full"]
            return next_f(*args, **kwargs)

        # ---- confidence + distogram heads on this device's pair rows
        if shard_conf:
            def dgram_rows(pos_rows, pos_all, config):
                """template_modules.dgram_from_positions (:46-73) for the row block: distances of this device's pseudo-beta positions to all."""
                lower_breaks = jnp.linspace(config.min_bin, config.max_bin, config.num_bins)
                lower_breaks = jnp.square(lower_breaks)
                upper_breaks = jnp.concatenate([lower_breaks[1:], jnp.array([1e8], dtype=jnp.float32)], axis=-1)
                dist2 = jnp.sum(jnp.square(jnp.expand_dims(pos_rows, axis=-2) - jnp.expand_dims(pos_all, axis=-3)), axis=-1, keepdims=True)
                return (dist2 > lower_breaks).astype(jnp.float32) * (dist2 < upper_breaks).astype(jnp.float32)

            def conf_call(self, dense_atom_positions, embeddings, seq_mask, token_atoms_to_pseudo_beta, asym_id):
                """confidence_head.py:101-279 on this device's pair rows: _embed_features per row block (:72-99), the pairformer on the local block
                (PairFormerIteration in-manual), the distance-logit symmetrisation as an all_to_all transpose (:175-178), psums for the global PDE mean
                (:198-200), the tm-adjusted pae per row block (:281-329), the [N, N] outputs all-gathered; the per-token heads unchanged."""
                if not _S["heads_rows_conf"]:
                    return stock_conf(self, dense_atom_positions, embeddings, seq_mask, token_atoms_to_pseudo_beta, asym_id)
                hm_, utils_, modules_ = CH.hm, CH.utils, CH.modules
                dtype = jnp.bfloat16 if self.global_config.bfloat16 == "all" else jnp.float32
                with utils_.bfloat16_context():
                    seq_mask_cast = seq_mask.astype(dtype)
                    pair_mask = seq_mask_cast[:, None] * seq_mask_cast[None, :]
                    pair_mask = pair_mask.astype(dtype)
                    pair_act = embeddings["pair"].astype(dtype)              # [N/P, N, C] this device's rows
                    single_act = embeddings["single"].astype(dtype)
                    target_feat = embeddings["target_feat"].astype(dtype)
                    num_residues = seq_mask.shape[0]
                    n_loc = int(pair_act.shape[0])
                    start = _shard.axis_index(AXIS) * n_loc
                    pm_rows = _rows_of(pair_mask, start, n_loc)
                    num_pair_channels = pair_act.shape[2]
                    pair_act += self._embed_features(dense_atom_positions, token_atoms_to_pseudo_beta, pm_rows, pair_act, target_feat)   # the rows version below (scope ~_embed_features)

                    def pairformer_fn(act):
                        pair_act, single_act = act
                        return modules_.PairFormerIteration(self.config.pairformer, self.global_config, with_single=True, name="confidence_pairformer")(
                            act=pair_act, single_act=single_act, pair_mask=pair_mask, seq_mask=seq_mask)
                    pairformer_stack = hk.experimental.layer_stack(self.config.pairformer.num_layer)(pairformer_fn)
                    pair_act, single_act = pairformer_stack((pair_act, single_act))
                    pair_act = pair_act.astype(jnp.float32)
                    assert pair_act.shape == (n_loc, num_residues, num_pair_channels)

                    left_distance_logits = hm_.Linear(self.config.num_bins, initializer=self.global_config.final_init, name="left_half_distance_logits")(hm_.LayerNorm(name="logits_ln")(pair_act))
                    distance_logits = _tr.symmetrize(left_distance_logits, AXIS)          # left + left^T across the devices (:175-178)
                    distance_breaks = jnp.linspace(0.0, self.config.max_error_bin, self.config.num_bins - 1)
                    step = distance_breaks[1] - distance_breaks[0]
                    bin_centers = distance_breaks + step / 2
                    bin_centers = jnp.concatenate([bin_centers, bin_centers[-1:] + step], axis=0)
                    distance_probs = jax.nn.softmax(distance_logits, axis=-1)
                    pred_distance_error = jnp.sum(distance_probs * bin_centers, axis=-1) * pm_rows
                    average_pred_distance_error = _shard.psum(jnp.sum(pred_distance_error, axis=[-2, -1]), AXIS) / _shard.psum(jnp.sum(pm_rows, axis=[-2, -1]), AXIS)   # the stock's mask mean (:198-200; no eps there)

                    pae_outputs = {}
                    pae_logits = hm_.Linear(self.config.pae.num_bins, initializer=self.global_config.final_init, name="pae_logits")(hm_.LayerNorm(name="pae_logits_ln")(pair_act))
                    pae_breaks = jnp.linspace(0.0, self.config.pae.max_error_bin, self.config.pae.num_bins - 1)
                    step = pae_breaks[1] - pae_breaks[0]
                    bin_centers = pae_breaks + step / 2
                    bin_centers = jnp.concatenate([bin_centers, bin_centers[-1:] + step], axis=0)
                    pae_probs = jax.nn.softmax(pae_logits, axis=-1)
                    seq_mask_bool = seq_mask.astype(bool)
                    pair_mask_bool = seq_mask_bool[:, None] * seq_mask_bool[None, :]
                    pmb_rows = _rows_of(seq_mask_bool, start, n_loc)[:, None] * seq_mask_bool[None, :]
                    pae = jnp.sum(pae_probs * bin_centers, axis=-1) * pmb_rows
                    pae_outputs.update({"full_pae": _shard.gather(pae, AXIS, dim=0)})

                def get_tmscore_adjusted_pae(num_interface_tokens, bin_centers, pae_probs):   # outside the bfloat16 context (:230-243 + :281-329 per row block)
                    clipped_num_res = jnp.maximum(num_interface_tokens, 19)
                    d0 = 1.24 * (clipped_num_res - 15) ** (1.0 / 3) - 1.8
                    d0 = d0[:, :, None]
                    bin_centers = bin_centers[None, None, :]
                    tm_per_bin = 1.0 / (1 + jnp.square(bin_centers) / jnp.square(d0))
                    return jnp.sum(pae_probs * tm_per_bin, axis=-1)
                x_full = asym_id[None, :] == asym_id[:, None]
                num_chain_tokens = jnp.sum(x_full * pair_mask_bool, axis=-1)                    # [N] (the full row sums; N² bool: small)
                x_rows = asym_id[None, :] == _rows_of(asym_id, start, n_loc)[:, None]
                num_interface_tokens = num_chain_tokens[None, :] + _rows_of(num_chain_tokens, start, n_loc)[:, None]
                num_interface_tokens -= x_rows * (num_interface_tokens // 2)
                num_interface_tokens = num_interface_tokens * pmb_rows
                num_global_tokens = jnp.full(shape=pmb_rows.shape, fill_value=seq_mask.sum())
                assert num_global_tokens.dtype == "int32"
                assert num_interface_tokens.dtype == "int32"
                global_apae = get_tmscore_adjusted_pae(num_global_tokens, bin_centers, pae_probs)
                interface_apae = get_tmscore_adjusted_pae(num_interface_tokens, bin_centers, pae_probs)
                pae_outputs.update({"tmscore_adjusted_pae_global": _shard.gather(global_apae, AXIS, dim=0),
                                    "tmscore_adjusted_pae_interface": _shard.gather(interface_apae, AXIS, dim=0)})
                single_act = single_act.astype("float32")
                plddt_logits = hm_.Linear((dense_atom_positions.shape[-2], self.config.num_plddt_bins), initializer=self.global_config.final_init, name="plddt_logits")(hm_.LayerNorm(name="plddt_logits_ln")(single_act))
                bin_width = 1.0 / self.config.num_plddt_bins
                bin_centers = jnp.arange(0.5 * bin_width, 1.0, bin_width)
                predicted_lddt = jnp.sum(jax.nn.softmax(plddt_logits, axis=-1) * bin_centers, axis=-1)
                predicted_lddt = predicted_lddt * 100.0
                experimentally_resolved_logits = hm_.Linear((dense_atom_positions.shape[-2], 2), initializer=self.global_config.final_init, name="experimentally_resolved_logits")(hm_.LayerNorm(name="experimentally_resolved_ln")(single_act))
                predicted_experimentally_resolved = jax.nn.softmax(experimentally_resolved_logits, axis=-1)[..., 1]
                return {"predicted_lddt": predicted_lddt, "predicted_experimentally_resolved": predicted_experimentally_resolved,
                        "full_pde": _shard.gather(pred_distance_error, AXIS, dim=0), "average_pde": average_pred_distance_error, **pae_outputs}
            stock_conf = _rebind(CH.ConfidenceHead, "__call__", conf_call, "ConfidenceHead.__call__")

            def embed_features_rows(self, dense_atom_positions, token_atoms_to_pseudo_beta, pair_mask, pair_act, target_feat):
                """confidence_head.py:72-99 for this device's rows (pair_mask = the row block): out[i, j] = right(tf)[i] + left(tf)[j] + Linear(dgram[i, j]);
                the Linears live under the stock's `~_embed_features` scope because this IS that method (haiku fact 3)."""
                if not _S["heads_rows_conf"]:
                    return stock_embed(self, dense_atom_positions, token_atoms_to_pseudo_beta, pair_mask, pair_act, target_feat)
                hm_, atom_layout_ = CH.hm, CH.atom_layout
                n_loc = int(pair_act.shape[0])
                start = _shard.axis_index(AXIS) * n_loc
                left_tf = hm_.Linear(pair_act.shape[-1], name="left_target_feat_project")(target_feat).astype(pair_act.dtype)
                right_tf = hm_.Linear(pair_act.shape[-1], name="right_target_feat_project")(target_feat).astype(pair_act.dtype)
                out = left_tf[None, :, :] + _rows_of(right_tf, start, n_loc)[:, None, :]
                positions = atom_layout_.convert(token_atoms_to_pseudo_beta, dense_atom_positions, layout_axes=(-3, -2))
                dgram = dgram_rows(_rows_of(positions, start, n_loc), positions, self.config.dgram_features)
                dgram *= pair_mask[..., None]
                out += hm_.Linear(pair_act.shape[-1], name="distogram_feat_project")(dgram.astype(pair_act.dtype))
                return out
            stock_embed = _rebind(CH.ConfidenceHead, "_embed_features", embed_features_rows, "ConfidenceHead._embed_features")

            def disto_call(self, batch, embeddings, return_distogram=False):
                """distogram_head.py:55-92 on this device's pair rows: the symmetrisation as an all_to_all transpose; contact_probs (and the distogram) all-gathered."""
                if not _S["heads_rows_conf"]:
                    return stock_disto(self, batch, embeddings, return_distogram)
                hm_ = DG.hm
                pair_act = embeddings["pair"]
                seq_mask = batch.token_features.mask.astype(bool)
                n_loc = int(pair_act.shape[0])
                start = _shard.axis_index(AXIS) * n_loc
                pair_mask = _rows_of(seq_mask, start, n_loc)[:, None] * seq_mask[None, :]
                left_half_logits = hm_.Linear(self.config.num_bins, initializer=self.global_config.final_init, name="half_logits")(pair_act)
                logits = _tr.symmetrize(left_half_logits, AXIS)
                probs = jax.nn.softmax(logits, axis=-1)
                breaks = jnp.linspace(self.config.first_break, self.config.last_break, self.config.num_bins - 1)
                bin_tops = jnp.append(breaks, breaks[-1] + (breaks[-1] - breaks[-2]))
                threshold = DG._CONTACT_THRESHOLD + DG._CONTACT_EPSILON
                is_contact_bin = 1.0 * (bin_tops <= threshold)
                contact_probs = jnp.einsum("ijk,k->ij", probs, is_contact_bin, precision=jax.lax.Precision.HIGHEST)
                contact_probs = pair_mask * contact_probs
                return_dict = {"bin_edges": breaks, "contact_probs": _shard.gather(contact_probs, AXIS, dim=0)}
                if return_distogram:
                    return_dict["distogram"] = _shard.gather(logits, AXIS, dim=0)
                return return_dict
            stock_disto = _rebind(DG.DistogramHead, "__call__", disto_call, "DistogramHead.__call__")
    else:
        def _atom_encoder_interceptor(next_f, args, kwargs, context):     # model group without the heads group: the interceptor is inert
            return next_f(*args, **kwargs)

    return record


def _return_arity(fn) -> int:
    """How many values ``fn``'s own top-level ``return`` statements yield (a tuple literal's length, else 1); mixed arities → 0 (refused by the caller)."""
    import ast as _ast  # noqa: PLC0415
    import inspect  # noqa: PLC0415
    import textwrap  # noqa: PLC0415
    tree = _ast.parse(textwrap.dedent(inspect.getsource(fn)))
    top = tree.body[0]
    arities = set()

    def visit(node):
        for child in _ast.iter_child_nodes(node):
            if isinstance(child, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.Lambda, _ast.ClassDef)):
                continue                                                      # nested definitions' returns are not the method's (``top`` itself is never a child)
            if isinstance(child, _ast.Return) and child.value is not None:
                arities.add(len(child.value.elts) if isinstance(child.value, _ast.Tuple) else 1)
            visit(child)
    visit(top)
    return arities.pop() if len(arities) == 1 else 0


def install_fields(record: Dict[str, Any]) -> Dict[str, Any]:
    """Blank-free lever-line evidence from :func:`install`'s record: ``sites=<n installed> heads=… conf=… diffusion=… schedule=…`` (pass ``sites`` itself
    as the ``sites=`` key of :func:`evidence.line` for the full list)."""
    return {"heads": record["heads"], "conf": record["conf"], "diffusion": record["diffusion"], "n_sites": len(record["sites"]), "library": record["library"]}
