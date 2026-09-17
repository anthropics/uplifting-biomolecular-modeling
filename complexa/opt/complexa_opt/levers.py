"""The levers — what ``exact`` / ``fast`` / ``big`` change inside upstream's generation process (``modes.KIT_MODES`` names each mode's set).

Every lever is a CLASS or MODULE-FUNCTION patch of upstream's pinned code, applied by ``install()`` right after ``proteinfoundation.proteina``
has been imported in the ``python -m proteinfoundation.generate`` process (``activate.enable``, fired by the autoload hook), before the
checkpoint loads; nothing edits the stock tree, nothing is staged. A lever that cannot be installed as written (an upstream function whose
source no longer carries the statement a lever rewrites, a class that lost the method a lever replaces) raises ``LeverError`` naming lever
and reason — the mode then refuses by name (all of its levers or none, ``activate.enable``); a lever never drops out quietly.

    lever            tier       replaces (upstream 1.1.0 @ 916eaaed)                                    effect
    onehot_f32       exact      feature_utils.bin_and_one_hot (and its by-name imports)                 the one-hot featurization tensors are
                                                                                                          built as float32 0.0/1.0 directly (stock:
                                                                                                          int64 one-hot, then `* 1.0`): identical
                                                                                                          values, no int64 [*, C] temporaries
    target_hoist     exact      ConcatPairFeaturesFactory.forward                                        the target x target pair block and the
                                                                                                          all-zero binder x target sequence-separation
                                                                                                          block — pure functions of the fixed target
                                                                                                          — are computed once per predict_step and
                                                                                                          reused by the other 399 steps
    pair_assembly    exact      ConcatPairFeaturesFactory.forward                                        with uniform masks (equal binder lengths in
                                                                                                          the batch: a fixed binder_length [L, L])
                                                                                                          the extended pair tensor is assembled by
                                                                                                          four block copies instead of three padded
                                                                                                          concatenations (each a host sync, two
                                                                                                          boolean gathers and mask multiplies by
                                                                                                          1.0); a padded batch (a binder_length
                                                                                                          range: one length drawn per design) takes
                                                                                                          stock's path on that call — counted
                                                                                                          (``padded``), and a run whose every call
                                                                                                          was padded reports the lever ``state=skipped
                                                                                                          reason=padded_batches``: its declared gate
                                                                                                          (``GATES``), same bytes, exit unaffected
    loop_desync      exact      ProductSpaceFlowMatcher.full_simulation, RDNFlowMatcher.simulation_step, the sampling loop branches on the CPU
                                rdn_flow_matcher.vf_to_score / score_to_vf (re-created from their own    float32 schedule value of the step instead
                                source with the statements below rewritten)                              of reading a CUDA scalar back (3-4 host
                                                                                                          syncs per data mode per step); comparisons
                                                                                                          stay float32 tensor vs python float, so
                                                                                                          every branch decides as stock does
    pair_bias_rows   exact      PairBiasAttention.forward (the source of the attention bias)             stock's per-layer pair LayerNorm -> to_bias
                     (memory)                                                                            evaluated on contiguous row blocks and
                                                                                                          written straight into [b, h, n, n]: the
                                                                                                          [b, n, n, 256] LayerNorm output per layer is
                                                                                                          never materialised (bitwise: cuBLAS's
                                                                                                          K-accumulation for this GEMM does not depend
                                                                                                          on M — holds on H100 and A100)
    pair_bias_fused  tolerance  PairBiasAttention.forward (the source of the attention bias)             the pair representation is constant across
                                                                                                          the 14 layers (update_pair_repr false), so
                                                                                                          its LayerNorm statistics are computed ONCE
                                                                                                          per forward and all layers' gamma-folded
                                                                                                          bias projections run as one [14*12, 256]
                                                                                                          GEMM (row-chunked); LayerNorm affine folded
                                                                                                          into the weights => reassociation-level
                                                                                                          differences (|d bias| ~ 1e-6)
    attn_sdpa        tolerance  PairBiasAttention._attn                                                  softmax(q k^T d^-1/2 + bias) v through
                                                                                                          torch's scaled_dot_product_attention with
                                                                                                          the bias (and -1e4 at masked pairs, as
                                                                                                          upstream's own flash variant does; skipped
                                                                                                          when the pair mask is all ones) as one float
                                                                                                          mask — online softmax, not bitwise

Plumbing every kit mode installs (not levers, no numeric effect): a ``Proteina.predict_step`` wrapper that opens the per-step memo scope the
hoist / uniform-mask checks live in and counts the steps, and a ``LocalLatentsTransformer.forward`` wrapper that wires the pair-bias bank onto
the transformer's attention layers at its first call, marks the bank stale at every call, and counts the forwards. Counters (``census()``)
feed the LEVER / TALLY lines and the activation record (``activate.py``). ``torch`` and ``proteinfoundation`` are imported inside functions
only: the registry is importable on a CPU box without either (the package's tests read it).
"""
from __future__ import annotations

import contextlib
import functools
import inspect
import re
import textwrap
import types
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

EXACT, TOLERANCE = "exact", "tolerance"


class LeverError(RuntimeError):
    """A lever that cannot be installed as written: ``.lever`` and the reason (the mode refuses by name)."""

    def __init__(self, lever: str, reason: str):
        super().__init__(f"lever {lever}: {reason}")
        self.lever, self.reason = lever, reason


class Lever:
    """One lever: its tier, its words, its installer; ``zeroed`` names counters its LEVER line carries even at zero (``install`` seeds them)."""
    __slots__ = ("name", "tier", "words", "installer", "zeroed")

    def __init__(self, name: str, tier: str, words: str, installer: Callable[[], None], zeroed: Sequence[str] = ()):
        self.name, self.tier, self.words, self.installer, self.zeroed = name, tier, words, installer, tuple(zeroed)


_STATS: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))     # lever -> counter -> n
_ORIG: dict = {}                                                              # (owner, attr) -> upstream's original object
_ACTIVE: set = set()                                                          # the installed lever names (read at call time by the shared patches)
_SCOPE = types.SimpleNamespace(armed=False, store={}, fms=[])                 # the per-predict_step memo and the flow matchers holding a CPU schedule value
_COUNTS = {"predict_steps": 0, "forwards": 0}
ROWS_BYTES = {"pair_bias_rows": 64 << 20, "pair_bias_fused": 256 << 20}      # LayerNorm output materialised per row block (bytes): the block size is free of numeric effect


# ---------------------------------------------------------------------------------------------------------------------------- plumbing
def _patch(owner, attr: str, new, lever: str) -> None:
    """Replace ``owner.attr`` by ``new`` once, keeping the original; a missing attribute is a LeverError naming the lever."""
    key = (owner, attr)
    if key in _ORIG:
        return
    if not hasattr(owner, attr):
        raise LeverError(lever, f"{getattr(owner, '__name__', owner)!r} has no attribute {attr!r} (upstream changed)")
    _ORIG[key] = getattr(owner, attr)
    setattr(owner, attr, new)


def _orig(owner, attr: str):
    return _ORIG[(owner, attr)]


CHECK = "check_"                                                              # the counter prefix of a memoised per-step CHECK (a mask test a lever decides on): never engagement


def _memo(key, fn, lever: str, counter: str = ""):
    """Compute ``fn()`` once per predict_step (``_SCOPE``) under ``key``; outside a scope it is simply computed (counted as unscoped). The
    lever's counters ``<counter>computed`` / ``<counter>served`` / ``<counter>unscoped`` book it: bare (``counter=""``) when the memoised value
    IS the lever's work (target_hoist's blocks), ``CHECK`` when it is only a test the lever branches on (pair_assembly's uniform-mask check,
    attn_sdpa's pair-mask check) — a check served from the memo is not the lever engaging (``ENGAGED_KEYS``)."""
    if not _SCOPE.armed:
        _STATS[lever][counter + "unscoped"] += 1
        return fn()
    if key not in _SCOPE.store:
        _SCOPE.store[key] = fn()
        _STATS[lever][counter + "computed"] += 1
    else:
        _STATS[lever][counter + "served"] += 1
    return _SCOPE.store[key]


@contextlib.contextmanager
def predict_step_scope():
    _SCOPE.armed, _SCOPE.store = True, {}
    try:
        yield _SCOPE
    finally:
        _SCOPE.armed, _SCOPE.store = False, {}
        for fm in _SCOPE.fms:                                  # a step's CPU schedule value never outlives its predict_step
            fm.__dict__.pop("_kit_t_cpu", None)
        _SCOPE.fms = []


def _retarget(func, module, edits: Sequence, lever: str, extra_globals=None):
    """Re-create ``func`` from ITS OWN source with the given (regex, replacement) edits, compiled in its module's namespace. An anchor
    that matches nothing means upstream's statement is not the one the lever was written against: LeverError (never a silent pass)."""
    try:
        src = textwrap.dedent(inspect.getsource(inspect.unwrap(func)))
    except (OSError, TypeError) as e:
        raise LeverError(lever, f"cannot read the source of {module.__name__}.{getattr(func, '__qualname__', func)}: {e}") from None
    for pattern, repl in edits:
        src, n = re.subn(pattern, repl, src, flags=re.M)
        if n < 1:
            raise LeverError(lever, f"anchor not found in {module.__name__}.{func.__qualname__} (upstream changed): {pattern!r}")
    ns_glob = module.__dict__
    if extra_globals:
        ns_glob.update(extra_globals)
    ns: dict = {}
    exec(compile(src, f"<complexa_opt.levers:{module.__name__}.{func.__qualname__}>", "exec"), ns_glob, ns)
    return ns[func.__name__]


# ------------------------------------------------------------------------------------------------------------------------- onehot_f32
def _bin_and_one_hot_f32(tensor, bin_limits):
    import torch
    idx = torch.bucketize(tensor, bin_limits)
    out = torch.zeros(*idx.shape, len(bin_limits) + 1, dtype=torch.get_default_dtype(), device=tensor.device)
    out.scatter_(-1, idx.unsqueeze(-1), 1.0)
    _STATS["onehot_f32"]["served"] += 1
    return out


def _install_onehot_f32():
    import importlib
    import pkgutil

    import proteinfoundation.nn.feature_factory as ffpkg
    import proteinfoundation.nn.feature_factory.feature_utils as fu

    stock = fu.bin_and_one_hot
    _patch(fu, "bin_and_one_hot", _bin_and_one_hot_f32, "onehot_f32")
    for m in pkgutil.iter_modules(ffpkg.__path__):                      # the feature modules bind the function BY NAME at import
        mod = importlib.import_module(f"{ffpkg.__name__}.{m.name}")
        if mod is not fu and getattr(mod, "bin_and_one_hot", None) is stock:
            _patch(mod, "bin_and_one_hot", _bin_and_one_hot_f32, "onehot_f32")


# ---------------------------------------------------------------------------------------------------- target_hoist + pair_assembly
def _cpf_forward(self, batch, orig_pair_rep, orig_seq_mask):
    """ConcatPairFeaturesFactory.forward for the protein-target configuration, statement for statement upstream's except the memoised
    target block / zero block (target_hoist) and the block-copy assembly under uniform masks (pair_assembly); any other configuration
    (motif, ligand, no target coordinates in the batch) is upstream's own forward."""
    import torch
    cls = type(self)
    stock = _orig(cls, "forward")
    hoist, lean = "target_hoist" in _ACTIVE, "pair_assembly" in _ACTIVE
    if not (self.enable_target and not self.enable_motif and not getattr(self, "enable_ligand", False)) \
            or self.coords_key not in batch or self.mask_key not in batch:
        _STATS["target_hoist" if hoist else "pair_assembly"]["fallback_stock"] += 1
        return stock(self, batch, orig_pair_rep, orig_seq_mask)
    from proteinfoundation.utils.tensor_utils import concat_padded_tensor

    memo = (lambda key, fn: _memo(key, fn, "target_hoist")) if hoist else (lambda key, fn: fn())
    _, _, _, pair_dim = orig_pair_rep.shape
    batch_with_chains = self._prepare_batch_with_chains(batch)
    ur_feat = self.upper_right_seq_sep
    if ur_feat.idx1_key != ur_feat.idx2_key:                                 # upstream returns torch.zeros(b, n1, n2, seq_sep_dim) for this block
        b_, n1 = orig_seq_mask.shape
        n2 = batch[self.mask_key].shape[1]
        upper_right_seq_sep = memo(("ur_seq_sep_zeros", b_, n1, n2, ur_feat.seq_sep_dim, str(orig_pair_rep.device)), lambda: ur_feat(batch_with_chains))
    else:
        upper_right_seq_sep = ur_feat(batch_with_chains)
    upper_right_combined = torch.cat([upper_right_seq_sep, self.upper_right_xt_dist(batch_with_chains), self.upper_right_chain(batch_with_chains),
                                      self.upper_right_hotspots(batch_with_chains)], dim=-1)
    upper_right_projected = self.ln_out(self.linear_out(upper_right_combined))

    def _lower_right():
        lower_right_combined = torch.cat([self.lower_right_seq_sep(batch_with_chains), self.lower_right_xt_dist(batch_with_chains),
                                          self.lower_right_chain(batch_with_chains), self.lower_right_hotspots(batch_with_chains)], dim=-1)
        return self.ln_out(self.linear_out(lower_right_combined))

    tgt = batch[self.coords_key]
    lower_right_projected = memo(("lower_right_projected", tuple(tgt.shape), str(tgt.device)), _lower_right)
    if self.dim_pair_out != pair_dim:
        raise ValueError(f"Configured output dimension {self.dim_pair_out} does not match pair dim {pair_dim}")
    concat_mask = batch[self.mask_key].sum(dim=-1).bool()
    uniform = False
    if lean:                                                                 # the uniform-mask CHECK, once per predict_step: booked as a check, never as the lever's work
        uniform = _memo(("uniform_masks", tuple(orig_seq_mask.shape), tuple(concat_mask.shape)),
                        lambda: bool(orig_seq_mask.all()) and bool(concat_mask.all()), "pair_assembly", CHECK)
    if uniform:
        b_, n_o = orig_seq_mask.shape
        n_c = concat_mask.shape[1]
        out = torch.empty(b_, n_o + n_c, n_o + n_c, pair_dim, dtype=orig_pair_rep.dtype, device=orig_pair_rep.device)
        out[:, :n_o, :n_o] = orig_pair_rep
        out[:, :n_o, n_o:] = upper_right_projected
        out[:, n_o:, :n_o] = upper_right_projected.transpose(1, 2)
        out[:, n_o:, n_o:] = lower_right_projected
        _STATS["pair_assembly"]["assembled"] += 1                             # the lever's work: this call's pair tensor by block copies
        return out
    if lean:
        _STATS["pair_assembly"]["padded"] += 1                                # binder lengths differ within the batch (a binder_length range): upstream's padded assembly below, the lever idle on this call (GATES)
    lower_left_projected = upper_right_projected.transpose(1, 2)
    orig_pair_rep = orig_pair_rep * orig_seq_mask[:, :, None, None] * orig_seq_mask[:, None, :, None]
    lower_left_projected = lower_left_projected * concat_mask[:, :, None, None] * orig_seq_mask[:, None, :, None]
    extended_pair_rep_left, extended_mask = concat_padded_tensor(a=orig_pair_rep, b=lower_left_projected, mask_a=orig_seq_mask, mask_b=concat_mask)
    upper_right_projected = upper_right_projected * orig_seq_mask[:, :, None, None] * concat_mask[:, None, :, None]
    lower_right_projected = lower_right_projected * concat_mask[:, :, None, None] * concat_mask[:, None, :, None]
    extended_pair_rep_right, extended_mask = concat_padded_tensor(a=upper_right_projected, b=lower_right_projected, mask_a=orig_seq_mask, mask_b=concat_mask)
    extended_pair_rep, extended_mask = concat_padded_tensor(a=extended_pair_rep_left.transpose(1, 2), b=extended_pair_rep_right.transpose(1, 2),
                                                            mask_a=orig_seq_mask, mask_b=concat_mask)
    extended_pair_rep = extended_pair_rep.transpose(1, 2)
    return extended_pair_rep * extended_mask[:, :, None, None] * extended_mask[:, None, :, None]


def _install_pair_factory(lever: str):
    from proteinfoundation.nn.feature_factory.concat_pair_feature_factory import ConcatPairFeaturesFactory
    _patch(ConcatPairFeaturesFactory, "forward", _cpf_forward, lever)


# ------------------------------------------------------------------------------------------------------------------------ loop_desync
def _kit_set_t_cpu(psfm, ts, step):
    """Inserted into full_simulation right after ``dt = …``: hand each base flow matcher this step's CPU float32 schedule value."""
    for dm in psfm.data_modes:
        bfm = psfm.base_flow_matchers[dm]
        bfm._kit_t_cpu = ts[dm][step]                                        # 0-dim float32 CPU tensor: the value t = ts[step] * ones(device) carries
        if bfm not in _SCOPE.fms:
            _SCOPE.fms.append(bfm)


def _kit_pick_t(bfm, t):
    """Replaces ``t_element = t.flatten()[0]; assert torch.all(t_element == t)`` in RDNFlowMatcher.simulation_step."""
    t_cpu = getattr(bfm, "_kit_t_cpu", None)
    if t_cpu is not None:
        assert bool(t_cpu < 1.0) and bool(t_cpu >= 0.0)                     # vf_to_score / score_to_vf's time preconditions, checked host-side
        _STATS["loop_desync"]["served"] += 1
        return t_cpu
    import torch                                                             # a simulation_step outside full_simulation: upstream's own statements
    _STATS["loop_desync"]["fallback_stock"] += 1
    t_element = t.flatten()[0]
    assert torch.all(t_element == t), "Sampling only implemented for same time for all samples"
    return t_element


def _kit_skip_device_assert(cond_fn):
    """Replaces the tensor-valued asserts of vf_to_score / score_to_vf (their precondition is checked host-side in ``_kit_pick_t``)."""
    return None


def _install_loop_desync():
    import proteinfoundation.flow_matching.product_space_flow_matcher as psm
    import proteinfoundation.flow_matching.rdn_flow_matcher as rdn

    g = {"_kit_set_t_cpu": _kit_set_t_cpu, "_kit_pick_t": _kit_pick_t, "_kit_skip_device_assert": _kit_skip_device_assert}
    L = "loop_desync"
    full_sim = _retarget(psm.ProductSpaceFlowMatcher.full_simulation, psm,
                         [(r"^(\s*)(dt = \{data_mode: ts\[data_mode\]\[step \+ 1\] - ts\[data_mode\]\[step\] for data_mode in self\.data_modes\})$",
                           r"\1\2\n\1_kit_set_t_cpu(self, ts, step)")], L, g)
    sim_step = _retarget(rdn.RDNFlowMatcher.simulation_step, rdn,
                         [(r"^(\s*)t_element = t\.flatten\(\)\[0\]\n\s*assert torch\.all\(t_element == t\), \"Sampling only implemented for same time for all samples\"$",
                           r"\1t_element = _kit_pick_t(self, t)")], L, g)
    v2s = _retarget(rdn.vf_to_score, rdn, [(r"^(\s*)assert torch\.all\(t < 1\.0\)[^\n]*$", r"\1_kit_skip_device_assert(lambda: t < 1.0)")], L, g)
    s2v = _retarget(rdn.score_to_vf, rdn, [(r"^(\s*)assert torch\.all\(t > 0\.0\)[^\n]*$", r"\1_kit_skip_device_assert(lambda: t > 0.0)")], L, g)
    _patch(psm.ProductSpaceFlowMatcher, "full_simulation", full_sim, L)
    _patch(rdn.RDNFlowMatcher, "simulation_step", sim_step, L)
    _patch(rdn, "vf_to_score", v2s, L)
    _patch(rdn, "score_to_vf", s2v, L)


# ------------------------------------------------------------------------------------------------ pair_bias_rows / pair_bias_fused
class PairBiasBank:
    """Per-LocalLatentsTransformer source of the attention biases: ``rows_bias`` (pair_bias_rows) evaluates stock's per-layer
    LayerNorm -> Linear on contiguous row blocks; ``fused_bias`` (pair_bias_fused) computes every layer's bias once per forward into one
    buffer (allocated once per shape, refilled in place) and serves layer slices of it."""

    def __init__(self, mhas):
        self.mhas, self.buf, self.fresh, self._folded = mhas, None, False, None

    def reset(self):
        self.fresh = False

    @staticmethod
    def _rows(nbytes: int, n: int, d: int, itemsize: int, b: int = 1) -> int:
        return max(1, nbytes // max(1, b * n * d * itemsize))

    def rows_bias(self, mha, pair_feats):
        import torch
        b, n, _, d = pair_feats.shape
        rows = self._rows(ROWS_BYTES["pair_bias_rows"], n, d, pair_feats.element_size())
        out = torch.empty(b, mha.heads, n, n, dtype=pair_feats.dtype, device=pair_feats.device)
        for bi in range(b):
            for i0 in range(0, n, rows):
                blk = mha.to_bias(mha.pair_norm(pair_feats[bi, i0:i0 + rows]))          # [r, n, h]: stock's kernels on M = r*n rows
                out[bi, :, i0:i0 + rows, :] = blk.permute(2, 0, 1)
        _STATS["pair_bias_rows"]["served"] += 1
        return out

    def fused_bias(self, layer: int, pair_feats):
        import torch
        import torch.nn.functional as F
        if not self.fresh:
            b, n, _, d = pair_feats.shape
            if self._folded is None:
                W = torch.cat([m.to_bias.weight * m.pair_norm.weight[None, :] for m in self.mhas], 0)        # [L*h, d]: gamma folded into the projection
                c = torch.cat([m.to_bias.weight @ m.pair_norm.bias for m in self.mhas], 0)                    # [L*h]:   beta through the projection
                self._folded = (W.contiguous(), c.contiguous(), self.mhas[0].pair_norm.eps)
            W, c, eps = self._folded
            Lh = W.shape[0]
            if self.buf is None or tuple(self.buf.shape) != (b, Lh, n, n) or self.buf.device != pair_feats.device:
                self.buf = torch.empty(b, Lh, n, n, dtype=pair_feats.dtype, device=pair_feats.device)
                _STATS["pair_bias_fused"]["alloc"] += 1
            rows = self._rows(ROWS_BYTES["pair_bias_fused"], n, d, pair_feats.element_size(), b)
            for i0 in range(0, n, rows):
                ph = F.layer_norm(pair_feats[:, i0:i0 + rows], (d,), None, None, eps)                        # [b, r, n, d], statistics once for all layers
                r = ph.shape[1]
                blk = torch.matmul(W, ph.reshape(b, r * n, d).transpose(1, 2))                               # [b, L*h, r*n]
                blk += c[None, :, None]
                self.buf[:, :, i0:i0 + rows, :] = blk.view(b, Lh, r, n)
            self.fresh = True
            _STATS["pair_bias_fused"]["computed"] += 1
        else:
            _STATS["pair_bias_fused"]["served"] += 1
        h = self.mhas[layer].heads
        return self.buf[:, layer * h:(layer + 1) * h]


def _pba_forward(self, node_feats, pair_feats, mask):
    """PairBiasAttention.forward: upstream's statements; only the source of the bias ``b`` differs (the bank wired at the transformer's
    first forward). A PairBiasAttention outside a wired transformer, or called without pair features, is upstream's own forward."""
    import torch
    from einops import rearrange
    bank, layer = getattr(self, "_kit_bank", None), getattr(self, "_kit_layer", None)
    if bank is None or pair_feats is None:
        _STATS["pair_bias_rows" if "pair_bias_rows" in _ACTIVE else "pair_bias_fused"]["fallback_stock"] += 1
        return _orig(type(self), "forward")(self, node_feats, pair_feats, mask)
    node_feats = self.node_norm(node_feats)
    q, k, v = self.to_qkv(node_feats).chunk(3, dim=-1)
    q = self.q_layer_norm(q)
    k = self.k_layer_norm(k)
    g = self.to_g(node_feats)
    b = bank.rows_bias(self, pair_feats) if "pair_bias_rows" in _ACTIVE else bank.fused_bias(layer, pair_feats)
    h = self.heads
    q, k, v, g = map(lambda t: rearrange(t, "b ... (h d) -> b h ... d", h=h), (q, k, v, g))
    attn_feats = self._attn(q, k, v, b, mask)
    attn_feats = rearrange(torch.sigmoid(g) * attn_feats, "b h n d -> b n (h d)", h=h)
    return self.to_out_node(attn_feats)


def _install_pair_bias(lever: str):
    from proteinfoundation.nn.modules.pair_bias_attn import PairBiasAttention
    _patch(PairBiasAttention, "forward", _pba_forward, lever)


def _wire_bank(nn_module) -> None:
    """At a LocalLatentsTransformer's first forward: one bank over its layers' PairBiasAttention modules (only when every layer's attention
    IS that class — upstream's flash / cuEquivariance variants keep upstream's path and the census says so)."""
    from proteinfoundation.nn.modules.pair_bias_attn import PairBiasAttention
    lever = "pair_bias_rows" if "pair_bias_rows" in _ACTIVE else "pair_bias_fused"
    if "pair_bias_fused" in _ACTIVE and getattr(nn_module, "update_pair_repr", False):
        _STATS[lever]["not_wired_pair_updates"] += 1                        # a pair representation updated per layer has per-layer statistics: stock's path
        return
    mhas = [layer.mhba.mha for layer in nn_module.transformer_layers]
    if not all(type(m) is PairBiasAttention for m in mhas):
        _STATS[lever]["not_wired_attention_class"] += 1
        return
    bank = PairBiasBank(mhas)
    for i, m in enumerate(mhas):
        m._kit_bank, m._kit_layer = bank, i
    nn_module._kit_bank = bank
    _STATS[lever]["wired"] += 1


# ------------------------------------------------------------------------------------------------------------------------- attn_sdpa
def _attn_sdpa(self, q, k, v, b, mask):
    import torch
    import torch.nn.functional as F
    attn_mask = None
    if torch.is_tensor(b):
        attn_mask = b
        if mask is not None:
            uniform = _memo(("pair_mask_uniform", tuple(mask.shape), str(mask.device)), lambda: bool(mask.all()), "attn_sdpa", CHECK)
            if not uniform:
                attn_mask = b.masked_fill(~mask[:, None, :, :], -1e4)
                _STATS["attn_sdpa"]["masked"] += 1
        if attn_mask.dtype != q.dtype:
            attn_mask = attn_mask.to(q.dtype)
    elif mask is not None:
        attn_mask = mask[:, None, :, :]
    _STATS["attn_sdpa"]["served"] += 1
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)


def _install_attn_sdpa():
    from proteinfoundation.nn.modules.pair_bias_attn import PairBiasAttention
    _patch(PairBiasAttention, "_attn", _attn_sdpa, "attn_sdpa")


# ---------------------------------------------------------------------------------------------------------------------------- registry
LEVERS: Dict[str, Lever] = {L.name: L for L in (
    Lever("onehot_f32", EXACT, "one-hot featurization built as float32 directly (no int64 temporaries)", _install_onehot_f32),
    Lever("target_hoist", EXACT, "target x target pair block computed once per predict_step", lambda: _install_pair_factory("target_hoist")),
    Lever("pair_assembly", EXACT, "extended pair tensor assembled by block copies under uniform masks", lambda: _install_pair_factory("pair_assembly"), zeroed=("assembled", "padded")),
    Lever("loop_desync", EXACT, "sampling loop branches on the CPU schedule value (no per-step device read-back)", _install_loop_desync),
    Lever("pair_bias_rows", EXACT, "per-layer pair LayerNorm->bias on row blocks (the [b,n,n,256] LayerNorm output never materialised)", lambda: _install_pair_bias("pair_bias_rows")),
    Lever("pair_bias_fused", TOLERANCE, "pair LayerNorm statistics once per forward, all layers' biases in one folded GEMM", lambda: _install_pair_bias("pair_bias_fused")),
    Lever("attn_sdpa", TOLERANCE, "pair-biased attention through scaled_dot_product_attention (bias+mask as one float mask)", _install_attn_sdpa),
)}
NEEDS_BANK = ("pair_bias_rows", "pair_bias_fused")


def tier_of(names: Sequence[str]) -> str:
    """``exact`` when every lever of the set is exact-class, else ``tolerance``."""
    return EXACT if all(LEVERS[n].tier == EXACT for n in names) else TOLERANCE


def _llt_forward_wrapper(orig):
    @functools.wraps(orig)
    def forward(self, *args, **kwargs):
        _COUNTS["forwards"] += 1
        if any(n in _ACTIVE for n in NEEDS_BANK):
            if not getattr(self, "_kit_wired", False):
                _wire_bank(self)
                self._kit_wired = True
            bank = getattr(self, "_kit_bank", None)
            if bank is not None:
                bank.reset()                                                 # a new forward: biases are recomputed (the buffer is kept)
        return orig(self, *args, **kwargs)
    return forward


def _predict_step_wrapper(orig):
    @functools.wraps(orig)
    def predict_step(self, *args, **kwargs):
        _COUNTS["predict_steps"] += 1
        with predict_step_scope():
            return orig(self, *args, **kwargs)
    return predict_step


def _install_plumbing():
    import importlib
    from proteinfoundation.proteina import Proteina
    for modname in ("proteinfoundation.nn.local_latents_transformer", "proteinfoundation.nn.local_latents_transformer_v2"):
        try:
            mod = importlib.import_module(modname)
        except ImportError:
            continue
        cls = mod.LocalLatentsTransformer
        _patch(cls, "forward", _llt_forward_wrapper(cls.forward), "plumbing")
    _patch(Proteina, "predict_step", _predict_step_wrapper(Proteina.predict_step), "plumbing")


def install(names: Sequence[str]) -> List[str]:
    """Install every lever of ``names`` (all of them: the first LeverError propagates and the caller refuses the mode) plus the
    per-step / per-forward plumbing; returns the names installed, in order. Idempotent per process."""
    unknown = [n for n in names if n not in LEVERS]
    if unknown:
        raise LeverError(",".join(unknown), f"unknown lever(s); the registry holds {'|'.join(LEVERS)}")
    if sum(n in NEEDS_BANK for n in names) > 1:
        raise LeverError("pair_bias", "pair_bias_rows and pair_bias_fused are alternative sources of the same bias: one per mode")
    for n in names:
        if n in _ACTIVE:
            continue
        try:
            LEVERS[n].installer()
        except LeverError:
            raise
        except Exception as e:                                               # an import that fails, a class that moved: the lever's name on the refusal
            raise LeverError(n, f"{type(e).__name__}: {e}") from None
        for k in LEVERS[n].zeroed:                                           # counters its LEVER line carries even when they never move (a true 0, not an absent key)
            _STATS[n][k] += 0
        _ACTIVE.add(n)
    _install_plumbing()
    return [n for n in names]


def active() -> List[str]:
    return [n for n in LEVERS if n in _ACTIVE]


def counts() -> dict:
    return dict(_COUNTS)


def census() -> Dict[str, Dict[str, int]]:
    """``{lever: {counter: n}}`` for every installed lever (an installed lever with no counter yet reads ``{}``)."""
    return {n: dict(_STATS.get(n, {})) for n in active()}


ENGAGED_KEYS = ("served", "computed", "assembled", "wired")                  # a lever counts as engaged when any of these moved (a CHECK-prefixed counter never does)


class Gate:
    """A lever's DECLARED input gate: the ``counter`` of calls its condition sent to upstream's own path, the ``reason`` word its LEVER line
    carries when every call went that way (``state=skipped reason=<reason>``), and the words a report says it with."""
    __slots__ = ("counter", "reason", "words")

    def __init__(self, counter: str, reason: str, words: str):
        self.counter, self.reason, self.words = counter, reason, words


GATES: Dict[str, Gate] = {                                                    # lever -> its declared gate (README 'Modes', CHANGES.md): idle by input, same bytes, exit unaffected
    "pair_assembly": Gate("padded", "padded_batches",
                          "binder lengths differ within a batch (a binder_length range [low, high] draws one length per design), so upstream's padded pair "
                          "assembly ran and the lever stayed idle — the same bytes at upstream's cost for that step; a fixed binder_length [L, L] engages it"),
}
GATE_REASONS = frozenset(g.reason for g in GATES.values())                   # the reason words that name a declared gate (never_engaged is not one: it is a partial activation)
NEVER_ENGAGED = "never_engaged"


def engaged(name: str) -> bool:
    c = _STATS.get(name, {})
    return any(c.get(k, 0) > 0 for k in ENGAGED_KEYS)


def status(name: str) -> Tuple[str, Optional[str]]:
    """A lever's state word and reason at report time — THE one reader of its counters: ``("on", None)`` when it engaged on at least one
    call; ``("skipped", <its gate's reason>)`` when it never engaged and its declared gate (``GATES``) took at least one call — idle by input,
    a named gate, not a partial activation; ``("skipped", "never_engaged")`` when its site was never reached (the mode is then partial)."""
    if engaged(name):
        return "on", None
    g = GATES.get(name)
    if g is not None and _STATS.get(name, {}).get(g.counter, 0) > 0:
        return "skipped", g.reason
    return "skipped", NEVER_ENGAGED


def gated(name: str) -> bool:
    """True when the lever's state is its declared gate's (``status`` says skipped with a ``GATE_REASONS`` word)."""
    state, reason = status(name)
    return state == "skipped" and reason in GATE_REASONS
