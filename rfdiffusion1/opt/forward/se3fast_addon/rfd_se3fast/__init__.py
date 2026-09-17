"""
rfd_se3fast — add-on for RFdiffusion v1 (RosettaCommons/RFdiffusion @86507b65) that replaces the DGL-based
SE(3)-Transformer evaluation inside Str2Str with a dense destination-major formulation (+ optional Triton kernels).

    RFD_SE3FAST=0        : nothing is patched (kit / stock path)
    RFD_SE3FAST=exact    : reserved for byte-equal levers (none exist -> behaves like 0 and says so)
    RFD_SE3FAST=t2       : Tier-2 dense path, Triton fused radial kernels           (falls back to t2torch if Triton is unusable)
    RFD_SE3FAST=t2torch  : Tier-2 dense path, pure torch (no Triton)
  Optional: RFD_SE3FAST_SCOPE=full|all   (default all: 36 full-graph calls AND the 4 top-k refinement calls per step)

Use: `import rfd_se3fast; rfd_se3fast.apply()` AFTER rfdiffusion is importable and BEFORE the model runs; the resident driver is
unmodified — rfdiffusion1_opt/driver_run.py calls apply() in the driver process when the mode's environment row sets RFD_SE3FAST,
and refuses the mode by name when the stats report the Triton line did not engage (a t2torch fallback is not run under `fast`).
Fidelity: TIER-2 (re-associated fp32 reductions; precision policy unchanged; not byte-equal to stock; run-to-run deterministic).
"""
import os, sys, types, functools
import torch

__version__ = "0.3.0"
STATS = dict(mode=None, scope=None, applied=False, triton=None, n_calls=0, n_full=0, n_topk=0, fallback_reason=None)
_ORIG = {}


def _mode():
    m = os.environ.get("RFD_SE3FAST", "0").strip().lower()
    return m if m in ("0", "exact", "t2", "t2torch") else "0"


def apply(mode=None, scope=None, verbose=True):
    """Patch rfdiffusion.Track_module.Str2Str.forward (class level -> extra, main and refinement instances)."""
    mode = (mode or _mode()); scope = (scope or os.environ.get("RFD_SE3FAST_SCOPE", "all")).lower()
    STATS['mode'] = mode; STATS['scope'] = scope
    if mode in ("0", "exact"):
        if mode == "exact" and verbose:
            print("[rfd_se3fast] RFD_SE3FAST=exact: this release carries no exact-mode SE(3) lever; stock path left untouched.", flush=True)
        return dict(STATS)
    import rfdiffusion.Track_module as TM
    from . import dense_torch as DT
    se3_fn = DT.se3_dense_forward
    if mode == "t2":
        try:
            import triton  # noqa
            from . import kernels  # noqa  (compiles lazily at first call)
            se3_fn = DT.se3_dense_forward_triton
            STATS['triton'] = triton.__version__
        except Exception as e:  # pragma: no cover
            STATS['fallback_reason'] = f"triton unavailable: {e!r}"; se3_fn = DT.se3_dense_forward
            print("[rfd_se3fast] WARNING:", STATS['fallback_reason'], "-> t2torch", flush=True)
    if 'Str2Str.forward' not in _ORIG:
        _ORIG['Str2Str.forward'] = TM.Str2Str.forward
    stock_forward = _ORIG['Str2Str.forward']

    @functools.wraps(stock_forward)
    def forward(self, msa, pair, R_in, T_in, xyz, state, idx, motif_mask, top_k=64, eps=1e-5, cyclic_reses=None):
        use = (msa.shape[0] == 1) and (scope == "all" or top_k == 0)
        if not use:
            return stock_forward(self, msa, pair, R_in, T_in, xyz, state, idx, motif_mask, top_k=top_k, eps=eps, cyclic_reses=cyclic_reses)
        STATS['n_calls'] += 1
        if top_k == 0:
            STATS['n_full'] += 1
        else:
            STATS['n_topk'] += 1
        out = DT.str2str_forward_dense(self, msa, pair, R_in, T_in, xyz, state, idx, motif_mask, top_k=top_k, eps=eps,
                                       cyclic_reses=cyclic_reses, se3_fn=se3_fn)
        return out

    TM.Str2Str.forward = forward
    STATS['applied'] = True
    if verbose:
        print(f"[rfd_se3fast] v{__version__} applied: mode={mode} scope={scope} se3_fn={se3_fn.__name__} triton={STATS['triton']}", flush=True)
    return dict(STATS)


def unapply():
    import rfdiffusion.Track_module as TM
    if 'Str2Str.forward' in _ORIG:
        TM.Str2Str.forward = _ORIG['Str2Str.forward']
    STATS['applied'] = False


def stats():
    from .dense_torch import STATS as DS
    return dict(STATS, **{f"dense_{k}": v for k, v in DS.items()})
