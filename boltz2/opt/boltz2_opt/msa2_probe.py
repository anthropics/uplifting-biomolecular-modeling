"""boltz2_opt.msa2_probe — a development instrument (in no mode row): per-op CUDA-event attribution of the trunk, centred on the MSA module.

Installed by ``boltz2_opt.msa2.apply()`` when the row word carries the ``probe`` unit (``BOLTZ_MSA2=probe[,...]``). Numerics unchanged: it wraps
callables with ``torch.cuda.Event`` pairs recorded on the current stream and copies ``MSALayer.forward``'s statements verbatim between the timers;
no synchronisation is added inside the forward (one ``torch.cuda.synchronize()`` per item, after ``Boltz2.predict_step`` returned — the PHASE hook's own
sync point). Leaf wrappers (triangle multiplications, triangle attentions, transitions, OPM, PWA) are installed LAZILY at the first ``Boltz2.forward``
call, i.e. after every kit lever (attach-time adapters AND the worker's own trunk/flash/graph patches) is in place, so each timer wraps the
implementation that actually serves. Attribution scope (ctx) = msa | tmpl | trunk | conf by the enclosing module.

Output per item: ``[msa2-probe] item=<k> tokens=<N> msa_rows=<S> <key>=<ms>/<calls> ...`` on stderr, and the same row as JSON appended to
``msa2_probe.jsonl`` in the launch's kit directory (``BOLTZ_OPT_WORKDIR``; outside a launch, a private tempfile.mkstemp file) — the file is
named on the ``installed`` line; never a fixed name in the shared temporary directory.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections import defaultdict
from typing import Any, Dict, List

_STATE: Dict[str, Any] = {"installed": False, "leaf": False, "pairs": [], "ctx": ["top"], "item": 0, "facts": {}, "rows": [],
                          "lock": threading.Lock(), "out": None}
WORKDIR_ENV = "BOLTZ_OPT_WORKDIR"                 # worker_launch.WORKDIR_ENV: the launch's kit directory (<out_dir>/_kit)
OUT_NAME = "msa2_probe.jsonl"


def out_path() -> str:
    """The JSON rows' file, settled once per process: ``<BOLTZ_OPT_WORKDIR>/msa2_probe.jsonl`` (beside the run, in the launch's own kit
    directory); outside a launch, a private file made for this process (tempfile.mkstemp: 0600, O_EXCL, a name no other process predicts).
    A fixed name in the shared temporary directory would append through whatever another account planted there."""
    if _STATE["out"] is None:
        wd = (os.environ.get(WORKDIR_ENV) or "").strip()
        if wd and os.path.isdir(wd):
            _STATE["out"] = os.path.join(os.path.abspath(wd), OUT_NAME)
        else:
            import tempfile
            fd, p = tempfile.mkstemp(prefix="msa2_probe_", suffix=".jsonl"); os.close(fd)
            _STATE["out"] = p
    return _STATE["out"]


def _ctx() -> str:
    return _STATE["ctx"][-1]


class _T:
    """CUDA-event span keyed `<ctx>.<name>` (or `name` when absolute)."""
    __slots__ = ("key", "a", "b")

    def __init__(self, name: str, absolute: bool = False):
        import torch
        self.key = name if absolute else f"{_ctx()}.{name}"
        self.a = torch.cuda.Event(enable_timing=True); self.b = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.a.record(); return self

    def __exit__(self, *exc):
        self.b.record(); _STATE["pairs"].append((self.key, self.a, self.b)); return False


class _Scope:
    __slots__ = ("name",)

    def __init__(self, name): self.name = name
    def __enter__(self): _STATE["ctx"].append(self.name)
    def __exit__(self, *exc): _STATE["ctx"].pop(); return False


def _wrap_leaf(cls, attr: str, name: str):
    orig = getattr(cls, attr)
    if getattr(orig, "_msa2_probe", False):
        return

    def timed(self, *a, **k):
        with _T(name):
            return orig(self, *a, **k)
    timed._msa2_probe = True; timed.__wrapped__ = orig
    setattr(cls, attr, timed)


def _wrap_scope(cls, attr: str, scope: str, name: str, absolute: bool = True):
    orig = getattr(cls, attr)
    if getattr(orig, "_msa2_probe", False):
        return

    def timed(self, *a, **k):
        with _Scope(scope), _T(name, absolute=absolute):
            return orig(self, *a, **k)
    timed._msa2_probe = True; timed.__wrapped__ = orig
    setattr(cls, attr, timed)


def _install_leaf():
    """After every lever is applied (first Boltz2.forward): wrap the serving implementations."""
    if _STATE["leaf"]:
        return
    from boltz.model.layers import triangular_mult as TM, transition as TRN, outer_product_mean as OPM, pair_averaging as PWA, pairformer as PF
    from boltz.model.layers.triangular_attention import attention as TA
    from boltz.model.modules import trunkv2 as TR
    _wrap_leaf(TM.TriangleMultiplicationOutgoing, "forward", "tri_mul_out")
    _wrap_leaf(TM.TriangleMultiplicationIncoming, "forward", "tri_mul_in")
    _wrap_leaf(TA.TriangleAttentionStartingNode, "forward", "tri_att_start")
    _wrap_leaf(TA.TriangleAttentionEndingNode, "forward", "tri_att_end")
    _wrap_leaf(TRN.Transition, "forward", "transition")
    _wrap_leaf(OPM.OuterProductMean, "forward", "opm")
    _wrap_leaf(PWA.PairWeightedAveraging, "forward", "pwa")
    _wrap_leaf(PF.AttentionPairBias if hasattr(PF, "AttentionPairBias") else __import__("boltz.model.layers.attentionv2", fromlist=["x"]).AttentionPairBias, "forward", "seq_attn")
    # module scopes
    _wrap_scope(TR.MSAModule, "forward", "msa", "msa_module")
    _wrap_scope(PF.PairformerModule, "forward", "trunk", "pairformer_module")   # main trunk AND the confidence module's pairformer (ctx tells)
    for nm in ("TemplateV2Module", "TemplateModule"):
        if hasattr(TR, nm):
            _wrap_scope(getattr(TR, nm), "forward", "tmpl", "template_module")
    _wrap_scope(TR.DistogramModule, "forward", "top", "distogram")
    _wrap_scope(TR.InputEmbedder, "forward", "top", "input_embedder")
    from boltz.model.modules import confidencev2 as CF, diffusionv2 as DF
    _wrap_scope(CF.ConfidenceModule, "forward", "conf", "confidence_module")
    _wrap_scope(DF.AtomDiffusion, "sample", "top", "diffusion_sample")
    # MSALayer: verbatim statements between timers (nobody else patches MSALayer.forward at n_gpu = 1)
    _STATE["msal_orig"] = TR.MSALayer.forward
    TR.MSALayer.forward = _msalayer_forward
    _STATE["leaf"] = True
    sys.stderr.write("[msa2-probe] leaf timers installed (after all levers)\n")


def _msalayer_forward(self, z, m, token_mask, msa_mask, chunk_heads_pwa=False, chunk_size_transition_z=None, chunk_size_transition_msa=None,
                      chunk_size_outer_product=None, chunk_size_tri_attn=None, use_kernels=False):
    # == trunkv2.py MSALayer.forward (boltz 2.2.1), statements verbatim; timers around them ==
    from boltz.model.layers.pairformer import get_dropout_mask
    f = _STATE["facts"]
    f.setdefault("msa_m_shape", tuple(m.shape)); f.setdefault("msa_z_shape", tuple(z.shape))
    f.setdefault("chunk", dict(chunk_heads_pwa=bool(chunk_heads_pwa), chunk_size_transition_z=chunk_size_transition_z, chunk_size_transition_msa=chunk_size_transition_msa,
                               chunk_size_outer_product=chunk_size_outer_product, chunk_size_tri_attn=chunk_size_tri_attn, use_kernels=bool(use_kernels)))
    f.setdefault("dtypes", dict(m=str(m.dtype), z=str(z.dtype), msa_mask=str(msa_mask.dtype), token_mask=str(token_mask.dtype)))
    with _Scope("msa"):
        with _T("pwa+resid"):
            msa_dropout = get_dropout_mask(self.msa_dropout, m, self.training)
            m = m + msa_dropout * self.pair_weighted_averaging(
                m, z, token_mask, chunk_heads_pwa
            )
        with _T("msa_transition+resid"):
            m = m + self.msa_transition(m, chunk_size_transition_msa)
        with _T("opm+resid"):
            z = z + self.outer_product_mean(m, msa_mask, chunk_size_outer_product)
        with _T("pairstack"):
            z = self.pairformer_layer(
                z, token_mask, chunk_size_tri_attn, use_kernels=use_kernels
            )
    return z, m


def _flush(tokens=None):
    import torch
    torch.cuda.synchronize()
    tot = defaultdict(float); cnt = defaultdict(int)
    for key, a, b in _STATE["pairs"]:
        try:
            tot[key] += a.elapsed_time(b); cnt[key] += 1
        except Exception:
            cnt[key + "!err"] += 1
    _STATE["pairs"].clear()
    f = dict(_STATE["facts"]); _STATE["facts"].clear()
    ms = f.get("msa_m_shape"); S = ms[1] if ms else None; N = ms[2] if ms else tokens
    row = {"item": _STATE["item"], "tokens": N, "msa_rows": S, "facts": f, "ms": {k: round(v, 3) for k, v in tot.items()}, "calls": dict(cnt), "t": time.time()}
    _STATE["rows"].append(row); _STATE["item"] += 1
    keys = sorted(tot, key=lambda k: -tot[k])
    sys.stderr.write("[msa2-probe] item=%d tokens=%s msa_rows=%s " % (row["item"], N, S) + " ".join(f"{k}={tot[k]:.1f}ms/{cnt[k]}" for k in keys) + "\n")
    try:
        with open(out_path(), "a") as fh:
            fh.write(json.dumps(row) + "\n")
    except OSError:
        pass


def install():
    """Wrap Boltz2.forward (lazy leaf install + whole-forward span) and Boltz2.predict_step (per-item flush)."""
    if _STATE["installed"]:
        return True
    from boltz.model.models import boltz2 as B2
    fwd = B2.Boltz2.forward

    def forward(self, *a, **k):
        _install_leaf()
        with _T("model_forward", absolute=True):
            return fwd(self, *a, **k)
    forward.__wrapped__ = fwd
    B2.Boltz2.forward = forward
    step = B2.Boltz2.predict_step

    def predict_step(self, batch, *a, **k):
        try:
            return step(self, batch, *a, **k)
        finally:
            try:
                tokens = int(batch["token_pad_mask"].shape[-1]) if isinstance(batch, dict) and "token_pad_mask" in batch else None
            except Exception:
                tokens = None
            _flush(tokens)
    predict_step.__wrapped__ = step
    B2.Boltz2.predict_step = predict_step
    _STATE["installed"] = True
    sys.stderr.write(f"[msa2-probe] installed on Boltz2.forward / Boltz2.predict_step (per-item CUDA-event attribution; numerics unchanged); rows -> {out_path()}\n")
    return True


def report() -> Dict[str, Any]:
    return {"items": len(_STATE["rows"]), "rows": _STATE["rows"], "out": _STATE["out"]}
