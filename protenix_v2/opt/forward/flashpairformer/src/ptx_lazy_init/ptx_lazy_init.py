"""ptx_lazy_init.py -- cold-start lever for Protenix v2 (protenix 2.0.0): skip the dead random initialisation of the
464 M parameters during model construction (the init functions the module constructors call are no-op'd during
construction and restored afterwards; adapted to the Protenix module tree).

Why it is exact: InferenceRunner.init_model() constructs Protenix(configs) -- every custom Linear runs a scipy/truncated-normal or
xavier/kaiming init on the CPU (most of a cold start) -- and then load_checkpoint() does
load_state_dict(strict=True) from protenix-v2.pt, which OVERWRITES every parameter and every persistent buffer. The random values are
never read. This module turns the init functions into no-ops ONLY while init_model() runs, then restores them.
Equality is checked, not assumed: PTX_LAZY_INIT=recheck constructs the model twice in one process (stock init, then lazy) and compares
sha256 of every state_dict tensor (params + buffers, incl. non-persistent buffers that the checkpoint does not cover) -> must be N/N equal.

Env: PTX_LAZY_INIT=0 (default, stock) | 1 (lazy) | check (both + per-tensor sha compare, writes $PTX_LAZY_INIT_REPORT)
"""
import os, sys, time, json, hashlib, contextlib
import torch, torch.nn as nn

_PATCHED = []
def _noop_ret(t, *a, **k): return t
def _noop_none(*a, **k): return None

def _collect_targets():
    """(module, attr_name, replacement) for every init function reachable in the Protenix tree + torch.nn.init."""
    T = []
    for name in ("trunc_normal_", "normal_", "xavier_uniform_", "xavier_normal_", "kaiming_uniform_", "kaiming_normal_", "uniform_", "zeros_", "ones_", "constant_"):
        # NOTE zeros_/ones_/constant_ are ALSO dead under strict load for parameters, but they may initialise NON-persistent buffers or
        # plain tensors used at run time -> keep them stock (cheap anyway). Only the expensive random inits are skipped.
        if name in ("zeros_", "ones_", "constant_"): continue
        if hasattr(nn.init, name): T.append((nn.init, name, _noop_ret))
    # the model's helper init functions (scipy truncnorm etc.), wherever they were imported to
    cand_attrs = ("trunc_normal_init_", "lecun_normal_init_", "he_normal_init_", "glorot_uniform_init_", "final_init_", "gating_init_", "normal_init_", "bias_init_zero_", "bias_init_one_", "ipa_point_weights_init_")
    for modname, mod in list(sys.modules.items()):
        if mod is None or not (modname.startswith("protenix") or modname.startswith("openfold")): continue
        for a in cand_attrs:
            if a in ("final_init_", "gating_init_", "bias_init_zero_", "bias_init_one_"):   # these write zeros/ones: cheap, and gating bias=1 matters only if not in ckpt (it is) -> keep stock to be safe
                continue
            if hasattr(mod, a) and callable(getattr(mod, a)): T.append((mod, a, _noop_none))
    # nn.Linear / nn.LayerNorm / nn.Embedding reset_parameters (kaiming_uniform_ + uniform_) are covered by nn.init patches above
    return T

@contextlib.contextmanager
def skip_random_init():
    targets = _collect_targets(); saved = []
    for mod, a, rep in targets:
        saved.append((mod, a, getattr(mod, a))); setattr(mod, a, rep)
    try:
        yield [f"{m.__name__}.{a}" for m, a, _ in targets]
    finally:
        for mod, a, orig in saved: setattr(mod, a, orig)

def _sd_hashes(model):
    out = {}
    sd = model.state_dict(keep_vars=False)
    for k, v in sd.items():
        t = v.detach().to("cpu").contiguous()
        out[k] = hashlib.sha256(t.view(torch.uint8).numpy().tobytes() if t.dtype != torch.bool else t.numpy().tobytes()).hexdigest() + f"|{tuple(t.shape)}|{t.dtype}"
    # non-persistent buffers are not in state_dict: hash them via named_buffers
    nb = {k for k, _ in model.named_buffers()} - set(sd.keys())
    for k, v in model.named_buffers():
        if k in nb:
            t = v.detach().to("cpu").contiguous()
            out["<nonpersistent>" + k] = hashlib.sha256(t.view(torch.uint8).numpy().tobytes() if t.dtype != torch.bool else t.numpy().tobytes()).hexdigest() + f"|{tuple(t.shape)}|{t.dtype}"
    return out

def install():
    """Monkeypatch runner.inference.InferenceRunner.init_model per PTX_LAZY_INIT. Returns the mode string."""
    mode = os.environ.get("PTX_LAZY_INIT", "0")
    if mode not in ("1", "recheck"): return "off"
    import runner.inference as RI
    # make sure the protenix model modules are imported so their init helpers are patchable
    import protenix.model.protenix  # noqa
    import protenix.model.modules.primitives  # noqa: trunc_normal_init_/normal_init_ import site 1
    import protenix.model.triangular.layers  # noqa: trunc_normal_init_ (scipy truncnorm) / lecun / he / glorot / normal import site 2
    orig = RI.InferenceRunner.init_model
    report = {"mode": mode}
    def init_model(self):
        if mode == "recheck":
            t = time.time(); orig(self); report["stock_construct_s"] = round(time.time() - t, 2)
            # load checkpoint into the stock model exactly like the runner will, hash, then discard
            t = time.time(); self.load_checkpoint() if hasattr(self, "load_checkpoint") else None; report["stock_ckpt_s"] = round(time.time() - t, 2)
            report["stock_hashes"] = _sd_hashes(self.model); del self.model; torch.cuda.empty_cache()
        t = time.time()
        with skip_random_init() as patched:
            orig(self)
        report["lazy_construct_s"] = round(time.time() - t, 2); report["patched_fns"] = patched
        self.print(f"[ptx_lazy_init] model constructed with random init skipped in {report['lazy_construct_s']:.1f}s ({len(patched)} init fns no-op'd; strict checkpoint load follows)")
    RI.InferenceRunner.init_model = init_model
    if mode == "recheck":
        orig_load = RI.InferenceRunner.load_checkpoint
        def load_checkpoint(self):
            orig_load(self)
            if "stock_hashes" in report and "lazy_hashes" not in report and "lazy_construct_s" in report:
                report["lazy_hashes"] = _sd_hashes(self.model)
                sh, lh = report["stock_hashes"], report["lazy_hashes"]
                keys = sorted(set(sh) | set(lh)); neq = [k for k in keys if sh.get(k) != lh.get(k)]
                report["n_tensors"] = len(keys); report["n_equal"] = len(keys) - len(neq); report["unequal"] = neq[:50]
                report["verdict"] = "IDENTICAL" if not neq else "DIFFERENT"
                self.print(f"[ptx_lazy_init] RECHECK state after checkpoint load: {report['n_equal']}/{report['n_tensors']} tensors sha256-identical (params+buffers+non-persistent buffers) -> {report['verdict']}")
                p = os.environ.get("PTX_LAZY_INIT_REPORT")
                if p:
                    slim = {k: v for k, v in report.items() if k not in ("stock_hashes", "lazy_hashes")}; slim["unequal_detail"] = {k: (sh.get(k), lh.get(k)) for k in neq[:50]}
                    json.dump(slim, open(p, "w"), indent=1)
        RI.InferenceRunner.load_checkpoint = load_checkpoint
    return mode
