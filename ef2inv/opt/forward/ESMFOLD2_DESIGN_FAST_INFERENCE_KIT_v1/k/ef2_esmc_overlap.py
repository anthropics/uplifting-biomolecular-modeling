"""ef2_esmc_overlap — run the design step's pseudo-perplexity term CONCURRENTLY with its fold (exact).

In the cookbook's `run_step` the two gradient terms are independent given the step's logits: the structure term (fold →
distogram losses → backward) and the pLM term (`compute_esmc_pseudoperplexity_nll` → backward), yet they run back to back,
and the pLM term (the ESMC-6B forward+backward) starts only after the trunk backward has been
launched and its own host syncs have drained the GPU queue. Both terms are pure functions of tensors that exist when the fold
is called: the fold receives the very `design` tensor (softmax of the logits) the pLM term will receive, and the seed the
cookbook will re-seed every RNG with (`seed_context(seed)`, which also restores every RNG state on exit) immediately before
the pLM term draws its masks. This lever wraps the two cookbook functions on the loaded cookbook module — the way the kit's
featurisation cache wraps `prepare_esmfold2_tensors` — and nothing else:

  * `fold_and_get_distogram` (design folds only: not a num_loops=3 confidence critic fold): after the stock fold has been
    launched, launch the stock pLM function on a side CUDA stream, on a detached copy of `design`, inside `seed_context(seed)`,
    followed by `autograd.grad(loss.mean(), copy)` — the arithmetic the cookbook is about to run, kernel for kernel — ordered
    after the fold's ESMC feature pass (a forward hook on the shared trunk records the event; the two never overlap, so the
    accelerated-library workspaces the LM's projections use are never shared between streams) and concurrent with the trunk,
    the structure module and the structure backward.
  * `compute_esmc_pseudoperplexity_nll`: if the call is the one prepared (same LM object, `binder_design` tensor-equal to the
    copy, same `score_mask` / `batch_size` / `n_passes`, grad mode on), return the early loss through an autograd Function whose
    backward hands the early gradient to `binder_design` (after the current stream waits on the side stream's event); anything
    else computes the stock function there and then, inside the cookbook's own seed context, and is counted by name. Nothing
    is drawn from any RNG inside the cookbook's seed context on the served path, and that context restores every RNG state on
    exit either way, so the RNG streams the rest of the step sees are stock's.

Exactness: every tensor the step produces (pLM loss, its gradient w.r.t. the logits, hence the update) is bitwise stock's —
same kernels, same inputs, same mask draws, launched earlier on another stream; the backward's only assumption (the cookbook
reduces the [B] loss with `.mean()`, which the early backward replicated) is checked against the incoming grad_output and a
different uniform reduction is served rescaled and counted (`grad_output_rescaled`, no longer a bitwise claim); a non-uniform
one raises. The first design step of a process computes in place (it records the reference arguments); so does any step
whose fold was not a design fold or whose arguments changed (`stats['fallback'][reason]`).

Ordering next to other levers' CUDA-graph captures: all of this lever's host-side work — the side-stream launches, the pLM
function's own host syncs (they wait on the side stream only), its allocations — happens inside the two wrappers on the loop's
one CPU thread, i.e. strictly before or after any capture another lever performs on that thread, never during one; the first
design step of a process (where captures of first use happen) computes in place; a wrapper that finds the current stream
capturing does nothing on the side stream (`capturing`). Device-side, kernels already in flight on the side stream are legal
under any capture mode, and the only cross-stream edges this lever creates (current stream waits on the side stream's event)
are issued in the pLM wrapper and in the served backward — outside any other lever's captured region.

Costs and limits: one extra [B, L_binder, 20] tensor and a [B] loss alive from fold to loss call; no extra LM memory (one pLM
computation in flight at a time, the same buffers). The gain is the overlap: large where the step is launch-bound (≈200
tokens), small where the trunk saturates the GPU (≥700). Meant to be stacked on ef2_pppl_graph (an eager pLM forward launched
inside the fold wrapper would hold the CPU for its ~3000 launches before the structure loss is queued).

Composition: the body launched early is resolved at call time (the module attribute, or the function under this lever's wrapper when the
attribute is that wrapper), so a sync-free pLM body or a timing wrapper stacked on the attribute is honoured; a lever that REPLACES the
attribute after this one is enabled also takes the loop's call away from this lever's wrapper — enable this lever after it (an early result the
loop never asks for is counted `superseded` at the next fold, never served stale).

Usage:
    import ef2_esmc_overlap as eeo
    h = eeo.enable(app)            # app: the cookbook's ESMFold2Design instance (its module and its .esmc_model), or
    h = eeo.enable(module, esmc_model=lm)
    h.stats                        # {'served': n, 'computed': n, 'fallback': {reason: n}}
    eeo.disable(app)
"""
from __future__ import annotations
import inspect
import sys
import types
import torch

_ATTR = "_ef2_esmc_overlap"
FOLD, PPPL = "fold_and_get_distogram", "compute_esmc_pseudoperplexity_nll"
_REQ = object()
FOLD_PARAMS = (("model", _REQ), ("target_seq", _REQ), ("target_one_hot", _REQ), ("design", _REQ), ("num_loops", 0), ("num_sampling_steps", 1),
               ("calculate_confidence", False), ("seed", None))                      # binder_design.fold_and_get_distogram
PPPL_PARAMS = (("esmc_model", _REQ), ("binder_design", _REQ), ("score_mask", _REQ), ("batch_size", 4), ("n_passes", 4))   # compute_esmc_pseudoperplexity_nll


class _Served(torch.autograd.Function):
    """loss (computed early) re-attached to the cookbook's `binder_design`; backward delivers the early d mean(loss)/d design."""

    @staticmethod
    def forward(ctx, binder_design, loss, grad, stats):
        ctx.grad, ctx.n, ctx.stats = grad, loss.numel(), stats
        return loss.clone()

    @staticmethod
    def backward(ctx, g_out):
        grad, n, stats = ctx.grad, ctx.n, ctx.stats
        ctx.grad = None
        expected = torch.ones((), dtype=g_out.dtype, device=g_out.device).expand(g_out.shape) / n     # what MeanBackward hands a [B] loss: expand / numel
        if torch.equal(g_out, expected):
            return grad, None, None, None
        first = g_out.reshape(-1)[:1]
        if not torch.equal(g_out.reshape(-1), first.expand(g_out.numel())):
            raise RuntimeError("ef2_esmc_overlap: the pLM loss was reduced with per-element weights; the early gradient "
                               "(computed for .mean()) cannot serve it — disable the lever for this loop")
        stats["fallback"]["grad_output_rescaled"] = stats["fallback"].get("grad_output_rescaled", 0) + 1
        return grad * (first * n).to(grad.dtype), None, None, None


class _Overlap:
    def __init__(self, module, lm):
        self.G = vars(module)                                   # the cookbook module's globals: run_step resolves both functions here per call
        self.lm = lm
        self.inner_fold, self.inner_pppl = self.G[FOLD], self.G[PPPL]      # the wrapped callables (restored by disable)
        self.seed_context = self.G["seed_context"]
        self._in_early = False
        self.stream = torch.cuda.Stream()
        self.ev_feat = None                                     # recorded after each no-grad forward of the shared trunk (the fold's feature pass)
        self.ref = None                                         # the last real call's (score_mask, batch_size, n_passes)
        self.pending = None
        self.stats = {"served": 0, "computed": 0, "fallback": {}}
        trunk = getattr(lm, "esmc", lm)
        self.hook = trunk.register_forward_hook(self._after_trunk)

    def _count(self, why):
        self.stats["fallback"][why] = self.stats["fallback"].get(why, 0) + 1

    @staticmethod
    def _bind(params, args, kw):
        """Arguments by the STOCK parameter order (wrappers stacked on the module attribute hide the signature; the loop's calls do not change)."""
        out = {name: default for name, default in params if default is not _REQ}
        out.update(zip((name for name, _ in params), args))
        out.update(kw)
        return out

    def _early_body(self):
        """The pLM body to launch early, resolved at CALL time: whatever the module attribute is now unless that is this lever's wrapper (then the
        function it wrapped). A wrapper stacked over it after enable (a timer, another lever) is thus what runs early, and it
        reaches the real body through this wrapper's pass-through (`_in_early`), never recursing."""
        cur = self.G.get(PPPL)
        return self.inner_pppl if getattr(cur, "_ef2_esmc_overlap", False) else cur

    def _after_trunk(self, mod, args, out):
        if not torch.is_grad_enabled():                        # the fold's feature pass (inference mode); the pLM term calls .transformer directly
            ev = torch.cuda.Event()
            ev.record(torch.cuda.current_stream())
            self.ev_feat = ev

    # ---- fold_and_get_distogram: launch the stock fold, then the pLM term early on the side stream
    def fold(self, *args, **kw):
        self.ev_feat = None
        self._drop_pending("superseded")                       # an early result nobody asked for: the loop did not reach this lever's pLM wrapper (a body installed over it?)
        out = self.inner_fold(*args, **kw)
        try:
            self._launch_early(self._bind(FOLD_PARAMS, args, kw))
        except Exception as e:  # noqa: BLE001 — the step then computes the term in place; the reason is counted
            self._drop_pending(f"early_failed:{type(e).__name__}")
        return out

    def _launch_early(self, p):
        design, seed = p.get("design"), p.get("seed")
        if self.ref is None:
            return self._count("no_reference_yet")
        if bool(p.get("calculate_confidence", False)) and p.get("num_loops", 0) == 3:
            return self._count("not_design_fold")
        if seed is None:
            return self._count("unseeded_fold")
        if not (torch.is_tensor(design) and design.is_cuda and torch.is_grad_enabled() and not torch.is_inference_mode_enabled()):
            return self._count("call_form")
        if torch.cuda.is_current_stream_capturing():           # inside another lever's CUDA-graph capture: no side-stream work, no syncs
            return self._count("capturing")
        ref = self.ref
        body = self._early_body()
        main = torch.cuda.current_stream()
        s = self.stream
        if self.ev_feat is not None:
            s.wait_event(self.ev_feat)                          # after the fold's ESMC feature pass; concurrent with everything the fold queued after it
        else:
            s.wait_stream(main)
        self._in_early = True
        try:
            with torch.cuda.stream(s), self.seed_context(seed):
                d_in = design.detach().clone().requires_grad_(True)
                loss = body(esmc_model=self.lm, binder_design=d_in, score_mask=ref["score_mask"],
                            batch_size=ref["batch_size"], n_passes=ref["n_passes"])
                grad, = torch.autograd.grad(loss.mean(), d_in)
                ev = torch.cuda.Event()
                ev.record(s)
        finally:
            self._in_early = False
        for t in (loss, grad, d_in):
            t.record_stream(main)
        self.pending = dict(d_in=d_in.detach(), loss=loss.detach(), grad=grad, ev=ev, batch_size=ref["batch_size"], n_passes=ref["n_passes"],
                            score_mask=ref["score_mask"])

    def _drop_pending(self, why=None):
        if self.pending is not None:
            torch.cuda.current_stream().wait_event(self.pending["ev"])
            self.pending = None
            if why:
                self._count(why)

    # ---- compute_esmc_pseudoperplexity_nll: serve the early result if it is this call's, else compute here
    def pppl(self, *args, **kw):
        if self._in_early:                                      # this lever's own early launch coming through a wrapper stacked over it: straight to the body
            return self.inner_pppl(*args, **kw)
        a = self._bind(PPPL_PARAMS, args, kw)
        lm, bd, sm, bs, npass = a.get("esmc_model"), a.get("binder_design"), a.get("score_mask"), a.get("batch_size"), a.get("n_passes")
        pend, self.pending = self.pending, None
        why = None                                              # a missing early result was counted when the fold declined to launch it (or this is the first call)
        if pend is None:
            pass
        elif torch.cuda.is_current_stream_capturing():
            why = "capturing"
        elif lm is not self.lm:
            why = "other_lm"
        elif not (torch.is_grad_enabled() and torch.is_tensor(bd) and bd.is_cuda and bd.requires_grad):
            why = "call_form"
        elif (bs, npass) != (pend["batch_size"], pend["n_passes"]) or bd.shape != pend["d_in"].shape or bd.dtype != pend["d_in"].dtype:
            why = "args_changed"
        elif not (torch.is_tensor(sm) and sm.shape == pend["score_mask"].shape and sm.dtype == pend["score_mask"].dtype):
            why = "args_changed"
        if pend is not None and why is None:
            torch.cuda.current_stream().wait_event(pend["ev"])
            if not (torch.equal(bd.detach(), pend["d_in"]) and torch.equal(sm.to(pend["score_mask"].device), pend["score_mask"])):
                why = "inputs_differ"
        if torch.is_tensor(sm) and lm is self.lm:               # the reference for the next step's early launch
            self.ref = dict(score_mask=sm.detach().clone(), batch_size=bs, n_passes=npass)
        if pend is None or why is not None:
            if why:
                self._count(why)
            if pend is not None:
                torch.cuda.current_stream().wait_event(pend["ev"])
            self.stats["computed"] += 1
            return self.inner_pppl(*args, **kw)
        self.stats["served"] += 1
        return _Served.apply(bd, pend["loss"], pend["grad"], self.stats)

    def install(self):
        self.G[FOLD] = self._mark(self.fold)
        self.G[PPPL] = self._mark(self.pppl)
        return self

    @staticmethod
    def _mark(fn):
        def wrapper(*args, **kw):
            return fn(*args, **kw)
        wrapper._ef2_esmc_overlap = True                        # disable() restores the inner function only where this lever's wrapper is still installed
        return wrapper

    def remove(self):
        self._drop_pending()
        self.stream.synchronize()
        for name, inner in ((FOLD, self.inner_fold), (PPPL, self.inner_pppl)):
            cur = self.G.get(name)
            if getattr(cur, "_ef2_esmc_overlap", False):
                self.G[name] = inner
        self.hook.remove()


def _resolve(target, esmc_model):
    if isinstance(target, types.ModuleType):
        module = target
    else:
        module = sys.modules.get(type(target).__module__) or inspect.getmodule(type(target))
        esmc_model = esmc_model if esmc_model is not None else getattr(target, "esmc_model", None)
    if module is None or FOLD not in vars(module) or PPPL not in vars(module) or "seed_context" not in vars(module):
        raise TypeError("enable(): expected the cookbook's design app (or its module) — a module defining "
                        f"{FOLD}, {PPPL} and seed_context")
    if esmc_model is None or not isinstance(getattr(esmc_model, "esmc", None), torch.nn.Module):
        raise TypeError("enable(): the ESMCForMaskedLM the loop scores with is required (app.esmc_model, or esmc_model=)")
    return module, esmc_model


def enable(target, esmc_model=None) -> _Overlap:
    module, lm = _resolve(target, esmc_model)
    h = getattr(module, _ATTR, None)
    if h is None:
        h = _Overlap(module, lm).install()
        setattr(module, _ATTR, h)
    return h


def disable(target, esmc_model=None) -> None:
    module, _ = _resolve(target, esmc_model)
    h = getattr(module, _ATTR, None)
    if h is not None:
        h.remove()
        delattr(module, _ATTR)


def handle(target, esmc_model=None):
    module, _ = _resolve(target, esmc_model)
    return getattr(module, _ATTR, None)
