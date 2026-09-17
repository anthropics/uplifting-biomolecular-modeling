"""The graphed diffusion sampler's poison self-test step-aside BY NAME — kit glue over the vendored loop.

The vendored unit (opt/forward/flashpairformer/src/infopt_graphs/protenix/graphed.py, ``GraphedDenoiseLoop``; its bytes are
the published ones and stay so — tests/test_sealed_third_party.py guards the whole third_party tree) proves once per process that a captured sampler
graph READS the live static buffers (``_poison_test``: the magnitude probe, with the DiT hoist bound; ``_poison_test_classes``: the per-class
NaN → magnitude probe) and RAISES when it cannot show it (``RuntimeError("biascache poison test FAILED …")`` / ``RuntimeError("poison test FAILED …")``).
Under stock's ``--dtype fp16`` on an input whose half-precision trunk overflowed, the step output the probe would compare is non-finite BEFORE anything
is poisoned (stock's own sampler is non-finite there too): no distance can be formed, the vendored probe raised, stock's runner caught the exception
per item and the run ended without that item's outputs (exit 1).  A kit accepts what stock accepts and a lever that cannot engage steps aside BY NAME,
so this module subclasses the vendored loop FROM THE OUTSIDE — no vendored byte changes — and binds the subclass as the class the unit's own
``install()`` constructs (``install()`` resolves ``GraphedDenoiseLoop`` from its module globals at call time; ``fpf_clisampler`` calls it after
``InferenceRunner.init_model``; the kit binds the name at activation, ``stack._apply``):

* a probe that cannot be judged (the reference replay's output is non-finite) returns its record with ``verdict="nonfinite"``
  (``reads_static_buffers`` None), poisons nothing, leaves the process verdict untouched (``sampler_prep.POISON_STATUS``: ``poison=not-run``) and the
  once-per-process test armed for the next captured signature; a judged probe carries ``verdict="ok"|"failed"``; neither raises to its caller;
* ``_capture``: the eager warm-up step's output already non-finite → nothing is captured and the signature steps aside (``stage=warmup``); a
  ``nonfinite`` / ``failed`` verdict on the freshly captured graph → the graph is dropped and the signature steps aside (``stage=selftest``); after a
  ``failed`` verdict every later signature of the process steps aside uncaptured (``stage=earlier_verdict``: no untested graph after a failed one) and
  the process verdict says ``poison=failed`` (the record's ``prep_poison``; ``stack.reconcile`` moves ``sampler_prep`` to the fallbacks by name:
  FINAL ``partial=true``, exit 3 — a defect stays a defect, now with the item's outputs written by the eager sampler);
* ``_step_aside``: the entry keeps its static buffers and is served by the loop's existing eager step body (the same step arithmetic as stock on the
  stock random stream — the route an entry the capture cannot serve already takes), the captured graph (if any) is dropped and its shared-pool share
  released as an eviction releases it, the aside is counted under its reason word (``stats["aside"]``), recorded as a ``sampler_aside`` event and said
  ONCE on stderr in the unit's grammar (the vendored file's own words, unchanged):
  ``[infopt_graphs] SAMPLER route=eager reason=nonfinite_probe|poison_mismatch stage=warmup|selftest|earlier_verdict N_token=<n> N_atom=<n> aside=<word:n[,word:n]> — <why>``
  (the ``poison_mismatch`` line carries the ``poison test FAILED`` words; the non-finite line never does);
* ``summary()`` gains a last key ``aside`` (``{reason: signatures}``; ``{}`` on a run where nothing stepped aside, so a bf16 run's truncated
  ``SUMMARY sampler=`` JSON reads as before) — the clisampler record's ``sampler.aside``, which ``stack.reconcile`` records under the run's ``asides``
  (``asides.sampler_graph``): declared, counted, by design, never a fallback.

The hooks, all from outside the sealed file (``loop_class``):
  ``_poison_test`` / ``_poison_test_classes`` — one reference replay first (x_l restored): non-finite → the ``nonfinite`` record; finite → the vendored
      probe runs unchanged and its own ``… poison test FAILED`` RuntimeError (that message only; anything else propagates) becomes the ``failed`` verdict
      on the record it had already stored (a poisoned replay that turns x_l non-finite READS the buffers: d = inf, the per-class probe's own rule);
  ``_capture`` — brackets the vendored capture.  The vendored frame consults ``self.disable_capture`` exactly once, between the eager warm-up step and
      the capture: here it is a property that answers the vendored switch (``INFOPT_GRAPHS_SAMPLER_CAPTURE=0``) OR "this warm-up cannot be judged /
      an earlier graph failed", so such a signature takes the vendored no-capture route and is then named by ``_step_aside``; a verdict reached on the
      fresh graph inside the vendored frame leaves it through a private signal (``_Verdict``, caught only here) and the signature steps aside with the
      graph dropped — the vendored frame's own post-capture accounting (capture count, eviction, ``sampler_captured`` event) does not run for it;
      records whose probe decided nothing wait outside the vendored frame while it runs (it disarms the once-per-process test on ANY record:
      the ``_poison_decided`` rule);
  ``_step_body`` — notes which static-buffer dict the warm-up step ran on (one dict write, no tensor op: capture-safe, replay-free);
  ``summary`` — appends ``aside``.
``install()`` binds the subclass on ``infopt_graphs.protenix.graphed`` (and the package's re-export); it prints nothing when it arms (the loop names an
aside when one happens) and ``[protenix-opt] SAMPLER_ASIDE:unavailable(<repr>)`` if the vendored module cannot be imported at activation (the sampler
hook then reports its own install failure by name, as before).
"""
from __future__ import annotations

import importlib
import math
import sys
from typing import Any, Dict, Optional

PREFIX = "[protenix-opt]"
UNIT_PREFIX = "[infopt_graphs]"                      # the step-aside line keeps the vendored unit's grammar (its words, verbatim)
GRAPHED_MODULE = "infopt_graphs.protenix.graphed"    # the copy this process imports (env.sh L6 order; stack.graphed_py names the file)
PACKAGE_MODULE = "infopt_graphs.protenix"            # re-exports GraphedDenoiseLoop (install() itself reads the graphed module's global)
ASIDE_NONFINITE = "nonfinite_probe"   # the step output the probe compares is non-finite before anything is poisoned (the input's own numerics — a half-precision
#                                       trunk that overflowed: stock's sampler is non-finite on that input too): no distance can be formed, nothing is judged; by design
ASIDE_MISMATCH = "poison_mismatch"    # judged on finite values and FAILED: a captured kernel reads a stale or private copy of a live static buffer — a defect signal:
#                                       said loudly (the `poison test FAILED` words), the process verdict carries poison=failed (sampler_prep.POISON_STATUS), and no
#                                       later signature of the process is captured
MARK = "_protenix_opt_poison_aside"                  # class attribute: this class is the kit's step-aside subclass (install is idempotent)
CLASS_NAME = "PoisonAsideDenoiseLoop"
_CLASSES: Dict[int, type] = {}                       # id(vendored class) -> the subclass built over it
_STATE: Dict[str, Any] = {"installed": None, "error": None, "graphed_file": None}


class _Verdict(Exception):
    """Private signal: a `nonfinite` / `failed` verdict reached INSIDE the vendored capture frame. Raised by the probe overrides only while a capture of
    this loop is running and caught only by the `_capture` override (a direct caller of the probes gets the record returned)."""

    def __init__(self, ent, rec):
        Exception.__init__(self, rec.get("verdict"))
        self.ent, self.rec = ent, rec


def loop_class(graphed) -> type:
    """The kit's step-aside subclass over ``graphed.GraphedDenoiseLoop`` (built once per vendored class object; the vendored class itself when it
    already is the subclass)."""
    base = graphed.GraphedDenoiseLoop
    if getattr(base, MARK, False):
        return base
    cls = _CLASSES.get(id(base))
    if cls is not None and cls.__mro__[1] is base:
        return cls
    torch = graphed.torch
    _sp = graphed._sp                                # infopt_graphs.protenix.sampler_prep as the vendored loop imported it (POISON_STATUS, the poison_once constants)

    class PoisonAsideDenoiseLoop(base):
        """``GraphedDenoiseLoop`` whose poison self-test never raises to the run: an unjudgeable or failed probe steps the signature aside BY NAME to the
        loop's eager step body (module doc)."""
        ASIDE_NONFINITE = ASIDE_NONFINITE
        ASIDE_MISMATCH = ASIDE_MISMATCH

        def __init__(self, *args, **kwargs):
            base.__init__(self, *args, **kwargs)
            self._arm_aside()

        # ----- state (also armed lazily: a loop built with __new__ and filled by hand — the unit tests' stand-ins — behaves the same)
        def _arm_aside(self) -> None:
            st = self.__dict__.get("stats")
            if isinstance(st, dict):
                st.setdefault("aside", {})                                # reason word -> signatures served by the eager step body
            self.__dict__.setdefault("_poison_failed", False)             # a graph FAILED the self-test in this process: every later signature is served eagerly by name too
            self.__dict__.setdefault("_probe", None)                      # per-capture scratch while the vendored _capture runs: {"st", "pre", "verdict"}

        @property
        def disable_capture(self) -> bool:
            """The vendored no-capture switch (INFOPT_GRAPHS_SAMPLER_CAPTURE=0), read by the vendored `_capture` once, after the eager warm-up step and before the
            capture — extended: also true for THIS capture when the warm-up's output is already non-finite (nothing could judge a graph of the signature) or an
            earlier signature's graph failed the self-test in this process. The reason is kept for `_capture`, which names the aside."""
            if self.__dict__.get("_capture_switch_off", False):
                return True
            p = self.__dict__.get("_probe")
            if not p or not self.__dict__.get("poison_check", True):
                return False
            if p["pre"] is None:                                          # decided once per capture, at the vendored frame's single read
                st = p["st"]
                if st is not None and not self._finite(st["x_l"]):
                    p["pre"] = (ASIDE_NONFINITE, "warmup", "x_l non-finite after the eager warm-up step")
                elif self.__dict__.get("_poison_failed", False):
                    p["pre"] = (ASIDE_MISMATCH, "earlier_verdict", "an earlier signature's graph failed the poison self-test in this process")
                else:
                    p["pre"] = False
            return bool(p["pre"])

        @disable_capture.setter
        def disable_capture(self, value) -> None:
            self.__dict__["_capture_switch_off"] = bool(value)

        def _step_body(self, denoise_net, st, *args, **kwargs):
            """The vendored step body (a staticmethod there, reached as `self._step_body` by the vendored capture's `body`); notes the static-buffer dict the
            warm-up runs on while a capture of this loop is open. No tensor op of its own: capture-safe."""
            p = self.__dict__.get("_probe")
            if p is not None and p["st"] is None:
                p["st"] = st
            return base._step_body(denoise_net, st, *args, **kwargs)

        @staticmethod
        def _finite(t) -> bool:
            """Every element of `t` finite (one device reduction + one sync; used once per capture / probe, never per step)."""
            return bool(torch.isfinite(t).all().item())

        def _poison_decided(self) -> bool:
            """lever sampler_prep[poison_once]: the self-test has produced a verdict in this process — a record whose probe was non-finite decided nothing and
            leaves the test armed for the next captured signature."""
            return any(r.get("reads_static_buffers") is not None for r in self.stats["poison"])

        def _aside_token(self) -> str:
            """`word:n[,word:n]` over the signatures stepped aside so far (the LEVER pair form of a counted aside), '' when none."""
            self._arm_aside()
            return ",".join(f"{k}:{int(v)}" for k, v in self.stats["aside"].items() if v)

        def _verdict_reached(self, ent, rec):
            """Hand the record back to a direct caller; inside a running capture a `nonfinite` / `failed` verdict leaves the vendored frame (see `_capture`)."""
            p = self.__dict__.get("_probe")
            if p is not None:
                p["verdict"] = rec
                if rec.get("verdict") in ("nonfinite", "failed"):
                    raise _Verdict(ent, rec)
            return rec

        def _reference_finite(self, ent) -> bool:
            """One replay of the entry's graph from a snapshot of x_l: is the step output finite? (x_l restored from the snapshot.)"""
            st = ent["st"]; g = ent["graph"]
            x_snap = st["x_l"].clone(); torch.cuda.synchronize()
            g.replay(); torch.cuda.synchronize()
            finite = self._finite(st["x_l"])
            st["x_l"].copy_(x_snap); torch.cuda.synchronize()
            return finite

        # ----- the probes: never raise to the run
        def _poison_test(self, ent, N_atom):
            """The vendored magnitude probe, judged as the module docstring states. Returns the record with rec["verdict"]: "ok" | "nonfinite" (the reference replay's output is
            non-finite: no distance can be formed, nothing is poisoned, the test stays armed) | "failed" (judged on finite values; the process verdict says
            poison=failed)."""
            self._arm_aside()
            if not self._reference_finite(ent):
                rec = {"N_atom": N_atom, "verdict": "nonfinite", "reads_static_buffers": None,
                       "why": "the reference replay's step output is non-finite before anything is poisoned: a poisoned-vs-normal distance cannot be formed"}
                self.stats["poison"].append(rec); ent["poison"] = rec
                return self._verdict_reached(ent, rec)
            try:
                rec = base._poison_test(self, ent, N_atom)
            except RuntimeError as ex:
                if not str(ex).startswith("biascache poison test FAILED"):
                    raise
                rec = ent["poison"]                                       # the vendored probe stored its record (stats["poison"], ent["poison"]) before raising
                d_ab, d_ac = rec.get("poisoned_vs_normal_max"), rec.get("normal_vs_normal_max")
                if isinstance(d_ab, float) and math.isnan(d_ab):          # the poisoned replay turned x_l non-finite while the reference was finite: it READS the buffers (d = inf)
                    d_ab = float("inf")
                    if self.prep.poison_once:                             # the vendored criteria, restated on d = inf
                        reads = bool(d_ab >= _sp.POISON_MIN_RATIO * max(d_ac, 1e-6) and d_ab >= _sp.POISON_MIN_ABS)
                    else:
                        reads = bool(d_ab > 100.0 * max(d_ac, 1e-6) and d_ab > 0.05)
                    rec["poisoned_vs_normal_max"] = d_ab; rec["reads_static_buffers"] = reads
            if rec.get("poisoned_vs_normal_max") == float("inf"):
                rec["poisoned_vs_normal_max"] = "nonfinite"
            rec["verdict"] = "ok" if rec.get("reads_static_buffers") else "failed"
            if not rec.get("reads_static_buffers"):                       # was a raise: the process verdict records it (poison=failed on the SAMPLER record), the signature steps aside by name
                _sp.POISON_STATUS.update(ran=True, failed=True)
            return self._verdict_reached(ent, rec)

        def _poison_classes_count(self, ent) -> int:
            """The class count the per-class probe reports for this entry (its own enumeration: the per-step buffers but x_l, the conditioning tensors one
            level deep, the bound hoist's slots)."""
            kinds = set()

            def add(name, t):
                if isinstance(t, torch.Tensor) and t.is_floating_point() and t.numel():
                    kinds.add(_sp.poison_class_key(name))
            for k, v in (ent.get("st") or {}).items():
                if k != "x_l":
                    add("st." + k, v)
            for k, v in (ent.get("cond") or {}).items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        add("cond." + k + "." + kk, vv)
                else:
                    add("cond." + k, v)
            bc = self.__dict__.get("biascache")
            if bc is not None and getattr(bc, "cur", None):
                for name, slot in (bc.cur.get("slots") or {}).items():
                    add("hoist." + str(name), getattr(slot, "buf", None))
            return len(kinds)

        def _poison_test_classes(self, ent, N_atom):
            """The vendored per-class probe, judged as the module docstring states (rec["verdict"]: "ok" | "nonfinite" | "failed"; never raises to the run)."""
            self._arm_aside()
            if not self._reference_finite(ent):                           # nothing to judge: NaN-poisoning a class cannot turn an already non-finite output non-finite
                rec = {"N_atom": N_atom, "mode": "per-class NaN, then magnitude", "classes": self._poison_classes_count(ent), "verdict": "nonfinite",
                       "reads_static_buffers": None, "why": "the reference replay's step output is non-finite before any class is poisoned: the per-class probe cannot be read"}
                self.stats["poison"].append(rec); ent["poison_classes"] = rec
                return self._verdict_reached(ent, rec)
            try:
                rec = base._poison_test_classes(self, ent, N_atom)        # prints its `poison per-class probe:` line and fills POISON_STATUS as before
            except RuntimeError as ex:
                if not str(ex).startswith("poison test FAILED"):
                    raise
                rec = ent["poison_classes"]                               # stored before the raise; POISON_STATUS already says failed
            rec["verdict"] = "ok" if rec.get("reads_static_buffers") else "failed"
            return self._verdict_reached(ent, rec)

        # ----- the step-aside by name
        def _step_aside(self, ent, key, reason: str, N_atom, *, stage: str, detail: str = ""):
            """Serve this signature with the eager step body, BY NAME: the captured graph (if any) is dropped — its pool share released as an eviction
            releases it —, the entry keeps its static buffers (its eager steps and a same-shape next item run on them; evicted by count like any entry),
            the aside is counted under `reason`, recorded as an event and said once on stderr.  Returns the entry (ent["graph"] is None,
            ent["unsupported"] names the aside)."""
            self._arm_aside()
            had_graph = ent.get("graph") is not None
            ent["graph"] = None
            if had_graph and self.pool_mode != "private":
                graphed.POOLS.release(self.family)
            self._prep_chain = None                                       # lever sampler_prep[pool_chain]: nothing borrows a pool past this point
            self.stats["aside"][reason] = int(self.stats["aside"].get(reason, 0)) + 1
            ent["aside"] = reason
            ent["unsupported"] = f"aside={reason} ({stage}): this signature's steps run the same step body eagerly" + (f" — {detail}" if detail else "")
            self.entries[key] = ent
            if key not in self.order:
                self.order.append(key)
            n_tok = key[4] if isinstance(key, tuple) and len(key) > 4 else None
            self.stats["events"].append({"event": "sampler_aside", "reason": reason, "stage": stage, "N_atom": N_atom, "N_token": n_tok, "detail": str(detail)[:300]})
            if reason == ASIDE_NONFINITE:
                why = ("the step output the self-test would compare is non-finite before anything is poisoned (the input's own numerics; the stock sampler is non-finite "
                       "there too), so no captured graph of this signature can be judged: its steps run the same step body eagerly (stock arithmetic, stock random "
                       "stream) — counted, by design, not a fallback")
            else:
                why = ("poison test FAILED: the captured graph does not read the live static buffers — the graph is dropped, this signature's steps (and every later "
                       "signature of this process) run the same step body eagerly (stock arithmetic), and the process verdict carries poison=failed")
            print(f"{UNIT_PREFIX} SAMPLER route=eager reason={reason} stage={stage} N_token={n_tok} N_atom={N_atom} aside={self._aside_token()} — {why}"
                  + (f" [{str(detail)[:400]}]" if detail else ""), file=sys.stderr, flush=True)
            return ent

        # ----- the capture, bracketed
        def _capture(self, key, denoise_net, cond, x0, dtype, batch_shape, chunk_n_sample, N_atom, *rest, **kwargs):
            self._arm_aside()
            probe: Dict[str, Any] = {"st": None, "pre": None, "verdict": None}
            parked: Optional[list] = None
            if self.stats["poison"] and not self._poison_decided():       # records that decided nothing (non-finite probes) leave poison_once armed: they wait outside the vendored frame
                parked, self.stats["poison"] = self.stats["poison"], []
            self._probe = probe
            try:
                try:
                    ent = base._capture(self, key, denoise_net, cond, x0, dtype, batch_shape, chunk_n_sample, N_atom, *rest, **kwargs)
                except _Verdict as sig:                                   # a verdict on the fresh graph, reached inside the vendored frame: the signature steps aside BY NAME
                    torch.cuda.synchronize()
                    rec = sig.rec
                    if rec.get("verdict") == "failed":
                        self._poison_failed = True
                        return self._step_aside(sig.ent, key, ASIDE_MISMATCH, N_atom, stage="selftest", detail=str({k: v for k, v in rec.items() if k != "why"})[:400])
                    return self._step_aside(sig.ent, key, ASIDE_NONFINITE, N_atom, stage="selftest", detail=str(rec.get("why") or ""))
            finally:
                self._probe = None
                if parked is not None:
                    self.stats["poison"] = parked + self.stats["poison"]
            pre = probe["pre"]
            if pre:                                                       # decided between the warm-up and the capture: the vendored frame took its no-capture route for this signature; name it
                reason, stage, detail = pre
                events = self.stats["events"]
                if events and events[-1].get("event") == "sampler_capture_disabled":
                    events.pop()                                          # that event names the environment switch; the aside's own event replaces it
                return self._step_aside(ent, key, reason, N_atom, stage=stage, detail=detail)
            return ent

        def summary(self):
            out = base.summary(self)
            self._arm_aside()
            out["aside"] = {k: int(v) for k, v in self.stats["aside"].items() if v}   # last key: a run where nothing stepped aside prints the SUMMARY sampler= JSON as before
            return out

    PoisonAsideDenoiseLoop.__name__ = CLASS_NAME
    PoisonAsideDenoiseLoop.__qualname__ = CLASS_NAME
    setattr(PoisonAsideDenoiseLoop, MARK, True)
    _CLASSES[id(base)] = PoisonAsideDenoiseLoop
    return PoisonAsideDenoiseLoop


def bind(graphed) -> type:
    """Bind the subclass as the loop class of ``graphed`` (the name its ``install()`` constructs) and of the package that re-exports it. Returns the class."""
    cls = loop_class(graphed)
    graphed.GraphedDenoiseLoop = cls
    pkg = sys.modules.get(PACKAGE_MODULE)
    if pkg is not None and hasattr(pkg, "GraphedDenoiseLoop"):
        pkg.GraphedDenoiseLoop = cls
    return cls


def install(graphed=None) -> str:
    """Arm the step-aside on the ``infopt_graphs.protenix.graphed`` this process imports (``graphed``: that module, else imported here). Never raises:
    returns ``armed(<class> over <file>)`` or ``unavailable(<repr>)`` (said once on stderr by name; the vendored loop's own semantics then apply and the
    sampler hook reports its state as before)."""
    try:
        mod = graphed if graphed is not None else importlib.import_module(GRAPHED_MODULE)
        cls = bind(mod)
        _STATE.update(installed=True, error=None, graphed_file=getattr(mod, "__file__", None))
        return f"armed({cls.__name__} over {getattr(mod, '__file__', '?')})"
    except Exception as e:  # noqa: BLE001 — said by name, never silent; activation continues
        _STATE.update(installed=False, error=repr(e)[:300])
        print(f"{PREFIX} SAMPLER_ASIDE:unavailable({e!r})"[:400], file=sys.stderr, flush=True)
        return f"unavailable({e!r})"[:300]


def state() -> Dict[str, Any]:
    return dict(_STATE)
