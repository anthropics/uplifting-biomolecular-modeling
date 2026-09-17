"""Lever `nosub` — the design step traced unchunked; the kit adapter of the shared sub-batch policy `opt_core.jax_design.subbatch_policy`
on ColabDesign's model preparation.

Stock (`colabdesign/af/prep.py:25-35`) builds both executables of a design model from ONE config object: it traces `subbatch_size=None`,
then — above 384 tokens — sets `subbatch_size=4` on that same object before either executable is traced, so the forward-only `fn` AND the
forward+backward `grad_fn` run chunked above 384 tokens. This lever replaces `_af_prep._prep_model` so that the two executables get their
own config objects and their own decisions:

* `fn` (prediction): stock's rule at every size — chunks of 4 above SIZE_GATE_TOKENS, none at or below (`subbatch_policy.choose(requested="stock")`);
* `grad_fn` (the design step): unchunked at every size (`subbatch_policy.fixed(value=None)`, source `kit`). No memory policy and no runtime
  fallback: chunking the remat'd backward pass lowers the peak of nothing this kit runs (with the `pallas` kernel the logits chunking bounded
  are never materialised; with stock attention chunking raises the peak). A complex too large for the card fails with XLA's out-of-memory
  error at the first design step, as stock does.

At or below SIZE_GATE_TOKENS both decisions equal stock's (no chunk) and the traced programs are stock's: the size gate `GATE` counts the
build as `gated` and the guarantee is bitwise. Above it the design step re-associates its accumulation (numerics class `precision`). The PATCH is this
kit's (it names ColabDesign's classes); the DECISION words and the census are the core's. Evidence: one LEVER line per distinct build
(`opt_core.report.lever_line`), e.g.

    [colabdesign-opt] LEVER name=nosub state=on impl=subbatch_policy@<core version> origin=core tokens=431 grad_subbatch=none grad_subbatch_source=kit
                      fn_subbatch=4 fn_subbatch_source=stock gate.nosub=min385 calls=1 served=1 gated=0 fallback=0

`state=skipped reason=gated` at or below the gate (stock's programs); `state=on` above it. Nothing changes a decision — no argument, no
environment variable, no module setting: a mode IS its lever set (modes.py).
"""
from __future__ import annotations

import copy
from typing import List

import opt_core
from opt_core import report as _core_report
from opt_core.attn.size_gate import SizeGate
from opt_core.jax_design import subbatch_policy as _policy

from .modes import SIZE_GATE_TOKENS, STOCK_SUBBATCH
from .names import LEVER_NOSUB, LEVER_NOSUB_FN, TAG

IMPL = f"subbatch_policy@{opt_core.__version__}"
STOCK_RULE = _policy.threshold_rule(SIZE_GATE_TOKENS, STOCK_SUBBATCH)               # stock's rule: chunks of 4 above the gate, unchunked at or below (prep.py:28-31)
GATE = SizeGate(LEVER_NOSUB, min_tokens=SIZE_GATE_TOKENS + 1, source="colabdesign/af/prep.py:30")   # served = the lever can change the design step's program
MARKER = "_colabdesign_opt_nosub"
FN_REQUEST = "stock"                                                                # the forward-only executable under `nosub` alone: stock's size rule
KIT, STOCK = "kit", "stock"                                                         # a policy per executable: `kit` = unchunked at every size (source kit), `stock` = stock's rule
POLICY = {"grad": STOCK, "fn": STOCK}                                              # what each executable gets in THIS process: `nosub` sets grad=kit (install), `nosub_fn` sets fn=kit (nosub_fn.py)
GATE_FN = SizeGate(LEVER_NOSUB_FN, min_tokens=SIZE_GATE_TOKENS + 1, source="colabdesign/af/prep.py:30")   # nosub_fn's size gate: the same rule, its own event
_BUILDS: List[dict] = []                                                            # one record per distinct (tokens, grad, fn) build in this process
_LINES: List[str] = []


def _decision(tokens: int, policy: str):
    """One executable's sub-batch decision under `policy`: kit = unchunked (value None, source kit); stock = stock's size rule (source stock)."""
    if policy == KIT:
        return _policy.fixed(tokens=tokens, value=None, stock_value=STOCK_SUBBATCH)
    return _policy.choose(tokens=tokens, stock_value=STOCK_SUBBATCH, requested=FN_REQUEST, stock_rule=STOCK_RULE)


def decide(tokens: int, grad: str = KIT, fn: str = STOCK) -> dict:
    """The two decisions for a model of `tokens` tokens: {"tokens", "grad": SubbatchDecision, "fn": SubbatchDecision, "gate": Decision, "gate_fn": Decision}.
    Defaults = lever `nosub` alone: grad unchunked, source `kit`; fn stock's rule, source `stock`. The hook passes this process's POLICY."""
    return {"tokens": int(tokens), "grad": _decision(tokens, grad), "fn": _decision(tokens, fn), "gate": GATE.decide(int(tokens)), "gate_fn": GATE_FN.decide(int(tokens)),
            "policy": {"grad": grad, "fn": fn}}


def line_of(d: dict) -> str:
    """The LEVER line of one build: the size gate decides the word (on | skipped gated)."""
    pairs = [("tokens", d["tokens"])] + list(d["grad"].fields("grad_subbatch").items()) + list(d["fn"].fields("fn_subbatch").items()) + GATE.evidence()
    state, reason = ("on", None) if d["gate"].served else ("skipped", "gated")
    return _core_report.lever_line(TAG, LEVER_NOSUB, state, *pairs, reason=reason, impl=IMPL, origin="core")


def line_of_fn(d: dict) -> str:
    """Lever nosub_fn's LEVER line of one build: the forward-only executable's decision (unchunked, source kit) and what stock's rule would have
    traced (`stock_fn_subbatch`); the size gate decides the word — at or below it stock's fn is unchunked too (skipped gated: stock's program)."""
    pairs = [("tokens", d["tokens"])] + list(d["fn"].fields("fn_subbatch").items()) + [("stock_fn_subbatch", STOCK_RULE(d["tokens"]))] + GATE_FN.evidence()
    state, reason = ("on", None) if d["gate_fn"].served else ("skipped", "gated")
    return _core_report.lever_line(TAG, LEVER_NOSUB_FN, state, *pairs, reason=reason, impl=IMPL, origin="core")


def builds() -> List[dict]:
    """Every distinct build decided in this process, as data."""
    return [{"tokens": b["tokens"], "grad": b["grad"].as_dict(), "fn": b["fn"].as_dict(), "gate": b["gate"].event, "gate_fn": b["gate_fn"].event,
             "policy": dict(b["policy"]), "line": b.get("line"), "lines": list(b.get("lines") or [])} for b in _BUILDS]


REFUSALS = ()                                                                  # nothing of this lever can fail to install (pure Python over colabdesign's prep)
NUMERICS = "precision"                                                         # registry.LEVERS[nosub].numerics: the unchunked program re-associates above the gate; stock's programs at or below it
_STOCK_PREP_MODEL = None                                                       # colabdesign's own `_af_prep._prep_model`, kept by install() for uninstall()


def off_line(reason: str) -> str:
    """The lever's line in a run that does not select it: state=off with the reason (impl = the policy module; nothing is decided)."""
    return _core_report.lever_line(TAG, LEVER_NOSUB, "off", reason=reason, impl=IMPL, origin="core")


def uninstall() -> None:
    """Lever nosub off again: the graded executable back at stock's rule; the hook itself goes once nosub_fn is off too."""
    POLICY["grad"] = STOCK
    uninstall_hook()


def evidence() -> dict:
    return {"builds": builds()}


def install_hook() -> None:
    """Replace `colabdesign.af.prep._af_prep._prep_model` (idempotent; marker MARKER) with the one that applies this process's POLICY. Both
    sub-batch levers (nosub: the graded executable; nosub_fn: the forward-only one) install it; call before any `mk_afdesign_model(...)`."""
    from colabdesign.af import prep as _prep_mod
    from colabdesign.af.prep import _af_prep
    if getattr(_af_prep._prep_model, MARKER, False):
        return
    global _STOCK_PREP_MODEL
    _STOCK_PREP_MODEL = _af_prep._prep_model                                  # kept for uninstall()

    def _prep_model(self, **kwargs):
        '''prep model (colabdesign_opt nosub): separate config objects and sub-batch decisions for the forward-only and the forward+backward executables'''
        if not hasattr(self, "_model") or self._cfg != self._model["runner"].config:
            L = int(sum(self._lengths))
            d = decide(L, grad=POLICY["grad"], fn=POLICY["fn"])
            gc = self._cfg.model.global_config
            gc.subbatch_size = d["grad"].value
            self._model = self._get_model(self._cfg)                     # grad_fn (+fn) traced later with the grad decision
            if d["fn"].value != d["grad"].value:
                cfg_fn = copy.deepcopy(self._cfg); cfg_fn.model.global_config.subbatch_size = d["fn"].value
                self._model["fn"] = self._get_model(cfg_fn)["fn"]        # fn gets its OWN config object at its own decision
            key = (L, d["grad"].value, d["fn"].value)
            if key not in {(b["tokens"], b["grad"].value, b["fn"].value) for b in _BUILDS}:
                d["lines"] = []
                if d["policy"]["grad"] == KIT:                             # lever nosub installed: its line (the graded executable's decision, and the fn one beside it)
                    d["line"] = _core_report.emit(line_of(d)); d["lines"].append(d["line"])
                if d["policy"]["fn"] == KIT:                               # lever nosub_fn installed: its line
                    d["lines"].append(_core_report.emit(line_of_fn(d)))
                _BUILDS.append(d); _LINES.extend(d["lines"])
        self._opt = _prep_mod.copy_dict(self.opt)
        self.restart(**kwargs)

    setattr(_prep_model, MARKER, True)
    _af_prep._prep_model = _prep_model


def hook_installed() -> bool:
    import sys
    prep = sys.modules.get("colabdesign.af.prep")
    return bool(prep and getattr(getattr(prep._af_prep, "_prep_model", None), MARKER, False))


def uninstall_hook() -> None:
    """Put colabdesign's own `_prep_model` back once no sub-batch lever holds a kit policy (models built afterwards are stock's)."""
    from colabdesign.af.prep import _af_prep
    if POLICY["grad"] == STOCK and POLICY["fn"] == STOCK and _STOCK_PREP_MODEL is not None and getattr(_af_prep._prep_model, MARKER, False):
        _af_prep._prep_model = _STOCK_PREP_MODEL


def install() -> None:
    """Lever nosub: the graded executable unchunked at every size (POLICY grad=kit) through the hook."""
    install_hook()
    POLICY["grad"] = KIT


def installed() -> bool:
    return POLICY["grad"] == KIT and hook_installed()
