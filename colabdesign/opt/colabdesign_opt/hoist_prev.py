"""Lever `hoist_prev` — the recycle features never leave the device between design steps (exact: the traced programs are stock's, every value
BindCraft or ColabDesign reads is stock's byte for byte).

What stock does every design step (`colabdesign/af/design.py`, `_af_design`):

* `_recycle` (design.py:162-181) rebuilds the recycle-0 features `prev = {prev_msa_first_row: zeros[L,256], prev_pair: zeros[L,L,128],
  prev_pos: zeros[L,37,3]}` as float64 NumPy arrays, which the forward-only executable `fn` receives as arguments: jit converts them
  float64→float32 on the host and copies them to the device — 21 MB at 200 tokens, 328 MB at 800 (float32), EVERY step, always zeros.
* `run` (design.py:97-106) pulls EVERY leaf of the step's aux to the host through `np.stack` over the models run and averages it
  (`mean(0)`); the aux of the design step carries `prev` = the last recycle's `prev_pair` (L×L×128 float32: 328 MB at 800 tokens) +
  `prev_msa_first_row` + `prev_pos`, which nothing on the host ever reads — a device→host copy and a host `mean` every step.

This lever replaces `_af_design._recycle` and `_af_design.run` with the same code except:

* the recycle-0 zeros are device-resident float32 arrays made once per (token count, dtype, device) and reused (`zero_prev`); an entry
  ColabDesign's `clear_mem()` deleted is remade. `prev_pos` under `use_initial_guess` (the target's coordinates) and `initial_atom_pos`
  are passed exactly as stock passes them (host arrays) — only the zeros are hoisted;
* when ONE model ran in the step (BindCraft's every design step and prediction: `num_models=1`), `prev` is taken out of the aux before the
  host stack and put back afterwards AS DEVICE ARRAYS: `aux["all"]["prev"]` = the last recycle's arrays behind a view that gives them their
  model axis on access (`_AllPrev`, no copy — stock: `np.stack` over one model, a plain copy), `aux["prev"][k]` = the same arrays with their
  zeros canonicalised on the device (`canonical_zeros`: −0.0 → +0.0, every other value untouched) — exactly what stock's host `mean(0)` over
  one model does to them (numpy's mean adds each element to +0.0 and divides by one: the bytes of every value but a negative zero survive,
  and the structure module does emit −0.0 in `prev_pos` for absent atoms). So `np.asarray` of either entry gives stock's bytes. With two or
  more models the step takes stock's host path unchanged (`np.stack` + `mean(0)` on the host — stock's bytes, stock's cost), counted
  `multi_model_stock_path`; no device mean is ever taken.

The one observable difference: the leaves of `af.aux["prev"]` / `af.aux["all"]["prev"]` (and of the best-step copy `af._tmp["best"]["aux"]`)
are `jax.Array`s, not `numpy.ndarray`s; `np.asarray(leaf)` gives stock's bytes. Readers checked (none reads them on the host):
ColabDesign — design.py:199/201 (`_inputs["prev"] = aux["prev"]`, `initial_atom_pos`: fed back to the executables, device arrays in stock
too at that point), model.py:208 (inside the traced program), `_save_results` / `predict` / `design_semigreedy` (dict handling only,
`copy_dict` = new containers, same leaves), utils.py / plot.py (never touch prev); BindCraft — colabdesign_utils.py:236
`pickle.dump(af_model.aux['all'], …)` under `save_trajectory_pickle` (false in every shipped settings file; with the lever the pickle holds
jax arrays — loadable where jax is), every other `aux[...]` read is `log` / `plddt` / `seq`; the kit — units.py / bindcraft.py read `log`,
`seq`, `plddt`, `atom_positions`. Device memory: the best-step aux keeps its prev arrays on the device instead of the host (+1 prev set, 0.33 GB at 800
tokens, while the best step is not the current one) + the canonical-zeros copy of the current step's prev (+0.33 GB @800) beside the raw arrays the
`all` view holds; the current step's prev arrays are alive between steps in stock too (`_inputs["prev"]`).

Evidence (`opt_core.report.lever_line`): one LEVER line at install and one at interpreter exit with the census (the last line of a name wins):

    [colabdesign-opt] LEVER name=hoist_prev state=on impl=hoist_prev@kit origin=kit patched=_recycle,run zero_builds=<n> zero_inits=<n>
                      device_prev_steps=<n> multi_model_stock_path=<n> run_calls=<n> source=<install|exit> pid=<pid>

Nothing changes a decision — no argument, no environment variable: a mode IS its lever set (modes.py).
"""
from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np

from opt_core import report as _core_report

from .names import TAG

LEVER = "hoist_prev"
IMPL = "hoist_prev@kit"
NUMERICS = "exact"                                                                 # registry.LEVERS[hoist_prev].numerics: stock's values byte for byte
# sha256 of the pinned bodies (colabdesign 1.1.3 @ e31a56fe, colabdesign/af/design.py) this module re-states (`_recycle`, `run`) or relies on
# (`_single`: called, not re-stated — its contract, aux["prev"] a dict of arrays and aux["grad"], is what `run` below assumes), each `def` as
# `ast` delimits it. install() refuses by name on any other body: the lever re-states THESE bodies and no others (tests/test_hoist_prev.py).
PINNED = {"_recycle": "097e396fafa0370870be9f96b905e0701449b369fbd40274508bc9e1ba963462",
          "run": "35952f7f70dab2df9ea44aff67d803cbd2bfb4fa811585b97fd4f28d5f802ea9",
          "_single": "7a39f289c90693fc60b04b94c9f19625f2700b9b05a8e4a90df222149be1acce"}


class HoistPrevError(RuntimeError):
    """colabdesign.af.design is not the pinned design loop: the lever refuses by name (cannot_run) rather than re-state a body over another."""
    cannot_run = True


REFUSALS = (HoistPrevError,)


def body_source(src: str, name: str) -> str:
    """The text of `def <name>` in module source `src`, as ast delimits it (KeyError when absent)."""
    import ast
    lines = src.splitlines()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return "\n".join(lines[node.lineno - 1: node.end_lineno])
    raise KeyError(name)


def require_pinned(module) -> Dict[str, str]:
    """{name: sha256} of the three bodies in `module`'s source file; HoistPrevError naming the first one that is absent or not the pinned digest."""
    import hashlib
    path = getattr(module, "__file__", None)
    try:
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
    except (TypeError, OSError) as e:
        raise HoistPrevError(f"hoist_prev: cannot read the source of {getattr(module, '__name__', module)!r} ({e}): the lever re-states colabdesign 1.1.3's _recycle/run and installs on no other design loop") from None
    got = {}
    for name, digest in PINNED.items():
        try:
            got[name] = hashlib.sha256(body_source(src, name).encode()).hexdigest()
        except KeyError:
            raise HoistPrevError(f"hoist_prev: {path} defines no `{name}`: not colabdesign 1.1.3's design loop (colabdesign/af/design.py) — the lever re-states its _recycle/run and installs on no other") from None
        if got[name] != digest:
            raise HoistPrevError(f"hoist_prev: `{name}` in {path} is not the pinned body (sha256 {got[name][:12]} != {digest[:12]}): the lever re-states colabdesign 1.1.3's _recycle/run and installs on no other")
    return got
_STOCK: Dict[str, object] = {}                                                     # colabdesign's own _recycle / run, kept by install() for uninstall()
MARKER = "_colabdesign_opt_hoist_prev"
PATCHED = ("_recycle", "run")
STOCK_SOURCE = {"_recycle": "colabdesign/af/design.py:147-206", "run": "colabdesign/af/design.py:80-133"}   # the stock bodies re-stated below (colabdesign 1.1.3 @ e31a56fe, stock/PINS.json)
_CENSUS: Dict[str, int] = {"zero_builds": 0, "zero_inits": 0, "device_prev_steps": 0, "multi_model_stock_path": 0, "run_calls": 0}
_ZEROS: Dict[Tuple, dict] = {}                                                      # (L, dtype name, use_dgram, device) -> {name: device zeros}
_LINES: list = []


class _AllPrev(dict):
    """`aux["all"]["prev"]` without a copy: the last recycle's device arrays, given their leading model axis (of one model) on access —
    the values stock's `np.stack` over one model holds. A dict subclass is a pytree LEAF to jax.tree_util (ColabDesign's `copy_dict` keeps it whole)."""
    def __getitem__(self, k):
        return dict.__getitem__(self, k)[None]

    def get(self, k, default=None):
        return self[k] if k in self else default

    def items(self):
        return [(k, self[k]) for k in self.keys()]

    def values(self):
        return [self[k] for k in self.keys()]

    def _raw(self) -> dict:
        return {k: dict.__getitem__(self, k) for k in self.keys()}

    def __reduce__(self):                                                          # pickle (BindCraft's save_trajectory_pickle) stores the un-expanded arrays: the view round-trips as a view
        return (_AllPrev, (self._raw(),))

    def __repr__(self):
        return f"_AllPrev({self._raw()!r})"


def _prev_dtype():
    """The dtype jit gives stock's float64 zeros: float32 (float64 only under jax x64, as stock's would be)."""
    import jax
    return jax.dtypes.canonicalize_dtype(np.float64)


def zero_prev(L: int, use_dgram: bool) -> dict:
    """Stock's recycle-0 zeros (design.py:163-173) as device arrays, made once per (L, dtype, use_dgram, default device) and reused;
    remade when ColabDesign's clear_mem() deleted them."""
    import jax
    import jax.numpy as jnp
    dt = _prev_dtype()
    dev = jax.devices()[0]
    key = (int(L), np.dtype(dt).name, bool(use_dgram), str(dev))
    z = _ZEROS.get(key)
    if z is not None and any(x.is_deleted() for x in z.values()):
        z = None
    if z is None:
        z = {'prev_msa_first_row': jnp.zeros([L, 256], dt),
             'prev_pair': jnp.zeros([L, L, 128], dt),
             'prev_pos': jnp.zeros([L, 37, 3], dt)}
        if use_dgram:
            z['prev_dgram'] = jnp.zeros([L, L, 64], dt)
        _ZEROS[key] = z
        _CENSUS["zero_builds"] += 1
    return z


_CANON: dict = {}


def canonical_zeros(tree):
    """Every −0.0 of the float leaves → +0.0 on the device, nothing else touched (a `where`, not an add XLA could fold): the bytes stock's host
    `np.stack(...).mean(0)` over ONE model gives these arrays (numpy adds each element to +0.0)."""
    import jax
    import jax.numpy as jnp
    fn = _CANON.get("fn")
    if fn is None:
        def _canon(t):
            return jax.tree_util.tree_map(lambda x: jnp.where(x == 0, jnp.zeros((), x.dtype), x) if jnp.issubdtype(x.dtype, jnp.floating) else x, t)
        fn = _CANON["fn"] = jax.jit(_canon)
    return fn(tree)


def census() -> dict:
    return dict(_CENSUS)


def line(source: str) -> str:
    return _core_report.lever_line(TAG, LEVER, "on", ("patched", ",".join(PATCHED)), *census().items(), impl=IMPL, origin="kit", source=source, pid=os.getpid())


def _exit_line() -> None:
    try:
        _core_report.emit(line("exit"))
    except Exception:  # noqa: BLE001 — an exit hook never raises
        pass


def install() -> dict:
    """Replace `_af_design._recycle` and `_af_design.run` (idempotent; marker MARKER). Call before or after models are built — the patched
    methods are looked up on the class at call time."""
    import atexit
    import jax
    from colabdesign.af.design import _af_design
    from colabdesign.shared.utils import to_float, to_list
    if getattr(getattr(_af_design, "_recycle", None), MARKER, False):
        return {"impl": IMPL, "patched": list(PATCHED), "already": True}
    import sys
    require_pinned(sys.modules["colabdesign.af.design"])                            # ColabDesign's bodies are the pinned ones, or the lever refuses by name

    def _recycle(self, model_params, num_recycles=None, backprop=True):
        '''multiple passes through the model (aka recycle) — colabdesign_opt hoist_prev: stock's body (design.py:147-206) with the recycle-0
        zeros device-resident (zero_prev) instead of float64 host arrays rebuilt every step'''
        a = self._args
        mode = a["recycle_mode"]
        if num_recycles is None:
            num_recycles = self.opt["num_recycles"]

        if mode in ["backprop", "add_prev"]:
            aux = self._single(model_params, backprop)
        else:
            L = self._inputs["residue_index"].shape[0]
            if "prev" not in self._inputs or a["clear_prev"]:
                prev = dict(zero_prev(L, a["use_dgram"]))                          # a fresh dict over the shared device zeros (stock: np.zeros, float64)
                _CENSUS["zero_inits"] += 1
                if a["use_initial_guess"] and "batch" in self._inputs:
                    prev["prev_pos"] = self._inputs["batch"]["all_atom_positions"]   # stock's host array, passed as stock passes it
                if a["use_initial_atom_pos"]:
                    if "batch" in self._inputs:
                        self._inputs["initial_atom_pos"] = self._inputs["batch"]["all_atom_positions"]
                    else:
                        self._inputs["initial_atom_pos"] = np.zeros([L, 37, 3])
            self._inputs["prev"] = prev                                            # noqa: F821 — as stock (design.py:181): unbound when clear_prev is off and prev is present
            cycles = (num_recycles + 1)
            mask = [0] * cycles
            if mode == "sample":  mask[np.random.randint(0, cycles)] = 1
            if mode == "average": mask = [1 / cycles] * cycles
            if mode == "last":    mask[-1] = 1
            if mode == "first":   mask[0] = 1
            grad = []
            for m in mask:
                if m == 0:
                    aux = self._single(model_params, backprop=False)
                else:
                    aux = self._single(model_params, backprop)
                    grad.append(jax.tree_util.tree_map(lambda x: x * m, aux["grad"]))
                self._inputs["prev"] = aux["prev"]
                if a["use_initial_atom_pos"]:
                    self._inputs["initial_atom_pos"] = aux["prev"]["prev_pos"]
            aux["grad"] = jax.tree_util.tree_map(lambda *x: np.stack(x).sum(0), *grad)

        aux["num_recycles"] = num_recycles
        return aux

    def run(self, num_recycles=None, num_models=None, sample_models=None, models=None,
            backprop=True, callback=None, model_nums=None, return_aux=False):
        '''run model to get outputs, losses and gradients — colabdesign_opt hoist_prev: stock's body (design.py:80-133); with one model the
        recycle features stay on the device instead of riding through np.stack / mean(0) on the host'''
        _CENSUS["run_calls"] += 1
        for fn in self._callbacks["design"]["pre"]: fn(self)

        if model_nums is None:
            model_nums = self._get_model_nums(num_models, sample_models, models)
        assert len(model_nums) > 0, "ERROR: no model params defined"

        auxs = []
        for n in model_nums:
            p = self._model_params[n]
            auxs.append(self._recycle(p, num_recycles=num_recycles, backprop=backprop))

        hoisted = None
        if len(auxs) == 1 and isinstance(auxs[0].get("prev"), dict):
            hoisted = auxs[0].pop("prev")                                          # the last recycle's device arrays: kept off the host path
            _CENSUS["device_prev_steps"] += 1
        elif any("prev" in x for x in auxs):
            _CENSUS["multi_model_stock_path"] += 1                                 # two or more models: stock's host stack + mean, unchanged
        auxs = jax.tree_util.tree_map(lambda *x: np.stack(x), *auxs)

        def avg_or_first(x):
            if np.issubdtype(x.dtype, np.integer): return x[0]
            else: return x.mean(0)

        self.aux = jax.tree_util.tree_map(avg_or_first, auxs)
        self.aux["atom_positions"] = auxs["atom_positions"][0]
        self.aux["all"] = auxs
        if hoisted is not None:
            self.aux["prev"] = canonical_zeros(hoisted)                            # stock: the host mean over one model of these arrays = these values with −0.0 → +0.0 (numpy's effect)
            self.aux["all"]["prev"] = _AllPrev(hoisted)                            # stock: np.stack over one model = these arrays with a model axis (raw bytes, −0.0 kept)

        for fn in (self._callbacks["design"]["post"] + to_list(callback)): fn(self)

        self.aux["log"] = {**self.aux["losses"]}
        self.aux["log"]["plddt"] = 1 - self.aux["log"]["plddt"]
        for k in ["loss", "i_ptm", "ptm"]: self.aux["log"][k] = self.aux[k]
        for k in ["hard", "soft", "temp"]: self.aux["log"][k] = self.opt[k]

        if self.protocol in ["fixbb", "partial"] or (self.protocol == "binder" and self._args["redesign"]):
            if self.protocol == "partial":
                aatype = self.aux["aatype"][..., self.opt["pos"]]
            else:
                aatype = self.aux["seq"]["pseudo"].argmax(-1)
            mask = self._wt_aatype != -1
            true = self._wt_aatype[mask]
            pred = aatype[..., mask]
            self.aux["log"]["seqid"] = (true == pred).mean()

        self.aux["log"] = to_float(self.aux["log"])
        self.aux["log"].update({"recycles": int(self.aux["num_recycles"]), "models": model_nums})

        if return_aux: return self.aux

    for f in (_recycle, run):
        setattr(f, MARKER, True)
    _STOCK.setdefault("_recycle", _af_design.__dict__.get("_recycle")); _STOCK.setdefault("run", _af_design.__dict__.get("run"))
    _af_design._recycle = _recycle
    _af_design.run = run
    _LINES.append(_core_report.emit(line("install")))
    atexit.register(_exit_line)
    return {"impl": IMPL, "patched": list(PATCHED), "already": False}


def off_line(reason: str) -> str:
    """The lever's line in a run that does not select it (state=off with the reason; no census)."""
    return _core_report.lever_line(TAG, LEVER, "off", reason=reason, impl=IMPL, origin="kit")


def uninstall() -> None:
    """Put colabdesign's own `_recycle` / `run` back on `_af_design` (models built afterwards AND already built take stock's path: the methods are looked up at call time)."""
    import sys
    d = sys.modules.get("colabdesign.af.design")
    if not d or not installed():
        return
    for n in PATCHED:
        f = _STOCK.get(n)
        if f is not None:
            setattr(d._af_design, n, f)
        elif n in d._af_design.__dict__:
            delattr(d._af_design, n)


def installed() -> bool:
    import sys
    d = sys.modules.get("colabdesign.af.design")
    return bool(d and all(getattr(getattr(d._af_design, n, None), MARKER, False) for n in PATCHED))


def evidence() -> dict:
    """The census ({} when not installed) and the lines printed so far."""
    if not installed():
        return {}
    return {**census(), "patched": list(PATCHED), "lines": list(_LINES)}
