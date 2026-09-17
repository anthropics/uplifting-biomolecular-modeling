"""
E10 layer_unroll — a scanned layer stack applied as a static Python loop, so every layer's weights are read at STATIC offsets.

joltz applies a stack of identical layers with `jax.lax.scan(body, carry, stacked_parameters)` (DiffusionTransformer2: the diffusion module's
token transformer, 24 layers × 25 sampling steps, and its atom encoder/decoder transformers): the scan's loop body receives layer i's parameters
as `dynamic-slice(stacked, i)` of every leaf, and XLA materialises each such slice that feeds a GEMM as a device-to-device copy — one per
parameter leaf per layer per iteration, every design step. Applying the same layers in a Python loop indexes the stacked leaves at constant
offsets (`leaf[i]`): the arithmetic, the kernels and the layer order are upstream's; only how the weights are addressed changes.

    from mosaic.fast import layer_unroll
    layer_unroll.install()               # once per process: rebinds joltz.DiffusionTransformer2.__call__
    layer_unroll.configure(None)         # = "sampler", the default setting (SETTING); "stock" = upstream's scan, untouched; clears the jax /
                                         # equinox trace caches either way
    layer_unroll.describe()              # the effective configuration, single-token scalar values (unrolled_traced counts the unrolled traces)
    layer_unroll.emit_line(tag)          # the LEVER line (opt_core.report grammar); the off line when not configured
    layer_unroll.gate()                  # AFTER the run (levers.finalize): LeverRefused when installed but not configured, or configured but never
                                         # engaged (no unrolled trace since configure) — fail closed; require_engaged is the same function

Modes (MODES): "stock" — DiffusionTransformer2.__call__ is upstream's function; "sampler" — the rebound call runs upstream's bias reshaping
verbatim, then `for i in range(depth): a = checkpoint(layer_i)(a, ...)` with layer_i = eqx.combine(static, leaf[i] of every stacked leaf)
(upstream checkpoints its scan body the same way). Nothing depends on N, the binder, the target or the depth. It is NOT bitwise equal to the scan
(XLA fuses across the layer boundaries the static loop exposes: 1-2 ulp on CPU already); its numerics class is the fast tier's.

COST, reported beside the number and never folded into it: the static loop is traced and compiled layer by layer, so every executable holding the
sampler (the design step's, and the refold's) compiles several-fold longer ONCE per process (results.json compile_plus_first_s carries it). The
steady per-step and per-refold times are what improve.
"""

import hashlib

LEVER = "E10"
NAME = "layer_unroll"
MODES = ("stock", "sampler")
SETTING = "sampler"                                                # the default setting: configure(None) / configure("") apply it
ENV_REQUIRED = {}                                                  # no environment variable takes part in this lever
CFG = {"mode": "stock", "unrolled_traced": 0, "depths": []}
_ORIG = {}


def _file_sha256():
    try:
        with open(__file__, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return "unavailable"


FILE_SHA256 = _file_sha256()


class LeverRefused(RuntimeError):
    """The lever refused by name (a spec that is not a mode; configure('sampler') before install())."""


def _dt2_call(self, a, s, bias=None, mask=None, to_keys=None, multiplicity=1):
    if CFG["mode"] != "sampler":
        return _ORIG["call:dt2"](self, a, s, bias=bias, mask=mask, to_keys=to_keys, multiplicity=multiplicity)
    import einops
    import equinox as eqx
    import jax
    if self.pair_bias_attn:                                               # upstream's bias handling, verbatim (joltz DiffusionTransformer2.__call__)
        B, N, M, D = bias.shape
        L = self.depth
        bias = bias.reshape(B, N, M, L, D // L)
    bias = einops.rearrange(bias, "... l p -> l ... p")
    static = self.static

    @jax.checkpoint
    def layer_fn(params, a, b):
        layer = eqx.combine(static, params)
        return layer(a=a, s=s, bias=b, mask=mask, to_keys=to_keys)

    depth = int(self.depth)
    for i in range(depth):
        params_i = jax.tree.map(lambda x: x[i], self.stacked_parameters)  # constant offsets: layer i's leaves
        a = layer_fn(params_i, a, bias[i])
    CFG["unrolled_traced"] += 1
    CFG["depths"].append(depth)
    return a


def install():
    """Rebind joltz.DiffusionTransformer2.__call__ (idempotent). Mode stays what configure() last set (stock at import)."""
    if _ORIG.get("installed"):
        return describe()
    import joltz
    cls = joltz.DiffusionTransformer2
    _ORIG[(cls, "__call__")] = cls.__dict__["__call__"]
    _ORIG["call:dt2"] = getattr(cls, "__call__")
    setattr(cls, "__call__", _dt2_call)
    _ORIG["installed"] = True
    return describe()


def uninstall():
    if _ORIG.get("installed"):
        for k, raw in list(_ORIG.items()):
            if isinstance(k, tuple):
                setattr(k[0], k[1], raw)
    _ORIG.clear()
    CFG.update({"mode": "stock", "unrolled_traced": 0, "depths": []})
    _clear_caches()
    return describe()


def _clear_caches():
    try:
        import jax
        jax.clear_caches()
    except Exception:  # noqa: BLE001
        pass
    try:
        import equinox as eqx
        eqx.clear_caches()
    except Exception:  # noqa: BLE001
        pass


def configure(spec=None) -> dict:
    """spec: None | '' -> SETTING ('sampler', the default setting); 'sampler' -> the diffusion transformers unrolled; 'stock' -> upstream's scan.
    A non-stock mode needs install() first (LeverRefused otherwise). Clears the jax / equinox caches and the counters. Returns describe()."""
    word = SETTING if spec in (None, "") else str(spec).strip().lower()
    if word not in MODES:
        raise ValueError(f"layer_unroll: unknown spec {spec!r}; one of {'|'.join(MODES)}")
    if word != "stock" and not _ORIG.get("installed"):
        raise LeverRefused(f"layer_unroll_not_installed: configure({spec!r}) = mode {word} before install() would change nothing — call layer_unroll.install() first")
    _clear_caches()
    CFG.update({"mode": word, "unrolled_traced": 0, "depths": []})
    return describe()


def installed() -> bool:
    return bool(_ORIG.get("installed"))


def gate():
    """Fail closed (run by mosaic_opt.levers.finalize() AFTER the phases): LeverRefused('layer_unroll_not_configured') when installed
    but mode stock (a run under the lever's name on upstream's scan); LeverRefused('layer_unroll_not_engaged') when installed and configured but no
    DiffusionTransformer2 trace took the unrolled path since configure() (a cached executable, or a stack that no longer routes through
    joltz.DiffusionTransformer2.__call__). Nothing to check when not installed."""
    if not installed():
        return describe()
    if CFG["mode"] == "stock":
        raise LeverRefused("layer_unroll_not_configured: E10 is installed but configure() left mode stock — the run would carry the lever's name on upstream's scanned stack")
    if CFG["unrolled_traced"] < 1:
        raise LeverRefused("layer_unroll_not_engaged: mode sampler, but no DiffusionTransformer2 trace took the unrolled path since configure() (a cached executable, or a stack "
                           "that no longer routes through joltz.DiffusionTransformer2.__call__)")
    return describe()


require_engaged = gate                                             # the post-run engagement check IS the gate (one home)


def describe() -> dict:
    """The effective configuration, every value a single-token scalar: {lever, lever_name, impl, file_sha256, origin, installed, mode,
    setting_of_record, stacks, engaged, unrolled_traced, depths}. `engaged` is process-level since configure(), never per phase."""
    on = CFG["mode"] == "sampler" and installed()
    depths = ",".join(str(d) for d in sorted(set(CFG["depths"]))) or "none"
    return {"lever": LEVER, "lever_name": NAME, "impl": __name__, "file_sha256": FILE_SHA256[:16], "origin": "kit", "installed": int(installed()), "mode": CFG["mode"],
            "setting_of_record": SETTING, "stacks": "DiffusionTransformer2" if on else "stock", "engaged": "yes" if (on and CFG["unrolled_traced"] >= 1) else "no",
            "unrolled_traced": int(CFG["unrolled_traced"]), "depths": depths}


def emit_line(tag: str = "") -> str:
    """The LEVER line of this process (opt_core.report's grammar, printed on stderr via emit): state=on with describe()'s facts when installed and
    configured, else the off line with its reason."""
    from opt_core.report import emit, lever_line
    d = describe()
    if not installed() or d["mode"] == "stock":
        return emit(lever_line(tag, f"{LEVER}.{NAME}", "off", reason=("not_installed" if not installed() else "not_configured"), impl=d["impl"], origin=d["origin"]))
    facts = {k: v for k, v in d.items() if k not in ("impl", "origin")}
    return emit(lever_line(tag, f"{LEVER}.{NAME}", "on", impl=d["impl"], origin=d["origin"], **{k: str(v) for k, v in facts.items()}))
