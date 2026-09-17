"""
E1 dead_template — the template module's contribution to the pair representation, skipped when it is identically zero.

Boltz-2's trunk adds the template module's output to z on every trunk pass (joltz `Joltz2.trunk_iteration`:
`z = z + self.template_module(z, feats, pair_mask, ...)`). `TemplateV2Module.__call__` runs z_norm -> z_proj -> a 2-block PairformerNoSeq at
template_dim -> v_norm and ends in

    u = (v * template_mask).sum(axis=1) / num_templates          # template_mask = jnp.any(feats["template_mask"], axis=2)
    return self.u_proj(self.relu(u))                             # u_proj: nn.Linear(template_dim, token_z, bias=False)  (boltz trunkv2.TemplateV2Module)

so a featurization whose `template_mask` has no nonzero entry makes the contribution identically zero forward AND backward (the cotangent reaching
the module's pairformer is multiplied by the same zero mask), while the stock step still executes that pairformer both ways: XLA cannot fold a
traced mask. boltz featurizes ONE all-zero dummy template whenever no template is staged (featurizerv2 `load_dummy_templates_features(tdim=1)`);
mosaic stages a real template only for a `TargetChain(template_chain=...)` (mosaic models/boltz2.py target_only_features / build_template_yaml) —
such a featurization is never skipped: the lever steps aside BY NAME (one stderr line, a count on its census line) and the template module runs as
stock for it. The lever accepts every input stock accepts; it never refuses one.

    from mosaic.fast import dead_template
    dead_template.install()               # once per process, before any model call: rebinds joltz.TemplateV2Module.__call__ and wraps
                                          # mosaic.models.boltz2.Boltz2.{build_loss, build_multisample_loss, model_output}
    dead_template.configure(None)         # = "skip", the default setting (SETTING); "stock" = upstream's code path, the wrappers inert; clears the
                                          # jax / equinox trace caches either way
    loss = model.build_loss(...)          # LICENSES the features it will trace: template_mask all zero -> a licence leaf is added to the dict the
                                          # loss holds; a real template -> the dict passes through unlicensed, named once (aside=real_template)
    ...                                   # design steps / refold
    dead_template.describe()              # the effective configuration incl. engaged=yes|no (process-level: was the skip branch traced since configure?)
                                          # and aside=<reason>:<count>,… when any call stepped aside
    dead_template.emit_line(tag)          # the LEVER line (opt_core.report grammar) of describe(); the off line when not configured
    dead_template.gate()                  # AFTER the run (levers.finalize): LeverRefused only when installed but not configured to skip (the run would
                                          # carry the lever's name with mode stock); a step-aside or an idle lever is reported, never raised;
                                          # require_engaged is the same function

Modes (MODES): "stock" — TemplateV2Module.__call__ runs upstream's function and the entry-point wrappers pass everything through untouched;
"skip" — the entry points license the features they receive and the rebound call returns zeros of z's shape and dtype for LICENSED features
(XLA's simplifier folds `z + 0`: nothing of the module is emitted and the backward has no path through it); for features without this process's
licence the rebound call runs upstream's function (the stock template path), named.

The licence (what exactly is guaranteed). `license(features, entry)` is a CONCRETE check made before tracing: when `features["template_mask"]`
([B, T, N]) has no nonzero entry it returns a shallow copy of the dict carrying one extra leaf, key LICENCE_PREFIX + <this process's install nonce>,
value a zero-size float32 array of shape (B, 0, T, N) — a leading batch axis like every features leaf, so mosaic's whole-tree operations on the
features (set_binder_sequence's `features | {...}`, boltz2_forward_from_trunk's de-batching `tree.map(lambda v: v[0], ...)`) carry it along
unharmed. Anything else steps aside by name and the dict goes on WITHOUT a licence (any licence leaf it carried is dropped): a real template
(ASIDES `real_template`), a dict with no template_mask (`no_mask`), a traced mask (`traced_mask` — nothing concrete to check). At trace time the
rebound call skips ONLY when the traced feats carry this process's licence key whose trailing (T, N) equals the traced template_mask's — the
licence travels with the dict it was issued for; every other trace runs the stock template module, named once per reason: no licence
(`unlicensed_loss_path`: a loss built outside the wrapped entry points — a hand-built mosaic.losses.boltz2.Boltz2Loss / MultiSampleBoltz2Loss,
boltz2_trunk, a features file — or features an entry point passed through unlicensed), a licence issued by another process (`stale_licence`), a
template_mask of another (T, N) than licensed (`shape`), a u_proj with a bias (`u_proj_bias`: the masked output would not be identically zero).
Residual, stated exactly: the check binds to the dict's licence leaf and the mask's static shape, not to the mask's traced VALUES — code that
overwrites `template_mask` of an already-licensed dict with a same-shape real mask defeats it; license() again after editing a licensed dict (the
wrapped entry points do: a real template there drops the stale licence).
configure() clears jax's and equinox's caches because joltz and mosaic cache their filter_jit traces per process: a mode change would otherwise be
ignored inside an already-traced executable (memlevers' reason). The skip is exact algebra; whether a run is BITWISE equal to stock is a measured
fact of the kit's tables, not a claim of this file (stock adds a computed +-0.0 to z; and a changed XLA module can be compiled with other kernel picks).
"""
import hashlib
import secrets
import sys

import numpy as np

LEVER = "E1"
NAME = "dead_template"
MODES = ("stock", "skip")
SETTING = "skip"                                                    # the default setting: configure(None) / configure("") apply it
ENV_REQUIRED = {}                                                   # no environment variable takes part in this lever
LICENCE_PREFIX = "dead_template_licence_"                           # the licence leaf's key = LICENCE_PREFIX + the install nonce of the issuing process
TAG = "mosaic-opt"                                                  # the kit's line tag for the step-aside notice
ASIDES = ("real_template", "no_mask", "traced_mask", "unlicensed_loss_path", "stale_licence", "shape", "u_proj_bias")   # the named step-asides: the stock template path runs, counted in CFG["aside"]
CFG = {"mode": "stock", "nonce": None, "licensed": [], "refused": 0, "skips_traced": 0, "stock_traced": 0, "aside": {}}
_ORIG = {}                                                          # (class, attribute name) -> the class __dict__ entry before install(); "call:<name>" -> the callable


class LeverRefused(RuntimeError):
    """The lever refused by name — kit-internal misuse only: mode skip without install(), gate() on an installed lever left in mode stock. An input
    is never refused: a featurization or call the lever cannot skip steps aside by name to the stock template path."""


def _reset_counters():
    CFG.update({"licensed": [], "refused": 0, "skips_traced": 0, "stock_traced": 0, "aside": {}})


def _aside(reason: str, text: str) -> None:
    """Count one named step-aside (CFG["aside"][reason]) and say it ONCE per reason since configure(): one stderr line in the kit's words, never
    silent, never a raise. The caller then takes the stock template path."""
    assert reason in ASIDES, reason
    n = CFG["aside"].get(reason, 0) + 1
    CFG["aside"][reason] = n
    if n == 1:
        sys.stderr.write(f"[{TAG}] {LEVER} {NAME}: {text} — stepping aside by name (aside={reason}); the template module runs as stock here\n")
        sys.stderr.flush()


def _aside_token() -> str:
    """`<reason>:<count>,…` in first-seen order; '' when nothing stepped aside."""
    return ",".join(f"{r}:{n}" for r, n in CFG["aside"].items())


def _unlicensed(features) -> dict:
    """`features` itself when it carries no licence leaf, else a shallow copy without any (a dict that steps aside must not keep a licence)."""
    if not _licence_keys(features):
        return features
    return {k: v for k, v in features.items() if not (isinstance(k, str) and k.startswith(LICENCE_PREFIX))}


def _file_sha256():
    try:
        with open(__file__, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return "unavailable"


FILE_SHA256 = _file_sha256()


def _is_tracer(x) -> bool:
    try:
        import jax
        return isinstance(x, jax.core.Tracer)
    except Exception:  # noqa: BLE001 — no jax: nothing is a tracer
        return False


def _licence_keys(features):
    return [k for k in features if isinstance(k, str) and k.startswith(LICENCE_PREFIX)]


def attest(features, entry="attest") -> dict:
    """The concrete facts of `features["template_mask"]` ([B, T, N]): {entry, template_mask_shape, templates, n_tokens, template_mask_nonzero};
    {} when the dict has no template_mask or the mask is traced (nothing concrete to read). Refuses nothing — license() is the gate."""
    if not isinstance(features, dict) or "template_mask" not in features or _is_tracer(features["template_mask"]):
        return {}
    a = np.asarray(features["template_mask"])
    return {"entry": str(entry), "template_mask_shape": [int(d) for d in a.shape], "templates": int(a.shape[1]) if a.ndim >= 3 else None,
            "n_tokens": int(a.shape[-1]) if a.ndim else None, "template_mask_nonzero": int(np.count_nonzero(a))}


def license(features, entry="license") -> dict:
    """Mode skip's check, before tracing: a shallow copy of `features` carrying this process's licence leaf when template_mask is all zero.
    Anything else steps aside by name (counted, said once) and goes on unlicensed — the dict itself, or a copy without the licence leaf it
    carried: a real template (`real_template`), no template_mask (`no_mask`), a traced mask (`traced_mask`). Under mode stock: `features`
    itself, untouched. LeverRefused only for mode skip without install() (kit-internal misuse)."""
    if CFG["mode"] != "skip":
        return features
    if not _ORIG.get("installed") or CFG["nonce"] is None:
        CFG["refused"] += 1
        raise LeverRefused("dead_template_not_installed: mode skip without install() — call dead_template.install() first")
    if not isinstance(features, dict) or "template_mask" not in features:
        _aside("no_mask", f"the features given to {entry} carry no 'template_mask' (nothing to license)")
        return _unlicensed(features) if isinstance(features, dict) else features
    if _is_tracer(features["template_mask"]):
        _aside("traced_mask", f"license() at {entry} received a traced template_mask (the licence is a concrete check made before tracing: nothing concrete to read)")
        return _unlicensed(features)
    facts = attest(features, entry)
    if facts["template_mask_nonzero"]:
        _aside("real_template", f"template_mask has {facts['template_mask_nonzero']} nonzero entries over {facts['templates']} template(s) "
                                f"(shape {facts['template_mask_shape']}) at {entry}: a real template, never skipped")
        return _unlicensed(features)
    out = {k: v for k, v in features.items() if not (isinstance(k, str) and k.startswith(LICENCE_PREFIX))}   # one licence leaf: this process's
    shape = facts["template_mask_shape"]                                                                       # [B, T, N]
    out[LICENCE_PREFIX + CFG["nonce"]] = np.zeros((shape[0], 0, *shape[1:]), np.float32)                     # [B, 0, T, N]: batch-leading like every leaf, zero-size
    CFG["licensed"].append(facts)
    return out


def _skip_or_aside(self, feats):
    """None when this trace may skip (the feats carry this process's licence for the traced template_mask's (T, N) and u_proj has no bias);
    else (reason, text) — the named step-aside, the stock template path runs."""
    keys = _licence_keys(feats)
    if not keys:
        return ("unlicensed_loss_path", "tracing the template module on features that carry no licence of this process (a loss built outside "
                "Boltz2.build_loss / build_multisample_loss / model_output, or features those entry points passed through unlicensed)")
    mine = LICENCE_PREFIX + str(CFG["nonce"])
    if keys != [mine]:
        return ("stale_licence", f"the traced features carry licence key(s) {keys} but this process issues {mine!r} (a licence read back from a file or issued by another install)")
    mask = feats.get("template_mask") if hasattr(feats, "get") else None
    if mask is None:
        return ("shape", "the traced features carry a licence but no 'template_mask'")
    lic = feats[mine]; lic_shape = tuple(int(d) for d in lic.shape[-2:]); mask_shape = tuple(int(d) for d in mask.shape[-2:])
    if lic.ndim != mask.ndim + 1 or 0 not in lic.shape or lic_shape != mask_shape:
        return ("shape", f"the traced template_mask has shape {tuple(mask.shape)} ((T, N) = {mask_shape}), its licence leaf {tuple(lic.shape)} was issued for (T, N) = {lic_shape}")
    if getattr(self.u_proj, "bias", None) is not None:
        return ("u_proj_bias", "this TemplateV2Module's u_proj carries a bias (its masked output is not identically zero)")
    return None


def _template_call(self, z, feats, pair_mask, *, key, deterministic):
    if CFG["mode"] != "skip":
        CFG["stock_traced"] += 1
        return _ORIG["call:template"](self, z, feats, pair_mask, key=key, deterministic=deterministic)
    verdict = _skip_or_aside(self, feats)
    if verdict is not None:                                         # a call this lever cannot skip: upstream's function, named — never a raise
        _aside(*verdict)
        CFG["stock_traced"] += 1
        return _ORIG["call:template"](self, z, feats, pair_mask, key=key, deterministic=deterministic)
    import jax.numpy as jnp
    CFG["skips_traced"] += 1
    return jnp.zeros(z.shape, z.dtype)


def _wrap_entry(name):
    def wrapper(self, *args, **kwargs):
        if CFG["mode"] == "skip" and "features" in kwargs:
            kwargs = dict(kwargs); kwargs["features"] = license(kwargs["features"], entry=name)
        return _ORIG[f"call:{name}"](self, *args, **kwargs)
    wrapper.__name__ = name
    wrapper.__qualname__ = f"Boltz2.{name}"
    wrapper.__doc__ = f"mosaic Boltz2.{name}; under dead_template mode skip its `features` are licensed first (mosaic.fast.dead_template.license)."
    return wrapper


def install():
    """Rebind joltz.TemplateV2Module.__call__ and wrap mosaic's Boltz2 entry points (idempotent); draws this process's licence nonce. Mode stays what
    configure() last set (stock at import)."""
    if _ORIG.get("installed"):
        return describe()
    import joltz
    from mosaic.models import boltz2 as mb
    targets = [(joltz.TemplateV2Module, "__call__", "template", _template_call)]
    targets += [(mb.Boltz2, n, n, _wrap_entry(n)) for n in ("build_loss", "build_multisample_loss", "model_output")]
    for cls, attr, key, new in targets:
        _ORIG[(cls, attr)] = cls.__dict__[attr]                     # the raw class entry, restored byte-for-byte by uninstall()
        _ORIG[f"call:{key}"] = getattr(cls, attr)                   # the callable as attribute access resolves it (equinox unwraps its method wrapper)
    for cls, attr, key, new in targets:
        setattr(cls, attr, new)
    _ORIG["installed"] = True
    CFG["nonce"] = secrets.token_hex(8)
    return describe()


def uninstall():
    """Restore every class entry install() replaced; back to mode stock, counters reset, nonce dropped (licences issued so far go stale)."""
    if _ORIG.get("installed"):
        for k, raw in list(_ORIG.items()):
            if isinstance(k, tuple):
                setattr(k[0], k[1], raw)
    _ORIG.clear()
    CFG.update({"mode": "stock", "nonce": None}); _reset_counters()
    _clear_caches()
    return describe()


def _clear_caches():
    try:
        import jax
        jax.clear_caches()
    except Exception:  # noqa: BLE001 — no jax / nothing traced: nothing to clear
        pass
    try:
        import equinox as eqx
        eqx.clear_caches()
    except Exception:  # noqa: BLE001
        pass


def configure(spec=None) -> dict:
    """spec: None | '' -> SETTING ('skip', the default setting); 'skip' -> mode skip; 'stock' -> mode stock (upstream's path, the wrappers inert).
    Mode skip needs install() first (LeverRefused otherwise). Clears the jax / equinox caches and the counters (licences issued before stay valid:
    same nonce). Returns describe()."""
    word = SETTING if spec in (None, "") else str(spec).strip().lower()
    if word not in MODES:
        raise ValueError(f"dead_template: unknown spec {spec!r}; one of {'|'.join(MODES)}")
    if word == "skip" and not _ORIG.get("installed"):
        raise LeverRefused(f"dead_template_not_installed: configure({spec!r}) = mode skip before install() would change nothing — call dead_template.install() first")
    _clear_caches()
    CFG["mode"] = word; _reset_counters()
    return describe()


def installed() -> bool:
    return bool(_ORIG.get("installed"))


def gate():
    """Run by mosaic_opt.levers.finalize() AFTER the phases. LeverRefused only when the lever is installed but not configured to skip
    (dead_template_not_configured: a run under the lever's name with mode stock — kit-internal misuse). Everything a run can meet is reported, not
    raised: a featurization or call that stepped aside is on the census line (describe()/emit_line: `engaged=no|yes aside=<reason>:<count>,…`), and
    mode skip that traced nothing since configure() reads `engaged=no` (no template-module trace at all: nothing for the lever to do). Nothing to check
    when not installed."""
    if not installed():
        return describe()
    if CFG["mode"] != "skip":
        raise LeverRefused("dead_template_not_configured: E1 is installed but configure() left mode stock — the run would carry the lever's name with the template module computed as stock")
    return describe()


require_engaged = gate                                              # the post-run engagement check IS the gate (one home)


def describe() -> dict:
    """The effective configuration, every value a single token: {lever, lever_name, impl, file_sha256, origin, installed, mode, setting_of_record,
    contribution, engaged, skips_traced, stock_traced, licensed, refused, last_licence} and, only when a call stepped aside since configure(),
    aside=<reason>:<count>,… (reasons: ASIDES). `engaged` is process-level since configure() (the skip branch was traced at least once), never a
    per-phase statement; `stock_traced` counts upstream's body traced (mode stock, or a step-aside under mode skip); `refused` counts LeverRefused
    raised by license() (mode skip without install() — 0 in a configured run)."""
    lic = CFG["licensed"][-1] if CFG["licensed"] else None
    last = "none" if lic is None else f"nonzero:{lic['template_mask_nonzero']},templates:{lic['templates']},n_tokens:{lic['n_tokens']},entry:{lic['entry']}"
    skip = CFG["mode"] == "skip" and installed()
    out = {"lever": LEVER, "lever_name": NAME, "impl": __name__, "file_sha256": FILE_SHA256[:16], "origin": "kit", "installed": int(installed()), "mode": CFG["mode"],
           "setting_of_record": SETTING, "contribution": "skipped" if skip else "stock", "engaged": "yes" if (skip and CFG["skips_traced"] >= 1) else "no",
           "skips_traced": int(CFG["skips_traced"]), "stock_traced": int(CFG["stock_traced"]), "licensed": len(CFG["licensed"]), "refused": int(CFG["refused"]),
           "last_licence": last}
    aside = _aside_token()
    if aside:                                                       # only when something stepped aside: a run on licensable features prints today's keys exactly
        out["aside"] = aside
    return out


def emit_line(tag: str = "") -> str:
    """The LEVER line of this process (opt_core.report's grammar, printed on stderr via emit): state=on with describe()'s facts when installed and
    configured to skip (aside=<reason>:<count>,… among them when a call stepped aside), else the off line with its reason."""
    from opt_core.report import emit, lever_line
    d = describe()
    if not installed() or d["mode"] != "skip":
        return emit(lever_line(tag, f"{LEVER}.{NAME}", "off", reason=("not_installed" if not installed() else "not_configured"), impl=d["impl"], origin=d["origin"]))
    facts = {k: v for k, v in d.items() if k not in ("impl", "origin")}
    return emit(lever_line(tag, f"{LEVER}.{NAME}", "on", impl=d["impl"], origin=d["origin"], **{k: str(v) for k, v in facts.items()}))
