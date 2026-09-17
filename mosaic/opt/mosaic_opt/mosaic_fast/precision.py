"""
P6 `precision` (opt-in): a per-REGION matmul-precision policy for joltz (Boltz-2 in JAX) as mosaic's design step runs it. Numerics class
`fast` by construction: a region set to any word rounds its dot products differently from stock, so a run under it is never bitwise with a stock run;
it changes no RNG draw, no dtype of a parameter or activation (float32 throughout), no softmax / LayerNorm arithmetic, no step count.

Regions, their entry points, and what STOCK runs there:
  trunk       Joltz2.embed_inputs, Joltz2.trunk_iteration (input embedder, template module, MSA module, the 64-block Pairformer2),
              DistogramModule2.__call__, DiffusionConditioning2.__call__      stock: no precision set = `jax_default_matmul_precision`
                                                                              unset = Precision.DEFAULT (one TF32 tensor-core pass per f32 dot on sm80+)
  diffusion   AtomDiffusion2.sample — the sampler: 25 steps x (atom encoder, 24-layer token transformer, atom decoder)
                                                                              stock: HIGHEST — IEEE fp32, no tensor cores: mosaic's loss calls it
                                                                              inside `jax.default_matmul_precision("float32")` (losses/boltz2.py)
  confidence  ConfidenceModule2.__call__                                      stock: as trunk

Words are the shared core's matmul vocabulary, `opt_core.precision.policy.MATMUL_PRECISIONS`: "highest" (IEEE fp32) | "high" (TF32) |
"medium" (bf16 operands, f32 accumulate); a region absent from the spec is not touched (stock). `JAX_WORDS` spells each word for
`jax.default_matmul_precision`.

Mechanism: install() wraps every entry point above so its body runs inside `jax.default_matmul_precision(<jax word>)`. The innermost precision
context is the one a dot sees when it is traced, so `diffusion=high` lifts mosaic's fp32 island to TF32 without editing an upstream file, and a
region's word reaches every dot traced inside it (nested filter_jit, checkpoint and scan bodies included). configure() clears the jax and
equinox staging caches: mosaic's jitted loss step (`optimizers._eval_loss_and_grad`) and joltz's inner filter_jits are cached per process and
would otherwise keep the precision they were first traced under (the same reason memlevers clears them).

  precision.install()                          # once per process, before the loss is traced (a second install() raises)
  precision.configure(None)                    # the default setting, SETTING = "diffusion=high"; or any spec: "trunk=medium,confidence=medium",
                                               # "stock" (every region untouched); returns describe()
  ... build the loss, run steps ...            # precision.describe() -> the flat record; precision.uninstall() restores joltz
ENV_REQUIRED is empty: the policy needs nothing from the environment.
"""
import contextlib

REGIONS = ("trunk", "diffusion", "confidence")
STOCK = "stock"                                                                     # the spec word for "no region touched"
SETTING = "diffusion=high"                                                          # the default setting: configure(None) applies it
ENV_REQUIRED = {}                                                                   # nothing of this lever lives in the environment
STOCK_WORDS = {"trunk": "default", "diffusion": "highest", "confidence": "default"}  # what stock runs per region (module docstring), reported by describe()
JAX_WORDS = {"highest": "highest", "high": "tensorfloat32", "medium": "BF16_BF16_F32"}
# JAX_WORDS is the JAX spelling of opt_core.precision.policy.MATMUL_PRECISIONS: the word `jax.default_matmul_precision` takes for each of the
# core's (torch-spelled) words.
TARGETS = {                                                                          # region -> (joltz class name, attribute) entry points
    "trunk": (("Joltz2", "embed_inputs"), ("Joltz2", "trunk_iteration"), ("DistogramModule2", "__call__"), ("DiffusionConditioning2", "__call__")),
    "diffusion": (("AtomDiffusion2", "sample"),),
    "confidence": (("ConfidenceModule2", "__call__"),),
}

PREC = {r: None for r in REGIONS}     # region -> core word | None (stock)
_ORIG = {}                            # (class name, attribute) -> the attribute as joltz defined it (restored by uninstall)


class PrecisionSpecError(ValueError):
    """A spec naming a region or a word this module does not have (the message names both vocabularies)."""


def words():
    """The core's matmul words (imported: the vocabulary has one home)."""
    from opt_core.precision.policy import MATMUL_PRECISIONS
    assert set(JAX_WORDS) == set(MATMUL_PRECISIONS), (sorted(JAX_WORDS), MATMUL_PRECISIONS)
    return tuple(MATMUL_PRECISIONS)


def parse(spec):
    """None (= SETTING) | `stock` | `REGION=WORD[,REGION=WORD...]` -> {region: word | None} over every region (absent = None). Raises PrecisionSpecError."""
    out = {r: None for r in REGIONS}
    text = (SETTING if spec is None else str(spec)).strip()
    if not text:
        raise PrecisionSpecError(f"precision spec {spec!r} is empty: want None (the default setting {SETTING!r}), {STOCK!r}, or REGION=WORD[,REGION=WORD...]")
    if text == STOCK:
        return out
    vocab = words()
    for item in text.split(","):
        region, sep, word = item.strip().partition("=")
        region, word = region.strip(), word.strip()
        if not sep or region not in REGIONS or word not in vocab:
            raise PrecisionSpecError(f"precision spec item {item!r}: want REGION=WORD with REGION in {REGIONS} and WORD in {vocab} (or the spec {STOCK!r})")
        out[region] = word
    return out


def canonical(prec=None):
    """The spec string of a policy, regions in REGIONS order (`stock` when none is set) — one spelling per policy."""
    prec = PREC if prec is None else prec
    items = [f"{r}={prec[r]}" for r in REGIONS if prec.get(r)]
    return ",".join(items) if items else STOCK


def _context(region):
    word = PREC[region]
    if word is None:
        return contextlib.nullcontext()
    import jax
    return jax.default_matmul_precision(JAX_WORDS[word])


def _wrap(region, raw):
    def call(self, *args, **kwargs):
        bound = raw.__get__(self, type(self))          # a plain function and an equinox filter_jit wrapper both bind this way
        with _context(region):
            return bound(*args, **kwargs)
    call.__precision_region__ = region
    call.__wrapped__ = raw
    return call


def _clear_caches():
    import jax
    import equinox as eqx
    jax.clear_caches()
    eqx.clear_caches()


def install():
    """Wrap the region entry points, once per process (a second install() raises: one installer per process). Nothing changes numerically until
    configure() sets a word."""
    if _ORIG:
        raise RuntimeError("precision.install() called twice in one process (uninstall() first)")
    import joltz
    for region, targets in TARGETS.items():
        for cls_name, attr in targets:
            key = (cls_name, attr)
            cls = getattr(joltz, cls_name)
            raw = cls.__dict__[attr] if attr in cls.__dict__ else getattr(cls, attr)
            _ORIG[key] = raw
            setattr(cls, attr, _wrap(region, raw))
    return sorted(_ORIG)


def installed():
    return bool(_ORIG)


def uninstall():
    """Restore joltz's own attributes and clear the staging caches."""
    import joltz
    for (cls_name, attr), raw in list(_ORIG.items()):
        setattr(getattr(joltz, cls_name), attr, raw)
        del _ORIG[(cls_name, attr)]
    _clear_caches()


def configure(spec):
    """Set the policy from a spec (see parse), clear the staging caches, return describe(). Refuses (RuntimeError) when install() has not run
    and the spec touches a region — a word that cannot take effect is never recorded as set."""
    prec = parse(spec)
    if any(prec.values()) and not installed():
        raise RuntimeError("precision.configure(%r) before precision.install(): the region entry points are not wrapped" % (spec,))
    import jax
    for word in filter(None, prec.values()):          # the running jax must accept the spelling (a config it refuses raises here, at configure, not at trace)
        with jax.default_matmul_precision(JAX_WORDS[word]):
            pass
    PREC.update(prec)
    _clear_caches()
    return describe()


def describe():
    """The FLAT record of the policy in force: spec (canonical), numerics_class (stock|fast), installed, and per region `<region>` = the core word
    or None (stock), `jax_<region>` = its JAX spelling, `stock_<region>` = what stock runs there."""
    rec = {"spec": canonical(), "setting_of_record": SETTING, "numerics_class": "stock" if canonical() == STOCK else "fast", "installed": installed()}
    for r in REGIONS:
        rec[r] = PREC[r]
        rec["jax_" + r] = JAX_WORDS[PREC[r]] if PREC[r] else None
        rec["stock_" + r] = STOCK_WORDS[r]
    return rec


def line():
    """`precision trunk=<word|stock> diffusion=<...> confidence=<...> class=<stock|fast>` — one log line."""
    d = describe()
    return "precision " + " ".join(f"{r}={(d[r] + '(' + d['jax_' + r] + ')') if d[r] else 'stock'}" for r in REGIONS) + f" class={d['numerics_class']}"
