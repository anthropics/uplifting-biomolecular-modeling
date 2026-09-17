"""DATTN — the model's two dense attention-with-pair-bias sites (``diffusion_transformer.self_attention``) served by the shared core's
JAX-family provider ``opt_core.kernels.pallas`` (``serve.attention``, the pair-bias attention core: softmax_k(scale q.k + bias[h,q,k] shared
over the batch rows + key mask) v) by the mode's tier word — ``fast`` under --mode fast, ``big`` under --mode big — instead of the stock
einsum -> softmax -> einsum with [heads, N, N] f32 logits. The provider resolves the word per call from its cell table (``PALLAS_CELLS.json``:
jax version x compute capability x dtype x family x token bucket) and serves that cell's row; this module names no row, no card and no size
rule of its own. Strategy id ``F5.flash_attn_dense``.

Sites (both call ``diffusion_transformer.self_attention``, which every caller resolves at call time, so one rebinding serves all):
  * ``diffusion`` — the diffusion transformer's 24 blocks per denoising step (``Transformer``; the FlashPairformer add-on's
    ``HoistTransformer`` calls the same function), per sample under the sampler's vmap: heads 16, head_dim 48 (768 channels), the block's
    pair logits [16, N, N] as the shared bias, the token mask as the key mask -> the provider's attention kind ``af3_dit`` (KINDS; the
    samples are the family's depth);
  * ``pairformer_single`` — the pairformer's single attention with pair bias (modules.py ``PairFormerIteration``: the 48-block trunk per
    recycle and the confidence head's pairformer per sample): heads 16, head_dim 24 (384 channels), same bias / mask form -> kind
    ``af3_single``.
A stack / dtype / shape outside every cell of the table is the provider's to name (its tier words resolve the stock statement ``xla``
there, recorded on its coverage census), never a guess here.

What changes: only the attention core — the adaptive layer norm, the q/k/v/gate projections (names and shapes, hence the parameters read),
the gating and the adaptive-zero output are the stock lines. The core receives q, k, v in the activation dtype, the softmax scale
key_dim**-0.5, the pair logits as the additive bias and the token mask as booleans over keys. Numerics: NOT bitwise with stock (stock forms
the logits from f32 q.k; a flash row multiplies bf16 operands with f32 accumulation and re-associates the softmax blockwise) — a fast-class
(tolerance) lever; ``--mode exact`` does not name it. Masking is the stock's in effect: stock adds ``1e9 * (mask - 1)`` per KEY to every
query row, this passes the same per-key mask as booleans; the two differ only when every key is masked (stock = mean of v, a flash row = 0),
which needs an input with no live token.

A call the provider refuses by name (``opt_core.kernels.pallas.Refusal``: an unknown word, a dtype it does not serve, ...) or a call outside
the face's form (a pair bias per batch row) STEPS ASIDE BY NAME: the stock core runs for that call and the reason is counted
(``report()["aside"]``, the census line's ``aside=``) — never silently, never a refusal of the run.

Switch: ``AF3_JAX_DATTN`` in the model process's environment, set by the mode table only (modes.TREE_LEVER_ENV / modes.lever_env; a
caller's own copy is stripped): ``1`` = ``fast``; ``big`` under --mode big; any other non-empty value is handed to the provider verbatim
(a tier word, a row ``tokamax``, an arm ``tokamax@triton``; an unknown word is the provider's refusal by name and every call steps aside).
``MODEL_OPT_LEVERS_OFF=DATTN`` removes the lever (nothing is rebound); ``MODEL_OPT_LEVERS_OFF=pallas[:<row>]`` is the provider's own row
switch.
Prints: the launcher's SERVED line carries ``dattn=<traced>|off dattn_sites=<site:n,...|none>`` from ``report()`` (installed / word /
traced / sites / rows / aside / uncovered); this module prints its own census line at exit, ``[af3-jax-opt] DATTN word=<word> served=<n>
rows=<arm:n,...|none> aside=<reason:n,...|none> uncovered=<family:n,...|none>`` (``census_line`` / ``scan``; ``census_impls`` maps the served
rows to the tokamax ``implementation=`` words the KERNELS census counts).
"""
import os
import re

ENV_SWITCH = "AF3_JAX_DATTN"
REBINDS = ("alphafold3.model.network.diffusion_transformer:self_attention",)   # what install() rebinds (module:attribute) — the DiT's and the pairformer's dense attention function (both call diffusion_transformer.self_attention); the launcher's INSTALL_ORDER table (fpf_launch.py) and tests/test_install_order.py read it
DEFAULT_WORD = "fast"                                                     # the provider's TIER word the switch value "1" names (the fast line); the memory line's program sets "big" (modes.lever_env)
BIG_WORD = "big"
TIER_WORDS = ("fast", "exact", "big")                                    # opt_core.kernels.pallas TIER_WORDS (repeated so the wrapper reads this module without the model stack)
KINDS = {"diffusion": "af3_dit", "pairformer_single": "af3_single"}       # site -> the provider's attention kind word (families af3_dit_s5_h16_d48 / af3_single_h16_d24 with the call's heads / head_dim / depth)
FACE = "attention"                                                        # the provider's serve face (serve.COUNTS keys 'served:attention:<arm>')
LINE_TAG = "DATTN"                                                        # this module's census line at exit: [af3-jax-opt] DATTN word= served= rows= aside= uncovered=
PREFIX = "[af3-jax-opt]"
LINE_RX = re.compile(r"\[af3-jax-opt\] DATTN word=(?P<word>\S+) served=(?P<served>\w+) rows=(?P<rows>\S+) aside=(?P<aside>\S+) uncovered=(?P<uncovered>\S+)")
_STATE = {"installed": False, "traced": 0, "sites": {}, "rows": {}, "aside": {}, "uncovered": {}, "word": None, "stock": None}


def wanted(environ=os.environ) -> bool:
    return (environ.get(ENV_SWITCH, "") or "").strip() not in ("", "0")


def word(environ=os.environ) -> str:
    """The provider word this process names: switch value ``1`` = DEFAULT_WORD (``fast``); any other non-empty value verbatim (``big`` on the
    memory line; a row / arm / tier word otherwise — an unknown word is the provider's refusal by name: every call steps aside, served=0)."""
    v = (environ.get(ENV_SWITCH, "") or "").strip()
    return DEFAULT_WORD if v in ("", "1") else v


# ----------------------------------------------------------------------------- the census line (pure text: the wrapper and the kernels gate read it without the model stack)
def _fmt(d: dict) -> str:
    return ",".join(f"{k}:{v}" for k, v in sorted(d.items())) or "none"


def _parse(text: str) -> dict:
    out = {}
    for tok in (text or "").split(","):
        if ":" in tok and tok != "none":
            k, n = tok.rsplit(":", 1)
            try:
                out[k] = out.get(k, 0) + int(n)
            except ValueError:
                out[k] = out.get(k, 0)
    return out


def census_line(rep: dict = None) -> str:
    """``[af3-jax-opt] DATTN word=<word> served=<n> rows=<arm:n,...|none> aside=<reason:n,...|none> uncovered=<family:n,...|none>`` from a report()."""
    r = report() if rep is None else rep
    return (f"{PREFIX} {LINE_TAG} word={r.get('word') or 'none'} served={int(r.get('traced') or 0)} rows={_fmt(r.get('rows') or {})} "
            f"aside={_fmt(r.get('aside') or {})} uncovered={_fmt(r.get('uncovered') or {})}")


def scan(lines) -> dict:
    """The LAST census line among ``lines`` parsed -> {word, served, rows {arm: n}, aside {reason: n}, uncovered {family: n}, line} or None."""
    last = None
    for ln in lines or ():
        m = LINE_RX.search(ln)
        if m:
            last = m
    if last is None:
        return None
    served = last.group("served")
    return {"word": last.group("word"), "served": int(served) if served.isdigit() else 0, "rows": _parse(last.group("rows")),
            "aside": _parse(last.group("aside")), "uncovered": _parse(last.group("uncovered")), "line": last.group(0)}


def census_impls(rows) -> list:
    """The tokamax ``implementation=`` words the served arms call under — what the KERNELS census counts (kernels_probe wraps
    tokamax.dot_product_attention): arm ``tokamax@<impl>`` -> ``<impl>``; the bare row ``tokamax`` -> ``None`` (the library's default chain);
    rows that are not tokamax (``cudnn`` = jax.nn.dot_product_attention, ``xla`` / ``xla_sdpa`` = XLA statements, the Pallas rows) make no
    tokamax call and contribute nothing. Sorted, unique."""
    out = set()
    for arm in rows or ():
        base = str(arm).split(":", 1)[0]
        row, _, setting = base.partition("@")
        if row == "tokamax":
            out.add(setting or "None")
    return sorted(out)


# ----------------------------------------------------------------------------- the binding
def _word(text: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "._-@") else "_" for ch in str(text))[:60].strip("_") or "unknown"


def _aside(reason: str) -> None:
    key = _word(reason)
    _STATE["aside"][key] = _STATE["aside"].get(key, 0) + 1


def refusal_word(exc) -> str:
    """One short word for a provider refusal: ``<row>:<kind>`` (opt_core.kernels.pallas.Refusal) or the exception's class."""
    row, kind = getattr(exc, "row", None), getattr(exc, "kind", None)
    if kind:
        return f"{row or 'none'}:{kind}"
    return type(exc).__name__


def attention_core(q, k, v, pair_logits, mask, key_dim, *, site: str):
    """q / k / v [B..., N, H, D] in the activation dtype, pair_logits [H, N, N] (shared over the batch rows) or None, mask [B..., N] (1 = a live
    token), key_dim = the per-head width whose **-0.5 scales the logits. Returns softmax_k(scale q.k + pair_logits + key mask) v as
    [B..., N, H, Dv] in q's dtype from the provider's row for the process's word, or None when the call steps aside (counted by name)."""
    import math
    from opt_core.kernels import pallas as P
    from opt_core.kernels.pallas import Refusal
    from opt_core.kernels.pallas import serve as PS
    lead = tuple(int(s) for s in q.shape[:-3])
    N, H, D = (int(s) for s in q.shape[-3:])
    Dv = int(v.shape[-1])
    B = int(math.prod(lead)) if lead else 1
    bias = pair_logits
    if bias is not None and bias.ndim != 3:
        if bias.ndim > 3 and all(int(s) == 1 for s in bias.shape[:-3]):
            bias = bias.reshape(bias.shape[-3:])
        else:                                                             # a pair bias per batch row is outside the face's shared-bias form: the stock core, by name
            _aside(f"bias_rank:{bias.ndim}"); return None
    q4, k4, v4 = q.reshape((B, N, H, D)), k.reshape((B, N, H, D)), v.reshape((B, N, H, Dv)).astype(q.dtype)
    key_mask = (mask.reshape((B, N)) > 0.5)
    w = _STATE["word"] or word()
    kind = KINDS[site]
    selection = None
    if w in TIER_WORDS:                                                   # a tier word: which cell serves this call — a family with no measured cell is the provider's stock statement by ITS rule; counted here by name (uncovered=), never re-pointed
        try:
            fam = P.family("attn", kind=kind, heads=H, head_dim=D, n_seq=(B if kind in PS.kinds_with_depth() else None))
            selection = PS.resolve("attn", fam, q4.dtype, N, word=w, direction="fwd", key_masked=True)
        except Refusal as e:
            _aside(refusal_word(e)); return None
        if getattr(selection, "cell_key", None) is None:
            _STATE["uncovered"][fam] = _STATE["uncovered"].get(fam, 0) + 1
    before = {k_: n for k_, n in PS.COUNTS.items() if k_.startswith(f"served:{FACE}:")}
    try:                                                                  # the provider walks the word's candidates; a row that cannot engage steps aside by name inside it (tier words) or raises (a row word)
        out = PS.attention(q4, k4, v4, bias, key_mask, float(key_dim) ** -0.5, word=w, kind=kind, direction="fwd", layout="BSHD", selection=selection)
    except Refusal as e:                                                  # refused by name (an unknown word, a dtype / shape rule, levers_off on a row word): this call keeps the stock core
        _aside(refusal_word(e)); return None
    except NotImplementedError as e:                                      # a transform a row's kernel does not cover: by name
        _aside("trace:" + type(e).__name__); return None
    served = [k_.split(":", 2)[2] for k_, n in PS.COUNTS.items() if k_.startswith(f"served:{FACE}:") and n > before.get(k_, 0)]
    arm = served[0] if served else w
    _STATE["traced"] += 1
    _STATE["rows"][arm] = _STATE["rows"].get(arm, 0) + 1
    _STATE["sites"][site] = _STATE["sites"].get(site, 0) + 1
    return out.reshape(lead + (N, H, Dv)).astype(q.dtype)


def stock_core(q, k, v, pair_logits, mask, key_dim, dtype):
    """The stock statement (diffusion_transformer.py self_attention: f32 q.k logits + 1e9 (mask - 1) per key + pair logits -> softmax -> cast -> @ v)."""
    import jax
    import jax.numpy as jnp
    bias = (1e9 * (mask - 1.0))[..., None, None, :]
    q = q.astype(jnp.float32)
    k = k.astype(jnp.float32)
    bias = bias.astype(jnp.float32)
    logits = jnp.einsum("...qhc,...khc->...hqk", q * key_dim ** (-0.5), k) + bias
    if pair_logits is not None:
        logits += pair_logits
    weights = jax.nn.softmax(logits, axis=-1)
    weights = jnp.asarray(weights, dtype=dtype)
    return jnp.einsum("...hqk,...khc->...qhc", weights, v)


def _make_self_attention(DT):
    """The stock ``self_attention`` with its einsum/softmax core served by the provider; every other line is the stock's."""
    import jax
    import jax.numpy as jnp
    hm = DT.hm

    def self_attention(x, mask, pair_logits, config, global_config, single_cond=None, name=""):
        assert len(mask.shape) == len(x.shape) - 1, f"{mask.shape}, {x.shape}"
        site = "pairformer_single" if str(name).startswith("single_attention") else "diffusion"

        x = DT.adaptive_layernorm(x, single_cond, name=name)

        num_channels = x.shape[-1]
        key_dim = config.key_dim if config.key_dim is not None else num_channels
        value_dim = config.value_dim if config.value_dim is not None else num_channels
        num_head = config.num_head
        assert key_dim % num_head == 0, f"{key_dim=} % {num_head=} != 0"
        assert value_dim % num_head == 0, f"{value_dim=} % {num_head=} != 0"
        key_dim = key_dim // num_head
        value_dim = value_dim // num_head

        qk_shape = (num_head, key_dim)
        q = hm.Linear(qk_shape, use_bias=True, name=f"{name}q_projection")(x)
        k = hm.Linear(qk_shape, use_bias=False, name=f"{name}k_projection")(x)
        v_shape = (num_head, value_dim)
        v = hm.Linear(v_shape, use_bias=False, name=f"{name}v_projection")(x)

        weighted_avg = attention_core(q, k, v, pair_logits, mask, key_dim, site=site)
        if weighted_avg is None:                                          # stepped aside by name: the stock core for this call
            weighted_avg = stock_core(q, k, v, pair_logits, mask, key_dim, x.dtype)
        weighted_avg = jnp.asarray(weighted_avg, dtype=x.dtype)
        weighted_avg = jnp.reshape(weighted_avg, weighted_avg.shape[:-2] + (-1,))

        gate_logits = hm.Linear(num_head * value_dim, bias_init=1.0, initializer="zeros", name=f"{name}gating_query")(x)
        weighted_avg *= jax.nn.sigmoid(gate_logits)

        return DT.adaptive_zero_init(weighted_avg, num_channels, single_cond, global_config, name)

    return self_attention


def _print_census() -> None:
    import sys
    try:
        sys.stdout.write(census_line() + "\n"); sys.stdout.flush()
    except Exception:                                                     # noqa: BLE001 — the exit hook never raises into the interpreter's shutdown
        pass


def install() -> bool:
    """Rebind ``diffusion_transformer.self_attention`` (idempotent) and register the census line at exit. Returns whether the lever is installed."""
    if _STATE["installed"]:
        return True
    import atexit
    from alphafold3.model.network import diffusion_transformer as DT
    _STATE["word"] = word()
    _STATE["stock"] = DT.self_attention
    DT.self_attention = _make_self_attention(DT)
    _STATE["installed"] = True
    atexit.register(_print_census)
    return True


def uninstall() -> None:
    if _STATE["installed"]:
        from alphafold3.model.network import diffusion_transformer as DT
        DT.self_attention = _STATE["stock"]
        _STATE["installed"] = False


def report() -> dict:
    rows = dict(_STATE["rows"])
    return {"installed": _STATE["installed"], "word": _STATE["word"] or word(), "traced": _STATE["traced"], "sites": dict(_STATE["sites"]), "rows": rows,
            "aside": dict(_STATE["aside"]), "uncovered": dict(_STATE["uncovered"]), "implementation": ",".join(census_impls(rows)) or "none"}
