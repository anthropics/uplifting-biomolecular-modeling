"""ptx_c64_routes — protenix_v2's c = 64 pair-stack routes (levers ``templ_trimul_tmk3`` / ``templ_trimul_esm`` and ``templ_triatt_core``).

Protenix-v2's TemplateEmbedder runs its own 2-block pairformer at c = 64 (TriMul c_z = c_hidden = 64; triangle attention 2 heads x 32) once per
distinct template per recycle; every other pair stack of the model is c = 256 (trunk 48 blocks, MSA module 4, confidence head 4) and is served
by the block path and the arm's c = 256 providers, which hand c != 256 modules to the stock forward by name.  This module serves the two c = 64
ops through the shared core's providers and touches nothing else:

  templ_trimul_tmk3   PTX_TEMPL_TRIMUL=tmk3   EXACT class — ARM E (exact): the shared core's ``exact`` tier row for the call's cell (``by_word(weights_of, 'exact')``)
  templ_trimul_esm    PTX_TEMPL_TRIMUL=esm    TOLERANCE class — ARM T (fast | big): the shared core's ``fast`` | ``big`` TIER ROW for the
                                              call's cell — ``by_word(weights_of, <the kit mode's tier word>)``: ``big`` under the kit mode big (``PROTENIX_OPT``,
                                              exported by the activation before this module is imported), ``fast`` otherwise; the row is the core's cell table's
                                              (``native`` / ``esm_v5_fwd`` / ``esm_shapes`` … per (cc, precision, 64, 64, N bucket, direction) and stack) and flips
                                              with the core's cells, the LEVER line names the row SERVED (``impl=trimul:<row>@<tier>/<cell>``, ``bind=tier:<word>``).
                                              opt_core older than ``TIER_CORE`` (0.5.86: the first core whose (64, 64) tier rows carry the sealed native TriMul
                                              payload) keeps the earlier route BY NAME: the ``opt_core.kernels.trimul_esm_shapes`` package directly (``_esm_provider``,
                                              ``bind=package:trimul_esm_shapes``; a call outside its envelope re-routes BY NAME to tmk3_exact)
      This module is the FPF_OPS provider of ``trimul_out`` / ``trimul_in`` (env.sh appends its two entries AFTER the arm's own: the registry
      binds the later entry of an op).  A call on a module with c_z == c_hidden == 64 on the library branch (triangle_multiplicative ==
      'cuequivariance') goes through ONE ``opt_core.trimul.Lever`` (mode 'exact', provider ``by_word(weights_of, 'exact')``, min_tokens 101):
      the cell table's exact row for (cc, bf16, C64, H64, N) — ``tmk3_exact`` on cc 9.0 and 8.0 (opt_core.kernels.trimul TRIMUL_CELLS.json; the
      TM-K3 'exact' construction: the stock op's floating-point operations in the stock order, the residual ``z + update`` fused like the
      library branch returns it) — bitwise == the stock op.  N <= 100 (the library's own torch branch), a cell without an exact row (N > 1200,
      another precision or card), a refusal of the row by name: the arm's forward serves that call, counted (``fallback_by=<row:kind>``).
      EVERY OTHER call (c = 256) is handed unchanged to the arm's own provider — the FPF_OPS entry of the op that precedes this module's,
      the *inner* provider;
      with none before it (FPF_TRIMUL_EXACT=0 under ARM E) the stock forward the registry captured.
  templ_triatt_core   PTX_TEMPL_TRIATT=core   TOLERANCE class — ARM T (fast | big); ARM E keeps triattn_exact on these calls
      Wraps ``protenix.model.triangular.layers.cuequivariance_triangular_attn`` AS FOUND at import — above ``triatt_headsplit_exact``'s
      per-head entry (the found callable's markers are carried forward, so a later idempotence check of that lever sees its own marker) — and
      routes ONLY the template stack's calls, H == 2 heads and D == 32, at N_token >= ``PTX_T_MIN_TOKENS`` (ARM T's one size gate, 300:
      below it the fast rows keep the library op here exactly as the block core keeps the exact path) through
      ``opt_core.kernels.triattn.triangle_attention(q, k, v, bias, mask, scale, word='fast', prefer=<the word's rows | None>, stock=<the
      found callable>)`` — the provider's TIER WORD ``fast`` (``bind=tier:fast`` on the LEVER line; the provider's tier words are fast |
      exact, so a big mode binds fast: a word it does not carry is refused by name, never guessed), the provider's ``select()`` choosing
      the row per cell; a row preference, when the switch word carries one, narrows the candidates in the kit's order (``prefer=`` on the
      line; ``none`` = the cell alone decides).  Words: ``core`` = the tier word ALONE (``prefer=none``: no row preference such as
      ('triattn_native@v10', 'cuda_sm90a', 'k2b') rides with it — the provider's cells decide): on the
      core's H2 / D32 cells cc 9.0 serves calls below 512 tokens with the core's small-N rows (cuda_sm90a / k2b per cell and call form) and
      512+ with the sealed triangle-attention package (``triattn_native``), cc 8.0 serves
      ``triattn_native`` per cell (the active package's sm_80 member); ``native`` | ``sm90a`` | ``k2b`` pin that one
      row (attribution words).  A row that refuses by name (no prebuilt for the stack, an offset bound, a shape, a package version) is
      re-routed BY NAME to the refusal's fallback row, at the end to the found callable — one stderr line per (row, kind), counted on the
      LEVER line.  H = 8 calls (trunk / MSA / confidence pair stacks) and per-head calls (H = 1) pass straight to the found callable: the
      block core's provider (PTX_BLK_ATT) and the exact tier's triattn_exact are neither wrapped nor consulted.

Both words are read once, at import (the module is imported by ``fpf.enable_from_env`` in the pairformer import hook, after the trunk levers
are applied and before any model object exists).  An absent word installs nothing for that lever: ``MODEL_OPT_LEVERS_OFF=<lever>`` removes the
word after env.sh ran, so the composite trimul provider stays bound and hands every call to the inner provider / the attention site is left as
found.  ``opt_core`` older than ``MIN_CORE`` (the first core with the trimul tier words, the H2 triangle-attention cells and the sealed package's
H2 build) or an import that fails makes the lever step aside BY NAME at import — never a silent stock call, never a refusal of the run.
stderr lines (opt_core's per-lever grammar for the LEVER lines; the exit lines through ``opt_core.report.register_exit_tally``)::

    [ptx_c64_routes] APPLIED trimul=<tmk3|esm|off|aside:why> bind=<tier:exact|tier:fast|tier:big|package:trimul_esm_shapes|none> triatt=<core|off|aside:why> inner_out=<mod:attr|stock> inner_in=<..> trimul_min_tokens=101 triatt_min_tokens=<n> prefer=<rows|none> triatt_bind=<tier:fast|none> opt_core=<version>@<file> wrapped_over=<..>
    [ptx_c64_routes] FIRST trimul <direction> N=.. C=.. dtype=.. residual=.. selection=[<the table's line>]          (first served call per direction)
    [ptx_c64_routes] FIRST triatt S=.. H=2 D=32 q=.. bias=.. mask=.. row=<row> selection=[<the table's line>]          (first served call)
    [ptx_c64_routes] REROUTE triatt <row>:<kind> -> <row> (<detail>)                                                    (once per refusal kind)
    [ptx_c64_routes] LEVER name=F2.trimul state=on impl=trimul:tmk3_exact@exact/<cell> origin=core served=<n> fallback=<n> fallback_by={..} min_tokens=101 shapes=.. mode=exact errors={} gate=ok word=tmk3 bind=tier:exact esm_rerouted=none lever=templ_trimul_tmk3 c256_inner=<n>
    [ptx_c64_routes] LEVER name=F2.trimul state=on impl=trimul:<row>@<fast|big>/<cell> origin=core served=<n> … mode=fast … word=esm bind=tier:<fast|big> esm_rerouted=none lever=templ_trimul_esm c256_inner=<n>      (ARM T, opt_core >= 0.5.86)
    [ptx_c64_routes] LEVER name=F2.trimul state=on impl=trimul_esm_shapes@<version> origin=core served=<n> … mode=fast … word=esm bind=package:trimul_esm_shapes esm_rerouted=<word:n|none> lever=templ_trimul_esm c256_inner=<n>      (ARM T, opt_core < 0.5.86)
    [ptx_c64_routes] LEVER name=F1.flash_triatt state=on impl=triattn:<rows SERVED>@fast origin=core served=<n> stock_by_name=<n> rerouted=<row:kind>word:n,..|none> errors=<T:n|none> passthrough=<n> min_tokens=<n> below_min_tokens=<n> shapes=<SxHxD/dtype:n,..> word=<w> prefer=<rows|none> bind=<tier:fast|none> gate=<ok|refused> lever=templ_triatt_core
"""
from __future__ import annotations

import functools
import importlib
import os
import sys
import threading

TAG = "ptx_c64_routes"
C64 = 64                                             # TemplateEmbedder pair stack: TriMul c_z == c_hidden == 64
TRIATT_H, TRIATT_D = 2, 32                           # ... and triangle attention 2 heads x 32 (every other pair stack: 8 x 32; per-head splits: 1 x 32)
TRIMUL_ENV, TRIATT_ENV = "PTX_TEMPL_TRIMUL", "PTX_TEMPL_TRIATT"
TRIMUL_WORDS = {"tmk3": ("exact", "templ_trimul_tmk3"),   # switch word -> (opt_core.trimul Lever mode word = the lever's CLASS, registry lever): by_word(weights_of, 'exact') = the cell's exact tier row (EXACT class; ARM E)
                "esm": ("fast", "templ_trimul_esm")}      # by_word(weights_of, 'fast' | 'big') = the cell's fast | big tier row (TOLERANCE class; ARM T); below TIER_CORE the trimul_esm_shapes package route, by name
MODE_ENV = "PROTENIX_OPT"                            # the kit mode word of this process (protenix_opt.stack exports it at activation, before the pairformer import hook imports this module; the multi-GPU line's ranks carry their base mode)
ARM_T_TIERS = ("fast", "big")                      # the core tier words ARM T's TriMul binds: the kit mode's own word when it is one of these, else 'fast' (a caller's PTX_TEMPL_TRIMUL=esm under another mode: the tolerance row)
TIER_CORE = (0, 5, 86)                               # opt_core >= 0.5.86: the first core whose kernels.trimul (64, 64) tier rows carry the sealed native TriMul payload (rows native / native_exact); ARM T binds the tier word from here, below it the package route by name
ESM_KERNEL = "opt_core.kernels.trimul_esm_shapes"    # ARM T's route below TIER_CORE (the package directly), kept by name
TRIATT_TIER = "fast"                                 # opt_core.kernels.triattn tier word the switch words below bind when the process carries no tolerance mode word; under the kit mode big the
                                                     # word is big (tier_word(): the provider carries fast | big | exact; which rows a word resolves to is the provider's cell table's)
TRIATT_BIND = "tier:" + TRIATT_TIER                  # the LEVER / APPLIED lines' bind= token, like the TriMul line's
TRIATT_CORE_PREFER = None                            # word core: the tier word ALONE — the provider's select() names the row per cell (no row preference such as
#                                                      ('triattn_native@v10', 'cuda_sm90a', 'k2b') rides with it: on the core's cells cc 9.0 serves N <= 511 with its small-N rows
#                                                      cuda_sm90a / k2b and N >= 512 with triattn_native, cc 8.0 serves triattn_native per cell)
TRIATT_WORDS = {"core": TRIATT_CORE_PREFER,          # switch word -> the tier's candidates (prefer=; None = the cell decides); the one-row words pin a row in the kit's order: the sealed package's
                "native": ("triattn_native@v11",), "sm90a": ("cuda_sm90a",), "k2b": ("k2b",)}   # H2 build, the prebuilt sm_90a row, the Triton row (attribution words; a pinned row's refusal re-routes by name)
TRIATT_PREFER = TRIATT_WORDS["core"]
TRIATT_GATE_ENV = "PTX_T_MIN_TOKENS"                   # ARM T's one size gate (env.sh: below it the block path is the exact path): template calls below it keep the found entry, too
TRIMUL_MIN_TOKENS = 101                              # the library routes N <= 100 to its own torch branch: those calls keep the arm's forward (counted below_min_tokens)
MIN_CORE = (0, 5, 47)                                # oldest opt_core with kernels.trimul_esm_shapes, no per-call host sync in triattn_native, the kernels.trimul tier words + C64 cells and the triattn H2 cells
OPS = ("trimul_out", "trimul_in")
STOCK_ROWS = ("cueq", "stock", "ds4sci", "sdpa")     # triattn rows that ARE the kit's stock callable (a re-route that ends here calls the found callable directly)
_LOCK = threading.Lock()
STATE = {"applied": False, "trimul": "off", "trimul_lever": None, "trimul_bind": "none", "triatt": "off", "opt_core": None, "inner": {}, "first": {}, "trimul_passthrough": 0, "trimul_target": 0, "esm_rerouted": {},
         "triatt_served": {}, "triatt_below_gate": 0, "triatt_min_tokens": 0, "triatt_prefer": TRIATT_PREFER, "triatt_stock_by_name": 0, "triatt_passthrough": 0, "triatt_rerouted": {}, "triatt_errors": {}, "triatt_shapes": {},
         "triatt_route": None, "triatt_bind": "none", "wrapped_over": None}
_TRIMUL_LEVER = None          # opt_core.trimul.Lever (None: the TriMul lever off / aside — the composite provider hands every call to the inner provider)
_ESM_CACHE = {}               # (id(module), direction) -> the esm package's packed-weights cache for that weight set
_INNER = {}                   # op -> the arm's provider callable (None: the registry's captured stock forward)
_INNER_SPEC = {}              # op -> "module:attr" | "stock"
_PRINTED = set()


def _say(msg: str) -> None:
    print(f"[{TAG}] {msg}", file=sys.stderr, flush=True)


def _bump(d: dict, k, n: int = 1) -> int:
    with _LOCK:
        d[k] = d.get(k, 0) + n
        return d[k]


def _tok(x) -> str:
    return str(x).replace(" ", "_").replace("\n", "_")[:160] or "none"


def prefer_token(prefer) -> str:
    """``row+row+..`` for a row preference, ``none`` for the tier word alone (the lines' prefer= value)."""
    return "+".join(prefer) if prefer else "none"


def version_tuple(v) -> tuple:
    """'0.5.35.1' -> (0, 5, 35, 1); non-numeric parts stop the parse ('0.5.47rc1' -> (0, 5))."""
    out = []
    for p in str(v or "").split("."):
        if not p.isdigit():
            break
        out.append(int(p))
    return tuple(out)


def core_ok(version) -> bool:
    """opt_core at or above MIN_CORE (the levers step aside by name below it)."""
    return version_tuple(version)[:len(MIN_CORE)] >= MIN_CORE


def core_at(version, floor: tuple) -> bool:
    """opt_core ``version`` at or above ``floor`` ((0, 5, 86) …); None / unparsable -> False."""
    return version is not None and version_tuple(version)[:len(floor)] >= tuple(floor)


def tier_word(environ=None) -> str:
    """The shared core's TriMul tier word ARM T binds in this process: the kit mode's own word when it is a tolerance tier (``fast`` |
    ``big``, read from ``PROTENIX_OPT``), else ``fast`` (the word esm under another mode, or no mode word: the tolerance row)."""
    environ = os.environ if environ is None else environ
    m = str(environ.get(MODE_ENV, "") or "").strip().lower()
    return m if m in ARM_T_TIERS else "fast"


def arm_t_binding(core_version, environ=None) -> tuple:
    """(``'tier'``, <fast|big>) when ``core_version`` >= TIER_CORE — ARM T's TriMul is the core's tier row by word; else (``'package'``,
    <why>) — the ``trimul_esm_shapes`` package route of the cores before it, kept BY NAME (``why`` names the core and the floor)."""
    if core_at(core_version, TIER_CORE):
        return "tier", tier_word(environ)
    return "package", "trimul_esm_shapes(opt_core%s<%s:tier_rows_without_native)" % (core_version, ".".join(map(str, TIER_CORE)))


def parse_inner(fpf_ops: str, me: str = __name__.rsplit(".", 1)[-1]) -> dict:
    """{op: 'module:attr' | 'stock'} — for each trimul op, the LAST FPF_OPS entry before this module's own (the arm's provider this module hands
    c = 256 calls to); 'stock' when no other entry names the op (FPF_TRIMUL_EXACT=0 under ARM E: the registry's captured stock forward)."""
    inner = {op: "stock" for op in OPS}
    for item in (fpf_ops or "").split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        name, target = item.split("=", 1)
        name, target = name.strip(), target.strip()
        if name in OPS and target.split(":", 1)[0].strip() not in (me, "ptx_c64_routes"):
            inner[name] = target
    return inner


def is_template_trimul(module, triangle_multiplicative) -> bool:
    """The template pair stack's TriMul on the library branch: c_z == c_hidden == 64 and triangle_multiplicative == 'cuequivariance'."""
    return getattr(module, "c_z", None) == C64 and getattr(module, "c_hidden", None) == C64 and triangle_multiplicative == "cuequivariance"


def is_template_triatt(heads: int, head_dim: int) -> bool:
    """The template pair stack's triangle attention: 2 heads x 32 (trunk / MSA / confidence: 8 x 32; a per-head split: 1 x 32)."""
    return int(heads) == TRIATT_H and int(head_dim) == TRIATT_D


def weights_of(m):
    """The ten canonical tensors (opt_core.trimul.WEIGHT_KEYS) of a protenix TriangleMultiplication{Outgoing,Incoming} (bias-free linear layers)."""
    return dict(ln_in_w=m.layer_norm_in.weight, ln_in_b=m.layer_norm_in.bias, w_ag=m.linear_a_g.weight, w_ap=m.linear_a_p.weight,
                w_bg=m.linear_b_g.weight, w_bp=m.linear_b_p.weight, ln_out_w=m.layer_norm_out.weight, ln_out_b=m.layer_norm_out.bias,
                w_o=m.linear_z.weight, w_og=m.linear_g.weight)


def _capturing() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())
    except Exception:  # noqa: BLE001
        return False


# ------------------------------------------------------------------------------------------------------------- TriMul (FPF_OPS provider)
def _resolve_inner(op: str):
    """The inner provider callable for ``op`` (imported once; None = the registry's captured stock forward, looked up at call time)."""
    if op in _INNER:
        return _INNER[op]
    spec = _INNER_SPEC.get(op, "stock")
    fn = None
    if spec != "stock":
        import importlib
        modpath, _, attr = spec.partition(":")
        fn = getattr(importlib.import_module(modpath.strip()), (attr or "fn").strip())   # an inner provider that cannot import raises here, as it would in fpf.enable_from_env
    _INNER[op] = fn
    return fn


def _stock(op: str):
    import fpf
    return fpf.original(op)


def _trimul_args(z, mask=None, inplace_safe=False, _add_with_inplace=False, _inplace_chunk_size=256, triangle_multiplicative="torch", **rest):
    return z, mask, inplace_safe, _add_with_inplace, triangle_multiplicative, rest


def torch_cc(t):
    """(major, minor) compute capability of a CUDA tensor's device ((0, 0) off CUDA)."""
    import torch
    return tuple(torch.cuda.get_device_capability(t.device)) if getattr(t, "is_cuda", False) else (0, 0)


def _esm_provider(weights_of):
    """ARM T's TriMul provider BELOW ``TIER_CORE`` (kept by name; from opt_core 0.5.86 the word ``esm`` binds the core's tier row instead, see
    :func:`arm_t_binding`): the shared core's ``trimul_esm_shapes`` line re-tiled for the template pair widths, the package directly
    (``opt_core.kernels.trimul_esm_shapes``: LN_in + gated dual projection -> bf16 planes, cuBLAS bmm, LN_out + out projection + gate (+ the
    residual the library branch returns), launch cells by (cc, io, c_z, c_hidden, N bucket, direction); TOLERANCE class).  A call outside
    that package's envelope (its ``supported()`` word: batch, io, width, N) is re-routed BY NAME to the cell's exact row ``tmk3_exact``
    (``by_word(weights_of, 'exact')``), counted ``esm_rerouted``; a call neither takes is :class:`opt_core.trimul.Refused` -> the arm's forward."""
    import torch
    from opt_core import trimul as T
    KE = importlib.import_module(ESM_KERNEL)
    tmk3 = T.by_word(weights_of, "exact", name="trimul:exact")

    def _facts(call):
        z = call.z
        zz = z[0] if (z.dim() == 4 and int(z.shape[0]) == 1) else z
        w = weights_of(call.module)
        return zz, tuple(torch.cuda.get_device_capability(z.device)) if z.is_cuda else (0, 0), str(zz.dtype).replace("torch.", ""), int(zz.shape[-1]), int(w["w_ap"].shape[0]), int(zz.shape[-2])

    def eligible(call):
        if not getattr(call.z, "is_cuda", False):
            raise T.Refused("device:%s" % getattr(getattr(call.z, "device", None), "type", "unknown"))
        zz, cc, dt, C, H, N = _facts(call)
        why = None
        if zz.dim() != 3:
            why = "batch:%s" % (tuple(call.z.shape)[:-3],)
        elif torch.is_grad_enabled() and call.z.requires_grad:
            why = "backward:forward_only"
        else:
            ok, word = KE.supported(cc, dt, C, H, N, batch=1)
            why = None if ok else str(word)
        if why is None:
            call.extra["c64_route"] = "esm"
            if prov.version is None:
                prov.version = str(getattr(KE, "__version__", None) or STATE["opt_core"].get("version")); prov.resolved_from = "core"
            return
        _bump(STATE["esm_rerouted"], _tok(why)[:60])
        n = sum(STATE["esm_rerouted"].values())
        if n == 1:
            _say(f"REROUTE trimul esm:{_tok(why)[:60]} -> tmk3_exact (the cell's exact row), counted esm_rerouted")
        tmk3.eligible(call)                                             # Refused -> the arm's forward (counted fallback_by on the lever's line)
        call.extra["c64_route"] = "tmk3"

    def fn(call):
        if call.extra.get("c64_route") == "tmk3":
            return tmk3.fn(call)
        z, mask = call.z, call.mask
        zz = z[0] if (z.dim() == 4 and int(z.shape[0]) == 1) else z
        N = int(zz.shape[-2])
        mm = mask
        if mm is not None and mm.dim() != 2 and mm.numel() == N * N:
            mm = mm.reshape(N, N)
        cache = _ESM_CACHE.setdefault((id(call.module), call.direction), {})
        try:
            out = KE.triangle_multiplication(zz, mm, direction=call.direction, weights=weights_of(call.module), residual=call.residual, levers=None, cache=cache)
        except KE.Unsupported as e:                                     # the package's own refusal with the tensors in hand: this call's route to the arm's forward, by name
            raise T.Refused(("esm:%s" % getattr(e, "word", e)).replace(" ", "_")[:120]) from None
        return out if out.shape == z.shape else out.reshape(z.shape)

    prov = T.Provider("trimul_esm_shapes", fn, eligible, kernel="trimul_esm_shapes")
    return prov


def _make_trimul(op: str):
    direction = "outgoing" if op == "trimul_out" else "incoming"

    def provider(module, *a, **kw):
        inner = _resolve_inner(op)

        def found():
            f = inner if inner is not None else _stock(op)
            return f(module, *a, **kw)
        if _TRIMUL_LEVER is None:
            return found()
        try:
            z, mask, inplace_safe, add_inplace, tm, rest = _trimul_args(*a, **kw)
        except TypeError:                                   # a call form this module does not know: the arm's provider's to handle
            _bump(STATE, "trimul_passthrough")
            return found()
        if rest or not is_template_trimul(module, tm):
            _bump(STATE, "trimul_passthrough")
            return found()
        _bump(STATE, "trimul_target")
        from opt_core import trimul as T
        residual = bool(inplace_safe is True and add_inplace)   # the library branch returns z + update exactly then, else the bare update: the row fuses the same add
        call = T.Call(module, z, mask, direction, residual=residual, orig=found)
        out = _TRIMUL_LEVER.serve(call)
        if direction not in STATE["first"] and not _capturing():
            with _LOCK:
                if direction not in STATE["first"]:
                    try:
                        from opt_core.kernels import trimul as KT
                        sel = call.extra.get("trimul_selection")
                        if call.extra.get("c64_route") == "esm":               # the package route (opt_core < TIER_CORE): the esm package's own launch cell for this call (kernels.trimul_esm_shapes.select_cell); the tier route carries trimul_selection (below)
                            KE = importlib.import_module(ESM_KERNEL)
                            ec = KE.select_cell(tuple(torch_cc(z)), str(z.dtype).replace("torch.", ""), int(z.shape[-1]), int(weights_of(module)["w_ap"].shape[0]), int(z.shape[-2]), direction)
                            row, cell, desc = "trimul_esm_shapes", ec.get("key"), "trimul_esm_shapes " + (KE.describe_cell(ec) if hasattr(KE, "describe_cell") else str(ec.get("key")))
                        else:
                            row, cell, desc = getattr(sel, "row", None), getattr(sel, "cell", None), (KT.describe(sel) if sel is not None else "none(the arm's forward served: see fallback_by)")
                        STATE["first"][direction] = f = {"N": int(z.shape[-2]), "C": int(z.shape[-1]), "dtype": str(z.dtype).replace("torch.", ""), "residual": residual,
                                                         "mask": None if mask is None else list(mask.shape), "row": row, "cell": cell, "selection": desc}
                        _say(f"FIRST trimul {direction} N={f['N']} C={f['C']} dtype={f['dtype']} residual={residual} mask={f['mask']} selection=[{f['selection']}]")
                    except Exception as e:  # noqa: BLE001 — a report line, never a failure of the call
                        STATE["first"][direction] = {"error": repr(e)}
                        _say(f"FIRST trimul {direction} facts unavailable: {e!r}")
        return out
    provider.__name__ = provider.__qualname__ = op
    provider._ptx_c64 = True
    return provider


trimul_out = _make_trimul("trimul_out")
trimul_in = _make_trimul("trimul_in")


def _trimul_line() -> str:
    from opt_core import report as R
    lever_name = STATE.get("trimul_lever") or "templ_trimul_tmk3"
    if _TRIMUL_LEVER is None:
        why = STATE["trimul"] if str(STATE["trimul"]).startswith("aside:") else f"word:{STATE['trimul']}"
        return R.lever_line(TAG, "F2.trimul", "skipped", reason=_tok(why), impl="none", origin="core", bind=STATE.get("trimul_bind", "none"), lever=lever_name, c256_inner=STATE["trimul_passthrough"])
    rr = ",".join(f"{k}:{v}" for k, v in sorted(STATE["esm_rerouted"].items())) or "none"
    return _TRIMUL_LEVER.line() + " " + R.kv(("word", STATE["trimul"]), ("bind", STATE.get("trimul_bind", "none")), ("esm_rerouted", rr), ("lever", lever_name), ("c256_inner", STATE["trimul_passthrough"]))


# ------------------------------------------------------------------------------------------------------- triangle attention (the tl site)
def _install_triatt(word: str, prefer, min_tokens: int) -> str:
    """Wrap the library attention entry; ``prefer``: the switch word's row preference (a tuple in the kit's order) or None (the tier word alone)."""
    import torch
    from opt_core.kernels import triattn as KA
    import protenix.model.triangular.layers as TL
    tier = tier_word()                                  # fast | big: the kit mode's own tolerance word
    STATE["triatt_tier"] = tier
    prev = TL.cuequivariance_triangular_attn
    if getattr(prev, "_ptx_c64", None):
        return word                                     # idempotent: already this module's site
    STATE["wrapped_over"] = f"{getattr(prev, '__module__', '?')}:{getattr(prev, '__qualname__', getattr(prev, '__name__', '?'))}"

    def stock(q, k, v, bias, mask=None, scale=None):
        return prev(q, k, v, bias, mask, scale)

    def _shape_out(out, q, v):
        if out.shape[:-1] != q.shape[:-1]:
            out = out.reshape(tuple(q.shape[:-1]) + (int(v.shape[-1]),))
        return out

    def serve(q, k, v, bias, mask, scale):
        """One template call through the core's provider; a refusal by name re-binds the route (word) for the rest of the process."""
        route = STATE["triatt_route"] or {"word": tier, "prefer": prefer}
        for _ in range(6):                              # a re-route chain ends at a stock row well before this
            if route["word"] in STOCK_ROWS:
                _bump(STATE, "triatt_stock_by_name")
                return stock(q, k, v, bias, mask, scale), route["word"]
            try:
                out = KA.triangle_attention(q, k, v, bias, mask=mask, scale=scale, word=route["word"], prefer=route["prefer"], stock=stock)
                return _shape_out(out, q, v), None
            except KA.Refusal as r:
                nxt = str(getattr(r, "fallback", None) or "cueq")
                key = f"{getattr(r, 'row', '?')}:{_tok(getattr(r, 'kind', '?'))}->{nxt}"
                if _bump(STATE["triatt_rerouted"], key) == 1 and not _capturing():
                    _say(f"REROUTE triatt {key} ({_tok(getattr(r, 'detail', '') or r)[:200]}) — bound by name for the rest of this process")
                route = {"word": nxt, "prefer": None}
                STATE["triatt_route"] = route
        _bump(STATE, "triatt_stock_by_name")
        return stock(q, k, v, bias, mask, scale), "reroute_chain"

    @functools.wraps(prev)                              # carries the found callable's markers (__dict__: _fpf_headsplit, ...) so their owners' idempotence checks hold
    def site(q, k, v, bias, mask=None, scale=None, *a, **kw):
        if a or kw:
            _bump(STATE, "triatt_passthrough")
            return prev(q, k, v, bias, mask, scale, *a, **kw)
        try:
            H, S, D = int(q.shape[-3]), int(q.shape[-2]), int(q.shape[-1])
        except Exception:  # noqa: BLE001
            _bump(STATE, "triatt_passthrough")
            return prev(q, k, v, bias, mask, scale)
        if not is_template_triatt(H, D):
            _bump(STATE, "triatt_passthrough")
            return prev(q, k, v, bias, mask, scale)
        if S < min_tokens:                              # ARM T's size gate: below it the fast rows keep the library op here as the block core does (fast == exact in the trunk there)
            _bump(STATE, "triatt_below_gate")
            return prev(q, k, v, bias, mask, scale)
        _bump(STATE["triatt_shapes"], f"{S}x{H}x{D}/{str(q.dtype).replace('torch.', '')}")
        try:
            out, stock_row = serve(q, k, v, bias, mask, scale)
        except Exception as e:  # noqa: BLE001 — counted, the first two printed, the found callable serves THIS call; an out-of-memory propagates
            from opt_core.oom import is_oom
            if is_oom(e):
                raise
            n = _bump(STATE["triatt_errors"], type(e).__name__)
            if sum(STATE["triatt_errors"].values()) <= 2 and not _capturing():
                _say(f"KERNEL ERROR triatt {type(e).__name__}: {str(e)[:300]} -> the found callable serves this call (n={n})")
            return prev(q, k, v, bias, mask, scale)
        if stock_row is not None:
            return out
        _bump(STATE["triatt_served"], label(q, D, H, S, bias, mask))
        return out

    _SEL = {}

    def label(q, D, H, S, bias, mask) -> str:
        """The row the table selects for this call shape under the bound route (once per (S, dtype, route); the first prints the table's line)."""
        route = STATE["triatt_route"] or {"word": tier, "prefer": prefer}
        key = (S, str(q.dtype), route["word"], route["prefer"])
        row = _SEL.get(key)
        if row is None:
            try:
                sel = KA.select(torch.cuda.get_device_capability(q.device), q.dtype, D, H, S, word=route["word"], prefer=route["prefer"])
                row = _SEL[key] = str(sel.row)
                if STATE["first"].get("triatt") is None and not _capturing():
                    STATE["first"]["triatt"] = f = {"S": S, "H": H, "D": D, "q": str(q.dtype).replace("torch.", ""), "bias": str(bias.dtype).replace("torch.", ""),
                                                    "mask": None if mask is None else list(mask.shape), "q_shape": list(q.shape), "row": sel.row, "cell": sel.cell,
                                                    "selection": KA.describe(sel)}
                    _say(f"FIRST triatt S={S} H={H} D={D} q={f['q']} bias={f['bias']} mask={f['mask']} row={sel.row} selection=[{f['selection']}]")
            except Exception as e:  # noqa: BLE001 — a census label, never a failure of the call
                row = _SEL[key] = _tok(f"{route['word']}?{type(e).__name__}")
                if STATE["first"].get("triatt") is None and not _capturing():
                    STATE["first"]["triatt"] = {"error": repr(e), "row": None}
                    _say(f"FIRST triatt facts unavailable: {e!r}")
        return row
    site._ptx_c64 = True
    site.__wrapped__ = prev
    TL.cuequivariance_triangular_attn = site
    return word


def _triatt_line() -> str:
    from opt_core import report as R
    st = STATE["triatt"]
    served = sum(STATE["triatt_served"].values()); errs = sum(STATE["triatt_errors"].values())
    pairs = [("served", served), ("stock_by_name", STATE["triatt_stock_by_name"]),
             ("rerouted", ",".join(f"{k}:{v}" for k, v in sorted(STATE["triatt_rerouted"].items())) or "none"),
             ("errors", ",".join(f"{k}:{v}" for k, v in sorted(STATE["triatt_errors"].items())) or "none"),
             ("passthrough", STATE["triatt_passthrough"]), ("min_tokens", STATE["triatt_min_tokens"]), ("below_min_tokens", STATE["triatt_below_gate"]),
             ("shapes", ",".join(f"{k}:{v}" for k, v in sorted(STATE["triatt_shapes"].items())) or "none"),
             ("word", st if st in TRIATT_WORDS else "none"), ("prefer", prefer_token(STATE["triatt_prefer"])), ("bind", STATE.get("triatt_bind", "none")),
             ("gate", "refused" if errs else "ok"), ("lever", "templ_triatt_core")]
    impl = "triattn:" + ("+".join(sorted(STATE["triatt_served"])) or "none") + "@" + STATE.get("triatt_tier", TRIATT_TIER)
    if st in TRIATT_WORDS:
        return R.lever_line(TAG, "F1.flash_triatt", "on", *pairs, impl=impl, origin="core")
    why = st if str(st).startswith("aside:") else f"word:{st}"
    return R.lever_line(TAG, "F1.flash_triatt", "skipped", *pairs, reason=_tok(why), impl=impl, origin="core")


# ----------------------------------------------------------------------------------------------------------------------------- apply
def apply(environ=None) -> dict:
    """Read the two words, check the core, build the TriMul lever and install the attention site.  Idempotent; returns STATE."""
    global _TRIMUL_LEVER
    environ = os.environ if environ is None else environ
    if STATE.get("applied"):
        return STATE
    STATE["applied"] = True
    _INNER_SPEC.update(parse_inner(environ.get("FPF_OPS", "")))
    STATE["inner"] = dict(_INNER_SPEC)
    w_tm, w_ta = environ.get(TRIMUL_ENV, ""), environ.get(TRIATT_ENV, "")
    core_v, core_f = None, None
    try:
        import opt_core
        core_v, core_f = getattr(opt_core, "__version__", "0"), getattr(opt_core, "__file__", "?")
    except Exception as e:  # noqa: BLE001
        core_v, core_f = None, f"import-failed:{e!r}"
    STATE["opt_core"] = {"version": core_v, "file": core_f, "min": ".".join(map(str, MIN_CORE))}
    aside = None if (core_v is not None and core_ok(core_v)) else f"aside:opt_core{core_v}<{'.'.join(map(str, MIN_CORE))}"
    # -- templ_trimul_tmk3
    if not w_tm:
        STATE["trimul"] = "off"
    elif w_tm not in TRIMUL_WORDS:
        STATE["trimul"] = f"aside:unknown_word:{_tok(w_tm)}"
    elif aside:
        STATE["trimul"] = aside
    else:
        try:
            from opt_core import trimul as T
            mode_w, lever_name = TRIMUL_WORDS[w_tm]                     # mode_w: the Lever's class word (exact | fast); the provider below binds the core's TIER word
            if w_tm == "esm":                                           # ARM T: the core's fast | big tier row for the call's cell, by word — the row flips with the core's cells
                kind, word = arm_t_binding(core_v, environ)
                if kind == "tier":
                    prov = T.by_word(weights_of, word, name=f"trimul:{word}")
                    STATE["trimul_bind"] = f"tier:{word}"
                else:                                                   # opt_core < TIER_CORE: the trimul_esm_shapes package route of those cores, by name (never silent)
                    prov = _esm_provider(weights_of)
                    STATE["trimul_bind"] = "package:trimul_esm_shapes"
                    _say(f"templ_trimul_esm binds the {ESM_KERNEL} package by name: {word} (the core's tier row from opt_core {'.'.join(map(str, TIER_CORE))})")
            else:                                                       # ARM E: the core's exact tier row for the call's cell, by word
                prov = T.by_word(weights_of, mode_w, name=f"trimul:{mode_w}")
                STATE["trimul_bind"] = f"tier:{mode_w}"
            _TRIMUL_LEVER = T.Lever(TAG, mode_w, provider=prov, min_tokens=TRIMUL_MIN_TOKENS, expected=("mode_stock", "below_min_tokens"))
            STATE["trimul"], STATE["trimul_lever"] = w_tm, lever_name
        except Exception as e:  # noqa: BLE001 — a core that cannot build the lever: aside by name, the arm's providers serve
            STATE["trimul"] = f"aside:{type(e).__name__}:{_tok(e)[:80]}"
    # -- templ_triatt_core
    if not w_ta:
        STATE["triatt"] = "off"
    elif w_ta not in TRIATT_WORDS:
        STATE["triatt"] = f"aside:unknown_word:{_tok(w_ta)}"
    elif aside:
        STATE["triatt"] = aside
    else:
        try:
            STATE["triatt_prefer"] = TRIATT_WORDS[w_ta]
            try:
                STATE["triatt_min_tokens"] = max(0, int(environ.get(TRIATT_GATE_ENV, "") or 0))
            except ValueError:
                STATE["triatt_min_tokens"] = 0
            STATE["triatt"] = _install_triatt(w_ta, TRIATT_WORDS[w_ta], STATE["triatt_min_tokens"])
            STATE["triatt_bind"] = "tier:" + STATE.get("triatt_tier", TRIATT_TIER)                          # the provider's tier word the site passes (word=fast | big: the kit mode's word); prefer= names the row preference riding with it, none = the cell decides
        except Exception as e:  # noqa: BLE001 — no protenix / no core provider in this interpreter: aside by name
            STATE["triatt"] = f"aside:{type(e).__name__}:{_tok(e)[:80]}"
    for k in ("trimul", "triatt"):
        if str(STATE[k]).startswith("aside:"):
            _say(f"{'templ_trimul_tmk3' if k == 'trimul' else 'templ_triatt_core'} steps aside by name: {STATE[k][6:]} (the arm's own providers serve; bytes unchanged)")
    try:                                                    # the exit lines (opt_core's per-lever grammar), once per process
        from opt_core import report as R
        if w_tm:
            R.register_exit_tally(TAG + "/F2.trimul", _trimul_line)
        if w_ta:
            R.register_exit_tally(TAG + "/F1.flash_triatt", _triatt_line)
    except Exception:  # noqa: BLE001
        pass
    try:                                                    # the trunk levers' tally carries this census into the kit's $PTX_LEVER_REPORT record
        LEV = sys.modules.get("ptx_trunk2_levers")
        if LEV is not None and isinstance(getattr(LEV, "_STATS", None), dict):
            LEV._STATS["c64_routes"] = STATE
    except Exception:  # noqa: BLE001
        pass
    _say(f"APPLIED trimul={STATE['trimul']} bind={STATE['trimul_bind']} triatt={STATE['triatt']} inner_out={_INNER_SPEC.get('trimul_out')} inner_in={_INNER_SPEC.get('trimul_in')} "
         f"trimul_min_tokens={TRIMUL_MIN_TOKENS} triatt_min_tokens={STATE['triatt_min_tokens']} prefer={prefer_token(STATE['triatt_prefer'])} triatt_bind={STATE['triatt_bind']} "
         f"opt_core={core_v}@{core_f} wrapped_over={STATE['wrapped_over']}")
    return STATE


def report() -> dict:
    """Plain-data census for a manifest: the words, the core, the inner providers, both levers' counters."""
    out = {k: v for k, v in STATE.items()}
    out["trimul_census"] = _TRIMUL_LEVER.census() if _TRIMUL_LEVER is not None else None
    return out


if os.environ.get("PTX_C64_ROUTES_NO_AUTOAPPLY", "") != "1":      # the kit process imports this module through FPF_OPS: apply at import (tests import it with the guard set)
    apply()
