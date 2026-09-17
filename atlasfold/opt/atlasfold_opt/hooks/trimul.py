"""Lever trimul_exact / trimul_v4 — the two TriangleMultiplication classes of atlasfold (stock triangle_update.py L119-290).

Stock: ``forward(self, z, mask, kernel_backend="torch")`` returns the UPDATE (the PairBlock adds it: block.py L255-267); with
kernel_backend == "cuequiv" it calls cuequivariance's fused ``triangle_multiplicative_update`` with (norm_in w/b, p_in [2C,C], g_in [2C,C],
norm_out w/b, p_out [C,C], g_out [C,C], eps 1e-5); the [2C,C] projections are [a;b] stacked (torch path: chunk(2) -> a first).
Here: the cuequiv branch goes through ONE ``opt_core.trimul.Lever`` per process whose provider is the shared triangle-multiplication provider
``opt_core.kernels.trimul`` (``opt_core.trimul.by_word``) bound by the TIER WORD OF THE MODE — exact -> ``exact``, fast -> ``fast``, big ->
``big``.  Which row serves a call (capability, precision, c_z, N bucket, direction) is the provider's measured cell on this stack: under the
exact word an exact-class row only where the provider holds its bitwise record on this stack, else the provider's stock cuEquivariance row
``cueq`` BY NAME; under fast / big the cell's fastest admitted row.  No kit table, floor or pin takes part: small N, packed batches and cards
without a cell are the provider's to decide, and it says so by name.  A call a row refuses BY NAME (``<row>:<kind>``) is served by the engine's
own forward (the stock op), counted on the LEVER line; kernel_backend "torch" stays the stock torch path, counted ``fallback:backend_torch``.
residual=False.  Developer overrides (engineering, never a mode): ``AFO_TRIMUL_EXACT_WORD`` / ``AFO_TRIMUL_WORD`` name a provider ROW word in
place of the tier word (the exact lever accepts exact-class and stock rows only)."""
import os

from . import Installed, rebind

TARGET = "atlasfold.model.network.primitives.triangle_update"
CLASSES = (("TriangleMultiplicationOutgoing", "outgoing"), ("TriangleMultiplicationIncoming", "incoming"))
WORD_ENV = {"exact": "AFO_TRIMUL_EXACT_WORD", "fast": "AFO_TRIMUL_WORD"}     # developer overrides: a provider row word (big reads the fast lever's variable)
TIER_WORD = {"exact": "exact", "fast": "fast", "big": "big"}            # mode -> the provider's tier word (big passes `big`, never `fast`)
EXPECTED = ("mode_stock", "backend_torch", "device:", "rank:")              # + every provider row's named refusal `<row>:` (install reads opt_core.kernels.trimul.ROW_NAMES)


def word(mode: str) -> str:
    """The word the lever passes to the provider: AFO_TRIMUL_EXACT_WORD (exact) / AFO_TRIMUL_WORD (fast, big) when set, else the TIER WORD of
    the mode (`exact` | `fast` | `big`) — the provider's cell for that tier serves, and a provider-side change of a cell reaches the mode
    without a kit change."""
    lever_cls = "exact" if mode == "exact" else "fast"
    w = (os.environ.get(WORD_ENV[lever_cls]) or "").strip()
    return w or TIER_WORD.get(mode, "fast")


def weights_of(m):
    """AtlasFold TriangleMultiplication{Outgoing,Incoming} -> the core's TriMul weight vocabulary (opt_core.trimul_weights.WEIGHT_KEYS)."""
    C = m.linear_out.weight.shape[0]
    w_in, g_in = m.linear_in.weight, m.linear_g_in.weight          # [2C, C] each: rows [0:C] -> a, [C:2C] -> b (torch.chunk order, L186)
    return {"ln_in_w": m.layernorm_in.weight, "ln_in_b": m.layernorm_in.bias,
            "w_ap": w_in[:C], "w_bp": w_in[C:], "w_ag": g_in[:C], "w_bg": g_in[C:],
            "ln_out_w": m.layernorm_out.weight, "ln_out_b": m.layernorm_out.bias,
            "w_o": m.linear_out.weight, "w_og": m.linear_g_out.weight}


def expected_words(row_names) -> tuple:
    """The fallback reasons the lever EXPECTS: the kit's own words plus every provider row's named refusal `<row>:<kind>` (that call is served by
    the engine's forward — a named step-aside to the stock op, exit 0); anything else refuses the gate."""
    return EXPECTED + tuple(f"{r}:" for r in row_names)


def _gate_with_boundary(lever):
    """The core Lever gate, except: when EVERY call fell back for reasons this kit lists as expected (every call refused BY NAME by the
    provider's rows -> the stock cuEquivariance TriMul ran, the arithmetic the exact tier promises), a `served 0` verdict is reported as ok
    with the boundary named, not as a refusal."""
    def gate():
        g = lever.gate()
        if getattr(g, "ok", True):
            return g
        reason = getattr(g, "reason", "") or ""
        c = lever.census() if hasattr(lever, "census") else {}
        fb = dict(c.get("fallback", {}))
        if reason.startswith("mode ") and " served 0 of " in reason and fb and all(lever._expected(r) for r in fb):
            from opt_core.gates import Gate
            return Gate(name=getattr(g, "name", "F2.trimul"), ok=True, reason="all calls at the documented boundary -> stock path: " + ",".join(sorted(fb)), details=c)
        return g
    return gate


def _line_with_boundary(lever, kit_gate):
    """The core Lever line, with the gate word made consistent with this kit's verdict: when the core says refused only because every call
    fell back for a reason the kit lists as expected (a named row refusal on every call), the kit gate is ok and the run continues on the
    stock cuEquivariance TriMul — the line then says so (`gate=ok fallback_to=stock_cueq reason=<reasons>`) instead of `gate=refused`."""
    def line():
        text = lever.line()
        core = lever.gate()
        if getattr(core, "ok", True):
            return text
        kit = kit_gate()
        if getattr(kit, "ok", False):
            from opt_core import report as _report
            fb = sorted((lever.census().get("fallback") or {}).keys())
            text = text.replace(" gate=refused", " gate=ok") + " " + _report.kv(("fallback_to", "stock_cueq"), ("reason", ",".join(fb) or "expected"))
        return text
    return line


def install(mode: str, tag: str, ctx: dict) -> Installed:
    import importlib
    lever_name = "trimul_exact" if mode == "exact" else "trimul_v4"
    lever_cls = "exact" if mode == "exact" else "fast"                  # opt_core.trimul.Lever classes: stock | exact | fast (big is fast-class)
    try:
        from opt_core import trimul as T
        from opt_core.kernels import trimul as KT                        # pure-stdlib import (torch only inside its serving functions)
    except Exception as e:  # noqa: BLE001
        return Installed(lever_name, False, reason=f"core:kernels.trimul:{type(e).__name__}")
    if not hasattr(T, "by_word"):
        return Installed(lever_name, False, reason="core:trimul_provider_missing(opt_core.trimul.by_word)")
    try:
        tu = importlib.import_module(TARGET)
    except Exception as e:  # noqa: BLE001
        return Installed(lever_name, False, reason=f"import:{TARGET}:{type(e).__name__}")
    if getattr(tu, "_cueq_triangle_multiplicative_update", None) is None:
        return Installed(lever_name, False, reason="stock_cuequivariance_missing (the lever composes over the cuequiv branch)")
    rows, tiers = tuple(KT.ROW_NAMES), tuple(KT.TIER_WORDS)
    w = word(mode)
    if w not in rows + tiers:
        return Installed(lever_name, False, reason=f"unknown_word:{w[:40]}".replace(" ", "_"))
    if lever_cls == "exact" and w not in tuple(KT.EXACT_ROWS) + tuple(KT.STOCK_ROWS) + ("exact",):
        return Installed(lever_name, False, reason=f"not_an_exact_word:{w[:40]}")   # the exact lever's class is exact: a tolerance word belongs to trimul_v4
    provider = T.by_word(weights_of, w, cache_key=f"atlasfold.by_word.{lever_cls}")
    lever = T.Lever(tag, lever_cls, provider=provider, min_tokens=0, expected=expected_words(rows))
    for r in ("fpf_trimul", "fpf_trimul_v4"):
        try:                                                        # route the carried kernel names to the core copies before first import
            from opt_core import kernels as _K
            if r in _K.names():
                _K.route(r)
        except Exception:  # noqa: BLE001 — a name the core does not carry stays whatever sys.path resolves (reported as origin=kit|none)
            pass

    def make(direction, stock_forward):
        def forward(self, z, mask, kernel_backend: str = "torch"):
            if kernel_backend != "cuequiv":
                lever._count(direction, "fallback:backend_torch")
                return stock_forward(self, z, mask, kernel_backend=kernel_backend)
            call = T.Call(self, z, mask, direction, False, lambda: stock_forward(self, z, mask, kernel_backend=kernel_backend))
            return lever.serve(call)
        forward.__qualname__ = f"TriangleMultiplication[{direction}].forward[atlasfold_opt:{lever_name}]"
        return forward

    for cls_name, direction in CLASSES:
        cls = getattr(tu, cls_name)
        stock = cls.forward
        rebind(cls, "forward", make(direction, stock), stock)
    kit_gate = _gate_with_boundary(lever)
    return Installed(lever_name, True, lines=[_line_with_boundary(lever, kit_gate)], gates=[kit_gate],
                     facts={"provider": provider.name, "classes": [c for c, _ in CLASSES], "word": w, "tier_word": w in tiers})
