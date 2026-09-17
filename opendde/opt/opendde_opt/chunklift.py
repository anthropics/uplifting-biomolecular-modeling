"""chunk_lift — the tree's own lever on upstream's dynamic attention-chunk resolution (class forward). Tolerance class: per element the
arithmetic is the un-chunked statement's, but a lifted chunk changes the kernels' reduction order at some sizes — byte-identical to stock under
the deterministic recipe at some token counts and not at others — so the lever rides the
fast line and the big lines below their offload size gate, not the exact line.

Upstream resolves one chunk size per stage from the item's token count (`opendde/model/opendde.py:1796-1836`): the threshold TABLE of
`configs.infer_setting.chunk_size_thresholds` (`opendde/config/model_base.py:44-51`: no chunking <= 1024 tokens, 512 rows <= 1536, 256 <= 2048,
128 <= 2560, 32 beyond — `_get_dynamic_chunk_size`), then a fixed score budget CLAMPS it (`_bound_pairformer_chunk_size`: the row chunk c is
bounded by the largest power of two <= 450e6 / N^2, so that c x N^2 <= 450 M score elements per head whatever the device). The resolution runs
twice per item in `_main_inference_loop` (:1859 at the residue token count N -> the trunk's pairformer / MSA-module / template-embedder
triangle attention and outer-product mean and the input embedder's atom attention; :1890 at the structural token count N_st ~ 1.9 N -> the
structural refiner's triangle attention and, dynamic chunking being on, the diffusion sampler's atom attention windows and the confidence
head's pairformer). Each consumer runs `chunk_layer` over row blocks of c (`opendde/model/utils.py`; `model/triangular/triangular.py:_chunk`,
`model/modules/primitives.py:_local_attention`, `model/triangular/layers.py:OuterProductMean._chunk`): per element the arithmetic is the
un-chunked statement's, only the launch count differs — at N = 1,000 the clamp (256 rows in the trunk, 64 at N_st ~ 1,910) makes the sampler's
atom attention alone run ~80 sub-batches per call, ~95,000 `chunk_layer` iterations per item of a dozen small kernels each.

The lever replaces the clamp's verdict by the TABLE's value when the probed device admits it, per item and per resolution site, and keeps
the clamp's value when it does not. Admission is the memory the table value needs at this token count in the reference (unfused) attention
statement — the [c, H, n, n] fp32 score tensor and its softmax copy of ONE triangle-attention call, H = 4 heads (`need_bytes`): the transient
the table was written for and the one the clamp bounds to 2 x 4 x 450e6 x 4 B = 14.4 GB — against the device's free bytes NOW (the caching
allocator's view: `opt_core.mem.budget.device_free_bytes`, driver free + reserved-unallocated) less a reserve of `RESERVE_FRAC` of the
device total plus `RESERVE_GIB` (`admit`). The fused triangle-attention kernels the kit's lines run (cuEquivariance, the arm's K2B cell) never
materialise the scores, so the rule over-states their need by construction: what actually grows when a site is admitted is the q/k/v/gate/output
projections of one triangle-attention call ([c, n, 128] fp32 each) and the outer-product-mean transient ([c, n, 32, 32] fp32) — ~5 GB at
N = 1,024 unchunked, against the ~32 GB the rule budgets. An 80 GB card admits the residue site at every N (the table's own design point) and the
structural site up to N_st = 2,560 when the trunk's residents leave the room; a 40 GB card clamps from N ~ 1,000 up; above N_st = 2,560
the table and the clamp agree (32 rows) and there is nothing to lift. No CUDA device / no probe -> the clamp's value (decision `no_probe`).

The residue site lifts only to an UN-CHUNKED table value (N <= the table's un-chunked gate, 1,024): where the table names a chunk larger than
the clamp (512 rows at 1,025..1,536, 256 at ..2,048) the clamp is kept and the decision reads `above_gate:<tokens>/<gate>` — larger trunk chunks
at 1,472 / 1,600 tokens were measured to cost the diffusion sampler several times what they save in the trunk. The structural site lifts
chunked values too (its consumers — refiner, sampler, confidence head — are the ones that gain); with both rules the lever is inert above ~1,340
residue tokens (structural tokens > 2,560: table == clamp) and engages from 345 (structural) / 663 (residue) tokens up.

Composition-time ceiling (`plan` / `compose`, the small-input floor's shape): the lever is composed INTO a `pred` call's line only when every
item counts <= CEILING (= the table's un-chunked gate, 1,024) residue tokens; above it, for a mixed query, and when no query was read it is
composed OUT — not installed, the model class untouched, LEVER row `state=off reason=above_gate:<tokens>/<gate>`, the ACTIVE / PRED lines
carrying `ceiling=above_gate:<tokens>/<gate>`. Measured on H100: with the lever installed (whatever chunk it served, the clamp's included)
the diffusion sampler of a 1,472 / 1,600-token item ran markedly slower than without it, cause not identified; at <= 1,000 tokens
the installed lever is a net gain. The residue site's `above_gate` decision below stays reachable: the ceiling counts polymer
residues, the model's token count adds ligand atoms.

One CHUNK line per decision on stdout (`line`), counted in STATS (`admitted`, `clamped_by_memory`, `stock` = table and clamp agree, `above_gate` =
the residue site above the un-chunked gate, `fixed` = dynamic chunking off: upstream's pass-through, `no_probe`); `calls` is the lever's ran
counter (ran.COUNTERS). Carried by `fast` and — below its
offload size gate, where the line is fast's resident lever set — `big` (the offload composition and `--n_gpu P>1` keep upstream's clamp:
modes.BIG_KIT_ROWS_OFF / BIG_TP_DROP). Installed from `stack._apply` for the lines that carry it through the core's per-site patch-at-import on `OpenDDE._resolve_pairformer_chunk_size` and
`OpenDDE._main_inference_loop` (the item context: which of the two resolutions this is), at once when the model module is already imported,
else right after its body executes. Numerics: byte-identity with the lever absent is a property of the device stack (a row-blocked launch of
the same fp32 GEMM / attention kernel vs the full-extent launch), established per line by the tree's equality tests; CHANGES.md states the tier.
"""
from __future__ import annotations

import sys

# opt_core (autoload, mem.budget) and .report are imported where they are used: modes.resolve composes the ceiling (compose / word) in processes
# that may hold no core at all (the absent-core refusal names the big lines before anything of the core is touched).

TARGET = "opendde.model.opendde"
CLASS = "OpenDDE"
RESOLVE, LOOP = "_resolve_pairformer_chunk_size", "_main_inference_loop"
TAG = "opendde-opt"
HEADS = 4                                   # triangle-attention heads on the pin (pairformer, MSA pair stack, template embedder, refiner, confidence)
SCORE_BYTES = 4                             # fp32 scores (the trunk runs fp32 on every line; bf16 arms up-cast the softmax input)
SCORE_COPIES = 2                            # the score tensor and its softmax output are alive together (layers.py `_attention`: a = softmax_no_cast(a))
RESERVE_FRAC = 0.15                         # of the device total: room for the stage's own residents that arrive after the decision
RESERVE_GIB = 1.0
GIB = float(1 << 30)
SITES = ("residue", "structural")           # the two resolutions of one item, in call order (opendde.py:1859, :1890)
UNCHUNKED_ONLY = {"residue": True, "structural": False}   # the residue site lifts only to the table's un-chunked value (N <= its un-chunked gate, 1,024 on the pin): a larger
                                            # CHUNK there (512 / 256 rows at 1,472 / 1,600 tokens) was measured to slow the diffusion sampler by more than it speeds the
                                            # trunk; the structural site (refiner, sampler, confidence: the consumers that gain) lifts chunked values too, memory admitting
ABOVE_GATE = "above_gate"                   # the decision word of that decline: `decision=above_gate:<tokens>/<gate>` (the size-gate vocabulary of the tree's LEVER rows)

STATS = {"installed": False, "armed": False, "calls": 0, "admitted": 0, "clamped_by_memory": 0, "stock": 0, "above_gate": 0, "fixed": 0, "no_probe": 0,
         "items": 0, "decisions": []}      # decisions: one dict per CHUNK line (the manifest's evidence; bounded to the last 64)
_CTX = {"item": 0, "site_calls": 0}         # the running item's resolution count (reset by the loop wrapper)


class ActivationError(RuntimeError):
    """The model class has no `_resolve_pairformer_chunk_size` / `_main_inference_loop` to wrap: the kit's activation fails by name."""


# ---------------------------------------------------------------- arithmetic (pure; the CPU tests call these)
PIN_THRESHOLDS = {"1024": -1, "1536": 512, "2048": 256, "2560": 128}   # upstream's infer_setting.chunk_size_thresholds on the pin (config/model_base.py:44-51); -1 = un-chunked


def gate_of(thresholds) -> int:
    """The un-chunked gate of a threshold table in tokens: its largest threshold whose value is -1 (no chunking), 0 when it has none."""
    items = thresholds.items() if hasattr(thresholds, "items") else dict(thresholds).items()
    return max([int(k) for k, v in items if int(v) == -1] or [0])


def table_chunk(n_token: int, thresholds: dict | None = None) -> int | None:
    """Upstream's threshold table verbatim (`_get_dynamic_chunk_size`): None = un-chunked. `thresholds` = infer_setting.chunk_size_thresholds."""
    th = thresholds if thresholds is not None else PIN_THRESHOLDS
    for t, c in sorted((int(k), v) for k, v in th.items()):
        if n_token <= t:
            return None if c == -1 else int(c)
    return 32


def stock_clamp(n_token: int, chunk: int | None, budget_elems: int = 450_000_000) -> int | None:
    """Upstream's `_bound_pairformer_chunk_size` verbatim: the table value bounded by the largest power of two <= budget / n^2."""
    requested = chunk or n_token
    b = max(1, budget_elems // max(1, n_token * n_token))
    p2 = 1 << (b.bit_length() - 1)
    bounded = min(requested, p2)
    if chunk is None and bounded >= n_token:
        return None
    return bounded


def need_bytes(n_token: int, chunk: int | None, heads: int = HEADS) -> int:
    """The reference statement's transient at row chunk `chunk` (None = all n rows): SCORE_COPIES x [c, H, n, n] fp32."""
    c = n_token if chunk is None else min(int(chunk), n_token)
    return SCORE_COPIES * c * heads * n_token * n_token * SCORE_BYTES


def reserve_bytes(total_bytes: int) -> int:
    return int(RESERVE_FRAC * total_bytes + RESERVE_GIB * GIB)


def admit(n_token: int, table: int | None, clamp: int | None, free_bytes, total_bytes, *, unchunked_only: bool = False, gate: int = 0) -> dict:
    """The decision of one resolution: {'chunk', 'decision', 'need', 'free', 'reserve'} — `chunk` is what the model gets. `unchunked_only` (the
    residue site): only an un-chunked table value (None) is lifted; a chunked table value larger than the clamp — the item is above the table's
    un-chunked gate `gate` — keeps the clamp, named `above_gate:<tokens>/<gate>` (measured: larger trunk chunks at 1,472 / 1,600 tokens cost the
    diffusion sampler more than they save the trunk)."""
    need = need_bytes(n_token, table)
    if table == clamp:
        return {"chunk": clamp, "decision": "stock", "need": need, "free": free_bytes, "reserve": None}
    if unchunked_only and table is not None:
        return {"chunk": clamp, "decision": f"{ABOVE_GATE}:{int(n_token)}/{int(gate)}", "need": need, "free": free_bytes, "reserve": None}
    if free_bytes is None or total_bytes is None:
        return {"chunk": clamp, "decision": "no_probe", "need": need, "free": None, "reserve": None}
    reserve = reserve_bytes(total_bytes)
    ok = need <= int(free_bytes) - reserve
    return {"chunk": table if ok else clamp, "decision": "admitted" if ok else "clamped_by_memory", "need": need, "free": int(free_bytes), "reserve": reserve}


def _word(c) -> str:
    return "none" if c is None else str(int(c))


def line(item, site: str, n_token: int, table, clamp, d: dict) -> str:
    """`[opendde-opt] CHUNK item=<i> site=residue|structural n=<tokens> table=<rows|none> clamp=<rows|none> chunk=<rows|none> decision=<word>
    need_gib=<f> free_gib=<f|none> reserve_gib=<f|none>` — one per resolution; `chunk` is the value handed to the model; <word> = admitted |
    clamped_by_memory | stock | above_gate:<tokens>/<gate> | fixed | no_probe."""
    from .report import PREFIX
    f = lambda b: "none" if b is None else f"{b / GIB:.2f}"   # noqa: E731
    return (f"{PREFIX} CHUNK item={item} site={site} n={int(n_token)} table={_word(table)} clamp={_word(clamp)} chunk={_word(d['chunk'])} "
            f"decision={d['decision']} need_gib={f(d['need'])} free_gib={f(d['free'])} reserve_gib={f(d['reserve'])}")


def unchunked_gate(model) -> int:
    """The table's un-chunked gate in tokens: the largest threshold of `configs.infer_setting.chunk_size_thresholds` whose value is -1 (no chunking;
    1,024 on the pin), 0 when the table has none."""
    try:
        return gate_of(getattr(model.configs.infer_setting, "chunk_size_thresholds"))
    except Exception:  # noqa: BLE001 — no table: no un-chunked gate
        return 0


# ---------------------------------------------------------------- the composition-time ceiling (cli.pred -> plan; modes.resolve -> compose) — smalln's shape
CEILING = gate_of(PIN_THRESHOLDS)           # 1,024 residue tokens: the table's un-chunked gate. The lever is composed INTO a pred call's line only when every item
                                            # counts <= it; above it, mixed, or no query read (check, the env route) it is composed OUT — nothing of it touches the
                                            # model class (no wrap, no probe, no CHUNK line) and its LEVER row reads `state=off reason=above_gate:<tokens>/<gate>`.
                                            # Measured: with the lever merely INSTALLED the diffusion sampler of a 1,472 / 1,600-token item runs markedly slower
                                            # whatever chunk it serves (cause not identified), while at <= 1,000 tokens the installed lever is a net gain.
_PLAN = {"plan": None}


def _reset() -> None:
    _PLAN["plan"] = None


def policy(tokens) -> dict:
    """The ceiling decision over ``tokens`` = ``big.count_tokens(jobs)`` ({item: {"residue_tokens", ...}}) or None when no query was read:
    ``{"gate", "max_tokens", "n_items", "within", "case", "reason"}``; ``within`` only when every item counts <= CEILING (case within | above | unknown)."""
    if tokens is None:
        return {"gate": CEILING, "max_tokens": None, "n_items": 0, "within": False, "case": "unknown",
                "reason": "no query read before activation: chunk_lift composed out (the safe side)"}
    counts = [int(t["residue_tokens"]) for t in tokens.values()]
    mx, n = max(counts, default=0), len(counts)
    if n and mx <= CEILING:
        return {"gate": CEILING, "max_tokens": mx, "n_items": n, "within": True, "case": "within",
                "reason": f"every item within the ceiling (largest {mx} residue tokens <= {CEILING}): chunk_lift composed in"}
    n_above = sum(1 for c in counts if c > CEILING)
    return {"gate": CEILING, "max_tokens": mx, "n_items": n, "within": False, "case": "above",
            "reason": f"{n_above} of {n} items above the ceiling (largest {mx} residue tokens > {CEILING}): chunk_lift composed out"}


def plan(res, query_path) -> dict:
    """The pre-activation hook (``cli.pred``): the ceiling decision for a line that carries chunk_lift, from the query's residue tokens (the kit's one
    counter, ``big.count_tokens``), kept in this process for :func:`compose`; ``{}`` for every other line and for ``off``."""
    line = getattr(res, "line", None)
    if line is None:                                                   # `off`: nothing to plan. Every kit line is planned: `res.line` may already be this module's own
        _PLAN["plan"] = None                                           # composed-out line (modes.resolve ran before the plan), so membership of chunk_lift in it says nothing;
        return {}                                                      # compose / gated_off sort out the lines whose rows never carry the lever
    tokens = None
    if query_path:
        from . import inputs as _inputs
        from .big import count_tokens
        tokens = count_tokens(_inputs.load_query(query_path))
    pol = policy(tokens)
    pol["tokens"] = tokens
    _PLAN["plan"] = pol
    return pol


def planned():
    """This process's ceiling decision (None until :func:`plan` ran for a line carrying the lever)."""
    return dict(_PLAN["plan"]) if _PLAN["plan"] is not None else None


def word():
    """``above_gate:<tokens>/<gate>`` when the lever is composed out (``<tokens>`` = the largest item's residue tokens, ``unknown`` when no query was
    read), None when it is composed in."""
    pol = _PLAN["plan"]
    if pol is not None and pol.get("within"):
        return None
    mx = pol.get("max_tokens") if pol else None
    return f"{ABOVE_GATE}:{mx if mx is not None else 'unknown'}/{CEILING}"


def compose(line):
    """``line`` as the plan composes it: itself when every item is within the ceiling, else without chunk_lift (``modes.line_without``)."""
    if line is None or "chunk_lift" not in line.levers or word() is None:
        return line
    from . import modes
    return modes.line_without(line, ("chunk_lift",))


def gated_off(line) -> dict:
    """``{"chunk_lift": "above_gate:<tokens>/<gate>"}`` when the ceiling composed the lever out of ``line`` (its static row carries it, the composed line
    does not, and the plan did not admit it); {} otherwise (composed in, or a line whose static row never carries it)."""
    from . import modes
    if line is None or "chunk_lift" in line.levers or word() is None:
        return {}
    static = modes.LINES.get(line.name)
    carries = static is not None and ("chunk_lift" in static.levers or line.name in modes.BIG_LINES)   # big: the resident composition carries it
    return {"chunk_lift": word()} if carries else {}


# ---------------------------------------------------------------- the device probe (torch lazily; None without CUDA)
def probe() -> tuple:
    """(free_bytes, total_bytes) of the current CUDA device as the caching allocator can still hand out, or (None, None)."""
    try:
        from opt_core.mem import budget as _budget
        free = _budget.device_free_bytes()
        if free is None:
            return None, None
        import torch  # noqa: WPS433
        _, total = torch.cuda.mem_get_info()
        return int(free), int(total)
    except Exception:  # noqa: BLE001 — a probe that raises is a probe that did not admit: the clamp's value, counted `no_probe`
        return None, None


def _item_name() -> str:
    ph = sys.modules.get("opendde_opt.phase")
    cur = getattr(ph, "CURRENT", None) if ph is not None else None
    name = (cur or {}).get("item") if isinstance(cur, dict) else None
    return str(name) if name else f"#{_CTX['item']}"


# ---------------------------------------------------------------- the wrappers installed on the class
def make_resolve(orig):
    """`OpenDDE._resolve_pairformer_chunk_size` under the lever: upstream's table, upstream's clamp, the kit's admission between them."""
    def _resolve_pairformer_chunk_size(self, n_token, chunk_size, *, dynamic_chunk_size):
        STATS["calls"] += 1
        site = SITES[min(_CTX["site_calls"], len(SITES) - 1)]
        _CTX["site_calls"] += 1
        if not dynamic_chunk_size:                                    # upstream honours a fixed chunk as given: so does the lever (counted, printed)
            STATS["fixed"] += 1
            d = {"chunk": chunk_size, "decision": "fixed", "need": need_bytes(int(n_token), chunk_size), "free": None, "reserve": None}
            _emit(site, int(n_token), chunk_size, chunk_size, d)
            return chunk_size
        table = self._get_dynamic_chunk_size(n_token)                  # upstream's own table read (infer_setting.chunk_size_thresholds)
        clamp = type(self)._bound_pairformer_chunk_size(n_token, table)   # upstream's own clamp, called not transcribed
        free, total = probe()
        d = admit(int(n_token), table, clamp, free, total, unchunked_only=UNCHUNKED_ONLY[site], gate=unchunked_gate(self))
        STATS[d["decision"].split(":")[0]] += 1
        _emit(site, int(n_token), table, clamp, d)
        return d["chunk"]
    _resolve_pairformer_chunk_size._chunk_lift = True
    _resolve_pairformer_chunk_size.__wrapped__ = orig
    return _resolve_pairformer_chunk_size


def make_loop(orig):
    """`OpenDDE._main_inference_loop` under the lever: the item context only (which resolution comes next); arguments and result untouched."""
    import functools

    @functools.wraps(orig)
    def _main_inference_loop(self, *a, **k):
        _CTX["item"] += 1
        _CTX["site_calls"] = 0
        STATS["items"] += 1
        return orig(self, *a, **k)
    _main_inference_loop._chunk_lift = True
    return _main_inference_loop


def _emit(site, n_token, table, clamp, d):
    rec = {"item": _item_name(), "site": site, "n": n_token, "table": table, "clamp": clamp, **d}
    STATS["decisions"] = (STATS["decisions"] + [rec])[-64:]
    print(line(rec["item"], site, n_token, table, clamp, d), flush=True)


# ---------------------------------------------------------------- install / state
def install() -> dict:
    """Idempotent: patches the class now when the model module is imported, else arms the per-site patches for its import."""
    if STATS["installed"]:
        return dict(STATS)
    from opt_core import autoload
    try:
        p1 = autoload.patch_attr_at_import(TARGET, f"{CLASS}.{RESOLVE}", make_resolve, tag=TAG, name="chunk_lift")
        p2 = autoload.patch_attr_at_import(TARGET, f"{CLASS}.{LOOP}", make_loop, tag=TAG, name="chunk_lift.item")
    except autoload.PatchError as e:
        raise ActivationError(f"chunk_lift: {e}") from None
    STATS["patches"] = (p1, p2)
    _sync()
    return dict(STATS)


def _sync() -> None:
    ps = STATS.get("patches") or ()
    if ps:
        STATS["installed"] = all(p.state == "installed" for p in ps)
        STATS["armed"] = any(p.state == "armed" for p in ps)


def fallbacks(planned) -> list:
    """The gate's named events for a planned chunk_lift: the model module imported but the class not wrapped. A process that never imported the
    model (nothing predicted here) has no event."""
    if "chunk_lift" not in (planned or []):
        return []
    _sync()
    m = sys.modules.get(TARGET)
    if m is None:
        return []
    cls = getattr(m, CLASS, None)
    if cls is None or not getattr(getattr(cls, RESOLVE, None), "_chunk_lift", False):
        return ["chunk_lift: the model module is imported but OpenDDE._resolve_pairformer_chunk_size is not wrapped (installed after its import without the patch)"]
    return []


def kit_stats() -> dict:
    _sync()
    return {k: v for k, v in STATS.items() if k not in ("patches", "decisions")} | {"last": (STATS["decisions"][-1] if STATS["decisions"] else None)}
