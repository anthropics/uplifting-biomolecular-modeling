"""ProGen2 kit v0_score — the LIKELIHOOD route (likelihood.py ``ll()``: one forward per sequence per direction under
autocast(fp16), the 25-aa renormalised log-softmax, the stock cross_entropy sum / mean) served by

  (1) RESIDENT ROTARY TABLES — modeling_progen.fixed_pos_embedding (L39-44: a CPU arange/einsum + a pageable H2D copy + device
      sin/cos, called once per layer per forward) replaced by the stock function's OWN output computed ONCE per (device, dim) for
      n_positions positions and sliced per call;
  (2) the stock prefill forward (ProGenForCausalLM.forward called VERBATIM with ``use_cache=False`` and no ``labels`` — the presents
      tuple and the full-vocab loss are not outputs of the route and touch no logit), eager, once per direction;
  (3) EXACT-LENGTH BATCHING — both directions of a sequence and every equal-length sequence in ONE unpadded [B, L] forward, no
      attention mask (the stock runs a 1-D [L] input viewed as [1, L]; the GEMMs see M = B*L);
  (4) the HOST PIPELINE — pinned ids/targets H2D, the per-row stock cross_entropy sum/mean kernels on the device, ONE pinned D2H
      per batch behind a CUDA event, and a host thread assembling the rows while the GPU runs the next batch;
  (5) RESIDENT fp16 LINEAR WEIGHTS (option, fp32-regime sizes under the autocast route only): autocast casts every Linear weight
      and bias fp32 -> fp16 on every forward (torch.autocast's cached_cast = ``arg.to(torch.float16)``); the same ``.to`` done ONCE
      at apply() (strides preserved: the weights are installed transposed) hands autocast fp16 operands it leaves alone —
      the same GEMM operands, one cast kernel per weight per process instead of per forward; refused for a non-autocast Scorer
      (fp32 GEMMs would become fp16 GEMMs: a different model).

The wrapper over the forward is likelihood.py's convention (the string as given, forward + reversed string, sum and mean,
.5*(lr+rl)): ``Scorer.score_T``.

    from engines.progen2.kits import v0_score_r3_1 as kit
    rec = kit.apply(model)                                   # tables + the module-global patch; no forward
    scorer = kit.Scorer(model, tokenizer, autocast=True)     # the forward runner + the host pipeline
    rows, meta = scorer.score_T(units)                      # [{unit_key, ok, ll_lr_sum, ll_rl_sum, ll_lr_mean, ll_rl_mean, ll_sum, ll_mean, shape, ...}]
    kit.remove(model)                                        # the stock function restored
"""
from __future__ import annotations

import contextlib
import hashlib
import queue
import threading

import numpy as np
import torch

from engines.progen2.score import route

KIT = "v0_score_r3_1"
ARM = "v0-score"
ROUTE = "likelihood"
COUNTERS = {"calls": 0, "eager": 0, "refusals": 0, "rot_builds": 0, "rot_hits": 0,
            "batches": 0, "post_rows": 0, "d2h": 0, "host_rows": 0, "ew_asserts": 0}
EXPECTED_PER_FORWARD = {"calls": 1, "eager": 1}                          # one eager stock forward per direction
MAX_ROWS_DEFAULT = 64
_state = {"applied": False, "orig": None, "mp": None, "tables": {}, "n_positions": None, "model": None, "fp32_linears": {}, "resident_fp16": False,
          "ew": None, "ew_tables_checked": None}

# v0_ew (the elementwise kit beside this one: gelu / rotary / residual / glue / ln kernels in one prebuilt .so) is composed FIRST.



class KitRefused(RuntimeError):
    pass


def counters_snapshot() -> dict:
    return dict(COUNTERS)


def counters_delta(before: dict) -> dict:
    return {k: COUNTERS[k] - before.get(k, 0) for k in COUNTERS}


def reset_counters() -> None:
    for k in COUNTERS:
        COUNTERS[k] = 0


def assert_counted_path(delta: dict, n_forwards: int) -> None:
    """The counters of n served forwards: n calls, n eager stock forwards (EXPECTED_PER_FORWARD each), 0 refusals."""
    want = {k: v * int(n_forwards) for k, v in EXPECTED_PER_FORWARD.items()}
    want["refusals"] = 0
    bad = {k: (delta.get(k), v) for k, v in want.items() if delta.get(k) != v}
    if bad:
        raise KitRefused(f"{KIT}: counters off the counted path (got, want): {bad}")


def sha256_bytes(a) -> str:
    a = a.detach().cpu().contiguous().numpy() if torch.is_tensor(a) else np.ascontiguousarray(a)
    return hashlib.sha256(a.tobytes()).hexdigest()


# ------------------------------------------------------------------------------------------------- (1) resident rotary tables
def _tables(x, dim: int):
    key = (str(x.device), int(dim))
    ent = _state["tables"].get(key)
    if ent is None:
        n = int(_state["n_positions"])
        sin, cos = _state["orig"](x, 1, seq_len=n)              # the stock function itself, once, for every position of the model
        ent = {"sin": sin, "cos": cos, "n": n, "sin_sha": sha256_bytes(sin), "cos_sha": sha256_bytes(cos)}
        _state["tables"][key] = ent
        COUNTERS["rot_builds"] += 1
    COUNTERS["rot_hits"] += 1
    return ent


def _fixed_pos_embedding_resident(x, seq_dim=1, seq_len=None):
    """modeling_progen.fixed_pos_embedding's contract (L39-44): (sin, cos) of shape [seq_len, dim/2] fp32 on x.device — served as
    slices of the resident tables (a slice is a view: the same bytes the stock computed, no CPU op, no H2D copy)."""
    dim = x.shape[-1]
    if seq_len is None:
        seq_len = x.shape[seq_dim]
    ent = _tables(x, dim)
    if seq_len > ent["n"]:
        COUNTERS["refusals"] += 1
        raise KitRefused(f"{KIT}: seq_len {seq_len} exceeds the resident table ({ent['n']} = n_positions)")
    return ent["sin"][:seq_len], ent["cos"][:seq_len]


def _ew_apply(model, size: str, levers=None) -> dict:
    """v0_ew applied BEFORE the resident tables (it binds ProGenAttention.forward/_attn, ProGenBlock.forward, ProGenMLP.act and the
    LayerNorm forwards; the resident tables bind after it). v0_ew's own apply() names what is untested about the
    box and engages (``notes``: a stack or gelu_new bytes off the tested ones, kernels JIT-compiled from PTX) and raises KitRefused BY
    LEVER NAME for what cannot run (its library missing / not loading / no image for the card): the mode refuses, nothing is composed in
    part. Nothing is hashed against this tree."""
    from engines.progen2.kits import v0_ew as ew
    kw = {"size": size}
    if levers is not None:
        kw["levers"] = tuple(levers)
    rec = ew.apply(model, **kw)
    _state["ew"] = ew
    return {"levers": list(rec["levers"]), "notes": list(rec.get("notes") or []), "ln_variant": rec.get("ln_variant"), "stock_bytes": rec.get("stock_bytes")}


def ew_tables_check() -> dict:
    """After the first forward: v0_ew's hoisted (cos, sin) tables (built lazily by its attention path from the stock function it
    saved at ITS apply) must equal this kit's resident tables byte for byte — the same sha definition (fp32 [n_positions, rd/2]
    contiguous bytes) on both sides; stamped, and refused on a mismatch."""
    ew = _state["ew"]
    if ew is None:
        return {"composed": False}
    theirs = ew.stamp()["rotary_tables"]                      # {"<device>/rd<dim>": {cos_sha256, sin_sha256, rows}}
    mine = {f"{k[0]}/rd{k[1]}": {"cos_sha256": v["cos_sha"], "sin_sha256": v["sin_sha"], "rows": v["n"]} for k, v in _state["tables"].items()}
    if not theirs:
        raise KitRefused(f"{KIT}: v0_ew has built no rotary table yet (its attention path did not run)")
    for key, t in theirs.items():
        m = mine.get(key)
        if m is None or m != t:
            raise KitRefused(f"{KIT}: rotary tables differ between v0_ew and {KIT} at {key}: {t} vs {m}")
    rec = {"composed": True, "equal": True, "tables": theirs}
    _state["ew_tables_checked"] = rec
    return rec


def ew_assert_path(snap: dict, n_forwards: int, autocast: bool) -> None:
    """v0_ew's own counted-path assertion on a counters delta (every patched site counted n_forwards x its expectation)."""
    ew = _state["ew"]
    if ew is not None:
        ew.assert_counted_path(ew.counters_delta(snap), n_forwards=n_forwards, autocast=autocast)
        COUNTERS["ew_asserts"] += 1


def ew_snapshot():
    ew = _state["ew"]
    return ew.counters_snapshot() if ew is not None else None


def apply(model, resident_fp16: bool = False, ew: bool = False, size: str = None, ew_levers=None) -> dict:
    """Bind the resident tables: the module-global fixed_pos_embedding of the stock module (looked up by name inside
    ProGenAttention.forward L185/L192 at every call) is replaced; the tables are built now through the ORIGINAL for every
    (device, dim) of the model. ``resident_fp16``: lever (5) — every fp32 nn.Linear weight/bias cast to fp16 once (.to, strides
    preserved; the originals kept for remove()). No forward. Refuses when already applied or when the function is
    not the stock's."""
    import models.progen.modeling_progen as mp
    if _state["applied"]:
        raise KitRefused(f"{KIT}: already applied")
    ew_rec = None
    if ew:
        if not size:
            raise KitRefused(f"{KIT}: ew=True needs the size (v0_ew.apply(size=...))")
        ew_rec = _ew_apply(model, size, ew_levers)
    f = mp.fixed_pos_embedding
    if getattr(f, "__module__", None) != mp.__name__ or f.__name__ != "fixed_pos_embedding":
        raise KitRefused(f"{KIT}: modeling_progen.fixed_pos_embedding is not the stock function ({f!r})")
    _state.update(orig=f, mp=mp, n_positions=int(model.config.n_positions), model=model, tables={}, fp32_linears={}, resident_fp16=False)
    mp.fixed_pos_embedding = _fixed_pos_embedding_resident
    _state["applied"] = True
    attn = model.transformer.h[0].attn
    dim = attn.rotary_dim if attn.rotary_dim is not None else attn.head_dim
    dev = next(model.parameters()).device
    ent = _tables(torch.zeros((1, 1, 1, dim), device=dev), dim)
    fp16_rec = {"enabled": False, "n_linear_cast": 0}
    if resident_fp16:
        n = 0
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear) and mod.weight.dtype == torch.float32:
                w, b = mod.weight.data, (mod.bias.data if mod.bias is not None else None)
                _state["fp32_linears"][name] = (w, b)
                mod.weight.data = w.to(torch.float16)                           # autocast's own cast (cached_cast: arg.to(fp16)), once
                if b is not None:
                    mod.bias.data = b.to(torch.float16)
                assert mod.weight.data.stride() == w.stride(), (name, mod.weight.data.stride(), w.stride())
                n += 1
        _state["resident_fp16"] = True
        fp16_rec = {"enabled": True, "n_linear_cast": n, "sample_stride": {k: list(model.get_submodule(k).weight.stride()) for k in list(_state["fp32_linears"])[:2]}}
    return {"kit": KIT, "arm": ARM, "rotary_dim": int(dim), "n_positions": _state["n_positions"], "device": str(dev),
            "tables": {f"{k[0]}:{k[1]}": {"n": v["n"], "sin_sha256": v["sin_sha"], "cos_sha256": v["cos_sha"]} for k, v in _state["tables"].items()},
            "patched": {"modeling_progen.fixed_pos_embedding": _fixed_pos_embedding_resident.__name__}, "resident_fp16": fp16_rec,
            "ew": ew_rec, "composition": ("v0_ew first (attention/block/MLP-act/LayerNorm forwards), then the resident tables — inert on the v0_ew attention path "
                                          "(it slices its own hoisted tables from the stock function saved at its apply); equality asserted after the first forward" if ew else None)}


def remove(model=None) -> dict:
    if not _state["applied"]:
        raise KitRefused(f"{KIT}: not applied")
    if _state["ew"] is not None and model is not None:
        _state["ew"].release(model)
        _state["ew"] = None
    mp = _state["mp"]
    if mp.fixed_pos_embedding is not _fixed_pos_embedding_resident:
        raise KitRefused(f"{KIT}: modeling_progen.fixed_pos_embedding was replaced by someone else since apply()")
    mp.fixed_pos_embedding = _state["orig"]
    n_restored = 0
    if _state["resident_fp16"] and model is not None:
        for name, (w, b) in _state["fp32_linears"].items():
            mod = model.get_submodule(name)
            mod.weight.data = w
            if b is not None:
                mod.bias.data = b
            n_restored += 1
    _state.update(applied=False, orig=None, mp=None, tables={}, model=None, fp32_linears={}, resident_fp16=False)
    return {"restored": True, "fixed_pos_embedding_is_stock": mp.fixed_pos_embedding.__module__ == mp.__name__, "n_linear_restored": n_restored}


def rotary_table_check(model, max_len: int = None) -> dict:
    """For every seq_len in [1, max_len]: the stock function's fresh (sin, cos) vs the resident slices — bit-pattern equality counts."""
    from compare.bit_equal import bit_equal
    attn = model.transformer.h[0].attn
    dim = attn.rotary_dim if attn.rotary_dim is not None else attn.head_dim
    n = int(max_len or _state["n_positions"])
    x = torch.zeros((1, 1, 1, dim), device=next(model.parameters()).device, dtype=next(model.parameters()).dtype)
    ok, bad = 0, []
    for L in range(1, n + 1):
        s0, c0 = _state["orig"](x, 1, seq_len=L)
        s1, c1 = _fixed_pos_embedding_resident(x, 1, seq_len=L)
        if bit_equal(s0, s1) and bit_equal(c0, c1):
            ok += 1
        else:
            bad.append(L)
    return {"n": n, "equal": ok, "n_mismatch": len(bad), "mismatch_lengths": bad[:20]}


# ------------------------------------------------------------------------------------------------- (2) the forward runner
class Runner:
    """Serves ``logits = ProGenForCausalLM.forward(input_ids=[B, L], use_cache=False).logits`` (fp32 [B, L, V]) eagerly through the stock
    call. The autocast context of the route (torch.cuda.amp.autocast(enabled=True), likelihood.py L176) is entered per forward with
    cache_enabled=False (every weight is used once per forward: the cast kernels are the same). Under no_grad, as the stock (L175)."""

    def __init__(self, model, *, autocast: bool):
        if not _state["applied"] or _state["model"] is not model:
            raise KitRefused(f"{KIT}: apply(model) first")
        if _state["resident_fp16"] and not autocast:
            raise KitRefused(f"{KIT}: resident fp16 Linear weights serve the autocast route only (a non-autocast forward would run fp16 GEMMs: a different model)")
        self.model, self.autocast = model, bool(autocast)
        self.device = next(model.parameters()).device

    def _ctx(self):
        return torch.cuda.amp.autocast(enabled=True, cache_enabled=False) if self.autocast else contextlib.nullcontext()

    def _forward(self, ids):
        with torch.no_grad(), self._ctx():
            return self.model(input_ids=ids, use_cache=False).logits

    def run(self, ids_host: torch.Tensor):
        """ids_host: a pinned int64 [B, L] CPU tensor. Returns (logits [B, L, V] fp32 device, ids_dev [B, L])."""
        COUNTERS["calls"] += 1
        ids_dev = ids_host.to(self.device, non_blocking=True)
        COUNTERS["eager"] += 1
        snap = ew_snapshot()
        logits = self._forward(ids_dev)
        if snap is not None:                                   # the forward IS the counted v0_ew path
            ew_assert_path(snap, 1, self.autocast)
            if _state["ew_tables_checked"] is None:
                ew_tables_check()
        return logits, ids_dev

    def record(self) -> dict:
        return {"autocast": self.autocast, "route": "eager"}


# ------------------------------------------------------------------------------------------------- (3) rows and batches
rows_for_unit = route.rows_for_unit                     # the route's row grammar (shared with the stock reference's same-shape rows)
bucket_rows = route.bucket_rows


# ------------------------------------------------------------------------------------------------- (4) post + host pipeline
def post_rows(logits, tg_dev, n_keep: int):
    """The route's post-processing on a batch: ``lg = logits[:, :n_keep, 5:30]`` (the shift + terminal drop = the first n_keep
    logit rows; likelihood.py L181-195) and the per-row STOCK cross_entropy sum and mean (the kernels the stock's .item() reads:
    likelihood.py L81-86 on the same [n, 25] strided view). Returns ce [B, 2] fp32 (sum, mean)."""
    B = logits.shape[0]
    lg = logits[:, :n_keep, route.FIRST_TOKEN:route.LAST_TOKEN + 1]
    ce = []
    for b in range(B):
        l_b = lg[b]
        t_b = tg_dev[b]
        ce.append(torch.nn.functional.cross_entropy(input=l_b.view(-1, l_b.size(-1)), target=t_b.view(-1), weight=None, size_average=None, reduce=None, reduction='sum'))
        ce.append(torch.nn.functional.cross_entropy(input=l_b.view(-1, l_b.size(-1)), target=t_b.view(-1), weight=None, size_average=None, reduce=None, reduction='mean'))
    COUNTERS["post_rows"] += B
    return torch.stack(ce).view(B, 2)


class _Set:
    """One slot of the pinned ring: host ids/targets (H2D) and ce (D2H) buffers + the event that frees it."""

    def __init__(self, max_rows: int, max_len: int, pin: bool = True):
        def buf(shape, dtype):
            t = torch.empty(shape, dtype=dtype)
            return t.pin_memory() if pin else t
        self.ids = buf((max_rows, max_len), torch.long)
        self.tg = buf((max_rows, max_len), torch.long)
        self.ce = buf((max_rows, 2), torch.float32)
        self.event = None
        self.meta = None


class _NoEvent:
    """CPU stand-in for torch.cuda.Event (the CPU contract test; the box is CUDA)."""

    def record(self, stream=None):
        pass

    def synchronize(self):
        pass

    def elapsed_time(self, other):
        return float("nan")


class Scorer:
    """The unit pipeline over Runner: rows → exact-length batches → forward → post → pinned D2H → host thread (row assembly)."""

    def __init__(self, model, tokenizer, *, autocast: bool, n_sets: int = 4, max_rows: int = MAX_ROWS_DEFAULT, max_tokens: int = None):
        self.model, self.tokenizer = model, tokenizer
        self.runner = Runner(model, autocast=autocast)
        self.n_positions = int(model.config.n_positions)
        self.max_rows, self.max_tokens, self.n_sets = int(max_rows), max_tokens, int(n_sets)
        self.cuda = self.runner.device.type == "cuda"
        self.sets, self.batch_clock = [], []

    def _event(self, timing: bool = False):
        return torch.cuda.Event(enable_timing=timing) if self.cuda else _NoEvent()

    def _sync(self):
        if self.cuda:
            torch.cuda.synchronize()

    # -- the batch loop (GPU thread)
    def _ensure_sets(self, max_len: int):
        if not self.sets or self.sets[0].ids.shape[1] < max_len:
            self.sets = [_Set(self.max_rows, max_len, pin=self.cuda) for _ in range(self.n_sets)]
            self.free = queue.Queue()
            for s in self.sets:
                self.free.put(s)

    def run_batches(self, batches: list, on_rows) -> dict:
        """Runs every batch; the host thread calls on_rows(list_of_row_results) per batch. Returns the batch clocks."""
        if not batches:
            return {"n_batches": 0}
        max_len = max(b["L"] for b in batches)
        self._ensure_sets(max_len)
        work = queue.Queue(maxsize=self.n_sets)
        errors = []

        def host_loop():
            while True:
                item = work.get()
                if item is None:
                    return
                st, meta = item
                try:
                    st.event.synchronize()
                    B, n = meta["B"], meta["n_keep"]
                    ce = st.ce[:B].numpy().copy()
                    results = []
                    for b, r in enumerate(meta["rows"]):
                        results.append({"unit_key": r["unit_key"], "part": r["part"], "n_keep": n, "L": meta["L"], "B": B,
                                        "ce_sum": float(ce[b, 0]), "ce_mean": float(ce[b, 1])})
                        COUNTERS["host_rows"] += 1
                    on_rows(results)
                except Exception as e:                                # a failed batch = named rows (every row of it), never silence
                    errors.append((meta, f"{type(e).__name__}: {e}"))
                    on_rows([{"unit_key": r["unit_key"], "part": r["part"], "ok": False, "error": f"{type(e).__name__}: {e}"} for r in meta["rows"]])
                finally:
                    self.free.put(st)
        th = threading.Thread(target=host_loop, daemon=True)
        th.start()
        stream = torch.cuda.current_stream() if self.cuda else None
        clocks = []
        try:
            for bi, batch in enumerate(batches):
                rows, L, n_keep, B = batch["rows"], batch["L"], batch["n_keep"], len(batch["rows"])
                st = self.free.get()
                ids_h, tg_h = st.ids[:B, :L], st.tg[:B, :n_keep]
                for b, r in enumerate(rows):
                    ids_h[b].copy_(torch.as_tensor(r["input_ids"], dtype=torch.long))
                    tg_h[b].copy_(torch.as_tensor(r["targets"], dtype=torch.long))
                e0, e1 = self._event(True), self._event(True)
                e0.record(stream)
                logits, ids_dev = self.runner.run(ids_h)
                tg_dev = tg_h.to(self.runner.device, non_blocking=True)
                ce = post_rows(logits, tg_dev, n_keep)
                st.ce[:B].copy_(ce, non_blocking=True)
                COUNTERS["d2h"] += 1
                e1.record(stream)
                st.event = self._event()
                st.event.record(stream)
                st.meta = {"B": B, "L": L, "n_keep": n_keep, "rows": rows}
                work.put((st, st.meta))
                COUNTERS["batches"] += 1
                clocks.append((B, L, e0, e1))
        finally:
            work.put(None)
            th.join()
        self._sync()
        out = {"n_batches": len(batches), "errors": errors,
               "batch_ms": [{"B": B, "L": L, "ms": e0.elapsed_time(e1)} for (B, L, e0, e1) in clocks]}
        self.batch_clock = out["batch_ms"]
        return out

    # -- the wrapper: likelihood.py's convention (units = [{unit_key, seq}] -> (rows, meta); a unit that fails is a named row, never a missing one)
    def score_T(self, units: list) -> tuple:
        rows = []
        failed = []
        for u in units:
            try:
                rows.extend(route.rows_for_unit(self.tokenizer, u))
            except Exception as e:
                failed.append({"unit_key": u["unit_key"], "ok": False, "error": f"{type(e).__name__}: {e}"})
        batches = route.bucket_rows(rows, self.max_rows, self.max_tokens)
        parts = {}

        def on_rows(results):
            for r in results:
                parts.setdefault(r["unit_key"], {})[r["part"]] = r
        clk = self.run_batches(batches, on_rows)
        out = []
        for u in units:
            if u["unit_key"] in [f["unit_key"] for f in failed]:
                continue
            p = parts.get(u["unit_key"], {})
            try:
                out.append(_assemble_T(u, p))
            except Exception as e:
                out.append({"unit_key": u["unit_key"], "ok": False, "error": f"{type(e).__name__}: {e}"})
        return out + failed, {"batches": [{"B": len(b["rows"]), "L": b["L"], "n_keep": b["n_keep"]} for b in batches], "clock": clk}


def _part_fields(p: dict) -> dict:
    """likelihood.py log_likelihood (L85): ll = -cross_entropy, per reduction."""
    return {"ll_sum": -p["ce_sum"], "ll_mean": -p["ce_mean"]}


def _assemble_T(u: dict, p: dict) -> dict:
    for d in ("lr", "rl"):
        if d not in p or p[d].get("ok") is False:
            raise KitRefused(f"direction {d} missing/failed: {p.get(d, {}).get('error')}")
    lr, rl = _part_fields(p["lr"]), _part_fields(p["rl"])
    comb = route.combine_T(lr, rl)
    return {"unit_key": u["unit_key"], "ok": True, "n_tokens": p["lr"]["L"], "n_pos": p["lr"]["n_keep"],
            "ll_lr_sum": lr["ll_sum"], "ll_rl_sum": rl["ll_sum"], "ll_lr_mean": lr["ll_mean"], "ll_rl_mean": rl["ll_mean"], "ll_sum": comb["ll_sum"], "ll_mean": comb["ll_mean"],
            "shape": {"B": p["lr"]["B"], "L": p["lr"]["L"]}, "shape_rl": {"B": p["rl"]["B"], "L": p["rl"]["L"]}}

