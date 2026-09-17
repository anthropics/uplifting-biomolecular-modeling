"""Per-item PHASE timing line — the same boundaries on every single-process route (``pred --mode off``: the stock subprocess, loaded there BY
PATH as a standalone standard-library module, the kit's ``opt/`` staying off ``sys.path``; ``exact`` / ``fast``: in-process, installed by
``cli.cmd_pred`` after activation so the timers sit OUTSIDE every lever's wrapper). It changes no computed value: each site is the callable
found on the class at install time, called unchanged between two CUDA events recorded on the current stream (``torch.cuda.Event(enable_timing=
True)``; no host synchronisation inside the item), read once after the item's ONE ``torch.cuda.synchronize()`` at the end of ``model_run``.

Sites (the pinned stock, ``stock/src`` v1.0.0 992067e; monomer AND multimer classes):
  trunk   = ``AtlasFold(_Multimer).run_trunk`` — s/z init, recycling embedder, AtlasLM + the 4-block LM stack (re-run every recycle with a fresh
            MLM mask), the template module (multimer, inert without templates) and the 48-block Pairformer over every recycle of the item;
  lm      = ``AtlasFold(_Multimer).run_lm_embedder`` — AtlasLM-3B + LM stack, NESTED INSIDE trunk_s (``lm_in=trunk_s`` on the PHASE-NOTE line;
            summed over the recycles) — informational, not one of the three phases;
  sampler = ``DiffusionHead.sample`` — the whole diffusion roll-out of the item, every step of every sample;
  conf    = ``ConfidenceHead_Monomer.forward`` / ``ConfidenceHead_Multimer.forward`` — the per-sample confidence stacks + logits heads (the
            pLDDT / PAE / pTM reductions ``inference`` computes after the head are in fwd_s, outside conf_s);
  fwd     = ``AtlasFold(_Multimer).inference`` — the model forward alone (rel-pos encoding, trunk, distogram head, sampler, confidence,
            metrics); the feature dict's device copy-in before it and the ``.cpu()`` copy-out after it are outside the span;
  total   = ``FoldingRunner.model_run`` / ``MultimerFoldingRunner.model_run`` — copy-in + forward + copy-out of ONE bucketed batch at ONE seed
            (host clock, ``time.perf_counter``, closed by the item's synchronize).
Item = the names of the FASTA records in the bucketed batch joined by ``+`` (``FoldingRunner._iter_batch`` / ``MultimerFoldingRunner._iter_batch``
yield the chunk; B > 1 items share one forward), seed = ``model_run``'s ``seed`` argument.

Lines (stdout, once per successful ``model_run``; nothing on an item that raised):
  ``PHASE item=<names> seed=<seed> lm_s=<f> trunk_s=<f> sampler_s=<f> conf_s=<f> fwd_s=<f> total_s=<f>``
  ``PHASE-NOTE item=<names> seed=<seed> engine=atlasfold bucket=<L> batch=<B> lm_in=trunk_s clock=cuda_events``
  ``PEAK item=<names> alloc_gib=<f> reserved_gib=<f>``  (torch's max_memory_allocated / max_memory_reserved since the item started, GiB = 2^30)
Guard: trunk_s + sampler_s + conf_s <= fwd_s <= total_s by construction (nested spans); a violation prints ``PHASE-NOTE guard …`` — never hidden.
Without a CUDA device the spans read the host clock (``clock=host``) and no PEAK line is printed (``PEAK-NOTE item=<names> cuda=absent``).
This module imports the standard library only; torch is read from ``sys.modules`` at call time (the stock code imports it)."""
import functools
import importlib
import sys
import time
from typing import Optional

PREFIX = "PHASE"
ENGINE = "atlasfold"
MODEL_SITES = (("atlasfold.model.model", "AtlasFold"), ("atlasfold.model.model_multimer", "AtlasFold_Multimer"))
RUNNER_SITES = (("atlasfold.runner", "FoldingRunner"), ("atlasfold.runner_multimer", "MultimerFoldingRunner"))
CONF_SITES = (("atlasfold.model.network.confidence_head", "ConfidenceHead_Monomer"), ("atlasfold.model.network.confidence_head", "ConfidenceHead_Multimer"))
SAMPLER_SITE = ("atlasfold.model.network.diffusion_head", "DiffusionHead", "sample")
PHASES = ("trunk", "sampler", "conf")                                   # the three phases of the family grammar
SPANS = ("lm",) + PHASES + ("fwd",)                                     # every device-timed span of an item
_STATE = {"installed": None, "sites": {}, "item": "unknown", "bucket": None, "batch": None, "events": {s: [] for s in SPANS},
          "host": {s: 0.0 for s in SPANS}, "noted": set(), "lines": 0, "last": None}


def _torch():
    return sys.modules.get("torch")


def _cuda():
    """torch.cuda when torch is imported and a device is present, else None."""
    cuda = getattr(_torch(), "cuda", None)
    try:
        return cuda if cuda is not None and cuda.is_available() else None
    except Exception:  # noqa: BLE001
        return None


def _note_once(key: str, text: str) -> None:
    if key not in _STATE["noted"]:
        _STATE["noted"].add(key)
        print(f"{PREFIX}-NOTE {text}", flush=True)


def _reset() -> None:
    for s in SPANS:
        _STATE["events"][s] = []; _STATE["host"][s] = 0.0


def timed(fn, span: str):
    """``fn`` between two CUDA events on the current stream (host clock without CUDA), appended to the item's ``span`` list (idempotent per span)."""
    if getattr(fn, "_afo_phase", None) == span:
        return fn

    @functools.wraps(fn)
    def wrapper(*a, **k):
        cuda = _cuda()
        if cuda is None:
            t0 = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                _STATE["host"][span] += time.perf_counter() - t0
        ev0 = cuda.Event(enable_timing=True); ev1 = cuda.Event(enable_timing=True)
        ev0.record()
        try:
            return fn(*a, **k)
        finally:
            ev1.record()
            _STATE["events"][span].append((ev0, ev1))
    wrapper._afo_phase = span
    try:
        wrapper.__wrapped_stock__ = getattr(fn, "__wrapped_stock__", fn)
    except Exception:  # noqa: BLE001
        pass
    return wrapper


def _span_seconds(span: str) -> float:
    """Sum of the span's event pairs (ms -> s) — valid after the item's synchronize — plus any host-clock seconds."""
    s = _STATE["host"][span]
    for ev0, ev1 in _STATE["events"][span]:
        try:
            s += ev0.elapsed_time(ev1) / 1000.0
        except Exception as e:  # noqa: BLE001
            _note_once("elapsed:" + span, f"span {span}: event elapsed_time raised {type(e).__name__}: {str(e)[:80]} (counted 0)")
    return s


def item_token(names) -> str:
    """Whitespace-free item token: record names joined by '+', blanks and '=' replaced."""
    toks = [(str(n).strip().replace(" ", "_").replace("=", "_").replace("+", "_") if n is not None else "") or "unnamed" for n in (names or [])]
    return "+".join(toks) if toks else "unknown"


def phase_line(item: str, seed, sec: dict, total: float) -> str:
    """``PHASE item=<item> seed=<seed> lm_s=<f> trunk_s=<f> sampler_s=<f> conf_s=<f> fwd_s=<f> total_s=<f>`` + guard notes."""
    line = (f"{PREFIX} item={item} seed={seed} lm_s={sec['lm']:.3f} " + " ".join(f"{p}_s={sec[p]:.3f}" for p in PHASES)
            + f" fwd_s={sec['fwd']:.3f} total_s={total:.3f}")
    s3 = sum(sec[p] for p in PHASES)
    if s3 > sec["fwd"] * 1.001 + 1e-3:
        line += f"\n{PREFIX}-NOTE guard violated on {item}: trunk_s+sampler_s+conf_s={s3:.3f} > fwd_s={sec['fwd']:.3f}"
    if sec["fwd"] > total * 1.001 + 1e-3:
        line += f"\n{PREFIX}-NOTE guard violated on {item}: fwd_s={sec['fwd']:.3f} > total_s={total:.3f}"
    if sec["lm"] > sec["trunk"] * 1.001 + 1e-3:
        line += f"\n{PREFIX}-NOTE guard violated on {item}: lm_s={sec['lm']:.3f} > trunk_s={sec['trunk']:.3f}"
    return line


def peak_lines(item: str) -> list:
    cuda = _cuda()
    if cuda is None:
        return [f"PEAK-NOTE item={item} cuda=absent"]
    gib = float(2 ** 30)
    return [f"PEAK item={item} alloc_gib={cuda.max_memory_allocated() / gib:.2f} reserved_gib={cuda.max_memory_reserved() / gib:.2f}"]


def timed_total(fn):
    """``model_run(self, feat, seed, ...)``: resets the item's spans and torch's peak counters, runs the call, synchronizes ONCE, prints PHASE /
    PHASE-NOTE / PEAK for the item."""
    if getattr(fn, "_afo_phase", None) == "total":
        return fn

    @functools.wraps(fn)
    def model_run(self, feat, *a, **k):
        seed = k.get("seed", a[0] if a else None)
        _reset()
        cuda = _cuda()
        if cuda is not None:
            try:
                cuda.reset_peak_memory_stats()
            except Exception:  # noqa: BLE001
                pass
        t0 = time.perf_counter()
        out = fn(self, feat, *a, **k)
        if cuda is not None:
            cuda.synchronize()
        total = time.perf_counter() - t0
        sec = {s: _span_seconds(s) for s in SPANS}
        item = _STATE["item"]
        B = _STATE["batch"]
        try:
            B = B if B is not None else (len(feat["seq_mask"]) if hasattr(feat.get("seq_mask", None), "__len__") and getattr(feat["seq_mask"], "ndim", 1) > 1 else 1)
        except Exception:  # noqa: BLE001
            pass
        print(phase_line(item, seed, sec, total), flush=True)
        print(f"{PREFIX}-NOTE item={item} seed={seed} engine={ENGINE} bucket={_STATE['bucket']} batch={B} lm_in=trunk_s clock={'cuda_events' if cuda is not None else 'host'}", flush=True)
        for ln in peak_lines(item):
            print(ln, flush=True)
        _STATE["lines"] += 1
        _STATE["last"] = {"item": item, "seed": seed, "total_s": total, **{f"{s}_s": sec[s] for s in SPANS}}
        return out
    model_run._afo_phase = "total"
    return model_run


def note_batch(chunk, bucket_length) -> None:
    """Record the batch about to be forwarded (record names, bucket, batch size) — the PHASE / PEAK lines of the next ``model_run`` carry it.  The
    ``_iter_batch`` wrapper calls this per yielded chunk; a runner that iterates ``_iter_batch`` with look-ahead (output_overlap's re-stated
    ``fold_iter_batch`` peeks one batch ahead to know the run's last batch) calls it again right before its forwards, so the label is the batch
    being timed, not the one peeked."""
    try:
        _STATE["item"] = item_token([getattr(c, "name", None) or "unnamed" for c in chunk])
        _STATE["bucket"] = int(bucket_length); _STATE["batch"] = len(chunk)
    except Exception:  # noqa: BLE001
        _STATE["item"], _STATE["bucket"], _STATE["batch"] = "unknown", None, None


def batch_recorder(fn):
    """``_iter_batch`` (a generator of (bucket_length, chunk)): records the chunk's record names and bucket before the caller's model_run."""
    if getattr(fn, "_afo_phase", None) == "batch":
        return fn

    @functools.wraps(fn)
    def _iter_batch(*a, **k):                                   # (self, …) for the multimer method, (…) for the monomer @staticmethod: passed through as given
        for bucket_length, chunk in fn(*a, **k):
            note_batch(chunk, bucket_length)
            yield bucket_length, chunk
    _iter_batch._afo_phase = "batch"
    return _iter_batch


def wrap_like(cls, attr: str, wrapper):
    """``wrapper`` dressed as the descriptor kind ``cls.<attr>`` has today: the stock monomer ``FoldingRunner._iter_batch`` is a ``@staticmethod``
    (runner.py L359-363; the multimer one is a plain method, runner_multimer.py L402-406) — re-set as a plain function it would receive ``self``
    as its first argument and the stock body would raise ``TypeError: _iter_batch() takes 2 positional arguments but 3 were given``."""
    for klass in cls.__mro__:
        if attr in vars(klass):
            raw = vars(klass)[attr]
            if isinstance(raw, staticmethod):
                return staticmethod(wrapper)
            if isinstance(raw, classmethod):
                return classmethod(wrapper)
            break
    return wrapper


def install() -> dict:
    """Wrap the sites in place (idempotent). Returns ``{"installed": bool, "sites": {span: [qualname, ...]}, "reason": str|None}``; a stock tree
    that is not importable leaves everything untouched and says so once (``PHASE-NOTE timing not installed: …``)."""
    sites = {}
    try:
        for mod, cls_name in MODEL_SITES:
            cls = getattr(importlib.import_module(mod), cls_name)
            for span, attr in (("fwd", "inference"), ("trunk", "run_trunk"), ("lm", "run_lm_embedder")):
                setattr(cls, attr, timed(getattr(cls, attr), span)); sites.setdefault(span, []).append(f"{mod}.{cls_name}.{attr}")
        mod, cls_name, attr = SAMPLER_SITE
        cls = getattr(importlib.import_module(mod), cls_name)
        setattr(cls, attr, timed(getattr(cls, attr), "sampler")); sites.setdefault("sampler", []).append(f"{mod}.{cls_name}.{attr}")
        for mod, cls_name in CONF_SITES:
            cls = getattr(importlib.import_module(mod), cls_name)
            setattr(cls, "forward", timed(getattr(cls, "forward"), "conf")); sites.setdefault("conf", []).append(f"{mod}.{cls_name}.forward")
        for mod, cls_name in RUNNER_SITES:
            cls = getattr(importlib.import_module(mod), cls_name)
            setattr(cls, "model_run", timed_total(getattr(cls, "model_run"))); sites.setdefault("total", []).append(f"{mod}.{cls_name}.model_run")
            setattr(cls, "_iter_batch", wrap_like(cls, "_iter_batch", batch_recorder(getattr(cls, "_iter_batch")))); sites.setdefault("item", []).append(f"{mod}.{cls_name}._iter_batch")
    except Exception as e:  # noqa: BLE001  (an absent stock tree: the route runs untimed, named)
        _STATE["installed"] = False
        _note_once("install", f"timing not installed: {type(e).__name__}: {e}")
        return {"installed": False, "sites": sites, "reason": f"{type(e).__name__}: {e}"}
    _STATE["installed"] = True; _STATE["sites"] = sites
    return {"installed": True, "sites": sites, "reason": None}


def lines_printed() -> int:
    return int(_STATE["lines"])


def last() -> Optional[dict]:
    """The last item's seconds as printed ({item, seed, total_s, lm_s, trunk_s, sampler_s, conf_s, fwd_s}) or None."""
    return dict(_STATE["last"]) if _STATE["last"] else None
