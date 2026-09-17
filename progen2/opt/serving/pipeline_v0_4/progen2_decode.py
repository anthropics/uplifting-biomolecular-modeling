"""In-process generation on the stock ProGen2 route (sample.py): the model loaded ONCE, every unit the fresh-process CLI's bytes.

``load``: the stock order — set_env, set_seed(rng_seed, rng_deterministic), the weights by the one-read loader (oneread_loader: the tensors
and dtypes create_model(ckpt, fp16) installs, `--fp16 false` included), `.to(device)`, create_tokenizer_custom — then the decode levers on the
stock module (components.json `decode_kit`: kit_t1 at level t3s = resident rotary tables + static K/V slots; every decode step runs the stock's
eager kernels), sized in LENGTH to the bucket of the max_length asked (256 / 512 / 1024 / 2048, capped at n_positions) and in BATCH to nothing yet.
sample.py's `--sanity` witness (its cross-entropy check for the size) runs once, before the first unit — the stock's own order: sanity, then
sampling — whatever the flag says.
``Handle.unit``: set_seed(rng_seed, rng_deterministic) as sample.py seeds its process, the exact sampler shadowing model.generate for the ONE
stock sample() call (sampler_exact: the same warpers, softmax and multinomial on the same philox stream, the integer bookkeeping fused),
truncate() per completion; the block = the lines sample.py prints (the context, per sample a blank line + index + truncation, `done.`).
Memory: static K/V slots per BATCH SIZE (num_samples) for the work in front of the model — allocated when the caller names its batch sizes
(``Handle.hold``: an --input job's num_samples values, a single call's own) or when a unit first names one — each after ONE fit rule
(static_kv_fits: the component's kv_bytes_per_sample x B against the free device memory less a margin of the card; the idle batch sizes are
released first when it does not fit beside them); a batch size that cannot fit is answered by the named out-of-memory error (OutOfMemory),
nothing allocated, nothing rerouted. A unit longer than the slots re-installs the levers at the next bucket (every slot dropped).
OOM_POLICY (the end of this file): an out-of-memory error is never answered by a fallback.

This module imports nothing but the standard library at import (torch arrives with ``load``); the package (progen2_opt.generate) puts this
directory and the component's on sys.path and drives it; `AUTO_LEVERS` is read from these bytes for the activation line.
"""
from __future__ import annotations

import gc
import hashlib
import inspect
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from _oom import is_oom   # noqa: E402 — the ONE out-of-memory test (OOM_POLICY below)

AUTO_LEVERS = ("oneread_mmap", "sampler_exact")   # the levers of this route beside the decode component's: the one-read mmap loader, the fused-bookkeeping sampler (CUDA route; the package refuses a cpu device by name before anything loads)
KIT_BUCKETS = (256, 512, 1024, 2048)              # the static K/V length the levers are sized for: the smallest bucket >= the max_length asked, capped at n_positions
BLOCK_END = "done.\n"                             # sample.py's closing line: the last line of every block


def bucket(max_length: int, n_positions: int) -> int:
    """max_slots for the decode levers: the smallest KIT_BUCKETS entry >= max_length, capped at the model's n_positions."""
    L = int(max_length)
    for b in KIT_BUCKETS:
        if b >= L:
            return min(b, int(n_positions))
    return int(n_positions)


def components(path: str = os.path.join(HERE, "components.json")) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)



# ----------------------------------------------------------------------------------------------------------------- the fit rule
STATIC_KV_FIT_MARGIN = 0.04   # the share of the card kept free of static K/V for the unit's own activations and the allocator's rounding (1.6 GiB of a 40 GB card, 3.2 of an 80 GB one: a decode unit's transient tensors are far below it)


def static_kv_fits(need_bytes, free_bytes, reserved_unused_bytes, total_bytes, margin=STATIC_KV_FIT_MARGIN):
    """The fit rule of a batch size's static K/V slots: the bytes they need against the device memory free now plus the allocator's cached-but-unused
    bytes, less `margin` of the card. Pure arithmetic (the one rule; kit_fits reads the four numbers off the device)."""
    return float(need_bytes) <= float(free_bytes) + float(reserved_unused_bytes) - float(margin) * float(total_bytes)


def kit_fits(kit, model, B, torch, margin=STATIC_KV_FIT_MARGIN):
    """would the static K/V slots for batch size B fit beside what is held? (static_kv_fits over the component's kv_bytes_per_sample(model) x B and
    the device's free memory); (True, info) when B is held already."""
    per_b = kit.kv_bytes_per_sample(model)
    if kit.allocated(B):
        return True, {"per_sample_bytes": per_b, "already_allocated": True}
    free, total = torch.cuda.mem_get_info()
    reserved_unused = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    need = per_b * int(B) - kit.kv_bytes_held(int(B))   # slots a forward already made for B (the sanity pass at batch 1) are held, not needed again
    return static_kv_fits(need, free, reserved_unused, total, margin), {"per_sample_bytes": per_b, "need_bytes": need, "free_bytes": free, "reserved_unused_bytes": reserved_unused, "total_bytes": total, "margin": margin}


# ----------------------------------------------------------------------------------------------------------------- the loaded model
class Handle:
    """One loaded size: the stock module, the model on its device, the tokenizer, the decode component and its state (the slots held).
    Made by ``load``; ``hold`` claims batch sizes ahead of the units, ``unit`` generates, ``close`` releases."""

    def __init__(self, **kw):
        self.__dict__.update(kw)
        self.units = 0
        self.closed = False

    # ---------------------------------------------------------------------------------------------------------- memory
    def held_batches(self) -> list:
        """the batch sizes whose static K/V slots are in memory now (the component's own count)"""
        return self.kit.allocated_batches()

    def resize(self, max_length: int) -> int:
        """Re-install the component at the bucket of `max_length` (a length beyond the slots): every slot is dropped first; returns max_slots."""
        kit, S = self.kit, self.kit.S
        kit.release(keep=()); S.rot.clear(); S.rot_il.clear()
        gc.collect(); self.torch.cuda.empty_cache()
        kit.install(self.model, max_slots=bucket(int(max_length), self.n_positions), **self.install_kw)
        return S.max_slots

    def ensure_slots(self, B: int, L: int, evict: bool = True):
        """The static K/V slots of batch size B at a length covering L, BEFORE the unit that needs them: a length beyond the slots re-installs the
        component at the next bucket (every slot dropped); a batch size not held is allocated after the fit check
        (kit_fits) — the idle batch sizes released first when it does not fit beside them (`evict`; batch 1, the sanity pass's, stays) — and when it cannot
        fit, NOTHING is allocated: the answer is the named out-of-memory row (OOM_POLICY's grammar). Returns (ok, mem_policy, oom_row)."""
        kit, model, torch, S = self.kit, self.model, self.torch, self.kit.S
        pol = {}
        if int(L) > S.max_slots:
            pol["reinstalled_max_slots"] = self.resize(int(L))
        if kit.allocated(B):
            pol["fit_info"] = {"already_allocated": True, "held_batches": self.held_batches()}
            return True, pol, None
        fits, finfo = kit_fits(kit, model, B, torch)
        if not fits and evict:
            pol["evicted"] = kit.release(keep={1, int(B)})
            fits, finfo = kit_fits(kit, model, B, torch); pol["fits_after_evict"] = fits
        pol["fit_info"] = {k: (round(v / 2**20, 1) if isinstance(v, (int, float)) and k.endswith("bytes") else v) for k, v in finfo.items()}
        where = f"sample N={B} L={L}"
        if not fits:
            msg = (f"the static K/V slots for {B} samples at {S.max_slots} positions need {finfo['need_bytes'] / 2**30:.2f} GiB and the card has "
                   f"{(finfo['free_bytes'] + finfo['reserved_unused_bytes']) / 2**30:.2f} GiB free of {finfo['total_bytes'] / 2**30:.2f} GiB (held: the weights and the slots of batches "
                   f"{batches_word(self.held_batches())}); not allocated")
            return False, pol, oom_row(where, msg, torch=torch, cuda=True, mem_policy=pol)
        t_a = time.monotonic()
        got, row = unit_call(lambda: kit.allocate(model, int(B)), f"static K/V allocation for {where}", torch=torch, cuda=True, mem_policy=pol)
        if row is not None:   # the estimate passed but the allocation itself ran out: the partial slots are released, the named row answers
            row["released_after"] = kit.release(keep={1})
            return False, pol, row
        pol["allocated"] = {"batch": int(B), "max_slots": S.max_slots, "gib": round(kit.kv_bytes_held(int(B)) / 2**30, 3), "seconds": round(time.monotonic() - t_a, 3), "held_batches": self.held_batches()}
        return True, pol, None

    def hold(self, shapes) -> dict:
        """Claim the static K/V slots of the work in front of the model: `shapes` = (num_samples, max_length) per unit to come (an --input job's items,
        a single call's own). The slots' length grows to the longest max_length first (the next bucket); every batch size is then allocated after
        the fit check — one that cannot fit is recorded, not raised: its units are answered by name when they come (OOM_POLICY). Returns the record
        {allocated, not_allocated, alloc_gib, alloc_seconds, held, max_slots}."""
        rec = {"allocated": [], "not_allocated": [], "alloc_gib": 0.0, "alloc_seconds": 0.0}
        shapes = [(int(shp[0]), int(shp[1])) for shp in shapes]
        if shapes and max(L for (_, L) in shapes) > self.kit.S.max_slots:   # a length beyond the slots: the component re-installed at its bucket before any batch size is allocated
            self.resize(max(L for (_, L) in shapes))
        for B in dict.fromkeys(B for (B, _) in shapes):
            ok, pol, row = self.ensure_slots(B, self.kit.S.max_slots, evict=False)
            if ok and pol.get("allocated"):
                rec["allocated"].append(B); rec["alloc_gib"] += pol["allocated"]["gib"]; rec["alloc_seconds"] += pol["allocated"]["seconds"]
            elif not ok:
                rec["not_allocated"].append({"batch": B, "error": row["error"]})
        rec.update(held=self.held_batches(), max_slots=self.kit.S.max_slots, alloc_gib=round(rec["alloc_gib"], 3), alloc_seconds=round(rec["alloc_seconds"], 3))
        self.held = rec
        return rec

    # ---------------------------------------------------------------------------------------------------------- the stock's sanity pass
    def sanity_witness(self) -> dict:
        """sample.py's --sanity section for this size (its `ce` closure and the checkpoint's reference cross-entropy, read from the stock main's own
        source): run once per loaded model, before the first unit; asserts as the stock asserts."""
        if self.sanity["ce"] is not None:
            return self.sanity
        stock, torch, model, tokenizer, device, fp16 = self.stock, self.torch, self.model, self.tokenizer, self.device, self.fp16

        def ce(tokens):  # sample.py L157-167
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=fp16):
                    target = torch.tensor(tokenizer.encode(tokens).ids).to(device)
                    logits = model(target, labels=target).logits
                    logits = logits[:-1, ...]
                    target = target[1:]
                    return stock.cross_entropy(logits=logits, target=target).item()

        self.sanity["ce"] = ce(self.sanity["tokens"]); self.sanity["when"] = "before the first sample unit"
        assert abs(self.sanity["ce"] - self.sanity["target"]) < 0.1, self.sanity
        return self.sanity

    # ---------------------------------------------------------------------------------------------------------- one unit
    def unit(self, context: str, max_length: int, num_samples: int, rng_seed: int, t: float, p: float, rng_deterministic: bool = True) -> dict:
        """ONE stock sample() call = one unit: the block sample.py prints for these flags plus the record (ids = the (num_samples, L) token tensor as
        sampled; sampler = tokens_sha256 / shape / shadow_calls; held_batches; max_slots; mem_policy; the allocator's high-water marks of this
        process). A batch size that cannot fit, or an out-of-memory error inside the call, raises OutOfMemory (the named row; OOM_POLICY)."""
        if self.closed:
            raise RuntimeError("the handle is closed")
        stock, torch, model, tokenizer, device, kit, S = self.stock, self.torch, self.model, self.tokenizer, self.device, self.kit, self.kit.S
        t0 = time.monotonic()
        self.sanity_witness()   # the --sanity witness of sample.py, once, before the first sample unit (the stock's own order within the route)
        B, L = int(num_samples), int(max_length)
        ok, mem_policy, row = self.ensure_slots(B, L)   # memory: the static K/V slots this unit's batch size needs, at a length covering it, BEFORE anything of the unit runs — allocated on first sight after the fit check, or the unit is answered by name
        if not ok:
            raise OutOfMemory(row)
        stock.set_seed(int(rng_seed), deterministic=bool(rng_deterministic))   # sample.py L130: the per-process set_seed(args.rng_seed, args.rng_deterministic), per unit here
        rec = {}
        restore_sampler = _install_sampler(model, self.sampler, rec)   # sampler_exact.generate_exact shadows model.generate for this ONE call
        try:
            _call = lambda: stock.sample(device=device, model=model, tokenizer=tokenizer, context=context, pad_token_id=tokenizer.encode("<|pad|>").ids[0],
                                         num_return_sequences=B, temp=float(t), top_p=float(p), max_length=L)
            completions, oom = unit_call(_call, f"sample N={B} L={L}", torch=torch, cuda=True, mem_policy=mem_policy or None)
            if oom is not None:   # after the named row: the idle static batches are released for the NEXT unit (allocator hygiene; this unit is not retried, the component stays installed)
                oom["released_after"] = kit.release(keep={1})
        finally:
            restore_sampler()
        if completions is None:
            raise OutOfMemory(oom)
        truncations = [stock.truncate(c, terminals=["1", "2"]) for c in completions]
        block = f"{context}\n" + "".join(f"\n{i}\n{tr}\n" for i, tr in enumerate(truncations)) + BLOCK_END
        self.units += 1
        # the sampler record is evidence on every unit, never a silent default — the shadow ran exactly once
        assert rec.get("tokens_sha256") and rec.get("shadow_calls") == 1 and rec.get("sampler") == "exact", f"sampler record missing on a sample unit (the generate shadow did not run): {rec}"
        ids = rec.pop("ids")
        ms = torch.cuda.memory_stats()
        record = {"unit_s": round(time.monotonic() - t0, 3), "sampler": rec, "held_batches": self.held_batches(), "max_slots": S.max_slots, "mem_policy": mem_policy,
                  "kit_stats": kit.stats(), "pid": os.getpid(),
                  "mem_stats": {k: ms.get(k) for k in ("num_alloc_retries", "num_device_alloc", "num_device_free", "reserved_bytes.all.current", "allocated_bytes.all.peak", "reserved_bytes.all.peak", "segment.all.current")}}
        return {"block": block, "ids": ids, "record": record}

    def peak_gib(self):
        """(allocated, reserved) high-water marks of this process's CUDA caching allocator, GiB — the memory of the process that holds the model."""
        return self.torch.cuda.max_memory_allocated() / 2**30, self.torch.cuda.max_memory_reserved() / 2**30

    def close(self) -> dict:
        """Release every slot and restore the stock module's own functions; the model is dropped (the caller's references aside)."""
        if self.closed:
            return {}
        self.closed = True
        r = self.kit.release(keep=())
        mp, orig = self.kit.mp, self.kit._ORIG   # the stock module's own functions back in place
        mp.fixed_pos_embedding = orig["fixed_pos_embedding"]; mp.apply_rotary_pos_emb = orig["apply_rotary_pos_emb"]
        mp.ProGenAttention.forward = orig["attn_forward"]
        self.kit.S.__init__()
        self.model = None
        gc.collect(); self.torch.cuda.empty_cache()
        return r


def batches_word(bs) -> str:
    return ",".join(str(b) for b in bs) if bs else "-"


def load(stock, upstream_name: str, workdir: str, *, fp16: bool = True, device: str = "cuda:0", rng_seed: int = 42, rng_deterministic: bool = True,
         max_length: int = 256) -> Handle:
    """Load `upstream_name` (progen2-<size>) from the stock run directory `workdir` (its `checkpoints/<name>` and `tokenizer.json`) the way sample.py's
    main does — set_env, set_seed, the weights (oneread), the tokenizer — on the CUDA card `device`, then install the decode component sized for
    `max_length` (no batch size allocated yet). `stock` = the imported stock `sample` module. Returns the Handle."""
    import torch
    t0 = time.monotonic()
    stock.set_env()
    stock.set_seed(int(rng_seed), deterministic=bool(rng_deterministic))
    dev = torch.device(device)   # sample.py's --device (L137); a card other than the current one is made current, so the levers' slots and memory readings are that card's
    assert dev.type == "cuda" and torch.cuda.is_available(), f"the decode levers run on a CUDA card (device {device!r})"
    if dev.index is not None and dev.index != torch.cuda.current_device():
        torch.cuda.set_device(dev)
    ckpt = os.path.join(workdir, "checkpoints", upstream_name)
    from oneread_loader import create_model_oneread          # oneread_loader.py beside this file
    model = create_model_oneread(ckpt, fp16=bool(fp16)).to(dev)
    tokenizer = stock.create_tokenizer_custom(file=os.path.join(workdir, "tokenizer.json"))
    t_weights = time.monotonic() - t0
    src = inspect.getsource(stock.main)   # the --sanity witness's inputs: the stock main's own sequence and reference cross-entropy for this size
    seqs = dict(re.findall(r"(x_[a-z0-9]+) = '([^']+)'", src))
    m = re.search(r"'%s': \((x_[a-z0-9]+), ([0-9.]+)\)" % re.escape(upstream_name), src)
    sanity = {"ce": None, "target": float(m.group(2)), "tokens": seqs[m.group(1)], "when": "deferred: runs once before the FIRST sample unit (sample.py's own order: sanity, then sampling)"}
    comp = components()
    dk = comp["decode_kit"]
    comp_dir = os.path.join(HERE, "components", dk["dir"])
    if comp_dir not in sys.path:
        sys.path.insert(0, comp_dir)
    import importlib
    kit = importlib.import_module(dk.get("module") or "kit_t1")
    flags_before = _global_flags(torch)
    n_positions = int(model.config.n_positions)
    max_slots = bucket(int(max_length), n_positions)
    install_kw = dict(level=dk["level"], device=str(dev))
    t_k = time.monotonic()
    info = kit.install(model, max_slots=max_slots, **install_kw)
    torch.cuda.synchronize()
    flags_after = _global_flags(torch)
    residue = {k: (v, flags_after[k]) for k, v in flags_before.items() if flags_after[k] != v}
    assert not residue, f"FLAG RESIDUE from the levers: {residue}"   # no lever sets a global numerics flag: the flags at rest are the stock's own (sample.py set_seed: cudnn.deterministic True, benchmark False)
    levers = tuple(AUTO_LEVERS) + tuple(dk["levers"] if info.get("level") == dk["level"] and kit.S.max_slots == max_slots else ())   # the component's levers count once its install reports the level asked at the length asked
    return Handle(stock=stock, torch=torch, model=model, tokenizer=tokenizer, device=dev, fp16=bool(fp16), upstream_name=upstream_name, kit=kit, sampler="exact",
                  levers=levers, install_kw=install_kw, n_positions=n_positions, sanity=sanity, kit_info=info, kv_per_sample_bytes=kit.kv_bytes_per_sample(model), held=None,
                  load_s=round(time.monotonic() - t0, 3), weights_s=round(t_weights, 3), install_s=round(time.monotonic() - t_k, 3))


def _global_flags(torch) -> dict:
    return {"matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32, "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark, "float32_matmul_precision": torch.get_float32_matmul_precision(), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}


def _install_sampler(model, sampler, rec):
    """Shadow model.generate for ONE stock.sample() call: 'exact' -> sampler_exact.generate_exact; 'stock' -> the stock generate.
    rec gains ids (the raw (B, L) int64 tensor as sampled, on the host), tokens_sha256 (its bytes hashed), shape, sampler and shadow_calls."""
    had_inst = "generate" in model.__dict__; prev_inst = model.__dict__.get("generate")
    orig = model.generate
    if sampler == "exact":
        from sampler_exact import generate_exact

        def gen(input_ids, **kw):
            return generate_exact(model, input_ids, max_length=kw["max_length"], temperature=kw["temperature"], top_p=kw["top_p"],
                                  num_return_sequences=kw.get("num_return_sequences", 1), pad_token_id=kw["pad_token_id"], top_k=kw.get("top_k"))
    else:
        gen = orig

    def shadow(input_ids, **kw):
        rec["shadow_calls"] = rec.get("shadow_calls", 0) + 1
        out = gen(input_ids, **kw)
        host = out.detach().cpu()
        rec["ids"] = host; rec["tokens_sha256"] = hashlib.sha256(host.numpy().tobytes()).hexdigest(); rec["shape"] = list(out.shape); rec["sampler"] = sampler
        return out

    model.generate = shadow

    def restore():
        if had_inst:
            model.generate = prev_inst        # an earlier instance attribute is put back
        else:
            del model.generate                # the class method is visible again
    return restore


OOM_POLICY = ("an out-of-memory error on this route is never answered by a fallback: the unit's ONE model call is not retried, not rerouted to "
              "the stock path, and the levers are never re-installed or disabled because of it; the unit raises OutOfMemory carrying the NAMED row "
              "{'error': 'OUT OF MEMORY at <where>: <message>; no fallback applied ...', 'oom': True} and the caller fails that unit by name")
OOM_ADVICE = "no fallback applied (no evict-and-retry, no stock path, no re-install); use fewer samples per call (--num-samples) or a shorter --max-length"
_OOM_MEM_STATS = ("num_alloc_retries", "allocated_bytes.all.current", "allocated_bytes.all.peak", "reserved_bytes.all.current", "reserved_bytes.all.peak", "segment.all.current")


class OutOfMemory(RuntimeError):
    """The named out-of-memory row of a unit (OOM_POLICY): str() = the row's `error`; `.row` = the row."""

    def __init__(self, row: dict):
        super().__init__(str((row or {}).get("error") or "OUT OF MEMORY"))
        self.row = row or {}


def oom_row(where, exc, *, torch=None, cuda=False, **fields):
    """The NAMED out-of-memory row of a unit (OOM_POLICY): {'error': 'OUT OF MEMORY at <where>: <message>; <OOM_ADVICE>', 'oom': True,
    <fields that are not None>, 'mem_stats' + 'mem_summary_head' on CUDA}. No result key."""
    row = {"error": f"OUT OF MEMORY at {where}: {str(exc)[:300]}; {OOM_ADVICE}", "oom": True}
    row.update({k: v for k, v in fields.items() if v is not None})
    if cuda and torch is not None:
        ms = torch.cuda.memory_stats()
        row["mem_stats"] = {k: ms.get(k) for k in _OOM_MEM_STATS}
        row["mem_summary_head"] = torch.cuda.memory_summary(abbreviated=True)[:1500]
    return row


def unit_call(call, where, *, torch=None, cuda=False, **fields):
    """Run a unit's ONE call. Returns (result, None); or (None, the NAMED out-of-memory row) when the call raises an out-of-memory
    error (is_oom) — the call is never retried and nothing is rerouted (OOM_POLICY). Every other exception propagates unchanged."""
    try:
        return call(), None
    except Exception as e:  # noqa: BLE001 — narrowed on the next line: only an out-of-memory error becomes the named row
        if not is_oom(e):
            raise
        row = oom_row(where, e, torch=torch, cuda=cuda, **fields)
    gc.collect()                                   # the failed call's frames + tensors are unreachable here (the except block has exited): collected, then the allocator's cache emptied
    if cuda and torch is not None:
        torch.cuda.empty_cache()
    return None, row
