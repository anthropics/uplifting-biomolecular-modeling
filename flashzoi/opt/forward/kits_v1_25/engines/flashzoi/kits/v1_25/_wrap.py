"""The kit wrapper: ``build`` returns the kit package namespace (KitRunner, describe, ALL, LEVERS, PINS, LEVER_CLASS, ...) parametrised by
the version's composition: ``stack_kwargs`` = the extra FastNCHW flags of the composition (decoder_fused / transformer_fused / relu_epilogue /
skip_fused) and ``pre_stack`` = the model patches the composition applies before the stack (``swap_layernorms`` for ln_fused)."""
from __future__ import annotations

import hashlib
import json
import importlib
import glob
import os
import sys
import threading
import time

import numpy as np
import torch
from torch import nn

from . import _pins as lane; from ._oom import is_oom                      # the kit's own constants / exceptions, and its one out-of-memory test (an OOM is never served by a fallback)

STACK_FLAG_ATTRS = ("crop", "bias_fold", "head", "tower", "decoder_fused", "transformer_fused", "relu_epilogue", "skip_fused", "decoder_fused2")


LEVER_COUNTER_KEYS = {            # SITE_CALLS are named by kernel, never by lever: the explicit lever -> kernel-counter map
    "decoder_fused": ["bn_gelu_nchw"], "decoder_fused2": ["up2_add", "head_gemm_softplus"], "skip_fused": ["bias_to_nchw"],
    "transformer_fused": ["ln_fused_add"], "relu_epilogue": ["relu_epilogue"], "ln_fused": ["ln_fused_add", "ln_fused_modules"],
    "sites": ["fused_site_nhwc", "fused_site_nchw"], "head": ["head_gemm_softplus"],
}
PRECAST_REQUIRED = {"transformer_fused", "relu_epilogue", "ln_fused", "decoder_fused", "decoder_fused2", "skip_fused"}   # fused fp16 kernels on the parameters: precast is a hard dependency
TAG_ONLY_LEVERS = {"crop", "precast"}                  # no kernel of their own (a stack flag / a parameter-dtype cast): the attach tag is the evidence
HEAD_TAG_ONLY_VALUES = {"baddbmm", "conv"}             # the baddbmm / cuDNN-conv head has no site counter; the fused softplus head does
COUNTER_PREFIXES = ("", "fz_", "site_", "capture_", "capture_fz_", "capture_site_")   # run-time + capture-time names as all_counts() emits them


_PROCESS_WARMED = {}       # (device, batch) -> the runner # that paid the warm-up forward in this process (once per process)
_PROCESS_RUNNERS = []      # the apply seconds of every runner constructed in this process, in order


def effective_flags_from(levers, counts: dict, tags: dict, now: dict) -> dict:
    """Per-lever EFFECTIVE flags = the lever's attach-time tag still holds now (the patched object is still the one installed / stack flag
    value) AND the lever's OWN counter moved — looked up by the explicit lever->kernel map under every name all_counts() emits
    (run-time fz_/site_ and capture-time capture_fz_/capture_site_ under the graphed path), never by name substring. Tag-only
    levers (crop, precast, a non-fused head) are effective iff their tag holds; a lever with no map entry is False (no evidence)."""
    graphed = "graph" in levers
    def moved(*keys):
        return any(counts.get(p + k, 0) > 0 for k in keys for p in COUNTER_PREFIXES)
    out = {}
    for lv in levers:
        same = tags.get(lv) == now.get(lv) and tags.get(lv) is not None
        if lv == "graph":
            eager_only = counts.get("graph_replays", 0) == 0 and counts.get("eager_fused_calls", 0) > 0 and counts.get("eager_fused_calls", 0) == counts.get("predict_calls", 0)
            out[lv] = same and (counts.get("graph_replays", 0) > 0 or eager_only)    # every plain call replays (the first captures); a job of variant surfaces only (mouse head, embeddings, features) runs the same kernels eagerly by construction
        elif lv == "stage1":
            out[lv] = same and (counts.get("capture_stage1_calls", 0) > 0 or counts.get("stage1_calls", 0) > 0)   # eager calls count stage1_calls per call; a captured graph counts it once at capture
        elif lv == "pinned":
            out[lv] = same and counts.get("leases", 0) > 0
        elif lv in TAG_ONLY_LEVERS or (lv == "head" and tags.get(lv) in HEAD_TAG_ONLY_VALUES):
            out[lv] = same
        else:
            keys = LEVER_COUNTER_KEYS.get(lv)
            out[lv] = bool(same and keys and moved(*keys))   # no map entry = no evidence = False
    return out


def build(pkg: str, ARM: str, PINS: dict, LEVERS: tuple, LEVER_CLASS: dict, *, stack_kwargs: dict | None = None,
          pre_stack: tuple = ()) -> dict:
    pkg_dir = os.path.dirname(importlib.import_module(pkg).__file__)
    FZ_EXACT_PATH = os.path.join(pkg_dir, "fz_exact.py")
    ALL = frozenset(LEVERS)
    stack_kwargs = dict(stack_kwargs or {})

    def _fz():
        """The kernel module, imported lazily (it imports triton): the tables and the sha can be read on a machine without a GPU."""
        return importlib.import_module(pkg + ".fz_exact")

    def _sha_self() -> str:
        return hashlib.sha256(open(FZ_EXACT_PATH, "rb").read()).hexdigest()

    PINS = dict(PINS)
    PINS["fz_exact_sha256"] = _sha_self()          # this build's own fz_exact.py bytes: the key a class record names (assert_device_class), reported in the activation manifest's kit table

    class _Counted:
        """Counts every call of a pinned path component (no silent fallback): the kernel module's objects are wrapped, never edited —
        ``Stage1.__call__`` runs the kernel once per forward."""

        def __init__(self, inner, counts: dict, key: str):
            self.inner, self.counts, self.key = inner, counts, key

        def __call__(self, *a, **k):
            self.counts[self.key] += 1
            return self.inner(*a, **k)

        def __getattr__(self, name):
            return getattr(self.inner, name)


    class KitRunner:
        """One replicate's stock model + the kit stack; ``predict`` / ``predict_tensor`` are the calls.

        ``counts`` = {predict_calls, graph_replays, stage1_calls, leases, releases}: every call through the kit's path is counted; a call
        that cannot take the kit's path RAISES (no captured graph for the shape, wrong dtype, leased slot reuse); nothing degrades to the
        stock route.
        """

        def __init__(self, model, levers=ALL, device: str = "cuda", batch: int = 1, pool_size: int | None = None, numerics: str = "tf32"):
            """apply() = pins + patch + ONE eager warm-up forward at ``batch`` (the first runner of the process only: the Triton kernels compile or
            load there); no graph capture and no pool allocation inside apply. A CUDA graph is captured per batch shape at that shape's FIRST call
            (one fixed cost per shape, stated on the line) and every call of the shape replays it; a capture the device refuses is named and that
            shape runs the same fused kernels eagerly. Every documented surface of the model is served (serve_surfaces): nothing is refused. The
            numerics class is read back and ONE line prints at the end of apply."""
            self.levers = frozenset(levers)
            self.graphs = {}; self.graph_keys = {}; self.capture_refused = {}; self.capture_s_by_batch = {}; self._graph_pool = None
            self.upstream_served = {}                                                                   # reason -> calls: inputs the kernels are not built for, served by upstream's own method (named once each)
            self.numerics = route_numerics("KitRunner(model)")                                          # the class the user actually runs under, read back: torch's defaults = the pinned class; any other class is NAMED on the line (drift), never refused
            if numerics not in ("tf32",):
                raise lane.PinDrift(f"numerics={numerics!r}: 'tf32' is the kit's one route (the exact route: the stock's TF32 class set inside each call and restored, bitwise to stock)")
            self.numerics_knob = numerics
            self.route_class = ("exact route: the stock's TF32 class set INSIDE each call (cudnn.allow_tf32 + matmul.allow_tf32) and restored — bitwise to stock" if self.numerics["pinned_class"]
                                else f"exact route under a numerics class at rest that is not torch's default ({self.numerics['class']}): unpinned — the head GEMM's TF32 follows cudnn.allow_tf32 at rest inside each call, restored after")
            self.drift = [] if self.numerics["pinned_class"] else [f"numerics at rest {self.numerics['class']}"]   # everything about THIS environment that is off the kit's pins: named on the apply line, never a reason to disengage
            self.class_label = "exact"
            self._flags_at_entry = numerics_flags_snapshot()                                            # no flag residue: read back after apply + after every call
            _pk0 = sys.modules.get(pkg)                                                                # the compile hook + the Triton cache dir read at the FIRST KitRunner —
            if _pk0 is not None and hasattr(_pk0, "jit_start"):                                        # after the model loads by construction, never at import
                self._jit_started_here = _pk0.jit_start("first KitRunner (after the model loads)")
            self._model_dict_before = dict(model.__dict__)                                              # the model's INSTANCE dict before apply (forward / the attach mark / the refused surfaces are added by the kit; restored exactly at close)
            self.arm = ARM
            self._jit_cache_status = getattr(sys.modules.get(pkg), "JIT_CACHE_STATUS", "not stated")     # set by the package's jit_start
            self._pkg = pkg
            self.model = model                                           # the attached model (graph cache pins read it before every replay)
            bad = self.levers - ALL
            if bad:
                raise ValueError(f"unknown levers {sorted(bad)}")
            if "sites" in self.levers and "stage1" not in self.levers:
                raise ValueError("the fused sites stack (FastNCHW) includes the stage-1 kernel: select stage1 with sites")
            needs_precast = self.levers & PRECAST_REQUIRED
            if needs_precast and "precast" not in self.levers:          # without precast the fused transformer path dies at construction on mixed dtypes
                raise ValueError(f"levers {sorted(needs_precast)} run the fused fp16 transformer/decoder kernels on the module parameters and "
                                 f"REQUIRE precast (fz.precast_bf16 -> ACT_DTYPE fp16); without it FastNCHW dies at construction "
                                 f"('mat1 and mat2 must have the same dtype, Half and Float', fz_exact.py transformer_fused_forward) — "
                                 f"select precast with them (the pinned composition always does)")
            _t_apply = time.perf_counter()                             # apply() = PINS + PATCH; no probe forward, no subprocess, no build
            self.device_name = torch.cuda.get_device_name(0) if str(device).startswith("cuda") and torch.cuda.is_available() else str(device)
            self.device_class = assert_device_class(PINS, self.device_name)   # the device's class: pinned | by a class record of the kit | unpinned (named on the line; the levers engage wherever the kernels run)
            if self.device_class.get("unpinned"): self.drift.append("device class")
            self.route_assertion = assert_center_bins_route(model)    # the default 6,144-bin route, asserted at attach
            self.bound_assertion = assert_batch_bound_dims(model)     # the filters the kernels' index arithmetic is written for, asserted at attach
            fz = self.fz = _fz()
            self.model, self.device, self.batch = model, device, batch
            self.hook_ns, self.hook_n = 0.0, 0                             # per-forward guard cost accumulators (seconds, calls)
            self.counts = {"predict_calls": 0, "graph_replays": 0, "stage1_calls": 0, "leases": 0, "releases": 0}
            self._lock = threading.Lock()                 # release() may be called from a consumer's threads
            self.applied = {}
            if "precast" in self.levers:
                self._orig = {"params": {}, "layernorms": {}}                                    # REMOVE: the originals BY REFERENCE (precast assigns NEW fp16 tensors to .data — the fp32 tensors stay untouched = bitwise on restore; no copy at apply)
                for _n, _mod in model.named_modules():
                    if _n in ("human_head", "mouse_head"): continue
                    if isinstance(_mod, (nn.Conv1d, nn.Linear)):
                        self._orig["params"][_n] = (_mod, _mod.weight.data, (_mod.bias.data if _mod.bias is not None else None))
                for _n, _mod in list(model.transformer.named_modules()):
                    for _cn, _child in list(_mod.named_children()):
                        if isinstance(_child, nn.LayerNorm): self._orig["layernorms"][(_n, _cn)] = (_mod, _child)
                self.applied["precast_modules"] = fz.precast_bf16(model)      # despite its name, precast_bf16 casts to ACT_DTYPE = fp16
            self.applied["rotary"] = assert_rotary_pinned(model)                # inv_freq float32 after the precast (buffers excluded); a table that differs from the pinned one is named (drift), not refused
            if self.applied["rotary"].get("drift"): self.drift.append(self.applied["rotary"]["drift"])
            if "stage1" in self.levers:
                for step in pre_stack:                                   # the composition's model patches (swap_layernorms for ln_fused)
                    self.applied[step] = getattr(fz, step)(model)
                self.stack = fz.FastNCHW(model, PINS["stage1"][0], tuple(PINS["stage1"][1]),
                                         crop=PINS["crop"] if "crop" in self.levers else False,
                                         bias_fold="sites" in self.levers, head=PINS["head"] if "head" in self.levers else "conv",
                                         stage1_order=PINS["stage1_order"], tower=PINS["tower"], **stack_kwargs)
                self.stack.st = _Counted(self.stack.st, self.counts, "stage1_calls")     # FastNCHW calls self.st(idx) once per forward
                self.fn = self.stack
            else:
                self.stack = None
                self.fn = lambda xt: fz.fwd(model, xt)
            self.graph = None
            self.pool = None
            self.leased = {}
            self.want = torch.float16
            example = torch.zeros((batch, 4, lane.SEQ_LEN), dtype=torch.float32, device=device)
            example[:, 0, :] = 1.0
            self.capture = None
            self.apply_s = {"patch_s": time.perf_counter() - _t_apply, "capture_s": 0.0, "jit_started_here": getattr(self, "_jit_started_here", None)}   # patch (pins + casts + module patches + the stack) vs the graph capture, per runner
            _t_cap = time.perf_counter()
            with torch.inference_mode():
                self._eager_fn = self.fn                                       # the fused stack itself (same kernels, no graph): the variant surfaces and a shape whose capture was refused run it
                self.apply_s["warmup_forward_s"] = 0.0
                _wkey = (str(device), int(batch))
                if _wkey in _PROCESS_WARMED:                                    # the warm-up forward runs on the FIRST runner in the process only
                    self.apply_s["warmup_skipped"] = f"done by runner #{_PROCESS_WARMED[_wkey]} in this process (the JIT cache + kernels are process-wide; this runner's own first-touch is paid by its first call)"
                else:                                                          # the ONE warm-up forward (JIT compile / cache load + first-touch) at the runner's batch
                    _t_w = time.perf_counter()
                    _bw = min(int(batch), KIT_MAX_BATCH)                       # one dispatch: at most KIT_MAX_BATCH windows (the stage-1 store's int32 offsets); a larger runner batch is served in chunks of it per call
                    xw = torch.zeros((_bw, 4, lane.SEQ_LEN), dtype=torch.float32, device=device); xw[:, 0, :] = 1.0
                    self._eager_fn(xw); del xw
                    if str(device).startswith("cuda"):
                        torch.cuda.synchronize()
                    self.apply_s["warmup_forward_s"] = time.perf_counter() - _t_w; self.apply_s["warmup_batch"] = _bw
                    _PROCESS_WARMED[_wkey] = len(_PROCESS_RUNNERS) + 1
                if "graph" in self.levers:
                    self.capture = {}                                          # the CAPTURE-TIME counts (structural evidence the kit's path is inside the graph), summed over captures
                    self._graph_pool = torch.cuda.graph_pool_handle()          # ONE pool shared by every batch's graph
                    self.counts.setdefault("eager_fused_calls", 0); self.counts.setdefault("lazy_captures", 0)
                    self.fn = _GraphDispatch(self, fz, device)                 # capture at a batch shape's first call, replay from then on
                    self.graph = self.graphs.get(self.batch)                   # None until the first call at the runner's batch
                # no probe replay inside apply: the first real call exercises the path. The eager (no-graph) composition runs its
                # first forward on the first call.
            if torch.cuda.is_available() and str(device).startswith("cuda"):
                torch.cuda.synchronize()
            _blk = time.perf_counter() - _t_cap
            self.apply_s["capture_s"] = float(sum(self.capture_s_by_batch.values()))            # 0 at apply (captures are lazy)
            self.apply_s["patch_s"] += max(0.0, _blk - self.apply_s["capture_s"] - self.apply_s["warmup_forward_s"])
            self.apply_s["apply_s"] = self.apply_s["patch_s"] + self.apply_s["capture_s"]
            for k in self.counts:
                self.counts[k] = 0                  # construction-time calls (the warm-up forward) are not rows
            self.served_surfaces = serve_surfaces(model, runner=self)                  # forward (every argument form) + get_embs_after_crop through the kit; predict / predict_gene_count / set_track_subset run as shipped on top; .to()/.half()… detach first
            self._helper_route = route_documented_helper()                            # the package's predict_tracks over kit-attached models -> the pinned path (exact by construction)
            if hasattr(self, "stack") and hasattr(self.stack, "counters"):
                self.stack.counters.clear()
            if hasattr(fz, "SITE_CALLS"):
                fz.SITE_CALLS.clear()
            if "pinned" in self.levers:                # the output dtype/shape pin is asserted on every predict_tensor (no probe forward in apply)
                n = pool_size or POOL_SLOTS_DEFAULT
                numel = 1
                for d in lane.OUTPUT_SHAPE: numel *= int(d)
                self._pool_n = n
                self.pool = None                                              # LAZY: allocated on the FIRST LEASE (predict()), never inside apply
                self.apply_s["pool_s"] = 0.0; self.apply_s["pool_lazy"] = True
                self.apply_s["pool_slots"] = n; self.apply_s["pool_bytes_pinned"] = int(n * numel * 4)
                self.applied["pinned_slots"] = n
            assert_no_flag_residue(self._flags_at_entry, "apply()")                                   # a lever that changed a process-global flag at rest = PinDrift
            self.apply_s["flags_at_rest"] = "unchanged (read back == snapshot at entry)"
            self.apply_s["captures"] = {str(b): round(s, 4) for b, s in self.capture_s_by_batch.items()}
            self.apply_s["numerics_class"] = self.numerics["class"]; self.apply_s["numerics_knob"] = self.numerics_knob; self.apply_s["exact"] = (self.numerics_knob == "tf32")
            _PROCESS_RUNNERS.append(self.apply_s["patch_s"] + self.apply_s.get("warmup_forward_s", 0.0) + self.apply_s["capture_s"])   # cumulative apply seconds in this process
            self.apply_s["runner_in_process"] = len(_PROCESS_RUNNERS); self.apply_s["process_paid_s"] = float(sum(_PROCESS_RUNNERS))
            self.apply_s["helper_route"] = getattr(self, "_helper_route", "not routed")
            self.apply_s["jit_cache"] = getattr(sys.modules.get(pkg), "JIT_CACHE_STATUS", self._jit_cache_status)   # worded by jit_start (dir, files at attach, hook)
            self.apply_s["jit_cache_dir"] = getattr(sys.modules.get(pkg), "JIT_CACHE_DIR", None)      # the effective Triton cache dir
            _imp = getattr(sys.modules.get(pkg), "IMPORT_WALL", None) or {}
            self.apply_s["import_s"] = float(_imp.get("import_s", 0.0) or 0.0)
            self.apply_s["apply_s"] = float(self.apply_s.get("patch_s", 0.0)) + float(self.apply_s.get("warmup_forward_s", 0.0)) + float(self.apply_s.get("capture_s", 0.0))
            self.apply_s["warmup_total_s"] = self.apply_s["apply_s"] + (float(self.apply_s.get("pool_s", 0.0) or 0.0) if not self.apply_s.get("pool_lazy") else 0.0)   # the fixed cost paid at apply
            self.apply_line = compose_apply_line(self)                                                  # ONE line, printed once per runner
            print(self.apply_line, flush=True)
            self.attach_tags = self._tag_snapshot()                   # the attach-time lever tags
            self.attach_effective = effective_flags_from(self.levers, self.all_counts(), self.attach_tags, self.attach_tags)

        def _tag_snapshot(self) -> dict:
            """The attach-time tag of every lever: the patched forward object, the stack's flag values, the LN modules."""
            st = self.stack
            snap = {"graph": id(self.fn) if "graph" in self.levers else None, "pinned": (id(self.pool) if getattr(self, "pool", None) is not None else ("lazy" if "pinned" in self.levers else None)),
                    "precast": self.applied.get("precast_modules"), "stage1": id(st.st) if st is not None and hasattr(st, "st") else None}
            if st is not None:
                snap["sites"] = (bool(getattr(st, "bias_fold", None)), getattr(st, "tower", None))
                snap["crop"] = getattr(st, "crop", None); snap["head"] = getattr(st, "head", None)
                for a in STACK_FLAG_ATTRS[4:]:
                    snap[a] = bool(getattr(st, a, False))
                snap["ln_fused"] = self.applied.get("swap_layernorms")
            return snap

        def effective_flags(self) -> dict:
            """Per-lever effective flags NOW (attach tag still holds + counter evidence) — stamped on every row."""
            return effective_flags_from(self.levers, self.all_counts(), self.attach_tags, self._tag_snapshot())

        def check_effective(self, when: str = "close") -> dict:
            """The second check (at close): every stamped lever must still be effective, else FAIL LOUD (KitPathRefused)."""
            eff = self.effective_flags()
            bad = sorted(k for k, v in eff.items() if not v)
            if bad:
                raise lane.KitPathRefused(f"kit {ARM} at {when}: stamped levers NOT effective {bad} (attach tags {self.attach_tags} vs now {self._tag_snapshot()}; counts {self.all_counts()})")
            return eff

        @property
        def hook_us(self) -> float | None:
            """Mean per-forward cost of the guards (_predict_tensor_checks + assert_graph_cache_pins), microseconds."""
            return (self.hook_ns / self.hook_n * 1e6) if self.hook_n else None

        def _ensure_pool(self) -> None:
            """The pinned lease pool, allocated on the first lease (OUTSIDE apply): n x (pinned host 187 MB + device 187 MB)."""
            fz = self.fz; n = self._pool_n
            _t_pool = time.perf_counter()
            template = torch.empty(lane.OUTPUT_SHAPE, dtype=torch.float32, device=self.device)
            self.pool = fz.OutputLeasePool(template, n=n)
            del template
            self.apply_s["pool_s"] = time.perf_counter() - _t_pool; self.apply_s["pool_first_lease"] = True
            self.applied["pinned_slots"] = n; self.attach_tags["pinned"] = id(self.pool)

        def close(self) -> dict:
            """The documented REMOVE for THIS runner: restore the model to its pre-apply state — forward()/the attach mark removed, the
            transformer's LayerNorm modules + the precast parameters restored BY REFERENCE (the original tensors/modules, never copied), the
            stack / graphs / pinned pool released; the process-level parts (the helper's __code__, the JIT hook) are restored by
            kit.remove(runners) once. Returns the check dict (every item True or a FAIL reason)."""
            model = self.model; out = {}
            try:
                before = getattr(self, "_model_dict_before", None)
                kit_attrs = [k_ for k_, v_ in list(model.__dict__.items()) if k_ in KIT_INSTANCE_ATTRS or getattr(v_, "_kit_move_wrapper", False)]
                for k_ in kit_attrs: del model.__dict__[k_]                        # ONLY what the kit put on the instance (forward, get_embs_after_crop, the attach mark, the move wrappers); what upstream's own code added since (set_track_subset's human_head_bak) stays
                if before is not None:                                              # an instance attribute the kit REPLACED is restored by reference
                    for k_, v_ in before.items():
                        if k_ in KIT_INSTANCE_ATTRS and model.__dict__.get(k_, None) is not v_: model.__dict__[k_] = v_
                added = [k_ for k_, v_ in model.__dict__.items() if (k_ in KIT_INSTANCE_ATTRS and (before is None or model.__dict__.get(k_) is not before.get(k_))) or getattr(v_, "_kit_move_wrapper", False)]
                out["forward_restored"] = "forward" not in model.__dict__ and getattr(model, "_flashzoi_kit_runner", None) is None and not added
                out["instance_attrs_added_by_kit_remaining"] = added
                o = getattr(self, "_orig", {}) or {}
                for (_n, _cn), (_mod, _child) in (o.get("layernorms") or {}).items(): setattr(_mod, _cn, _child)
                out["layernorms_restored"] = all(isinstance(getattr(_mod, _cn), nn.LayerNorm) and getattr(_mod, _cn) is _child for (_n, _cn), (_mod, _child) in (o.get("layernorms") or {}).items())
                for _n, (_mod, _wd, _bd) in (o.get("params") or {}).items():
                    _mod.weight.data = _wd
                    if _bd is not None: _mod.bias.data = _bd
                def _same(a_, b_): return a_.data_ptr() == b_.data_ptr() and a_.dtype == b_.dtype and a_.shape == b_.shape   # `.data` returns a fresh alias each access: the same storage = the same tensor
                out["params_restored"] = all(_same(_mod.weight.data, _wd) and (_bd is None or _same(_mod.bias.data, _bd)) for _n, (_mod, _wd, _bd) in (o.get("params") or {}).items())
                out["params_dtype"] = sorted({str(_wd.dtype) for _n, (_mod, _wd, _bd) in (o.get("params") or {}).items()})
            except Exception as e:                                                  # noqa: BLE001 — a FAIL word, never silent
                out["FAIL"] = f"restore raised {type(e).__name__}: {str(e)[:200]}"
            self.stack = None; self.fn = None; self.graphs = {}; self._graph_pool = None
            self.pool = None; self.leased = {}; self.closed = True
            out["stack_graphs_pool_released"] = self.stack is None and not self.graphs and self.pool is None
            try:
                out["flags_at_rest"] = "unchanged" if numerics_flags_snapshot() == self._flags_at_entry else "FAIL: flags differ from the snapshot at apply"
            except Exception as e:                                                  # noqa: BLE001
                out["flags_at_rest"] = f"FAIL: {type(e).__name__}"
            return out

        def all_counts(self) -> dict:
            """The runner's counters + the kernel module's own (FastNCHW.counters, SITE_CALLS) in one dict, plus the capture-time
            counts under ``capture_*`` (graphed path: the per-replay evidence is structural, see ``capture``)."""
            d = dict(self.counts)
            if self.capture:
                d.update({f"capture_{k}": int(v) for k, v in self.capture.items()})
            st = getattr(self, "stack", None)
            if st is not None and hasattr(st, "counters"):
                d.update({f"fz_{k}": int(v) for k, v in st.counters.items()})
            if hasattr(self.fz, "SITE_CALLS"):
                d.update({f"fz_{k}": int(v) for k, v in self.fz.SITE_CALLS.items()})
            return d

        def release(self, arr: np.ndarray) -> None:
            """Return a leased pinned slot (idempotent for arrays that are not leased views)."""
            with self._lock:
                k = self.leased.pop(id(arr), None)
                if k is not None:
                    self.pool.release(k)
                    self.counts["releases"] += 1

        def jit_after_job(self) -> dict:
            """The compile counter after the LAST forward of a job (the package's log of Triton compile() calls in this process)."""
            m_ = sys.modules.get(getattr(self, "_pkg", ""))
            return m_.jit_after_job() if m_ is not None and hasattr(m_, "jit_after_job") else {"compile_calls": None, "verdict": "no package compile log"}

        def predict_tensor(self, xt):
            """The plain forward(x, is_human=True) on the attached model, graph-dispatched: ``xt`` = a (B, 4, 524288) tensor (moved + cast to float32
            on the runner's device here); returns the fp32 (B, rows, 6144) output of the model's CURRENT human head (7,611 rows as shipped, or the
            subset upstream's set_track_subset installed) as a FRESH device tensor (cloned out of the graph's static buffer: the caller owns it)."""
            _flags_before = numerics_flags_snapshot()                    # no flag residue: the call leaves every flag as it FOUND it (a caller's own autocast scope is the caller's)
            _t_hook = time.perf_counter()
            _predict_tensor_checks(self, xt)
            assert_graph_cache_pins(self)                                              # before every replay: no cache moved/reallocated since capture (cached refs)
            self.hook_ns += time.perf_counter() - _t_hook; self.hook_n += 1             # hook_us = the guard's per-forward cost
            xt = xt.to(device=self.device, dtype=torch.float32)
            if xt.shape[0] > KIT_MAX_BATCH:                                  # a batch above what one dispatch serves (upstream's own maximum at this length) is DISPATCHED IN CHUNKS
                self.counts["chunked_calls"] = self.counts.get("chunked_calls", 0) + 1     # through the same kit path — never an illegal access, never a refusal
                outs = []
                for piece in torch.split(xt, KIT_MAX_BATCH, dim=0):
                    self.counts["chunks_dispatched"] = self.counts.get("chunks_dispatched", 0) + 1
                    outs.append(self.predict_tensor(piece))
                res = torch.cat(outs, dim=0)
                assert_no_flag_residue(_flags_before, "predict_tensor() [chunked]")
                return res
            self.counts["predict_calls"] += 1; self.counts["predict_device_calls"] = self.counts.get("predict_device_calls", 0) + 1   # a device-side output (no lease)
            _tf = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
            if self.numerics_knob == "tf32":                                  # the exact route, for THIS call only (no residue): the head GEMM's TF32 = the class stock's cuDNN conv head runs under = cudnn.allow_tf32 at rest
                torch.backends.cuda.matmul.allow_tf32 = bool(_tf[0])           # torch's defaults at rest (cudnn TF32 on): (True, True) — the pinned class; cudnn TF32 off at rest: strict fp32 on both arms (drift, named at apply)
            try:
                with torch.inference_mode():
                    out = self.fn(xt)
            finally:
                if self.numerics_knob == "tf32":
                    torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = _tf
            rows = int(self.model.human_head.weight.shape[0])                # the model's current human head (upstream's set_track_subset may have replaced it)
            if out.dtype != torch.float32 or tuple(out.shape[1:]) != (rows, lane.OUTPUT_SHAPE[1]):
                raise lane.PinDrift(f"kit output {tuple(out.shape)} {out.dtype} != (B, {rows}, {lane.OUTPUT_SHAPE[1]}) float32")
            res = out.clone()                                         # ownership: a fresh tensor per call
            assert_no_flag_residue(_flags_before, "predict_tensor()")         # after vs before THE CALL: nothing global moved (the extended route calls under its own autocast scope)
            return res

        def _eager_call(self, xt, **kw):
            """One eager pass of the fused stack (no graph) under the same in-call TF32 pair and no-residue rule as predict_tensor — the variant
            surfaces (mouse head, the data_parallel_training form, return_embeddings, get_embs_after_crop) run the same kernels without capture."""
            _flags_before = numerics_flags_snapshot()
            xt = xt.to(device=self.device, dtype=torch.float32)
            self.counts["predict_calls"] += 1; self.counts["eager_fused_calls"] = self.counts.get("eager_fused_calls", 0) + 1
            _tf = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
            torch.backends.cuda.matmul.allow_tf32 = bool(_tf[0])                # the head GEMM's TF32 = the class the stock's cuDNN conv head runs under, for this call only
            try:
                with torch.inference_mode():
                    out = self._eager_fn(xt, **kw)
            finally:
                torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = _tf
            res = tuple(t.clone() for t in out) if isinstance(out, tuple) else out.clone()
            assert_no_flag_residue(_flags_before, "forward() [eager surface]")
            return res

        def _chunks(self, x, call):
            """A batch above what one dispatch serves goes through ``call`` in chunks of KIT_MAX_BATCH (tuples concatenated per member)."""
            if int(x.shape[0]) <= KIT_MAX_BATCH:
                return call(x)
            self.counts["chunked_calls"] = self.counts.get("chunked_calls", 0) + 1
            outs = []
            for piece in torch.split(x, KIT_MAX_BATCH, dim=0):
                self.counts["chunks_dispatched"] = self.counts.get("chunks_dispatched", 0) + 1
                outs.append(call(piece))
            if isinstance(outs[0], tuple):
                return tuple(torch.cat([o[i] for o in outs], dim=0) for i in range(len(outs[0])))
            return torch.cat(outs, dim=0)

        def not_served(self, x):
            """None when the kit's kernels serve this input; else the reason upstream's own method serves it (a shape they are not built for)."""
            if not torch.is_tensor(x) or x.ndim != 3 or tuple(x.shape[1:]) != (4, lane.SEQ_LEN):
                return f"input {tuple(getattr(x, 'shape', ()))} is not (B, 4, {lane.SEQ_LEN})"
            if int(x.shape[0]) < 1:
                return "empty batch"
            return None

        def _upstream(self, name: str, why: str):
            """Upstream's own ``name`` (the class's method as shipped, before any hook) bound to this model, for an input the kernels do not
            serve — said once per reason on one line, counted in upstream_served."""
            import types
            cls_fn = getattr(type(self.model), name); cls_fn = getattr(cls_fn, "__wrapped__", cls_fn)
            key = f"{name}: {why}"
            if key not in self.upstream_served:
                print(f"[flashzoi kit {ARM}] {name}(): {why} — upstream's own {name} serves this call (the kit's kernels are built for (B, 4, {lane.SEQ_LEN}) windows)", flush=True)
            self.upstream_served[key] = self.upstream_served.get(key, 0) + 1
            return types.MethodType(cls_fn, self.model)

        def forward_any(self, x, is_human=True, data_parallel_training=False, return_embeddings=False):
            """upstream's Borzoi.forward in every argument form, through the kit: the plain call (human head, no embeddings) is the graph-dispatched
            path (predict_tensor); the mouse head, the data_parallel_training form and return_embeddings run the same fused kernels eagerly with
            upstream's own head arithmetic on the kit's features. An input the kernels are not built for is served by upstream's forward, named."""
            why = self.not_served(x)
            if why is not None:
                return self._upstream("forward", why)(x, is_human=is_human, data_parallel_training=data_parallel_training, return_embeddings=return_embeddings)
            m = self.model
            if is_human and not data_parallel_training and not return_embeddings:
                return self.predict_tensor(x)
            head = m.human_head if is_human else m.mouse_head                                  # AttributeError exactly as upstream when the checkpoint has no such head
            aux = (m.mouse_head if is_human else m.human_head) if data_parallel_training else None
            return self._chunks(x, lambda piece: self._eager_call(piece, head_module=head, aux_head=aux, with_embeddings=bool(return_embeddings)))

        def embs_after_crop(self, x):
            """upstream's get_embs_after_crop(x) through the kit: the fused trunk up to the crop, the (B, 1536, 6144) features returned."""
            why = self.not_served(x)
            if why is not None:
                return self._upstream("get_embs_after_crop", why)(x)
            return self._chunks(x, lambda piece: self._eager_call(piece, upto="crop"))

        def detach(self, reason: str) -> dict:
            """Take the kit off this model — the documented remove for one runner (upstream's modules, parameters and methods restored by reference;
            graphs and pools dropped) — when the caller moves or re-types the model (.to / .cuda / .cpu / .half / …): upstream serves the model from
            then on, and the kit re-attaches at its next forward if it is a CUDA fp32 model again. One line names it."""
            out = self.close()
            self.detached = reason
            print(f"[flashzoi kit {ARM}] detached from model ({reason}): upstream's modules and methods restored by reference; the kit re-attaches at the model's next forward on a CUDA device", flush=True)
            return out

        def predict(self, x: np.ndarray, device: str | None = None, autocast_dtype: str = lane.AUTOCAST_DTYPE) -> tuple[list[np.ndarray], dict]:
            if autocast_dtype != lane.AUTOCAST_DTYPE:
                raise lane.PinDrift("the kit runs the stock autocast dtype only")
            if x.ndim != 3 or x.shape[1:] != (4, lane.SEQ_LEN) or x.dtype != np.uint8:
                raise ValueError(f"input {x.shape} {x.dtype} != (B, 4, {lane.SEQ_LEN}) uint8")
            if x.shape[0] > KIT_MAX_BATCH:                                             # one dispatch serves at most KIT_MAX_BATCH windows (the stage-1 store's int32 offsets): refused by name before any launch
                raise ValueError(f"predict(): a batch of {x.shape[0]} windows > {KIT_MAX_BATCH} per dispatch — forward() / predict_tensor() serve larger batches in chunks of {KIT_MAX_BATCH}; the leased host route serves at most {KIT_MAX_BATCH}")
            if self.graph is not None and not hasattr(self, "graphs") and x.shape[0] != self.batch:
                raise ValueError(f"graph captured at batch {self.batch}, got {x.shape[0]}")
            assert_graph_cache_pins(self)                                              # before every replay: no cache moved/reallocated since capture
            xt = torch.from_numpy(np.ascontiguousarray(x)).to(device=self.device, dtype=torch.float32)
            torch.cuda.synchronize()
            self.counts["predict_calls"] += 1; self.counts["predict_host_calls"] = self.counts.get("predict_host_calls", 0) + 1   # the leased host route (predict); predict_tensor never leases
            t0 = time.perf_counter()
            with torch.inference_mode():
                out = self.fn(xt)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            if out.dtype != torch.float32:
                raise lane.PinDrift(f"kit output dtype {out.dtype} != float32")
            outs = []
            if "pinned" in self.levers and self.pool is None:
                self._ensure_pool()                                                   # the lazy pool: its cost lands on the first lease, stamped (pool_s / pool_first_lease)
            if self.pool is not None:
                slots = [self.pool.lease(out[i]) for i in range(out.shape[0])]       # ownership inside clock 1
                self.counts["leases"] += len(slots)
                for k in slots:
                    a = self.pool.pinned(k).numpy()                                 # waits for that slot's D2H event
                    self.leased[id(a)] = k
                    outs.append(a)
            else:
                outs = [lane.canonical(out[i]) for i in range(out.shape[0])]      # the kit's own canonical (_pins)
            t2 = time.perf_counter()
            for a in outs:
                if a.shape != lane.OUTPUT_SHAPE or a.dtype != np.float32 or not a.flags["C_CONTIGUOUS"]:
                    raise lane.PinDrift(f"kit output {a.shape} {a.dtype} contiguous={a.flags['C_CONTIGUOUS']}")
            return outs, {"forward_only": t1 - t0, "in_process": t2 - t0}


    def describe(levers=ALL) -> dict:
        """Per-lever class stamps (LEVER_CLASS) for the levers named, the kit version (ARM) and the pins."""
        levers = sorted(levers)
        return {"arm": ARM, "levers": levers, "per_lever": {lv: LEVER_CLASS[lv] for lv in levers}, "pins": dict(PINS)}

    KIT_DIR = pkg_dir
    WRAPPER_FILE = os.path.join(pkg_dir, "_wrap.py") if os.path.isfile(os.path.join(pkg_dir, "_wrap.py")) else os.path.abspath(__file__)

    return {k: v for k, v in locals().items() if k in ("KitRunner", "describe", "ALL", "LEVERS", "PINS", "LEVER_CLASS",
                                                         "FZ_EXACT_PATH", "ARM", "_Counted", "_fz", "_sha_self", "KIT_DIR", "WRAPPER_FILE")}


def assert_center_bins_route(model) -> dict:
    """Attach-time assertion (OUTPUT PARITY): the kit's surface is forward(is_human=True) under the documented default
    route — model.config.return_center_bins_only is True and config.bins_to_return == 6144 == model.crop.target_length; the
    16,384-bin config route (return_center_bins_only=False: TargetLengthCrop(16384 - 32)) REFUSES the kit loudly (PinDrift naming the
    flag) — never fewer bins silently. Objects without a config (tests' fakes) pass through."""
    cfg = getattr(model, "config", None)
    if cfg is None:
        return {"checked": False}
    flag = getattr(cfg, "return_center_bins_only", None)
    if not bool(flag):
        raise lane.PinDrift("kit on a model with config.return_center_bins_only=False: the 16,384-bin route (crop 16352) is the stock's only — the cropped "
                            "decoder (crop='aligned') is an approved intermediate ONLY under the documented default (return_center_bins_only=True)")
    bins = getattr(cfg, "bins_to_return", None)
    crop = getattr(getattr(model, "crop", None), "target_length", None)
    if bins != 6144 or (crop is not None and crop != 6144):
        raise lane.PinDrift(f"kit on a model with bins_to_return={bins} / crop.target_length={crop}: the kit's surface is the 6,144-bin default")
    return {"checked": True, "return_center_bins_only": True, "bins_to_return": 6144, "crop_target_length": crop}


KIT_INSTANCE_ATTRS = ("forward", "get_embs_after_crop", "_flashzoi_kit_runner")   # what serve_surfaces puts on the attached instance (+ the move wrappers, marked _kit_move_wrapper)
SERVED_SURFACES = ("forward", "get_embs_after_crop")   # served through the kit on the attached instance; predict / predict_gene_count / set_track_subset / reset_track_subset are upstream's own code on top of them


_HELPER_STOCK = {}         # the package's own predict_tracks, kept once (restored at remove; a name imported BEFORE apply keeps the stock helper object)


def _flashzoi_helper_body(models, sequence_one_hot, slices):
    """The body swapped INTO the stock ``predict_tracks`` function object: closure-free, the stock's parameter
    names; every name it uses is injected into the package module's globals at apply (the stock function's __globals__)."""
    if all(getattr(m_, "_flashzoi_kit_runner", None) is not None for m_ in models):
        return _flashzoi_kit_predict_tracks(models, sequence_one_hot, slices)    # noqa: F821 — injected into the package module's dict
    return _flashzoi_kit_stock_predict_tracks(models, sequence_one_hot, slices)  # noqa: F821 — the stock body under its original code object


def route_documented_helper() -> str:
    """The DOCUMENTED helper call ``borzoi_pytorch.pytorch_borzoi_helpers.predict_tracks(models, x, slices)`` does a PAGEABLE device-to-host
    copy after each forward, and a notebook binds the name BEFORE apply, so a module-attribute patch never reaches it. The kit therefore
    performs IN-PLACE FUNCTION SURGERY on the STOCK function object: its __code__ is swapped for a closure-free body with the stock's own
    parameter names (the free-variable set asserted equal = empty; __defaults__/__kwdefaults__ carried over), so EVERY already-bound name runs
    the pinned/non-blocking body over kit-attached models (predict_tracks below = engines/flashzoi/predict_tracks_fast: bytes + strides
    identical to the stock helper's) and the stock body for any other call; restored at remove."""
    try:
        import borzoi_pytorch.pytorch_borzoi_helpers as H
    except Exception as e:                                                  # noqa: BLE001 — no package module, nothing to route (stated)
        return f"not routed (package helper module unavailable: {type(e).__name__})"
    stock = H.predict_tracks
    if getattr(stock, "_flashzoi_kit_code_swapped", False):
        return "borzoi_pytorch.pytorch_borzoi_helpers.predict_tracks -> helper: code-swap on the bound object (already applied in this process)"
    import types
    orig = stock.__code__
    new_code = _flashzoi_helper_body.__code__
    assert new_code.co_freevars == () and orig.co_freevars == (), (new_code.co_freevars, orig.co_freevars)     # the free-variable set asserted equal (empty)
    assert orig.co_varnames[:orig.co_argcount] == ("models", "sequence_one_hot", "slices"), orig.co_varnames[:orig.co_argcount]
    stock_copy = types.FunctionType(orig, stock.__globals__, stock.__name__ + "_stock", stock.__defaults__, stock.__closure__)
    stock_copy.__kwdefaults__ = stock.__kwdefaults__
    _HELPER_STOCK["stock"] = stock; _HELPER_STOCK["code"] = orig; _HELPER_STOCK["defaults"] = (stock.__defaults__, stock.__kwdefaults__); _HELPER_STOCK["module"] = H
    H.__dict__["_flashzoi_kit_predict_tracks"] = predict_tracks; H.__dict__["_flashzoi_kit_stock_predict_tracks"] = stock_copy
    stock.__code__ = new_code                                                # every bound name (a notebook's `from ... import predict_tracks` BEFORE apply) now runs this body
    stock.__defaults__ = _flashzoi_helper_body.__defaults__; stock.__kwdefaults__ = _flashzoi_helper_body.__kwdefaults__
    stock._flashzoi_kit_code_swapped = True
    return ("borzoi_pytorch.pytorch_borzoi_helpers.predict_tracks -> helper: code-swap on the bound object (the stock function's __code__ replaced in place: "
            "every name bound before apply runs the kit's pinned/non-blocking path over kit-attached models — the package's pageable .numpy(force=True) replaced, "
            "bytes + strides identical — and the stock body otherwise; restored at remove)")


def unroute_documented_helper() -> bool:
    """Restore the stock function object's code/defaults and remove the injected names (remove)."""
    st = _HELPER_STOCK.get("stock")
    if st is None or not getattr(st, "_flashzoi_kit_code_swapped", False):
        return False
    st.__code__ = _HELPER_STOCK["code"]; st.__defaults__, st.__kwdefaults__ = _HELPER_STOCK["defaults"]
    for k in ("_flashzoi_kit_predict_tracks", "_flashzoi_kit_stock_predict_tracks"):
        _HELPER_STOCK["module"].__dict__.pop(k, None)
    st._flashzoi_kit_code_swapped = False
    return True


def serve_surfaces(model, runner) -> list:
    """The attached model serves EVERY documented surface of borzoi-pytorch, nothing refused: forward(x, is_human, data_parallel_training,
    return_embeddings) and get_embs_after_crop(x) run through the kit (KitRunner.forward_any / embs_after_crop — the plain human-head call
    graph-dispatched, the other forms on the same fused kernels eagerly with upstream's own head arithmetic); predict, predict_gene_count and
    set_track_subset / reset_track_subset are upstream's own code on top of those two and run as shipped (a new head = a new graph at the next
    call); .to() / .cuda() / .cpu() / .half() / … take the kit off the model first (serve_module_moves)."""
    import types
    def _forward(self, x, is_human=True, data_parallel_training=False, return_embeddings=False):
        return runner.forward_any(x, is_human=is_human, data_parallel_training=data_parallel_training, return_embeddings=return_embeddings)
    def _get_embs_after_crop(self, x):
        return runner.embs_after_crop(x)
    model._flashzoi_kit_runner = runner                                    # the attach mark the helper router tests (kit-attached models only)
    model.forward = types.MethodType(_forward, model)
    model.get_embs_after_crop = types.MethodType(_get_embs_after_crop, model)
    moves = serve_module_moves(model, runner)
    return list(SERVED_SURFACES) + [f"{n}() -> detach" for n in moves]


def _predict_tensor_checks(runner, xt) -> bool:
    """predict_tensor's own guard (the served surfaces route any other input to upstream's method before it gets here): (B, 4, 524288), never
    empty. The numerics class is read per call (route_numerics), never a refusal: the in-call TF32 setting mirrors it."""
    if not hasattr(xt, "shape") or xt.ndim != 3 or tuple(xt.shape[1:]) != (4, lane.SEQ_LEN):
        raise lane.PinDrift(f"forward() through the kit: input {getattr(xt, 'shape', None)} != (B, 4, {lane.SEQ_LEN})")
    if getattr(runner, "graph", None) is not None and not hasattr(runner, "graphs") and xt.shape[0] != runner.batch:
        raise lane.PinDrift(f"forward() through the kit: graph captured at batch {runner.batch}, got {xt.shape[0]} — the captured batch shapes only")
    if xt.shape[0] < 1:
        raise lane.PinDrift("forward() through the kit: an empty batch")
    route_numerics("forward() through the kit")
    return True


def device_capability():
    """((major, minor), total memory MiB) of CUDA device 0, or (None, None) where no CUDA device is visible (CPU tests)."""
    try:
        if not torch.cuda.is_available(): return None, None
        p = torch.cuda.get_device_properties(0)
        return (int(p.major), int(p.minor)), int(p.total_memory // (1024 * 1024))
    except Exception:  # noqa: BLE001
        return None, None


def assert_device_class(PINS: dict, device_name: str, cc=None, memory_mib=None) -> dict:
    """The device's class, decided at construction and worded on the apply line — never a refusal: PINNED are the kit's pinned classes (PINS['device_classes'] by
    compute capability, PINS['device_names'] by name) and any class named by a class record of the kit dir (class_records/*.json: the
    device's class BY CAPABILITY — `sm` == its compute capability; memory is documentation: a part of the same capability with another
    memory size is served, with a `memory_note` clause on the apply line — or by name in its 'device_names', and whose 'kit' is this kit's
    fz_exact sha: _class_records.record_serves); any other NVIDIA device is served UNPINNED — the levers engage wherever the kernels run and
    the line names it ({'unpinned': <why>}: bitwise equality to stock is not established on that class); a CPU / non-CUDA device passes through (the kit
    cannot run there anyway; the caller refuses a non-CUDA model by name)."""
    pinned = tuple(PINS.get("device_names") or ())
    if not device_name.startswith("NVIDIA"):
        return {"device_name": device_name, "pinned": False, "record": None}
    if cc is None: cc, memory_mib = device_capability()                   # the device's class by capability (cc + total memory), its name second
    from ._class_records import pinned_serves, record_serves
    from ._class_records import class_row, memory_note
    if pinned_serves(PINS, device_name, cc, memory_mib):                # the kit's pinned classes: PINS['device_classes'] by capability, PINS['device_names'] by name
        note = memory_note(class_row(PINS.get("device_classes"), cc, memory_mib), memory_mib)   # a part of a pinned capability with another memory size: served, one clause on the line
        return {"device_name": device_name, "pinned": True, "record": None, **({"memory_note": note} if note else {})}
    here = os.path.dirname(os.path.abspath(__file__))                   # class records resolve from the kit dir: class_records/*.json — every record a candidate
    candidates = sorted(glob.glob(os.path.join(here, "class_records", "*.json")))
    in_dir = [p for p in candidates if os.path.isfile(p)]
    def _names(p):
        try: rec = json.load(open(p))
        except (OSError, ValueError): return False
        return record_serves(rec, PINS.get("fz_exact_sha256"), device_name, cc, memory_mib)
    def _served(path):                                                  # the record that serves + the memory clause when this part's memory differs from the record's
        try: note = memory_note(json.load(open(path)), memory_mib)
        except (OSError, ValueError): note = None
        return {"device_name": device_name, "pinned": False, "record": path, **({"memory_note": note} if note else {})}
    hit = next((p for p in in_dir if _names(p)), None)                   # the dir's record naming THIS device for THIS kit (its fz_exact digest)
    if hit:
        return _served(hit)
    why = (f"no pinned class {list(pinned)} and no class record of the kit dir ({', '.join(os.path.basename(p) for p in in_dir) or 'none'}) names "
           f"{device_name} (cc {cc}, {memory_mib} MiB) for kit {str(PINS.get('fz_exact_sha256'))[:12]}: the levers engage, bitwise vs stock unverified on this class")
    return {"device_name": device_name, "pinned": False, "record": None, "unpinned": why}


# ---- THE USER'S ROUTE (from helpers.predict_tracks @ v0.5.1 8a05eb15 lines 4-13): the package's documented route is serial per fold —
# forward, then `.numpy(force=True)` = a PAGEABLE full-array device-to-host copy (7611 x 6144 fp32) during which the GPU idles, THEN the host
# track slice, np.concatenate over folds, swapaxes. The kit ships the SAME function (same signature, byte-identical output incl. strides) with
# the host side rewritten: engines/flashzoi/predict_tracks_fast.py (the one implementation). Works over kit-attached models (forward -> the
# runner's predict_tensor) AND over stock models (then it is the pipeline lever alone).


def predict_tracks(models, sequence_one_hot, slices):
    """borzoi_pytorch.pytorch_borzoi_helpers.predict_tracks, byte-identical output (shape, dtype, strides, bytes), with the host side of the
    user's route rewritten — engines/flashzoi/predict_tracks_fast.py (the slice on the device before the copy, pinned buffers on a side stream
    overlapped with the next fold's forward, the result allocated ONCE from a page-locked pool with each fold landed straight into its slot). This wrapper vendors it BY IMPORT —
    no second implementation lives here; absent on this tree it REFUSES by name. Over kit-attached models the per-fold forward is the runner's
    predict_tensor."""
    try:
        from engines.flashzoi import predict_tracks_fast as PF
    except ImportError as e:
        raise lane.PinDrift("engines.flashzoi.predict_tracks_fast (the host-side route) is not on this tree; "
                            "the kit ships no second implementation") from e
    return PF.predict_tracks_fast(models, sequence_one_hot, slices)


# ---- GRAPH USE-AFTER-FREE RULE: a captured CUDA graph holds device ADDRESSES; any lazily-allocated cache the graph reads
# (flash_attn.layers.rotary.RotaryEmbedding._cos_cached/_sin_cached/_cos_k_cached: REALLOCATED by _update_cos_sin_cache when
# seqlen > _seq_len_cached, or the device/dtype changes, or (training and inference tensor) — rotary.py:410-453 @ v2.7.0.post2) must be
# pinned at the max shape BEFORE capture and asserted unchanged before every replay; a post-capture .to() clears them (.device change) ->
# refused. Flashzoi's token length is FIXED (4,096 for the 524,288-bp input the kit serves; other input shapes are refused at predict), so
# the max shape IS the captured shape; the pins make a silent reallocation impossible to miss.
LAZY_CACHE_ATTRS = ("_cos_cached", "_sin_cached", "_cos_k_cached", "_sin_k_cached")


def graph_cache_pins(model) -> list:
    """Every lazily-allocated device tensor a captured graph may address: (module path, attr, data_ptr, shape, dtype, device,
    _seq_len_cached) for each RotaryEmbedding-like module (attrs in LAZY_CACHE_ATTRS) after the capture warm-up."""
    pins = []
    for name, m in model.named_modules():
        if not any(hasattr(m, a) for a in LAZY_CACHE_ATTRS):
            continue
        for a in LAZY_CACHE_ATTRS:
            t = getattr(m, a, None)
            if t is None or not hasattr(t, "data_ptr"):
                continue
            pins.append({"module": name, "attr": a, "data_ptr": int(t.data_ptr()), "shape": tuple(t.shape), "dtype": str(t.dtype), "device": str(t.device),
                         "seq_len_cached": getattr(m, "_seq_len_cached", None)})
    return pins



ROTARY_INV_FREQ_SHA256 = "af9890bba64c7a3f"    # sha256(inv_freq fp32 bytes) prefix, torch 2.5.1 on the pinned stack's host


def assert_rotary_pinned(model) -> dict:
    """Flashzoi's rotary frequencies are a NON-PERSISTENT host-built fp32 buffer (flash_attn rotary.py:387-388; the formula IS the
    pinned rotary; the checkpoint carries none). The kit never re-derives them (the MHA is called unchanged) and the precast casts only
    Conv1d/Linear weights (fz_exact.precast_bf16) — asserted here on every attach: dtype float32 and sha256 per layer == the pinned table."""
    shas, bad = {}, []
    for name, mod in model.named_modules():
        if mod.__class__.__name__ == "RotaryEmbedding":
            iv = getattr(mod, "inv_freq", None)
            if iv is None:
                bad.append(f"{name}: no inv_freq"); continue
            if iv.dtype != torch.float32:
                bad.append(f"{name}: inv_freq dtype {iv.dtype} != float32 (the precast must exclude buffers)")
            shas[name] = hashlib.sha256(iv.detach().float().cpu().numpy().tobytes()).hexdigest()
    if bad:
        raise lane.PinDrift("rotary pin: " + "; ".join(bad))
    distinct = sorted(set(shas.values()))
    rec = {"layers": len(shas), "distinct_sha256": distinct, "dtype": "float32", "form": "constructed, persistent=False (flash_attn 2.7.0.post2 rotary.py:387-388); never re-derived by the kit"}
    if shas and any(not s.startswith(ROTARY_INV_FREQ_SHA256) for s in distinct):   # another flash-attn build or host libm: the model's own table is what both routes run — named, not refused
        rec["drift"] = f"rotary inv_freq table {[s[:12] for s in distinct]} differs from the pinned flash-attn 2.7.0.post2 table {ROTARY_INV_FREQ_SHA256[:12]}"
    return rec

POOL_SLOTS_DEFAULT = 4                                                           # the pinned lease pool's depth (predict(): one unit x 2 strands per runner, a lease + one in copy-out = 4)


def graph_pin_refs(runner) -> list:
    """The per-forward guard reads CACHED host state — the pinned modules' object refs + the captured (data_ptr, shape, dtype,
    device, seq_len_cached) tuples are resolved ONCE (here, at capture / first use) so no module-tree walk or driver query sits in the
    forward; the per-forward check is len(pins) attribute reads."""
    refs = getattr(runner, "_graph_pin_refs", None)
    if refs is not None:
        return refs
    mods = dict(runner.model.named_modules())
    refs = []
    for p in (getattr(runner, "_graph_pins", None) or []):
        m = mods.get(p["module"])
        refs.append((m, p["attr"], (p["data_ptr"], tuple(p["shape"]), p["dtype"], p["device"], p["seq_len_cached"]), p["module"]))
    runner._graph_pin_refs = refs
    return refs


def assert_graph_cache_pins(runner) -> None:
    """PinDrift if any pinned cache moved, was reallocated, changed shape/dtype/device or its cached length since capture
    (cached refs: no per-forward module walk, no driver call; data_ptr()/shape/dtype/device are host attribute reads)."""
    if not getattr(runner, "_graph_pins", None) or getattr(runner, "graph", None) is None:
        return
    for m, attr, want, name in graph_pin_refs(runner):
        t = getattr(m, attr, None) if m is not None else None
        now = None if t is None else (int(t.data_ptr()), tuple(t.shape), str(t.dtype), str(t.device), getattr(m, "_seq_len_cached", None))
        if now != want:
            raise lane.PinDrift(f"graph cache pin moved: {name}.{attr} captured {want} now {now} — a replay would read freed/reused memory; refusing")


def serve_module_moves(model, runner) -> list:
    """.to() / .cuda() / .cpu() / .half() / .float() / .bfloat16() / .double() / .type() on an attached model first take the kit off it
    (runner.detach: a captured graph holds device addresses and the kit's tables hold the loaded dtype/device), then run the module's own
    method — nothing is refused; the kit re-attaches at the model's next CUDA forward."""
    names = ("to", "cuda", "cpu", "half", "float", "bfloat16", "double", "type")
    def _wrapped(name):
        def f(*a, **k):
            runner.detach(f"model.{name}() called")                                   # removes these instance wrappers too (close restores the instance dict)
            return getattr(type(model), name)(model, *a, **k)                         # the nn.Module method itself
        f._kit_move_wrapper = True
        return f
    done = []
    for n in names:
        if callable(getattr(model, n, None)):
            object.__setattr__(model, n, _wrapped(n)); done.append(n)
    return done


def graph_head_key(model) -> tuple:
    """The human head a graph is captured with: upstream's set_track_subset / reset_track_subset install a NEW head module, so a graph captured with
    the old one must not replay (it would read the old weights' addresses)."""
    h = model.human_head
    return (id(h), int(h.weight.data_ptr()), tuple(h.weight.shape))

class _GraphedShared:
    """fz_exact.Graphed's capture form (3 warm-up replays on a side stream, then torch.cuda.graph) with ONE memory pool shared across the
    batch-keyed graphs (lazily captured shapes share one pool). The kernels are fz_exact's, byte-identical; only when/where the
    capture happens differs."""

    def __init__(self, fn, example, pool, warmup: int = 3):
        self.fn = fn; self.x = example.clone()
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup):
                fn(self.x)
        torch.cuda.current_stream().wait_stream(s)
        self.g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g, pool=pool):
            self.y = fn(self.x)
        self.replays = 0

    def __call__(self, x):
        assert tuple(x.shape) == tuple(self.x.shape) and x.dtype == self.x.dtype and x.device == self.x.device, \
            f"no captured graph for shape {tuple(x.shape)}/{x.dtype} (captured {tuple(self.x.shape)}/{self.x.dtype})"
        self.x.copy_(x); self.g.replay(); self.replays += 1; return self.y

    def status(self):
        d = {"graph_replays": self.replays, "graph_shape": list(self.x.shape)}
        if hasattr(self.fn, "counters"):
            d.update(self.fn.counters)
        return d


def _capture_graph(runner, fz, batch: int, device: str):
    """Capture the fused stack at ``batch`` into the runner's shared pool; the capture-time counters are the structural evidence
    (stage-1 inside the graph)."""
    example = torch.zeros((int(batch), 4, lane.SEQ_LEN), dtype=torch.float32, device=device)
    example[:, 0, :] = 1.0
    g = _GraphedShared(runner._eager_fn, example, runner._graph_pool)
    del example
    return g


class _GraphDispatch:
    """The kit's plain forward by batch shape: a shape's FIRST call captures the fused stack into a CUDA graph (3 warm-up runs and the capture on a
    static input, one memory pool shared by every shape; the seconds recorded per shape) and every call of that shape replays it, the first
    included. A graph is keyed by the batch AND the human head it was captured with (graph_head_key): after upstream's set_track_subset the next call
    captures afresh and the stale graph is dropped. A capture the device refuses (other than an out-of-memory error, which propagates) is recorded
    by name and that shape runs the same kernels eagerly. Counts: graph_replays per replay, eager_fused_calls per eager forward, lazy_captures;
    capture-time launches go to runner.capture (structural evidence), never to run-time."""

    def __init__(self, runner, fz, device):
        self.runner, self.fz, self.device = runner, fz, device

    def _capture_now(self, b: int):
        r = self.runner; fz = self.fz
        before = dict(r.counts)
        st = getattr(r, "stack", None)
        st_before = dict(st.counters) if st is not None and hasattr(st, "counters") else {}
        site_before = dict(fz.SITE_CALLS) if hasattr(fz, "SITE_CALLS") else {}
        _t = time.perf_counter()
        with torch.inference_mode():
            g = _capture_graph(r, fz, b, self.device)
        r.capture_s_by_batch[b] = time.perf_counter() - _t
        # the capture-time deltas = the structural evidence; the run-time counters are restored (a replay moves no Python counter)
        cap = {"stage1_calls": r.counts["stage1_calls"] - before["stage1_calls"]}
        r.counts["stage1_calls"] = before["stage1_calls"]
        if st_before or st is not None and hasattr(st, "counters"):
            for k, v in list(st.counters.items()):
                d = int(v) - int(st_before.get(k, 0)); cap[f"fz_{k}"] = d; st.counters[k] = st_before.get(k, 0)
        if hasattr(fz, "SITE_CALLS"):
            for k, v in list(fz.SITE_CALLS.items()):
                d = int(v) - int(site_before.get(k, 0)); cap[f"site_{k}"] = d; fz.SITE_CALLS[k] = site_before.get(k, 0)
        if "stage1" in r.levers and cap["stage1_calls"] < 1:
            raise lane.PinDrift(f"kit: the CUDA-graph capture at batch {b} ran no stage-1 launch ({cap}) — the kit's stage-1 path is not in the graph")
        for k, v in cap.items():
            r.capture[k] = r.capture.get(k, 0) + int(v)
        r.capture["captures"] = r.capture.get("captures", 0) + 1
        r.graphs[b] = g; r.graph_keys[b] = graph_head_key(r.model)
        if r.batch == b:
            r.graph = g
        r._graph_pins = graph_cache_pins(r.model); r._graph_pin_refs = None; graph_pin_refs(r)      # GRAPH UAF RULE: the pins (re)taken after every capture
        return g

    def _graph_for(self, b: int):
        r = self.runner
        key = graph_head_key(r.model)
        g = r.graphs.get(b)
        if g is not None and r.graph_keys.get(b) == key:
            return g
        if g is not None:                                                          # the human head changed since the capture (set_track_subset / reset_track_subset): drop the stale graph
            del r.graphs[b]; r.graph_keys.pop(b, None)
            if r.graph is g: r.graph = None
            del g
            if not r.graphs and torch.cuda.is_available():                            # the shared pool's last graph is gone: the allocator retires that private pool, so the next capture opens a fresh one
                r._graph_pool = torch.cuda.graph_pool_handle()
        if b in r.capture_refused:
            return None
        try:
            g = self._capture_now(b)
            r.counts["lazy_captures"] += 1
            return g
        except (RuntimeError, MemoryError) as e:                                  # an out-of-memory error during the capture PROPAGATES (never the eager path: no fallback on OOM); any OTHER capture failure = that shape stays eager, by name
            if is_oom(e):
                raise
            r.capture_refused[b] = f"capture refused at batch {b}: {type(e).__name__}: {str(e)[:160]} — eager fused path (same kernels, no graph)"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None

    def __call__(self, xt):
        b = int(xt.shape[0]); r = self.runner
        g = self._graph_for(b)
        if g is None:
            r.counts["eager_fused_calls"] += 1
            return r._eager_fn(xt)
        r.counts["graph_replays"] += 1
        return g(xt)

    @property
    def replays(self):
        return sum(g.replays for g in self.runner.graphs.values())

    def status(self):
        r = self.runner
        return {"graph_replays": self.replays, "graph_shapes": {str(b): list(g.x.shape) for b, g in r.graphs.items()},
                "capture_s": {str(b): round(s_, 4) for b, s_ in r.capture_s_by_batch.items()}, "capture_refused": dict(r.capture_refused)}

    def __getattr__(self, name):
        gs = self.runner.graphs
        if not gs:
            raise AttributeError(name)
        return getattr(gs.get(self.runner.batch) or next(iter(gs.values())), name)


def read_numerics_class() -> dict:
    """The numerics class the user's process actually runs under, read from torch's own switches (never assumed). Classes:
    'torch-default' (cudnn TF32 on, matmul TF32 off, fp32 matmul precision highest), 'tf32-off' (the package notebook's recipe),
    'tf32-on' (both on / precision high|medium), each '+autocast:<dtype>' when an autocast context is active at apply."""
    d = {"cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32), "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
         "float32_matmul_precision": str(torch.get_float32_matmul_precision()), "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
         "cudnn_deterministic": bool(torch.backends.cudnn.deterministic), "autocast": bool(torch.is_autocast_enabled() or torch.is_autocast_cpu_enabled()),
         "autocast_dtype": (str(torch.get_autocast_gpu_dtype()) if torch.is_autocast_enabled() else (str(torch.get_autocast_cpu_dtype()) if torch.is_autocast_cpu_enabled() else None))}
    tf32_mm = d["matmul_allow_tf32"] or d["float32_matmul_precision"] != "highest"
    if d["cudnn_allow_tf32"] and not tf32_mm:
        cls = "torch-default"
    elif not d["cudnn_allow_tf32"] and not tf32_mm:
        cls = "tf32-off"
    else:
        cls = "tf32-on"
    if d["autocast"]:
        cls += "+autocast:" + str(d["autocast_dtype"]).replace("torch.", "")
    d["class"] = cls
    d["kit_decision"] = ("pinned to the stock's torch-default TF32 class: inside each call the head GEMM runs the TF32 class stock's cuDNN conv head runs "
                         "under (cudnn.allow_tf32 at rest) and the switches are restored after it — bitwise to the stock and faster than it; a process whose "
                         "switches at rest name another class (matmul TF32 on, a lowered float32 matmul precision, cuDNN TF32 off) is served the same way and "
                         "named on the apply line as drift")
    return d


TF32_CLASS_SERVED = "torch-default"   # the TF32 class the exact route reproduces: the stock's own (cuDNN TF32 on, matmul TF32 off, float32 matmul precision 'highest')


def route_numerics(where: str) -> dict:
    """The numerics class the process runs under at rest (read_numerics_class), with `pinned_class`: True at torch's own defaults — cudnn TF32 on,
    matmul TF32 off, float32 matmul precision 'highest' — the class the kit's rows are pinned to; False for any other class, which is NAMED on
    the apply line (drift) and served: inside each call the head GEMM's TF32 follows cudnn.allow_tf32 at rest (the switch
    stock's cuDNN conv head obeys), so both routes run one class; nothing is refused over it. `where` names the reader for the record."""
    d = read_numerics_class()
    d["pinned_class"] = str(d["class"]).split("+", 1)[0] == TF32_CLASS_SERVED
    d["read_at"] = where
    return d


def kit_max_batch(seq_len: int = lane.SEQ_LEN, stem_channels: int = lane.STEM_CHANNELS, stock_max_batch: int = lane.STOCK_MAX_BATCH) -> int:
    """The batch one dispatch serves, FROM BYTES: the kit's Triton launches index each window's slab at int32 offsets from a 64-bit window
    base, except the stage-1 kernel, which stores (n * Lo + q) * 512 + c flat — stem_channels x (seq_len // 2) = 2**27 elements per window,
    so 2**31 // 2**27 = 16 windows fit its int32 offsets. Upstream's own forward serves at most stock_max_batch = 15 windows at this length
    (torch.max_pool1d refuses 16), which is the smaller: one dispatch serves every batch the stock serves, as one call of that size; a larger
    batch is dispatched in chunks of it (each chunk = the stock forward's bytes on those windows as one call of that size)."""
    lo = seq_len // 2
    flat_bound = max(1, (2 ** 31) // (stem_channels * lo))             # the one remaining flat int32 launch (the stage-1 store)
    return min(flat_bound, stock_max_batch)


def assert_batch_bound_dims(model) -> dict:
    """Attach-time assertion: the filters the kernels' index arithmetic is written for (lane.STEM_CHANNELS — the stage-1 store's literal 512 —
    and lane.TOWER1_CHANNELS, the first site's per-window extent) are the attached model's own (conv_dna, res_tower.0) — a model of other dims
    is refused by name rather than dispatched through kernels cut for another shape. Objects without those modules (tests' fakes) pass through."""
    try:
        stem = int(model.conv_dna.conv_layer.out_channels); tower1 = int(model.res_tower[0].conv_layer.out_channels)
    except (AttributeError, IndexError, TypeError):
        return {"checked": False}
    if (stem, tower1) != (lane.STEM_CHANNELS, lane.TOWER1_CHANNELS):
        raise lane.PinDrift(f"kit on a model with conv_dna / res_tower.0 filters ({stem}, {tower1}) != ({lane.STEM_CHANNELS}, {lane.TOWER1_CHANNELS}): "
                            f"the kernels' index arithmetic ({KIT_MAX_BATCH} windows per dispatch) is written for the pinned filters only")
    return {"checked": True, "stem_channels": stem, "tower1_channels": tower1, "max_batch": KIT_MAX_BATCH}


KIT_MAX_BATCH = kit_max_batch()


def compose_apply_line(runner) -> str:
    """ONE line at the end of apply — GPU class, numerics class read back, levers ON, what apply did (seconds), the graph rule (one capture per
    batch shape at its first call) and the lazy pool, the Triton cache and compiles, the helper route, skipped items and why."""
    a = runner.apply_s
    a["class_label"] = getattr(runner, "class_label", "exact")
    paid = float(a.get("patch_s", 0.0)) + float(a.get("warmup_forward_s", 0.0)) + float(a.get("capture_s", 0.0))
    skipped = []
    if "graph" not in runner.levers:
        skipped.append("graph: not selected")
    if a.get("warmup_skipped"):
        skipped.append("warm-up forward: " + a["warmup_skipped"])
    for b, why in sorted(runner.capture_refused.items()):
        skipped.append(f"graph@batch {b}: {why}")
    jit = a.get("jit_cache", "not stated")
    log = list(getattr(sys.modules.get(getattr(runner, "_pkg", "")), "JIT_COMPILE_LOG", []) or [])
    jit += f" | compile(): {len(log)} calls" + (f", max {max(log) * 1000:.1f} ms" if log else "")
    dc = getattr(runner, "device_class", None) or {}
    cls_word = "pinned" if dc.get("pinned", True) else (f"by record {os.path.basename(str(dc.get('record')))}" if dc.get("record") else f"unpinned — {dc.get('unpinned')}")   # PINS['device_names' / 'device_classes'] = pinned; else the class record of the kit dir that names the device
    if dc.get("memory_note"): cls_word += f"; {dc['memory_note']}"       # a part of the class's capability with another memory size: served, said here (never a key)
    return (f"[flashzoi kit {runner.arm}] GPU {runner.device_name} (device class: {cls_word}) | numerics: {runner.numerics['class']} "
            f"(cudnn.allow_tf32={runner.numerics['cudnn_allow_tf32']}, matmul.allow_tf32={runner.numerics['matmul_allow_tf32']}, "
            f"fp32 matmul precision {runner.numerics['float32_matmul_precision']}) -> {getattr(runner, 'route_class', 'exact route')} | levers ON: {', '.join(sorted(runner.levers))} "
            f"| act dtype fp16 "
            f"| apply {paid:.2f} s = patch {float(a.get('patch_s', 0.0)):.2f} + warm-up forward@batch {a.get('warmup_batch', '-')} {float(a.get('warmup_forward_s', 0.0)):.2f} "
            f"| graphs: one CUDA-graph capture per batch shape at its first call, replay from then on; pinned pool {a.get('pool_slots')} slots at the first lease "
            f"| batch bound: {KIT_MAX_BATCH} windows per dispatch, larger batches in chunks of {KIT_MAX_BATCH} "
            f"| jit: {jit} | flags at rest: {a.get('flags_at_rest', 'not read back')} | helper: {a.get('helper_route', 'not routed')} "
            f"| skipped: {'; '.join(skipped) if skipped else 'none'}"
            + (f" | drift: {'; '.join(runner.drift)}" if getattr(runner, "drift", None) else ""))


NUMERICS_FLAGS = ("cudnn_allow_tf32", "matmul_allow_tf32", "float32_matmul_precision", "cudnn_benchmark", "cudnn_deterministic", "deterministic_algorithms", "autocast_cuda", "autocast_cpu")


def numerics_flags_snapshot() -> dict:
    """The process-global numerics state a user owns: read only, never written by the kit."""
    return {"cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32), "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "float32_matmul_precision": str(torch.get_float32_matmul_precision()), "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic), "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            "autocast_cuda": bool(torch.is_autocast_enabled()), "autocast_cpu": bool(torch.is_autocast_cpu_enabled())}


def assert_no_flag_residue(at_entry: dict, when: str) -> None:
    """PinDrift on any process-global numerics flag that differs from the snapshot at apply's entry: the kit sets nothing at rest
    (fz_exact's autocast is a per-forward `with` scope; its set_det() helper is never called by the kit)."""
    now = numerics_flags_snapshot()
    diff = {k: (at_entry.get(k), now.get(k)) for k in NUMERICS_FLAGS if at_entry.get(k) != now.get(k)}
    if diff:
        raise lane.PinDrift(f"kit {when}: process-global numerics flag residue {diff} (nothing global may move at rest)")

def remove(runners, *, stock_clock_s: float | None = None, quiet: bool = False) -> dict:
    """THE DOCUMENTED REMOVE: `kit.remove(runners)` — every runner's model restored to its pre-apply state (KitRunner.close(): forward /
    the attach mark, the LayerNorm modules, the precast parameters BY REFERENCE (the original objects); the stack / graphs / pinned pool released),
    the documented helper's __code__ restored by reference on the bound object, the page-locked result pools + the background-prepared pool released,
    the JIT compile hook detached, the flags read back. Prints ONE line: 'removed: …; helper = stock (…)' — with FAIL words if any lever stays active."""
    import gc
    checks = {}
    for k_, r_ in enumerate(runners if isinstance(runners, (list, tuple)) else [runners]):
        checks[f"runner{k_}"] = r_.close()
    helper_line = "helper = not routed (nothing to restore)"
    st = _HELPER_STOCK.get("stock")
    if st is not None and getattr(st, "_flashzoi_kit_code_swapped", False):
        ok = unroute_documented_helper(); same = st.__code__ is _HELPER_STOCK.get("code")
        helper_line = f"helper = stock (__code__ restored by reference: {ok and same})" + (f" ({stock_clock_s * 1000:.0f} ms before apply per the caller)" if stock_clock_s else "")
        checks["helper_restored"] = bool(ok and same)
    pkg = sys.modules.get(__name__.rsplit(".", 1)[0])
    try:
        from engines.flashzoi import predict_tracks_fast as PF
        pools = getattr(PF, "_RESULT_POOLS", None)
        if isinstance(pools, dict): pools.clear()
        checks["result_pools_released"] = True
    except Exception:                                                          # noqa: BLE001 — the helper module absent: nothing to release
        checks["result_pools_released"] = "n/a (predict_tracks_fast not imported)"
    if pkg is not None:
        checks["jit_hook_detached"] = pkg._unhook_triton_compile() if hasattr(pkg, "_unhook_triton_compile") else "n/a"
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:                                                          # noqa: BLE001
        pass
    fails = [f"{k_}: {v_}" for k_, v_ in checks.items() if v_ is False or (isinstance(v_, str) and v_.startswith("FAIL"))] + [f"{k_}.{kk}: {vv}" for k_, v_ in checks.items() if isinstance(v_, dict) for kk, vv in v_.items() if vv is False or (isinstance(vv, str) and vv.startswith("FAIL"))]
    line = ("removed: " + ", ".join(f"{k_} {'ok' if not isinstance(v_, dict) else ('ok' if not [kk for kk, vv in v_.items() if vv is False or (isinstance(vv, str) and vv.startswith('FAIL'))] else 'FAIL')}" for k_, v_ in checks.items())
            + f"; {helper_line}" + (("; FAIL: " + " | ".join(fails)) if fails else "; every lever removed"))
    if not quiet: print(line, flush=True)
    return {"line": line, "checks": checks, "fail": bool(fails)}
