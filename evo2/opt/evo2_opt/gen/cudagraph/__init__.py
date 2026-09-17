"""evo2_opt.gen.cudagraph — replay the stock single-token decode step of Evo 2 generation as CUDA graphs (numerics class: exact).

What it does. `Evo2.generate` runs vortex's `Generator.generate` (vtx 1.1.0 `vortex/model/generation.py`): one prefill forward over the
prompt, then per generated token one call `StripedHyena.forward(x:(B,1), inference_params_dict)` (`vortex/model/model.py` `forward` ->
`stateful_forward`) followed by `sample()` on the host side of the loop. At batch 1 that decode step is ~1.3k (7B) / ~2.3k (40B) small kernel
launches issued from Python, and the step is host-dispatch-bound. This lever captures the decode step in `torch.cuda.CUDAGraph`s at the
SECOND decode step of each `generate()` call and REPLAYS them for every later step (the first decode step runs the stock eager code: it is the
step at which stock converts the prefill's bf16 state views into the fp32 state tensors every later step reads and rebinds). Sampling stays
outside the graphs (unchanged stock code). Granularity:
  * one device (7B on one GPU): ONE graph = the whole step, embedding -> every block -> final norm -> unembed ("whole");
  * several devices (40B: vortex puts blocks 0-24 on cuda:0 and 25-49 on cuda:1): one graph per DEVICE SEGMENT of consecutive blocks,
    anchored at the segment's first block; the stock `stateful_forward` still drives the step — embedding, the `cross_device_transfer` /
    `x.to(device 0)` hops, final norm and unembed run as stock eager code, the anchor block's call replays its segment and the other blocks of
    the segment pass their input through ("segments"). Inside a segment body the lever calls each block of the segment in order with its
    family's inference params — the one place it re-states a stock loop (3 lines of `stateful_forward`), because a graph cannot span devices.

What is made static for replay (the stock step's per-token host values):
  * the token input / each segment's input -> a static buffer, `copy_` before replay; the output -> a static buffer returned as `.clone()`;
  * attention position/length: stock passes `inference_params.seqlen_offset` (a Python int) as `cache_seqlens` to
    `flash_attn_with_kvcache`, whose wrapper turns an int into `torch.full((B,), n, int32)` (`vortex/ops/attn_interface.py`); the lever
    sets `inference_params.lengths_per_sample` to a static int32 tensor (one per device) filled with `seqlen_offset` before each replay — the
    same kernel with the same argument kind — and resets it to None after the call (prefill never sees it);
  * hyena FIR / IIR states: stock REBINDS `fir_state_dict[i]`, `fir_inner_state_dict[i]`, `state_dict[i]` to new tensors every step
    (`HyenaInferenceEngine.step_fir/step_iir`); inside the captured region the lever copies each new state back into the tensor the first
    decode step produced and rebinds the dict to it (bit-exact copies), so replay N+1 reads what replay N wrote;
  * the kv cache needs nothing: stock preallocates it to (max_batch, prompt+n_tokens, 2, H, Dh) at prefill and the flash kernel writes
    slot `cache_seqlens` in place;
  * Transformer-Engine FP8 projections (40B): every `fp8_meta` scale / scale_inv / amax_history tensor of the modules inside a segment is
    snapshot before the warm-up runs and restored after them and after capture, so the captured call sequence sees exactly the FP8 state the
    stock step would have; whatever per-call amax/scale kernels stock launches inside the step are captured and replayed like any other kernel.
Capture protocol (per generate() call, at its second decode step, per segment): snapshot the hyena states (a few MB) and fp8 metas ->
`warmup` eager runs of the body on a side stream, restoring the snapshot after each (the kv slot is rewritten with identical values) ->
capture on the side stream with `CUDAGraph.capture_begin/capture_end` (kernels are recorded, not run; unlike the `torch.cuda.graph` context
manager nothing calls `torch.cuda.empty_cache()`, so no cached memory is unmapped and the next call's prefill pays no re-mapping) -> restore ->
replay = the real step. Each segment owns ONE private pool for the arm's life, held in use by an empty sentinel graph; every re-capture reuses
the blocks the previous graph returned to it, so memory is flat across generate() calls under the default allocator and under
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` alike. A capture that raises is REFUSED by name, never a silent fallback: the capture
is ended, the states are restored, the capture state is healed (one empty capture clears the CUDA generators' capture registration) and
`CaptureRefused` propagates out of the call (an out-of-memory propagates as itself). A condition the lever names itself (the checks listed
below) is different: that generate() call runs the stock eager step and the next call captures again.

Refusals (named, printed once per event, recorded in `status(model)["refusals"]`; the stock eager step runs instead):
  * `install`: no CUDA device; a block on a non-cuda device; `use_flash_attn` off, rotary dim % 16 != 0 or flash-attn's kvcache kernel
    absent (the non-fused decode path slices the kv cache with Python ints); a block type the lever does not know.
  * per generate() call: batch above the cache's max_batch_size; `lengths_per_sample` already set by the caller; any hyena FIR state still
    growing (prompt shorter than `min_prompt_tokens` = longest FIR filter - 1 = 127 for evo2_7b/40b: `step_fir` uses torch.cat until the
    state is full); a state tensor of unexpected shape/device. A capture that RAISES is not in this list: REFUSED names the exception and
    `CaptureRefused` propagates out of the call (above).
Hooking discipline: the lever sets the INSTANCE attribute `forward` of the StripedHyena object (and of each block in segments mode) through
`chain_install`, remembering any instance hook another party (e.g. the scoring kit) put there first, delegating to it (else to the class method
resolved at call time), and putting it back exactly at `remove` (LIFO: a hook installed on top of this lever must come off first, raised by name).
Numerics: a replay runs the stock step's kernels on the stock step's buffers, so per-step logits and sampled tokens are the stock's.
Memory: + the graph pools (one step's activations) and one cuBLAS workspace for the lever's side stream per device while armed; flat across
generate() calls.

    from evo2 import Evo2
    from evo2_opt.gen.cudagraph import arm          # = install
    m = Evo2("evo2_7b", local_path=...); h = arm(m) # prints [evo2-gen cudagraph] ARMED: ... ; h.describe(), h.record (= status(m))
    m.generate([...], n_tokens=..., ...)            # prints [evo2-gen cudagraph] INSTALLED: ... at the second decode step of the call
    h.uninstall()                                   # = remove(m): stock forward restored, graphs and pools dropped
"""
import sys
from evo2_opt._oom import is_oom
import time
import weakref


from evo2_opt.gen._chain import chain_install, chain_next, chain_uninstall as _chain_uninstall   # the one chaining discipline of every gen arm

TAG = "[evo2-gen cudagraph]"


def chain_uninstall(obj, fn, prev):
    """`_chain.chain_uninstall` with this arm's tag on the LIFO refusal."""
    return _chain_uninstall(obj, fn, prev, tag=TAG)

LEVER = "cudagraph"
__all__ = ["arm", "install", "remove", "status", "Handle", "CaptureRefused", "TAG", "chain_install", "chain_next", "chain_uninstall"]


class CaptureRefused(RuntimeError):
    """The decode step cannot be captured for this model / call; `.reason` is the printed reason."""

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def _say(word, reason, verbose=True):
    if verbose:
        print(f"{TAG} {word}: {reason}", file=sys.stdout, flush=True)


def _striped(model):
    """Accept an `Evo2` instance (has `.model`) or the `StripedHyena` module itself."""
    inner = getattr(model, "model", model)
    if not hasattr(inner, "blocks") or not hasattr(inner, "block_idx_to_name"):
        raise TypeError(f"{TAG} install() expects an Evo2 instance or its StripedHyena module, got {type(model).__name__}")
    return inner


class _Seg:
    """One captured region on one device: the consecutive blocks `blocks` ("segments") or the whole forward (`blocks is None`, "whole")."""

    def __init__(self, device, blocks):
        self.device = device
        self.blocks = blocks
        self.graph = None
        self.pool = None            # ONE private mempool per segment for the arm's life, kept in use by `sentinel` (an empty graph captured into it)
        self.sentinel = None        # so the pool's use count never reaches zero between captures and its blocks are reused by every re-capture
        self.static_in = None
        self.static_out = None
        self.entries = []           # [(dict, key, static_tensor)] hyena state slots rebound inside this region
        self.meta = []              # fp8_meta tensors (TE) inside this region: snapshot/restore around warm-up and capture

    def drop(self):
        self.graph = None           # the graph's blocks go back to the segment's pool (still held by the sentinel) for the next capture to reuse
        self.static_in = self.static_out = None
        self.entries = []
        self.meta = []


class _Runner:
    """Owns the dispatching `forward` bound on one StripedHyena instance, the block hooks (segments mode) and the live graphs."""

    def __init__(self, sh, *, warmup, min_prompt_tokens, verbose, capture_error_mode, step_tokens=1):
        import torch  # noqa: F401  (torch is a run-time dependency of the model, never of `import evo2_opt`)
        self.sh = sh
        self.prev_forward = None                  # the instance `forward` hook found at install (another party's) or None; set by install() via chain_install
        self.block_prev = [None for _ in sh.blocks]     # segments mode: per block, the instance `forward` hook found at install (or None), via chain_install
        self.warmup = int(warmup)
        self.step_tokens = int(step_tokens)   # tokens per captured step: 1 = the stock decode step; k+1 = a (B,k+1) target step of a caller's own loop
        self.verbose = verbose
        self.capture_error_mode = capture_error_mode
        self.min_prompt_tokens = min_prompt_tokens if min_prompt_tokens is not None else self._min_prompt_tokens(sh)
        self.segs = self._segments(sh)
        self.mode = "whole" if len(self.segs) == 1 else "segments"
        if self.mode == "whole":
            self.segs = [_Seg(self.segs[0].device, None)]
        self.devices = sorted({str(s.device) for s in self.segs})
        self.record = {"lever": LEVER, "armed": True, "installed": False, "mode": self.mode,
                       "segments": [{"device": str(s.device), "blocks": (None if s.blocks is None else [s.blocks[0], s.blocks[-1]])} for s in self.segs],
                       "captures": 0, "replays": 0, "eager_decode_steps": 0, "prefills": 0, "refusals": [], "disabled": None,
                       "capture_failures": 0, "fallbacks": 0, "captures_since_failure": 0,
                       "last_capture": None, "warmup": self.warmup, "min_prompt_tokens": self.min_prompt_tokens, "devices": self.devices}
        self.lens = {}             # device str -> static int32 (max_batch,) tensor = flash-attn cache_seqlens for the layers on that device
        self.mha_ref = None        # weakref to the InferenceParams the live graphs were captured against
        self.batch = None
        self.refused_mha = None    # weakref: the InferenceParams of a generate() call already refused (print once per call)
        self.pending_mha = None    # weakref: the InferenceParams whose FIRST decode step ran eagerly; its next decode step captures
        self.phase = "eager"       # segments mode: what the block hooks do — "eager" (stock), "capture", "replay"
        self.cur_ipd = None        # segments mode: the inference_params_dict of the step in flight (hooks read it)
        self.snaps = {}            # during a capture step: id(seg) -> (state clones, fp8_meta clones) taken BEFORE any segment ran
        self.side_streams = {}     # device str -> the one side stream used for warm-up and capture on that device
        if self.mode == "segments":
            self._install_hooks()

    # ----- model facts -----
    @staticmethod
    def _segments(sh):
        import torch
        segs = []
        for idx in range(len(sh.blocks)):
            dev = torch.device(sh.block_idx_to_device[idx])
            if dev.type != "cuda":
                raise CaptureRefused(f"block {idx} is on {dev}: CUDA graphs need cuda devices")
            if segs and segs[-1].device == dev:
                segs[-1].blocks.append(idx)
            else:
                if any(s.device == dev for s in segs):
                    raise CaptureRefused(f"blocks on {dev} are not consecutive (block {idx}): one segment per device expected")
                segs.append(_Seg(dev, [idx]))
        return segs

    @staticmethod
    def _min_prompt_tokens(sh):
        longest = 0
        for idx, blk in enumerate(sh.blocks):
            name = sh.block_idx_to_name(idx)
            if name == "mha":
                mha = blk.inner_mha_cls
                if not getattr(mha, "use_flash_attn", False):
                    raise CaptureRefused("config.use_flash_attn is off: the non-flash decode path indexes the kv cache with Python ints")
                if getattr(mha, "rotary_emb_dim", 0) % 16 != 0:
                    raise CaptureRefused(f"rotary_emb_dim {mha.rotary_emb_dim} % 16 != 0: stock takes the non-fused decode path")
                continue
            filt = getattr(blk, "filter", None)
            if filt is None or not hasattr(filt, "short_filter_length"):
                raise CaptureRefused(f"block {idx} ({type(blk).__name__}) has no hyena filter the lever knows")
            longest = max(longest, int(filt.short_filter_length))
            if filt.fir_inner_filter_length is not None:
                longest = max(longest, int(filt.fir_inner_filter_length))
        try:
            from vortex.model.attention import local_flash_attn_with_kvcache
        except Exception as exc:  # pragma: no cover - import failure is itself the refusal
            raise CaptureRefused(f"vortex.model.attention import failed: {exc!r}")
        if local_flash_attn_with_kvcache is None:
            raise CaptureRefused("flash-attn kvcache kernel absent (vortex.ops local_flash_attn_with_kvcache is None)")
        return longest - 1

    # ----- segments mode: block hooks -----
    def _install_hooks(self):
        anchors = {s.blocks[0]: s for s in self.segs}
        self.block_hooks = []
        for idx, blk in enumerate(self.sh.blocks):
            hook = self._make_hook(idx, blk, anchors.get(idx))
            self.block_prev[idx] = chain_install(blk, hook)
            self.block_hooks.append(hook)

    def _block_next(self, idx, *args, **kwargs):
        """Block idx's forward as it was before this arm's hook (previous instance hook, else the class method resolved at call time)."""
        return chain_next(self.sh.blocks[idx], self.block_prev[idx], *args, **kwargs)

    def _make_hook(self, idx, blk, seg):
        runner = self

        def hooked(u, inference_params=None, padding_mask=None, *args, **kwargs):
            if runner.phase == "eager" or padding_mask is not None:
                return runner._block_next(idx, u, inference_params, padding_mask, *args, **kwargs)
            if seg is None:
                return u, None                   # inside a replayed segment: its anchor already produced this segment's output
            return runner._anchor(seg, u), None
        hooked._evo2_gen_cudagraph_idx = idx
        return hooked

    def _remove_hooks(self):
        for idx, blk in enumerate(self.sh.blocks):
            if idx < len(getattr(self, "block_hooks", [])):
                chain_uninstall(blk, self.block_hooks[idx], self.block_prev[idx])

    def _next(self, *args, **kwargs):
        """The model's forward as it was before this arm: the previous instance hook if any, else the class method resolved at call time."""
        return chain_next(self.sh, self.prev_forward, *args, **kwargs)

    # ----- dispatch -----
    def forward(self, x, inference_params_dict=None, padding_mask=None):
        if inference_params_dict is None or padding_mask is not None:
            return self._next(x, inference_params_dict=inference_params_dict, padding_mask=padding_mask)
        mha = inference_params_dict.get("mha") if isinstance(inference_params_dict, dict) else None
        decode = (mha is not None and x.dim() == 2 and x.shape[1] == self.step_tokens and mha.seqlen_offset > 0
                  and self.record["disabled"] is None)
        if not decode:
            if mha is not None and mha.seqlen_offset == 0:
                self.record["prefills"] += 1
            return self._next(x, inference_params_dict=inference_params_dict, padding_mask=padding_mask)
        if self.refused_mha is not None and self.refused_mha() is mha:
            self.record["eager_decode_steps"] += 1
            return self._next(x, inference_params_dict=inference_params_dict, padding_mask=padding_mask)
        import torch
        with torch.inference_mode():
            live = (self.mha_ref is not None and self.mha_ref() is mha and x.shape[0] == self.batch
                    and all(s.graph is not None for s in self.segs) and self._slots_intact(inference_params_dict))
            if not live:
                if self.pending_mha is None or self.pending_mha() is not mha:
                    # the first decode step of this generate() call: stock eager (it rebinds the prefill's bf16 state views to the fp32
                    # tensors that every later step produces; those become the graphs' static state slots at the next step)
                    self.pending_mha = weakref.ref(mha)
                    self.record["eager_decode_steps"] += 1
                    return self._next(x, inference_params_dict=inference_params_dict, padding_mask=padding_mask)
                self.pending_mha = None
                try:
                    return self._capture(x, inference_params_dict)
                except CaptureRefused as exc:
                    self._refuse(exc.reason, mha)
                except Exception as exc:   # capture raised: the member cannot run here — refused BY NAME (the states restored first); never a silent eager fallback
                    if is_oom(exc): raise                                  # an out-of-memory propagates as itself
                    reason = f"capture raised {type(exc).__name__}: {str(exc).splitlines()[0][:300]}"
                    self.record["capture_failures"] = self.record.get("capture_failures", 0) + 1
                    self._refuse(reason, mha)
                    raise CaptureRefused(reason + " — CUDA-graph capture of the decode step cannot run on this stack; unset EVO2_OPT to generate on the stock loop") from exc
                self._drop_graphs()
                self.record["eager_decode_steps"] += 1
                return self._next(x, inference_params_dict=inference_params_dict, padding_mask=padding_mask)
            return self._replay(x, inference_params_dict)

    def _refuse(self, reason, mha):
        self.record["refusals"].append({"t": time.time(), "reason": reason})
        self.refused_mha = weakref.ref(mha) if mha is not None else None
        _say("REFUSED", reason + " — the stock eager decode step runs for this generate() call", self.verbose)

    def _slots_intact(self, ipd):
        for seg in self.segs:
            for d, k, static in seg.entries:
                if d.get(k) is not static:
                    return False
        return ipd.get("mha").lengths_per_sample is None

    def _drop_graphs(self):
        for seg in self.segs:
            seg.drop()
        self.mha_ref = None
        self.lens = {}
        self.phase = "eager"
        self.cur_ipd = None

    # ----- capture / replay -----
    def _collect_entries(self, ipd, batch, block_indices, device):
        """Every hyena state slot the decode step rebinds inside these blocks, with the shape checks that make replay legal."""
        entries = []
        for idx in block_indices:
            blk = self.sh.blocks[idx]
            name = self.sh.block_idx_to_name(idx)
            if name == "mha":
                continue
            params = ipd.get(name)
            if params is None:
                raise CaptureRefused(f"inference_params_dict has no '{name}' entry")
            filt = blk.filter
            expect = {"fir_state_dict": int(filt.short_filter_length) - 1}
            if filt.fir_inner_filter_length is not None:
                expect["fir_inner_state_dict"] = int(filt.fir_inner_filter_length) - 1
            else:
                expect["state_dict"] = None
            for attr, full in expect.items():
                d = getattr(params, attr, None)
                if d is None or idx not in d:
                    raise CaptureRefused(f"{name}.{attr}[{idx}] absent at the second decode step (prefill did not fill it)")
                t = d[idx]
                if full is not None and t.shape[-1] != full:
                    raise CaptureRefused(f"{name}.{attr}[{idx}] holds {t.shape[-1]} of {full} taps: the FIR state is still growing "
                                         f"(prompt shorter than {self.min_prompt_tokens} tokens; stock step_fir uses torch.cat until full)")
                if t.shape[0] != batch:
                    raise CaptureRefused(f"{name}.{attr}[{idx}] batch {t.shape[0]} != input batch {batch}")
                if t.device != device:
                    raise CaptureRefused(f"{name}.{attr}[{idx}] is on {t.device}, its block on {device}")
                entries.append((d, idx, t))
        return entries

    def _collect_meta(self, block_indices):
        """TE fp8_meta tensors (scale, scale_inv, amax_history) of every module inside these blocks (empty without Transformer Engine)."""
        import torch
        out, seen = [], set()
        for idx in block_indices:
            for m in self.sh.blocks[idx].modules():
                meta = getattr(m, "fp8_meta", None)
                if not isinstance(meta, dict):
                    continue
                for key in ("scaling_fwd", "scaling_bwd"):
                    obj = meta.get(key)
                    if obj is None:
                        continue
                    for attr in ("scale", "scale_inv", "amax_history"):
                        t = getattr(obj, attr, None)
                        if isinstance(t, torch.Tensor) and id(t) not in seen:
                            seen.add(id(t))
                            out.append(t)
        return out

    def _body(self, seg, ipd):
        """The captured region of one segment: the STOCK code on the static input, then copy-back + rebind of every rebound state slot."""
        mha = ipd["mha"]
        mha.lengths_per_sample = self.lens[str(seg.device)]
        try:
            if seg.blocks is None:
                out, _ = self._next(seg.static_in, inference_params_dict=ipd)
            else:
                u = seg.static_in
                for idx in seg.blocks:      # = stateful_forward's per-block call for the blocks of this device (no transfer inside a segment)
                    u, _ = self._block_next(idx, u, inference_params=ipd[self.sh.block_idx_to_name(idx)])
                out = u
        finally:
            mha.lengths_per_sample = None
        for d, k, static in seg.entries:
            new = d[k]
            if new is not static:
                if new.shape != static.shape or new.dtype != static.dtype or new.device != static.device:
                    raise CaptureRefused(f"state slot {k} changed inside the step: {tuple(static.shape)} {static.dtype} {static.device} -> "
                                         f"{tuple(new.shape)} {new.dtype} {new.device}")
                static.copy_(new)
                d[k] = static
        return out

    @staticmethod
    def _restore(seg, snap):
        states, metas = snap
        for (d, k, static), saved in zip(seg.entries, states):
            static.copy_(saved)
            d[k] = static
        for t, saved in zip(seg.meta, metas):
            t.copy_(saved)

    def _seg_capture(self, seg, ipd, u):
        """Warm up, capture and leave `seg.graph` ready; state as before the call. Does NOT run the real step."""
        import torch
        if u.device != seg.device:
            raise CaptureRefused(f"segment input on {u.device}, segment device {seg.device}")
        seg.static_in = u.clone()
        snap = self.snaps[id(seg)]          # taken in _capture before any segment of this step ran (segments touch disjoint blocks)
        cur = torch.cuda.current_stream(seg.device)
        side = self._side_stream(seg.device)   # ONE side stream per device for the life of the arm: cuBLAS keeps a workspace (32 MiB on H100)
                                               # per (handle, stream) that is never freed, so a new stream per capture leaks 32 MiB per generate() call
        side.wait_stream(cur)
        try:
            with torch.cuda.device(seg.device):
                with torch.cuda.stream(side):
                    for _ in range(self.warmup):
                        self._body(seg, ipd)
                        self._restore(seg, snap)
                cur.wait_stream(side)
                self._ensure_pool(seg, side)
                torch.cuda.synchronize(seg.device)
                graph = torch.cuda.CUDAGraph()
                # Low-level capture_begin/capture_end instead of the `torch.cuda.graph` context manager: the context manager calls
                # torch.cuda.empty_cache() at every capture, which under PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True unmaps the cached
                # free pages (the previous prefill's multi-GiB transient among them) and makes the NEXT generate() call's prefill re-map them,
                # a per-call stall that grows with the model. Nothing is unmapped here; the capture allocates from the segment's persistent pool.
                with torch.cuda.stream(side):
                    graph.capture_begin(pool=seg.pool, capture_error_mode=self.capture_error_mode)
                    try:
                        seg.static_out = self._body(seg, ipd)
                    finally:
                        graph.capture_end()          # always end a begun capture, also when the body raised
                cur.wait_stream(side)
        except BaseException:
            graph = None
            self._heal_capture_state(seg.device)   # a capture that raised may leave the CUDA RNG registered for capture
            self._restore(seg, snap)     # states back to their pre-step values; the stock eager step re-writes the kv slot
            ipd["mha"].lengths_per_sample = None
            raise
        self._restore(seg, snap)
        seg.graph = graph

    def _ensure_pool(self, seg, side):
        """Create the segment's persistent private pool once, held in use by an empty sentinel graph: re-capturing into a pool whose use
        count dropped to zero trips the caching allocator's private-pool bookkeeping under expandable_segments (torch 2.7.1), and a pool
        that stays in use lets every re-capture reuse the blocks the previous graph returned to it (no growth, no empty_cache needed)."""
        import torch
        if seg.pool is not None and seg.sentinel is not None:
            return
        seg.pool = torch.cuda.graph_pool_handle()
        sentinel = torch.cuda.CUDAGraph()
        with torch.cuda.stream(side):
            sentinel.capture_begin(pool=seg.pool, capture_error_mode=self.capture_error_mode)
            sentinel.capture_end()
        seg.sentinel = sentinel

    def _side_stream(self, device):
        import torch
        key = str(device)
        if key not in self.side_streams:
            self.side_streams[key] = torch.cuda.Stream(device=device)
        return self.side_streams[key]

    @staticmethod
    def _heal_capture_state(device):
        """After a failed capture: make the process usable again. torch.cuda.graph's __exit__ ends a capture that began; a failure INSIDE
        capture_begin (before the stream started capturing) skips __exit__ and leaves the CUDA generators registered for capture, so the next
        sampling op raises 'Offset increment outside graph capture'. One empty capture on a fresh graph + fresh pool runs the generator
        prologue/epilogue pair and clears that state; whatever still fails here is reported by the REFUSED line that follows, never hidden."""
        import torch
        try:
            with torch.cuda.device(device):
                cur = torch.cuda.current_stream(device)
                if torch.cuda.is_current_stream_capturing():
                    try:
                        torch.cuda.CUDAGraph().capture_end()
                    except Exception:
                        pass
                torch.cuda.synchronize(device)
                g = torch.cuda.CUDAGraph()
                side = torch.cuda.Stream(device=device)
                side.wait_stream(cur)
                with torch.cuda.graph(g, pool=torch.cuda.graph_pool_handle(), stream=side):
                    pass
                cur.wait_stream(side)
                del g
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
            return True
        except Exception as exc:  # noqa: BLE001
            _say("REFUSED", f"capture-state heal after a failed capture also raised {type(exc).__name__}: {exc}", True)
            return False

    def _seg_replay(self, seg, u):
        seg.static_in.copy_(u)
        seg.graph.replay()
        self.record["replays"] += 1
        return seg.static_out.clone()

    def _anchor(self, seg, u):
        """Segments mode: called from the anchor block's hook inside the stock stateful_forward."""
        if self.phase == "capture":
            self._seg_capture(seg, self.cur_ipd, u)
        return self._seg_replay(seg, u)

    def _fill_lens(self, off):
        for t in self.lens.values():
            t.fill_(off)

    def _capture(self, x, ipd):
        import torch
        t0 = time.perf_counter()
        mha = ipd["mha"]
        batch = int(x.shape[0])
        if batch > int(mha.max_batch_size):
            raise CaptureRefused(f"batch {batch} > inference_params.max_batch_size {mha.max_batch_size}")
        if mha.lengths_per_sample is not None:
            raise CaptureRefused("inference_params.lengths_per_sample is already set by the caller (per-sample lengths): not the stock loop")
        if not mha.key_value_memory_dict:
            raise CaptureRefused("mha.key_value_memory_dict is empty at the decode step (no prefill on this inference_params)")
        self._drop_graphs()                  # the previous call's graphs release their blocks into the persistent pools; nothing is unmapped
        self.batch = batch
        for seg in self.segs:
            blocks = seg.blocks if seg.blocks is not None else list(range(len(self.sh.blocks)))
            seg.entries = self._collect_entries(ipd, batch, blocks, seg.device)
            seg.meta = self._collect_meta(blocks)
            self.lens[str(seg.device)] = torch.zeros(int(mha.max_batch_size), dtype=torch.int32, device=seg.device)
        for idx, kv in mha.key_value_memory_dict.items():
            want = self.sh.block_idx_to_device[idx]
            if kv.device != torch.device(want):
                raise CaptureRefused(f"kv cache of layer {idx} on {kv.device}, its block on {want}")
        self._fill_lens(int(mha.seqlen_offset))
        self.snaps = {id(seg): ([static.clone() for _, _, static in seg.entries], [t.clone() for t in seg.meta]) for seg in self.segs}
        try:
            if self.mode == "whole":
                seg = self.segs[0]
                self._seg_capture(seg, ipd, x)
                logits = self._seg_replay(seg, x)          # the real step
            else:
                self.phase, self.cur_ipd = "capture", ipd
                try:
                    logits, _ = self._next(x, inference_params_dict=ipd)   # anchors capture + replay their segments; the rest is stock eager
                finally:
                    self.phase, self.cur_ipd = "eager", None
        except BaseException:
            # a segment failed after earlier segments may already have run their real step: put EVERY segment's states / fp8 metas back to
            # their pre-step values so the caller's stock eager fallback recomputes this step exactly (the kv slot is rewritten identically)
            for seg in self.segs:
                self._restore(seg, self.snaps[id(seg)])
            mha.lengths_per_sample = None
            raise
        finally:
            self.snaps = {}
        self.mha_ref = weakref.ref(mha)
        self.record["captures"] += 1
        self.record["captures_since_failure"] += 1
        self.record["installed"] = True
        cap = {"t": time.time(), "batch": batch, "seqlen_offset_at_capture": int(mha.seqlen_offset), "max_seqlen": int(mha.max_seqlen),
               "state_slots": sum(len(s.entries) for s in self.segs), "fp8_meta_tensors": sum(len(s.meta) for s in self.segs),
               "kv_layers": len(mha.key_value_memory_dict), "capture_s": round(time.perf_counter() - t0, 3), "warmup": self.warmup,
               "pool": "persistent per segment (sentinel-held)", "mode": self.mode, "graphs": len(self.segs), "devices": self.devices,
               "memory_reserved_mib": {str(d): torch.cuda.memory_reserved(d) >> 20 for d in self.devices}}
        self.record["last_capture"] = cap
        what = ("the whole decode step as one CUDA graph" if self.mode == "whole"
                else f"the decode step as {cap['graphs']} per-device CUDA graphs (segments {self.record['segments']}; hops, embedding, norm, "
                     f"unembed stock eager)")
        _say("INSTALLED", f"captured {what} on {','.join(self.devices)} (batch {batch}, kv max_seqlen {cap['max_seqlen']}, "
                          f"{cap['kv_layers']} attn layers via lengths_per_sample tensor, {cap['state_slots']} hyena state slots static, "
                          f"{cap['fp8_meta_tensors']} fp8_meta tensors snapshot-restored) at seqlen_offset {cap['seqlen_offset_at_capture']} in "
                          f"{cap['capture_s']} s (warmup {self.warmup}; persistent pool; reserved MiB {cap['memory_reserved_mib']}); replaying per token", self.verbose)
        return logits, ipd

    def _replay(self, x, ipd):
        mha = ipd["mha"]
        off = int(mha.seqlen_offset)
        if off + 1 > int(mha.max_seqlen):
            raise RuntimeError(f"{TAG} seqlen_offset {off} + 1 exceeds the kv cache max_seqlen {mha.max_seqlen} (stock sizes the cache "
                               f"to prompt + n_tokens; refusing to write past it)")
        self._fill_lens(off)
        if self.mode == "whole":
            return self._seg_replay(self.segs[0], x), ipd
        self.phase, self.cur_ipd = "replay", ipd
        try:
            logits, _ = self._next(x, inference_params_dict=ipd)
        finally:
            self.phase, self.cur_ipd = "eager", None
        return logits, ipd


class Handle:
    """What `install` returns (the generation-arm handle): `.name`, `.describe()`, `.uninstall()`, `.generate`
    (None: callers use the public `model.generate`), plus `.record` = the lever's live record (== `status(model)`)."""
    name = LEVER
    generate = None

    def __init__(self, model, record, knobs, verbose):
        self._model = model
        self.record = record
        self.knobs = knobs
        self._verbose = verbose

    @property
    def armed(self):
        return bool(self.record.get("armed"))

    def describe(self):
        r = self.record
        return {"name": LEVER, "knobs": dict(self.knobs), "numerics_class": "exact", "armed": bool(r.get("armed")), "mode": r.get("mode"),
                "segments": r.get("segments"), "devices": r.get("devices"), "min_prompt_tokens": r.get("min_prompt_tokens"),
                "captures": r.get("captures", 0), "replays": r.get("replays", 0), "eager_decode_steps": r.get("eager_decode_steps", 0),
                "refusals": [x["reason"] for x in r.get("refusals", [])], "disabled": r.get("disabled"),
                "what": "CUDA-graph replay of the stock vortex decode step (StripedHyena.forward on one token): whole step in one graph on one "
                        "device, one graph per device segment on several; captured at the 2nd decode step of each generate() call; token, "
                        "flash-attn cache_seqlens (lengths_per_sample tensor) and hyena FIR/IIR state slots static; sampling outside; stock "
                        "eager path by named refusal otherwise"}

    def uninstall(self):
        remove(self._model, verbose=self._verbose)


WARMUP = 2   # eager decode steps on the capture stream before the capture (lazy library initialisation and the private pool settle there); a fixed per-call cost, not a knob


def install(model, *, warmup=WARMUP, min_prompt_tokens=None, verbose=True, capture_error_mode="global", step_tokens=1):
    """Arm the CUDA-graph decode step on a constructed stock `Evo2` (or `StripedHyena`) instance; returns a `Handle`. Idempotent per
    instance. knobs: warmup (eager side-stream runs before capture, default 2), min_prompt_tokens (override of the derived FIR bound;
    informational), verbose (print the lever's lines), capture_error_mode (torch.cuda.graph's), step_tokens (tokens per captured step:
    1 = the stock decode step `x:(B,1)`; k+1 lets a caller that drives its own (B,k+1) target step through this model — e.g. a speculative
    decoder — have those captured and replayed instead; calls of any other width take the stock path)."""
    knobs = {"warmup": warmup, "min_prompt_tokens": min_prompt_tokens, "capture_error_mode": capture_error_mode, "step_tokens": step_tokens}
    sh = _striped(model)
    existing = getattr(sh, "_evo2_gen_cudagraph", None)
    if existing is not None:
        return Handle(model, existing.record, knobs, verbose)
    try:
        import torch
        if not torch.cuda.is_available():
            raise CaptureRefused("torch.cuda is not available")
        runner = _Runner(sh, warmup=warmup, min_prompt_tokens=min_prompt_tokens, verbose=verbose, capture_error_mode=capture_error_mode,
                         step_tokens=step_tokens)
    except CaptureRefused as exc:
        record = {"lever": LEVER, "armed": False, "installed": False, "refusals": [{"t": time.time(), "reason": exc.reason}],
                  "disabled": exc.reason}
        sh._evo2_gen_cudagraph_record = record
        _say("REFUSED", exc.reason + " — not armed; every call takes the stock path", verbose)
        return Handle(model, record, knobs, verbose)
    sh._evo2_gen_cudagraph = runner
    runner.prev_forward = chain_install(sh, runner.forward)   # instance attribute (nn.Module.__call__ resolves self.forward to it); chains onto any existing hook
    shape = ("ONE whole-step graph" if runner.mode == "whole"
             else f"{len(runner.segs)} per-device segment graphs {runner.record['segments']} (hops/embedding/norm/unembed stock eager)")
    _say("ARMED", f"CUDA graph replay of the stock decode step: {shape} on {','.join(runner.devices)} (capture at the second decode step of "
                  f"each generate() call after {runner.warmup} warm-up runs, the first runs stock eager; sampling outside the graph; refuses "
                  f"prompts shorter than {runner.min_prompt_tokens} tokens by name)", verbose)
    return Handle(model, runner.record, knobs, verbose)


arm = install


def remove(model, verbose=True):
    """Restore the instance: drop the dispatching forward, the block hooks and the graphs (their pools are released with them)."""
    sh = _striped(model)
    runner = getattr(sh, "_evo2_gen_cudagraph", None)
    if runner is None:
        return {"lever": LEVER, "armed": False, "removed": False}
    chain_uninstall(sh, runner.forward, runner.prev_forward)   # first: raises by name, touching nothing, if a later hook sits on top of this arm's
    runner._remove_hooks()
    runner._drop_graphs()
    for seg in runner.segs:
        seg.sentinel = None                  # last user of the pool: the pool becomes releasable
        seg.pool = None
    del sh._evo2_gen_cudagraph
    runner.side_streams = {}
    rec = dict(runner.record, armed=False, removed=True)
    sh._evo2_gen_cudagraph_record = rec
    try:
        import torch
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    except Exception:
        pass
    _say("REMOVED", f"stock forward restored after {rec['captures']} captures / {rec['replays']} replays", verbose)
    return rec


def status(model):
    """The lever's live record for this instance ({armed, installed, mode, captures, replays, eager_decode_steps, refusals, ...})."""
    sh = _striped(model)
    runner = getattr(sh, "_evo2_gen_cudagraph", None)
    if runner is not None:
        return runner.record
    return getattr(sh, "_evo2_gen_cudagraph_record", {"lever": LEVER, "armed": False, "installed": False, "refusals": []})
