"""evo2_opt.gen.specdec.core — the speculative-sampling loop for Evo 2 and the two mechanisms it adds to vortex's cached decode path.

Mechanism 1 — the multi-token warm-cache call (the target's target call). Stock vortex feeds ONE token per cached call: `HyenaCascade.forward`
takes `sequential_forward` once the layer's FIR state exists and that function keeps only the last position of a longer input
(`vortex/model/model.py:224-229, 328-333`). `install_hooks(sh)` binds, on every HyenaCascade INSTANCE of a StripedHyena `sh`, a forward that —
for a cached call with a time axis longer than 1 — runs the STOCK `sequential_forward` once per position, in order, and concatenates the
outputs. No Hyena arithmetic is re-implemented: each position's FIR/IIR update is the stock step function on the stock state slots. The
position-wise dense work (embedding, projections, out_filter_dense, MLP, norms, unembed) and attention (`flash_attn_with_kvcache`, k+1 causal
queries at `seqlen_offset`) see a (1, k+1, ·) input instead of (1, 1, ·): that call's logits define the distribution p the loop samples from.

Mechanism 2 — rollback = per-position state stack + index commit. A `StateStack` enumerates every recurrent state slot of `sh`'s inference
params (per Hyena layer: the featurizer FIR state and the inner FIR state or the IIR state; `vortex/model/cache.py`), snapshots all of them
before a call (entry 0) and, while recording, each layer's slots after every position the call consumes (entries 1..n). `commit(j)` copies
entry j back INTO the live slot tensors (copy_, never rebinding — CUDA-graph replays that hold those tensors static stay valid) = the state
after the first j positions of the call, bitwise what the call itself computed there. Attention needs no copy: rolling back = lowering
`seqlen_offset`; later calls overwrite the stale kv entries. Memory: (k+2) x the recurrent state bytes per model.

The loop (`generate`): prefill target and draft on the prompt (the stock forward with fresh inference params, as `Generator.generate`); the
first token is sampled from the target's prefill logits (as stock); then per round: the draft consumes the head token and proposes d_1..d_k one
single-token cached step at a time (q_i = its transformed distribution), the target consumes [head, d_1..d_k] in ONE cached call (p_0..p_k), the
rule of `rule.py` accepts a prefix, corrects the first rejected position from the residual or adds the bonus token, and both models are
committed to the state after their last kept position. The sampler transform is vortex's (`vortex/model/sample.py:30-59`: nan/inf scrub, top-k
on logits, temperature, top-p, softmax over the kept logits) applied in the logits' own dtype to target and draft alike. Every round is an
entry of the record (`round_rows`); every degraded path is a named event (`events`).
"""
import contextlib
from evo2_opt._oom import is_oom
import time

TAG = "[evo2-gen specdec]"


# =====================================================================================================================================
# 1. multi-token warm-cache path
# =====================================================================================================================================
def _is_hyena(sh, idx):
    return sh.block_idx_to_name(idx) != "mha"


def _make_filter_forward(stock_bound, filt, sh):
    """Instance-level forward for one HyenaCascade: positions of a cached multi-token call go through the stock sequential step in order."""
    import torch

    def forward(u, inference_params=None, padding_mask=None, *args, **kwargs):
        if (inference_params is not None and padding_mask is None and torch.is_tensor(u) and u.dim() == 3 and u.shape[1] > 1
                and filt.layer_idx in inference_params.fir_state_dict):
            stack = sh.__dict__.get("_evo2_gen_specdec_stack")
            ys = []
            for i in range(u.shape[1]):
                y, inference_params = stock_bound(u[:, i:i + 1], inference_params, None)
                ys.append(y)
                if stack is not None and stack.recording:
                    stack.record_layer(filt.layer_idx, i + 1)
            return torch.cat(ys, dim=1), inference_params
        out = stock_bound(u, inference_params, padding_mask, *args, **kwargs)
        stack = sh.__dict__.get("_evo2_gen_specdec_stack")
        if (stack is not None and stack.recording and inference_params is not None and torch.is_tensor(u) and u.dim() == 3
                and u.shape[1] == 1 and filt.layer_idx in inference_params.fir_state_dict):
            stack.record_layer(filt.layer_idx, stack.single_step_entry)
        return out
    forward._evo2_gen_specdec = True
    return forward


def install_hooks(sh):
    """Bind the multi-token forward on every HyenaCascade instance of `sh` (idempotent). Returns the number of layers hooked."""
    n = 0
    for idx, blk in enumerate(sh.blocks):
        if not _is_hyena(sh, idx):
            continue
        filt = blk.filter
        cur = filt.__dict__.get("forward")
        if cur is not None:
            if getattr(cur, "_evo2_gen_specdec", False):
                continue
            raise RuntimeError(f"{TAG} block {idx}.filter already carries an instance-level forward ({cur!r}); refusing to stack hooks")
        filt.forward = _make_filter_forward(filt.forward, filt, sh)
        n += 1
    return n


def remove_hooks(sh):
    n = 0
    for idx, blk in enumerate(sh.blocks):
        if not _is_hyena(sh, idx):
            continue
        cur = blk.filter.__dict__.get("forward")
        if cur is not None and getattr(cur, "_evo2_gen_specdec", False):
            del blk.filter.__dict__["forward"]
            n += 1
    sh.__dict__.pop("_evo2_gen_specdec_stack", None)
    return n


# =====================================================================================================================================
# 2. per-position state stack
# =====================================================================================================================================
class StateStack:
    """Recurrent state slots of one StripedHyena's inference_params_dict, with (depth+1) snapshot entries.

    slots: [(dict, layer_idx)] — for each Hyena layer its `fir_state_dict` slot and its `fir_inner_state_dict` (HCS/HCM) or `state_dict` (HCL)
    slot. entry e: one buffer per slot (allocated lazily with the slot's shape/dtype/device; re-allocated if the slot's dtype/shape changes,
    which stock does exactly once — the first cached step turns the prefill's bf16 views into fp32 tensors). `events` counts named paths."""

    def __init__(self, sh, ipd, depth):
        self.sh = sh
        self.depth = int(depth)
        self.slots = []
        self.layer_slots = {}
        for idx, blk in enumerate(sh.blocks):
            name = sh.block_idx_to_name(idx)
            if name == "mha":
                continue
            params = ipd[name]
            filt = blk.filter
            mine = [(params.fir_state_dict, idx)]
            if filt.fir_inner_filter_length is not None:
                mine.append((params.fir_inner_state_dict, idx))
            else:
                mine.append((params.state_dict, idx))
            self.layer_slots[idx] = list(range(len(self.slots), len(self.slots) + len(mine)))
            self.slots += mine
        self.entries = [[None] * len(self.slots) for _ in range(self.depth + 1)]
        self.recording = False
        self.single_step_entry = 0
        self.flat = None                      # [(flat buffer, [slot indices], [views])] after flatten(); snapshot/commit become single copies
        self.flat_entries = None
        self.events = {"realloc": 0, "rebind_restore": 0, "records": 0, "commits": 0, "snapshots": 0, "flatten": 0, "flat_fallbacks": 0}

    def flatten(self):
        """Rebind every slot to a view of ONE flat buffer per (device, dtype), values preserved, so that snapshot(e)/commit(e) are one copy per
        buffer instead of one per slot. Call after the first cached step (states are then fp32 tensors) and BEFORE a CUDA-graph capture adopts
        the slots as its statics (evo2_opt.gen.cudagraph rebinds slots back to the tensors it found at capture, i.e. to these views)."""
        import torch
        groups = {}
        for si, (d, k) in enumerate(self.slots):
            t = d[k]
            groups.setdefault((t.device, t.dtype), []).append(si)
        self.flat = []
        for (dev, dt), sis in groups.items():
            n = sum(self.slots[si][0][self.slots[si][1]].numel() for si in sis)
            buf = torch.empty(n, device=dev, dtype=dt)
            views, off = [], 0
            for si in sis:
                d, k = self.slots[si]
                t = d[k]
                v = buf[off:off + t.numel()].view(t.shape)
                v.copy_(t)
                d[k] = v
                views.append(v)
                off += t.numel()
            self.flat.append((buf, sis, views))
        self.flat_entries = [[None] * len(self.flat) for _ in range(self.depth + 1)]
        self.events["flatten"] += 1

    def _flat_intact(self):
        return self.flat is not None and all(self.slots[si][0][self.slots[si][1]] is v for buf, sis, views in self.flat for si, v in zip(sis, views))

    def _save(self, e, si):
        d, k = self.slots[si]
        t = d[k]
        buf = self.entries[e][si]
        if buf is None or buf.shape != t.shape or buf.dtype != t.dtype or buf.device != t.device:
            if buf is not None:
                self.events["realloc"] += 1
            buf = t.detach().clone()
            self.entries[e][si] = buf
        else:
            buf.copy_(t)

    def snapshot(self, e=0):
        """All slots -> entry e (one copy per flat buffer when flattened and the views are still the live slots; per slot otherwise)."""
        if self.flat is not None:
            if self._flat_intact():
                for gi, (buf, sis, views) in enumerate(self.flat):
                    fe = self.flat_entries[e][gi]
                    if fe is None:
                        self.flat_entries[e][gi] = buf.clone()
                    else:
                        fe.copy_(buf)
                self.events["snapshots"] += 1
                return
            self.flat, self.flat_entries = None, None            # a stock eager step rebound the slots: back to per-slot copies (named)
            self.events["flat_fallbacks"] += 1
        for si in range(len(self.slots)):
            self._save(e, si)
        self.events["snapshots"] += 1

    def record_layer(self, layer_idx, e):
        """This layer's slots -> entry e (called from the hook right after the layer consumed position e-1 of the call)."""
        if e > self.depth:
            raise RuntimeError(f"{TAG} state stack depth {self.depth} exceeded (entry {e})")
        for si in self.layer_slots[layer_idx]:
            self._save(e, si)
        self.events["records"] += 1

    def commit(self, e):
        """Live slots <- entry e (copy INTO the live tensors; rebind only if stock changed the slot's shape/dtype since, a named event)."""
        if self.flat is not None and self._flat_intact() and all(fe is not None for fe in self.flat_entries[e]):
            for gi, (buf, sis, views) in enumerate(self.flat):
                buf.copy_(self.flat_entries[e][gi])
            self.events["commits"] += 1
            return
        for si, (d, k) in enumerate(self.slots):
            buf = self.entries[e][si]
            if buf is None:
                raise RuntimeError(f"{TAG} commit({e}): slot {si} (layer {k}) was never recorded at that entry")
            cur = d[k]
            if cur.shape == buf.shape and cur.dtype == buf.dtype and cur.device == buf.device:
                cur.copy_(buf)
            else:
                d[k] = buf.clone()
                self.events["rebind_restore"] += 1
        self.events["commits"] += 1

    @contextlib.contextmanager
    def record(self, single_step_entry=0):
        self.recording, self.single_step_entry = True, single_step_entry
        try:
            yield self
        finally:
            self.recording = False

    def nbytes(self):
        n = sum(b.numel() * b.element_size() for row in self.entries for b in row if b is not None)
        if self.flat_entries:
            n += sum(fe.numel() * fe.element_size() for row in self.flat_entries for fe in row if fe is not None)
        return n


def set_offsets(ipd, off):
    for name in ("mha", "hcl", "hcm", "hcs"):
        p = ipd.get(name)
        if p is not None:
            p.seqlen_offset = int(off)


# =====================================================================================================================================
# 3. the stock sampler transform as a distribution
# =====================================================================================================================================
def transformed_probs(logits, top_k, top_p, temperature):
    """Dense (B, V) sampling distribution that `vortex.model.sample.sample(logits, top_k, top_p, temperature)` draws from, computed with the
    same operations in the same dtype (`sample.py:35-59`); greedy (top_k == 1) -> one-hot argmax. Returned as float32, rows normalised."""
    import torch
    from vortex.model.sample import modify_logits_for_top_p_filtering
    logits = torch.nan_to_num(logits)
    logits = torch.where(logits == float("-inf"), 0, logits)
    logits = torch.where(logits == float("inf"), 0, logits)
    V = logits.shape[-1]
    if top_k == 1:
        idx = logits.argmax(dim=-1, keepdim=True)
        return torch.zeros(logits.shape, dtype=torch.float32, device=logits.device).scatter_(-1, idx, 1.0)
    if top_k > 0:
        kk = min(top_k, V)
        logits_top, indices = torch.topk(logits, kk, dim=-1)
        if temperature != 1.0:
            logits_top /= temperature
        modify_logits_for_top_p_filtering(logits_top, top_p)
        pk = torch.softmax(logits_top, dim=-1)
        probs = torch.zeros(logits.shape, dtype=pk.dtype, device=logits.device).scatter_(-1, indices, pk)
    else:
        logits_top = logits / temperature if temperature != 1.0 else logits.clone()
        modify_logits_for_top_p_filtering(logits_top, top_p)
        probs = torch.softmax(logits_top, dim=-1)
    probs = probs.float()
    return probs / probs.sum(-1, keepdim=True)


# =====================================================================================================================================
# 4. draft construction on one device
# =====================================================================================================================================
@contextlib.contextmanager
def single_device_construction(device_index=0):
    """vortex places blocks by `torch.cuda.device_count()` (`model.py:683-702`); inside this context the count reads 1 so a model built here
    lands entirely on cuda:0 (the target keeps its own placement). A named, scoped patch: restored on exit."""
    import torch
    if device_index != 0:
        raise ValueError(f"{TAG} single_device_construction supports cuda:0 only (vortex names devices cuda:<i> from index 0)")
    real = torch.cuda.device_count
    torch.cuda.device_count = lambda: 1
    try:
        yield
    finally:
        torch.cuda.device_count = real


def load_draft(model_name, local_path):
    """The draft through the STOCK constructor (`Evo2(model_name, local_path=...)`, upstream's own download when local_path is None) on the
    first device, the constructor's kit wrap standing aside for it (evo2_opt.activation.plain_construction: the draft never scores)."""
    from evo2 import Evo2
    from evo2_opt import activation
    with single_device_construction(0), activation.plain_construction():
        return Evo2(model_name, local_path=local_path)


# =====================================================================================================================================
# 5. the loop
# =====================================================================================================================================
def _striped(model):
    return getattr(model, "model", model)


class Refused(ValueError):
    """A call the loop does not take, by name; the caller runs it on the stock loop."""


def generate(target, draft, prompt_seqs, *, k, n_tokens=500, temperature=1.0, top_k=4, top_p=1.0, batched=True, cached_generation=True,
             verbose=1, force_prompt_threshold=None, tokenizer=None, slide_keep=4096, progress=None):
    """Speculative sampling with `draft` proposing `k` tokens per round for `target` (Evo2 instances or StripedHyena modules carrying the hooks of
    `install_hooks`); the keywords of `Evo2.generate` (evo2/models.py:146; `batched` has no effect: prompts run one after the other, each at batch 1). Raises
    `Refused` (nothing run) for `cached_generation=False` and for `force_prompt_threshold` set. Returns vortex's `GenerationOutput` (sequences,
    logits = per prompt the (1, n, vocab) float32 target-call rows that judged / produced each emitted token, logprobs_mean) with `.speculative`
    = the per-prompt records. `progress` (a dict the caller holds, e.g. the member's record['progress']) is updated in place after every round:
    {prompt: index, emitted: tokens emitted so far of this prompt, rounds} — the call's live state for a caller that watches it."""
    from vortex.model.generation import GenerationOutput
    if not cached_generation:
        raise Refused("cached_generation=False (speculative sampling drives the cached path)")
    if force_prompt_threshold is not None:
        raise Refused(f"force_prompt_threshold={force_prompt_threshold} (prompt forcing is not part of this loop; Evo2.generate passes None)")
    tsh, dsh = _striped(target), _striped(draft)
    tok = tokenizer or getattr(target, "tokenizer", None)
    if tok is None:
        from vortex.model.tokenizer import CharLevelTokenizer
        tok = CharLevelTokenizer(512)
    seqs, logits_out, lp_means, records = [], [], [], []
    for i, prompt in enumerate(prompt_seqs):
        if progress is not None:
            progress.update(prompt=i, emitted=0, rounds=0)
        r = _generate_one(tsh, dsh, tok, prompt, n_tokens=int(n_tokens), temperature=temperature, top_k=top_k, top_p=top_p, k=int(k),
                          slide_keep=int(slide_keep), progress=progress)
        seqs.append(r["sequence"]); logits_out.append(r["logits"]); lp_means.append(r["logprobs_mean"]); records.append(r["record"])
        if verbose:
            print(f'Prompt: "{prompt}",\tOutput: "{r["sequence"]}",\tScore: {r["logprobs_mean"]}')
    out = GenerationOutput(sequences=seqs, logits=logits_out, logprobs_mean=lp_means)
    out.speculative = records
    return out


def _generate_one(tsh, dsh, tok, prompt, *, n_tokens, temperature, top_k, top_p, k, slide_keep=4096, draft_context=None, progress=None):
    import torch
    from vortex.model.generation import logits_to_logprobs
    from . import rule as R
    t_start = time.perf_counter()
    dev_t = torch.device(tsh.block_idx_to_device[0])
    dev_d = torch.device(dsh.block_idx_to_device[0])
    ids = list(tok.tokenize(prompt))
    L = len(ids)
    dmax = int(dsh.config.get("max_seqlen", 8192))
    if draft_context is not None:                        # a smaller draft context forces the sliding re-prefill early (named in the record)
        dmax = min(dmax, int(draft_context))
    if slide_keep < 8 * (k + 2) or dmax < 16 * (k + 2):
        raise Refused(f"slide_keep {slide_keep} / draft context {dmax} too small for k={k}")
    greedy = top_k == 1
    x = torch.tensor(ids, dtype=torch.long, device=dev_t)[None]
    tot = L + n_tokens                                   # = vortex's tot_length: the target's kv caches are sized exactly as stock sizes them
    ipd_t = tsh.initialize_inference_params(max_seqlen=tot); ipd_t["mha"].max_batch_size = 1
    d_tot = min(tot, dmax)                               # the draft's caches never exceed its context; it slides instead (below)
    ipd_d = dsh.initialize_inference_params(max_seqlen=d_tot); ipd_d["mha"].max_batch_size = 1
    d_base = 0                                           # absolute position of the draft cache's row 0 (moves at each sliding re-prefill)
    bg = tsh.__dict__.get("_evo2_gen_specdec_targetgraph")
    if bg is not None:
        bg.set_width(k + 1)                              # the target's cudagraph runner captures width-(k+1) calls during this call; restored at the end
    tstack = StateStack(tsh, ipd_t, depth=k + 1)
    dstack = StateStack(dsh, ipd_d, depth=k + 1)
    tsh._evo2_gen_specdec_stack = tstack
    dsh._evo2_gen_specdec_stack = dstack
    rounds = []
    events = {"k_shrunk_at_end": 0, "residual_zero_mass": 0, "draft_slides": 0, "draft_slide_positions": [], "draft_stack_unflattened": 0}
    t_draft = t_target = 0.0
    try:
        with torch.inference_mode():
            # ---- prefill (the stock forward, offsets 0); the draft's prompt is its last dmax - (k+2) tokens when the prompt is longer ------
            logits_t, _ = tsh(x, inference_params_dict=ipd_t)
            if L > d_tot - (k + 2):
                d_base = L - min(slide_keep, d_tot - (k + 2))
                events["draft_slides"] += 1; events["draft_slide_positions"].append(L)
            _ = dsh(x[:, d_base:].to(dev_d), inference_params_dict=ipd_d)
            V = logits_t.shape[-1]
            last = logits_t[:, -1]                                   # the target's logits for position L
            p0 = transformed_probs(last, top_k, top_p, temperature)
            head = int(p0.argmax(-1)) if greedy else R.draw(p0[0])
            out_tokens = [head]
            out_logits = [last[0].float()]
            torch.cuda.synchronize(dev_t)
            t_prefill = time.perf_counter()
            m = L                                                    # position of the head token (chosen, not yet consumed by either model)
            set_offsets(ipd_t, L); set_offsets(ipd_d, L - d_base)   # stock: seqlen_offset = prompt length at the first cached step
            d_consumed = L                                           # the draft's state has consumed positions < d_consumed
            n_round = 0
            tested_tot = acc_tot = prop_tot = 0; alpha_sum = 0.0
            while len(out_tokens) < n_tokens:
                n_round += 1
                k_eff = min(k, n_tokens - len(out_tokens))           # never write kv rows past tot-1 (= stock's cache size), never over-generate
                if k_eff < k:
                    events["k_shrunk_at_end"] += 1
                seq_now = ids + out_tokens                           # tokens at positions 0..m (all final)
                t0 = time.perf_counter()
                # ---- draft context: slide (re-prefill on the last `slide_keep` final tokens) when this round would write past its cache ----
                if (m + k_eff) - d_base > d_tot - 1:
                    keep = min(slide_keep, m, d_tot - (k + 2))       # re-prefill positions [m-keep, m); room for this round's k steps + replay
                    d_base = m - keep
                    for name in ("mha", "hcl", "hcm", "hcs"):        # fresh recurrent state; the kv tensor is re-used by the prefill's write
                        pp = ipd_d[name]
                        for attr in ("fir_state_dict", "fir_inner_state_dict", "state_dict"):
                            if hasattr(pp, attr):
                                getattr(pp, attr).clear()
                    set_offsets(ipd_d, 0)
                    dsh(torch.tensor([seq_now[d_base:m]], dtype=torch.long, device=dev_d), inference_params_dict=ipd_d)
                    d_consumed = m
                    dstack.__init__(dsh, ipd_d, depth=k + 1)         # the slots are new objects after a prefill
                    events["draft_slides"] += 1; events["draft_slide_positions"].append(m)
                # ---- draft: replay not-yet-consumed final tokens (only d_k after an all-accepted round), then head, then propose ------------
                qs = []
                for pos in range(d_consumed, m):
                    set_offsets(ipd_d, pos - d_base)
                    dsh(torch.tensor([[seq_now[pos]]], dtype=torch.long, device=dev_d), inference_params_dict=ipd_d)
                d_consumed = m
                cur = torch.tensor([[head]], dtype=torch.long, device=dev_d)
                d_toks = []
                for j in range(1, k_eff + 1):                        # step j consumes position m+j-1 -> q_j for position m+j
                    set_offsets(ipd_d, m + j - 1 - d_base)
                    ld, _ = dsh(cur, inference_params_dict=ipd_d)
                    if dstack.flat is None and dstack.events["flat_fallbacks"] == 0 and not events["draft_stack_unflattened"]:
                        try:
                            dstack.flatten()                         # after the first cached step the slots are fp32 tensors; before a capture
                        except Exception as exc:
                            if is_oom(exc):
                                raise
                            events["draft_stack_unflattened"] += 1   # named in the record; snapshots stay per slot
                    dstack.snapshot(j)                               # from the loop, not the hook: a graph-replayed step runs no Python
                    q = transformed_probs(ld[:, -1], top_k, top_p, temperature)
                    nxt = int(q[0].argmax(-1)) if greedy else R.draw(q[0])
                    cur = torch.tensor([[nxt]], dtype=torch.long, device=dev_d)
                    qs.append(q[0].to(dev_t)); d_toks.append(nxt)
                d_consumed = m + k_eff                               # consumed head .. d_{k_eff-1}
                torch.cuda.synchronize(dev_d); t1 = time.perf_counter(); t_draft += t1 - t0
                # ---- target: ONE cached call over [head, d_1..d_k] at positions m..m+k (entries 1..k+1 recorded per layer by the hook) -----
                set_offsets(ipd_t, m)
                block = torch.tensor([[head] + d_toks], dtype=torch.long, device=dev_t)
                with tstack.record():
                    lt, _ = tsh(block, inference_params_dict=ipd_t)  # (1, k_eff+1, V); lt[:, i] judges d_{i+1}; lt[:, k_eff] -> the bonus
                P = transformed_probs(lt[0], top_k, top_p, temperature)      # (k_eff+1, V) float32
                torch.cuda.synchronize(dev_t); t2 = time.perf_counter(); t_target += t2 - t1
                # ---- the rule ----------------------------------------------------------------------------------------------------------
                drafts = block[0, 1:]
                Q = torch.stack(qs) if qs else P[:0]
                if greedy:
                    n_acc = R.greedy_prefix(P[:k_eff], drafts)
                    alphas = [1.0] * k_eff
                else:
                    u = torch.rand(k_eff, dtype=torch.float64, device=dev_t)
                    n_acc = R.accepted_prefix(P[:k_eff], Q, drafts, u)
                    alphas = [R.acceptance_probability(P[i], Q[i]) for i in range(k_eff)]
                if n_acc == k_eff:                                   # all accepted (or nothing proposed): one more token from the next row
                    row = P[k_eff]
                    new = int(row.argmax(-1)) if greedy else R.draw(row)
                    kind = "bonus"
                else:                                                # first rejection at draft n_acc+1: the residual's token
                    if greedy:
                        new = int(P[n_acc].argmax(-1))
                    else:
                        res, mass = R.residual(P[n_acc], Q[n_acc])
                        if mass <= 0.0:
                            events["residual_zero_mass"] += 1
                        new = R.draw(res)
                    kind = "corrected"
                    tstack.commit(n_acc + 1)                         # target state <- after block positions 0..n_acc (head + accepted)
                    if n_acc + 1 < k_eff:                            # draft consumed through d_{k_eff-1}; keep through d_{n_acc}
                        dstack.commit(n_acc + 1)
                        d_consumed = m + n_acc + 1
                kept = d_toks[:n_acc] + [new]
                for i, t_id in enumerate(kept):
                    if len(out_tokens) >= n_tokens:
                        break
                    out_tokens.append(t_id)
                    out_logits.append(lt[0, i].float())              # the target-call row that judged / produced this token
                n_tested = n_acc + (1 if kind == "corrected" else 0)          # drafts after the first rejection are never tested
                tested_tot += n_tested; acc_tot += n_acc; prop_tot += k_eff; alpha_sum += sum(alphas[:n_tested])
                rounds.append({"r": n_round, "m": m, "k": k_eff, "accepted": n_acc, "tested": n_tested, "extra": kind})
                if progress is not None:
                    progress["emitted"] = len(out_tokens); progress["rounds"] = n_round      # the call's live state (the accept test read the target's rows: the round's device work is complete)
                m = m + n_acc + 1
                head = new
            torch.cuda.synchronize(dev_t)
            t_end = time.perf_counter()
    finally:
        tsh.__dict__.pop("_evo2_gen_specdec_stack", None); dsh.__dict__.pop("_evo2_gen_specdec_stack", None)
        if bg is not None:
            bg.restore()
    gen_ids = torch.tensor(out_tokens[:n_tokens], dtype=torch.long)
    logits = torch.stack(out_logits[:n_tokens])[None]                # (1, n, V) float32, like vortex's `scores`
    lp = logits_to_logprobs(logits, gen_ids[None].to(logits.device)).float().cpu().numpy()
    n_out = int(gen_ids.numel())
    record = {"prompt_len": L, "n_tokens": n_out, "k": k, "greedy": greedy, "temperature": temperature, "top_k": top_k, "top_p": top_p,
              "rounds": n_round, "target_calls": n_round + 1, "proposed": prop_tot, "accepted": acc_tot, "tested": tested_tot,
              "accepted_per_proposed": acc_tot / max(prop_tot, 1),             # what sets tokens per target call (untested tail drafts count as lost)
              "acceptance_per_test": acc_tot / max(tested_tot, 1),             # observed P(accept | tested)
              "alpha_expected": alpha_sum / max(tested_tot, 1),                # mean Σ min(p, q) over the same tested positions (its expectation)
              "tokens_per_target_call": (n_out - 1) / max(n_round, 1),
              "wall_s": {"total": t_end - t_start, "prefill_and_first_token": t_prefill - t_start, "decode": t_end - t_prefill,
                         "draft": t_draft, "target": t_target},
              "decode_tok_s": (n_out - 1) / max(t_end - t_prefill, 1e-9), "draft_context": dmax, "draft_cache_len": d_tot,
              "events": {**events, "target_stack": dict(tstack.events), "draft_stack": dict(dstack.events)},
              "state_stack_bytes": {"target": tstack.nbytes(), "draft": dstack.nbytes()}, "slide_keep": slide_keep,
              "target_graph": (None if bg is None else bg.describe()), "round_rows": rounds}
    return {"sequence": tok.detokenize(gen_ids.tolist()), "logits": logits, "logprobs_mean": float(lp.mean()), "record": record,
            "token_ids": gen_ids.tolist()}
