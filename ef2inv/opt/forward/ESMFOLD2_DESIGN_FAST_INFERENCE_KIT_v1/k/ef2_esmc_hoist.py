"""ef2_esmc_hoist — the ESMC-6B feature pass of a design step computed for the binder's rows only: the target chain's hidden states
hoisted out of the loop (numerics class: exact).

The fold's language-model input is `[BOS] target [EOS] [BOS] binder [EOS]` (one `[BOS]…[EOS]` run per chain, the binder last) with a
chain id per position (`sequence_id`), and the trunk masks attention to positions of the same chain
(`modeling_esmc._scaled_dot_product_attention`: `mask = sequence_id[:, :, None] == sequence_id[:, None, :]`). Two facts follow for the
rows of the LAST chain (the "live" rows, `[b0, L)`): no other row attends to them, and they attend to no other row. Every other module of
a block (TE LayerNormLinear, the q/k LayerNorms, the rotary, TE Linear, TE LayerNormMLP, the residual updates, the final LayerNorm) is
row-wise. So within one design — the target never changes, the binder changes every step — the 81 hidden states of the rows `[0, b0)`
are the same tensor on every step, and the live rows' states are a function of the live rows alone.

This lever wraps the trunk's `forward` (an instance attribute on the shared `ESMCModel`, over whatever is installed — install it AFTER
`ef2_esmc_graph` so it is the outer wrapper; a wrapper under a CUDA-graph replay would never run). Per call of the feature-pass form
(`input_ids=[B, L]` CUDA, `sequence_id=[B, L]`, `output_hidden_states=True`, no grad) it reads the two small integer tensors back to the
host (ONE transfer), finds the live block, and

  * FULL  — first call of a (shape, flags) key, or the frozen rows' ids / the chain ids differ from the remembered ones: the inner forward
            runs (the graph replay when `ef2_esmc_graph` is under it), its `hidden_states` [n_layers+1, B, L, D] are copied into this
            lever's persistent tensor for the key, the inner output is returned as is;
  * HOIST — the frozen rows match: the reduced forward runs — per block the block's own modules on the live rows, the rotary at the rows'
            ORIGINAL positions (`RotaryEmbedding.forward(q, k, seqlen_offset=b0)`, the module's own parameter), scaled-dot-product
            attention of the live queries against key / value rows AT THEIR ORIGINAL COLUMNS (zero-filled [B, L, D] buffers whose live
            rows are written each layer; the frozen columns are masked out exactly as in the full pass, so the softmax reduction runs over
            the same L columns in the same blocks — the binder-alone, re-based form is NOT bitwise: max|d| 7.8e-3 at block 0 on the pinned
            stack), the residual updates, and each block input / the final norm written into the live rows of the persistent tensor; an
            `ESMCOutput` over that tensor is returned (`hidden_states` = it, `last_hidden_state` = its last layer). The reduced forward is
            CUDA-graph captured per key at its first use (eager it is as launch-bound as the full pass) and replayed.

Exact on the pinned stack because the per-row arithmetic of those kernels does not depend on the number of rows (TE LayerNormLinear /
Linear / LayerNormMLP, torch LayerNorm, the rotary: tensor-equal on a row subset at 195 / 431 / 700 tokens) and masked SDPA is per-query-row
invariant when the key columns keep their positions. ANCHOR: the first hoisted call of every key also runs the inner forward on the same
input and the two [n_layers+1, B, L, D] tensors must be tensor-equal — all rows: the frozen rows prove the isolation and the freshness of
the remembered states, the live rows the reduced route. Not equal → that call returns the inner result and the lever switches itself off
BY NAME for the rest of the process (`fallback['anchor']`, every later call to the inner forward); a reduced-forward capture that raises
does the same (`capture_failed`); a trunk this file does not describe (flash-attn varlen blocks, SAE taps, attention weights requested)
is never hoisted (`structure:<what>`, decided at enable). PER CARD: the row-count invariance is a property of the kernels the stack selects on
the device, so the lever installs only on a compute capability listed in `PROVEN_CC` (sm_90: every module tensor-equal on a row subset at
199-804 positions); on any other capability it STEPS ASIDE BY NAME at enable — the trunk's forward is left untouched, `stats()` reports
`stepped_aside='cc_unproven:sm_NN'`, served 0 (sm_80, A100: TransformerEngine's cuBLASLt GEMMs pick their algorithm by row count there —
`layernorm_qkv` / `out_proj` / the MLP differ between 82 and 199-435 rows by up to 1 bf16 ulp of the activations, so no reduced pass reproduces
the full pass's bits; SDPA and the LayerNorms are row-invariant on both cards). The run-time anchor stays for the unexpected case (a listed
capability whose kernels diverge). By-design inner calls, counted as words: `grad` (grad-enabled calls — the
pseudo-perplexity passes do not come through `forward` at all), `form` (another argument set, CPU tensors, `output_hidden_states` not
True), `layout` (padding present, one chain only, or a live block that does not start at the same column in every batch row),
`capturing` (called while the current stream is being captured by someone else: nothing of this lever may enter a foreign graph).

Memory: the persistent hidden-state tensor per key ((n_layers+1) · B · L · D elements in the dtype the trunk returns — float32 under the
loop's autocast, `torch.stack` promoting the bf16 block inputs with the fp32 final norm), two [B, L, D] key/value buffers, the
[B, 1, L-b0, L] mask and the reduced graph's private pool (the live rows' intermediates); `POOL_BYTES` is the figure the design kit's
memory planner is given for all of it. The hoisted pass is bound by the weight read, so its time is flat in L.

    import ef2_esmc_hoist as eh
    h = eh.enable(model)        # model: an ESMFold2 model (._esmc), an ESMCForMaskedLM (.esmc) or the ESMCModel; idempotent per trunk
    eh.stats(model)             # {'served', 'full', 'anchored', 'captures', 'replays', 'entries', 'fallback': {word: n}, 'off': reason|None, 'live': 'rows/L', 'cc': 'sm_NN'|None, 'stepped_aside': 'cc_unproven:sm_NN'|None}
    eh.release(model)           # drop the remembered states and the captured graphs (the wrapper stays; the next call is FULL)
    eh.disable(model)           # hands the trunk back the forward that was installed under this lever
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

_ATTR = "_ef2_esmc_hoist"
NAME = "ef2_esmc_hoist"
KWARGS = frozenset(("input_ids", "sequence_id", "output_hidden_states"))
POOL_BYTES = 0.75 * float(2 ** 30)         # what fastkit hands the memory planner for this lever's resident tensors + graph pool

# Compute capabilities on which the reduced pass is PROVEN tensor-equal to the full pass (every row-wise kernel the pinned stack selects there is
# row-count invariant). A capability absent here: the lever steps aside by name at enable (nothing installed):
#   (8, 0) A100: TransformerEngine LayerNormLinear / Linear / LayerNormMLP through cuBLASLt choose their GEMM algorithm by row count, so a
#   row subset differs from the full pass by up to one bf16 ulp from block 0 on; SDPA / LayerNorm are equal.
PROVEN_CC = {
    (9, 0): "sm_90 (H100 / H200): TE LayerNormLinear, Linear, LayerNormMLP, torch LayerNorm, the rotary and masked SDPA tensor-equal on a row subset at 199 / 435 / 704 / 804 positions",
}


def cc_word(cc) -> str:
    return f"sm_{int(cc[0])}{int(cc[1])}"


def cc_route(cc=None):
    """(proven, 'sm_NN') for a compute capability (default: the current CUDA device's; (False, None) without CUDA)."""
    if cc is None:
        if not torch.cuda.is_available():
            return False, None
        cc = torch.cuda.get_device_capability()
    cc = (int(cc[0]), int(cc[1]))
    return cc in PROVEN_CC, cc_word(cc)


# --------------------------------------------------------------------------------------------------------------------------- host side
def live_block(ids: np.ndarray, seq: np.ndarray):
    """The hoistable layout of one LM input, decided on host copies ([B, L] integer arrays): returns (b0, key_bytes) — the live rows are
    `[b0, L)`, the last chain of every batch row, and `key_bytes` identifies everything the frozen rows' hidden states depend on (the frozen
    ids and every chain id) — or (None, reason) when the layout is not hoisted: `layout` = padding (a negative chain id), a single chain
    (nothing frozen), a last chain that starts at different columns across the batch, or chain ids that are not one contiguous run."""
    if ids.ndim != 2 or seq.shape != ids.shape or ids.shape[1] < 2:
        return None, "layout"
    if (seq < 0).any():
        return None, "layout"
    last = seq[:, -1:]
    live = seq == last                                     # [B, L]
    b0s = live.argmax(axis=1)
    b0 = int(b0s[0])
    if b0 <= 0 or not (b0s == b0).all():
        return None, "layout"
    if not live[:, b0:].all() or live[:, :b0].any():        # the last chain is exactly the trailing block (chain ids are runs)
        return None, "layout"
    return b0, ids[:, :b0].tobytes() + b"|" + seq.tobytes()


def structure_problem(esmc) -> str | None:
    """None when the trunk is the form this file computes (SDPA blocks with QK-LayerNorm and the plain rotary, no SAE taps, no attention
    weights); else the word naming what differs."""
    try:
        from transformers.models.esmc import modeling_esmc as ME
    except Exception as e:                                  # noqa: BLE001
        return f"import:{type(e).__name__}"
    if not isinstance(esmc, ME.ESMCModel):
        return f"type:{type(esmc).__name__}"
    if getattr(esmc, "_use_flash_attn", False):
        return "flash_attn_varlen"
    if len(getattr(esmc, "_sae_models", {}) or {}):
        return "sae"
    if getattr(esmc.config, "output_attentions", False):
        return "output_attentions"
    tr = getattr(esmc, "transformer", None)
    if tr is None or not hasattr(tr, "blocks") or not hasattr(tr, "norm") or not hasattr(esmc, "embed"):
        return "modules"
    for blk in tr.blocks:
        at = getattr(blk, "attn", None)
        if type(at) is not ME.MultiHeadAttention or type(getattr(at, "rotary", None)) is not ME.RotaryEmbedding:
            return f"attn:{type(at).__name__}"
        if not all(hasattr(at, n) for n in ("layernorm_qkv", "q_ln", "k_ln", "out_proj", "n_heads", "d_head")) or not hasattr(blk, "ffn") or not hasattr(blk, "scaling_factor"):
            return "attn_modules"
    return None


# --------------------------------------------------------------------------------------------------------------------------- the lever
class _Entry:
    """Per (B, L, b0, dtypes, autocast / inference flags): the persistent hidden states, the static inputs and the reduced graph."""

    def __init__(self, B, L, b0, ids_like, seq, n_states, d_model):
        self.B, self.L, self.b0 = B, L, b0
        self.key_bytes = None                                # what the frozen rows of `hs` were computed from (None = nothing remembered)
        self.ids_live = torch.zeros((B, L - b0), dtype=ids_like.dtype, device=ids_like.device)
        self.mask = (seq[:, b0:].unsqueeze(-1) == seq.unsqueeze(-2)).unsqueeze(1).contiguous()   # [B, 1, L-b0, L]: the full pass's mask, live rows
        self.hs = None                                        # [n_states, B, L, D], allocated from the first FULL output (its dtype)
        self.kv = {}                                          # dtype -> (k_pad, v_pad) [B, L, D] zero-filled; live rows written per layer
        self.graph = None
        self.anchored = False                                 # the run-time anchor met for this key
        self.n_anchor = 0
        self.n_states, self.d_model = n_states, d_model


class _Hoist:
    def __init__(self, esmc, inner, anchor=1, graphs=True, n_warmup=2, max_entries=2, cc=None):
        self.esmc, self.inner = esmc, inner
        self.anchor, self.graphs, self.n_warmup, self.max_entries = int(anchor), bool(graphs), int(n_warmup), int(max_entries)
        self.entries = {}
        self.installed = False                                # set by enable(): the trunk's forward is this object
        proven, self.cc = cc_route(cc) if (cc is not None or torch.cuda.is_available()) else (True, None)   # no CUDA (unit tests on CPU): nothing to key on
        self.stepped_aside = None if proven else f"cc_unproven:{self.cc}"   # decided once, here: the lever does not install on this card (by name; stats() carries it)
        problem = structure_problem(esmc)
        self.off = None if problem is None else f"structure:{problem}"   # None = engaged; else the reason every call goes to the inner forward
        self.stats = dict(served=0, full=0, anchored=0, captures=0, replays=0, fallback={})
        self.live = None

    # -- census helpers
    def _fb(self, word, n=1):
        self.stats["fallback"][word] = self.stats["fallback"].get(word, 0) + n

    def _switch_off(self, word):
        self.off = word
        self.entries.clear()

    # -- the reduced forward: the trunk's own modules on the live rows, keys / values at their original columns
    def _reduced(self, ent: _Entry):
        esmc, tr = self.esmc, self.esmc.transformer
        B, L, b0 = ent.B, ent.L, ent.b0
        nl = L - b0
        hs = ent.hs
        x = esmc.embed(ent.ids_live)                                                  # [B, nl, D]
        for i, blk in enumerate(tr.blocks):
            hs[i, :, b0:].copy_(x)                                                    # collected[i] = the block's input (torch.stack copies it in the full pass)
            at = blk.attn
            H, Dh = at.n_heads, at.d_head
            qkv = at.layernorm_qkv(x)
            q, k, v = torch.chunk(qkv, 3, dim=-1)
            q = at.q_ln(q).to(q.dtype)
            k = at.k_ln(k).to(q.dtype)
            q = q.unflatten(-1, (H, Dh)); k = k.unflatten(-1, (H, Dh))                # MultiHeadAttention._apply_rotary, at the rows' original positions
            q, k = at.rotary(q, k, seqlen_offset=b0)
            q = q.flatten(-2, -1); k = k.flatten(-2, -1)
            bufs = ent.kv.get(k.dtype)
            if bufs is None:
                bufs = (torch.zeros((B, L, H * Dh), dtype=k.dtype, device=k.device), torch.zeros((B, L, H * Dh), dtype=k.dtype, device=k.device))
                ent.kv[k.dtype] = bufs
            kp, vp = bufs
            kp[:, b0:].copy_(k); vp[:, b0:].copy_(v)
            q4 = q.view(B, nl, H, -1).transpose(1, 2)                                 # _scaled_dot_product_attention's masked branch, live query rows
            k4 = kp.view(B, L, H, -1).transpose(1, 2)
            v4 = vp.view(B, L, H, -1).transpose(1, 2)
            ctx = F.scaled_dot_product_attention(q4, k4, v4, ent.mask)
            _, h, _, d_out = ctx.shape
            ctx = ctx.transpose(1, 2).reshape(B, nl, h * d_out)
            attn_out = at.out_proj(ctx)
            x = x + attn_out / blk.scaling_factor                                     # UnifiedTransformerBlock.forward
            x = x + blk.ffn(x) / blk.scaling_factor
        hs[len(tr.blocks), :, b0:].copy_(tr.norm(x))                                  # collected[n_layers] = the post-norm output

    def _capture(self, ent: _Entry):
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(self.n_warmup):
                self._reduced(ent)
        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._reduced(ent)
        torch.cuda.synchronize()
        ent.graph = g
        self.stats["captures"] += 1

    def _output(self, ent: _Entry):
        from transformers.models.esmc.modeling_esmc import ESMCOutput
        return ESMCOutput(last_hidden_state=ent.hs[-1], hidden_states=ent.hs, sae_outputs=None, attentions=None)

    # -- the trunk's forward
    def __call__(self, *args, **kw):
        if self.off is not None:
            self._fb(self.off)
            return self.inner(*args, **kw)
        input_ids = kw.get("input_ids", args[0] if args else None)
        sequence_id = kw.get("sequence_id"); ohs = kw.get("output_hidden_states")
        if torch.is_grad_enabled():
            self._fb("grad"); return self.inner(*args, **kw)
        form = (torch.is_tensor(input_ids) and input_ids.dim() == 2 and torch.is_tensor(sequence_id) and sequence_id.shape == input_ids.shape
                and sequence_id.device == input_ids.device and ohs is True and len(args) <= 1 and set(kw) <= KWARGS
                and not input_ids.dtype.is_floating_point and not sequence_id.dtype.is_floating_point
                and (input_ids.is_cuda or not self.graphs))
        if not form:
            self._fb("form"); return self.inner(*args, **kw)
        if input_ids.is_cuda and torch.cuda.is_current_stream_capturing():
            self._fb("capturing"); return self.inner(*args, **kw)
        host = torch.stack((input_ids.to(torch.int64), sequence_id.to(torch.int64))).cpu().numpy()     # ONE device->host transfer: [2, B, L]
        b0, key_bytes = live_block(host[0], host[1])
        if b0 is None:
            self._fb(key_bytes); return self.inner(*args, **kw)
        B, L = input_ids.shape
        ac = torch.is_autocast_enabled(input_ids.device.type) if input_ids.is_cuda else False
        key = (B, L, b0, input_ids.dtype, sequence_id.dtype, str(input_ids.device), ac, (torch.get_autocast_dtype(input_ids.device.type) if ac else None),
               torch.is_inference_mode_enabled())
        ent = self.entries.get(key)
        if ent is None or ent.key_bytes != key_bytes or ent.hs is None:
            out = self.inner(*args, **kw)                                              # FULL: the inner forward; remember its hidden states
            hsf = getattr(out, "hidden_states", None)
            if not torch.is_tensor(hsf) or hsf.dim() != 4 or tuple(hsf.shape[1:3]) != (B, L):
                self._fb("form"); return out
            if ent is None:
                if len(self.entries) >= self.max_entries:
                    self.entries.pop(next(iter(self.entries)))
                ent = _Entry(B, L, b0, input_ids, sequence_id, hsf.shape[0], hsf.shape[-1]); self.entries[key] = ent
            if ent.hs is None or ent.hs.shape != hsf.shape or ent.hs.dtype != hsf.dtype:
                ent.hs = torch.empty_like(hsf, memory_format=torch.contiguous_format)
            ent.hs.copy_(hsf); ent.key_bytes = key_bytes
            self.stats["full"] += 1
            self.live = f"{L - b0}/{L}"
            return out
        # HOIST: the frozen rows are the remembered ones
        ent.ids_live.copy_(input_ids[:, b0:])
        try:
            if self.graphs and input_ids.is_cuda:
                if ent.graph is None:
                    self._capture(ent)
                ent.graph.replay(); self.stats["replays"] += 1
            else:
                self._reduced(ent)
        except Exception as e:                                                       # noqa: BLE001 — a capture / replay that raises: the inner forward from now on, by name
            self._switch_off(f"capture_failed:{type(e).__name__}")
            self._fb(self.off)
            return self.inner(*args, **kw)
        out = self._output(ent)
        if not ent.anchored and self.anchor > 0:                                     # the run-time anchor: the inner forward on the same input, all rows tensor-equal
            ref = self.inner(*args, **kw)
            hr = getattr(ref, "hidden_states", None)
            ok = torch.is_tensor(hr) and hr.shape == ent.hs.shape and hr.dtype == ent.hs.dtype and bool(torch.equal(hr, ent.hs))
            if not ok:
                self._switch_off("anchor"); self._fb("anchor")
                return ref
            ent.n_anchor += 1
            if ent.n_anchor >= self.anchor:
                ent.anchored = True
            self.stats["anchored"] += 1
        self.stats["served"] += 1
        return out


# --------------------------------------------------------------------------------------------------------------------------- module API
def _esmc_of(model):
    for name in ("_esmc", "esmc"):
        sub = getattr(model, name, None)
        if isinstance(sub, torch.nn.Module):
            return sub
    if isinstance(model, torch.nn.Module):
        return model
    raise TypeError("enable(model): expected an ESMFold2 model (._esmc), an ESMCForMaskedLM (.esmc) or an ESMC module")


def enable(model, anchor: int = 1, graphs: bool = True, n_warmup: int = 2, max_entries: int = 2, cc=None) -> _Hoist:
    """Install the hoist as the trunk's forward over whatever forward is installed now (idempotent per trunk; returns the handle). On a compute
    capability not in PROVEN_CC the handle is registered but NOT installed: `handle.stepped_aside` names why, the trunk's forward is untouched
    (`cc=(major, minor)` overrides the probed capability — tests)."""
    esmc = _esmc_of(model)
    h = getattr(esmc, _ATTR, None)
    if h is not None:
        return h
    h = _Hoist(esmc, esmc.forward, anchor=anchor, graphs=graphs and torch.cuda.is_available(), n_warmup=n_warmup, max_entries=max_entries, cc=cc)
    setattr(esmc, _ATTR, h)
    if h.stepped_aside is None:
        esmc.forward = h
        h.installed = True
    return h


def disable(model) -> None:
    esmc = _esmc_of(model)
    h = getattr(esmc, _ATTR, None)
    if h is None:
        return
    if h.installed and vars(esmc).get("forward") is h:
        inner = h.inner
        if getattr(inner, "__self__", None) is esmc and getattr(inner, "__func__", None) is type(esmc).forward:
            del esmc.forward                                  # the class's own forward was under this wrapper: leave no instance attribute behind
        else:
            esmc.forward = inner
    h.entries.clear()
    delattr(esmc, _ATTR)


def handle(model):
    return getattr(_esmc_of(model), _ATTR, None)


def stats(model) -> dict | None:
    h = handle(model)
    if h is None:
        return None
    return dict(h.stats, fallback=dict(h.stats["fallback"]), entries=len(h.entries), off=h.off, live=h.live, cc=h.cc, stepped_aside=h.stepped_aside, installed=h.installed)


def release(model) -> None:
    """Drop the remembered hidden states and the reduced graphs (the wrapper stays installed; the next call of a key is FULL)."""
    h = handle(model)
    if h is not None:
        h.entries.clear()
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize(); torch.cuda.empty_cache()
