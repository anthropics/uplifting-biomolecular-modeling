"""ef2_loop_pppl — the cookbook's ESMC pseudo-perplexity loss (binder_design.py `compute_esmc_pseudoperplexity_nll`,
l.853-947, Algorithm 14) issued WITHOUT host synchronisations (loop level, exact).

Stock blocks the host three times per call (design batch 1): `score_mask[b].nonzero()` to count the scorable positions,
`pass_masks.nonzero()` to place the mask token, and the boolean gather `nlls[pass_masks]` before the mean. Each block
parks the host until the current stream drains — behind the fold's whole backward when the loss runs in `run_step`'s
order, or in the middle of the fold's launch sequence when the loss is launched early on a side stream (ef2_esmc_overlap) —
and every block is followed by a launch gap on the GPU. It also constructs an `ESMCTokenizer()` per call.

This lever's function computes the same tensors with the same kernels in the same order, minus the three blocking reads:
  * the scorable positions are a constant of the design (`score_mask = gradient_mask.sum(-1) > 0`, run_step l.1085, with
    `gradient_mask` built once by `build_gradient_mask`, l.1040): this lever wraps `BD.build_gradient_mask` to take the
    positions to the host ONCE per design (the loop has not started; nothing is in flight), and keeps them as host ints +
    a device index tensor. A call whose mask was not seen through `build_gradient_mask` (a transcribed single step) reads
    it once, blocking, and memoises it (`first_read` in the census). Every call still CHECKS the caller's mask against
    the memo — on the device, in stream order, the flag copied to pinned memory without blocking and read at the next call
    (and by `stats()` / `disable()`): a changed mask fails loudly one call later instead of blocking every call now;
  * `torch.rand((n_passes, num_positions))` exactly as stock (same shape → the same Philox draws and generator advance),
    `topk` as stock, the pass masks by the same `index_put`; the mask token placed with `torch.where` on the padded pass
    mask (a select, no arithmetic; its backward zeroes the same gradient entries the in-place index_put's backward zeroes);
  * the NLL gather by integer indices — rows × the masked columns SORTED ascending, i.e. exactly the row-major element
    order the boolean gather returns — so `.mean()` reduces the same values in the same order (bitwise), and its backward
    is the same `index_put(accumulate)` onto zeros at unique positions;
  * the tokenizer's two ids read once.
The ESMC transformer / lm_head calls are the cookbook's own lines, so the design kit's pPPL fwd+bwd graph
(`esmc.transformer.forward`) and any lever wrapping this function from outside compose unchanged.

Exact: proof = k/test_ef2_loop.py::test_pppl_bitwise (A/B/A in one process under the det recipe: pLM loss, both
gradients, updated logits, the masked LM inputs of every pass, and the CUDA RNG state at the call's entry AND exit
tensor-equal over 12 steps incl. confidence steps, two sizes). Census: stats() → served, first_read, checked, designs.

    import ef2_loop_pppl as lp
    lp.enable(BD)        # replaces BD.compute_esmc_pseudoperplexity_nll and wraps BD.build_gradient_mask
    lp.disable(BD)
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

_ATTR = "_ef2_loop_pppl"


class _PPPL:
    def __init__(self, BD):
        self.BD = BD
        self.orig = BD.compute_esmc_pseudoperplexity_nll
        self.orig_bgm = BD.build_gradient_mask
        tok = BD.ESMCTokenizer()
        self.cls_id, self.eos_id = int(tok.cls_token_id), int(tok.eos_token_id)
        self.memo = {}                    # (B, L, device) -> dict(mask=bool [B, L] device, pos=[LongTensor per row], n=[int per row])
        self.check = None                 # (pinned int32 flag, event) of the previous call's mask comparison
        self.flag_host = None
        self.stats = dict(served=0, first_read=0, checked=0, designs=0)

    # ---- the design's constant: taken to the host once, when the loop builds it (nothing in flight then) ----
    def build_gradient_mask(self, *args, **kw):
        gm = self.orig_bgm(*args, **kw)
        if torch.is_tensor(gm) and gm.dim() >= 2:
            self._memoise(gm.sum(dim=-1) > 0)                    # run_step l.1085's expression, read once here
            self.stats["designs"] += 1
        return gm

    def _memoise(self, score_mask):
        sm = score_mask.to(dtype=torch.bool)
        if sm.ndim == 1:
            sm = sm.unsqueeze(0)
        host = sm.cpu()                                          # the one blocking read
        pos = [row.nonzero(as_tuple=False).flatten() for row in host]
        ent = dict(mask=sm.clone(), pos=[p.to(sm.device) for p in pos], n=[int(p.numel()) for p in pos])
        self.memo[(sm.shape[0], sm.shape[1], sm.device)] = ent
        return ent

    def _read_check(self):
        if self.check is None:
            return
        flag, ev = self.check; self.check = None
        ev.synchronize()
        self.stats["checked"] += 1
        if int(flag.item()) != 0:
            raise RuntimeError("ef2_loop_pppl: the score mask passed by the loop differs from the design's memoised mask "
                               "(gradient_mask.sum(-1) > 0 read at build_gradient_mask) — the previous call used the memo. Disable this lever.")

    # ---- the loss: binder_design.py l.863-947, same tensors, no blocking reads ----
    def __call__(self, esmc_model, binder_design, score_mask, batch_size: int = 4, n_passes: int = 4):
        BD = self.BD
        self._read_check()
        device = binder_design.device
        lm_vocab_size = esmc_model.config.vocab_size
        model_dtype = esmc_model.esmc.embed.weight.dtype

        target_esm = binder_design @ BD._folding_trunk_to_lm_aa_vocab_matrix(device)
        input_esm = BD._straight_through(BD._one_hot_from_probs(target_esm), target_esm)
        input_ids = torch.zeros((binder_design.size(0), binder_design.size(1) + 2, lm_vocab_size), dtype=model_dtype, device=device)
        input_ids[:, 0, self.cls_id] = 1
        input_ids[:, -1, self.eos_id] = 1
        input_ids[:, 1:-1, 4:24] = input_esm.to(model_dtype)

        if score_mask.ndim == 1:
            score_mask = score_mask.unsqueeze(0).expand(binder_design.size(0), -1)
        elif score_mask.shape != binder_design.shape[:2]:
            raise ValueError(f"Expected score_mask with shape {(binder_design.size(0), binder_design.size(1))}, got {tuple(score_mask.shape)}")
        score_mask = score_mask.to(device=device, dtype=torch.bool)
        ent = self.memo.get((score_mask.shape[0], score_mask.shape[1], score_mask.device))
        if ent is None:
            self.stats["first_read"] += 1
            ent = self._memoise(score_mask)

        mask_token = torch.zeros(lm_vocab_size, dtype=model_dtype, device=device)
        mask_token[esmc_model.config.mask_token_id] = 1
        esmc = esmc_model.esmc

        all_masked_sequences, all_gathers = [], []
        rows = torch.arange(n_passes, device=device)[:, None]
        for batch_idx in range(binder_design.size(0)):
            position_indices, num_positions = ent["pos"][batch_idx], ent["n"][batch_idx]
            if num_positions == 0:
                raise ValueError("ESMC pseudoperplexity score mask selected zero positions.")
            num_masked = max(1, math.ceil(BD.ESMC_MASK_FRACTION * num_positions))
            random_scores = torch.rand((n_passes, num_positions), device=device)             # the stock draw: same shape, same generator
            masked_offsets = random_scores.topk(num_masked, dim=-1, largest=False).indices
            cols = position_indices[masked_offsets]                                          # [n_passes, num_masked]
            pass_masks = torch.zeros((n_passes, binder_design.size(1)), dtype=torch.bool, device=device)
            pass_masks[rows, cols] = True
            masked_sequences = input_ids[batch_idx: batch_idx + 1].repeat(n_passes, 1, 1)
            masked_sequences = torch.where(F.pad(pass_masks, (1, 1))[:, :, None], mask_token, masked_sequences)   # stock: [mask rows, cols + 1] = mask_token
            all_masked_sequences.append(masked_sequences)
            all_gathers.append((rows, cols.sort(dim=-1).values))                            # row-major order of the True entries = the boolean gather's order

        logit_chunks = []
        masked_sequence_rows = torch.cat(all_masked_sequences, dim=0)
        for start in range(0, masked_sequence_rows.size(0), batch_size):
            chunk = masked_sequence_rows[start: start + batch_size]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                hidden, *_ = esmc.transformer(chunk @ esmc.embed.weight.to(chunk.dtype), sequence_id=None, layers_to_collect=[], output_attentions=False)
                logit_chunks.append(esmc_model.lm_head(hidden))
        logits = torch.cat(logit_chunks, dim=0)

        losses = []
        for batch_idx, (r, c) in enumerate(all_gathers):
            start = batch_idx * n_passes
            stop = start + n_passes
            target_weights = target_esm[batch_idx]
            log_probs = logits[start:stop].log_softmax(dim=-1)[:, 1:-1, 4:24]
            nlls = -(log_probs * target_weights.to(log_probs.dtype).unsqueeze(0)).sum(dim=-1)
            losses.append(nlls[r, c].reshape(-1).mean())                                   # stock: nlls[pass_masks].mean()
        out = torch.stack(losses, dim=0)

        # compare the caller's mask with the memo, in stream order, without blocking
        flag = (score_mask != ent["mask"]).any().to(torch.int32)
        if self.flag_host is None:
            self.flag_host = torch.empty((), dtype=torch.int32, pin_memory=device.type == "cuda")
        self.flag_host.copy_(flag, non_blocking=True)
        ev = torch.cuda.Event() if device.type == "cuda" else None
        if ev is not None:
            ev.record()
            self.check = (self.flag_host, ev)
        elif int(self.flag_host.item()) != 0:
            raise RuntimeError("ef2_loop_pppl: score mask differs from the memo")
        self.stats["served"] += 1
        return out


def enable(BD) -> _PPPL:
    cur = getattr(BD, _ATTR, None)
    if cur is not None:
        return cur
    lever = _PPPL(BD)
    setattr(BD, _ATTR, lever)
    BD.compute_esmc_pseudoperplexity_nll = lever
    BD.build_gradient_mask = lever.build_gradient_mask
    return lever


def disable(BD) -> None:
    lever = getattr(BD, _ATTR, None)
    if lever is None:
        return
    if BD.compute_esmc_pseudoperplexity_nll is lever:
        BD.compute_esmc_pseudoperplexity_nll = lever.orig
    if getattr(BD.build_gradient_mask, "__self__", None) is lever:
        BD.build_gradient_mask = lever.orig_bgm
    lever._read_check()
    delattr(BD, _ATTR)


def stats(BD) -> dict:
    lever = getattr(BD, _ATTR, None)
    if lever is None:
        return dict(served=0, first_read=0, checked=0, designs=0)
    lever._read_check()
    return dict(lever.stats)
