"""ef2_loop_prep — the design fold's entry re-plumbed: one device→host transfer for the designed sequence, the language-model
input built on the host and the ESMC-6B forward launched BEFORE the CPU featurisation (the two overlap), the hidden states
handed to the model through its own `lm_hidden_states=` argument, and a small memo of those hidden states for repeated
sequences (loop level, exact).

Stock `fold_and_get_distogram` (binder_design.py l.665-748) per call: (a) `int(tkn.item())` per binder position to spell the
designed sequence — one device sync each; (b) `prepare_esmfold2_tensors` per sequence — pure-CPU featurisation whose cost grows
with length, a miss of the design kit's featurisation cache on every step whose argmax sequence
is new (all early steps); (c) 25 `.cuda()` copies; (d) `model(**inputs, ...)`, inside which `_compute_lm_hidden_states`
(modeling_esmfold2_common.py `compute_lm_hidden_states`) rebuilds the LM input with ~10 syncing ops (boolean indexing,
`torch.unique`, per-chain `.nonzero()`, a Python loop over a device tensor) before the ESMC-6B forward and gathers the 81
hidden layers back to token order after it. During (a)-(c) and the prep of (d) the GPU idles (the step boundary synchronises
on the loss bookkeeping), so every host millisecond there is a step millisecond.

This lever's `fold_and_get_distogram` does the same work in a different order:
  1. designed sequence from ONE `.tolist()` of the argmax (same strings);
  2. LM input ids / sequence_id / token→LM-position map built ON THE HOST from the sequence strings through a letter→input-id
     table LEARNED FROM THE FEATURISER'S OWN OUTPUT (no featuriser logic restated), uploaded, and the ESMC-6B forward launched
     at once (`torch.inference_mode()` under the same bf16 autocast the model's forward has active at that point, so the design
     kit's ESMC graph sees the same key) — the GPU runs it while the host featurises;
  3. featurisation exactly as installed (`BD.prepare_esmfold2_tensors`, i.e. the design kit's cache or stock), upload as stock;
  4. the LM input the model WOULD have built, recomputed on the host from the uploaded features' CPU originals by an op-for-op
     replica of `compute_lm_hidden_states`' prep (written for what the cookbook's fold builds: protein chains — the target's
     chain(s) plus the binder —, no MSA, `lm_mask_pct` 0; the replica follows the model's token collapse per (asym_id,
     residue_index) and its per-chain BOS/EOS layout) and COMPARED with step 2's early guess — equal: the early hidden states
     are used; not equal or no guess yet (letter table incomplete): the forward is launched now with the replica's input
     (`late` in the census). ANCHOR: on the first call(s) the model's own `_compute_lm_hidden_states` also runs on the uploaded
     features (its syncing prep, one more ESMC pass) and its result must be tensor-equal to this lever's gathered hidden
     states; if it is not, that call uses the model's tensor and every later call lets the model compute its own
     (`fallback_anchor`, counted per call) — the lever then only keeps steps 1 and 3;
  5. the gather to token order with the same two indexing ops as stock (index tensors from the host instead of `.nonzero()`);
  6. `model(**inputs, lm_hidden_states=<5>, ...)` inside `seed_context(seed)` as stock — the model skips its own LM step when
     the hidden states are given (forward step 4) and continues identically.
A memo (default 1 entry, keyed by the ESMC module and the sequence strings; one set of LM hidden states per entry) returns the
gathered hidden states of a repeated sequence without running ESMC — late design steps repeat the sequence once T is small.
The memo holds its own tensor (the gather writes a fresh one), never the graph's static output buffer.

Exact: same integers into the same ESMC call under the same autocast / inference state, same gather ops, same model call;
`lm_mask_pct` must be 0 (it is for the design models; else the model's own path runs: `fallback_lm_mask`), the model must hold
an ESMC (`_esmc`; else stock path: `fallback_no_esmc`); a caller with another parameter list gets the stock function
(`fallback_signature`). The feature uploads of (c) and the gather's index uploads go through ONE page-locked staging buffer with
non-blocking copies (`PinnedUploader`, `pinned=`): a pageable `.cuda()` synchronises the stream, so stock's host waited there for the
ESMC forward it had just launched and the GPU idled through the uploads and the fold's preamble launches; staged, that host work
overlaps the language model (census `pinned` / `pageable`; pinning refused → pageable by name, `fallback_pageable`). Proof: k/test_ef2_loop.py (A/B/A bitwise in one process under the det recipe, two sizes, 12 steps
incl. confidence steps; a CPU test of the host replica against `compute_lm_hidden_states` with three chains, batch 2,
duplicate residue keys, a non-protein token and padding). Census: stats() → served, early, late, memo_hits, mismatch,
anchored, fallback_*.

    import ef2_loop_prep as lp
    lp.enable(BD, memo=1, anchor=1)     # replaces BD.fold_and_get_distogram
    lp.disable(BD)
"""
from __future__ import annotations

import collections

import numpy as np
import torch
import torch.nn.functional as F

_ATTR = "_ef2_loop_prep"
BOS, PAD, EOS = 0, 1, 2          # compute_lm_hidden_states' literals (modeling_esmfold2_common.py)


def host_lm_inputs(input_ids, asym_id, residue_index, mol_type, token_mask):
    """compute_lm_hidden_states' LM-input construction, op for op, on host arrays ([B, L] int64 / bool numpy).
    Returns (lm_input_ids [B, max_len] int64, sequence_id [B, max_len] int64, [(prot_pos_b, em_b) per batch row])."""
    B, L = input_ids.shape
    protein_mask = (mol_type == 0) & token_mask.astype(bool)
    seqs, lengths, maps = [], [], []
    for b in range(B):
        mb = protein_mask[b]
        ids_b, asym_b, res_b = input_ids[b][mb], asym_id[b][mb], residue_index[b][mb]
        keys = np.stack((asym_b, res_b), axis=1)
        if keys.shape[0] == 0:
            unique_keys, inverse = keys, np.zeros(0, dtype=np.int64)
        else:
            unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
            inverse = inverse.reshape(-1)
        n_unique = unique_keys.shape[0]
        first_pos = np.full((n_unique,), keys.shape[0], dtype=np.int64)
        np.minimum.at(first_pos, inverse, np.arange(keys.shape[0], dtype=np.int64))
        ordered = np.argsort(first_pos, kind="stable")
        first_pos_ordered = first_pos[ordered]
        ids_collapsed = ids_b[first_pos_ordered]
        asym_collapsed = asym_b[first_pos_ordered]
        remap = np.empty_like(ordered); remap[ordered] = np.arange(n_unique, dtype=np.int64)
        inverse_ordered = remap[inverse]
        chain_ids = np.unique(asym_collapsed)
        idt = input_ids.dtype                    # stock: lm_input_ids carries input_ids' dtype
        parts = [np.array([BOS], dtype=idt)]
        per_token_lm_pos = np.empty(n_unique, dtype=np.int64)
        cursor = 1
        for i, cid in enumerate(chain_ids):
            in_chain = np.nonzero(asym_collapsed == cid)[0]
            parts.append(ids_collapsed[in_chain].astype(idt))
            per_token_lm_pos[in_chain] = np.arange(cursor, cursor + in_chain.shape[0], dtype=np.int64)
            cursor += in_chain.shape[0]
            if i < len(chain_ids) - 1:
                parts.append(np.array([EOS, BOS], dtype=idt)); cursor += 2
        parts.append(np.array([EOS], dtype=idt))
        lm_seq = np.concatenate(parts)
        seqs.append(lm_seq); lengths.append(lm_seq.shape[0])
        prot_pos = np.nonzero(mb)[0].astype(np.int64)
        em = per_token_lm_pos[inverse_ordered]          # LM position of every protein token, in token order
        maps.append((prot_pos, em))
    max_len = max(lengths)
    lm_input_ids = np.full((B, max_len), PAD, dtype=input_ids.dtype)
    for b in range(B):
        lm_input_ids[b, : lengths[b]] = seqs[b]
    sequence_id = np.cumsum(lm_input_ids == BOS, axis=1).astype(np.int64) - 1
    sequence_id[lm_input_ids == PAD] = -1
    return lm_input_ids, sequence_id, maps


class SpliceFeaturiser:
    """The design fold's featurisation with the target's part computed once: `prepare_esmfold2_tensors` (as installed) runs on
    the BINDER ALONE per call and its rows are spliced after the target's rows kept from the first full featurisation of the
    design; index-valued fields get the constant offset the full featurisation gives them (learned from it, per field), pair
    fields are block-diagonal, atom rows are re-padded to the featuriser's multiple of 32. Nothing of the featuriser is
    restated: the layout is LEARNED from its own two outputs (complex, binder alone) for the design's first sequence and the
    splice of those two must reproduce the full output field for field (else the design keeps the full featuriser:
    `splice_off`), and the next `checks` spliced results are compared with the full featuriser too (`splice_checked`). A field
    name this class does not know, an unexpected shape, batch featurisation (`max_atoms`), or a binder whose sequence equals a
    target chain's (entity sharing) → the full featuriser (`splice_unknown` / `splice_batch` / `splice_dup`)."""

    TOKEN_PLAIN = ("residue_index", "mol_type", "res_type", "input_ids", "token_attention_mask", "pocket_feature", "deletion_mean")
    TOKEN_OFFSET = ("token_index", "asym_id", "entity_id", "sym_id", "distogram_atom_idx", "frames_idx")
    PAIR = ("token_bonds", "disto_cond", "disto_cond_mask")
    ATOM_PLAIN = ("ref_pos", "ref_element", "ref_charge", "ref_atom_name_chars", "atom_attention_mask", "is_resolved")
    ATOM_OFFSET = ("ref_space_uid", "atom_to_token")
    ATOM_DIM1 = ("gt_coords",)
    MSA = ("msa", "deletion_value", "has_deletion", "msa_attention_mask")
    KNOWN = frozenset(TOKEN_PLAIN + TOKEN_OFFSET + PAIR + ATOM_PLAIN + ATOM_OFFSET + ATOM_DIM1 + MSA)

    def __init__(self, BD, checks: int = 2):
        self.BD = BD
        self.checks = int(checks)
        self.designs = {}            # (target chains, binder length) -> learned layout dict, or None = full featuriser for this design
        self.stats = dict(spliced=0, splice_full=0, splice_learned=0, splice_checked=0, splice_off=0, splice_unknown=0, splice_batch=0, splice_dup=0)

    # -- the featuriser as installed, on a chain list --
    def _featurise(self, chains, first_id, max_atoms=None):
        BD = self.BD
        sequences = {sequence: [str(first_id + idx)] for idx, sequence in enumerate(chains)}
        raw = BD.StructurePredictionInput(sequences=[BD.ProteinInput(id=cid, sequence=sq, msa=None) for sq, cid in sequences.items()])
        return BD.prepare_esmfold2_tensors(raw, max_atoms=max_atoms)

    def __call__(self, seq: str, max_atoms=None):
        """== BD.prepare_esmfold2_tensors on the cookbook's input for `seq` = 'target[|target…]|binder' (fold_and_get_distogram l.700-712)."""
        chains = seq.split("|")
        if max_atoms is not None or len(chains) < 2:
            self.stats["splice_batch" if max_atoms is not None else "splice_unknown"] += 1; self.stats["splice_full"] += 1
            return self._featurise(chains, 0, max_atoms)
        target, binder = tuple(chains[:-1]), chains[-1]
        if binder in target:
            self.stats["splice_dup"] += 1; self.stats["splice_full"] += 1
            return self._featurise(chains, 0)
        key = (target, len(binder))
        lay = self.designs.get(key, False)
        if lay is None:                                             # this design keeps the full featuriser
            self.stats["splice_full"] += 1
            return self._featurise(chains, 0)
        alone = self._featurise([binder], len(target))             # the binder alone, under the chain id it has in the complex
        if lay is False:                                            # first sequence of the design: learn the layout, return the full output
            full = self._featurise(chains, 0); self.stats["splice_full"] += 1
            lay = self._learn(full, alone, len(binder))
            self.designs[key] = lay
            self.stats["splice_learned" if lay is not None else "splice_off"] += 1
            return full
        out = self._splice(lay, alone)
        if lay["checks_left"] > 0:                                  # the first spliced results are compared with the full featuriser
            full = self._featurise(chains, 0); self.stats["splice_full"] += 1
            if not _same_features(out, full):
                self.designs[key] = None; self.stats["splice_off"] += 1
                return full
            lay["checks_left"] -= 1; self.stats["splice_checked"] += 1
        self.stats["spliced"] += 1
        return out

    # -- layout from the featuriser's own two outputs --
    def _learn(self, full, alone, Lb):
        try:
            if set(full) != set(alone) or not set(full) <= self.KNOWN:
                self.stats["splice_unknown"] += 1
                return None
            L = int(full["token_index"].shape[0]); Lt = L - Lb
            am_f, am_a = full["atom_attention_mask"], alone["atom_attention_mask"]
            A_real, Ab = int(am_f.sum()), int(am_a.sum()); At = A_real - Ab
            if Lt <= 0 or At <= 0 or int(alone["token_index"].shape[0]) != Lb:
                return None
            if not (bool(am_f[:A_real].all()) and not bool(am_f[A_real:].any()) and bool(am_a[:Ab].all()) and not bool(am_a[Ab:].any())):
                return None                                         # real atoms are not a prefix: not a layout this class splices
            if am_f.shape[0] != _padded_atoms(A_real) or am_a.shape[0] != _padded_atoms(Ab):
                return None
            lay = dict(Lt=Lt, At=At, checks_left=self.checks, fields={})
            for k, f in full.items():
                a = alone[k]
                if k in self.TOKEN_PLAIN or k in self.TOKEN_OFFSET:
                    if f.shape[0] != L or a.shape[0] != Lb or f.shape[1:] != a.shape[1:]:
                        return None
                    off = None
                    if k in self.TOKEN_OFFSET:
                        d = f[Lt:] - a
                        if not bool((d == d.reshape(-1)[0]).all()):
                            return None
                        off = d.reshape(-1)[0].clone()
                    elif not torch.equal(f[Lt:], a):
                        return None
                    lay["fields"][k] = ("token", f[:Lt].clone(), off)
                elif k in self.PAIR:
                    if f.shape[:2] != (L, L) or a.shape[:2] != (Lb, Lb) or f.shape[2:] != a.shape[2:]:
                        return None
                    if not torch.equal(f[Lt:, Lt:], a) or bool((f[:Lt, Lt:] != 0).any()) or bool((f[Lt:, :Lt] != 0).any()):
                        return None
                    lay["fields"][k] = ("pair", f[:Lt, :Lt].clone(), None)
                elif k in self.ATOM_PLAIN or k in self.ATOM_OFFSET or k in self.ATOM_DIM1:
                    fd, ad = (f[0], a[0]) if k in self.ATOM_DIM1 else (f, a)
                    if (k in self.ATOM_DIM1 and (f.shape[0] != 1 or a.shape[0] != 1)) or fd.shape[0] != am_f.shape[0] or ad.shape[0] != am_a.shape[0] or fd.shape[1:] != ad.shape[1:]:
                        return None
                    if bool((fd[A_real:] != 0).any()) or bool((ad[Ab:] != 0).any()):
                        return None                                 # padding rows are zeros in both
                    off = None
                    if k in self.ATOM_OFFSET:
                        d = fd[At:A_real] - ad[:Ab]
                        if not bool((d == d.reshape(-1)[0]).all()):
                            return None
                        off = d.reshape(-1)[0].clone()
                    elif not torch.equal(fd[At:A_real], ad[:Ab]):
                        return None
                    lay["fields"][k] = ("atom1" if k in self.ATOM_DIM1 else "atom", fd[:At].clone(), off)
                else:                                               # MSA [M, L]
                    if f.dim() != 2 or f.shape[1] != L or a.shape != (f.shape[0], Lb) or not torch.equal(f[:, Lt:], a):
                        return None
                    lay["fields"][k] = ("msa", f[:, :Lt].clone(), None)
            if not _same_features(self._splice(lay, alone), full):  # the splice must reproduce this very output
                return None
            return lay
        except Exception:                                           # any surprise: the full featuriser
            return None

    @staticmethod
    def _splice(lay, alone):
        Lt, At = lay["Lt"], lay["At"]
        Ab = int(alone["atom_attention_mask"].sum())
        A_real = At + Ab; A = _padded_atoms(A_real)
        out = {}
        for k, (kind, tpart, off) in lay["fields"].items():
            a = alone[k]
            if kind == "token":
                out[k] = torch.cat((tpart, a if off is None else a + off), dim=0)
            elif kind == "pair":
                L = Lt + a.shape[0]
                o = torch.zeros((L, L) + tuple(tpart.shape[2:]), dtype=tpart.dtype)
                o[:Lt, :Lt] = tpart; o[Lt:, Lt:] = a
                out[k] = o
            elif kind in ("atom", "atom1"):
                o = torch.zeros((A,) + tuple(tpart.shape[1:]), dtype=tpart.dtype)
                o[:At] = tpart
                o[At:A_real] = a[0, :Ab] if kind == "atom1" else (a[:Ab] if off is None else a[:Ab] + off)
                out[k] = o.unsqueeze(0) if kind == "atom1" else o
            else:
                out[k] = torch.cat((tpart, a), dim=1)
        return out


def _padded_atoms(n_real: int) -> int:
    return -(-n_real // 32) * 32 if n_real > 0 else 32          # the featuriser pads atoms to a multiple of 32 (learned outputs are checked against it)


def _same_features(x, y) -> bool:
    return set(x) == set(y) and all(x[k].dtype == y[k].dtype and x[k].shape == y[k].shape and torch.equal(x[k], y[k]) for k in x)


def _fold_args(model, target_seq, target_one_hot, design, num_loops=0, num_sampling_steps=1, calculate_confidence=False, seed=None):
    """binder_design.fold_and_get_distogram's parameter list (l.665-674): binds a call's arguments by name whatever wraps the function."""
    return dict(model=model, target_seq=target_seq, target_one_hot=target_one_hot, design=design, num_loops=num_loops,
                num_sampling_steps=num_sampling_steps, calculate_confidence=calculate_confidence, seed=seed)


class PinnedUploader:
    """The fold's host->device feature uploads through ONE page-locked staging buffer with non-blocking copies (exact: the same
    bytes land in the same dtypes and shapes). Stock uploads each of the ~25 feature tensors with `.cuda()` from pageable memory:
    every such copy synchronises the stream — the host waits for the ESMC-6B forward launched just before — and the GPU then idles
    while the host uploads and launches the fold's preamble. Staged through pinned memory the copies are stream-ordered and return
    at once, so that host work runs while the language model computes. The staging buffer is rewritten only after the event
    recorded behind the previous call's copies has completed. No CUDA / pinning refused by the driver -> pageable `.to(device)` by
    name (`pageable`), counted."""

    ALIGN = 64

    def __init__(self, enabled: bool = True):
        self.enabled = bool(enabled)
        self.buf = None
        self.event = None
        self.stats = dict(pinned=0, pageable=0, grown=0)
        self.failed = None

    def __call__(self, tensors: dict, device) -> dict:
        dev = torch.device(device)
        if not self.enabled or dev.type != "cuda" or self.failed is not None or not tensors:
            self.stats["pageable"] += len(tensors)
            return {k: v.to(dev) for k, v in tensors.items()}
        items = [(k, (v if v.is_contiguous() else v.contiguous())) for k, v in tensors.items()]
        offs, total = [], 0
        for _, v in items:
            offs.append(total); total += (v.numel() * v.element_size() + self.ALIGN - 1) // self.ALIGN * self.ALIGN
        try:
            if self.event is not None:
                self.event.synchronize()                              # the previous call's copies out of the staging buffer are done
            if self.buf is None or self.buf.numel() < total:
                self.buf = torch.empty(max(total, 2 * (self.buf.numel() if self.buf is not None else 0)), dtype=torch.uint8, pin_memory=True)
                self.stats["grown"] += 1
        except Exception as e:                                        # noqa: BLE001 — pinning refused: pageable uploads from now on, by name
            self.failed = f"{type(e).__name__}"; self.buf = None
            self.stats["pageable"] += len(tensors)
            return {k: v.to(dev) for k, v in tensors.items()}
        out = {}
        for (k, v), off in zip(items, offs):
            n = v.numel() * v.element_size()
            stage = self.buf[off:off + n].view(v.dtype).view(v.shape) if n else v
            if n:
                stage.copy_(v)                                        # host memcpy into page-locked memory
            out[k] = stage.to(dev, non_blocking=True) if n else v.to(dev)
        self.event = torch.cuda.Event(); self.event.record()
        self.stats["pinned"] += len(items)
        return out


class _Prep:
    def __init__(self, BD, memo=1, anchor=1, splice=True, splice_checks=2, pinned=True):
        self.BD = BD
        self.uploader = PinnedUploader(enabled=pinned)   # step 3's feature uploads: one pinned staging buffer, non-blocking copies
        self.idx_uploader = PinnedUploader(enabled=pinned)   # the gather's index uploads: its own staging buffer (a second use of the first one within a call would wait for its copies, i.e. for the LM)
        self.orig = BD.fold_and_get_distogram
        self.splice = SpliceFeaturiser(BD, checks=splice_checks) if splice else None
        self.memo_size = int(memo)
        self.memo = collections.OrderedDict()
        self.letter_ids = {}                 # residue letter -> LM input id, learned from the featuriser's output
        self.ids_dtype = None                # the featuriser's input_ids dtype (the LM input carries it, as stock)
        self.stats = dict(served=0, early=0, late=0, memo_hits=0, mismatch=0, anchored=0, fallback_anchor=0, fallback_no_esmc=0, fallback_lm_mask=0, fallback_signature=0)
        self.anchor_left = int(anchor)       # first calls also run the model's own LM path and require tensor-equal hidden states
        self.anchor_failed = False

    # ---------------------------------------------------------------------------------------------------------------- LM side
    def _guess(self, seq_list):
        """The LM input from the sequence strings alone (one protein token per letter, chains '|'-separated in asym order,
        residue keys unique) — None until the letter table covers every letter present. Compared with the features' replica input later."""
        tbl = self.letter_ids
        if self.ids_dtype is None:
            return None
        rows = []
        for seq in seq_list:
            chains = seq.split("|")
            try:
                ids = [np.fromiter((tbl[c] for c in ch), dtype=self.ids_dtype, count=len(ch)) for ch in chains]
            except KeyError:
                return None
            L = sum(len(ch) for ch in chains)
            n = np.arange(L, dtype=np.int64)
            asym = np.concatenate([np.full(len(ch), i, dtype=np.int64) for i, ch in enumerate(chains)])
            res = np.concatenate([np.arange(len(ch), dtype=np.int64) for ch in chains])
            rows.append((np.concatenate(ids), asym, res, np.zeros(L, dtype=np.int64), np.ones(L, dtype=bool)))
        if len({r[0].shape[0] for r in rows}) != 1:
            return None
        cols = [np.stack([r[i] for r in rows]) for i in range(5)]
        return host_lm_inputs(*cols)

    def _learn_letters(self, seq_list, feats_cpu):
        ids = feats_cpu["input_ids"].numpy(); mol = feats_cpu["mol_type"].numpy(); mask = feats_cpu["token_attention_mask"].numpy().astype(bool)
        self.ids_dtype = ids.dtype
        for b, seq in enumerate(seq_list):
            letters = seq.replace("|", "")
            prot = (mol[b] == 0) & mask[b]
            if prot.sum() != len(letters) or prot.sum() != prot.shape[0]:
                continue                     # not one protein token per letter: nothing to learn from this row
            for c, i in zip(letters, ids[b][prot]):
                self.letter_ids.setdefault(c, int(i))

    @staticmethod
    def _launch(esmc, lm_input_ids, sequence_id, device):
        ids = torch.from_numpy(lm_input_ids).to(device)
        sid = torch.from_numpy(sequence_id).to(device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.bfloat16), torch.inference_mode():
            out = esmc(input_ids=ids, sequence_id=sid, output_hidden_states=True)
        return out.hidden_states                                  # [n_layers+1, B, max_len, D]

    @staticmethod
    def _gather(hs, maps, L, device, upload=None):
        n, B, _, D = hs.shape
        result = torch.zeros(B, L, n, D, device=device, dtype=hs.dtype)
        for b, (prot_pos, em) in enumerate(maps):
            if upload is not None:
                idx = upload({"em": torch.from_numpy(em), "pos": torch.from_numpy(prot_pos)}, device); em_t, pos_t = idx["em"], idx["pos"]
            else:
                em_t = torch.from_numpy(em).to(device); pos_t = torch.from_numpy(prot_pos).to(device)
            gathered = hs[:, b, em_t, :].permute(1, 0, 2)
            result[b, pos_t] = gathered
        return result.detach()

    # ---------------------------------------------------------------------------------------------------------------- the fold
    def __call__(self, model, *args, **kw):
        BD = self.BD
        try:
            A = _fold_args(model, *args, **kw)
        except TypeError:                                   # not the cookbook's parameter list: stock path
            self.stats["fallback_signature"] += 1
            return self.orig(model, *args, **kw)
        target_seq, target_one_hot, design = A["target_seq"], A["target_one_hot"], A["design"]
        num_loops, num_sampling_steps, calculate_confidence, seed = A["num_loops"], A["num_sampling_steps"], A["calculate_confidence"], A["seed"]
        esmc = getattr(model, "_esmc", None)
        if esmc is None:
            self.stats["fallback_no_esmc"] += 1
            return self.orig(model, *args, **kw)
        if float(getattr(model.config, "lm_mask_pct", 0.0) or 0.0) > 0.0:
            self.stats["fallback_lm_mask"] += 1
            return self.orig(model, *args, **kw)
        device = design.device

        # 1. the designed sequence: one transfer (stock: one .item() per position)
        padded_design = F.pad(design, (2, 11), mode="constant", value=0)
        token_lists = torch.argmax(padded_design, dim=-1)
        designed_seq = [[BD.PROTEIN_3TO1[BD.TOKENS[int(t)]] for t in row] for row in token_lists.tolist()]
        seq_list = [target_seq + "|" + "".join(seq) for seq in designed_seq]
        max_atoms = None if len(seq_list) == 1 else ((len(seq_list[0]) - 1) * 14) // 32 * 32

        # 2. memo / early LM launch
        key = (id(esmc), tuple(seq_list))
        lm_hs = None if self.anchor_failed else self.memo.get(key)
        hs = guess = None
        if lm_hs is not None:
            self.memo.move_to_end(key); self.stats["memo_hits"] += 1
        elif not self.anchor_failed:
            guess = self._guess(seq_list)
            if guess is not None:
                hs = self._launch(esmc, guess[0], guess[1], device)

        # 3. featurisation (as installed: the design kit's cache or stock) and upload, as stock
        inputs_list = []
        for seq in seq_list:
            if self.splice is not None:
                inputs_list.append(self.splice(seq, max_atoms=max_atoms))
                continue
            sequences = {sequence: [str(idx)] for idx, sequence in enumerate(seq.split("|"))}
            inputs_raw = BD.StructurePredictionInput(sequences=[BD.ProteinInput(id=chain_id, sequence=sequence, msa=None) for sequence, chain_id in sequences.items()])
            inputs_list.append(BD.prepare_esmfold2_tensors(inputs_raw, max_atoms=max_atoms))
        cpu = {k: torch.stack([inp[k] for inp in inputs_list], dim=0) for k in inputs_list[0]}
        inputs = self.uploader(cpu, device)                     # stock: {k: v.cuda()} — 25 pageable copies, each a stream synchronisation
        inputs["res_type_soft"] = torch.cat((target_one_hot.repeat(design.size(0), 1, 1), padded_design), dim=1)

        # 4./5. the replica LM input from the features; use the early result when it matches, else launch now; gather
        if self.anchor_failed:
            self.stats["fallback_anchor"] += 1                    # lm_hidden_states=None: the model computes its own
        elif lm_hs is None:
            ref = host_lm_inputs(cpu["input_ids"].numpy(), cpu["asym_id"].numpy(), cpu["residue_index"].numpy(), cpu["mol_type"].numpy(), cpu["token_attention_mask"].numpy())
            if guess is not None and np.array_equal(guess[0], ref[0]) and np.array_equal(guess[1], ref[1]) and all(
                    np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) for a, b in zip(guess[2], ref[2])):
                self.stats["early"] += 1
            else:
                if guess is not None:
                    self.stats["mismatch"] += 1
                self.stats["late"] += 1
                hs = self._launch(esmc, ref[0], ref[1], device)
            if len(self.letter_ids) < 32:
                self._learn_letters(seq_list, cpu)
            lm_hs = self._gather(hs, ref[2], cpu["input_ids"].shape[1], device, self.idx_uploader)
            if self.anchor_left > 0:                              # the model's own LM path on the same features must give the same tensor
                self.anchor_left -= 1
                with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.bfloat16):    # the model forward's context at its LM step
                    own = model._compute_lm_hidden_states(inputs["input_ids"], inputs["asym_id"], inputs["residue_index"], inputs["mol_type"],
                                                          inputs["token_attention_mask"], lm_mask_pct=0.0)
                if own.shape == lm_hs.shape and own.dtype == lm_hs.dtype and torch.equal(own, lm_hs):
                    self.stats["anchored"] += 1
                else:
                    self.anchor_failed = True; self.stats["fallback_anchor"] += 1
                    self.memo.clear()
                    lm_hs = own
            if self.memo_size > 0 and not self.anchor_failed:
                self.memo[key] = lm_hs
                while len(self.memo) > self.memo_size:
                    self.memo.popitem(last=False)

        # 6. the model call, as stock, with the hidden states given
        with BD.seed_context(seed):
            output = model(**inputs, lm_hidden_states=lm_hs, num_diffusion_samples=1, num_sampling_steps=num_sampling_steps,
                           num_loops=num_loops, calculate_confidence=calculate_confidence, seed=seed)
        self.stats["served"] += 1
        result = {"distogram_logits": output["distogram_logits"], "inputs": inputs, "inputs_list": inputs_list, "output": output, "seq_list": seq_list}
        if calculate_confidence:
            result.update({"ptm": output.get("ptm"), "iptm": output.get("iptm"), "plddt": output.get("plddt")})
        return result


def enable(BD, memo: int = 1, anchor: int = 1, splice: bool = True, splice_checks: int = 2, pinned: bool = True) -> _Prep:
    cur = getattr(BD, _ATTR, None)
    if cur is not None:
        return cur
    lever = _Prep(BD, memo=memo, anchor=anchor, splice=splice, splice_checks=splice_checks, pinned=pinned)
    setattr(BD, _ATTR, lever)
    BD.fold_and_get_distogram = lever
    return lever


def disable(BD) -> None:
    lever = getattr(BD, _ATTR, None)
    if lever is None:
        return
    if BD.fold_and_get_distogram is lever:
        BD.fold_and_get_distogram = lever.orig
    lever.memo.clear()
    delattr(BD, _ATTR)


def stats(BD) -> dict:
    lever = getattr(BD, _ATTR, None)
    if lever is None:
        return {}
    out = dict(lever.stats)
    if lever.splice is not None:
        out.update(lever.splice.stats)
    ups = (lever.uploader, lever.idx_uploader)
    out.update(pinned=sum(u.stats["pinned"] for u in ups), pageable=sum(u.stats["pageable"] for u in ups), fallback_pageable=sum(u.stats["pageable"] for u in ups if u.failed),
               pin_failed=next((u.failed for u in ups if u.failed), None))
    return out


def release(BD) -> None:
    """Drop the memoised hidden states (the lever stays installed)."""
    lever = getattr(BD, _ATTR, None)
    if lever is not None:
        lever.memo.clear()
