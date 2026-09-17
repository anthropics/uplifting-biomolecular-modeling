"""The 40b's pipelined scoring surface: on a model split over two devices, ``score_sequences(seqs, batch_size=1)`` with >= 2 sequences runs
consecutive same-length windows two-in-flight across vortex's layer split (pipeline.py) — the stock's own per-sequence form (evo2/scoring.py:
prepare_batch([seq]) -> logits -> logits_to_logprobs -> reduce), scheduling only: every window is the stock's one-row forward, and a
group of consecutive same-length windows is bounded by the bytes of the (1, L, V) logits it holds on device 0 (HELD_BYTES_MAX). Batched calls
(``score_sequences(batch_size >= 2)``, ``evo2_model(ids)`` with B >= 2) run the stock's batched forward: its B == 1 and B >= 2 GEMM paths are
not the same kernels, so rows are never split. OFF, named on the APPLIED line, on one device or when the installed evo2/scoring.py is not the
one whose per-sequence tail this file follows."""
from __future__ import annotations

import hashlib

from evo2_opt._oom import is_oom

HELD_BYTES_MAX = 2 << 30                                     # a pipelined group grows while the per-window logits it holds on device 0 stay within this many bytes:
                                                             # (1, L, V) per window counted at 4 L V bytes (16 MiB at L = 8,192, V = 512 -> up to 128 windows), so
                                                             # consecutive same-length windows form ONE group (one fill and one drain) at genome-window lengths
SCORING_MODULE_SHA256 = "17e6985cef3a62c2a488133ab5a65050dcf0abeec2f55d1c0b068dfe37c44de6"   # evo2 0.6.0's evo2/scoring.py: the per-sequence tail followed below
_P = {"ok": False, "why": "not set up", "fwd": None, "windows": 0, "groups": {}, "calls": 0}


def setup(evo2_model) -> dict:
    """Build the pipelined forward once (the two stage streams); {'ok': False, 'why': …} names why it is off. Never a refusal of the model."""
    import torch
    global _P
    try:
        import evo2.scoring as _sc
        sha = hashlib.sha256(open(_sc.__file__, "rb").read()).hexdigest()
    except Exception as e:                                   # noqa: BLE001 — the package's scoring module unreadable: the pipeline is off with the reason
        _P = dict(_P, ok=False, why=f"evo2.scoring unreadable: {type(e).__name__}: {str(e)[:100]}"); return _P
    if sha != SCORING_MODULE_SHA256:
        _P = dict(_P, ok=False, why=f"evo2/scoring.py sha256 {sha[:12]} is not the pinned {SCORING_MODULE_SHA256[:12]}"); return _P
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        _P = dict(_P, ok=False, why="one device: no layer split to pipeline"); return _P
    try:
        from evo2_opt.kit.pipeline import make_pipelined_forward_windows
        fwd, topology = make_pipelined_forward_windows(evo2_model.model)
    except Exception as e:                                   # noqa: BLE001 — a placement that is not the two-device split: off with the reason
        if is_oom(e):
            raise
        _P = dict(_P, ok=False, why=f"placement: {type(e).__name__}: {str(e)[:160]}"); return _P
    _P = {"ok": True, "why": None, "fwd": fwd, "topology": topology, "windows": 0, "groups": {}, "calls": 0}
    return _P


def words() -> str:
    if not _P.get("ok"):
        return f"off({_P.get('why')})"
    st = _P["topology"]["stages"]                                            # device -> [first block, last block, count]
    return "on(score_sequences batch_size=1: %d stages, %s, one cuBLAS workspace per device)" % (
        len(st), " | ".join(f"{d} blocks {a}-{b}" for d, (a, b, n) in st.items()))


def tally() -> dict:
    return {"pipelined_windows": _P.get("windows", 0), "pipelined_calls": _P.get("calls", 0), "groups": dict(_P.get("groups") or {})}


def _score_pipelined(evo2_model, seqs: list, ensure_shape, *, prepend_bos: bool = False, reduce_method: str = "mean") -> list:
    import numpy as np
    import torch
    from evo2.scoring import logits_to_logprobs, prepare_batch
    if reduce_method == "sum":
        reduce_func = np.sum
    elif reduce_method == "mean":
        reduce_func = np.mean
    else:
        raise ValueError(f"Invalid reduce_method {reduce_method}")
    tok = evo2_model.tokenizer; model = evo2_model.model; vocab = int(model.config.vocab_size)
    scores = [None] * len(seqs); group: list = []

    def tail(idx, logits, ids, seq_len):
        logprobs = logits_to_logprobs(logits, ids).float().cpu().numpy()
        scores[idx] = reduce_func(logprobs[0][:seq_len])

    def flush():
        if not group:
            return
        n = len(group); L = int(group[0][1].shape[1])
        if n == 1:                                           # one window: the gated model call, no pipeline
            idx, ids, sl = group[0]
            with torch.inference_mode():
                logits, _ = model(ids)
            tail(idx, logits, ids, sl); group.clear(); return
        ensure_shape(1, L)                                   # the pipelined forward calls the blocks directly: the shape manager is told here
        outs = _P["fwd"](torch.cat([g[1] for g in group], 0), n)
        for (idx, ids, sl), out in zip(group, outs):
            tail(idx, out, ids, sl)
        _P["groups"][n] = _P["groups"].get(n, 0) + 1; _P["windows"] += n
        group.clear()

    with torch.no_grad():                                    # the wrapper's own no_grad (evo2/models.py:138)
        for idx, seq in enumerate(seqs):
            ids, seq_lengths = prepare_batch([seq], tok, device="cuda:0", prepend_bos=prepend_bos)
            if group and (int(group[0][1].shape[1]) != int(ids.shape[1]) or (len(group) + 1) * 4 * int(ids.shape[1]) * vocab > HELD_BYTES_MAX):
                flush()
            group.append((idx, ids, seq_lengths[0]))
        flush()
    _P["calls"] += 1
    return scores


def install(evo2_model, ensure_shape) -> dict:
    """Shadow ``score_sequences`` on the INSTANCE (the class untouched) when the pipeline is on; returns setup()'s record."""
    P = setup(evo2_model)
    if not P["ok"]:
        return P
    orig_score = evo2_model.score_sequences                  # the bound method (evo2/models.py:121)

    def score_sequences(seqs, batch_size: int = 1, prepend_bos: bool = False, reduce_method: str = "mean", average_reverse_complement: bool = False):
        if average_reverse_complement or batch_size != 1 or len(seqs) < 2:
            return orig_score(seqs, batch_size=batch_size, prepend_bos=prepend_bos, reduce_method=reduce_method, average_reverse_complement=average_reverse_complement)
        try:
            return _score_pipelined(evo2_model, seqs, ensure_shape, prepend_bos=prepend_bos, reduce_method=reduce_method)
        except Exception as e:                               # the package's own wrapping (evo2/models.py:139-142)
            raise RuntimeError(f"Error during sequence scoring: {str(e)}") from e

    evo2_model.score_sequences = score_sequences
    return P
