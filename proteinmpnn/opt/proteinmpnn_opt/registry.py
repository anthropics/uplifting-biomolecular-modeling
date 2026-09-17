"""The lever registry: every runtime optimization of the kit executables, keyed by its switch (the executable's flag), with the class it
belongs to, its numerics tier and the file that implements it. Modes are lists of levers (modes.py resolves a mode to the kit's
README row, whose flags are keys of this registry); this file names what a flag is, never which flags a mode uses.

Classes: forward = one kit process per pass that exits (the worker); datapath = the read-once PDB parser.
Tiers: 1 = byte-identical to stock under the deterministic recipe; 2 = not bitwise (the
tolerance tier); None = the kit states a condition, not a tier. A tier can depend on the set (``tier_of``): ``--bb_batch`` above 1 is tier 1
with ``--chunk_gemm`` (every decode-step GEMM at the stock shape) and tier 2 without it (batched-GEMM accumulation differs from the
per-backbone call's).
File paths are relative to opt/forward/ (the directory names are modes.WORKER_DIR / modes.PARSER_DIR); the one package lever (``lowmem``, the
transform of the base variants' exact executable) points at ``opt/proteinmpnn_opt/lowmem.py``.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

from .modes import WORKER_DIR, PARSER_DIR

LEVERS: Dict[str, dict] = {
    # the worker's original levers (addon/mpnn_worker2.py) and the read-once parser (kit/fast_parse.py)
    "--mode": {"name": "stream replay", "cls": "forward", "tier": 1, "file": f"{WORKER_DIR}/addon/mpnn_worker2.py",
               "what": "stream: replays the one-process stock RNG stream over the whole jsonl (per-backbone Philox offsets)"},
    "--bb_batch": {"name": "cross-backbone batching", "cls": "forward", "tier": 1, "file": f"{WORKER_DIR}/addon/mpnn_worker2.py",
                   "what": "K backbones decoded per forward, each on its own generator; the value is the batch width. Tier 1 with --chunk_gemm "
                           "(the decode-step GEMMs at the stock shape); K > 1 without it is tier 2 (tier_of)"},
    "--sort_by_length": {"name": "length-sorted batches", "cls": "forward", "tier": 1, "file": f"{WORKER_DIR}/addon/mpnn_worker2.py",
                         "what": "groups backbones of similar length per batch; stream mode keeps each backbone's own offset"},
    "--chunk_gemm": {"name": "per-backbone GEMM chunking", "cls": "forward", "tier": 1, "file": f"{WORKER_DIR}/addon/mpnn_worker2.py",
                     "what": "every decode-step GEMM at the stock per-backbone shape (the stock kernels)"},
    "fast_parse": {"name": "read-once PDB parser", "cls": "datapath", "tier": 1, "file": f"{PARSER_DIR}/kit/fast_parse.py",
                   "what": "the stock parse_multiple_chains.py reading each PDB once (byte-identical parsed.jsonl)"},
    # the worker's exact set (addon/mpnn_worker2.py --x_all) and its probe-gated lever
    "--graph_rng": {"name": "whole-step CUDA graph", "cls": "forward", "tier": 1, "file": f"{WORKER_DIR}/addon/mpnn_worker2.py",
                    "what": "the whole autoregressive decode step incl. the per-backbone multinomial on registered generators, in one graph per batch"},
    "--single_graph": {"name": "one graph per decode shape", "cls": "forward", "tier": 1, "file": f"{WORKER_DIR}/addon/mpnn_worker2.py",
                       "what": "one all-draw CUDA graph per decode shape, replayed L_max times per batch with a device-side step counter and for every later batch of that shape; upstream's two scoring forwards per backbone captured per shape likewise and run on their own streams underneath the decode steps"},
    "--fused_draw": {"name": "fused draw kernel", "cls": "forward", "tier": 1, "file": f"{WORKER_DIR}/addon/mpnn_worker2.py",
                     "what": "the decode step's per-backbone torch.multinomial draws in ONE kernel that computes torch's own sampler bit for bit (Philox4x32-10 exponential draws + argmax, in the probabilities' dtype); probed against torch.multinomial on the device at start-up — the job is refused by name if it does not reproduce it; CUDA + Triton"},
    "--cache_enc_ctx": {"name": "encoder context cache", "cls": "forward", "tier": 1, "file": f"{WORKER_DIR}/addon/mpnn_worker2.py",
                        "what": "the decoder's encoder context assembled per decoded position from h_V / h_E with stock's gathers and product (never the all-positions [N, L, K, 3H] tensors), its masks indexed directly (data movement)"},
    "--analytic_offsets": {"name": "analytic RNG offsets", "cls": "forward", "tier": 1, "file": f"{WORKER_DIR}/addon/mpnn_worker2.py",
                           "what": "each backbone's stream offset computed from per-call Philox increments instead of replayed draws"},
    "--stock_shape_enc": {"name": "stock-shape encoder", "cls": "forward", "tier": 1, "file": f"{WORKER_DIR}/addon/mpnn_worker2.py",
                          "what": "encoder and scoring forwards per backbone on the stock tensors, cached within a batch"},
    "--hybrid_gemm": {"name": "batched message GEMMs (probe-gated)", "cls": "forward", "tier": 1, "file": f"{WORKER_DIR}/addon/mpnn_worker2.py",
                      "what": "the decoder message GEMMs (W1/W2/W3, M = 8*48 rows per backbone) of all backbones of a batch in one call; the worker probes every decode-step cell of the job on the device before any output and applies it only when bit-identical to the per-backbone call (else the job is refused by name before any output: the line is all of its levers; --hybrid_gemm 0 runs it without the lever)"},
    "--x_all": {"name": "the exact set", "cls": "forward", "tier": 1, "file": f"{WORKER_DIR}/addon/mpnn_worker2.py",
                "what": "the worker's recommended exact set (its --x_all help text names the flags; modes.x_all_expansion reads it)"},
    # package (proteinmpnn_opt/lowmem.py) — the transform of the base variants' exact executable, keyed by the lever name the worker record carries (not a
    # flag).
    "lowmem": {"name": "low-memory featuriser + decoding-order mask", "cls": "forward", "tier": 1, "file": "../proteinmpnn_opt/lowmem.py",
               "what": "the O(L^2) sites of protein_mpnn_utils (masked top-k distances, RBF distances, residue offsets / same-chain flags) and the "
                       "dense decoding-order mask (one_hot + einsum + gather, in protein_mpnn_utils and the worker) recomputed on [B,L,K] tensors with the "
                       "same float32 arithmetic per element: working set linear in L, the same output bytes (base variants' exact line; B=8: 17.5k residues on one "
                       "80 GB H100 where the dense form stops at 12,411; B=1: 100,076 residues, no OOM reached at that size)"},
}


def lever(flag: str) -> dict:
    return LEVERS[flag]


def by_class(cls: str) -> Dict[str, dict]:
    return {k: v for k, v in LEVERS.items() if v["cls"] == cls}


def tier_of(tokens: Iterable[str]) -> Optional[int]:
    """The tier of a flag set given as the row's tokens (flags with their values, ``--x_all`` already expanded: modes.expanded_flags).
    2 as soon as one flag is tolerance-tier, or when ``--bb_batch`` above 1 runs without ``--chunk_gemm``;
    None when a flag's tier is a condition the kit states without a tier; else 1."""
    toks = list(tokens)
    flags = [t for t in toks if t.startswith("--")]
    tiers = [LEVERS[f]["tier"] for f in flags if f in LEVERS]
    if any(t is None for t in tiers):
        return None
    bb = 1
    for i, t in enumerate(toks):
        if t == "--bb_batch" and i + 1 < len(toks) and not toks[i + 1].startswith("--"):
            bb = int(toks[i + 1])
    if bb > 1 and "--chunk_gemm" not in flags:
        return 2
    return max(tiers, default=1)
