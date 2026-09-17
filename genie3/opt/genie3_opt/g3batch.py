"""g3batch — the batched capture line: upstream's own batch semantics at batch size B under the kit's levers.
``python <genie3_opt>/g3batch.py --config <request.yaml> --outdir <out> --batch-size B [--cuda-graphs] [--hoist] [--reuse-graphs R]
[--pt-chunk design] [--compile] [--tf32] [--trimul stock|fpf] [--timings <json>] [--limit N] [--seed S]``, cwd = the Genie 3 checkout.

What a pass computes is what ``genie3 generate`` computes with ``generation.dataset.batch_size: B`` (``generation.compile: false``): the
designs of the request in dataset order, B per denoiser call (data_module.py GenieTestSampler: consecutive index blocks; the dataset's
``__getitem__`` featurises and pads the block, feat_utils.batchify_np_features), the initial noise one ``torch.randn_like`` over the batch's
``gt_atom_positions`` and one ``torch.randn_like(xs)`` per DDIM step (sampler.py:98-110, ddim.py ``_step``), the timestep vector
``torch.Tensor([step] * B).int()`` per step, then — when the request sets ``predict_sequence`` — upstream's own sequence stage on the batch
(Sampler._sample_sequence, eagerly on the denoiser module: its one noise draw follows the loop's, as in Sampler.sample), the PDB files through
upstream's ``postprocess`` per batch (runner.py on_test_batch_end). ``--shard-id K --num-shards M`` slice the request as upstream's loader does
(the shard's share of n_sample, named from its sample_index_offset). A request that sets `inference.search` (beam search) or the sampler's
`predict_sidechain` stage is refused by name. At B = 1 this is upstream's per-design stream; at B > 1 it is what stock computes at batch size B.

How it computes it is the kit's: the resident driver ``opt/forward/fast_inference/driver/g3fast.py`` imported unchanged as a library (model /
sampler / data module set-up = the stock path's, build_everything; the sync-free step tables and DDIM arithmetic, StepTables / ddim_math; the
CUDA-graph wrapper of the denoiser core, GraphedDenoiser; the process-global numerics guard, global_numerics_state / assert_global_state), its
value-identical forward patches (g3fast_patches.apply), and the capture-cell module ``opt/forward/g3cap/g3cap.py`` imported unchanged for the
pair featuriser's loop-invariant hoist (install_hoist) and the graph cache keyed by feature-shape signature (feat_signature / GraphCache /
rebind_graph: one capture per distinct batch shape per process, later batches of that shape replay the kept graph). Levers on this line and
their switches: ``--cuda-graphs`` (L4), ``--hoist`` (L8), ``--reuse-graphs R`` (L9), ``--batch-size B`` (L11, the batch semantics above; batches
are featurised one at a time, so the resident feature set is one batch's whatever the request's size),
``--lean-pair`` (L17: the pair stack's reference lifetimes — LatentTransformer ownership passing and the stock TriMul forward with in-place gating,
g3lean.py; byte-identical, one pair tensor fewer at the core's activation peak, which is what sizes the graph's private pool), ``--wide-capture`` (L18: the
hoisted featuriser's per-step tail inside the captured region and the hoist's terms as graph-owned buffers — g3fast.GraphedDenoiser wide=True; needs
--cuda-graphs --hoist --lean-pair), ``--alloc expandable`` (L19: the caching allocator's expandable segments, set before CUDA initialises, and the capacity
probe once per feature-shape signature), ``--pt-chunk design`` (L12: upstream's PairTransition runs its two-Linear MLP through ``chunk_layer`` in 4-row chunks at inference, transition.py:102 —
B·N sequential chunk calls per layer per step; with the switch the chunk is one design's rows, chunk_size = N set per batch shape: B large chunks per call,
the whole-tensor GEMM path with one design's 4·c hidden intermediate live instead of the batch's — identical arithmetic per element, another cuBLAS kernel:
class 2), ``--compile`` (L16: the denoiser core compiled by inductor and the compiled callable recorded by the CUDA graph — g3fast.GraphedDenoiser
compile='inductor'; class 2-3, mode fast; one compile per captured batch shape per process), ``--tf32`` (L13: TF32 tensor-core
matmuls, class 3 — the tolerance tier's precision lever, never on an exact line; the switches move only through the shared core's
``opt_core.precision.policy``: without the flag the line PROVES upstream's fp32 policy is live and sets nothing (``expect``), with it the TF32
preset is applied (``apply``); the live readback — not the flag — is what ``timings.tf32`` / ``numerics_readback`` record, and a TF32 library
override present in the environment (``NVIDIA_TF32_OVERRIDE``, ``TORCH_ALLOW_TF32_CUBLAS_OVERRIDE``) is refused by name before torch
initialises), ``--trimul fpf`` (L7: the ten TriangleMultiplicativeUpdate modules served by the shared core's fused TriMul kernel through
``genie3_opt.trimul`` — opt_core.trimul's ladder, provider fpf_v4, strategy F2.fpf_trimul_fast; class 3, bf16 GEMM operands with fp32 accumulate,
mode fast's provider (modes.FAST_FLAGS pass it), never on an exact line; patched in before the capture so the graphs replay the kernel; unsupported
calls run the module's own forward counted by reason and an unexpected reason refuses the lever's gate; a request whose every call is under the
kernel's token floor declines the lever by name — ``timings.trimul`` and the KERNELS census line printed at exit carry the census). The per-batch record ``per_batch`` [{names, n_token_pad,
n_token_real, wall_s, probe_s, graph}] is the clock a throughput measurement reads (wall from the batch's features resident on the device to its last coordinate
materialised: the capacity probe — one eager forward at batch 1 plus one ``empty_cache`` — graph capture or rebind, the hoist's cache fill and
the 100 steps inside, the probe's seconds also apart as ``probe_s``; featurisation and PDB writing outside, as upstream's
on_test_batch_start/end bracket excludes its dataloader and runs its writer after the bracket).

Nothing here substitutes a smaller configuration for a failing one. Before each batch runs, the driver measures the allocator high-water of ONE eager
forward at batch 1 on the batch's first design (probe_forward_gb: this input, this line's model as configured), projects the batch's need as
what is resident + CAPACITY_MULT[regime] × B × that probe (regime: eager, graph replay), and
compares it with the device's total: a batch that does not fit is refused by name before anything of it runs (``CAPACITY … verdict=refused:
set generation.dataset.batch_size <= <b>``, exit 5; capacity_verdict) — the guard refuses, it never reduces. Kept CUDA graphs of earlier batches (L9)
yield first: while the projection does not fit and the cache keeps graphs, the oldest is freed and the batch judged again on what is resident then
(capacity_verdict_yielding) — the device the pass runs on, not a kept pool sized by an earlier batch, decides; a refusal names a batch that does
not fit the card on its own. An out-of-memory the model did not foresee, or any other error, propagates (the LEVER line prints from
``finally``), and design.py records the pass as failed.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from typing import Dict, Iterator, List, Optional, Tuple

from opt_core.precision import policy as P                              # the shared core's numerics policy: presets, apply / expect / live, the TF32 override names

PREFIX = "[genie3-opt]"
POLICY_FP32 = P.Policy("genie3_stock_fp32", matmul="highest", note="upstream sets no numerics switch: torch's defaults — IEEE fp32 matmuls (`highest`); cuDNN TF32 left at torch's default (the model has no cuDNN op)")
POLICY_TF32 = P.FP32_TF32                                                 # L13: TF32 tensor cores for fp32 matmuls (`high`) and cuDNN
EXIT_CAPACITY = 5                                        # a batch refused by name before it runs: its measured-and-scaled resident memory exceeds the device (capacity_verdict)
CAPACITY_HEADROOM = 0.90                                 # the fraction of the device's total memory a batch's projected need may claim
CAPACITY_MULT = {"eager": 1.35, "graph": 2.0}         # projected need ÷ (B × the batch-1 probe's high-water), from the line's own measurements on binder bin l
                                                         # (670 padded tokens, H100 80 GB): an eager forward at batch B peaks at 1.31 × B ×
                                                         # probe (21.3 GB at B=8, 42.4 GB at B=16 for probe 2.02 GB) → eager 1.35; a captured graph's private pool KEEPS
                                                         # 1.16 (chunked transition) – 1.35 (unchunked, L12) × B × probe for the whole request and the eager pre-part
                                                         # (frames, single and pair featurisers) runs at batch B beside it every step → graph 2.0 (exact at 16 ran inside
                                                         # it; fast at 16 died at ≥ 1.8 with 57.9 GiB in pools). A projection with margin, not a guarantee: an out-of-memory it did not
                                                         # foresee still propagates by name


def probe_slice(bd: Dict, b: int) -> Dict:
    """The batch's first design as a batch of one: every tensor whose leading dimension is the batch sliced to [:1], the rest as is; the hoist's
    per-batch keys left out (the probe must not fill or read the static-term cache)."""
    import torch
    out = {}
    for k, v in bd.items():
        if str(k).startswith("_g3cap"):
            continue
        out[k] = v[:1] if (torch.is_tensor(v) and v.dim() > 0 and int(v.shape[0]) == b and b > 1) else v
    return out


def probe_forward_gb(model, bd: Dict, b: int, s_vec, n_timestep: int) -> float:
    """The allocator high-water DELTA (GB) of one eager denoiser forward at batch 1 on this batch's first design, at the first step's timestep:
    the line's own model as configured (chunked or unchunked pair transition, the patches, the hoist off for the probe), so what it measures is
    this input's and this line's per-design activation footprint — binder conditioning, motif features and token count included. The forward
    draws no random number and writes nothing the sampling reads; its output is dropped."""
    import torch
    b1 = probe_slice(bd, b)
    x1 = torch.zeros_like(b1["gt_atom_positions"])
    t1 = (s_vec[:1] / n_timestep)
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        out = model(batch=b1, xl=x1, t=t1)["xl"]
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del out, b1, x1, t1
    return max(0.0, (peak - base) / 2 ** 30)


def capacity_regime(cuda_graphs: bool) -> str:
    return "graph" if cuda_graphs else "eager"


def device_resident_gb() -> Tuple[float, float]:
    """(resident GB, total GB) of the device after returning torch's cached free blocks: resident = total − free as the driver reports it, i.e.
    the CUDA context, live tensors and the private pools of kept CUDA graphs (which no other allocation can use) — not torch's reusable cache."""
    import torch
    torch.cuda.synchronize()
    torch.cuda.empty_cache()                                              # cached free blocks go back to the device; graph pools and live tensors stay
    free_b, total_b = torch.cuda.mem_get_info()
    return (total_b - free_b) / 2 ** 30, total_b / 2 ** 30


def capacity_verdict(b: int, probe_gb: float, resident_gb: float, total_gb: float, regime: str, headroom: float = CAPACITY_HEADROOM) -> Tuple[float, bool, Optional[int]]:
    """(projected need GB, fits, the largest smaller batch that fits or None) for a batch of `b` designs about to run: need = what is resident on
    the device now (context, weights, this batch's features, kept graph pools of other shapes; device_resident_gb) + CAPACITY_MULT[regime] × b ×
    the batch-1 probe's high-water; it fits when need ≤ headroom × the device's total. The guard refuses by name before the batch runs; it never
    reduces the batch (the caller chooses: `--batch-size`); kept graphs of earlier batches yield to it first (capacity_verdict_yielding)."""
    mult = CAPACITY_MULT[regime]
    need = float(resident_gb) + mult * b * float(probe_gb)
    budget = headroom * float(total_gb)
    fits = [x for x in (1, 2, 4, 8, 16, 32, 64, 128) if x < b and float(resident_gb) + mult * x * float(probe_gb) <= budget]
    return need, need <= budget, (max(fits) if fits else None)


def evict_oldest_graph(cache) -> bool:
    """Drop the graph cache's least-recently kept graph (the step GraphCache.give takes past its capacity), so its private pool can go back to
    the device; False when the cache keeps none."""
    if cache is None or not cache.store:
        return False
    k0 = next(iter(cache.store)); lst = cache.store[k0]
    old = lst.pop(0); cache.n -= 1
    del old
    if not lst:
        del cache.store[k0]
    return True


def capacity_verdict_yielding(b: int, probe_gb: float, regime: str, cache, measure=None) -> Tuple[float, bool, Optional[int], float, float, int]:
    """The capacity verdict for a batch about to run, the kept graphs (L9) yielding to it: while the projection does not fit and the graph
    cache keeps captured graphs of earlier batches, the oldest is freed (its private pool returned to the device: gc + empty_cache inside
    `measure`) and the batch is judged again on what is resident then. A kept graph is a reuse opportunity sized by what the device had
    free when it was captured — on a smaller card, or beside a batch near the card's ceiling, it is never the reason a batch that fits by
    itself is refused (a batch of a kept shape is then captured afresh: the same kernels, the same bytes). Constants unchanged: need is
    capacity_verdict's at every step. Returns (need, fits, b_ok, resident_gb, total_gb, evicted)."""
    measure = measure or device_resident_gb
    resident_gb, total_gb = measure()
    need, fits, b_ok = capacity_verdict(b, probe_gb, resident_gb, total_gb, regime)
    evicted = 0
    while not fits and evict_oldest_graph(cache):
        evicted += 1
        gc.collect()
        resident_gb, total_gb = measure()
        need, fits, b_ok = capacity_verdict(b, probe_gb, resident_gb, total_gb, regime)
    return need, fits, b_ok, resident_gb, total_gb, evicted


def capacity_line(b: int, n_token: int, regime: str, probe_gb: float, need: float, resident_gb: float, total_gb: float, ok: bool, b_ok: Optional[int]) -> str:
    """``CAPACITY batch=<B> n_token=<N> regime=<eager|graph> probe_gb=<p> need_gb=<x> resident_gb=<y> total_gb=<z> verdict=ok|refused: <advice>``."""
    advice = "" if ok else (f"refused: set generation.dataset.batch_size <= {b_ok} or lower" if b_ok else "refused: this input does not fit at batch 1 on this device")
    return (f"{PREFIX} CAPACITY batch={b} n_token={n_token} regime={regime} probe_gb={probe_gb:.2f} need_gb={need:.1f} resident_gb={resident_gb:.1f} "
            f"total_gb={total_gb:.1f} verdict={'ok' if ok else advice}")


def kit_dirs():
    """(<tree>/opt/forward/fast_inference/driver, <tree>/opt/forward/g3cap) of this package's tree (stack.kit_dir: GENIE3_OPT_HOME, else
    MODEL_OPT, else the package's own location; registry.KIT_DIRS). Refused by name when either carried driver is missing."""
    from genie3_opt.registry import DRIVER, G3CAP, G3CAP_DRIVER, G3LEAN, KIT
    from genie3_opt.stack import kit_dir
    kitdrv = os.path.join(kit_dir(KIT), *os.path.dirname(DRIVER).split("/"))
    capdir = kit_dir(G3CAP)
    for p in (os.path.join(kitdrv, os.path.basename(DRIVER)), os.path.join(capdir, G3LEAN), os.path.join(capdir, G3CAP_DRIVER)):
        if not os.path.isfile(p):
            raise SystemExit(f"g3batch.py: the carried driver it imports is missing: {p}")
    return kitdrv, capdir


def chunk_pair_transition_per_design(model, n_rows: int) -> int:
    """L12: every PairTransition runs its two-Linear MLP in chunks of ONE DESIGN (chunk_size = the pair tensor's row count N: upstream's
    chunk_layer flattens the (B, N) leading dims into B·N rows of [N, c], so N rows = one design's N×N×c slab, B chunks per call) instead
    of upstream's 4-row chunks (B·N/4 sequential calls per layer per step). The hidden 4·c intermediate is one design's, never the whole
    batch's: at 708 tokens × batch 8 that is 32 GB → 4 GB of live tensors for the same arithmetic. Set per batch shape (before a capture
    records it); returns the number of modules set."""
    from genie3.generation.model.module.transition import PairTransition
    n = 0
    for m in model.modules():
        if isinstance(m, PairTransition):
            m.chunk_size = int(n_rows)
            n += 1
    return n


def n_designs_of(dm, limit=None) -> int:
    """The request's design count (the dataset's length, or --limit of it)."""
    n = len(dm.dataset)
    return n if limit is None else min(int(limit), n)


def featurize_batches(dm, device, batch_size: int, limit=None) -> Iterator[Tuple[List[str], Dict, Dict]]:
    """Per batch exactly as stock at this batch size, ONE BATCH AT A TIME (a generator: the resident feature set is one batch's, whatever
    the request's size — upstream's DataLoader featurises lazily in the same order): GenieTestSampler's consecutive index blocks
    (np.arange(n)[i*B:(i+1)*B]) through the dataset's own __getitem__ (identical numpy RNG draws in identical order), the DataLoader's
    leading dimension of 1, then prepare_tensor_features(reduce_dim=0) on the device (runner.py test_step). Yields (names, device batch,
    cpu raw features)."""
    import numpy as np
    import torch
    from genie3.generation.utils.feat_utils import prepare_tensor_features
    ds = dm.dataset
    n = n_designs_of(dm, limit)
    idx = np.arange(n)
    for i in range(0, n, batch_size):
        feats_np, names = ds[idx[i:i + batch_size]]
        raw = {k: torch.as_tensor(v).unsqueeze(0) for k, v in feats_np.items()}
        bd = prepare_tensor_features({k: v.to(device) for k, v in raw.items()}, reduce_dim=0)
        bd["_g3fast_cond_group_max"] = int(bd["cond_group"].max().item())          # the static cond-group count the patched pair featuriser reads (g3fast_patches.py V1PairFeatureNet patch)
        yield [str(x) for x in names], bd, raw


def step_vectors(tab, B: int, device) -> list:
    """The per-step timestep vectors as upstream builds them (sampler.py:109: torch.Tensor([step] * batch_size).int().to(device)), built once
    per batch size instead of once per step: same values, dtype, device."""
    import torch
    return [torch.Tensor([s] * B).int().to(device) for s in tab.steps]



def write_batch(config, raw_cpu: Dict, names: List[str], xs, outdir: str, ai=None) -> None:
    """Upstream's writer on this batch, called as runner.py on_test_batch_end calls it (postprocess.py: one PDB per design name; ``ai`` = the
    sequence stage's one-hot residue types when the request sets predict_sequence, written as upstream writes them)."""
    from genie3.generation.runner.postprocess import postprocess
    outputs = {"xl": xs.detach().cpu()}
    if ai is not None:
        outputs["ai"] = ai.detach().cpu()
    postprocess(source=config.dataset.source, outputs=outputs, batch=(raw_cpu, [(n,) for n in names]), outdir=outdir)


def lever_line(ev: dict) -> str:
    return (f"{PREFIX} LEVER g3batch batch={ev.get('batch')} cuda_graphs={int(bool(ev.get('cuda_graphs')))} hoist={int(bool(ev.get('hoist')))} "
            f"reuse_graphs={ev.get('reuse_graphs')} pt_chunk={ev.get('pt_chunk') or 'stock'} compile={ev.get('compile') or 'none'} lean_pair={int(bool(ev.get('lean_pair')))} wide={int(bool(ev.get('wide')))} alloc={ev.get('alloc') or 'default'} tf32={int(bool(ev.get('tf32')))} policy={ev.get('policy')} graph_captures={ev.get('graph_captures')} "
            f"graph_reuses={ev.get('graph_reuses')} cache_hits={ev.get('cache_hits')} cache_misses={ev.get('cache_misses')} hoist_fills={ev.get('hoist_fills')} "
            f"hoist_anomalies={ev.get('hoist_anomalies')} patches={int(bool(ev.get('patches')))} trimul={ev.get('trimul', 'stock')} "
            f"batches={ev.get('batches_done')}/{ev.get('batches')} finished={int(bool(ev.get('finished')))} capacity={'refused' if ev.get('capacity_refused') else 'ok'}")


def main(argv=None) -> int:
    from genie3_opt import jitdirs, kernels as KK, trimul as KT      # the compile caches' directory rule (jitdirs), L7's adapter over the shared core (trimul) and the KERNELS census words (kernels); light imports, no torch
    from genie3_opt import sampling as SMP                  # the per-batch sampling loop (sample_batch): one function, called once per batch
    ap = argparse.ArgumentParser(prog="g3batch.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--batch-size", type=int, default=8, help="designs per denoiser call, grouped and noised as upstream's generation.dataset.batch_size (L11)")
    ap.add_argument("--cuda-graphs", action="store_true", help="CUDA-graph replay of the denoiser core (L4)")
    ap.add_argument("--hoist", action="store_true", help="the pair featuriser's step-invariant terms once per batch (L8)")
    ap.add_argument("--reuse-graphs", type=int, default=0, help="keep up to R captured graphs keyed by feature-shape signature and replay them for later batches of the same shapes (L9)")
    ap.add_argument("--pt-chunk", choices=("design",), default=None, help="PairTransition chunk = one design's rows per call instead of upstream's 4-row chunks (L12: design)")
    ap.add_argument("--compile", action="store_true", help="torch.compile (inductor) of the denoiser core, recorded inside the kit's CUDA graph (L16, class 2-3; needs --cuda-graphs)")
    ap.add_argument("--lean-pair", action="store_true", help="reference-lifetime levers of the pair stack: LatentTransformer ownership passing + the stock TriMul forward with in-place gating (L17, class 1)")
    ap.add_argument("--wide-capture", action="store_true", help="the pair featuriser's per-step tail inside the CUDA graph, the hoist's terms as graph-owned buffers (L18, class 1; needs --cuda-graphs --hoist --lean-pair)")
    ap.add_argument("--alloc", choices=("expandable",), default=None, help="the CUDA caching allocator with expandable segments + the capacity probe once per feature shape (L19)")
    ap.add_argument("--tf32", action="store_true", help="TF32 tensor-core matmuls (L13, class 3: the tolerance tier's precision lever; never on an exact line)")
    ap.add_argument("--trimul", choices=KT.CHOICES, default=KT.DEFAULT, help="the TriangleMultiplicativeUpdate provider: stock (the module's own forward) | fpf (L7: the shared core's fused kernel, class 3 — mode fast's; never on an exact line)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--timings", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--verbose", action="store_true", help="per-batch detail on stderr (upstream's --verbose, passed through by the package)")
    ap.add_argument("--shard-id", type=int, default=0, help="upstream's `generate --shard-id K`: this pass samples shard K's share of every problem's n_sample (0 <= K < M)")
    ap.add_argument("--num-shards", type=int, default=1, help="upstream's `generate --num-shards M`: the request sliced as upstream's loader slices it (1 = the whole request)")
    args = ap.parse_args(argv)
    if args.batch_size < 1:
        raise SystemExit("g3batch.py: --batch-size must be >= 1")
    if args.reuse_graphs and not args.cuda_graphs:
        raise SystemExit("g3batch.py: --reuse-graphs needs --cuda-graphs (a graph cache of no graphs)")

    hits = P.tf32_override_env()                                          # a TF32 library override in this process's environment moves every line off its policy: refused by name before torch initialises (the package's child environment strips them; this is the bare driver's own gate)
    if hits:
        raise SystemExit(f"g3batch.py: {P.tf32_override_refusal(hits)}")
    kitdrv, capdir = kit_dirs()
    sys.path.insert(0, kitdrv)
    import g3fast as G                       # noqa: E402  the resident driver, unchanged
    import g3fast_patches                    # noqa: E402
    sys.path.insert(0, capdir)
    import g3cap as C                        # noqa: E402  the capture-cell module, unchanged (hoist, graph cache)
    import g3lean as L                       # noqa: E402  the capture line's memory levers (L17 lean pair stack, L18 wide capture)
    import logging
    import torch

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
    if args.alloc == "expandable":                                       # L19: the caching allocator maps its segments expandable (less fragmentation beside the graph pools); read at CUDA init, so set first
        if torch.cuda.is_initialized():
            raise SystemExit("g3batch.py: --alloc expandable must be applied before CUDA is initialised in this process")
        conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        if "expandable_segments" not in conf:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (conf + "," if conf else "") + "expandable_segments:True"
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    T: dict = {"argv": sys.argv[1:] if argv is None else list(argv), "driver": "g3batch", "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "tag": args.tag}
    ev: dict = {"batch": args.batch_size, "cuda_graphs": args.cuda_graphs, "hoist": args.hoist, "reuse_graphs": args.reuse_graphs, "pt_chunk": None, "compile": None, "lean_pair": 0, "wide": 0, "alloc": None, "tf32": args.tf32,
                "policy": None, "graph_captures": 0, "graph_reuses": 0, "graph_evictions": 0, "cache_hits": 0, "cache_misses": 0, "hoist_fills": 0, "hoist_anomalies": 0, "patches": False, "batches": None, "batches_done": 0, "finished": False,
                "trimul": args.trimul}
    if args.alloc == "expandable":
        T["alloc"] = {"conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"), "probe": "once-per-shape"}; ev["alloc"] = "expandable"
    t_all = time.perf_counter()
    rc = 0
    try:
        if not (args.num_shards >= 1 and 0 <= args.shard_id < args.num_shards):
            raise SystemExit(f"g3batch.py: --shard-id {args.shard_id} --num-shards {args.num_shards}: the shard index is 0 <= K < M")
        exp, config, model, diffusion, sampler, dm, setup_s, seed = G.build_everything(args.config, dev, args.seed, args.outdir, shard_id=args.shard_id, num_shards=args.num_shards)   # the shard flags slice the request as upstream's generate does
        if args.num_shards > 1:
            T["shard"] = f"{args.shard_id}/{args.num_shards}"                     # the run record names the shard; n_designs below is the shard's share
        beyond = [k for k in ("predict_sidechain",) if getattr(sampler, k, False)]
        if getattr(sampler, "_beam_search", None) is not None or getattr(getattr(config, "inference", None), "search", None) is not None:
            beyond.append("inference.search")
        if beyond:                                                        # the line computes upstream's generate without beam search / the sidechain stage: a request beyond it is refused by name, never run in part (design.py refuses it before the driver starts)
            raise SystemExit(f"g3batch.py: the request sets {', '.join(beyond)} — outside the batched capture line; run it on the stock route (--mode off)")
        predict_sequence = bool(getattr(sampler, "predict_sequence", False))   # upstream's sequence stage (Sampler.sample: `_sample_sequence` after the structure loop) runs after this driver's loop, per batch
        T["predict_sequence"] = predict_sequence
        T["model_setup_s"] = round(setup_s, 3); T["seed"] = seed
        # The line's numerics policy, through the shared core (opt_core.precision.policy — the one home of the switches): the fp32 line
        # PROVES upstream's own policy is in force and sets nothing (expect: IEEE fp32 matmuls, torch's `highest`; a live TF32 state —
        # e.g. a library override — is a PolicyMismatch and nothing runs); --tf32 APPLIES the TF32 preset (matmul `high`, cuDNN TF32 on).
        if args.tf32:                                                     # class 3, whole process, recorded; the exact line never passes it
            T["precision_policy"] = P.apply(POLICY_TF32, torch)
        else:
            T["precision_policy"] = P.expect(POLICY_FP32, torch)
        rb = P.live(torch)
        if bool(rb["matmul_tf32"]) is not bool(args.tf32):                # the readback, not the flag, is the fact: TF32 matmuls live exactly when the line asked for them
            raise SystemExit(f"g3batch.py: numerics readback matmul_tf32={rb['matmul_tf32']} matmul={rb['matmul']} contradicts the line (--tf32={bool(args.tf32)}); nothing runs")
        T["numerics_readback"] = rb
        T["tf32"] = bool(rb["matmul_tf32"])
        T["matmul_precision"] = rb["matmul"]
        ev["tf32"] = T["tf32"]; ev["policy"] = T["precision_policy"]["policy"]
        g3fast_patches.apply(verbose=False)
        from genie3_opt.stack import kit_levers_applied
        markers = kit_levers_applied()
        T["patches"] = len(markers) == 2 and markers; ev["patches"] = bool(T["patches"])
        if args.hoist:
            C.install_hoist()
            T["hoist"] = True
        if args.compile and not args.cuda_graphs:
            raise SystemExit("g3batch.py: --compile is a CUDA-graph lever (the compiled core is what the graph records): it needs --cuda-graphs")
        if args.wide_capture and not (args.cuda_graphs and args.hoist and args.lean_pair):
            raise SystemExit("g3batch.py: --wide-capture records the featuriser tail inside the CUDA graph over the hoist's terms and hands the pair tensor over by ownership: it needs --cuda-graphs --hoist --lean-pair")
        if args.lean_pair:                                                # L17: reference lifetimes only — the same statements, one pair tensor fewer alive at the core's peak
            n_lt = L.install_lt_release()
            n_tm = L.install_trimul_lean() if args.trimul == "stock" else 0   # the stock provider's forward (mode fast's fused kernel replaces the forward: nothing to lean)
            T["lean_pair"] = {"lt_release": n_lt, "trimul_lean": n_tm}; ev["lean_pair"] = 1
        if args.wide_capture:
            T["wide_capture"] = {"graphs": 0}; ev["wide"] = 1
        if args.compile:
            os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", jitdirs.inductor_cache_dir())   # a warm inductor cache turns tens of seconds per shape into seconds; it is read back as code, so its directory is one this user alone writes (jitdirs.py) — never a fixed shared name
        if args.trimul == "fpf":                                          # L7: patch the TriMul class BEFORE the first capture so the graphs record the kernel's launches (route refusals raise here, by name)
            T["trimul_route"] = KT.enable()
        tab = G.StepTables(sampler, dev)
        T["sampler"] = {"n_timestep": tab.n_timestep, "n_sample_step": tab.n_sample_step, "step_size": tab.step_size,
                        "eta": tab.eta, "noise_scale": tab.noise_scale, "direction_scale": tab.direction_scale}
        B = int(args.batch_size)
        T["n_designs"] = n_designs_of(dm, args.limit); T["stock_batch"] = B
        n_batches = -(-T["n_designs"] // B); T["batches"] = n_batches; ev["batches"] = n_batches
        T["n_token"] = []; T["featurize_s"] = 0.0
        cache = C.GraphCache(int(args.reuse_graphs)) if args.cuda_graphs else None
        gen = torch.cuda.default_generators[dev.index or 0]
        svec_by_b: Dict[int, list] = {}
        probe_seen: Dict[tuple, float] = {}
        per_batch: List[dict] = []
        write_s = 0.0
        n_checks = 0
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        peak_sofar = 0
        with torch.inference_mode():
            STATE0 = G.global_numerics_state(); T["global_numerics_state"] = STATE0   # the process-global numerics switches as the loop runs under them (inference mode included); asserted unchanged after every batch
            feed = featurize_batches(dm, dev, B, args.limit)              # lazy: one batch featurised at a time, before its clock starts (upstream: the DataLoader's fetch precedes on_test_batch_start)
            for bi in range(n_batches):
                t_f = time.perf_counter()
                names, bd, raw = next(feed)
                T["featurize_s"] += time.perf_counter() - t_f; T["n_token"].append(int(bd["token_mask"].shape[1]))
                b = int(bd["gt_atom_positions"].shape[0])
                n_tok = int(bd["token_mask"].shape[1])
                if args.pt_chunk == "design":                             # L12: PairTransition chunk = one design's rows at THIS batch shape (a capture records it; a kept graph keeps its own)
                    n_pt = chunk_pair_transition_per_design(model, n_tok)
                    rows = T.setdefault("pt_chunk", {"chunk": "design", "modules": n_pt, "rows": []})["rows"]
                    if n_tok not in rows:
                        rows.append(n_tok)
                    ev["pt_chunk"] = "design"
                if b not in svec_by_b:
                    svec_by_b[b] = step_vectors(tab, b, dev)
                t_b = time.perf_counter()                                # the batch's clock starts here, BEFORE the probe: the probe is per-batch work of this line and its seconds are inside wall_s (reported apart as probe_s)
                t_p = t_b
                peak_sofar = max(peak_sofar, torch.cuda.max_memory_allocated())      # the probe resets the peak counter; the loop's peak is kept by hand
                saved = gen.get_state()                                   # the probe's forward draws nothing; its generator state is bracketed like the capture's (by-construction neutrality)
                psig = (C.feat_signature(bd), b) if args.alloc else None   # L19: the probe's answer is a function of the feature-shape signature — probed once per shape per process
                if psig is not None and psig in probe_seen:
                    probe_gb = probe_seen[psig]
                else:
                    probe_gb = probe_forward_gb(model, bd, b, svec_by_b[b][0], tab.n_timestep)
                    if psig is not None:
                        probe_seen[psig] = probe_gb
                torch.cuda.synchronize(); gen.set_state(saved)
                regime = capacity_regime(bool(args.cuda_graphs))
                need, fits, b_ok, resident_gb, total_gb, evicted = capacity_verdict_yielding(b, probe_gb, regime, cache)   # kept graphs of earlier batches yield (oldest first) before a batch that fits by itself is refused
                ev["graph_evictions"] += evicted
                probe_s = time.perf_counter() - t_p                       # the probe forward + empty_cache + the verdict (+ any eviction): a component of this batch's wall_s
                T.setdefault("capacity", []).append({"batch": b, "n_token": n_tok, "regime": regime, "probe_gb": round(probe_gb, 3), "need_gb": round(need, 2), "resident_gb": round(resident_gb, 2),
                                                     "total_gb": round(total_gb, 1), "fits": fits, "probe_s": round(probe_s, 3), "evicted_graphs": evicted})
                T["probe_s"] = round(T.get("probe_s", 0.0) + probe_s, 3)
                print(capacity_line(b, n_tok, regime, probe_gb, need, resident_gb, total_gb, fits, b_ok), file=sys.stderr, flush=True)   # one line per batch: ok, or the refusal with its advice
                if not fits:                                              # refused by name BEFORE the batch runs (no capture, no partial OOM): the pass ends failed, exit 5, the earlier batches' PDB files stay written
                    ev["capacity_refused"] = {"batch": b, "n_token": n_tok, "regime": regime, "probe_gb": round(probe_gb, 3), "need_gb": round(need, 1), "suggest_batch": b_ok}
                    raise SystemExit(EXIT_CAPACITY)
                SB = svec_by_b[b]
                mask = bd["gt_atom_mask"]
                if args.hoist:
                    bd["_g3cap_hoist"] = True
                hoist_pre = "_g3cap_static" in bd                         # the batch's own dict arrives without a static-term cache (the capture's warm-up or the first eager step fills it from this batch's values)
                gd, how, sig = None, "eager", None
                if args.cuda_graphs:
                    sig = C.feat_signature(bd)
                    gd = cache.take(sig)
                    if gd is None:
                        t_c = time.perf_counter()
                        saved = gen.get_state()                           # the capture registers the default generator; the forward draws nothing — restore its state (as g3fast.GraphedDenoiser's own capture does)
                        gd = G.GraphedDenoiser(model, bd, tab.n_timestep, pool=None, compile=("inductor" if args.compile else None), wide=bool(args.wide_capture))
                        if args.wide_capture:
                            T["wide_capture"]["graphs"] += 1
                        if args.compile:                                  # L16's record: the backend word once, the shapes compiled (one per capture)
                            T["compile"] = {"backend": "inductor", "shapes": ev["graph_captures"] + 1}; ev["compile"] = "inductor"
                        torch.cuda.synchronize(); gen.set_state(saved)
                        T.setdefault("graph_capture_s", []).append(round(time.perf_counter() - t_c, 3)); ev["graph_captures"] += 1
                        how = "capture"
                    else:
                        C.rebind_graph(gd, bd)
                        how = "reuse"
                live = gd.batch if gd is not None else bd                 # the dict the denoiser reads this batch (a kept graph's static dict after rebind, else the batch's own)
                if how == "reuse":
                    hoist_pre = hoist_pre or "_g3cap_static" in live      # a kept graph's dict must come back from rebind without the previous batch's cache
                xs = SMP.sample_batch(gd, model, bd, tab, SB, mask, G.ddim_math)   # the batch's reverse-diffusion sampling (genie3_opt.sampling: the initial noise, then per DDIM step one denoiser call + the step
                                                                          # arithmetic — the statements of sampler.py:98-110 / ddim.py _step at batch B); its per-step temporaries are the function's locals,
                                                                          # released at return, so a graph the guard evicts frees its whole pool
                ai, seq_s = None, 0.0
                if predict_sequence:                                      # upstream's sequence stage, the SAME statements (Sampler._sample_sequence: x re-noised at t=1, the denoiser's `ai` logits decoded in the
                    t_q = time.perf_counter()                             # sampler's decoding order), run eagerly on the denoiser module right after the structure loop — where Sampler.sample runs it, so the
                    seq = {k: v for k, v in live.items() if not str(k).startswith("_g3cap")}   # one noise draw it makes follows the loop's draws as on the stock path; a private copy of the batch's
                    ai = sampler._sample_sequence(model, seq, xs)         # features without the kit's hoist / capture keys (upstream's own featuriser math; the graph's static dict is never rebound)
                    del seq
                    torch.cuda.synchronize(); seq_s = time.perf_counter() - t_q
                hoist_how = "off"
                if args.hoist:                                            # one fill per batch is the lever's design (the terms are per batch); a cache found before the batch or absent after it is a named, counted event
                    filled = "_g3cap_static" in live
                    hoist_how = "fill" if (filled and not hoist_pre) else ("stale" if hoist_pre else "miss")
                    ev["hoist_fills"] += int(hoist_how == "fill"); ev["hoist_anomalies"] += int(hoist_how != "fill")
                if gd is not None:
                    cache.give(sig, gd); gd = None
                ev["graph_reuses"] += int(how == "reuse")
                torch.cuda.synchronize()
                wall = time.perf_counter() - t_b
                per_batch.append({"names": names, "n_token_pad": int(bd["token_mask"].shape[1]), "n_token_real": [int(v) for v in bd["token_mask"].sum(dim=1).tolist()],
                                  "wall_s": round(wall, 3), "probe_s": round(probe_s, 3), "graph": how, "hoist": hoist_how})
                G.assert_global_state(STATE0, f"after batch {bi + 1}"); n_checks += 1
                t_w = time.perf_counter()
                write_batch(config, raw, names, xs, args.outdir, ai=ai)  # this batch's PDBs as the batch completes, outside the batch's clock (upstream: after its on_test_batch bracket); `ai` when the request predicts a sequence
                G.assert_global_state(STATE0, f"after PDB writing, batch {bi + 1}")
                dtw = time.perf_counter() - t_w; write_s += dtw
                per_batch[-1]["write_s"] = round(dtw, 3)
                if predict_sequence:
                    per_batch[-1]["sequence_s"] = round(seq_s, 3)         # the sequence stage's wall, inside the batch's wall_s and outside the sampling unit
                ev["batches_done"] = bi + 1
                print(f"{PREFIX} BATCH {bi + 1}/{n_batches} designs={len(names)} n_token={int(bd['token_mask'].shape[1])} graph={how} hoist={hoist_how} wall_s={wall:.3f}", flush=True)
                bd.pop("_g3cap_static", None)
                del bd, raw, live, ai                                     # the batch's device features (and the sequence stage's output) are released before the next batch is featurised
            T["featurize_s"] = round(T["featurize_s"], 3)
            if ev["batches_done"] != n_batches or sum(len(p["names"]) for p in per_batch) != T["n_designs"]:
                raise SystemExit(f"g3batch.py: {ev['batches_done']}/{n_batches} batches, {sum(len(p['names']) for p in per_batch)}/{T['n_designs']} designs sampled — the request is not complete")
            T["numerics_readback_end"] = P.live(torch)                    # the switches as the last batch left them (STATE0 asserted unchanged throughout; this is the record's own end-of-run readback)
        if cache is not None:
            ev["cache_hits"], ev["cache_misses"] = cache.hits, cache.misses
            if args.reuse_graphs > 0:                                     # L9's record: written only when a cache was asked for
                T["graph_cache"] = {"capacity": cache.capacity, "hits": cache.hits, "misses": cache.misses, "kept": cache.n, "captures": ev["graph_captures"], "evictions": ev["graph_evictions"]}
            cache.clear()
        T["per_batch"] = per_batch
        T["sampling_loop_s"] = round(sum(p["wall_s"] for p in per_batch), 3)
        T["s_per_design_sampling"] = round(T["sampling_loop_s"] / max(1, T["n_designs"]), 4)
        if len(per_batch) > 1:                                            # the throughput statistic: batches after the first / their designs
            rest = per_batch[1:]
            T["s_per_design_steady"] = round(sum(p["wall_s"] for p in rest) / max(1, sum(len(p["names"]) for p in rest)), 4)
        T["write_s"] = round(write_s, 3)                                # PDB-writer seconds: the sum of per_batch[].write_s (outside every batch's wall_s)
        T["peak_mem_GB_sampling"] = round(max(peak_sofar, torch.cuda.max_memory_allocated()) / 2 ** 30, 3)
        T["D59_global_state_checks_passed"] = n_checks
        ev["finished"] = True
    finally:
        T["total_s"] = round(time.perf_counter() - t_all, 3)
        if T.get("n_designs"):
            T["s_per_design_total_inprocess"] = round(T["total_s"] / T["n_designs"], 4)
        T["trimul"] = KT.evidence()                                       # L7's record: mode, the core census (served / fallback by reason / errors), the gate, the route, the cell row, the KERNELS word
        T["kernels"] = KK.census(T["trimul"], tf32=T.get("tf32"))
        T["lever"] = ev
        if args.timings:
            os.makedirs(os.path.dirname(os.path.abspath(args.timings)), exist_ok=True)
            json.dump(T, open(args.timings, "w"), indent=1, default=str)
        if args.trimul == "fpf":
            print(KT.lever_line(), file=sys.stderr, flush=True)           # the shared core's activation-evidence line for the TriMul lever (name, state, impl, served, fallback_by, gate)
        print(KK.line(T["kernels"]), file=sys.stderr, flush=True)         # the KERNELS census line (kernels.py): what the pair stack's kernel-servable ops ran on in this process
        print(lever_line(ev), file=sys.stderr, flush=True)
    short = {k: v for k, v in T.items() if k not in ("per_batch", "n_token", "global_numerics_state")}
    print(json.dumps(short, indent=1, default=str))
    return rc


if __name__ == "__main__":
    sys.exit(main())
