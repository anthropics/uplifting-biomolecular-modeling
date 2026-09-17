"""ef2_bwd_ckpt — memory-planned activation checkpointing of the FoldingTrunk under grad (numerics class: exact).

Stock checkpoints every PairUpdateBlock of every grad-enabled trunk pass (modeling_esmfold2_common.FoldingTrunk.forward), so the
backward recomputes one full block forward per block: at 2 trunk passes that is 48 block forwards per design step, ~1/3 of the
trunk's GPU time in the backward. Keeping a block's activations instead costs memory that scales with the pair count L² and with
the kernel set that produced them. This lever chooses HOW MANY blocks keep their activations from the memory actually free on the
device when it is enabled (weights, other models and any CUDA-graph pools already resident are thereby accounted for), a
per-block footprint per kernel set (KEPT_BYTES_PER_POS), a bound on the step's transient working set and a reserve; it installs the choice as the checkpoint policy of
ef2_autograd_kernels' patched trunk forward ('none' | 'ckpt:m' = checkpoint the first m blocks of each pass, keep the rest | 'block').
Two rules, both planned (the estimate and the budget computed either way): `budget` (default) = keep the most blocks whose estimated peak
fits the budget — the card is FILLED; `floor` (plan/enable floor=True) = the plan pinned to its floor: policy 'block', every block of
every pass checkpointed, kept 0, the record's reason word 'memory_floor' (stock's activation footprint: a mode whose axis is peak memory
chooses it by composition, never by hand — a hand-given `policy=` is a different thing, marked forced).

    import ef2_autograd_kernels as agk, ef2_bwd_ckpt
    agk.enable(model, ..., checkpoint="block")            # any policy; this lever replaces it
    rec = ef2_bwd_ckpt.enable(model, n_tokens=N)          # after every other resident allocation (graph pools) exists, before the loop
    ...                                                   # rec = {"policy", "kept_per_pass", "est_peak_gib", "budget_gib", "kernels", ...}
    ef2_bwd_ckpt.disable(model)                           # restores the policy agk.enable was given

Numerics: exact. A checkpointed block recomputes the same kernels on the same saved input inside the backward; kept or recomputed,
every tensor of the forward and of d(loss)/d(inputs) is bitwise identical (test_ef2_bwd_ckpt.py: 'none' / 'ckpt:m' / 'block' agree
bitwise on outputs and input-gradients, stock kernels and agk3 kernels).  Failure mode is memory, never numerics: the plan is an
upper bound built from the coefficients below plus `reserve_frac` of the device; a caller that adds resident memory AFTER
enable() (e.g. captures graphs later) must pass it as `extra_reserved_bytes`.

Footprint model (bf16 autocast, d_pair 256, 24 blocks; every term is bytes per pair position, × L², for the kernel set that serves):
    kept block-pass      KEPT_BYTES_PER_POS[kernels]: 'stock' (cuEquivariance triangle multiplication + the compiled stock transition),
                         'agk3' (K-A2 bmm2 + K-D3 refround_lean), 'fused3' (ef2_trimul fused + K-D3 refround_lean), 'ref3' (the fork's
                         reference triangle multiplication under grad, eager, + K-D3 refround_lean: what an ablated trimul lever leaves, the
                         loader's 'fused' backend being a no-grad path), 'refstock' (the stock COMPILED block on the reference triangle
                         multiplication: trimul and agk_transition both ablated)
    checkpointed block   512 B (its bf16 input pair tensor) in either kernel set
    step transient       one kept-block equivalent + 1 GiB (one block's recompute + backward working set, fp32 distogram logits and
                         losses, allocator rounding); eager fragmentation above that bound is what the reserve is for
    budget               (1 - reserve_frac) x total device memory
"""
from __future__ import annotations

import sys

import torch

import ef2_autograd_kernels as agk

NAME = "ef2_bwd_ckpt"
KIB = 1024.0
GIB = float(2 ** 30)

# bytes per pair position (L*L) per kernel set, see module docstring
KEPT_BYTES_PER_POS = {"stock": 11.2 * KIB, "agk3": 7.1 * KIB, "fused3": 5.0 * KIB,   # fused3 = ef2_trimul variant "fused" + K-D3 lean transition
                      "ref3": 19.0 * KIB, "refstock": 13.5 * KIB}                     # ref3 = the fork's REFERENCE triangle multiplication under grad, EAGER (the loader's kernel backend
                      # 'fused' is a no-grad path; backend None), beside K-D3's lean transition in agk's eager block path — what MODEL_OPT_LEVERS_OFF=trimul leaves on fast /
                      # big. refstock = the stock COMPILED pair block on that reference triangle multiplication (trimul AND agk_transition off: agk's block path steps
                      # back to the compiled block, whose graph saves less than the eager reference). A cuEquivariance or kit triangle multiplication beside the stock
                      # transition prices at "stock" (an upper bound there)
CKPT_INPUT_BYTES_PER_POS = 512.0
TRANSIENT_KEPT_EQUIV = 1.0
TRANSIENT_CONST_BYTES = 1.0 * GIB
LM_GRAPH_POOL_BYTES = 2.4 * GIB         # the ESMC-forward + pPPL CUDA-graph pools (ef2_esmc_graph + ef2_pppl_graph), 195..700 tokens, binder 80:
                                        # pass as extra_reserved_bytes when those graphs will be captured AFTER enable()
DEFAULT_RESERVE_FRAC = 0.10          # of total device memory, kept free on top of the transient bound (fragmentation, other streams)


class PlanError(RuntimeError):
    """Raised by name when the lever cannot be applied (agk not enabled on the model) — never a silent fallback."""


def kernels_of(model: torch.nn.Module) -> str:
    """The kernel set the plan prices a kept block-pass with — the set that ACTUALLY serves the grad-mode pair block (kernel_set): 'fused3'
    when ef2_trimul's fused triangle multiplication AND agk's lean transition serve (fast / big as shipped), 'agk3' when agk's own triangle
    multiplication (K-A2) and its lean transition serve, 'stock' when the cuEquivariance triangle multiplication serves (exact; the row is
    cuEquivariance + the compiled transition, an upper bound beside the lean one) or a kit triangle multiplication runs beside the stock
    compiled transition (MODEL_OPT_LEVERS_OFF=agk_transition: the row is an upper bound for it), 'ref3' when the fork's REFERENCE triangle multiplication serves under grad,
    eager, beside the lean transition — the loader's kernel backend 'fused' is a no-grad path, so that is what MODEL_OPT_LEVERS_OFF=trimul leaves
    on fast / big (the eager reference path keeps several times the fused kernel's bytes per pair position, hence
    its own row) —, 'refstock' when agk's block path steps back to the stock COMPILED block on that
    reference triangle multiplication (trimul and agk_transition both ablated). PlanError when agk is not enabled at all (the policy lives in agk's trunk forward)."""
    blocks = [m for m in model.modules() if getattr(m, "_agk_cfg", None) is not None]
    if not blocks:
        raise PlanError(f"{NAME}: ef2_autograd_kernels is not enabled on this model (agk.enable installs the trunk forward whose checkpoint policy this lever sets)")
    return kernel_set(blocks[0]._agk_cfg, getattr(model, "__dict__", {}).get("_ef2_trimul_handle"), cueq=any(_runs_cueq(b) for b in blocks))


def _runs_cueq(block) -> bool:
    """True when the block's stock triangle multiplication runs the cuEquivariance kernels (the fork's switch attributes, as ef2_trimul reads them)."""
    if getattr(block, "_kernel_backend", None) == "cuequivariance":          # PairUpdateBlock carries the fork's switch word itself
        return True
    tm = getattr(block, "tri_mul_out", None)
    return tm is not None and (getattr(tm, "_kernel_backend", None) == "cuequivariance" or bool(getattr(getattr(tm, "_engine", None), "_use_kernels", False)))


def kernel_set(cfg, trimul_handle=None, cueq: bool = False) -> str:
    """kernels_of's rule on one block's agk Config, the model's ef2_trimul handle (if any) and whether the stock triangle multiplication runs the
    cuEquivariance kernels (else the fork's reference path serves it under grad) — see kernels_of."""
    lean = getattr(cfg, "transition", None) is not None                      # agk's lean transition serves (else the stock compiled transition)
    fused = trimul_handle is not None and getattr(trimul_handle, "variant", None) == "fused"
    if lean and fused:
        return "fused3"                                                  # ef2_trimul's fused triangle multiplication + agk's lean transition
    if lean and getattr(cfg, "trimul", None) is not None:
        return "agk3"                                                    # agk's K-A2 triangle multiplication + its lean transition
    if fused or getattr(cfg, "trimul", None) is not None or cueq:
        return "stock"                                                   # a kit or the cuEquivariance triangle multiplication beside the stock transition, or cuEquivariance beside the lean one: the stock row bounds them
    return "ref3" if lean else "refstock"                                # the fork's reference triangle multiplication under grad: eager beside the lean transition / inside the stock compiled block


def n_blocks_of(model: torch.nn.Module, trunks: str = "main") -> int:
    import transformers.models.esmfold2.modeling_esmfold2_common as C
    for name, m in model.named_modules():
        if isinstance(m, C.FoldingTrunk) and not (trunks == "main" and "confidence" in name):
            return len(m.blocks)
    if isinstance(model, C.FoldingTrunk):
        return len(model.blocks)
    raise PlanError(f"{NAME}: no FoldingTrunk in the model")


def estimate_peak_bytes(n_tokens: int, kept_per_pass: int, num_passes: int, kernels: str, n_blocks: int = 24,
                        base_bytes: float = 0.0, copies: int = 1) -> float:
    """Upper bound of allocated bytes at the step's peak for a policy keeping `kept_per_pass` blocks per grad-enabled pass.
    `copies` > 1 when the kept activations live in per-model CUDA-graph pools of several resident models (each holds its own)."""
    L2 = float(n_tokens) * float(n_tokens)
    kept = KEPT_BYTES_PER_POS[kernels] * L2
    per_pass = kept_per_pass * kept + (n_blocks - kept_per_pass) * CKPT_INPUT_BYTES_PER_POS * L2
    transient = TRANSIENT_KEPT_EQUIV * kept + TRANSIENT_CONST_BYTES
    return base_bytes + copies * num_passes * per_pass + transient


FLOOR_REASON = "memory_floor"                              # the floor plan's reason word (record "reason", the LEVER line's reason=)


def plan(n_tokens: int, num_passes: int = 2, kernels: str = "agk3", budget_bytes: float | None = None, n_blocks: int = 24,
         base_bytes: float = 0.0, copies: int = 1, reserve_frac: float = DEFAULT_RESERVE_FRAC, total_bytes: float | None = None,
         floor: bool = False) -> dict:
    """Rule `budget` (floor=False): the policy keeping the most blocks whose estimated peak fits `budget_bytes` (default: total device
    memory × (1 − reserve)); 'block' when not even one kept block fits. Rule `floor` (floor=True): the all-checkpoint policy 'block'
    (kept 0) whatever fits, reason 'memory_floor' — the budget and the estimate (of the kept-0 peak) computed and returned all the same.
    Returns {"policy", "kept_per_pass", "est_peak_bytes", "budget_bytes", "rule", "floor", ["reason"], ...}."""
    if kernels not in KEPT_BYTES_PER_POS:
        raise PlanError(f"{NAME}: unknown kernel set {kernels!r} (known: {sorted(KEPT_BYTES_PER_POS)})")
    if total_bytes is None:
        total_bytes = float(torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory) if torch.cuda.is_available() else 80 * GIB
    if budget_bytes is None:
        budget_bytes = total_bytes * (1.0 - reserve_frac)
    best = 0
    if not floor:
        for k in range(n_blocks, -1, -1):                   # most kept first
            if estimate_peak_bytes(n_tokens, k, num_passes, kernels, n_blocks, base_bytes, copies) <= budget_bytes:
                best = k
                break
    policy = "none" if best >= n_blocks else ("block" if best <= 0 else f"ckpt:{n_blocks - best}")
    rec = {"policy": policy, "kept_per_pass": best, "n_blocks": n_blocks, "num_passes": num_passes, "kernels": kernels, "copies": copies,
           "est_peak_bytes": estimate_peak_bytes(n_tokens, best, num_passes, kernels, n_blocks, base_bytes, copies),
           "budget_bytes": budget_bytes, "base_bytes": base_bytes, "total_bytes": total_bytes, "n_tokens": n_tokens,
           "rule": "floor" if floor else "budget", "floor": bool(floor)}
    if floor:
        rec["reason"] = FLOOR_REASON
    return rec


def device_bytes_in_use() -> float:
    """Bytes held on the current device that a future eager allocation can NOT reuse: everything the driver has handed out (this
    process's live tensors and CUDA-graph private pools, the CUDA context, other processes) minus the free space cached inside this
    process's DEFAULT allocator pool (reusable; a graph's private pool is not). mem_get_info alone would count the allocator's cache
    of earlier steps as used and under-plan by tens of GiB."""
    free, total = torch.cuda.mem_get_info()
    reusable = 0
    for seg in torch.cuda.memory_snapshot():
        if tuple(seg.get("segment_pool_id", (0, 0))) == (0, 0) and seg.get("device", torch.cuda.current_device()) == torch.cuda.current_device():
            reusable += int(seg["total_size"]) - int(seg.get("allocated_size", seg["total_size"]))
    return float(total - free - reusable)


def enable(model: torch.nn.Module, n_tokens: int, num_passes: int = 2, kernels: str | None = None, extra_reserved_bytes: float = 0.0,
           copies: int = 1, reserve_frac: float = DEFAULT_RESERVE_FRAC, budget_bytes: float | None = None, policy: str | None = None,
           floor: bool = False, verbose: bool = True) -> dict:
    """Plan from the memory free on the device NOW (+ `extra_reserved_bytes` the caller will still allocate) and install the policy
    on every agk-patched trunk of `model` (instance-level; reversible via disable()). `floor=True` plans by the floor rule (policy
    'block', reason 'memory_floor': planned, not forced). `policy=` forces a policy by hand (same grammar; marked forced)."""
    kern = kernels or kernels_of(model)
    nb = n_blocks_of(model)
    if torch.cuda.is_available():
        base = device_bytes_in_use() + float(extra_reserved_bytes)      # everything anyone holds on the device now + what the caller adds later
        total = torch.cuda.mem_get_info()[1]
    else:
        total = 80 * GIB; base = float(extra_reserved_bytes)
    rec = plan(n_tokens, num_passes, kern, budget_bytes=budget_bytes, n_blocks=nb, base_bytes=base, copies=copies,
               reserve_frac=reserve_frac, total_bytes=float(total), floor=floor)
    if policy is not None:
        agk._ckpt_this_block(policy, 0, nb)                             # validates the grammar (raises ValueError on a bad policy)
        rec["policy"] = policy; rec["forced"] = True
    cfgs = {}                                                           # the distinct agk Config objects (trunk and blocks share one)
    for m in model.modules():
        cfg = getattr(m, "_agk_cfg", None)
        if cfg is not None:
            cfgs[id(cfg)] = cfg
    if not hasattr(model, "_bwd_ckpt_prev"):
        model._bwd_ckpt_prev = [(cfg, cfg.checkpoint) for cfg in cfgs.values()]   # the policy agk.enable was given, restored by disable()
    for cfg in cfgs.values():
        cfg.checkpoint = rec["policy"]
    rec.update({"name": NAME, "applied": bool(cfgs), "configs": len(cfgs), "est_peak_gib": round(rec["est_peak_bytes"] / GIB, 2),
                "budget_gib": round(rec["budget_bytes"] / GIB, 2), "base_gib": round(base / GIB, 2)})
    model._bwd_ckpt = rec
    if verbose:
        print(describe(model), file=sys.stderr, flush=True)              # the policy depends on the memory free at this moment: always named
    return rec


def disable(model: torch.nn.Module) -> None:
    for cfg, prev in getattr(model, "_bwd_ckpt_prev", []):
        cfg.checkpoint = prev
    for attr in ("_bwd_ckpt_prev", "_bwd_ckpt"):
        if hasattr(model, attr):
            delattr(model, attr)


def describe(model: torch.nn.Module) -> str:
    """One status line in the kit's LEVER style (the caller prints it; this module prints nothing)."""
    rec = getattr(model, "_bwd_ckpt", None)
    if not rec:
        return f"LEVER name={NAME} state=off"
    reason = f" reason={rec['reason']}" if rec.get("reason") else ""          # the floor plan names its reason (memory_floor); the budget rule's line is unchanged
    return (f"LEVER name={NAME} state=on policy={rec['policy']}{' (forced)' if rec.get('forced') else ''}{reason} kept_per_pass={rec['kept_per_pass']}/{rec['n_blocks']} "
            f"passes={rec['num_passes']} copies={rec['copies']} kernels={rec['kernels']} tokens={rec['n_tokens']} est_peak_gib={rec['est_peak_gib']} "
            f"budget_gib={rec['budget_gib']} base_gib={rec['base_gib']}")
