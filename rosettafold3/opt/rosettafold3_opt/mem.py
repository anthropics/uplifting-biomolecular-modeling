"""The memory policy of the kit rows (lever MEM): ``apply()`` at activation (the allocator export, the trunk-graph cache budget), ``wrap()``
when the kit module executes (the roll-out seams), ``report()`` (the exit tally's ``mem`` block). Every kit row carries it — ``exact``,
``fast`` and the memory mode ``big`` (big.py: the memory levers) alike; the core holds the allocator primitive
(``opt_core.mem.torch_alloc``), this module is the engine's install of it. Allocator-level only: no tensor, kernel or draw changes (a
bitwise row stays bitwise).

The kit's CUDA-graph roll-out (``opt/forward/rf3_xattempt_addon/patched/rf3/diffusion_samplers/inference_sampler.py``) warms the
step up on a side stream, captures it into a private graph pool and deletes the graph after the roll-out. Every item leaves the caching
allocator holding blocks it cannot reuse for the next allocation class: the trunk's blocks (main stream) cannot serve the warm-up (side
stream), the warm-up's cannot serve the trunk of the next item, and the deleted graph's pool stays cached. Reserved device memory then
grows until ``cudaMalloc`` fails and the allocator frees its cache to retry — the peak the device sees is the card, not the model
(reserved memory far above what is allocated). The policy (``POLICY`` = ``capped``):
  * returns the free cached blocks at the two seams the kit itself calls at every roll-out, ``rf3.graph_flags.hoist_begin`` (before the
    warm-up) and ``hoist_end`` (after the graph is deleted) — ``wrap()``;
  * exports ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.5`` at activation for the kit process (the
    shared core's ``opt_core.mem.torch_alloc`` export: refused when torch's CUDA allocator is already initialised or the variable names
    another configuration); the reclaim threshold returns unused cached blocks once reserved memory passes half the card instead of at the
    first failed ``cudaMalloc`` — a budget on the reserved figure, never a cap on what the model may allocate;
  * on a row whose FPF arm carries ``tg``, sets the add-on's trunk-graph cache budget ``FPF_RF3_TG_MAX=1`` (one captured shape at a time,
    the least recently used evicted — a recapture per shape change instead of up to six resident pools; an ``FPF_RF3_TG_MAX`` already in
    the environment stands and is the value recorded).
Every release is counted (``STATE``) and the counts ride the exit tally (report.tally ``mem``); the policy is named on the activation
report and the ``[rosettafold3-opt] MEM`` line.
"""
import os
import sys
from typing import Dict, Optional

from . import _core

_alloc = _core.load("mem.torch_alloc")              # the shared core's allocator-policy primitive (conf strings, the export and its refusals)
_mem = _core.load("mem")

POLICY = "capped"                                 # the one memory policy of the kit rows (named on the MEM line and the activation report)
EXPORTS = ("PYTORCH_CUDA_ALLOC_CONF", "FPF_RF3_TG_MAX")   # every name the policy may export into the kit process
GC_THRESHOLD = 0.5                                # the allocator reclaims its unused cached blocks once reserved memory passes this fraction of the card
TG_BUDGET = "1"                                   # rows whose arm carries `tg`: the add-on's trunk-graph cache holds ONE shape (FPF_RF3_TG_MAX; the add-on's default 6 —
                                                  # each cached shape keeps a private graph pool the size of the trunk's activations)
ENV_TG_MAX = "FPF_RF3_TG_MAX"
ALLOC_CONF = _alloc.conf_for(POLICY, GC_THRESHOLD)   # PYTORCH_CUDA_ALLOC_CONF (the core's words), read by torch at its first CUDA allocation
TRIGGER = "rf3.graph_flags"                       # the kit module whose roll-out seams the policy wraps (imported by rf3.model.RF3_structure)
SEAMS = ("hoist_begin", "hoist_end")
STATE: Dict[str, object] = {"policy": None, "installed": False, "wrapped": False, "releases": 0, "freed_gb": 0.0, "max_reserved_gb": 0.0,
                            "max_allocated_gb": 0.0, "reserved_at_begin_gb": [], "reserved_at_end_gb": []}


class MemPolicyError(RuntimeError):
    """The policy cannot apply in this process (the allocator export refused by name, or the kit module lacks the seams)."""


def line(policy: str = POLICY, fpf_tg_max: str = None) -> str:
    seams = f"seams={','.join(TRIGGER + '.' + s for s in SEAMS)}"
    return f"[rosettafold3-opt] MEM policy={policy} {seams} alloc_conf={ALLOC_CONF}" + (f" fpf_tg_max={fpf_tg_max}" if fpf_tg_max else "")


def _release(where: str) -> None:
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available():
        return
    before = torch.cuda.memory_reserved()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    after = torch.cuda.memory_reserved()
    STATE["releases"] += 1
    STATE["freed_gb"] = round(STATE["freed_gb"] + max(0, before - after) / 1e9, 3)
    STATE["max_reserved_gb"] = round(max(STATE["max_reserved_gb"], torch.cuda.max_memory_reserved() / 1e9), 3)
    STATE["max_allocated_gb"] = round(max(STATE["max_allocated_gb"], torch.cuda.max_memory_allocated() / 1e9), 3)
    STATE[f"reserved_at_{where}_gb"].append(round(before / 1e9, 2))


def wrap(module) -> None:
    """Wrap the two seams of the imported kit module (called once by the import watch); a module without them is not the kit's."""
    missing = [s for s in SEAMS if not callable(getattr(module, s, None))]
    if missing:
        raise MemPolicyError(f"{module.__name__} has no {', '.join(missing)}: not the shipped kit module (the memory policy cannot apply)")
    begin, end = module.hoist_begin, module.hoist_end

    def hoist_begin(*a, **k):
        _release("begin")
        return begin(*a, **k)

    def hoist_end(*a, **k):
        r = end(*a, **k)
        _release("end")
        return r
    hoist_begin.__wrapped__, hoist_end.__wrapped__ = begin, end
    module.hoist_begin, module.hoist_end = hoist_begin, hoist_end
    STATE["wrapped"] = True


def apply(rep: dict) -> dict:
    """Export the allocator configuration (refused by name when the core declines it), set the trunk-graph cache budget on a row whose
    arm carries ``tg``, and record the policy on the activation report; the seams are wrapped by ``wrap`` when the kit module executes
    (stack's watch on ``TRIGGER``)."""
    torch = sys.modules.get("torch")
    inited = bool(torch is not None and getattr(getattr(torch, "cuda", None), "is_initialized", lambda: False)())
    try:                                                     # the core's export with its named refusals (CUDA already initialised; another configuration set)
        _alloc.export(POLICY, lever="mem", gc_threshold=GC_THRESHOLD, cuda_initialized=inited,
                      graphs_on=True, allow_with_graphs=True)          # every kit row runs the sampler CUDA graph: the shipped composition, declared
    except _mem.MemLeverRefused as e:
        raise MemPolicyError(f"memory policy {POLICY}: {e.reason}") from e
    tg = None
    f = rep.get("fpf") or {}
    comps = f.get("components") or ((f.get("arm") or "").split("@")[0].split("+") if f.get("arm") else [])
    if "tg" in comps:                                        # the budget is for the add-on's trunk-graph cache: every mode / row whose arm carries `tg` (exact)
        cur = os.environ.get(ENV_TG_MAX)
        if cur not in (None, ""):
            tg = cur                                  # the environment's own budget stands (the add-on's knob; the activation report notes it) — recorded on the MEM line
        else:
            os.environ[ENV_TG_MAX] = tg = TG_BUDGET
    STATE["policy"] = POLICY
    STATE["installed"] = True
    rep["mem"] = {"policy": POLICY, "trigger": TRIGGER, "seams": list(SEAMS), "applied": "deferred",
                  "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"), "fpf_tg_max": tg}
    return rep["mem"]


def report() -> Optional[dict]:
    """The exit tally's ``mem`` block (None when the policy was not applied in this process)."""
    if STATE["policy"] is None:
        return None
    return dict(STATE)
