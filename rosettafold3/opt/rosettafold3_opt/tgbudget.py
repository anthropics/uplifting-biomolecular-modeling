"""The trunk-graph token budget: the FPF add-on's pairformer-stack CUDA graph (arm component ``tg``: ``Recycler.forward`` replays the 48-block
stack as one ``torch.cuda.CUDAGraph`` per token count) pays where the per-block launch overhead is visible — small inputs — and holds a graph
pool that grows with the pair tensor. This module routes by ONE rule,
decided per call from the pair tensor's token count and logged: ``I <= max_i`` → the add-on's graph forward (capture once, replay); ``I > max_i``
→ the forward ``Recycler`` carried before the add-on installed its graph (the stock recycle loop over the classes' current block forwards: stock
kernels on ``exact``, the arm's kernels on a ``fast_tg`` row) — a NAMED skip (``tg:budget``), counted, never a fallback: both paths are the
row's declared numerics (on stock kernels both are bitwise to stock under the deterministic recipe).

The budget is ``MAX_I`` tokens on stock trunk kernels and ``MAX_I_FAST`` on the FPF-fast kernels (``budget_for``: the adapter's own trimul mode names
the arm's kernel class): the fused kernels' eager trunk is launch-bound only at the smallest inputs, so their graph pays over a shorter range.
Installed by ``stack.fpf_apply`` after ``apply_arm`` for every mode / row whose arm carries ``tg``; the EXIT tally's ``fpf_tg`` LEVER line carries
``budget_max_i=<n> graphed=<n> skipped=<n>``.
"""
from collections import Counter

MAX_I = 1000                                          # stock-kernel arms (exact's tg+sapb): above this the graph pool's growth with the pair tensor outweighs the launch overhead the graph saves
MAX_I_FAST = 300                                      # fast-kernel arms: the fused kernels' eager trunk is launch-bound only at the smallest inputs, so their graph pays over a shorter range
BUDGETS = {"stock": MAX_I, "fast": MAX_I_FAST}         # by the adapter's trimul mode (fpf_rf3_adapter.describe()["mode"])
CARD_BUDGETS = {(8, 0): {"fast": 0}}                   # compute capability 8.0, fast-kernel arms: no trunk graph at any size — the fast arm's pair-bias
                                                      # attention row on sm_80 (SDPA, cuDNN) allocates workspace inside the capture and the capture is
                                                      # invalidated (cudaErrorStreamCaptureInvalidated / CUDNN_STATUS_INTERNAL_ERROR_DEVICE_ALLOCATION_FAILED
                                                      # on the pinned stack); every trunk call takes the pre-graph forward BY
                                                      # NAME (census skipped:<n>, EXIT levers_budget_skipped=fpf_tg(max_i=0,...)). The stock-kernel arm
                                                      # (exact's tg+sapb) captures on 8.0 as on 9.0: its budget is MAX_I on both cards.
STATE = {"on": False, "max_i": None, "reason": None, "card": None}
CENSUS = Counter()


class TgBudgetRefused(RuntimeError):
    """The budget cannot be installed (the add-on's graph forward is absent)."""


def compute_capability():
    """``(major, minor)`` of the running GPU, or None (no torch / no GPU: the dry run)."""
    try:
        import torch
        if torch.cuda.is_available():
            return tuple(int(x) for x in torch.cuda.get_device_capability())
    except Exception:
        pass
    return None


def budget_for(adapter, cc="probe") -> int:
    """The arm's budget: ``MAX_I_FAST`` when the adapter serves the FPF-fast trimul (a fast-kernel arm), ``MAX_I`` otherwise (stock kernels);
    a card row (``CARD_BUDGETS[cc][mode]``) takes precedence on that compute capability (``cc``: probed from the running GPU by default)."""
    try:
        mode = (adapter.describe() or {}).get("mode")
    except Exception:
        mode = None
    cc = compute_capability() if cc == "probe" else (tuple(cc) if cc else None)
    card = CARD_BUDGETS.get(cc) or {}
    if mode in card:
        STATE["card"] = "cc%d.%d:%s:max_i=%d" % (cc[0], cc[1], mode, card[mode])
        return int(card[mode])
    STATE["card"] = None
    return BUDGETS.get(mode, MAX_I)


def enable(adapter, max_i: int = None) -> dict:
    """Wrap ``rf3.model.RF3_structure.Recycler.forward`` (the add-on's graph forward, installed by ``apply_arm``) with the budget rule.
    ``adapter`` is the imported ``fpf_rf3_adapter`` module: its ``_ORIG['rec_forward']`` is the Recycler forward it replaced; ``max_i``
    defaults to the arm's budget (``budget_for``)."""
    import rf3.model.RF3_structure as RS
    max_i = budget_for(adapter) if max_i is None else int(max_i)
    graph_forward = RS.Recycler.forward
    eager_forward = (getattr(adapter, "_ORIG", {}) or {}).get("rec_forward")
    if eager_forward is None or graph_forward is eager_forward:
        raise TgBudgetRefused("the add-on's trunk graph is not installed on Recycler.forward (arm without tg, or apply_arm did not run)")
    def forward(self, f, S_inputs_I, S_init_I, Z_init_II, S_I, Z_II):
        n = int(Z_II.shape[-2])
        if n > max_i:
            CENSUS[f"skipped:{n}"] += 1
            return eager_forward(self, f, S_inputs_I, S_init_I, Z_init_II, S_I, Z_II)
        CENSUS[f"graphed:{n}"] += 1
        return graph_forward(self, f, S_inputs_I, S_init_I, Z_init_II, S_I, Z_II)

    forward.__name__ = "recycler_forward_tg_budget"
    forward.__wrapped_graph__ = graph_forward
    RS.Recycler.forward = forward
    STATE.update({"on": True, "max_i": max_i, "reason": None})
    return describe()


def census() -> dict:
    c = dict(CENSUS)
    return {"graphed": sum(v for k, v in c.items() if k.startswith("graphed:")), "skipped": sum(v for k, v in c.items() if k.startswith("skipped:")), "by_key": c}


def describe() -> dict:
    return {"on": STATE["on"], "max_i": STATE["max_i"], "card": STATE.get("card"), "rule": "tg:budget (I > max_i -> the pre-graph Recycler.forward, named skip)", "census": census() if STATE["on"] else None,
            "reason": STATE["reason"]}


def lever_evidence() -> list:
    c = census()
    return [("budget_max_i", STATE["max_i"]), ("graphed", c["graphed"]), ("skipped", c["skipped"])]
