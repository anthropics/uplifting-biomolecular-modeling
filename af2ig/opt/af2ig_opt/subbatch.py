"""L9 — the inference sub-batch of the vendored AlphaFold.

Every ``Attention`` module of the model (triangle attention starting / ending node, MSA row attention with pair bias, MSA column
attention, template attention) and every ``Transition`` runs through ``alphafold.model.mapping.inference_subbatch``: the rows of its
input are processed in chunks of ``global_config.subbatch_size`` rows inside an XLA while loop (``config.py`` ``subbatch_size: 4`` — a
value sized for 16 GB cards). At 4 rows the pair track of an N-residue complex is N/4 loop iterations of small GEMMs and elementwise
kernels; at ``ROWS`` rows it is N/ROWS iterations of large ones. The per-row math is unchanged (rows are independent); the compiled
program differs, so the lever's numerics class is stated as Tier-2 (not bitwise), not assumed.

The driver flag is ``-subbatch N`` (``predict_pdb.py``); the value of the kit's line is ``modes.SUBBATCH_ROWS``. The decision and its
record are the model-opt tree's one sub-batch policy, :mod:`opt_core.jax_design.subbatch_policy` (shared with the other JAX kits):
``choose(tokens=<compiled length>, stock_value=STOCK_ROWS, requested=N)`` → a ``SubbatchDecision`` whose ``as_dict()`` is the driver's
``subbatch`` timer record per compiled length (``value``, ``source='requested'``, ``stock_value``, ``tokens``) — the evidence
``stack.applied`` reads for L9. Nothing here holds a device estimate: the kit requests a finite chunk (never unchunked), whose largest
per-chunk intermediate is the attention logits ``[ROWS, 4 heads, N, N]`` f32 = ROWS·16·N² bytes (2.0 GB at N=1000, 4.4 GB at N=1472,
18 GB at N=3000 for ROWS=128) inside XLA's preallocated pool.
"""
from typing import Optional

STOCK_ROWS = 4                     # alphafold/model/config.py global_config.subbatch_size (the vendored stock value)
POLICY = "opt_core.jax_design.subbatch_policy"
_STATE = {"requested": None, "records": []}


def _policy():
    try:
        from opt_core.jax_design import subbatch_policy as SBP   # noqa: WPS433 — the core pin gate ran at package import (af2ig_opt/__init__)
    except ImportError as e:                                     # a pinned core without the policy: named, never a silent stock run
        raise RuntimeError(f"af2ig_opt.subbatch: core_missing:{POLICY} ({e})") from None
    return SBP


def setup(rows: int) -> dict:
    """Validate the requested chunk once, before the model is built: an int >= 1 (0 is not a spelling of stock: omit the flag)."""
    rows = int(rows)
    if rows < 1:
        raise SystemExit(f"predict_pdb.py: -subbatch {rows}: rows per chunk must be >= 1 (omit the flag for the stock {STOCK_ROWS})")
    _policy()
    _STATE["requested"] = rows
    return {"rows": rows, "stock_rows": STOCK_ROWS, "policy": POLICY}


def rows() -> int:
    if _STATE["requested"] is None:
        raise RuntimeError("af2ig_opt.subbatch: setup() was not called (predict_pdb.py calls it when -subbatch is given)")
    return int(_STATE["requested"])


def record(tokens: int) -> dict:
    """The policy's decision for one compiled length, as the fields of the driver's ``subbatch`` timer record."""
    dec = _policy().choose(tokens=int(tokens), stock_value=STOCK_ROWS, requested=rows())
    d = dict(dec.as_dict())
    d["policy"] = POLICY
    _STATE["records"].append(d)
    return d


def records() -> list:
    return list(_STATE["records"])
