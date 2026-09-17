"""Bindings of Protenix-v2 modules to the shared core's row-sharded pair statements (``opt_core.mem.rowpair``) for the multi-GPU
line of ``big`` (``tp.py``). A binding packs one Protenix module's per-element statements (LayerNorms, projections, gates, attention) into
the core's callables (``TriMulFns`` / ``TriAttFns`` / ``PairBlockFns`` / ``DiTBlockFns`` / the ``*_rows`` statement functions) and calls the
core's ONE driver for that mechanism; it owns no schedule, no communication and no block arithmetic. A seam of the carried unit
(``opt/forward/PTX_TP/PTX_TP_ADDON/ptx_tp``) runs through a binding when ``tp_route.ROUTES`` names it (the ACTIVE line's ``routed=`` list is
the statement of what the core serves in a run; every other seam is the carried unit's own module-bound statement, ``impl=ptx_tp``).

Modules: ``pairstack`` (triangle multiplication out / in, triangle attention start / end, pair transition, attention-pair-bias: the block driver of the
trunk Pairformer, the MSA module's pair stack, the template pair stack and the confidence pair stack), ``diffusion``, ``trunk``, ``template``,
``relpos`` (the row producer they share) and the rank-side helpers ``census``, ``launch``, ``layout_guard``.
"""
