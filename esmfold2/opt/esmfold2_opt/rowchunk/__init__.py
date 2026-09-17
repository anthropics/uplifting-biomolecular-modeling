"""esmfold2_opt.rowchunk -- row-chunking levers of the row-sharded (``--n_gpu > 1``) pair stack.

These modules bound the memory of statements that the sharded pair stack issues on WHOLE row shards.
Sharding splits the pair-shaped objects ACROSS ranks; these levers bound them WITHIN a rank by issuing
the same statements per row block of the local shard: the recycle inject (``rowpair._inject_rows``), the
confidence head's pairwise prologue and heads (``confrows``), the pair state handed to the confidence
head (``zbf16rows``), and the sampler's pair-bias buffers released once the roll-out returns
(``confmem``). ``tn_shim`` restores the truncated-normal form the sharded pair init requires on a torch
that lacks it (inert on the pinned torch).

NUMERICS.  Not a bit-exact transform: an op reissued per row block changes GEMM extents and reduction
order -- the same statement ``rowpair.py`` makes for its own sharded line -- and the bf16 members
(``confbf16``, ``zbf16``) demote a pair-sized activation the stock statements hold in fp32.

Installed from ``rowpair.install_rank`` after the pair stack is sharded; nothing here is reachable at
``--n_gpu 1``.
"""
