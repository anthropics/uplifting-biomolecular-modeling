"""opt_core.mem.rowpair_jax — the row-sharded pair stack of a JAX engine over P local devices (the ``--n_gpu P`` resource axis of a kit's
``big`` mode): ONE 1-D device mesh, the pair representation ``[N, N, C]`` sharded over its ROW axis, the pair sub-layers run per device on
the local row block ``[N/P, N, C]`` inside ``shard_map`` regions with an ENUMERABLE set of collectives, the heads' pair consumers served from
the local rows. Engine-free: which stock class is rebound, the mode table, the token buckets and every threshold are the kit's adapter
(``<pkg>.big``); the primitives here are the mesh, the layouts, the collectives, the two cross-row contractions, the evidence and the Haiku
frame recipe. Submodules are imported by name (``from opt_core.mem.rowpair_jax import mesh, shard, trimul``); every module is standard library
at import and imports jax / haiku inside the functions that need them (``_lazy``), so a kit on another framework never pays for them and a
missing jax is a refusal naming the lever (:class:`opt_core.mem.MemLeverRefused`), never an ImportError.

    mesh        :func:`mesh.build` — the mesh over the FIRST ``n_gpu`` visible devices (explicit P; ``refused: n_gpu=P visible=K`` when fewer are
                visible; never auto-sized — the token text and refusal sentences of the axis are :mod:`opt_core.mem.ngpu`'s — one
                producer), the device facts (kind, bytes_limit — read from the runtime, no card constant), the mode gate
                (``--n_gpu>1`` outside the kit's TP modes is refused by name), the XLA client-memory environment as found / its NCCL headroom gate
    shard       partition specs (rows ``P(axis, None, None)`` / cols / replicated), ``with_sharding_constraint`` at the pair sites, the ``shard_map``
                compatibility call (jax 0.5 … 0.10), the row-extent gate (``N % P``), and the collectives of the plan: ``gather`` (all_gather of a
                row block), ``rows_to_cols`` / ``cols_to_rows`` (all_to_all re-layout), ``transpose_block`` (the block of ``x^T`` for my rows),
                ``local_block`` (my slice of a replicated operand), ``psum``
    trimul      :func:`trimul.contract` — triangle multiplication's cross-row contraction for the four stock equations (outgoing ``ikc,jkc->ijc`` /
                ``cik,cjk->cij``: all_gather of the partner operand; incoming ``kjc,kic->ijc`` / ``ckj,cki->cij``: all_gather + all_to_all so the
                k-contraction is COMPLETE per device — no partial sums), schedules ``gather`` (one full operand transient) and ``ring`` (P
                permute steps, transient one block; the same numerics class)
    triatt      triangle attention plumbing: starting node = row-local given the gathered bias (:func:`triatt.bias_full`); ending node = the
                transposed tensor's row block via one all_to_all in and one out (:func:`triatt.enter_transposed` / :func:`triatt.exit_transposed`);
                mask blocks; the per-shard kernel gate (a Pallas kernel runs per shard when its block constraints hold on the local extent, else the
                kit prints the jnp path as a NAMED lever state)
    transition  the row-local sub-layers and the pair CONSUMERS: pair transition / outer-product-mean operands sliced to my rows / the MSA and
                single-attention pair-bias gather / the symmetrisation of half-logits across devices / the recycle-carry constraint
    evidence    the activation-evidence fields (``n_gpu=P sharding=rowpair``), per-device peak memory (``jax`` device ``memory_stats``), the XLA
                memory environment as found, the per-site plan table (locality, collectives, numerics class), the card/arch declaration, and the
                ONE ``LEVER`` line through :func:`opt_core.report.lever_line`
    <recipe>    the model-library recipe (the module(s) of this package named for the Haiku model library they bind): ``install(rmesh, ...)``
                rebinds the stock trunk / template / head bodies on row blocks over these primitives — the Evoformer-class blocks as
                regions, the pair sub-layers on the local row block, the heads that read the pair served from local rows — pinned to the
                stock tree it transcribes (refused by name on other bytes); the only place in this package that names model classes
    rowchunk    the single-device ROW-CHUNK of the stock ``TriangleMultiplication`` (the P=1 ``trimul_chunk`` lever; one producer for the
                fused and unfused forms): :func:`rowchunk.chunked_class`
    haiku       the Haiku recipe: re-binding an ``hk.Module`` method WITH its name scope (``wrap_method``), the rng-less temporary frame a
                ``shard_map`` body runs under (``hk.scan`` inside would otherwise leak the rng tracer), the region depth (a patched body dispatches
                to the stock body outside a sharded region), ``jit`` of the transformed apply with replicated in/out shardings, ``device_put`` to the mesh

Contract (every submodule):
  * EXPLICIT P. ``n_gpu`` is an integer the user typed (absent = 1 in the kit's CLI; an explicit ``--n_gpu 1`` is the same run); P=1 installs
    NOTHING — structurally: :func:`mesh.build` refuses ``n_gpu=1`` by name, so the kit's single-device program runs byte for byte as tested and the
    adapter prints ``LEVER name=rowpair state=off reason=n_gpu=1`` (the XLA memory environment at P=1 is the tested kit's, untouched; the line records
    it as found). Fewer than P visible devices of the platform is ``refused: n_gpu=P visible=K``; a wrong platform (jax fell back to CPU) is refused by
    name; every size has a shard plan (:func:`shard.pad_plan`: N is padded to a multiple of P — a ladder bin is never refused); ``--n_gpu>1`` under a
    non-TP mode (``exact``, ``fast``) is refused by name — row-sharded contractions cannot be bit-exact to the single-device program.
  * FAIL-LOUD. A primitive acts or raises :class:`opt_core.mem.MemLeverRefused` (lever ``rowpair`` unless the adapter names another). No silent
    single-device fallback: a per-shard kernel that cannot run on the local extent is a NAMED state on the lever line (``kernel=jnp reason=…``).
  * NUMERICS CLASSES, named per primitive (:mod:`evidence` ``CLASS_*``): ``moves_bytes`` (collectives, constraints, re-layouts, padding: values
    unchanged — the ONLY class asserted bit-exact), ``row_local`` (the same per-element arithmetic on a row subset; library code generation depends on
    the row extent → stated tolerance, bit-exact reported per stack), ``complete_contraction`` (trimul: every output element's reduction has the dense
    length and operands; the GEMM shape differs from the dense call → stated tolerance, tier-2 by construction), ``reordered`` (cross-device ``psum``
    means: the sum order changes → tolerance). Stated tolerances of the unit tests (relative to max|ref|): fp32 1e-5, bf16 2^-7; reordered 1e-5. A
    kit's P>1 result is a BAND vs its P=1 run, never bit-exact.
  * EVIDENCE. One ``LEVER name=rowpair state=on impl=opt_core.mem.rowpair_jax@<version> origin=core n_gpu=P sharding=rowpair axis=… visible=…
    devices=… schedule=… kernel=… peak_bytes=d0:…,d1:… prealloc=… mem_fraction=…`` line per process (:func:`evidence.line`); the kit's ACTIVE / EXIT
    lines carry ``n_gpu=P sharding=rowpair`` (:func:`evidence.active_fields`).
  * SINGLE PROCESS, MULTI-DEVICE: jax drives P local devices from one interpreter — the kit launches no workers and the user never types a launcher;
    ``CUDA_VISIBLE_DEVICES`` (the kit's) bounds what is visible.
  * CARD-GENERIC: device kind, count and memory are read from the runtime (:func:`mesh.device_facts`); nothing here assumes a card or a size.
  * Standard library at import; Python 3.8 syntax; no engine name; jax ≥ 0.4.30-class ``shard_map`` (``jax.shard_map`` or
    ``jax.experimental.shard_map``) — an older jax is refused by name.
"""
from __future__ import annotations

LEVER = "rowpair"                     # the default lever name on refusals and evidence (the adapter may pass its own registry name)
AXIS = "row"                          # the default mesh axis name
SHARDING = "rowpair"                  # the word of the ACTIVE / EXIT / LEVER lines: sharding=rowpair

__all__ = ["LEVER", "AXIS", "SHARDING"]
