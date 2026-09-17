fpf_triattn_cuda — sm_90a CUDA triangle-attention forward for the Protenix v2 kit (lever `triattn_cuda`).

PROVENANCE. csrc/triattn_mw.cu and csrc/mw_ptx.h: written in this repository by the fpf_triattn_cuda authors (kernel lineage: the
repository's sm_90 triangle-attention kernel, version 20, in this repository's history), then adapted for this kit (S cap removed —
per-row mask tables in global memory —, kit-shaped Python op, prebuilt loader with a load-time bitwise check). __init__.py, build_prebuilt.py:
written by the same authors for this kit. prebuilt/<stack key>/*: built by build_prebuilt.py from exactly the csrc/ files whose sha256 the
manifest records (compiled from a neutral staging copy: no build-machine path in the binary).

NOTICE. No third-party source is included: the PTX wrappers in mw_ptx.h (mbarrier, cp.async.bulk.tensor, ldmatrix, mma.sync, ex2.approx) are
original; there is no CUTLASS / CuTe / flash-attention code and no vendor kernel binary. The op follows the cuEquivariance `triangle_attention`
calling convention so it can stand in for it; cuEquivariance is neither copied nor linked.
Licence: the package licence (common/opt_core/LICENSE); third-party components of the package and their licence texts are listed in
common/opt_core/THIRD_PARTY_NOTICES.md — this directory carries none.
