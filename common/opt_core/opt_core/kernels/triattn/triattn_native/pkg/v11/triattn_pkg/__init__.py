"""triattn_pkg -- a triangle-attention forward for H100 (cc 9.0) and A100 (cc 8.0), package generation 11: the routed entry `best`
(dispatch/candidate.py) with every routed kernel directory at the commit that sealed it (PINS).  Generation 11 = generation 10 + the sm_80 member
(`cuda_80`); the cc 9.0 routes, kernels, binaries and test vectors are generation 10's, unchanged.

    out = triattn_pkg.triangle_attention(q, k, v, bias, mask=None, scale=None)

    q, k, v   [B, N, H, S, D] or [N, H, S, D], bf16 or fp16, CUDA; contiguous or strided views with stride(-1) == 1 (an "ending node" call is the
              same op on transposed views -- no copies are made); any N (pair rows), incl. N != S
    bias      [B, 1, H, S_q, S_kv] (or 4-D without B), fp32 or the 16-bit dtype of q -- ONE pair bias shared by all N rows
    mask      [B, N, 1, 1, S_kv] bool, True = attend (a key mask per pair row; padding masks and per-row masks), or None
    scale     softmax scale, default D ** -0.5
    returns   [B, N, H, S_q, D] (or 4-D) in q's dtype:  softmax_k(scale * q.k + bias (+ -inf where mask == 0)) @ v;
              a pair row whose keys are all masked yields the uniform average of v

Anything no sealed kernel serves -- fp32 q/k/v, head dims outside {16, 32, 64, 128}, per-row biases, per-query masks, non-CUDA tensors, a missing
binary with no CUDA toolkit to build it -- raises `triattn_pkg.Unsupported` (a NotImplementedError naming the reason) BEFORE any work; nothing is
substituted silently.  `triattn_pkg.route(q, k, v, bias, mask)` names the kernel a call takes.

Routes (`dispatch/candidate.py` docstring = the full table; CELLS.json = the measured cells, keyed by cc):
    cc 9.0, bf16, D = 32, S_q == S_kv:  S < 512 contiguous -> k13 (Gluon, Triton >= 3.6) | S < 512 strided -> cuda (wgmma/TMA) |
                                        512 <= S <= 3072 and S > 4096 -> cuda_b | 3072 < S <= 4096 -> cuda_c
    cc 9.0, fp16 or D = 16 (S_q == S_kv): S <= 640 -> k13, above -> tri (k12 Gluon);   D in {64, 128} or S_q != S_kv -> tri (k10 Triton)
    cc 8.0, bf16, D in {16, 32, 64}, S_q == S_kv, contiguous or strided: every S -> cuda_80 (mma.sync / ldmatrix / cp.async);
            fp16, D = 128, S_q != S_kv -> tri (k10; unmeasured there)
    other devices: tri for every servable shape (unmeasured).
The CUDA routes are torch C++ extensions specific to (torch version, CUDA major, CPython): `prebuilt/<torch version>-<SOABI>/` holds them for the
stacks listed in prebuilt/INDEX.json; on another stack they are JIT-built on first use when nvcc + ninja (+ CUTLASS headers at $CUTLASS_PATH for
cuda / cuda_b) are present, else the call raises Unsupported naming the missing binary.  TRIATTN_PKG_PREBUILT=never forces the JIT build, =always
refuses to JIT.  cuda / cuda_c / cuda_b and the Gluon kernels (k13, k12) are sm_90a code (wgmma / TMA / setmaxnreg); cuda_80 is sm_80 code and is
routed on cc 8.0 only.
"""
from ._face import BEST_COMMIT, CUDA_SEATS, PINS, prebuilt_status, route, stack_tag, triangle_attention  # noqa: F401
from ._face import __getattr__  # noqa: F401  (`triattn_pkg.Unsupported`: the router's typed refusal, resolved with the router on first access -- importing it loads torch + triton)

__version__ = "11"
__all__ = ["triangle_attention", "route", "Unsupported", "prebuilt_status", "stack_tag", "PINS", "BEST_COMMIT", "__version__"]
