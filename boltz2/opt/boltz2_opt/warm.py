"""``warm``: four public-input predictions in ONE worker process on the mode's worker route, so that every Triton kernel class the mode
routes (the fused triangle multiplication, the flash triangle attention and pair block, the fused transitions and MSA-module kernels —
whichever the mode routes) compiles on this machine and lands in the Triton cache (``TRITON_CACHE_DIR``, configs/*.env); later ``pred``
calls on the machine start from that cache. The sampler's CUDA-graph capture is per worker process and per input shape: every ``pred``
captures its own (a fraction of a second), ``warm`` does not pre-pay it.

Inputs: ``WARM_INPUTS`` under inputs/ (single-sequence, one seed, default 0), run in ascending token count: 398 and 480 tokens (the two
integer-specialization classes, N != 0 and N == 0 (mod 16), inside the K2B window 300-511), 576 (the >= 512 class; on H100 the CUDA
triangle-attention row of ``fast`` / ``big``, a prebuilt extension with nothing to compile, serves from 512 tokens) and 1056 (the
>= 1,024 class: the 1,024+-row cell configurations and the exact tier's residual-fused LayerNorm cubins, which engage from 725 tokens).
Triton specializes a kernel's plain integer arguments by class (divisible by 16 or not) and the routed cells take the token count as such
an argument, so these inputs meet both classes in each kernel window and a later ``pred`` of either class starts from the cache (a first
input of an uncompiled class costs seconds of JIT otherwise). They run through ``pred`` (worker.run, tag ``warm``): the outputs
(``<out>/by_seed/<input stem>/s<seed>/``), the lines and the exit rule are pred's; the run record carries ``"warm": true``. The JIT cache
is a pure function of the stack's pins and the GPU architecture, keyed under the kit's cache root (configs/*.env): a container without
that keyed volume pays the per-class compile once per mode; an image build can seed it by running ``warm`` for each mode.
"""
from __future__ import annotations

import os

from . import stack, worker

WARM_INPUT = os.path.join("inputs", "1BRS_x2_barnase_barstar.yaml")     # relative to the boltz2/ tree (stack.tree_dir()): 398 tokens, N != 0 (mod 16)
WARM_INPUT_2 = os.path.join("inputs", "1BRS_a2b4_barnase_barstar.yaml") # 576 tokens = 36 x 16, N == 0 (mod 16): the token count's other specialization class
WARM_INPUT_3 = os.path.join("inputs", "1BRS_a4t2_barnase_trpcage.yaml")   # 480 tokens = 30 x 16: N == 0 (mod 16) INSIDE the K2B window (300-511)
WARM_INPUT_4 = os.path.join("inputs", "1BRS_a6b4t2_barnase_barstar_trpcage.yaml")   # 1056 tokens = 66 x 16: the >= 1,024-token class (1,024+-row cell configurations, the exact tier's residual-fused LayerNorm cubins from 725 tokens)
WARM_INPUTS = (WARM_INPUT, WARM_INPUT_3, WARM_INPUT_2, WARM_INPUT_4)     # all run in ONE worker process, in this order (ascending token count)


def warm_input(rel: str = WARM_INPUT) -> str:
    p = os.path.join(stack.tree_dir(), rel)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"the tree's public warm-up input is missing: {p}")
    return p


def warm_inputs() -> list:
    """The warm-up inputs, in run order: one per token-count specialization class and kernel window (398 / 480 / 576 / 1056 tokens)."""
    return [warm_input(rel) for rel in WARM_INPUTS]


def run(mode: str, out_dir: str, seed: int = 0, allow_partial: bool = False, n_gpu=1) -> int:
    return worker.run(mode, warm_inputs(), out_dir, [seed], tag="warm", extra_record={"warm": True}, allow_partial=allow_partial, n_gpu=n_gpu)
