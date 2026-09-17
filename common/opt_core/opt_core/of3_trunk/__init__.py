"""opt_core.of3_trunk — trunk-side levers of the AF3-architecture engines of this tree (the OF3 code family: the 0.4.x release and its 0.5.x
fork share one module layout), one implementation each; the engines' kits bind them through thin adapters (`<kit>/cells/<lever>.py`: switch
name, log prefix, the engine's module paths, registry row, modes line) with `configure(...)` and re-export `STATE / requested / install /
census_line`.

    castcache    exact class: the engines' `Linear` / `LayerNorm` primitives cast fp32 weights to bf16 on every bf16 call; the copies
                 memoised per module (key: the fp32 tensors' storage + version) and handed to the same F.linear / F.layer_norm call
    apb_hoist    exact class: `AttentionPairBias._prep_bias`' key-mask bias `inf * (mask - 1)` memoised per stack call (constant over the
                 pairformer blocks of a recycle pass) instead of rebuilt per block
    trunk_graph  exact class: `PairFormerStack.forward` captured into a CUDA graph per (instance, input signature, flags, numerics mode) at
                 its second call and replayed after; first call eager; token gate, grad, DS4Sci attention, a failed capture refused by name
    tuner_guard  exact class: the engine's `ChunkSizeTuner._compare_arg_caches` answers 'changed' for argument records of different
                 structure (the confidence stack's batched vs per-sample call forms) instead of raising: one process serves items of any
                 mix of token counts
    templ_embed  fast class: the template embedder (TemplateEmbedderAllAtom.forward) on the carried kernels `templ_embed` — the eight-Linear
                 template feature embedding as ONE kernel writing the template pair stack's input, the stack untouched (whatever the line
                 serves it with), the mean over templates / relu / linear_t as ONE kernel; refusals by name run the stock forward, counted

One engine per process: `configure()` rebinds module globals, so a process binds one kit's names. Torch and the engine's modules are
imported inside `install()` / at call time; this module imports nothing (PEP 562).
"""
from __future__ import annotations

_MODULES = ("templ_embed", "castcache", "apb_hoist", "trunk_graph")


def __getattr__(name):
    if name in _MODULES:
        import importlib
        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
