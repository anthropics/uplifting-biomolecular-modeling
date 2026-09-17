"""evo2_opt — the Evo 2 inference kit as a drop-in for the ``evo2`` package (evo2 0.6.0 on vtx 1.1.0).

    export EVO2_OPT=exact          # before python starts; or, first thing in the process:  import evo2_opt; evo2_opt.enable()
    from evo2 import Evo2
    model = Evo2("evo2_7b")        # any construction: the kit reads the route from the object (use_kernels on / off, TE present / absent)
    model.score_sequences([...]); model(ids); model.generate(...)   # the stock's outputs bit for bit, faster

Modes: ``exact`` — every scoring forward (``score_sequences``, ``model(ids)``) takes the kit's levers and returns the logits the stock
returns on this stack, bit for bit; ``generate`` runs the stock decode loop with the generation members (a fused Hyena state update and
CUDA-graph replay of the one-token step: the same kernels, the same sampled tokens). ``fast`` — the same scoring, and ``generate`` becomes
speculative sampling with the ``evo2_1b_base`` draft (every token an exact sample from the target's transformed distribution; the sequence
for a seed is not the stock sampler's). ``EVO2_OPT=off`` (or unset) is the stock: nothing of the kit is imported (this module stays
import-light for that reason: the names below load on first use).

Lines (stderr): ``[evo2-opt] ACTIVE mode=<mode> …`` once per process when the switch engages, the environment it read named on it (package
versions against ``stock/PINS.json``, Transformer Engine present or absent, the GPU listed in it or not — an unlisted GPU or a library version off its pin
is named, never refused); ``[evo2-opt] NOT ACTIVE: <reason>`` when it cannot run here (no CUDA device; ``evo2`` / ``vtx`` not the pinned
stock) — ``enable()`` raises ``Evo2OptRefused``, ``EVO2_OPT=<mode>`` exits 3; ``[evo2-kit] APPLIED …`` per constructed model (route, levers);
``[evo2-opt] GENERATION …`` (the members armed); ``[evo2-kit] EXIT …`` at interpreter exit (forwards on the kit path / on the stock path by
reason). A model the levers cannot serve as constructed raises ``Evo2OptRefused`` from its constructor naming the condition: a mode is all
of its levers or refuses, never a subset under its name.
"""
__version__ = "0.5.1"
__all__ = ["enable", "status", "Evo2OptRefused", "__version__"]


def __getattr__(name):
    if name in ("enable", "status", "Evo2OptRefused", "check"):
        from evo2_opt import activation
        return getattr(activation, name)
    raise AttributeError(name)
