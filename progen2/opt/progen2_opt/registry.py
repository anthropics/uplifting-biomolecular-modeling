"""The registry of optimizations: one row per optimization the kits carry — its numerics class, its tier, the switch that turns
it on or off (the kit's own), the route it acts on and the code that implements it (a path in the tree + the symbol).
Nothing here is a measurement.

Classes: ``exact`` = bit-for-bit the stock's outputs at the kit's own exact shapes (its shape table). Tiers: ``1`` = exact.
Switch spellings are the kits' own: the generation kit's lever table (`progen2_decode.py` ``AUTO_LEVERS``) and its components.json decode
component; the scoring kit's ``apply(model, ew=True, size=...)`` keywords and the ``Scorer`` constructor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Optimization:
    name: str
    kit: str                 # "serving" | "scoring" | "ew"
    route: str               # "sample" | "score"
    klass: str               # numerics class
    tier: int
    switch: str              # the kit's own switch spelling
    code: str                # tree path (+ symbol)
    what: str


OPTIMIZATIONS: Dict[str, Optimization] = {
    # ------------------------------------------------------------------------------------------------ the generation kit (sample.py's route, in-process)
    "oneread_mmap": Optimization("oneread_mmap", "serving", "sample", "exact", 1, "progen2_decode.py AUTO_LEVERS (always on, CUDA route)",
                          "opt/serving/pipeline_v0_4/oneread_loader.py create_model_oneread; progen2_decode.py load",
                          "the weights read once (torch.load mmap) into the meta-initialised stock model, the same tensors and dtypes create_model(ckpt, fp16) installs (the fp32 route included)"),
    "sampler_exact": Optimization("sampler_exact", "serving", "sample", "exact", 1, "progen2_decode.py AUTO_LEVERS (always on, CUDA route)",
                           "opt/serving/pipeline_v0_4/sampler_exact.py generate_exact; progen2_decode.py _install_sampler",
                           "model.generate shadowed for the ONE stock sample() call of a unit: the stock warpers, softmax and multinomial on the stock philox stream with the integer bookkeeping fused (tokens bit-identical)"),
    "resident_rotary": Optimization("resident_rotary", "serving", "sample", "exact", 1, "components.json decode_kit (kit_t1 level t3s, installed by progen2_decode.py load)",
                           "opt/serving/pipeline_v0_4/components/plm_transfer_t1_v0/kit_t1.py fixed_pos_embedding_cached / apply_rotary_pos_emb_cached",
                           "the stock rotary function's own sin/cos tables computed once on the device for the slots' length and sliced per decode step (the stock recomputes them on the CPU per layer per step)"),
    "static_kv": Optimization("static_kv", "serving", "sample", "exact", 1, "components.json decode_kit (kit_t1 level t3s, installed by progen2_decode.py load)",
                           "opt/serving/pipeline_v0_4/components/plm_transfer_t1_v0/kit_t1.py attn_forward_static / allocate / release; progen2_decode.py Handle.hold / ensure_slots",
                           "the per-step torch.cat K/V cache replaced by static slot buffers per (layer, batch size) in the stock's own key/value dtypes, written at slot t and viewed over [0, t]; held per batch size for the work in front of the model (allocated when a job names its batch sizes or a unit first names one, after a fit check; a batch size that cannot fit is a named out-of-memory, no fallback)"),
    # ------------------------------------------------------------------------------------------------ the scoring kit (likelihood.py's route)
    "rotary_tables": Optimization("rotary_tables", "scoring", "score", "exact", 1, "v0_score_r3_1.apply(model, ...) (always)",
                           "opt/forward/engines/progen2/kits/v0_score_r3_1/__init__.py apply / _fixed_pos_embedding_resident",
                           "the stock fixed_pos_embedding evaluated once per (device, dim) and sliced per call (a module-global patch of the stock module); rotary-table check n_positions/n_positions"),
    "one_forward_per_direction": Optimization("one_forward_per_direction", "scoring", "score", "exact", 1, "Scorer.score_T(units)",
                                       "opt/forward/engines/progen2/kits/v0_score_r3_1/__init__.py Scorer / post_rows",
                                       "sum and mean of a direction from one forward (the stock CLI runs four); both reductions read from the same logits"),
    "host_pipeline": Optimization("host_pipeline", "scoring", "score", "exact", 1, "Scorer(..., n_sets=4)",
                           "opt/forward/engines/progen2/kits/v0_score_r3_1/__init__.py Scorer.run_batches",
                           "tokenisation, pinned H2D/D2H and the row assembly on a host thread overlapping the GPU"),
    # ------------------------------------------------------------------------------------------------ module B: v0_ew, composed first
    "ew:gelu": Optimization("ew:gelu", "ew", "score", "exact", 1, "v0_score_r3_1.apply(model, ew=True, size=...) (v0_ew composed first, its exact table)",
                     "opt/forward/engines/progen2/kits/v0_ew/patches.py; ext.py; libew_progen2.so", "the stock MLP activation as one fused kernel (bit-for-bit the stock's, the kit's own check)"),
    "ew:rotary": Optimization("ew:rotary", "ew", "score", "exact", 1, "same", "opt/forward/engines/progen2/kits/v0_ew/patches.py; libew_progen2.so", "the rotary application as one fused kernel"),
    "ew:residual": Optimization("ew:residual", "ew", "score", "exact", 1, "same", "opt/forward/engines/progen2/kits/v0_ew/patches.py; libew_progen2.so", "the residual adds fused"),
    "ew:glue": Optimization("ew:glue", "ew", "score", "exact", 1, "same", "opt/forward/engines/progen2/kits/v0_ew/patches.py; libew_progen2.so", "the attention glue (reshape/cast copies) fused"),
    "ew:ln": Optimization("ew:ln", "ew", "score", "exact", 1, "same (ln_variant = the kit's default)", "opt/forward/engines/progen2/kits/v0_ew/patches.py; libew_progen2.so", "the layer norm as one kernel"),
}
