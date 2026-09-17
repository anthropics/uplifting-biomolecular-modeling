"""The levers of the pipeline_tf v17 kit as the modes compose them — what each one does, its class and tier, its switch and the code
it touches.

Descriptive only: the kit's own entry script constructs and applies every lever (``opt/datapath/pipeline_tf/v17/borzoi_sad.py``, the
lines cited are that carried file's); this table is what ``check`` prints. No lever value is
transcribed here — switches are named, their values live in the kit. Every lever below is part of the one kit mode's composition
(``exact``: ``kit.KIT_COMPOSITION`` sets the kit's forward switch; every other lever is always on in the kit entry).

Classes: ``datapath`` (the input and host-post terms around the forward), ``forward`` (the model call's host side), ``observability``
(the stamp, which changes no byte). Tiers are the kit's own words (``kitlib/writer.py``, the module docstrings): ``exact`` (the kit states
the stock's bytes by construction), ``none`` (no numerics).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

KIT = "opt/datapath/pipeline_tf/v17"


@dataclass(frozen=True)
class Lever:
    name: str
    cls: str                          # datapath | forward | observability
    tier: str                         # exact | none
    file: str                         # the kit file that implements it (relative to KIT)
    lines: str                        # file:line anchors in the carried bytes (entry script + module)
    what: str
    switch: str                       # the kit's own switch, or "none (always on in the kit entry)"
    modes: Tuple[str, ...]            # the package modes in which this lever is part of the composition
    default_in_kit: str = ""          # what the kit does when its switch is unset
    notes: Tuple[str, ...] = field(default_factory=tuple)


LEVERS: Dict[str, Lever] = {
    "onehot_lut": Lever(
        "onehot_lut", "datapath", "exact", "kitlib/onehot.py", "borzoi_sad.py:49-50 (install at import); kitlib/onehot.py:30-68",
        "the one-hot of each 524,288-bp window by a (256, 4) lookup table bound in place of baskerville.dna.dna_1hot (the values 0, 1, 0.25 "
        "are exact in float16; the row per byte is the stock's) — paths the table cannot reproduce verbatim (n_sample, seq_len trim/pad with "
        "n_uniform=False) are delegated to the stock function",
        "none (always on in the kit entry)", ("exact",)),
    "cores_probe": Lever(
        "cores_probe", "datapath", "none", "kitlib/sad_post.py", "borzoi_sad.py:17-22 (started before the TensorFlow import), :278 (joined at the post); kitlib/sad_post.py:186-250",
        "sizes the host post's thread pool from the machine's deliverable cores: a numpy subprocess probe started asynchronously before the "
        "TensorFlow import and joined at the first post (cgroup quota + affinity + the measured ladder); when the probe cannot run or its "
        "table is implausible the pool is sized from affinity/quota instead and that is said on one FALLBACK line (outputs unaffected)",
        "none (always on in the kit entry)", ("exact",), default_in_kit="the probe's pool size"),
    "pipelined_chunked_post": Lever(
        "pipelined_chunked_post", "datapath", "exact", "kitlib/sad_post.py", "borzoi_sad.py:274-282 (setup), :309-312 (submit per variant), :352-355 (flush before write_pct); kitlib/sad_post.py:263-267 (applies), :299-387 (ChunkedPost), :390- (PipelinedChunkedPost)",
        "variant k's host post (untransform, strand collapse, the per-column statistics, the HDF5 row writes) runs per column chunk on a "
        "thread pool while the main thread goes on to variant k+1's one-hot and forward; every chunk sees the identical column bytes, every "
        "row is written in variant order; applies only to the option set the chunked post supports (a targets file, strand sums, the length-preserving "
        "write, per-column statistics) — any other option set runs the stock post path in the entry, untouched; the writer this pipeline calls per chunk is the mode's own (post_writer)",
        "none (always on in the kit entry)", ("exact",),
        default_in_kit="pipelined, one post in flight"),
    "post_writer": Lever(
        "post_writer", "datapath", "exact", "kitlib/writer.py", "kitlib/writer.py:28-140 (write_snp_len_exact: the statistics), :144-153 (make_writer)",
        "the per-variant statistics writer (the stock write_snp_len, borzoi_sad.py:452-): the stock's expressions with only the intermediates "
        "the requested statistics use (the stock computes every intermediate whatever --stats asks for) — the stock's bits by construction; "
        "a requested statistic outside the writer's set sends the whole call to the stock function",
        "none (always on in the kit entry)", ("exact",)),
    "graph_forward": Lever(
        "graph_forward", "forward", "exact", "kitlib/forward.py", "borzoi_sad.py:240-244 (install after the model load); kitlib/forward.py:36-78 (ForwardCall), :81-92 (install)",
        "how the forward is dispatched: the first call of the process is the stock's eager Keras call (cuDNN selects its algorithms there, as "
        "in a stock process); the same model is then traced ONCE with tf.function(model, jit_compile=False) — no XLA, no fusion, the same ops "
        "on the same kernels — and every later call runs that one graph instead of thousands of eager op dispatches from Python; the "
        "documented command's input has one shape, so one trace (the stamp counts traces, eager and graph calls); the graph keeps its "
        "intermediate buffers on the device, so the process's device high-water mark is higher than the eager call's",
        "KIT_FWD=1 (set by this package in the kit mode: kit.KIT_COMPOSITION)", ("exact",), default_in_kit="off (the stock's eager call)"),
    "copy_free_forward": Lever(
        "copy_free_forward", "forward", "exact", "kitlib/forward.py", "borzoi_sad.py:240-244 (install after the model load); kitlib/forward.py:36-78 (ForwardCall), :81-92 (install)",
        "the forward call's host side: the stock returns preds = model(x).numpy().astype(dtype) — the device-to-host copy, then a second "
        "float32 -> float32 copy of the (2, L, T) array; the kit returns the .numpy() array as is and converts only when the requested dtype "
        "differs (the same float32 values, one host copy fewer); the stamp counts every call",
        "KIT_FWD=1 (set by this package in the kit mode: kit.KIT_COMPOSITION)", ("exact",), default_in_kit="off (the stock's two copies)"),
    "kit_stamp": Lever(
        "kit_stamp", "observability", "none", "borzoi_sad.py", "borzoi_sad.py:45-60 (the KIT_STAMP record and its atexit dump)",
        "one `KIT_STAMP {json}` line on stdout at exit (the kit name, the writer, the pool, post_applies, the forward counters, the main() "
        "wall) and, with KIT_STAMP_DIR set, the same record as kit_stamp.json there — never into the output directory",
        "KIT_STAMP_DIR", ("exact",)),
}


def by_mode(mode: str) -> Dict[str, Lever]:
    return {n: lv for n, lv in LEVERS.items() if mode in lv.modes}
