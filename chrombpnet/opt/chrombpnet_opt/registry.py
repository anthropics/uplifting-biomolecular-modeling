"""The kit's levers and switches, as DATA. Nothing here is a value the kit decides at run time — every default, class table and route
decision stays in the kit's bytes (tf/chrombpnet_fastkit/fastdefault.py, torch/chrombpnet_k1/arch_tiles.json) and the package never
re-derives them. LEVERS: what the kit applies under `fast` / `exact` (name, what it does, where in the kit: `kit_file`). KIT_SWITCH_NAMES:
the kit's own internal environment names; found in a caller's environment they are removed for the run and named, never refused.
"""
from typing import Dict

FASTDEFAULT = "tf/chrombpnet_fastkit/fastdefault.py"
PRED_BW_FAST = "tf/pred_bw_fast.py"
ARCH_TILES = "torch/chrombpnet_k1/arch_tiles.json"


class Lever(object):
    __slots__ = ("name", "what", "kit_file")

    def __init__(self, name, what, kit_file):
        self.name, self.what, self.kit_file = name, what, kit_file

    def as_dict(self):
        return {"name": self.name, "what": self.what, "kit_file": self.kit_file}


LEVERS = {
    "forward_route": Lever("forward_route",
        "the forward route: k1 (the kit's Triton forward under torch, torch/chrombpnet_k1) | tf_function (the stock Keras graph in a tf.function) | "
        "keras_predict_fileorder (the stock's own predict() in input order); chosen per (GPU class, mode) from the kit's own tables "
        "(a class in neither table engages k1 with a cold JIT), the K1 cache-pins precondition and the K1 stack probe; on a k1 class whose stack or kernels cannot start the mode refuses by name (exit 3)",
        ARCH_TILES + " entries[*].default_route (fastdefault.arch_default_route); class table + class_route (" + FASTDEFAULT + "); resolve (" + FASTDEFAULT + "); " + PRED_BW_FAST),
    "native_dilation": Lever("native_dilation",
        "the TF route's dilated convolutions run natively instead of the stock's space_to_batch form; ON/OFF by GPU class; OFF under the deterministic recipe by detection",
        FASTDEFAULT + ":54-55 (class table), :236-257 (decision); " + PRED_BW_FAST),
    "tail": Lever("tail",
        "the remainder batch padded to the batch shape (no new kernel shape for N < batch) or the stock's own tail shape; by GPU class",
        FASTDEFAULT + ":14-22 (class table), :259-261; " + PRED_BW_FAST),
    "warmup": Lever("warmup", "a load-time warm-up of the forward at the job's batch shapes", FASTDEFAULT + ":258,262"),
    "jit_cache": Lever("jit_cache",
        "the driver ComputeCache tarball installed add-only into the process's cache dir before the first CUDA context (TF routes only; the K1 route skips it)",
        PRED_BW_FAST + " _JIT_TAR; tf/chrombpnet_fastkit/__init__.py jit_cache_boot"),
    "bigwig_writer": Lever("bigwig_writer", "the kit's numpy bigWig writer in the stock's block layout, in a separate writer process overlapped with the forward",
        "tf/chrombpnet_fastkit/bigwig_numpy.py; tf/chrombpnet_fastkit/__init__.py predict_pipelined"),
    "metrics_stage": Lever("metrics_stage", "the kit's metrics stage (numpy bigWig reader for the observed counts, vectorised profile metrics) and its overlap with the forward",
        "tf/chrombpnet_fastkit/metrics_overlap.py, bigwig_reader.py, h5_fast.py; " + PRED_BW_FAST + " (the -bw stage)"),
    "h5_writer": Lever("h5_writer", "the parallel-deflate predictions.h5 writer in the stock's layout",
        "tf/chrombpnet_fastkit/h5_fast.py; " + PRED_BW_FAST),
    "prefetch": Lever("prefetch", "the forked observed-counts prefetch worker (replaced by the metrics overlap when -bw is given) and the featurise prefetch thread",
        PRED_BW_FAST),
    "preimport": Lever("preimport", "the stage-2 import chain imported on a daemon thread under stage 1",
        "torch/chrombpnet_k1/_preimport.py; " + PRED_BW_FAST),
    "exit_fast": Lever("exit_fast", "outputs witnessed on disk, then the process leaves without the interpreter's teardown",
        "torch/chrombpnet_k1/_exit.py; " + PRED_BW_FAST),
    "k1_batch": Lever("k1_batch", "the K1 route's internal batch, pinned host buffers and shipped Triton cache dir",
        "torch/chrombpnet_k1/ apply()/predict(); torch/triton_cache_of_record.json; " + PRED_BW_FAST),
    "malloc_env": Lever("malloc_env", "mallopt tunables and their environment export before numpy/TF load",
        "tf/chrombpnet_fastkit/malloc_env.py (applied at the entry script's import, " + PRED_BW_FAST + ")"),
}   # type: Dict[str, Lever]

ROUTES = ("k1", "tf_function", "keras_predict_fileorder", "stock_cli")        # the forward words: the kit's three (fastdefault.resolve) and stock_cli = the --mode off arm's word on its EXIT line (never a route of the fast process)

# THE KIT'S OWN SWITCH NAMES — found in a caller's environment they are removed for the run and named (stack.kit_switches_set / kit_env; the stock child's forbidden list in
# stock_pred_bw.py is the same names plus the package's own CHROMBPNET_OPT prefix, locked in step by a test). Source: the names the kit's
# own code reads from the environment (os.environ / getenv over opt/kit_ho/tf and opt/kit/torch; torch/vendor and the kit's test_*.py
# excluded), plus K1_EXIT_TEARDOWN, which torch/chrombpnet_k1/_exit.py receives as a parameter. Framework variables the kit also reads
# (TRITON_CACHE_DIR, CUDA_CACHE_PATH, CUDA_CACHE_MAXSIZE, TF_DETERMINISTIC_OPS, NPY_DISABLE_CPU_FEATURES, PYTHONPATH) are not switches
# and are never removed.
# Never a prefix rule: a caller's own bookkeeping variables may share a prefix with the kit's namespace.
KIT_SWITCH_NAMES = (
    'CHROMBPNET_DET_SEED',
    'CHROMBPNET_DET_SUBPROCESS',
    'CHROMBPNET_DET_SUBPROCESS_APPLIED',
    'CHROMBPNET_FASTKIT_GPU_CLASS_OVERRIDE',
    'CHROMBPNET_FASTKIT_RECORD',
    'CHROMBPNET_FASTKIT_TAIL',
    'CHROMBPNET_FASTKIT_TAIL_SOURCE',
    'CHROMBPNET_JIT_CACHE_TAR',
    'CHROMBPNET_K1_DIR',
    'CHROMBPNET_K1_STREAMS',
    'K1_BLOCK_K',
    'K1_BLOCK_M',
    'K1_BLOCK_N',
    'K1_EXIT_TEARDOWN',
    'K1_NUM_STAGES',
    'K1_NUM_WARPS',
    'K1_PREIMPORT',
)
KIT_SWITCH_DET_NAMES = ("CHROMBPNET_DET_SUBPROCESS", "CHROMBPNET_DET_SEED", "CHROMBPNET_DET_SUBPROCESS_APPLIED")   # the recipe's names (read by tf/det_subprocess/sitecustomize.py): the package composes them under --det
KIT_CACHE_TAR_NAME = "CHROMBPNET_JIT_CACHE_TAR"                                                                    # composed by the package (stack.cache_tar_env), never by the caller
KIT_RECORD_NAME = "CHROMBPNET_FASTKIT_RECORD"                                                                      # composed by the package (cli.run_fast), never by the caller: the file the kit's entry script writes its run record to, folded into opt_manifest.json
