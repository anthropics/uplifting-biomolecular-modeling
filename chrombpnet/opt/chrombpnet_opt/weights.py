"""``run.sh install --weights DIR``: the three stock weight files under DIR checked against ``stock/PINS.json`` ``weights`` (sha256).

Upstream chrombpnet 1.0.1 trains models and ships no downloader for trained ones, and the pinned weights are files, not URLs — the
ENCODE GM12878 ATAC-seq fold-0 ChromBPNet model (experiment ENCSR637XSC) as STOCK.md 'Weights' lists it: ``bias_scaled.h5`` and
``nobias.h5`` are, byte for byte, the members ``model.bias_scaled.fold_0.ENCSR637XSC.h5`` and
``model.chrombpnet_nobias.fold_0.ENCSR637XSC.h5`` of the ENCODE models archive ENCFF142IOR
(https://www.encodeproject.org/files/ENCFF142IOR/); ``chrombpnet_recompiled.h5`` — the ``-cm`` model every route reads — is those two
recombined into one Keras model file the way upstream composes them for training (``recombine`` below; ENCODE's own combined member
is a Keras 2.4 file the pinned stack cannot load). So this step fetches nothing: when ``chrombpnet_recompiled.h5`` is absent and the
two members are in place at their pins it builds it first (CPU, seconds, the same bytes every time under the pinned stack), then hashes
every pinned file present under DIR with the pin gate's own digest routine (stock/check_pins.py ``sha256_file``), prints one line per
file (``ok``, ``MISSING`` with where the file comes from, or ``MISMATCH`` with both digests) and exits 1 naming any file absent or off
its pin — files are left in place, never deleted. All pinned: ``WEIGHTS OK: 3/3 under DIR`` and DIR is the CHROMBPNET_OPT_WEIGHTS every
route reads (configs/*.env derive CHROMBPNET_OPT_MODEL from it). The driver-cache tarball the configs name beside the weights
(``cache/nv_compute_cache_<class>.tar``) is not a weight and is not checked here: without it the kit's TensorFlow routes compile their
kernels once and say so (STOCK.md).

    python -m chrombpnet_opt.weights DIR        (what `run.sh install --weights DIR` runs after the pin check)
"""
import importlib.util
import os
import sys
from typing import Dict, List, Optional, Tuple

from . import stack

PINS_RELPATH = os.path.join("stock", "PINS.json")
CHECK_PINS_RELPATH = os.path.join("stock", "check_pins.py")     # the pin gate: read_pins() and sha256_file() are its routines, used here as they are
ROOT_KEY = "root"                                                # PINS.json weights: the one non-directory key (the prose naming $CHROMBPNET_OPT_WEIGHTS)
ORIGIN = {                                                       # where each pinned file comes from, printed beside a MISSING line (STOCK.md 'Weights')
    "bias_scaled.h5": "member model.bias_scaled.fold_0.ENCSR637XSC.h5 of ENCODE ENCFF142IOR (https://www.encodeproject.org/files/ENCFF142IOR/)",
    "nobias.h5": "member model.chrombpnet_nobias.fold_0.ENCSR637XSC.h5 of ENCODE ENCFF142IOR (https://www.encodeproject.org/files/ENCFF142IOR/)",
    "chrombpnet_recompiled.h5": "bias_scaled.h5 + nobias.h5 recombined into one Keras model: built here once both members are in place (run.sh install --weights DIR)",
}
RECOMBINED = "chrombpnet_recompiled.h5"                          # built by recombine() from the two members below when absent
COMPONENTS = ("bias_scaled.h5", "nobias.h5")                     # (bias model, bias-free model): the ENCODE members, inputs of recombine()
LOGSUMEXP_SRC = "lambda x: tf.math.reduce_logsumexp(x, axis=-1, keepdims=True)"   # upstream's log-counts combination (chrombpnet_with_bias_model.py)
LOGSUMEXP_FILENAME = "chrombpnet_with_bias_model.py"             # the source name recorded in the saved Lambda: fixed, so the file's bytes do not depend on where this runs


def pin_gate(tree: Optional[str] = None):
    """The tree's stock/check_pins.py loaded as a module (stdlib only; nothing of the stack is imported)."""
    path = os.path.join(tree or stack.tree_home(), CHECK_PINS_RELPATH)
    spec = importlib.util.spec_from_file_location("chrombpnet_stock_check_pins", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def recombine(bias_h5: str, nobias_h5: str, out_h5: str) -> str:
    """Build the ``-cm`` model from the two component models the way upstream composes them for training
    (chrombpnet.training.models.chrombpnet_with_bias_model: both models on one ``sequence`` input, the profile logits added, the log
    counts combined by log-sum-exp) and save it as a Keras HDF5 file without optimizer state. CPU only (no device kernels are compiled
    for this); deterministic: the same two inputs under the same stack give the same bytes wherever the kit is installed (the Lambda
    layer's function is compiled from a fixed source string, so no install path is recorded in the file)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"; os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    from tensorflow.keras.layers import Add, Concatenate, Input, Lambda
    from tensorflow.keras.models import Model, load_model
    wo_bias = load_model(nobias_h5, compile=False, custom_objects={"tf": tf}); wo_bias._name = "model_wo_bias"   # upstream's names for the two sub-models
    bias = load_model(bias_h5, compile=False, custom_objects={"tf": tf}); bias._name = "model"
    inp = Input(shape=tuple(wo_bias.inputs[0].shape[1:]), name="sequence")
    out, bias_out = wo_bias(inp), bias(inp)
    logsumexp = eval(compile(LOGSUMEXP_SRC, LOGSUMEXP_FILENAME, "eval"), {"tf": tf, "__name__": "__main__"})   # a global `tf`, no closure: what stock's loader resolves through its custom_objects
    concat = Concatenate(axis=-1, name="concatenate")([out[1], bias_out[1]])
    profile = Add(name="logits_profile_predictions")([out[0], bias_out[0]])
    counts = Lambda(logsumexp, name="logcount_predictions")(concat)
    part = out_h5 + ".part"
    Model(inputs=[inp], outputs=[profile, counts]).save(part, include_optimizer=False, save_format="h5"); os.replace(part, out_h5)
    return out_h5


def build_missing(root: str, files: List[Tuple[str, str, int]], gate, stream) -> Optional[str]:
    """Build RECOMBINED under ``root`` when it is absent and both COMPONENTS sit beside its pinned path at their pinned digests;
    returns the built path, or None (nothing to build, an input absent or off its pin, or the build failed — said on ``stream``)."""
    want = {os.path.basename(rel): (rel, sha) for rel, sha, _ in files}
    if RECOMBINED not in want or any(c not in want for c in COMPONENTS):
        return None
    out = os.path.join(root, want[RECOMBINED][0])
    if os.path.isfile(out):
        return None
    parts = [os.path.join(root, want[c][0]) for c in COMPONENTS]
    if not all(os.path.isfile(p) and gate.sha256_file(p) == want[c][1] for p, c in zip(parts, COMPONENTS)):
        return None
    print("BUILDING  {}  from {} + {} (upstream's with-bias composition; CPU, under a minute)".format(out, *COMPONENTS), file=stream); stream.flush()
    try:
        return recombine(parts[0], parts[1], out)
    except Exception as e:                                       # the digest lines below then report the file MISSING
        print("NOT BUILT {}  ({}: {})".format(out, type(e).__name__, str(e).splitlines()[0][:200] if str(e) else ""), file=stream)
        return None


def pinned_files(pins: dict) -> List[Tuple[str, str, int]]:
    """[(relative path under the weights root, sha256, size_bytes)] from PINS.json ``weights`` — every directory key, every file under it."""
    out = []
    for sub, files in pins["weights"].items():
        if sub == ROOT_KEY or not isinstance(files, dict):
            continue
        for name, ent in files.items():
            out.append((os.path.join(sub, name), ent["sha256"], int(ent.get("size_bytes") or 0)))
    return sorted(out)


def check(root: str, tree: Optional[str] = None, stream=None) -> Dict[str, dict]:
    """Hash every pinned file under ``root``; returns {rel: {status: ok|MISSING|MISMATCH, want, have, path}} and prints one line per file."""
    stream = stream or sys.stdout
    gate = pin_gate(tree)
    pins = gate.read_pins(os.path.join(tree or stack.tree_home(), PINS_RELPATH))
    result = {}
    build_missing(root, pinned_files(pins), gate, stream)
    for rel, want, size in pinned_files(pins):
        path = os.path.join(root, rel)
        if not os.path.isfile(path):
            result[rel] = {"status": "MISSING", "want": want, "have": None, "path": path}
            print("MISSING   {}  ({}; {} bytes expected)".format(path, ORIGIN.get(os.path.basename(rel), "see STOCK.md 'Weights'"), size), file=stream)
            continue
        have = gate.sha256_file(path)
        if have == want:
            result[rel] = {"status": "ok", "want": want, "have": have, "path": path}
            print("ok        {}  sha256 {}".format(path, have), file=stream)
        else:
            result[rel] = {"status": "MISMATCH", "want": want, "have": have, "path": path}
            print("MISMATCH  {}  sha256 {} != pinned {} (left in place)".format(path, have, want), file=stream)
    return result


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1 or argv[0].startswith("-"):
        print("usage: python -m chrombpnet_opt.weights DIR   (DIR = the weights root, your CHROMBPNET_OPT_WEIGHTS; run.sh install --weights DIR)", file=sys.stderr)
        return 2
    root = os.path.abspath(argv[0])
    if not os.path.isdir(root):
        print("WEIGHTS FAILED: {} is not a directory (the weights root holding GM12878_ATAC/fold_0/…, STOCK.md 'Weights')".format(root), file=sys.stderr)
        return 1
    result = check(root)
    bad = sorted(rel for rel, r in result.items() if r["status"] != "ok")
    if bad:
        print("WEIGHTS FAILED: {}/{} not as pinned under {}: {} (stock/PINS.json weights; nothing was deleted)".format(len(bad), len(result), root, ", ".join(bad)), file=sys.stderr)
        return 1
    print("WEIGHTS OK: {n}/{n} under {root} — export CHROMBPNET_OPT_WEIGHTS={root}".format(n=len(result), root=root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
