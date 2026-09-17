"""The deterministic recipe — the settings under which `exact` claims byte-identical outputs, stated once, here, as data. OFF by default
(`design --det 0`, upstream's own numerics: unseeded unless the request file or `--seed` seeds it); `design --det 1` applies it. Byte-identical
for this generator = byte-identical PDB files at matched (seed, design name `<problem>_<i>`), one process per request on both arms.

(1) seed: `experiment.seed` in the request file — upstream's own key (src/genie3/config/models.py:18, default None = unseeded); the
    stock path seeds through `lightning.seed_everything(seed, workers=True)` before the model is built
    (src/genie3/generation/workflow.py:78,175-176) and the driver does the same in the same order (driver/g3fast.py build_everything). A request file
    that sets its own seed keeps it on every mode and `design --seed S` composes S on every mode, with or without the recipe; under `--det 1` a
    request with neither gets the recipe's seed, 0. Under one seed the binder lengths (numpy draws in the featuriser) and the noise stream (the
    one default CUDA generator) are the same on both arms; at batch size B both arms draw the batch's noise as one tensor per step, in dataset order.
(2) kernel selection: the stock path runs under `lightning.Trainer(deterministic=True)` (workflow.py:183), which calls
    `torch.use_deterministic_algorithms(True)` and sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`; the driver sets exactly the same before the
    model reaches the GPU (g3fast.py lightning_equivalent_determinism). Nothing is exported by the package: both arms set it themselves, at every det level.
(3) fp32: the exact line passes no `--tf32`, and before anything runs the driver proves torch's own fp32 matmul policy is live
    (opt/genie3_opt/g3batch.py: opt_core.precision.policy expect(FP32); a live TF32 state refuses the pass) — the stock path's default.
    NVIDIA_TF32_OVERRIDE — read by cuBLAS / cuDNN, not by any kit or by torch's flags — is dropped from both arms
    (stack.DROP_ENV_NAMES; stock/PINS.json must_be_absent_prefixes), so no inherited setting moves either arm off strict fp32 silently.
(4) one process per request: the request file names the problems and `n_sample`; every design of the request is sampled in that
    process, in the stock order.
(5) the stock line's `--log-dir`: `<out>/logs` when the pass names an output directory (src/genie3/cli.py:71-72; upstream's default is
    `logs/runs` under the process cwd, which is the checkout — the package keeps the checkout free of run logs); the caller's own
    `--log-dir` is passed through unchanged.
"""
from typing import Dict, Optional

LEVELS = (0, 1)
DEFAULT_LEVEL = 0                                                            # off: upstream's own numerics (config/models.py:18: experiment.seed None = unseeded)
SEED = 0                                                                     # (1) the recipe's seed
SEED_KEY = ("experiment", "seed")                                            # (1) src/genie3/config/models.py:18
SEED_DOC = "src/genie3/generation/workflow.py:78,175-176; driver/g3fast.py build_everything"
DETERMINISM = {"stock": "lightning.Trainer(deterministic=True) (src/genie3/generation/workflow.py:183): torch.use_deterministic_algorithms(True), CUBLAS_WORKSPACE_CONFIG=:4096:8",
               "driver": "lightning_equivalent_determinism() + setdefault (driver/g3fast.py build_everything): the same two"}   # (2)
TF32 = {"driver": "no --tf32: opt_core.precision.policy expect(FP32) before anything runs", "stock": "torch defaults (allow_tf32 False)", "doc": "opt/genie3_opt/g3batch.py (the line's numerics policy)"}   # (3)
LOG_DIR_FLAG = "--log-dir"                                                   # (5) src/genie3/cli.py:71-72


def request_seed(request: dict, seed_override=None, det_level: int = DEFAULT_LEVEL) -> Optional[int]:
    """The seed a request runs with, or None (leave `experiment.seed` as the file has it: absent = unseeded, upstream's default): `--seed`,
    else the request file's experiment.seed, else — under `det_level` 1 only — the recipe's 0."""
    if seed_override is not None:
        return int(seed_override)
    v = (request.get("experiment") or {}).get("seed")
    if v is not None:
        return int(v)
    return SEED if int(det_level) >= 1 else None


def recipe(det_level: int = DEFAULT_LEVEL):
    """The recipe in the shared shape (opt_core.det.Recipe): level 1 = (1)-(5) above — it EXPORTS nothing (both arms set their determinism
    state themselves; the seed travels in the request copy), so its ``env`` / ``unset`` / ``pythonpath`` are empty and the stock proof needs no
    carve-out (opt_core.det.stock_exception is None); level 0 = production numerics, unseeded unless the request seeds itself (opt_core.det.PRODUCTION)."""
    from . import _core
    _core.ensure_importable()
    from opt_core.det import PRODUCTION, Recipe
    if int(det_level) == 0:
        return PRODUCTION
    return Recipe(level=1, note="experiment.seed in the request copy (0 unless the request or --seed sets one); torch determinism set by each arm itself; strict fp32; one process per request")


def describe(seed, det_level: int = DEFAULT_LEVEL) -> Dict[str, object]:
    """The recipe as recorded in opt_manifest.json (``line``: the shared one-line form, opt_core.det.describe)."""
    from opt_core.det import describe as _line
    return {"det": int(det_level), "line": _line(recipe(det_level)), "seed": seed, "seed_key": ".".join(SEED_KEY), "seed_doc": SEED_DOC,
            "determinism": dict(DETERMINISM), "tf32": dict(TF32), "process_rule": "one process per request, every design of the request in it",
            "log_dir": "<out>/logs"}
