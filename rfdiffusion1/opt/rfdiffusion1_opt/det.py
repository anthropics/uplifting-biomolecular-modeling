"""The deterministic recipe — `design --det 1`: the equality settings under which `exact` claims byte-equal outputs with `off` at the
same design index, stated once, here, as data (README.md 'Modes'; upstream_issues/RFD1-001, RFD1-002).
Default is `--det 0`: upstream's unseeded default on every arm.

(1) seed: `inference.deterministic=True` — upstream's own key: before every design upstream seeds torch / numpy / random with the design
    index (stock/src/scripts/run_inference.py:32-35,71-73), and the resident drivers seed the same way under the same key
    (opt/forward/fast_inference/drivers/rfd_bench.py:95-96,144-145). Under `--det 1` the stock arm appends the override to the typed ones
    (stock_overrides) and the driver line composes it (modes.settings_of, DRIVER_FIXED); under `--det 0` neither does and upstream's
    default (False) is composed on the driver line — a typed `inference.deterministic=…` stands on both arms either way.
(2) fp32: the driver's `--tf32 0` default sets allow_tf32 False for matmul and cudnn (rfd_bench.py:31,45-46); the stock command line runs
    at torch's defaults (no code on the stock path). NVIDIA_TF32_OVERRIDE and TORCH_ALLOW_TF32_CUBLAS_OVERRIDE are dropped from the kit's own
    driver child (stack.DROP_ENV_NAMES, named in the manifest's env_dropped); the stock process receives the caller's environment minus the kit
    namespace only (stock/PINS.json must_be_absent_prefixes) — an inherited TF32 override reaches upstream as the caller set it, the caller's to record.
(3) CPU host class: the equality claim holds within one CPU host class (ISA-dispatched CPU math in the CPU-side diffuser and
    the host-built IGSO3 schedule cache give a different, internally consistent trajectory per class); both arms of an equality run execute
    on one class. The cross-host settings (MKL_CBWR and friends, a fresh `inference.schedule_directory_path`) are the caller's to
    export and type: the environment reaches both arms unchanged and the key is upstream's own.
(4) process position: the first design of a process is the fresh-process numerics class, the later ones the warmed class; equality is
    judged at matched positions (one process per arm and design block).
(5) DGLBACKEND=pytorch on every arm (the driver's setdefault: rfd_bench.py:42; stack.DGL_BACKEND; DGL's backend selection, not numerics).
"""
from typing import List

SEED_OVERRIDE = "inference.deterministic=True"                                   # (1)
SEED_KEY = SEED_OVERRIDE.split("=", 1)[0]
HOST_CLASS_RULE = "byte-equal equality within one CPU host class and one process-position class (upstream_issues/RFD1-001, RFD1-002)"   # (3) (4)
LEVELS = (0, 1)                                                                  # `design --det 0|1`
DEFAULT_LEVEL = 0
LEVEL_WHAT = {0: "upstream's unseeded default on every arm (no seed override); two passes are not byte-equal to each other",
              1: f"seed override {SEED_OVERRIDE} on the stock command line and the kit line alike (seed = design index)"}


def stock_overrides(det: bool = False, typed=()) -> List[str]:
    """The recipe's overrides for the stock command line, appended after the typed ones: the seed override under `--det 1` unless the caller
    typed the key (a typed value stands); nothing under `--det 0`."""
    typed_keys = {str(o).split("=", 1)[0] for o in (typed or [])}
    return [SEED_OVERRIDE] if det and SEED_KEY not in typed_keys else []
