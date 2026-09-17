"""The stock caller — mode ``off``: one pipeline step of upstream ``boltzgen`` in a clean subprocess whose environment is checked before it runs.

``design --mode off`` runs ``boltzgen configure`` and then, per step i of ``<run_dir>/steps.yaml``, execs this module through the shared
core's stock-proof contract (``opt_core.stock_proof.stock_command``: ``python -s -m boltzgen_opt.stock_design --proof-json
<run_dir>/stock_env_proof_<step>.json --env-absent <prefixes> --kit-dirs <dirs> --module-prefixes <p> -- <seed+i|none>
<run_dir>/<config_i>``) with every environment name under stock/PINS.json ``stock_environment.must_be_absent_prefixes`` and the exact
``must_be_absent_names`` stripped (no PYTHONPATH, no BOLTZGEN_OPT), ``BOLTZGEN_PIPELINE_STEP=<step>`` (as upstream's own ``run`` sets it)
and ``HF_HUB_OFFLINE=1``. Before anything of upstream is imported it checks its environment with the core's one proof
(``opt_core.stock_proof.env_proof``: no forbidden variable present, no kit module loaded, no kit directory on sys.path, the package's
autoload finder not armed, no kit sitecustomize, torch not yet imported, no core module beyond the proof) and writes it. Any finding is a
refusal (exit 3, nothing ran) — the stock arm never runs with the kit within reach.

Nothing else runs in this process: no census, counter, wrap or timer — the stock step is upstream alone after the proof and the seed line
(what the accelerators did is upstream's own `Using kernels:` resolution line in the configure log; this caller adds nothing).

Then the step, seeded when the caller passed ``--seed`` (the recipe the kit modes apply in-process in ``bg_inproc.py`` and ``bg_hook.py``
— one contract on every arm) and exactly as upstream's ``run`` leaves it, unseeded, when not::

    pl.seed_everything(seed, workers=True); random.seed(seed); np.random.seed(seed % (2**32)); torch.manual_seed(seed)   # --seed given
    seed_default_rng(np)                                                                                                # --seed given
    <resources/main.py>.main(config, [])

The seed line is ``SEED_LINE`` below and is executed verbatim (the kit's unit tests hold it equal to ``bg_inproc.py``'s line);
``seed_default_rng`` seeds upstream's featurizer generator — ``np.random.default_rng(None)``, OS entropy in upstream's own form — from the
seeded numpy stream's state, the same derivation as ``bg_hook.py``'s (the kit's unit tests hold the two equal draw for draw). Together
they fix WHICH noise a stock process draws (upstream's ``run`` has no seed argument), never how: the same sampler, steps, schedule and batch.
"""
from __future__ import annotations

import os
import sys
from typing import List, Optional

from opt_core import stock_proof

from . import TAG, print_fresh

PREFIX = f"[{TAG} stock]"
EXIT_NOT_STOCK = stock_proof.EXIT_NOT_STOCK                                 # the package's EXIT_NOT_ACTIVE (3): nothing ran
SEED_LINE = "pl.seed_everything(seed, workers=True); random.seed(seed); np.random.seed(seed % (2**32)); torch.manual_seed(seed)"
MAIN_REL = ("resources", "main.py")
NO_SEED = "none"                                                            # the seed slot's word for an unseeded step (upstream's own form)


def main_py() -> str:
    import boltzgen                                                       # noqa: WPS433 — after the proof
    return os.path.join(os.path.dirname(os.path.abspath(boltzgen.__file__)), *MAIN_REL)


def seed_default_rng(np) -> None:
    """``np.random.default_rng(None)`` seeded from the seeded global numpy stream's STATE (MT19937 key + position, hashed with the call's
    ordinal: deterministic under the seeded stream, distinct per call, the stream itself not advanced — a draw here would shift what
    upstream's data modules read next); an explicit seed passes through. The ordinal counts the seedless calls of THIS process under THIS
    stream key (keyed by pid and key): a forked DataLoader worker — where upstream's featurizer draws the generator — counts from its own
    first call, and a re-seed of the stream opens a fresh count, so the seeds a step's workers derive are a function of the step's seed, the
    worker and the order of its items, not of what the parent process drew before upstream's step began. The derivation is the seed
    hook's (``bg_hook.py``), so every route hands upstream's conformer sampler the same generators at the same ``--seed``."""
    import hashlib
    orig, calls = np.random.default_rng, {}

    def seeded_default_rng(seed=None, *a, **kw):
        if seed is None:
            st = np.random.get_state()
            key = np.asarray(st[1], dtype=np.uint32).tobytes()
            k = (os.getpid(), hashlib.sha256(key).digest()[:8])
            calls[k] = calls.get(k, 0) + 1
            seed = int.from_bytes(hashlib.sha256(key + int(st[2]).to_bytes(4, "little")
                                                 + calls[k].to_bytes(8, "little")).digest()[:4], "little") & (2**31 - 1)
        return orig(seed, *a, **kw)
    np.random.default_rng = seeded_default_rng


def run_step(seed: Optional[int], config: str) -> None:
    """The seed line when ``seed`` is not None, then upstream's step function on the step's config — what upstream's ``run`` executes per
    step (``python <resources/main.py> <config>``), in this process."""
    import importlib.util
    if seed is not None:
        import random
        import numpy as np
        import torch
        import pytorch_lightning as pl
        exec(SEED_LINE, {"pl": pl, "random": random, "np": np, "torch": torch, "seed": seed})
        seed_default_rng(np)
    path = main_py()
    spec = importlib.util.spec_from_file_location("bg_main", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.main(config, [])


def parse_seed(word: str) -> Optional[int]:
    """``none`` -> None (unseeded); else the step's integer seed."""
    return None if word == NO_SEED else int(word)


def main(argv: Optional[List[str]] = None) -> int:
    a, stock = stock_proof.parse_stock_argv(list(sys.argv[1:] if argv is None else argv), prog="python -s -m boltzgen_opt.stock_design")
    proof = stock_proof.env_proof(env_absent=a.env_absent, kit_dirs=a.kit_dirs, module_prefixes=a.module_prefixes)
    proof["step"] = os.environ.get("BOLTZGEN_PIPELINE_STEP")
    proof["pid"] = os.getpid()
    stock_proof.write_proof(a.proof_json, proof)
    if not proof["ok"]:
        print_fresh(f"{PREFIX} NOT STOCK: {stock_proof.violations_sentence(proof)}")
        print_fresh(f"{PREFIX} refused: the stock step does not run with the kits within reach (exit {EXIT_NOT_STOCK})")
        return EXIT_NOT_STOCK
    if len(stock) != 2:
        print_fresh(f"{PREFIX} usage: ... -- <seed|none> <config.yaml>")
        return 2
    seed, config = parse_seed(stock[0]), stock[1]
    print_fresh(f"{PREFIX} environment proven clean; step={proof['step']} seed={NO_SEED if seed is None else seed} config={config} {stock_proof.clean_sentence(proof)}")
    run_step(seed, config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
