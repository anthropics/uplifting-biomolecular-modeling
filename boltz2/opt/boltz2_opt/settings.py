"""`boltz predict`'s own settings — the same inputs on every mode, with upstream's flag names, semantics and defaults; no presets.

KNOBS lists the `boltz predict` options `pred` takes by name, in boltz/main.py's declaration order, with upstream's kinds: ``--checkpoint``
(path), ``--recycling_steps`` (3), ``--sampling_steps`` (200), ``--diffusion_samples`` (1), ``--max_parallel_samples`` (5), ``--step_scale``
(None = 1.5 for Boltz-2, main.py predict), ``--write_full_pae`` / ``--write_full_pde`` (flags), ``--output_format`` (mmcif | pdb),
``--num_workers`` (2), ``--override`` (flag), ``--seed`` (None), ``--use_msa_server`` (flag), ``--msa_server_url``, ``--msa_pairing_strategy``
(greedy), ``--use_potentials`` (flag), ``--method`` (None), ``--preprocessing-threads``, ``--affinity_mw_correction`` (flag),
``--sampling_steps_affinity`` (200), ``--diffusion_samples_affinity`` (5), ``--affinity_checkpoint`` (path), ``--max_msa_seqs`` (8192),
``--subsample_msa`` (flag), ``--num_subsampled_msa`` (1024), ``--no_kernels`` (flag), ``--write_embeddings`` (flag). The defaults are
upstream's, read from stock/PINS.json ``cli_defaults`` (the one place they are stated, = main.py's). A knob the caller does not give is
absent: on ``--mode off`` the argument is not passed (the bare command is upstream exactly as shipped); on the worker route (exact | fast |
big) the worker keeps upstream's default for it. Any other `boltz predict` option reaches the stock CLI verbatim after ``--`` on ``--mode off``.

The worker route serves WORKER_KNOBS, each forwarded to the one site the persistent worker states it (bz_worker_lev*.py): ``predict`` =
Boltz2's predict_args, step_scale, write_full_pae / pde, output_format, the checkpoint, the DataLoader's num_workers, and the batch json's
``options`` — process_inputs' MSA-server / max_msa_seqs / preprocessing-threads arguments, MSAModuleArgs' subsampling, the steering switches
(``--use_potentials``), the data module's ``--method``, the writer's ``--write_embeddings``, upstream's affinity-leg options (affinity_leg.request).
``--seed`` is the pass's one seed, ``--seeds a,b`` several; ``--override`` is what the worker always does (it recomputes). It refuses BY NAME
the one knob its line cannot serve (WORKER_REFUSED): ``--no_kernels`` — every kit row runs upstream's kernels on; the kernels-off line is the
stock route's (``--det`` likewise, cli). ``--use_potentials`` on exact / fast: the CUDA-graphed sampler and the DiT hoist replay one fixed
denoising step and cannot carry upstream's per-step steering, so that run takes them OFF by name (modes.RUN_DROPS: two ``LEVER … state=off
reason=use_potentials`` lines; every trunk lever stays on; upstream's eager sampler steers as `boltz predict` does).
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

from . import modes, stack

KNOBS = (   # (name, kind) in boltz/main.py's declaration order; kind: int | float | str | "flag" | "path" (must exist, as click.Path(exists=True)) | tuple of choices
    ("checkpoint", "path"),
    ("recycling_steps", int), ("sampling_steps", int), ("diffusion_samples", int), ("max_parallel_samples", int), ("step_scale", float),
    ("write_full_pae", "flag"), ("write_full_pde", "flag"), ("output_format", ("pdb", "mmcif")), ("num_workers", int), ("override", "flag"),
    ("seed", int), ("use_msa_server", "flag"), ("msa_server_url", str), ("msa_pairing_strategy", str), ("use_potentials", "flag"), ("method", str),
    ("preprocessing_threads", int), ("affinity_mw_correction", "flag"), ("sampling_steps_affinity", int), ("diffusion_samples_affinity", int),
    ("affinity_checkpoint", "path"), ("max_msa_seqs", int), ("subsample_msa", "flag"), ("num_subsampled_msa", int), ("no_kernels", "flag"),
    ("write_embeddings", "flag"),
)
KNOB_NAMES = tuple(k for k, _ in KNOBS)
KIND = dict(KNOBS)
FLAG_SPELLING = {"preprocessing_threads": "--preprocessing-threads"}   # main.py spells this one option with a hyphen; every other one is --<name>
MINIMUM = {"recycling_steps": 0, "sampling_steps": 1, "diffusion_samples": 1, "max_parallel_samples": 1, "num_workers": 0, "preprocessing_threads": 1,
           "sampling_steps_affinity": 1, "diffusion_samples_affinity": 1, "max_msa_seqs": 1, "num_subsampled_msa": 1}
PREDICT_FLAGS = ("recycling_steps", "sampling_steps", "diffusion_samples", "max_parallel_samples")          # Boltz2's predict_args dims (the batch json's `predict`)
PROCESS_INPUTS_KNOBS = ("use_msa_server", "msa_server_url", "msa_pairing_strategy", "preprocessing_threads", "max_msa_seqs")   # boltz main.py process_inputs' arguments (the worker's PROCESS_INPUTS_KW)
MSA_ARGS_KNOBS = ("subsample_msa", "num_subsampled_msa")                                                        # MSAModuleArgs at model construction
AFFINITY_KNOBS = ("affinity_mw_correction", "sampling_steps_affinity", "diffusion_samples_affinity", "affinity_checkpoint")   # upstream's affinity leg (affinity_leg.request overrides)
OPTION_KNOBS = ("checkpoint", "use_potentials", "method", "write_embeddings") + PROCESS_INPUTS_KNOBS + MSA_ARGS_KNOBS + AFFINITY_KNOBS   # the batch json's `options`: only the ones the caller gave (absent = the worker's stated upstream default)
WORKER_KNOBS = PREDICT_FLAGS + ("step_scale", "write_full_pae", "write_full_pde", "output_format", "override", "seed", "num_workers") + OPTION_KNOBS
WORKER_REFUSED = {
    "no_kernels": "every kit row runs upstream's kernels on (modes.py rows, the KERNELS census expects them engaged); the kernels-off line is the stock route's: --mode off --no_kernels",
}


def flag(name: str) -> str:
    """The command-line spelling of a knob: ``--<name>``, except main.py's ``--preprocessing-threads``."""
    return FLAG_SPELLING.get(name, f"--{name}")
SINGLE_SHAPE_LEVERS = ("rollout", "graph_sampler", "dit_hoist")  # the roll-out / the CUDA-graphed sampler and the DiT pair-bias hoist capture ONE sample-chunk shape per prediction: they serve a
                                                                # sampling whose chunks (upstream_sample_chunks) all have one size and step aside BY NAME for a run with distinct sizes
                                                                # (modes.RUN_DROPS distinct_sample_chunks, every row; the stock eager loop serves it) — never a refusal


def upstream_sample_chunks(diffusion_samples: int, max_parallel_samples: int) -> List[int]:
    """The sample chunk sizes upstream's sampler runs per denoising step — boltz/model/modules/diffusionv2.py ``sample``:
    ``torch.arange(multiplicity).chunk(multiplicity % max_parallel_samples + 1)`` (torch.chunk: ``n`` requested chunks of size
    ``ceil(D / n)``, the last one shorter, possibly fewer than ``n``). E.g. (D, M) = (5, 5) → [5]; (10, 5) → [10]; (3, 5) → [1, 1, 1];
    (6, 5) → [3, 3]; (7, 5) → [3, 3, 1]; (3, 2) → [2, 1]."""
    D, M = int(diffusion_samples), int(max_parallel_samples)
    n_req = D % M + 1
    size = -(-D // n_req)
    return [size] * (D // size) + ([D % size] if D % size else [])
BOLTZ2_STEP_SCALE = 1.5                                        # main.py predict: `step_scale = 1.5 if step_scale is None else step_scale` for Boltz-2


def stock_defaults() -> dict:
    """`boltz predict`'s defaults for KNOBS (stock/PINS.json cli_defaults = main.py's click defaults; step_scale None → 1.5 for Boltz-2)."""
    d = stack.load_pins()["cli_defaults"]
    out = {k: d[k] for k in KNOB_NAMES}
    out["use_kernels"] = bool(d.get("use_kernels", True))
    return out


def given(flags: Optional[dict]) -> dict:
    """The knobs the caller gave: value options that are not None, flag options that are True — validated (ValueError names the knob)."""
    out = {}
    for k in KNOB_NAMES:
        v = (flags or {}).get(k)
        if v is None or (KIND[k] == "flag" and v is False):
            continue
        kind = KIND[k]
        if kind == "flag":
            if v is not True:
                raise ValueError(f"{flag(k)} is a flag (given {v!r})")
        elif isinstance(kind, tuple):
            if v not in kind:
                raise ValueError(f"{flag(k)} {v!r}: one of {'|'.join(kind)}")
        elif kind is str:
            v = str(v)
            if k == "method":                                          # main.py:1150-1157: boltz refuses an unknown method before anything runs, with this message; so does every mode
                try:
                    from boltz.data import const as _const
                except ImportError:                                    # boltz absent: the pins gate names that before any launch
                    _const = None
                if _const is not None and v.lower() not in _const.method_types_ids:
                    raise ValueError(f"Method {v} not supported. Supported: {list(_const.method_types_ids.keys())}")
        elif kind == "path":
            v = str(v)
            if not os.path.exists(v):                                 # click.Path(exists=True): boltz refuses a missing path before anything runs; so does this
                raise ValueError(f"{flag(k)} {v!r}: no such file or directory")
        else:
            try:
                v = kind(v)
            except (TypeError, ValueError):
                raise ValueError(f"--{k} {v!r} is not {'an integer' if kind is int else 'a number'}") from None
            if k in MINIMUM and v < MINIMUM[k]:
                raise ValueError(f"--{k} {v} is below {MINIMUM[k]}")
        out[k] = v
    unknown = sorted(set(flags or {}) - set(KNOB_NAMES))
    if unknown:
        raise ValueError(f"unknown boltz predict setting(s) {unknown}: {' '.join('--' + k for k in KNOB_NAMES)}")
    return out


def stock_argv(flags: Optional[dict], order: Optional[Sequence[str]] = None) -> List[str]:
    """The given knobs as `boltz predict` arguments (``--k v``, a flag as ``--k``), in the caller's command-line order (`order`: knob names as
    they appeared) for those named there, then main.py's declaration order. Nothing given → ``[]``: the bare upstream command."""
    g = given(flags)
    names = [k for k in (order or []) if k in g] + [k for k in KNOB_NAMES if k in g and k not in (order or [])]
    seen, argv = set(), []
    for k in names:
        if k in seen:
            continue
        seen.add(k)
        argv += [flag(k)] if KIND[k] == "flag" else [flag(k), str(g[k])]
    return argv


def worker_settings(mode: str, flags: Optional[dict]) -> dict:
    """The worker route's settings in force for a `pred --mode <kit mode>` call: upstream's defaults with the given knobs over them —
    ``{recycling_steps, sampling_steps, diffusion_samples, max_parallel_samples, step_scale, write_full_pae, write_full_pde, output_format,
    override, kernels, given}`` (``given``: the knob names the caller gave). A knob the worker line cannot serve is a ValueError naming it
    and why (WORKER_REFUSED). A shape a lever of the row cannot serve is never refused: sample chunks of DISTINCT sizes per denoising step
    (upstream's own chunking of ``diffusion_samples`` by ``max_parallel_samples``, upstream_sample_chunks) take the single-shape sampler levers
    (SINGLE_SHAPE_LEVERS) off by name for that run (``run_off``, modes.RUN_DROPS distinct_sample_chunks) and the stock eager loop serves it, as
    upstream does. ``seed`` is not a setting here (worker.run takes the seeds)."""
    g = given(flags)
    row = modes.resolve(mode)
    refused = [f"--{k}: {WORKER_REFUSED[k]}" for k in g if k in WORKER_REFUSED]
    if refused:
        raise ValueError(f"not served on the worker route (--mode {row['mode']}): " + "; ".join(refused))
    d = stock_defaults()
    eff = {k: g.get(k, d[k]) for k in WORKER_KNOBS if k not in ("seed", "num_workers") + OPTION_KNOBS}
    eff["num_workers"] = int(g["num_workers"]) if "num_workers" in g else None        # None: the worker line's own (modes.WORKER_ARGS num_workers, 1); given: the DataLoader's, as boltz predict --num_workers
    eff["options"] = {k: (os.path.abspath(g[k]) if KIND[k] == "path" else g[k]) for k in OPTION_KNOBS if k in g}   # the batch json's `options` (a path absolute: the worker runs in its own directory)
    eff["run_off"] = modes.run_drops_for(row["mode"], g)                                # levers this run takes off by name for an option or a shape they cannot carry (modes.RUN_DROPS: exact / fast --use_potentials ->
                                                                                     # the graphed sampler, the hoist; every row on sample chunks of DISTINCT sizes per denoising step -> the single-shape
                                                                                     # sampler levers SINGLE_SHAPE_LEVERS, `distinct_sample_chunks` — never a refusal)
    eff["step_scale"] = float(BOLTZ2_STEP_SCALE if eff["step_scale"] is None else eff["step_scale"])
    for k in ("write_full_pae", "write_full_pde", "override"):
        eff[k] = bool(eff[k])
    eff["kernels"] = modes.worker_args(mode)["kernels"]
    eff["given"] = [k for k in KNOB_NAMES if k in g]
    return eff


def settings_tokens(eff: dict) -> str:
    """``k=v`` for every worker knob away from upstream's default (the ACTIVE line's settings words; empty at the defaults)."""
    d = stock_defaults(); d["step_scale"] = float(BOLTZ2_STEP_SCALE if d["step_scale"] is None else d["step_scale"])
    flat = {**{k: v for k, v in eff.items() if k not in ("options", "run_off")}, **(eff.get("options") or {})}
    words = []
    for k in WORKER_KNOBS:
        if k == "seed" or k not in flat or (k == "num_workers" and flat[k] is None):
            continue
        v, dv = flat[k], d[k]
        if (bool(v) != bool(dv)) if KIND[k] == "flag" else (v != dv):
            words.append(f"{k}={int(v) if KIND[k] == 'flag' else v}")
    return " ".join(words)


def settings_word(flags: Optional[dict]) -> str:
    """``defaults`` when the caller gave no knob, else ``flags`` (the KERNELS census settings token)."""
    return "flags" if given(flags) else "defaults"


def manifest_settings(mode: str, eff: dict, seeds: List[int], n_gpu: int) -> Dict:
    """The manifest's ``settings`` of a worker-route run."""
    return {**{k: eff[k] for k in WORKER_KNOBS if k in eff and k != "seed"}, "options": dict(eff.get("options") or {}), "run_off": dict(eff.get("run_off") or {}),
            "kernels": eff["kernels"], "given": list(eff["given"]), "seeds": list(seeds), "n_gpu": int(n_gpu),
            "source": "boltz predict's own flags over upstream's defaults (stock/PINS.json cli_defaults = boltz/main.py); kernels: the mode row"}
