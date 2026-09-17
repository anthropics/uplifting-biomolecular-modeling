"""The knobs of a generation run: upstream's own Hydra overrides, passed through verbatim.

``complexa generate <config> [<override> …]`` takes its settings as Hydra override tokens (``++key=value``) and its two command-line options
(``--verbose``, ``--job-id N``); this package adds no knob of its own. Everything given after ``--`` on the ``design`` command line reaches
the child unchanged, in order, AFTER the tokens the package composes (``stock_design.compose``) — so a caller's token for a key the package
also spells (the checkpoints, the task and its target entry) is applied last and wins, exactly as Hydra
applies a later override; nothing is refused here. A value not given is the shipped configuration's
(``configs/search_binder_local_pipeline.yaml`` → ``configs/pipeline/binder/binder_generate.yaml`` → ``configs/pipeline/model_sampling.yaml``)
— ``SHIPPED`` lists those values with their file and line and ``tests/test_settings.py`` holds the list to the carried config files
under ``stock/src``. Four of them the package also READS from the tokens, for its own accounting only (``values_of``): the design count of
the run = ``generation.dataloader.dataset.nres.nsamples`` (shipped 4) × ``…dataset.nrepeat_per_sample`` (shipped 1) — the number of PDB
files the census expects (outputs.census) — the ``seed`` (shipped 5; upstream applies ``lightning.seed_everything(seed + job_id)``
itself) and ``generation.dataloader.batch_size`` (shipped 16), both for the record.
"""
from __future__ import annotations

import re
from typing import Dict, Optional, Sequence

NSAMPLES_KEY = "generation.dataloader.dataset.nres.nsamples"      # binder_generate.yaml:26 `nsamples: 4` — designs drawn per run (× nrepeat_per_sample)
NREPEAT_KEY = "generation.dataloader.dataset.nrepeat_per_sample"   # binder_generate.yaml:28 `nrepeat_per_sample: 1`
BATCH_KEY = "generation.dataloader.batch_size"                     # binder_generate.yaml:19 `batch_size: 16` — designs per network call
SEED_KEY = "seed"                                                  # search_binder_local_pipeline.yaml:30 `seed: 5` (generate.py:72-75 seeds with it)
TASK_KEY = "generation.task_name"                                  # the target: an entry name in upstream's target dictionary
RUN_NAME_KEY = "run_name"                                          # the suffix of upstream's run root (search_binder_local_pipeline.yaml:21 `run_name: search_binder_local`)
SHIPPED_NSAMPLES = 4
SHIPPED_NREPEAT = 1
SHIPPED_BATCH = 16
SHIPPED_SEED = 5                                                   # search_binder_local_pipeline.yaml:30 `seed: 5` — the value a run uses when no ++seed token is given
SHIPPED_RUN_NAME = "search_binder_local"
SHIPPED_SEARCH = "best-of-n"                                       # binder_generate.yaml:48 — scored by the reward model of :128 (upstream's structure-prediction reward; its parameters under $AF2_DIR); the generation stage alone is the caller's `++generation.search.algorithm=single-pass ++generation.reward_model=null`

# The shipped values a run uses when no token names them. file = the carried copy under stock/src/<file>; `line` holds the text.
SHIPPED = (
    {"key": "generation.args.nsteps", "value": "400", "file": "configs/pipeline/model_sampling.yaml", "line": "nsteps: 400"},
    {"key": "generation.args.self_cond", "value": "True", "file": "configs/pipeline/model_sampling.yaml", "line": "self_cond: True"},
    {"key": "generation.args.guidance_w", "value": "1.0", "file": "configs/pipeline/model_sampling.yaml", "line": "guidance_w: 1.0"},
    {"key": "generation.args.ag_ratio", "value": "0.0", "file": "configs/pipeline/model_sampling.yaml", "line": "ag_ratio: 0.0"},
    {"key": "generation.model.bb_ca.schedule", "value": "mode log, p 2.0", "file": "configs/pipeline/model_sampling.yaml", "line": "mode: log"},
    {"key": "generation.model.bb_ca.gt", "value": "mode 1/t, p 1.0", "file": "configs/pipeline/model_sampling.yaml", "line": "mode: 1/t"},
    {"key": "generation.model.local_latents.schedule", "value": "mode power, p 2.0", "file": "configs/pipeline/model_sampling.yaml", "line": "mode: power"},
    {"key": "generation.model.local_latents.gt", "value": "mode tan, p 1.0", "file": "configs/pipeline/model_sampling.yaml", "line": "mode: tan"},
    {"key": "generation.model.*.simulation_step_params.sampling_mode", "value": "sc", "file": "configs/pipeline/model_sampling.yaml", "line": "sampling_mode: sc"},
    {"key": "generation.model.*.simulation_step_params.sc_scale_noise", "value": "0.1", "file": "configs/pipeline/model_sampling.yaml", "line": "sc_scale_noise: 0.1"},
    {"key": "generation.model.*.simulation_step_params.t_lim_ode", "value": "0.98", "file": "configs/pipeline/model_sampling.yaml", "line": "t_lim_ode: 0.98"},
    {"key": "generation.model.*.simulation_step_params.t_lim_ode_below", "value": "0.02", "file": "configs/pipeline/model_sampling.yaml", "line": "t_lim_ode_below: 0.02"},
    {"key": BATCH_KEY, "value": str(SHIPPED_BATCH), "file": "configs/pipeline/binder/binder_generate.yaml", "line": "batch_size: 16"},
    {"key": NSAMPLES_KEY, "value": str(SHIPPED_NSAMPLES), "file": "configs/pipeline/binder/binder_generate.yaml", "line": "nsamples: 4"},
    {"key": NREPEAT_KEY, "value": str(SHIPPED_NREPEAT), "file": "configs/pipeline/binder/binder_generate.yaml", "line": "nrepeat_per_sample: 1"},
    {"key": SEED_KEY, "value": str(SHIPPED_SEED), "file": "configs/search_binder_local_pipeline.yaml", "line": "seed: 5"},
    {"key": RUN_NAME_KEY, "value": SHIPPED_RUN_NAME, "file": "configs/search_binder_local_pipeline.yaml", "line": "run_name: search_binder_local"},
    {"key": "generation.search.algorithm", "value": SHIPPED_SEARCH, "file": "configs/pipeline/binder/binder_generate.yaml", "line": "algorithm: best-of-n"},
    {"key": "generation.reward_model", "value": "upstream's composite structure-prediction reward (af_params_dir: ${oc.env:AF2_DIR})", "file": "configs/pipeline/binder/binder_generate.yaml", "line": "af_params_dir: ${oc.env:AF2_DIR}"},
    {"key": "ckpt_name", "value": "complexa.ckpt", "file": "configs/search_binder_local_pipeline.yaml", "line": "ckpt_name: complexa.ckpt"},
)

OVERRIDE_RE = re.compile(r"^(?P<prefix>\+\+|\+|~)?(?P<key>[A-Za-z_][A-Za-z0-9_./@]*)(?:=(?P<value>.*))?$", re.S)
INT_KEYS = (NSAMPLES_KEY, NREPEAT_KEY, BATCH_KEY, SEED_KEY)         # the tokens the package reads back for its accounting


def last_value(tokens: Sequence[str], key: str) -> Optional[str]:
    """The value of the LAST ``[++|+]key=value`` token of ``key`` (as Hydra applies them), or None when no token names it (or the last one
    deletes it, ``~key``). Read only: the tokens themselves reach the child untouched."""
    found = None
    for t in tokens:
        m = OVERRIDE_RE.match(str(t))
        if not m or m.group("key") != key:
            continue
        found = None if m.group("prefix") == "~" else m.group("value")
    return found


def _int(text: Optional[str]) -> Optional[int]:
    try:
        return int(str(text).strip()) if text is not None else None
    except ValueError:
        return None


def values_of(tokens: Sequence[str]) -> Dict[str, object]:
    """What the run will use, read from the tokens (the last token of a key wins) with the shipped value where none is given: ``{designs,
    nsamples, nrepeat, batch, seed, seed_given, batch_given, counted}`` — ``designs`` = nsamples × nrepeat, the number of PDB files the run
    writes; ``counted`` is False (and ``designs`` None) when a given nsamples / nrepeat value is not an integer the package can count with
    (the token still reaches upstream, which answers for it). For the package's accounting and record only."""
    raw = {k: last_value(tokens, k) for k in INT_KEYS}
    given = {k: raw[k] is not None for k in INT_KEYS}
    nsamples = _int(raw[NSAMPLES_KEY]) if given[NSAMPLES_KEY] else SHIPPED_NSAMPLES
    nrepeat = _int(raw[NREPEAT_KEY]) if given[NREPEAT_KEY] else SHIPPED_NREPEAT
    counted = nsamples is not None and nrepeat is not None
    batch = _int(raw[BATCH_KEY]) if given[BATCH_KEY] else SHIPPED_BATCH
    seed = _int(raw[SEED_KEY]) if given[SEED_KEY] else SHIPPED_SEED
    return {"designs": (nsamples * nrepeat) if counted else None, "nsamples": nsamples if nsamples is not None else raw[NSAMPLES_KEY],
            "nrepeat": nrepeat if nrepeat is not None else raw[NREPEAT_KEY], "counted": counted,
            "batch": batch if batch is not None else raw[BATCH_KEY], "batch_given": given[BATCH_KEY],
            "seed": seed if seed is not None else raw[SEED_KEY], "seed_given": given[SEED_KEY]}
