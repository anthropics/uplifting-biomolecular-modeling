"""`af3-torch-opt pred | check | stock | exports` (== `python -m af3_torch_opt …` == `run.sh <cmd> --config h100 …`).

pred: one prediction per fold-input JSON and seed in one process chain — featurise (the fork, JAX venv, CPU: the stock CLI's data pipeline
first when `--run_data_pipeline` (its flag, default and database / binary flags, passed through), then one feature batch per seed of the fold
input's `modelSeeds`) → forward (the kit, torch venv, the GPU; every (input, seed) of the call in ONE model
process) → postprocess (the fork's writers, JAX venv, CPU) — the ACTIVE line first, one COMMAND line per step, one ITEM line (+ one PHASE line: its trunk / sampler / confidence-head walls) per (input,
seed) after the forward step (`name= seed= forward_s= tokens= bucket= peak_gb= ok=`: the model process's wall for that item, the per-item
time), one KERNELS line (`routed=… ok= <name>=<version>:core|none`: the kernels served from the shared core's carried copies, the model process's
gate verdict and where each was imported from), one LEVER line per registry lever (`name=<family>.<lever> state=on|off|skipped arm= [reason=] <evidence>`:
whether the lever ran in this arm and the run record that proves it), the DONE line last. Layout under --output_dir: `<name>/` per input in the fork's
own form (`seed-<s>_sample-<k>/<name>_seed-<s>_sample-<k>_{model.cif, confidences.json, summary_confidences.json}`, the best-ranked copy
as `<name>_model.cif` / `<name>_confidences.json` / `<name>_summary_confidences.json`, `ranking_scores.csv`, `<name>_data.json`) — the stock
layout and nothing else once the pred succeeds: the steps' hand-offs (`.af3_torch_opt_work/<name>/seed-<s>/{batch.npz, batch.pkl, result.npz}`, the step logs and
reports) are removed on success and kept, named on the DONE line (`work=`), when anything failed. An item with a failed seed is failed-with-reason
(its seeds that ran are still written out by postprocess: the unit is the (input, seed)). A degraded lever set — the model
process applied fewer levers than the mode names (`LEVERS requested=… applied=… agree=0 missing=…`), a kernel lever that served the stock
path or died at run time (the kit's per-call census: `FALLBACK lever=… fallback_<reason>=n expected=0|1 dead=… observed=n accounted=n size_gated=…`,
judged per item against the kernels' declared gates — registry.fallback_account; the per-item counters ride on every
`forwards[]` record and the process census on the run record), or `off` running with a kernel census at all (`STOCK census=present ok=0`) —
is PARTIAL: outputs kept (the run record's `partial.events` names every event), rc 3 and a `PARTIAL refused=<kinds> rc=3 opt_out=--allow-partial`
line unless `--allow-partial` was passed, which the DONE line says (`partial=allowed:<kinds>`). An (input, seed) that failed inside the model
process is named with its error on a `FAILED item=… seed=… error=…` line (the words of the exception — e.g. the core's RowpairRefused under
`--n_gpu P`, which names `--mode off` / `ROWPAIR_TRIMUL_KERNELS=torch` — beside the ITEM line's ok=0); the pred then exits 1.
Exit 0 ok, 1 a step or an item failed (named on its ITEM / FAILED line and the DONE line's failed=: total accounting), 2 usage, 3 not active (the gates: interpreters, kit,
parameters) or partial.
check: the dry run — the activation report, nothing launched (`--json` for the dict).
warm: the kit's one-time kernel and cache setup, ahead of the first pred (`cmd_warm`): one short pred per requested mode (`--mode M`, or
`all` = every mode in turn) over three synthetic single-chain fold inputs of 448 / 832 / 1216 tokens (`WARM_SIZES`: the padded lengths a
400 / 800 / 1200-token input runs at under fast / big) or over the fold inputs named on the command line, with the cheapest protocol that
still reaches every compiled, autotuned and captured path once per length (`WARM_KNOBS`: two trunk passes, eight denoising steps, the stock
sample count; the data pipeline off — the inputs carry their MSAs, nothing is searched, no network). It fills the Triton / Inductor / JAX
caches under `AF3_TORCH_CACHE_ROOT` through the pred's own three processes and prints ONE `WARM mode= sizes= inputs= cache_root= cache_before=
cache_after= new_files= seconds= rc=` line per mode after that pred's own lines; the throwaway inputs and outputs live in one temporary
directory removed at the end. A second run over a warm root finds nothing to compile (new_files=0 or thereabouts) and takes the model build
plus three short items.
stock: xfold's own CLI at its shipped defaults on the composed stock venv (stock_cli.py). exports: the config's required-variable checks.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from typing import List, Optional

from opt_core.cli import input_paths, item_name, read_report, step_reason  # noqa: F401 — the pass helpers are the core's (input_paths(a): --json_path list + --input_dir/*.json)
from opt_core.mem.rowpair import launch as _rowpair_launch
from opt_core.process import run_step as _core_run_step
from opt_core.report import EXIT_FAIL as EXIT_FAILED, EXIT_NOT_ACTIVE, EXIT_OK, EXIT_USAGE

from . import big as _big, modes as _modes, pipeline_stream as PS, registry as _registry, run_record as _run_record, stack, stock_cli
from .report import PREFIX, RANKENV_MARK, TAG, activation_line, big_line, dump, emit, kernels_line, lever_line, line, phase_fields

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURISE = os.path.join(PKG_DIR, "featurise.py")
FORWARD = os.path.join(PKG_DIR, "forward.py")
POSTPROCESS = os.path.join(PKG_DIR, "postprocess.py")
WORK_DIR = ".af3_torch_opt_work"                 # under --output_dir while a pred runs (cmd_pred): batch / result hand-offs between the three processes, their logs and reports
# EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_NOT_ACTIVE = 0, 1, 2, 3: the core's exit table (opt_core.report), imported above


# The stock CLI's data-pipeline flags `pred` passes through unchanged (xfold run_alphafold.py's names and spellings, plus alphafold3's
# --max_template_date: the pipeline that runs is the fork's). featurise.py builds the fork's DataPipelineConfig from them as its own
# run_alphafold.main does; a flag not given is not sent and takes the fork CLI's default there (featurise.PIPELINE_FLAGS names the same set).
STOCK_PIPELINE_STRINGS = ("db_dir", "jackhmmer_binary_path", "nhmmer_binary_path", "hmmalign_binary_path", "hmmsearch_binary_path", "hmmbuild_binary_path",
                          "small_bfd_database_path", "mgnify_database_path", "uniprot_cluster_annot_database_path", "uniref90_database_path",
                          "ntrna_database_path", "rfam_database_path", "rna_central_database_path", "pdb_database_path", "seqres_database_path", "max_template_date")
STOCK_PIPELINE_INTS = ("jackhmmer_n_cpu", "nhmmer_n_cpu")


def bool_flag(name):
    """An absl-style boolean for `--<name>[=true|false]` (bare = true; `--no<name>` is added beside it by the parser)."""
    def parse(text):
        v = str(text).strip().lower()
        if v in ("1", "true", "t", "yes", "y"): return True
        if v in ("0", "false", "f", "no", "n"): return False
        raise argparse.ArgumentTypeError(f"--{name}={text!r}: expected true|false")
    return parse


def pipeline_argv(a) -> List[str]:
    """featurise.py's share of the stock flags: --run_data_pipeline / --run_inference as 0|1 and every data-pipeline path / cpu flag the user gave."""
    argv = ["--run_data_pipeline", str(int(a.run_data_pipeline)), "--run_inference", str(int(a.run_inference))]
    for name in STOCK_PIPELINE_STRINGS + STOCK_PIPELINE_INTS:
        v = getattr(a, name, None)
        for x in (v if isinstance(v, list) else [v] if v is not None else []):   # --db_dir repeats (searched in order, as the fork's CLI does)
            argv += [f"--{name}", str(x)]
    return argv


PADDING_POLICIES = ("none", "kernel_tile")      # modes.MODE_PADDING values this wrapper serves
NO_PADDING_ROW = "none"                         # the featuriser's word for no token padding (featurise.py --buckets none: featurise_input(buckets=None), each input at its own token count)
TILE_ROW = "tile:"                              # the featuriser's word for the open-ended kernel-tile row (featurise.py --buckets tile:<k>: every multiple of k, no largest bucket)


def bucket_row(policy: str) -> str:
    """The featuriser's token-bucket row under a padding policy (modes.MODE_PADDING): ``none`` = no padding (the word ``none``: each input at
    its own token count, what xfold's CLI does as shipped); ``kernel_tile`` = every input padded to the next multiple of the shared core's
    KERNEL_TILE at any size (the word ``tile:<KERNEL_TILE>``: no largest bucket, so no input runs unpadded or is refused for its length —
    opt_core.shape_policy.padded_len(n, 'kernel_tile') for every n). An unknown policy is refused by name."""
    if policy == "none":                        # xfold as shipped: the stock CLI names no buckets, every input runs at its own token count
        return NO_PADDING_ROW
    if policy == "kernel_tile":                 # fast / big: the core's kernel tile, open-ended
        from opt_core import shape_policy as _sp  # opt_core >= 0.5.1 (stack.padding_gate refuses the mode by name before this on an older core)
        return f"{TILE_ROW}{int(_sp.KERNEL_TILE)}"
    raise ValueError(f"unknown padding policy {policy!r} (policies: {', '.join(PADDING_POLICIES)})")


def census_word(key: str) -> str:
    """A census counter key as a line word: `fallback:N<101` -> `fallback_Nlt101`, `fallback:c=384,rows<4096` -> `fallback_c=384_rowslt4096`."""
    return str(key).replace(":", "_").replace("<", "lt").replace(",", "_")


def templates_declared(json_path: str) -> int:
    """How many templates the fold input DECLARES: the entries of every protein chain's `templates` list in the AlphaFold 3 input JSON, once per
    chain id the entry names (a homodimer entry with ids [A, B] and four templates declares eight; a chain with `templates` absent or null
    declares none) — handed to the model process (forward.py --templates-declared) for the template census: every ITEM line says
    templates=<live>/<declared> in these units (live = per chain, the template slots whose atoms reached the model on that chain's tokens:
    forward.template_entries_live), the TEMPLATES line counts the pass (real=<live slots>/<slots featurised>: the featuriser's T slots span every
    chain; form=dense|row_born: templates_form; dummy_only = items whose declared templates all featurised dummy: the model ran them
    untemplated, as stock does; the line counts them)."""
    try:
        with open(json_path, encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, ValueError):
        return 0                                                                # an unreadable input is the featuriser's named failure, not this count's
    n = 0
    for entry in d.get("sequences", []) if isinstance(d, dict) else []:
        chain = entry.get("protein") if isinstance(entry, dict) else None
        if isinstance(chain, dict) and isinstance(chain.get("templates"), list):
            ids = chain.get("id")
            copies = len(ids) if isinstance(ids, list) and ids else 1                     # one entry may name several chain ids (copies): each copy carries the templates
            n += len(chain["templates"]) * copies
    return n


TEMPLATES_NONE = "none"                                          # the ITEM line's templates word for an item whose fold input declares no templates: it ran untemplated, as stock does


def templates_none(rec: dict) -> bool:
    """The item's fold input declares no templates and none were featurised live (it runs untemplated, exactly as stock does)."""
    return int(rec.get("templates_declared") or 0) == 0 and int(rec.get("templates_live") or 0) == 0


def templates_word(rec: dict):
    """The ITEM line's `templates=` word: `none` when the fold input declares no templates (the item ran untemplated), else
    `<live>/<declared>` (template slots featurised live / templates the fold input declares; `0/<k>` is the refused item); None when the
    model process recorded neither (a step that never reached the item)."""
    if rec.get("templates_live") is None and rec.get("templates_declared") is None:
        return None
    if templates_none(rec):
        return TEMPLATES_NONE
    return f"{int(rec.get('templates_live') or 0)}/{int(rec.get('templates_declared') or 0)}"


def templates_census(items: List[dict]) -> dict:
    """{items, declared, live, dummy_only, none} over every (input, seed) forward record of a pred: the TEMPLATES line and DONE's
    templates=l/d; `none` = records whose fold input declares no templates (ran untemplated); `slots` / `slots_live` = template slots featurised / with atoms (real=slots_live/slots)."""
    c = {"items": 0, "declared": 0, "live": 0, "dummy_only": 0, "none": 0, "slots": 0, "slots_live": 0}
    for it in items:
        for rec in it.get("forwards") or []:
            c["items"] += 1; c["declared"] += int(rec.get("templates_declared") or 0); c["live"] += int(rec.get("templates_live") or 0); c["slots"] += int(rec.get("templates_slots") or 0); c["slots_live"] += int(rec.get("templates_slots_live") or 0)
            c["dummy_only"] += int(int(rec.get("templates_declared") or 0) > 0 and int(rec.get("templates_live") or 0) == 0)
            c["none"] += int(templates_none(rec) and (rec.get("templates_live") is not None or rec.get("templates_declared") is not None))
    return c


def run_step(step: str, argv: List[str], env: dict, log_path: str, timeout: Optional[float] = None, on_line=None) -> dict:
    """One model process (opt_core.process.run_step under this package's prefix): argv on the COMMAND line, the merged transcript to
    log_path (each line also handed to ``on_line`` when given — the stock route's STOCK-ITEM tee), the STEP line; {step, argv, rc, wall_s,
    log, ok, error?} (rc -1 = the clock ran out, -2 = could not launch)."""
    return _core_run_step(step, list(argv), prefix=PREFIX, env=env, log_path=log_path, timeout_s=timeout, on_line=on_line)



def refused_mode_line(a) -> Optional[str]:
    """The NOT ACTIVE line of a mode this package refuses BY NAME (modes.REFUSED_MODES: `exact`, with its one reason), or None: printed by
    `pred` and `check` before anything else is read, exit EXIT_NOT_ACTIVE — the same words the autoload hook prints on the AF3_TORCH_OPT route."""
    mode = a.mode or _modes.mode_from_env()
    if mode in _modes.REFUSED_MODES:
        return activation_line({"active": False, "mode": mode, "n_gpu": getattr(a, "n_gpu", None), "reason": _modes.REFUSED_MODES[mode]})
    return None

fastnn_flag = bool_flag("fastnn")   # `--fastnn[=true|false]` in the stock CLI's absl spelling (bare `--fastnn` = true; `--nofastnn` = false)


def fastnn_of(a, mode: str) -> int:
    """The model process's fastnn switch: xfold's shipped Triton fastnn kernels (1) under modes.FASTNN_MODES unless `--nofastnn`; the kit's kernel
    levers serve those modules elsewhere (0, nothing passed) — the stock CLI's `--fastnn` default is on."""
    return int(mode in _modes.FASTNN_MODES and getattr(a, "fastnn", None) is not False)


def protocol_named(a, mode: str) -> bool:
    """Whether the SETTINGS line prints: always under modes.FASTNN_MODES (the stock kernels' switch is part of what ran) and whenever a protocol
    knob departs from the stock CLI's / the model constructor's defaults (a reduced protocol is a named event, never silent)."""
    return (mode in _modes.FASTNN_MODES or getattr(a, "fastnn", None) is not None or a.num_recycles is not None or a.diffusion_steps is not None
            or a.num_diffusion_samples != _modes.STOCK_NUM_DIFFUSION_SAMPLES)


def no_compile_alias(a) -> bool:
    """``--no-compile``: the opt-out of the kit's ``compile`` lever (torch.compile of the step glue) — an ALIAS of the ablation switch, nothing
    else: when the mode's own selection carries ``compile`` (fast / big), ``compile`` is appended to ``MODEL_OPT_LEVERS_OFF`` for this process
    (modes.ENV_LEVERS_OFF: the model is built without the lever, its LEVER line reads ``state=off reason=levers_off``, the ACTIVE line
    ``compile=off:user levers_off=…,compile``); under ``off`` / ``exact``, which never compile, there is nothing to drop — the flag is accepted and
    the ACTIVE line says ``compile=off:mode``. Returns whether the variable was extended. The model process inherits the variable as the switch's
    own route does (stack.model_process_env)."""
    if not getattr(a, "no_compile", False):
        return False
    off = list(_modes.levers_off())
    if "compile" in off:                                                                     # named already: the switch's own words stand
        return False
    try:
        carries = "compile" in (_modes.resolve(a.mode or _modes.mode_from_env(), environ={}).get("levers") or ())
    except _modes.UnsupportedMode:                                                           # an unknown / refused mode: the entry route below refuses it in its own words
        return False
    if not carries:
        return False
    os.environ[_modes.ENV_LEVERS_OFF] = ",".join(off + ["compile"])
    return True


def cmd_pred(a) -> int:
    if (refused := refused_mode_line(a)):                                                    # a mode refused by name (modes.REFUSED_MODES): nothing read or run
        emit(refused); return EXIT_NOT_ACTIVE
    no_compile_alias(a)                                                                      # --no-compile: MODEL_OPT_LEVERS_OFF gains `compile` before the mode resolves (one mechanism)
    inputs = input_paths(a)
    if not inputs:
        emit("pred: no input (--json_path <fold input json> [...] or --input_dir <dir>)"); return EXIT_USAGE
    missing = [p for p in inputs if not os.path.isfile(p)]
    if missing:
        emit(f"pred: input not found: {missing}"); return EXIT_USAGE
    if not a.output_dir:
        emit("pred: --output_dir is required"); return EXIT_USAGE
    if not a.run_inference and not a.run_data_pipeline:                                   # the stock CLI's own rule and words (run_alphafold.main)
        emit("pred: At least one of --run_inference or --run_data_pipeline must be set to true."); return EXIT_USAGE
    if a.model_dir:                                                                         # the stock CLI's --model_dir: the weights directory for this call (else $AF3_TORCH_PARAMS_DIR)
        os.environ["AF3_TORCH_PARAMS_DIR"] = os.path.abspath(a.model_dir)
    if (why := _modes.fastnn_refusal(a.mode or _modes.mode_from_env(), a.fastnn)):         # a --fastnn / --nofastnn value this mode cannot serve: refused by name, nothing runs
        emit(activation_line({"active": False, "mode": a.mode or _modes.mode_from_env(), "n_gpu": a.n_gpu, "reason": why})); return EXIT_NOT_ACTIVE
    try:
        rep = stack.activate(a.mode, quiet=a.quiet, n_gpu=a.n_gpu)
    except _modes.UnsupportedMode as e:
        emit(f"pred: {e}"); return EXIT_USAGE
    n_gpu_mismatch = False
    if not rep["active"]:
        return EXIT_NOT_ACTIVE
    emit(stack.cache_seed_line(stack.seed_cache_root()))                                 # a cache root with no compiled kernel yet takes this card's pre-filled tree from the environment when it carries one (stack.seed_cache_root: seconds; ONE CACHE line either way) — before any model process starts
    out = os.path.abspath(a.output_dir); work = os.path.join(out, WORK_DIR)               # the steps' hand-off files and logs: removed when the pred succeeds, kept (and named on the DONE line) when it does not
    os.makedirs(work, exist_ok=True)
    names = [item_name(p) for p in inputs]
    if len(set(names)) != len(names):
        emit(f"pred: input file stems must be distinct (they name the output dirs): {names}"); return EXIT_USAGE
    if any("=" in os.path.abspath(p) for p in inputs) or "=" in out:
        emit(f"pred: '=' is the item-spec delimiter of the model processes; input paths and --output_dir must not contain it: {inputs} {out}"); return EXIT_USAGE
    fastnn = fastnn_of(a, rep["mode"])                                          # xfold's shipped fastnn kernels under off / exact (forward.py --fastnn 1); the kit's kernels elsewhere
    rep["settings"] = {"num_recycles": a.num_recycles, "num_diffusion_samples": a.num_diffusion_samples, "diffusion_steps": a.diffusion_steps, "fastnn": fastnn, "confidence": "once_per_sample"}
    if protocol_named(a, rep["mode"]):                                           # the stock kernels' switch under off / exact, and any knob off its default in every mode: named, never silent
        emit(line("SETTINGS", num_recycles=a.num_recycles, num_diffusion_samples=a.num_diffusion_samples, diffusion_steps=a.diffusion_steps, fastnn=fastnn, confidence="once_per_sample"))
    env_jax, env_torch = stack.model_process_env(jax=True), stack.model_process_env()
    if "autotune_cache" in (rep.get("package_levers") or ()):                       # package lever autotune_cache: Triton's autotune result cache on in the model process (its autotuners read the knob when their modules are imported; forward.run sets it too)
        env_torch["TRITON_CACHE_AUTOTUNING"] = "1"
    else:
        env_torch.pop("TRITON_CACHE_AUTOTUNING", None)
    if int(rep["n_gpu"]) > 1:                                                    # n_gpu > 1 only: the shared core's launcher and the ranks print their lines under the kit's tag (`[af3-torch-opt] RANKENV …`, `… SCHEDULE …`, `… LEVER name=F2.trimul_rows …`)
        env_torch[_rowpair_launch.ENV_TAG] = TAG
    proof = stack.proof(env_torch)
    emit(line("STOCK", env_prefixes_absent=",".join(proof["prefixes"]), env_present=proof["env_present"] or "none", proof="ok" if proof["ok"] else "FAILED"))
    if rep.get("big"):
        emit(big_line(rep))                                                    # the memory levers in force beside ACTIVE (whose levers= names the kit levers the model is built with)
    steps: List[dict] = []
    items = {n: {"name": n, "input": p, "ok": False} for n, p in zip(names, inputs)}
    reports: dict = {}
    feat = None; ok_names: List[str] = []; seeds: dict = {}
    # The chain runs STREAMED when the mode carries the package levers prefetch / write_behind (pipeline_stream): the featuriser, the model
    # process and the writers are launched together and hand items over one by one — the model is built while the first input featurises,
    # input i+1 featurises while input i is on the GPU (prefetch), input i is written while input i+1 is on the GPU (write_behind). The same
    # three processes, files and reports as the sequential chain; the accounting below reads the same records once each process has ended.
    pkg = tuple(rep.get("package_levers") or ())
    stream = {l: l in pkg for l in PS.STREAM_LEVERS}
    stream_skip = None
    if any(stream.values()) and (int(rep["n_gpu"]) > 1 or not a.run_inference):        # the row-sharded launch runs its ranks' own item loop; a data-pipeline-only run has no model process: the levers step aside BY NAME
        stream_skip = "n_gpu>1:rowpair_ranks" if int(rep["n_gpu"]) > 1 else "run_inference=false"
        stream = {l: False for l in stream}
    rep["stream"] = {"selected": [l for l in PS.STREAM_LEVERS if l in pkg], "engaged": [], "skipped": ({l: stream_skip for l in PS.STREAM_LEVERS if l in pkg} if stream_skip else {})}
    workers = 1                                                                  # feat_par: the featurise step as pipeline_stream.FEAT_WORKERS processes when the chain streams and the pred has that many inputs
    if stream.get("feat_par"):
        if stream["prefetch"] and len(names) >= 2:
            workers = min(PS.FEAT_WORKERS, len(names))
        else:
            stream["feat_par"] = False; rep["stream"]["skipped"].setdefault("feat_par", "single_input" if stream["prefetch"] else "requires:prefetch")
    rep["feat_par"] = {"workers": workers, "inputs": len(names)}
    jobs: dict = {}                                                              # step -> pipeline_stream.Job: the steps of the chain running beside this process
    launched: list = []                                                          # every Job started (their spans give the overlap the LEVER lines report)
    wb_post: dict = {}                                                           # the writers' side of write_behind's census (postprocess.json), merged into the LEVER line
    t_chain = time.monotonic()

    def bg(step, argv_, env_, log_name, on_line=None, mark_done=True, register=True):
        """Start one step of the chain in the background (pipeline_stream.Job over run_step: the same COMMAND / STEP lines, log and record)."""
        j = PS.Job(step, work, run_step, argv_, env_, os.path.join(work, log_name), on_line=on_line, mark_done=mark_done); launched.append(j)
        if register:
            jobs[step] = j
        return j
    feat_extra: list = []                                                        # feat_par: the featurise step's further worker processes (Jobs), joined with the first

    def join_featurisers():
        """Join every featurise process; under feat_par merge the workers' reports into featurise.json and their step records into one, then mark the step done."""
        rec0 = jobs.pop("featurise").join()
        if not feat_extra:
            return rec0
        recs = [rec0] + [j.join() for j in feat_extra]
        merged = PS.merge_featurise_reports([read_report(os.path.join(work, "featurise.json"))] + [read_report(os.path.join(work, f"featurise.w{k}.json")) for k in range(1, len(recs))], names)
        with open(os.path.join(work, "featurise.json"), "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=1)
        PS.touch(PS.step_done_path(work, "featurise"))                          # all featurisers have ended: an input nobody handed over is withdrawn downstream from here on
        rep["feat_par"]["walls"] = ",".join(str(round(float(r.get("wall_s") or 0), 1)) for r in recs)
        return PS.merge_step_records(recs)

    def account_featurise():
        """The featurise report settles which inputs go on (ok_names) and their seeds; a failed input keeps its named reason. Reads steps[-1] as the featurise step."""
        nonlocal ok_names, seeds, feat
        feat = read_report(os.path.join(work, "featurise.json")) or {"items": []}; reports["featurise"] = feat
        ok_names = [it["name"] for it in feat.get("items", []) if it.get("ok")]
        seeds = {}
        for it in feat.get("items", []):
            items.get(it["name"], {}).update({k: it.get(k) for k in ("n_tokens", "bucket", "featurise_s", "msa_rows", "seeds") if k in it})
            seeds[it["name"]] = [int(x) for x in (it.get("seeds") or [])]
            if not it.get("ok"):
                items.get(it["name"], {})["error"] = f"featurise: {it.get('error')}"
            elif not seeds[it["name"]]:
                items.get(it["name"], {})["error"] = "featurise: report names no seed"; ok_names = [n for n in ok_names if n != it["name"]]
        for n in names:
            if n not in {it.get("name") for it in feat.get("items", [])}:
                items[n]["error"] = f"featurise: no report entry ({step_reason(steps[-1])})"
        return ok_names, seeds

    # 1. featurise — the fork under the JAX venv, CPU only
    buckets = bucket_row(rep["padding"])                                        # the mode's padding policy (modes.MODE_PADDING): none on off / exact (xfold as shipped), the core's kernel tile at any size on fast / big
    argv = [stack.jax_python(), FEATURISE, "--buckets", buckets, "--report", os.path.join(work, "featurise.json"), *pipeline_argv(a), *(["--repo_dir", stack.jax_repo()] if (a.run_data_pipeline or not a.run_inference) else []),
            *(["--stream", "1"] if stream["prefetch"] else [])]                  # prefetch: load every input, announce the seeds, hand each seed over as it completes
    item_args = [["--item", f"{n}={os.path.abspath(p)}={os.path.join(work if a.run_inference else out, n)}"] for n, p in zip(names, inputs)]   # --norun_inference (the stock CLI's data-pipeline-only run): the item's dir is its output dir, where the fork writes <name>_data.json as run_alphafold does; nothing else runs
    base_argv = list(argv)
    for k, ia in enumerate(item_args):                                           # worker 0's items (every item when workers == 1); the further workers' below
        if k % workers == 0:
            argv += ia
    if stream["prefetch"]:                                                       # the featuriser(s) in the background: the model process is launched the moment every input's seeds are announced
        j0 = bg("featurise", argv, env_jax, "featurise.log", mark_done=(workers == 1))
        for wk in range(1, workers):                                             # feat_par: inputs wk, wk+W, … on their own featuriser process (its own report and log, merged into featurise.json when all end)
            argv_w = [x if x != os.path.join(work, "featurise.json") else os.path.join(work, f"featurise.w{wk}.json") for x in base_argv] + [x for k, ia in enumerate(item_args) if k % workers == wk for x in ia]
            feat_extra.append(bg("featurise", argv_w, env_jax, f"featurise.w{wk}.log", mark_done=False, register=False))
        tables = [j0.wait_announced()] + [j.wait_announced() for j in feat_extra]
        announced = None if any(t is None for t in tables) else {k: v for t in tables for k, v in t.items()}
        if feat_extra:
            rep["stream"]["engaged"].append("feat_par")
        if announced is None:                                                    # a featuriser ended without announcing (an input list nothing of which loaded, or a featuriser without the line): on from the report(s), sequentially
            steps.append(join_featurisers()); account_featurise()
        else:                                                                    # provisional: every input that loaded, with its seeds — the featurise report settles it once the process ends (a failed input is withdrawn downstream)
            seeds = {n: list(v) for n, v in announced.items() if n in items}
            ok_names = [n for n in names if seeds.get(n)]
            rep["stream"]["engaged"].append("prefetch")
    else:
        steps.append(run_step("featurise", argv, env_jax, os.path.join(work, "featurise.log")))
        account_featurise()
    # 2. forward — the kit under the torch venv, one process for every featurised (item, seed)
    partial = None
    if not a.run_inference:                                                       # --norun_inference: the data pipeline ran (or was skipped) and <name>_data.json is written; no model, no writers
        for n in ok_names:
            items[n]["ok"] = "error" not in items[n]
        ok_names = []
    if ok_names:
        argv = [stack.torch_python(), FORWARD, "--kit", stack.kit_home(), "--dtk-home", stack.dtk_home(), "--params", rep["weights"]["file"], "--weights-sha256", rep["weights"]["sha256"], "--weights-pinned", str(int(rep["weights"]["pinned"])),
                "--levers", ",".join(rep["levers"]), "--dtk", str(int(rep["dtk"])), "--report", os.path.join(work, "forward.json"),
   # n_gpu > 1: the group's collective timeout (default: forward.py's)
                "--opt-core", stack.core_dir(), "--route", ",".join(stack.kernel_routes()),      # the routed kernels: the core's carried copies, gated in the model process (registry.KERNEL_ROUTES)
                *_big.forward_argv(rep.get("big")),                                          # big: the memory levers the model process acts on / reads back
                "--n-gpu", str(rep["n_gpu"]),
                *(["--stream-root", work] if (stream["prefetch"] or stream["write_behind"]) else []),        # the streamed chain: wait for each batch's hand-off (prefetch) / hand each result over as it closes (write_behind)
                "--package-levers", ",".join(l for l in (rep.get("package_levers") or ()) if l not in rep["stream"]["skipped"]), "--padding", rep.get("padding") or "none",   # the package levers and padding policy of the mode (modes.MODE_PACKAGE_LEVERS / MODE_PADDING), composed in the model process
                ]
        declared = {n: templates_declared(p) for n, p in zip(names, inputs) if n in ok_names}
        for n, k in declared.items():                                                              # templates the fold input DECLARES: the model process counts the live ones
            if k:                                                                                  # (the template census: declared per item, live per item and chain)
                argv += ["--templates-declared", f"{n}={k}"]
        for k, v in (("--num_recycles", a.num_recycles), ("--num_samples", a.num_diffusion_samples), ("--diffusion_steps", a.diffusion_steps)):   # the model constructor's names (forward.py)
            if v is not None:
                argv += [k, str(v)]
        if fastnn:                                                               # off / exact: xfold's shipped fastnn kernels; absent otherwise (the kit's kernel levers serve those modules)
            argv += ["--fastnn", "1"]
        for n in ok_names:
            for sd in seeds[n]:
                d = os.path.join(work, n, f"seed-{sd}")
                argv += ["--item", f"{n}={sd}={os.path.join(d, 'batch.npz')}={os.path.join(d, 'result.npz')}"]
        relay = (lambda ln: emit(ln.rstrip("\n")) if RANKENV_MARK in ln else None) if int(rep["n_gpu"]) > 1 else None   # n_gpu > 1 only: the launcher's ONE `[af3-torch-opt] RANKENV hashseed=<v> source=default|inherited ranks=P` line (the PYTHONHASHSEED every rank starts with), relayed verbatim from the model process's transcript the moment it is printed
        if stream["prefetch"] or stream["write_behind"]:                        # the model process beside this one (its RANKENV relay as before); joined below
            bg("forward", argv, env_torch, "forward.log", on_line=relay)
        else:
            steps.append(run_step("forward", argv, env_torch, os.path.join(work, "forward.log"), on_line=relay))
    # 3. postprocess — the fork's writers under the JAX venv, CPU only
    post_argv = None
    if ok_names:
        post_argv = [stack.jax_python(), POSTPROCESS, "--repo_dir", stack.jax_repo(), "--report", os.path.join(work, "postprocess.json"), "--opt_core", stack.core_dir(),
                     *(["--follow", work] if stream["write_behind"] else [])]              # write_behind: each item written as the model process hands it over
        if stream["write_behind"]:                                               # the writers beside the model process, over every input still in the chain (a withdrawn one leaves no row)
            bg("postprocess", post_argv + [x for n in ok_names for x in ("--item", f"{n}={os.path.join(work, n)}={os.path.join(out, n)}")], env_jax, "postprocess.log")
            rep["stream"]["engaged"].append("write_behind")
    # the chain's processes end in order — featuriser, model process, writers — and each report is read once its process has ended
    if "featurise" in jobs:
        steps.append(join_featurisers()); account_featurise()                    # final: a featurise failure named as always; the model process withdrew that input (no record), so did the writers
        ok_names = [n for n in ok_names if n in seeds]
    if "forward" in jobs:
        steps.append(jobs["forward"].join())
    post_rec = None
    if "postprocess" in jobs:                                                    # the writers end right after the model process (the last input's write): their record joins the steps after the forward's accounting, their census the LEVER line
        post_rec = jobs.pop("postprocess").join()
        wb_post = dict((read_report(os.path.join(work, "postprocess.json")) or {}).get("write_behind") or {})
    fwd = None
    if ok_names and (steps and steps[-1]["step"] == "forward"):
        fwd = read_report(os.path.join(work, "forward.json")) or {"items": []}; reports["forward"] = fwd
        rep["levers_applied"] = fwd.get("levers_applied"); rep["package_levers_applied"] = fwd.get("package_levers_applied"); rep["template_dedupe"] = fwd.get("template_dedupe"); rep["dev_scalars"] = fwd.get("dev_scalars"); rep["tri_layout"] = fwd.get("tri_layout"); rep["ln_rows"] = fwd.get("ln_rows"); rep["attn_layout"] = fwd.get("attn_layout"); rep["gate_fuse"] = fwd.get("gate_fuse"); rep["castcache"] = fwd.get("castcache"); rep["canonical_noise"] = fwd.get("canonical_noise"); rep["forward_build_s"] = fwd.get("build_s"); rep["device"] = fwd.get("device"); rep["graph_resets"] = fwd.get("graph_resets"); rep["pool_resets"] = fwd.get("pool_resets"); rep["graph_captures"] = fwd.get("graph_captures")
        rep["feat_par"] = {**(rep.get("feat_par") or {}), "engaged": int("feat_par" in rep["stream"]["engaged"])}   # package lever feat_par's census: featuriser processes, inputs, their walls
        rep["autotune_cache"] = fwd.get("autotune_cache")                                    # package lever autotune_cache's census (Triton autotune disk-cache hits / benchmarked-now counts)
        rep["package_levers_skipped"] = {**dict(fwd.get("package_levers_skipped") or {}), **rep["stream"]["skipped"]}   # a package lever named-not-applied by rule (canonical_noise: n_gpu>1; prefetch / write_behind: n_gpu>1, run_inference=false): its LEVER line says skipped:<rule>
        rep["prefetch"] = {**(fwd.get("prefetch") or {}), "engaged": int("prefetch" in rep["stream"]["engaged"]), "overlap_s": PS.overlap_s([j.span for j in launched])}; rep["write_behind"] = {**(fwd.get("write_behind") or {}), **{k: v for k, v in wb_post.items() if k != "streamed"}, "engaged": int("write_behind" in rep["stream"]["engaged"])}   # the streamed chain's census for the LEVER lines (pipeline_stream): the model process's waits / hand-offs; the writers' side is added once they end
        if fwd.get("error"):
            for n in ok_names:
                items[n]["error"] = f"forward: {fwd['error']}"
            ok_names = []
        else:
            active_p, world = fwd.get("n_gpu"), fwd.get("world")                       # fail-closed: the model process REPORTS the n_gpu it ran (and, under n_gpu > 1, the launcher's
            if steps[-1]["ok"] and (int(active_p or 0) != int(rep["n_gpu"]) or (int(rep["n_gpu"]) > 1 and int(world or 0) != int(rep["n_gpu"]))):   # rank census) — anything but the requested P is NOT ACTIVE by name, never a pass (a step that failed keeps its own named reason)
                emit(line("NOT ACTIVE", mode=rep["mode"], reason="n_gpu_mismatch", requested=rep["n_gpu"], active=active_p, world=world))
                for n in ok_names:
                    items[n]["ok"] = False; items[n]["error"] = f"forward: n_gpu_mismatch requested={rep['n_gpu']} active={active_p} world={world}"
                n_gpu_mismatch = True
                ok_names = []
            done = {(it["name"], it.get("seed")): it for it in fwd.get("items", [])}
            for n in list(ok_names):
                per_seed = []
                for sd in seeds[n]:
                    it = done.get((n, sd))
                    rec = {"seed": sd, "ok": bool(it and it.get("ok"))}
                    if it:
                        rec.update({k: it.get(k) for k in ("forward_s", "peak_mem_gb", "peak_reserved_gb", "n_tokens", "bucket", "fallbacks", "dead", "rng", "graph_reset", "graph_capture",
                                                             "templates_live", "templates_declared", "templates_slots", "templates_slots_live", "reason", "peak_mem_gb_ranks", "guards", "layout",
                                                             "diff_free", "diff_freed", "diff_freed_trimul_mib", "diff_freed_trimul", "phase_s") if k in it})   # phase_s: the item's trunk / sampler / confidence-head walls (the PHASE line)     # + the template slots featurised live / declared; big's per-item facts (present under mode big only); per-rank peaks (n_gpu > 1)
                    if not rec["ok"]:
                        rec["error"] = it.get("error") if it else f"no report entry ({step_reason(steps[-1])})"
                    per_seed.append(rec)
                    emit(line("ITEM", name=n, seed=sd, forward_s=rec.get("forward_s"), tokens=rec.get("n_tokens"), bucket=rec.get("bucket"),
                              peak_gb=rec.get("peak_mem_gb"), ok=int(rec["ok"]), reason=rec.get("reason"), templates=templates_word(rec), rng=rec.get("rng"), graph_reset=rec.get("graph_reset"), graph_capture=rec.get("graph_capture"),
                              **({"diff_free": rec.get("diff_free", 0)} if rep.get("big") else {}),
                              **({"diff_freed_trimul_mib": rec.get("diff_freed_trimul_mib")} if rep.get("big") and rec.get("diff_freed_trimul_mib") is not None else {}),
                              **({"peak_gb_ranks": rec.get("peak_mem_gb_ranks")} if int(rep.get("n_gpu") or 1) > 1 else {})))
                    if not rec["ok"]:                                          # a failed (input, seed) is named WITH its error: the exception's own words (a RowpairRefused names its opt-out)
                        emit(line("FAILED", item=n, seed=sd, error=str(rec["error"]).replace("\n", " ")[:400]))
                    if rec.get("phase_s"):                                    # ONE PHASE line per (input, seed) that ran: the forward's phase walls beside its total (report.phase_fields)
                        emit(line("PHASE", item=n, seed=sd, **phase_fields(rec["phase_s"], rec.get("forward_s"))))

                items[n]["forwards"] = per_seed
                bad = [r for r in per_seed if not r["ok"]]
                if bad:
                    items[n]["error"] = "forward: " + "; ".join(f"seed {r['seed']}: {r['error']}" for r in bad)
                    if len(bad) == len(per_seed):
                        ok_names.remove(n)                        # nothing ran: nothing to write; a partly failed item is written out and stays failed
        rep["census"] = fwd.get("census"); rep["fallback_events"] = fwd.get("fallback_events") or {}; rep["dead"] = fwd.get("dead") or {}
        rep["kernel_routes"] = fwd.get("kernel_routes") or {}; rep["compiled"] = fwd.get("compiled"); rep["dtk_swap_s"] = fwd.get("dtk_swap_s"); rep["big_record"] = fwd.get("big")
        emit(kernels_line(rep["kernel_routes"], stack.kernel_routes(), rep["census"]))   # the routed kernels: gate verdict + where each was imported from in the model process; the device's arch + cells
        lever_states = {}
        for lever in list(_registry.LEVERS) + list(_big.LEVER_ORDER):          # one activation-evidence line per lever per pred: on/off/skipped in this arm, and the run record that proves it
            text, lever_states[lever] = lever_line(lever, rep)
            emit(text)
        rp = fwd.get("rowpair") or {}                                                     # n_gpu > 1 only (absent at n_gpu = 1): rank 0's rowpair record
        if rp.get("trimul_rows") is not None:                                              # the core's fused TriMul rows account, for the run record
            rep["trimul_rows"] = rp["trimul_rows"]
        if rp.get("trimul_rows_line"):                                                     # its ONE line as the core rendered it in the model process, relayed verbatim
            emit(rp["trimul_rows_line"])
        events = []
        skipped = {l: st[1] for l, st in lever_states.items() if st[0] == "skipped" and l in _big.LEVER_ORDER}
        if skipped:                                                              # a big lever selected but not shown by the model process's record: the run is not the mode it claims
            events.append({"kind": "big", "skipped": skipped})
            emit(line("BIG", agree=0, skipped=",".join(sorted(skipped))))
        skipped = rep["package_levers_skipped"]                                                                # a package lever named-not-applied by rule is accounted, not missing
        if rep.get("package_levers_applied") is not None and tuple(rep["package_levers_applied"]) != tuple(l for l in (rep.get("package_levers") or ()) if l not in skipped):
            missing = [l for l in (rep.get("package_levers") or ()) if l not in rep["package_levers_applied"] and l not in skipped]
            events.append({"kind": "levers", "requested": list(rep.get("package_levers") or ()), "applied": list(rep["package_levers_applied"]), "missing": missing})
            emit(line("LEVERS", requested=",".join(rep.get("package_levers") or ()) or "none", applied=",".join(rep["package_levers_applied"]) or "none", agree=0, missing=",".join(missing) or "none", scope="package"))
        if rep.get("levers_applied") is not None and tuple(rep["levers_applied"]) != tuple(rep["levers"]):
            missing = [l for l in rep["levers"] if l not in rep["levers_applied"]]
            events.append({"kind": "levers", "requested": list(rep["levers"]), "applied": list(rep["levers_applied"]), "missing": missing})
            emit(line("LEVERS", requested=_modes.levers_label(rep["levers"]), applied=_modes.levers_label(rep["levers_applied"]), agree=0,
                      missing="+".join(missing) or "none"))
        acct = _registry.fallback_account([{**r, "name": n} for n, it in items.items() for r in (it.get("forwards") or [])], rep["fallback_events"])   # the kit's declared gates vs degradation, per item
        rep["fallbacks_expected"], unexpected = acct["expected"], acct["unexpected"]
        for lever in sorted(set(rep["fallback_events"]) | set(rep["dead"]) | set(unexpected)):   # every fallback is named on stderr, expected or not: the counters, the verdict
            lv = acct["levers"].get(lever) or {"observed": 0, "accounted": 0, "size_gated": {}}    # (expected=1: every call covered by a declared gate), the calls observed / accounted, the size-gated items
            emit(line("FALLBACK", lever=lever, **{census_word(k): v for k, v in (rep["fallback_events"].get(lever) or {}).items()},
                      expected=int(lever not in unexpected and lever not in rep["dead"]), dead=int(lever in rep["dead"]), observed=lv["observed"], accounted=lv["accounted"],
                      size_gated=",".join(f"{census_word(k).replace('fallback_', '', 1)}:{'+'.join(str(n) for n in names)}" for k, names in sorted(lv["size_gated"].items())) or "none"))
        if unexpected or rep["dead"]:                            # a kernel lever served the stock path outside its declared coverage, or died: not the mode it claims
            events.append({"kind": "fallback", "counts": unexpected, "dead": rep["dead"], "expected": rep["fallbacks_expected"]})
        if rep["mode"] == "off" and (rep["census"] is not None or fwd.get("kernels_enabled")):  # off: the stock proof includes an empty census (keyed on the mode: an ablated run of another mode may carry no kit lever and still its package levers / DTK)
            events.append({"kind": "stock", "census": rep["census"], "kernels_enabled": bool(fwd.get("kernels_enabled"))})
            emit(line("STOCK", census="present", kernels_enabled=int(bool(fwd.get("kernels_enabled"))), ok=0))
        if events:
            partial = {"events": events, "kinds": [e["kind"] for e in events], "allow_partial": bool(a.allow_partial)}
    post = None
    if post_rec is not None:                                                     # the writers ran beside the model process (joined above): their record and report, whatever became of ok_names
        steps.append(post_rec)
        post = read_report(os.path.join(work, "postprocess.json")) or {"items": []}; reports["postprocess"] = post
    elif ok_names and post_argv is not None:                                     # the sequential chain: the writers over the inputs that ran
        for n in ok_names:
            post_argv += ["--item", f"{n}={os.path.join(work, n)}={os.path.join(out, n)}"]
        steps.append(run_step("postprocess", post_argv, env_jax, os.path.join(work, "postprocess.log")))
        post = read_report(os.path.join(work, "postprocess.json")) or {"items": []}; reports["postprocess"] = post
    if ok_names and post is not None:
        done = {it["name"]: it for it in post.get("items", [])}
        for n in list(ok_names):
            it = done.get(n)
            if it and it.get("ok"):
                items[n].update(files=it.get("files"), ranking=it.get("ranking"), top=it.get("top"))
                items[n]["ok"] = "error" not in items[n]           # every seed ran and the writers wrote: ok; a failed seed keeps its reason
            else:
                items[n]["error"] = "; ".join(x for x in (items[n].get("error"), f"postprocess: {it.get('error') if it else 'no report entry (' + step_reason(steps[-1]) + ')'}") if x); ok_names.remove(n)
    chain_wall = round(time.monotonic() - t_chain, 1)                            # the chain's own wall: the DONE line's wall_s when the steps overlapped (their sum otherwise, as before)
    n_ok = sum(1 for it in items.values() if it["ok"])
    ok = n_ok == len(items) and all(s["ok"] for s in steps)
    # a degraded run is a named event, PARTIAL: the model process applied fewer levers than the mode asked for (kind levers), a kernel lever
    # served the stock path or died at run time (kind fallback: the kit's per-call census), or off ran with a kernel census (kind stock).
    # The outputs are kept, the run is not the mode it claims; rc 3 unless --allow-partial was passed (the DONE line says partial=allowed:<kinds>)
    rc = EXIT_OK if ok else EXIT_FAILED
    if partial is not None and ok and not partial["allow_partial"]:
        rc = EXIT_NOT_ACTIVE
        emit(line("PARTIAL", refused="+".join(partial["kinds"]), rc=EXIT_NOT_ACTIVE, opt_out="--allow-partial", stock="--mode off"))   # the mode is a contract: a lever that did not engage ends the run non-zero; the line names the one opt-out flag (and the stock route)
    if n_gpu_mismatch:                                                             # the axis dropped inside the model process: NOT ACTIVE (rc 3), never a pass, never --allow-partial's
        rc = EXIT_NOT_ACTIVE
    global _LAST_RUN
    _LAST_RUN = _run_record.build(rep, steps, [items[n] for n in names], inputs, sys.argv, ok, proof=proof, partial=partial, reports=reports)
    tc = templates_census([items[n] for n in names])                               # every (input, seed): template slots featurised live / declared; dummy_only = declared templates that all featurised dummy (ran untemplated)
    emit(line("TEMPLATES", items=tc["items"], declared=tc["declared"], live=tc["live"], refused=0, reason=None, dummy_only=tc["dummy_only"], none=tc["none"], real=f"{tc['slots_live']}/{tc['slots']}", form=templates_form(rep["n_gpu"])))   # refused=0 reason=None: no item is refused for its templates (the tokens the line has always carried)   # none = items whose input declares no templates (ran untemplated)
    emit(line("DONE", ok=int(ok), rc=rc, items=f"{n_ok}/{len(items)}", failed=",".join(n for n in names if not items[n]["ok"]) or "none",
              partial=(("allowed" if partial["allow_partial"] else "REFUSED") + ":" + "+".join(partial["kinds"])) if partial else "none",
              templates=f"{tc['live']}/{tc['declared']}",
              **dict(stack.n_gpu_fields(rep["n_gpu"])),                                   # the EXIT evidence of the resource axis: n_gpu=P sharding=rowpair|none (the core's words)
              wall_s=(chain_wall if rep["stream"]["engaged"] else round(sum(s["wall_s"] for s in steps), 1)), **({} if rc == EXIT_OK else {"work": work})))
    if rc == EXIT_OK:                                                              # a clean pred leaves the stock layout only: the hand-off files, logs and step reports go
        shutil.rmtree(work, ignore_errors=True)                                     # (a failed or partial one keeps them: the DONE line names the directory)
    return rc


TEMPLATE_FORMS = ("dense", "row_born")   # the TEMPLATES line's form= word: the stock embedder's dense N×N statement (one GPU) | each rank's own rows of the template pair inputs from the per-token template atoms (--n_gpu P > 1: rowpair_xfold template_rows_fn)


def templates_form(n_gpu: int) -> str:
    """``row_born`` when the pair representation is row-sharded (--n_gpu P > 1, sharding=rowpair), ``dense`` on one GPU."""
    return TEMPLATE_FORMS[1] if dict(stack.n_gpu_fields(int(n_gpu))).get("sharding") == "rowpair" else TEMPLATE_FORMS[0]


def cmd_check(a) -> int:
    if (refused := refused_mode_line(a)):
        emit(refused); return EXIT_NOT_ACTIVE
    no_compile_alias(a)
    try:
        rep = stack.check(a.mode, n_gpu=a.n_gpu)
    except _modes.UnsupportedMode as e:
        emit(f"check: {e}"); return EXIT_USAGE
    if a.json:
        print(dump(rep))
    else:
        from .report import activation_line
        emit(activation_line(rep))
    return EXIT_OK if rep["active"] else EXIT_NOT_ACTIVE


def cmd_exports(a) -> int:
    """``exports``: the shell lines configs/<gpu>.env evaluates (stack.config_exports: one presence check per required deployment variable —
    the two interpreters and the fork checkout, stack.REQUIRED; an unset one is named, rc 3). Printed on stdout, nothing else; the producers
    gate ran first (__main__)."""
    print(stack.config_exports())
    return EXIT_OK


_LAST_RUN = None      # the record of the last `pred` in this process (run_record.build): what its printed lines were rendered from


def last_run():
    """The last `pred`'s record in this process (activation report, stock proof, partial verdict, steps, items; None before any pred) — for
    callers driving the package from Python; the printed lines carry the same account."""
    return _LAST_RUN


# ---- warm: the kit's one-time kernel and cache setup, done ahead of the first pred --------------------------------------------------------
# The first model process on a cold cache root compiles the kit's Triton kernels, autotunes the tunable ones, compiles the `compile` lever's
# glue (Inductor) and stores all of it under AF3_TORCH_CACHE_ROOT (stack.cache_dirs: <root>/triton, <root>/inductor, <root>/jax); every later
# process at the same padded length loads it in seconds. `warm` pays that once, ahead of time, through the pred's own three processes — no
# separate code path: a pred with the protocol knobs turned down (WARM_KNOBS) over inputs that need no data pipeline.
WARM_ALL = "all"                                   # `warm --mode all`: every mode of the table in turn (modes.MODES), one short pred each
WARM_SIZES = (448, 832, 1216)                      # the token counts of warm's synthetic inputs: the kernel-tile lengths a 400 / 800 / 1200-token input pads to under fast / big (opt_core.shape_policy, 64-token tiles); one chain of exactly that many residues, so off / exact (no padding) run the same three lengths
WARM_KNOBS = {"num_recycles": 1, "diffusion_steps": 8}   # the cheapest protocol that still reaches every compiled / autotuned / captured path once per length: two trunk passes (the recycle's previous-pass embedding included), eight denoising steps (the step graph runs one step eagerly — where the glue compiles — and captures on the next; its two retries ride the steps after), the stock sample count (the batched sampler's shapes are the five-sample ones a default pred runs); named on the pred's SETTINGS line as any reduced protocol is
WARM_RESIDUES = "AQVINTFDGVADYLQTYHKLPDNYITKSEAQALGWVASKGNLADVAPGKSIGGDIFSNREGKLPGKSGRTWREADINYTSGFRNSDRILYSSDWLIYKTTDHYQTFTKIR"   # the example input's first chain (inputs/1BRS.json, 110 residues): warm's synthetic chains repeat it to length, so the atoms per token are a real protein's
WARM_PREFIX = "warm"                               # the synthetic inputs' stems (warm0448 / warm0832 / warm1216): the item names on the pred's ITEM lines


def warm_modes(mode: Optional[str]) -> List[str]:
    """The modes `warm` runs, in order: ``all`` = every mode of the table (modes.MODES); a mode name = that one; None = the mode a pred without
    ``--mode`` runs (AF3_TORCH_OPT, else the package default). An unknown name raises UnsupportedMode (warm's usage refusal, rc 2)."""
    if mode == WARM_ALL:
        return list(_modes.MODES)
    mode = (mode or _modes.mode_from_env()).strip()
    if mode in _modes.REFUSED_MODES:
        raise _modes.UnsupportedMode(_modes.REFUSED_MODES[mode])
    if mode not in _modes.MODES:
        raise _modes.UnsupportedMode(f"unknown mode {mode!r}; modes are {' | '.join(_modes.MODES)} (or {WARM_ALL}: each in turn)")
    return [mode]


def warm_sequence(n_tokens: int) -> str:
    """A protein sequence of exactly ``n_tokens`` standard residues (one token each): WARM_RESIDUES repeated and cut to length."""
    n = int(n_tokens)
    if n < 1:
        raise ValueError(f"warm: a synthetic input needs at least one residue, got {n_tokens!r}")
    return (WARM_RESIDUES * (-(-n // len(WARM_RESIDUES))))[:n]


def warm_fold_input(n_tokens: int, seed: int = 1) -> dict:
    """One synthetic AlphaFold 3 fold input of exactly ``n_tokens`` tokens in the example's own form (inputs/1BRS.json): a single protein
    chain with a query-only unpaired MSA, an empty paired MSA and no templates — nothing for the data pipeline to search — and one model seed."""
    seq = warm_sequence(n_tokens)
    return {"name": f"{WARM_PREFIX}{int(n_tokens):04d}", "dialect": "alphafold3", "version": 2,
            "sequences": [{"protein": {"id": "A", "sequence": seq, "modifications": [], "unpairedMsa": f">query\n{seq}\n", "pairedMsa": "", "templates": []}}],
            "modelSeeds": [int(seed)]}


def warm_inputs(dir_: str, sizes=WARM_SIZES) -> List[str]:
    """Write warm's synthetic fold inputs under ``dir_``, one JSON per distinct size in ascending order (``<dir_>/warm<size>.json``); their paths."""
    os.makedirs(dir_, exist_ok=True)
    paths = []
    for n in sorted({int(s) for s in sizes}):
        fi = warm_fold_input(n)
        p = os.path.join(dir_, fi["name"] + ".json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(fi, f)
        paths.append(p)
    return paths


def cache_census(root: Optional[str]) -> tuple:
    """``(files, bytes)`` under the cache root (stack.cache_root(): the Triton / Inductor / JAX compile caches and the weights digest memo) — the
    WARM line's before / after counts. A missing root is (0, 0); a file that vanishes mid-walk (another process's cache write) is not counted."""
    files = size = 0
    if root and os.path.isdir(root):
        for dp, _, fs in os.walk(root):
            for f in fs:
                try:
                    size += os.path.getsize(os.path.join(dp, f))
                except OSError:
                    continue
                files += 1
    return files, size


def cache_word(c: tuple) -> str:
    """A cache census as the WARM line's word: ``<files>files/<megabytes>MB``."""
    return f"{int(c[0])}files/{c[1] / 1e6:.1f}MB"


def warm_pred_args(mode: str, inputs: List[str], output_dir: str, quiet: bool = False) -> argparse.Namespace:
    """The namespace of warm's one pred under ``mode``, built by the pred parser itself (every stock flag at the pred's own default): the data
    pipeline off (``--norun_data_pipeline``: the inputs carry their MSAs), WARM_KNOBS, the inputs as ``--json_path``, the output dir."""
    argv = ["pred", "--mode", mode, "--output_dir", output_dir, "--norun_data_pipeline"] + (["--quiet"] if quiet else [])
    for k, v in WARM_KNOBS.items():
        argv += [f"--{k}", str(v)]
    for p in inputs:
        argv += ["--json_path", p]
    return build_parser().parse_args(argv)


def warm_sizes_word(run: Optional[dict], requested: Optional[List[int]]) -> str:
    """``sizes=`` on the WARM line: the synthetic inputs' token counts (``requested``: exact by construction, tile multiples — the lengths every
    mode ran them at); for named inputs the padded token counts they ran at (the featuriser's bucket per input, in input order — the ITEM
    lines' ``bucket=``) when the pred got that far, else ``named``."""
    if requested:
        return ",".join(str(int(s)) for s in requested)
    items = (run or {}).get("items") or []
    got = [it.get("bucket") or it.get("n_tokens") for it in items]
    if got and all(g is not None for g in got):
        return ",".join(str(int(g)) for g in got)
    return "named"


def warm_rc(rcs: List[int]) -> int:
    """warm's exit code over its modes' preds: 0 when every one succeeded; else a failed pred (1) first, then not-active / partial (3), then
    usage (2), then any other code a pred returned — total accounting, the per-mode codes are on the WARM lines."""
    for code in (EXIT_FAILED, EXIT_NOT_ACTIVE, EXIT_USAGE):
        if code in rcs:
            return code
    return next((r for r in rcs if r != EXIT_OK), EXIT_OK)


def cmd_warm(a) -> int:
    """``warm [--mode M|all] [--json_path P …] [--input_dir D] [P …]``: one short pred per mode (warm_modes) over the synthetic inputs
    (warm_inputs) or the named ones, each followed by ONE WARM line: mode, the sizes that ran, the cache root's file count and megabytes before
    and after, the files this mode added, the wall seconds, the pred's rc. The synthetic inputs and every output live under one temporary
    directory (``$TMPDIR/af3_torch_warm_*``) that is removed at the end, whatever happened; the caches stay. rc: warm_rc over the modes."""
    try:
        wanted = warm_modes(a.mode)
    except _modes.UnsupportedMode as e:
        emit(f"warm: {e}"); return EXIT_USAGE
    named = [os.path.abspath(p) for p in list(a.inputs or []) + input_paths(a)]
    missing = [p for p in named if not os.path.isfile(p)]
    if missing:
        emit(f"warm: input not found: {missing}"); return EXIT_USAGE
    global _LAST_RUN
    root = stack.cache_root()
    emit(stack.cache_seed_line(stack.seed_cache_root(root)))                              # the environment's pre-filled tree first, when it carries one and the root is cold: the WARM lines' cache_before then counts it and new_files is what this box still had to compile
    tmp = tempfile.mkdtemp(prefix="af3_torch_warm_")
    rcs: List[int] = []
    try:
        inputs = named or warm_inputs(os.path.join(tmp, "inputs"))
        requested = None if named else [int(item_name(p)[len(WARM_PREFIX):]) for p in inputs]
        for m in wanted:
            before = cache_census(root); t0 = time.monotonic(); prev = _LAST_RUN
            rc = cmd_pred(warm_pred_args(m, inputs, os.path.join(tmp, "out", m), quiet=a.quiet))
            run = _LAST_RUN if _LAST_RUN is not prev else None                              # the record of THIS pred (None when it stopped before the chain: usage / not active)
            after = cache_census(root)
            emit(line("WARM", mode=m, sizes=warm_sizes_word(run, requested), inputs="named" if named else "synthetic", cache_root=root,
                      cache_before=cache_word(before), cache_after=cache_word(after), new_files=after[0] - before[0], seconds=round(time.monotonic() - t0, 1), rc=rc))
            rcs.append(rc)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)                                                    # warm's own temporary directory only; the caches it filled stay
    return warm_rc(rcs)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="af3-torch-opt", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--mode", default=None, help=f"{' | '.join(_modes.MODES)} (default: $AF3_TORCH_OPT, else {_modes.DEFAULT_MODE})")
        p.add_argument("--n_gpu", type=int, default=_modes.N_GPU_DEFAULT, help=f"GPUs of this box the pair stack is row-sharded over (mode big only; {'|'.join(str(x) for x in _modes.N_GPU_SUPPORTED)}; default 1 = one GPU; never auto-detected: fewer visible GPUs than P is refused by name)")
        p.add_argument("--quiet", action="store_true")
        p.add_argument("--no-compile", dest="no_compile", action="store_true", help=f"run the mode without the kit's torch.compile lever (`compile`: the step glue compiled on first use at each token length): an alias of {_modes.ENV_LEVERS_OFF}=compile — the ACTIVE line says compile=off:user; accepted and without effect under off / exact, which never compile (compile=off:mode)")

    p = sub.add_parser("pred", help="predict every fold-input JSON: featurise -> forward -> postprocess"); common(p)
    p.add_argument("--json_path", action="append", help="a fold-input JSON (repeatable)")
    p.add_argument("--input_dir", help="every *.json in the dir")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model_dir", default=None, help="the stock CLI's --model_dir: the directory holding the weights file (default: $AF3_TORCH_PARAMS_DIR)")
    for name, what in (("run_data_pipeline", "run the fork's data pipeline (MSA / template search) on each fold input first — the stock CLI's flag and default (true); chains that already carry unpairedMsa / pairedMsa / templates are skipped as upstream does"),
                       ("run_inference", "run the model — the stock CLI's flag and default (true); false = data pipeline only: <name>_data.json is written per input, nothing else")):
        p.add_argument(f"--{name}", nargs="?", const=True, default=True, type=bool_flag(name), metavar="true|false", help=what)
        p.add_argument(f"--no{name}", dest=name, action="store_const", const=False, help=f"the stock CLI's --no{name} (= --{name}=false)")
    for name in STOCK_PIPELINE_STRINGS:                                                  # the stock CLI's data-pipeline flags, passed through unchanged (default: upstream's, applied in the fork's interpreter)
        if name == "db_dir":
            p.add_argument("--db_dir", action="append", default=None, help="the stock CLI's --db_dir (repeatable, searched in order; default upstream's $HOME/public_databases)")
        else:
            p.add_argument(f"--{name}", default=None, help=f"the stock CLI's --{name} (default upstream's: ${{DB_DIR}}/… for databases, the binary on PATH, 2021-09-30 for --max_template_date)")
    for name in STOCK_PIPELINE_INTS:
        p.add_argument(f"--{name}", type=int, default=None, help=f"the stock CLI's --{name} (default: upstream's, min(cpu_count, 8))")
    p.add_argument("--num_diffusion_samples", type=int, default=_modes.STOCK_NUM_DIFFUSION_SAMPLES, help="diffusion samples per seed — the stock CLI's flag and default (xfold run_alphafold.py --num_diffusion_samples, 5); 1 = the single-sample form, not upstream's protocol")
    p.add_argument("--fastnn", nargs="?", const=True, default=None, type=fastnn_flag, metavar="true|false", help="xfold's shipped Triton fastnn kernels (layer norm / dot-product attention / gated linear unit) — the stock CLI's flag, default on: served under --mode off | exact (their base); fast / big serve those modules with the kit's kernel levers and refuse an explicit --fastnn by name")
    p.add_argument("--nofastnn", dest="fastnn", action="store_const", const=False, help="the stock CLI's --nofastnn: the eager port (xfold's torch ops) instead of its fastnn kernels — refused by name under exact (whose byte-equality is stated on the fastnn base); no effect under fast / big, whose kernel levers serve those modules")
    p.add_argument("--num_recycles", type=int, default=None, help="the model constructor's num_recycles (xfold AlphaFold3(num_recycles=10): 10 recycles = 11 trunk passes); absent = the constructor's default; upstream's CLI has no flag for it")
    p.add_argument("--diffusion_steps", type=int, default=None, help="the model constructor's diffusion_steps (xfold AlphaFold3(diffusion_steps=200)); absent = the constructor's default; upstream's CLI has no flag for it")
    p.add_argument("--allow-partial", action="store_true", help="accept a degraded lever set (the model process applied fewer levers than the mode names): rc 0 instead of 3, said on the DONE line (partial=allowed:<kinds>)")
    p.set_defaults(fn=cmd_pred)
    p = sub.add_parser("stock", help="the stock route: xfold's own CLI (run_alphafold.py, the pinned archive's bytes) at its shipped defaults on the composed stock venv ($AF3_TORCH_STOCK_PY); no mode, no lever")
    stock_cli.add_args(p); p.set_defaults(fn=lambda a: stock_cli.run(a, run_step))
    p = sub.add_parser("check", help="the dry run: resolve + gate, nothing launched"); common(p)
    p.add_argument("--json", action="store_true"); p.set_defaults(fn=cmd_check)
    p = sub.add_parser("warm", help="the kit's one-time kernel and cache setup ahead of the first pred: one short pred per mode over synthetic 448 / 832 / 1216-token inputs (or the named fold inputs), filling the Triton / Inductor / JAX caches under AF3_TORCH_CACHE_ROOT; one WARM line per mode")
    p.add_argument("--mode", default=None, help=f"{' | '.join(_modes.MODES)} | {WARM_ALL} (default: $AF3_TORCH_OPT, else {_modes.DEFAULT_MODE}; {WARM_ALL} = every mode in turn)")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--json_path", action="append", help="warm on this fold-input JSON instead of the synthetic inputs (repeatable): the caches are then keyed by its own padded lengths")
    p.add_argument("--input_dir", help="every *.json in the dir, likewise")
    p.add_argument("inputs", nargs="*", metavar="JSON", help="fold-input JSONs to warm on (the positional form of --json_path)")
    p.set_defaults(fn=cmd_warm)
    p = sub.add_parser("exports", help="print the shell lines configs/<gpu>.env evaluates (one presence check per required deployment variable); nothing else")
    p.set_defaults(fn=cmd_exports)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
