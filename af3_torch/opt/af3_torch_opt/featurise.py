#!/usr/bin/env python
"""featurise.py — fold-input JSON -> [the stock CLI's data pipeline] -> per-seed batch.npz (+ batch.pkl) with the AF3 fork's own featuriser
(alphafold3.data.featurisation), the kit's featurisation recipe (featurise_input(buckets, verbose=False, resolve_msa_overlaps=False),
remove_invalidly_typed_feats, np.savez of np.asarray of every numeric feature).

The data pipeline is upstream's, passed through: `--run_data_pipeline 1` (the stock CLI's default) runs alphafold3.data.pipeline.DataPipeline
on each fold input first, configured from the stock CLI's flags (--db_dir, the nine database paths, the five HMMER binary paths, --max_template_date,
--jackhmmer_n_cpu / --nhmmer_n_cpu; a flag not given takes the fork CLI's own default) exactly as the fork's run_alphafold.main builds its
DataPipelineConfig; chains that already carry their MSAs / templates are skipped by the pipeline itself. `--run_inference 0` (the stock CLI's
data-pipeline-only run) writes <sanitised name>_data.json into the item's directory with the fork's write_fold_input_json and featurises nothing.

Runs on the JAX venv's interpreter (AF3_TORCH_JAX_PY). Featurisation only: no model, no parameters, no GPU (the caller sets JAX_PLATFORMS=cpu; the report records
the JAX backend seen by this process when the fork imported jax).

    python featurise.py --item NAME=<fold_input.json>=<work_dir> [--item ...] [--run_data_pipeline 0|1 --db_dir … --run_inference 0|1 --repo_dir …] \
        [--buckets none | tile:64 | 256,512,…] [--stream 0|1] --report <featurise.json>

Seeds: the fold input's own modelSeeds, all of them; featurise_input returns one example per rng seed
(alphafold3.data.featurisation.featurise_input:104); they are indexed by seed here.
Per item and seed s: <work_dir>/seed-<s>/batch.npz (numeric features; what af3_torch_api.batch_from_npz reads) and
<work_dir>/seed-<s>/batch.pkl (pickle {'name': the fold input's name field, 'seed', 'example': the featurised example BEFORE
remove_invalidly_typed_feats (what alphafold3.model.model.Model.get_inference_result needs), 'fold_input_json': path,
'fold_input': the folding_input.Input after the data pipeline (what run_alphafold.write_fold_input_json writes), 'buckets'}).
Report: {items: [{name, json, work_dir, ok, seeds, n_tokens, bucket, featurise_s, pipeline_s, msa_rows, fold_input_name, sanitised_name, error?}],
ok, run_data_pipeline, run_inference, buckets, jax_backend, stream}. Exit 0 only when every item is ok; a failed item is named and the others still run.
`--stream 1` (the wrapper's package lever `prefetch`: the model process runs beside this one): every fold input is loaded first and ONE
`SEEDS {name: [seeds]}` line printed (the wrapper launches the model process on it), then each item featurises as before and hands each seed
over the moment its two files are closed (pipeline_stream: seed-<s>/batch.ready, then featurise.ok | featurise.failed per item) — the same
bytes, written by the same statements, marked complete for a reader that is already waiting.
"""
import argparse
import json
import os
import pickle
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_stream as PS  # noqa: E402 — the streamed chain's hand-off markers (standard library only)

# The stock CLI's data-pipeline flags `pred` passes through (cli.STOCK_PIPELINE_STRINGS / _INTS name the same set): each is also a flag of the
# fork's own run_alphafold.py, whose DataPipeline is the one that runs here. A flag not given takes the fork CLI's own default, read from its
# flag table — nothing is restated (pipeline_config).
PIPELINE_FLAGS = ("db_dir", "jackhmmer_binary_path", "nhmmer_binary_path", "hmmalign_binary_path", "hmmsearch_binary_path", "hmmbuild_binary_path",
                  "small_bfd_database_path", "mgnify_database_path", "uniprot_cluster_annot_database_path", "uniref90_database_path",
                  "ntrna_database_path", "rfam_database_path", "rna_central_database_path", "pdb_database_path", "seqres_database_path",
                  "max_template_date", "jackhmmer_n_cpu", "nhmmer_n_cpu")


def pipeline_config(args, ra):
    """alphafold3.data.pipeline.DataPipelineConfig built as the fork's run_alphafold.main builds it: every field from the flag of the same name —
    the value given on the command line, else the fork CLI's own default (ra.flags.FLAGS[name].default) — with main's two conversions: database
    paths expanded over the --db_dir list (the first directory holding the file, as the fork's CLI picks; plain ${DB_DIR} substitution as
    xfold's CLI does when none holds it) and --max_template_date parsed as a date."""
    import dataclasses, datetime, string
    from alphafold3.data import pipeline
    F = ra.flags.FLAGS
    db_dirs = tuple(args.db_dir) if args.db_dir else tuple(F["db_dir"].default)
    kw = {}
    for f in dataclasses.fields(pipeline.DataPipelineConfig):
        given = getattr(args, f.name, None)
        if given is not None:
            v = given
        elif f.name in F:
            v = F[f.name].default
        elif f.default is not dataclasses.MISSING:
            v = f.default
        else:
            raise SystemExit(f"featurise.py: the fork's DataPipelineConfig field {f.name!r} has no flag and no default (run_alphafold.py changed shape)")
        if f.name.endswith("_database_path") and isinstance(v, str):   # xfold's CLI substitutes ${DB_DIR} (string.Template, no check); the fork's searches its --db_dir
            try:                                                       # list for the file and refuses when none holds it — even for inputs that need no search.
                v = ra.replace_db_dir(v, list(db_dirs))                # The fork's pick when the file exists somewhere; xfold's substitution (the first --db_dir)
            except FileNotFoundError:                                  # otherwise, so a missing database fails the search that needs it, by name, and nothing else
                v = string.Template(v).substitute(DB_DIR=db_dirs[0])
        if f.name == "max_template_date" and isinstance(v, str):
            v = datetime.date.fromisoformat(v)
        kw[f.name] = v
    return pipeline.DataPipelineConfig(**kw)


def parse_item(spec):
    """'NAME=<fold_input.json>=<work_dir>' -> (name, json_path, work_dir)."""
    parts = spec.split("=", 2)
    if len(parts) != 3 or not all(parts):
        raise ValueError(f"--item must be NAME=<fold_input.json>=<work_dir>, got {spec!r}")
    return parts[0], parts[1], parts[2]


NO_BUCKETS = "none"          # --buckets none: no token padding — featurise_input(buckets=None) pads each input to its own token count (what the stock CLI does when it names no buckets)
TILE_BUCKETS = "tile:"       # --buckets tile:<k>: every input pads to the next multiple of k tokens, at any size (an open-ended row: featurise_input bisects it like a list)
TILE_ROW_LEN = 8192          # the open-ended row is materialised lazily as range(k, k * TILE_ROW_LEN + 1, k): 524288 tokens at k = 64, past anything the featuriser itself can lay out


def parse_buckets(s):
    """``none`` -> None (no padding); ``tile:<k>`` -> range(k, k*TILE_ROW_LEN+1, k) (the next multiple of k, any size); else the explicit increasing row."""
    s = s.strip()
    if s.lower() == NO_BUCKETS:
        return None
    if s.lower().startswith(TILE_BUCKETS):
        k = int(s[len(TILE_BUCKETS):])
        if k <= 0:
            raise ValueError(f"--buckets {s!r}: the tile must be a positive integer")
        return range(k, k * TILE_ROW_LEN + 1, k)
    b = tuple(int(x) for x in s.split(",") if x)
    if not b or list(b) != sorted(set(b)):
        raise ValueError(f"--buckets must be strictly increasing integers, {TILE_BUCKETS}<k> or {NO_BUCKETS!r}, got {s!r}")
    return b


def buckets_word(buckets):
    """The report's record of a bucket row: None, 'tile:<k>' for the open-ended row, else the list."""
    if buckets is None:
        return None
    if isinstance(buckets, range):
        return f"{TILE_BUCKETS}{buckets.step}"
    return list(buckets)


def load_one_fold_input(json_path):
    """The single fold input of a file (a file with several is refused rather than silently truncated)."""
    from alphafold3.common import folding_input
    fis = list(folding_input.load_fold_inputs_from_path(json_path))
    if len(fis) != 1:
        raise ValueError(f"{json_path}: {len(fis)} fold inputs in the file; exactly one is featurised per item")
    return fis[0]


def featurise_one(json_path, work_dir, buckets, pipeline=None, run_inference=True, write_data_json=None, fold_input=None, on_seed=None):
    """[Run the data pipeline on, then] featurise one fold input for its seeds; writes seed-<s>/batch.npz + batch.pkl into work_dir; returns the
    report row fields. run_inference=False: write_data_json(fold_input, work_dir) and no featurisation (the stock CLI's data-pipeline-only run).
    ``fold_input``: the input already loaded (--stream: every input is loaded once, up front, for the SEEDS line — a second load of a seedless
    server-dialect input would draw another random seed); ``on_seed(seed_dir)``: called once a seed's two files are closed (--stream: the hand-off marker)."""
    from alphafold3.constants import chemical_components
    from alphafold3.data import featurisation
    from alphafold3.model.components import utils

    fi = fold_input if fold_input is not None else load_one_fold_input(json_path)
    t_pipe = None
    if pipeline is not None:                                       # run_alphafold.process_fold_input: the pipeline first, on the fold input as read
        t0 = time.time(); fi = pipeline.process(fi); t_pipe = round(time.time() - t0, 3)
    seeds = tuple(int(s) for s in fi.rng_seeds)
    if not seeds:
        raise ValueError(f"{json_path}: no rng seeds (modelSeeds empty)")
    fi1 = fi
    if not run_inference:
        os.makedirs(work_dir, exist_ok=True); write_data_json(fi1, work_dir)
        return dict(seeds=list(seeds), pipeline_s=t_pipe, featurise_s=None, fold_input_name=fi.name, sanitised_name=fi1.sanitised_name(), data_json=os.path.join(work_dir, f"{fi1.sanitised_name()}_data.json"))
    t0 = time.time()
    examples = featurisation.featurise_input(fold_input=fi1, buckets=buckets, ccd=chemical_components.Ccd(user_ccd=fi.user_ccd),
                                             verbose=False, resolve_msa_overlaps=False)
    t_feat = time.time() - t0
    if len(examples) != len(seeds):
        raise RuntimeError(f"featurise_input returned {len(examples)} examples for {len(seeds)} seeds")
    row = dict(seeds=list(seeds), pipeline_s=t_pipe, featurise_s=round(t_feat, 3), fold_input_name=fi.name, sanitised_name=fi1.sanitised_name())
    for s, ex in zip(seeds, examples):
        n_tok = int(np.asarray(ex["seq_length"]))
        bucket = int(np.asarray(ex["token_index"]).shape[0])
        msa_rows = [int(x) for x in np.asarray(ex["msa"]).shape]
        ex_num = utils.remove_invalidly_typed_feats(ex)
        d = os.path.join(work_dir, f"seed-{s}")
        os.makedirs(d, exist_ok=True)
        np.savez(os.path.join(d, "batch.npz"), **{k: np.asarray(v) for k, v in ex_num.items()})
        with open(os.path.join(d, "batch.pkl"), "wb") as f:
            pickle.dump({"name": fi.name, "seed": int(s), "example": ex, "fold_input_json": os.path.abspath(json_path),
                         "fold_input": fi1, "buckets": buckets_word(buckets)}, f, protocol=pickle.HIGHEST_PROTOCOL)
        if on_seed is not None:                                     # --stream: both files of the seed are closed — hand it over
            on_seed(d)
        row.update(n_tokens=n_tok, bucket=bucket, msa_rows=msa_rows, n_numeric_feats=len(ex_num), n_feats=len(ex))
    return row


def preload(items):
    """--stream: every item's fold input loaded ONCE, before any is featurised → ({name: fold_input}, {name: failed row}). A file that does not load
    is the item's named failure exactly as featurise_one would have named it; it is absent from the SEEDS line (nothing downstream waits for it)."""
    loaded, failed = {}, {}
    for name, json_path, work_dir in items:
        try:
            loaded[name] = load_one_fold_input(json_path)
        except Exception as e:  # named per item, as in run()
            failed[name] = dict(name=name, json=json_path, work_dir=work_dir, ok=False, error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-4000:])
    return loaded, failed


def run(items, buckets, report_path, pipeline=None, run_inference=True, write_data_json=None, stream=False):
    """Featurise every item (total accounting: each row ok or failed with its named reason); returns the report dict. ``stream``: the inputs are
    loaded first and the SEEDS line printed, each seed is handed over as it completes and each item marked ok / failed (pipeline_stream)."""
    rows = []
    loaded, failed = preload(items) if stream else ({}, {})
    if stream:                                                          # the model process is launched on this line: every input that loaded, with its seeds
        PS.exit_with_parent()
        print(PS.seeds_line({name: [int(x) for x in loaded[name].rng_seeds] for name, _, _ in items if name in loaded}), flush=True)
    for name, json_path, work_dir in items:
        row = dict(name=name, json=json_path, work_dir=work_dir, ok=False)
        try:
            if name in failed:
                row = failed.pop(name); raise _Preloaded()
            row.update(featurise_one(json_path, work_dir, buckets, pipeline=pipeline, run_inference=run_inference, write_data_json=write_data_json,
                                     fold_input=loaded.get(name), on_seed=(lambda d: PS.touch(os.path.join(d, PS.BATCH_READY))) if stream else None))
            row["ok"] = True
        except _Preloaded:      # the load failure recorded by preload(), named as featurise_one names it
            pass
        except Exception as e:  # named per item; the run continues with the next item
            row["error"] = f"{type(e).__name__}: {e}"
            row["traceback"] = traceback.format_exc()[-4000:]
        if stream:                                                      # the item's terminal marker: complete (every seed handed over) or withdrawn downstream
            PS.touch(os.path.join(work_dir, PS.ITEM_OK if row["ok"] else PS.ITEM_FAILED))
        print(json.dumps({k: v for k, v in row.items() if k != "traceback"}), flush=True)
        rows.append(row)
    jax_backend = None
    if "jax" in sys.modules:
        try:
            jax_backend = sys.modules["jax"].default_backend()
        except Exception as e:  # reported, never hidden
            jax_backend = f"unavailable: {type(e).__name__}: {e}"
    report = dict(items=rows, ok=all(r["ok"] for r in rows), run_data_pipeline=int(pipeline is not None), run_inference=int(bool(run_inference)), buckets=buckets_word(buckets),
                  jax_backend=jax_backend, jax_platforms_env=os.environ.get("JAX_PLATFORMS"), stream=int(bool(stream)))
    if report_path:
        os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=1)
    return report


class _Preloaded(Exception):
    """An item whose fold input failed to load in preload(): its row is already written."""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--item", action="append", required=True, help="NAME=<fold_input.json>=<work_dir> (repeatable)")
    ap.add_argument("--stream", type=int, choices=(0, 1), default=0, help="1 = the streamed chain (package lever prefetch): load every input, print the SEEDS line, hand each seed over as it completes (pipeline_stream)")
    ap.add_argument("--buckets", default=NO_BUCKETS, help="none (xfold as shipped: no padding) | tile:<k> (every multiple of k) | a comma list (bisected like alphafold3's buckets)")
    ap.add_argument("--report", required=True, help="report json path")
    ap.add_argument("--run_data_pipeline", type=int, choices=(0, 1), default=1, help="the stock CLI's --run_data_pipeline (default 1, upstream's)")
    ap.add_argument("--run_inference", type=int, choices=(0, 1), default=1, help="the stock CLI's --run_inference: 0 = data pipeline only, <name>_data.json written into the item's dir")
    ap.add_argument("--repo_dir", default=None, help="the fork checkout holding run_alphafold.py (its write_fold_input_json writes <name>_data.json under --run_inference 0)")
    for name in PIPELINE_FLAGS:                                          # the stock CLI's data-pipeline flags; absent = the fork CLI's own default (pipeline_config)
        if name == "db_dir":
            ap.add_argument("--db_dir", action="append", default=None, help="database directory (repeatable: searched in order, as the fork's CLI does); default the fork CLI's ($HOME/public_databases)")
        elif name.endswith("_n_cpu"):
            ap.add_argument(f"--{name}", type=int, default=None, help=f"the stock CLI's --{name}; default the fork CLI's (min(cpu_count, 8))")
        else:
            ap.add_argument(f"--{name}", default=None, help=f"the stock CLI's --{name}; default the fork CLI's")
    args = ap.parse_args(argv)
    items = [parse_item(s) for s in args.item]
    if not args.run_inference and not args.run_data_pipeline:
        raise SystemExit("featurise.py: At least one of --run_inference or --run_data_pipeline must be set to true.")   # run_alphafold.main's own rule
    pipeline = write_data_json = None
    if args.run_data_pipeline or not args.run_inference:              # the fork's own run_alphafold module (loaded as postprocess.py loads it): its flag defaults configure the
        if not args.repo_dir:                                         # pipeline, its write_fold_input_json writes <name>_data.json
            raise SystemExit("featurise.py: --run_data_pipeline 1 / --run_inference 0 need --repo_dir (the fork checkout holding run_alphafold.py)")
        from postprocess import load_run_alphafold
        ra = load_run_alphafold(args.repo_dir)
        if args.run_data_pipeline:                                    # run_alphafold.main: DataPipelineConfig from the flags; process_fold_input: DataPipeline(config).process(fold_input)
            from alphafold3.data import pipeline as _pipeline
            pipeline = _pipeline.DataPipeline(pipeline_config(args, ra))
        if not args.run_inference:
            write_data_json = ra.write_fold_input_json
    report = run(items, parse_buckets(args.buckets), args.report, pipeline=pipeline, run_inference=bool(args.run_inference), write_data_json=write_data_json,
                 stream=bool(args.stream) and bool(args.run_inference))   # --run_inference 0 featurises nothing: nothing to hand over
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
