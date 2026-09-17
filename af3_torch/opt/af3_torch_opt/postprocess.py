#!/usr/bin/env python
"""postprocess.py — per-seed result.npz (the torch forward's arrays) + batch.pkl (the featurised example) -> the AF3 fork's own output
files, written by the fork's own writers exactly as run_alphafold.py writes them: alphafold3.model.model.Model.get_inference_result
(structure + confidence metrics per sample) -> run_alphafold.ResultsForSeed per seed -> run_alphafold.write_fold_input_json +
run_alphafold.write_outputs (post_processing.write_output per sample: seed-<s>_sample-<k>/<job>_seed-<s>_sample-<k>_{model.cif,
confidences.json,summary_confidences.json}; the best-ranked copy <job>_model.cif / _confidences.json / _summary_confidences.json;
<job>_ranking_scores.csv over every (seed, sample); TERMS_OF_USE.md; <sanitised json name>_data.json).
<out_dir> corresponds to run_alphafold.py's <output_dir>/<sanitised_name>/; <job> = the item NAME (write_outputs takes job_name
explicitly — run_alphafold passes fold_input.sanitised_name(); write_fold_input_json always names by the JSON name).

Runs on the JAX venv's interpreter (AF3_TORCH_JAX_PY) on CPU (JAX_PLATFORMS=cpu is fine: the writers are numpy).

    python postprocess.py --item NAME=<work_dir>=<out_dir> [--item ...] --report <postprocess.json> --repo_dir <the fork checkout> [--follow <work root>]

--follow <work root> (the wrapper's package lever `write_behind`: this process runs BESIDE the model process): each item, in order, is written
the moment the model process has handed over every seed's result.npz (pipeline_stream: seed-<s>/result.ready) — while the next item is on the
GPU — or when the model process is gone; an item that never received a result (its featurisation or every forward failed) is withdrawn, as the
sequential chain never hands it here. The same functions write the same files; the report's rows carry `follow_wait_s` / `early`.

<work_dir>/seed-<s>/batch.pkl: featurise.py's pickle ({'name','seed','example','fold_input_json','fold_input',...}); every seed-<s>/ dir
with a batch.pkl must also hold result.npz, else the item fails naming the seeds without results.
<work_dir>/seed-<s>/result.npz keys (the kit's e2e form as forward.py saves them; what af3_torch_api.forward returns):
    atom_positions   [S,N,24,3] float32 (a [N,24,3] array is accepted as one sample)
    conf_<key>       every key of af3_torch_api.run_confidence: predicted_lddt [N,24], predicted_experimentally_resolved [N,24],
                     full_pae [N,N], full_pde [N,N], average_pde [], tmscore_adjusted_pae_global [N,N], tmscore_adjusted_pae_interface [N,N]
                     (per sample, or with a leading sample axis S; the fork's result layout carries the sample axis, added here when absent)
    contact_probs    [N,N] the distogram head's contact probabilities (its dict keys saved as-is; bin_edges optional)
    seed             0-d int: the torch RNG seed of the sample(s); must equal batch.pkl's seed
    model_id         the params file's __meta__/__identifier__ record (uint8[64]) as forward.py writes it from the loaded model
Report: {items: [{name, work_dir, out_dir, ok, seeds, files: [{path, bytes, sha256}], ranking: [{seed, sample, ranking_score}],
top: {seed, sample, ranking_score}, samples, n_tokens, n_atoms, conversions, model_id_source, job_name, sanitised_name, error?}], ok}.
Exit 0 only when every item is ok.
"""
import argparse
import importlib.util
import json
import os
import pickle
import re
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_stream as PS  # noqa: E402 — the streamed chain's hand-off markers (standard library only)


def file_hasher(opt_core_dir):
    """The package's one file hasher (outputs.sha256 = the core's opt_core.gates.sha256_file): this script runs standalone under the JAX
    venv, so the core's directory is passed explicitly (--opt_core, the wrapper names the core it imported) and placed on sys.path first."""
    if opt_core_dir and opt_core_dir not in sys.path:
        sys.path.insert(0, opt_core_dir)
    from outputs import sha256   # noqa: E402
    return sha256

# per-sample rank of every confidence-head output (alphafold3.model.network.confidence_head / xfold nn.head.ConfidenceHead)
CONF_RANK = {"predicted_lddt": 2, "predicted_experimentally_resolved": 2, "full_pae": 2, "full_pde": 2, "average_pde": 0,
             "tmscore_adjusted_pae_global": 2, "tmscore_adjusted_pae_interface": 2}
REQUIRED_CONF = ("predicted_lddt", "full_pae", "full_pde", "average_pde", "tmscore_adjusted_pae_global", "tmscore_adjusted_pae_interface")
DISTOGRAM_KEYS = ("contact_probs", "bin_edges")  # the distogram head's dict (xfold nn.head.DistogramHead), saved under its own key names
SEED_DIR = re.compile(r"^seed-(\d+)$")
SHA256 = None            # set by main(): file_hasher(--opt_core)


def parse_item(spec):
    """'NAME=<work_dir>=<out_dir>' -> (name, work_dir, out_dir)."""
    parts = spec.split("=", 2)
    if len(parts) != 3 or not all(parts):
        raise ValueError(f"--item must be NAME=<work_dir>=<out_dir>, got {spec!r}")
    return parts[0], parts[1], parts[2]


def load_run_alphafold(repo_dir):
    """Import the fork's run_alphafold.py (a script, not a package module) for ResultsForSeed / write_fold_input_json / write_outputs.
    write_outputs reads the absl flag --of3_weights (the OF3 output terms), so the flags are parsed here with exactly that flag."""
    path = os.path.join(repo_dir, "run_alphafold.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"run_alphafold.py not found under --repo_dir {repo_dir}")
    spec = importlib.util.spec_from_file_location("run_alphafold", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from absl import flags
    if flags.FLAGS.is_parsed():
        flags.FLAGS.of3_weights = True
    else:
        flags.FLAGS(["postprocess.py", "--of3_weights"])
    return mod


def build_result(z, conversions):
    """result.npz arrays -> the ModelResult dict layout of alphafold3.model.model.Model.__call__ + ModelRunner.run_inference
    (numpy, float32, leading sample axis). Every conversion performed is appended to `conversions` by name."""
    xyz = np.asarray(z["atom_positions"])
    if xyz.dtype != np.float32:
        conversions.append(f"atom_positions {xyz.dtype} -> float32"); xyz = xyz.astype(np.float32)
    if xyz.ndim == 3:
        conversions.append("atom_positions: sample axis added ([N,24,3] -> [1,N,24,3])"); xyz = xyz[None]
    if xyz.ndim != 4 or xyz.shape[-1] != 3:
        raise ValueError(f"atom_positions must be [S,N,24,3], got {xyz.shape}")
    S = int(xyz.shape[0])
    result = {"diffusion_samples": {"atom_positions": xyz}, "distogram": {}}
    for k in z.files:
        if k.startswith("conf_"):
            key = k[len("conf_"):]
            arr = np.asarray(z[k])
            if arr.dtype.kind == "f" and arr.dtype != np.float32:
                conversions.append(f"{key} {arr.dtype} -> float32"); arr = arr.astype(np.float32)
            rank = CONF_RANK.get(key)
            if rank is not None:
                if arr.ndim == rank:
                    if S != 1:
                        raise ValueError(f"{key} has no sample axis (shape {arr.shape}) but atom_positions carries S={S} samples")
                    conversions.append(f"{key}: sample axis added ({arr.shape} -> {(1,) + arr.shape})"); arr = arr[None]
                elif not (arr.ndim == rank + 1 and arr.shape[0] == S):
                    raise ValueError(f"{key}: shape {arr.shape} is neither per-sample rank {rank} nor [S={S}, ...]")
            elif not (arr.ndim >= 1 and arr.shape[0] == S):
                if S != 1:
                    raise ValueError(f"{key}: unknown confidence key with shape {arr.shape}; cannot place a sample axis for S={S}")
                conversions.append(f"{key}: sample axis added ({arr.shape} -> {(1,) + arr.shape})"); arr = arr[None]
            result[key] = arr
        elif k in DISTOGRAM_KEYS:
            result["distogram"][k] = np.asarray(z[k])
    missing = [k for k in REQUIRED_CONF if k not in result]
    if missing:
        raise KeyError(f"result.npz lacks conf_ keys {missing}")
    if "contact_probs" not in result["distogram"]:
        raise KeyError("result.npz lacks contact_probs")
    return result, S


def file_rows(out_dir):
    rows = []
    for dp, _, fns in os.walk(out_dir):
        for fn in sorted(fns):
            p = os.path.join(dp, fn)
            rows.append(dict(path=os.path.relpath(p, out_dir), bytes=os.path.getsize(p), sha256=SHA256(p)))
    return sorted(rows, key=lambda r: r["path"])


def seed_dirs(work_dir):
    """The seed-<s>/ dirs of a work dir that hold a batch.pkl, ordered by seed."""
    found = []
    for d in os.listdir(work_dir):
        m = SEED_DIR.match(d)
        if m and os.path.isfile(os.path.join(work_dir, d, "batch.pkl")):
            found.append((int(m.group(1)), os.path.join(work_dir, d)))
    if not found:
        raise FileNotFoundError(f"{work_dir}: no seed-<s>/batch.pkl")
    return sorted(found)


def results_for_seed(seed, d, RA, conversions):
    """One seed dir -> (run_alphafold.ResultsForSeed, fold_input, sample rows, facts)."""
    from alphafold3.model import model as af3_model
    with open(os.path.join(d, "batch.pkl"), "rb") as f:
        pk = pickle.load(f)
    for key in ("name", "seed", "example", "fold_input"):
        if key not in pk:
            raise KeyError(f"{d}/batch.pkl lacks {key!r}")
    if int(pk["seed"]) != seed:
        raise ValueError(f"{d}: batch.pkl seed {pk['seed']} != dir seed {seed}")
    z = np.load(os.path.join(d, "result.npz"))
    if "seed" in z.files and int(np.asarray(z["seed"]).reshape(-1)[0]) != seed:
        raise ValueError(f"{d}: result.npz seed {int(np.asarray(z['seed']).reshape(-1)[0])} != {seed}")
    result, S = build_result(z, conversions)
    if "model_id" not in z.files:
        raise KeyError(f"{d}: result.npz has no 'model_id' (forward.py writes the loaded model's identifier record)")
    result["__identifier__"] = np.asarray(z["model_id"]).tobytes(); model_id_source = "result.npz:model_id"
    inference_results = list(af3_model.Model.get_inference_result(batch=pk["example"], result=result, target_name=pk["name"]))
    rfs = RA.ResultsForSeed(seed=seed, inference_results=inference_results, full_fold_input=pk["fold_input"], embeddings=None, distogram=None)
    samples = [dict(seed=seed, sample=i, ranking_score=float(r.metadata["ranking_score"]), ptm=float(r.metadata["ptm"]), iptm=float(r.metadata["iptm"]),
                    num_atoms=int(r.predicted_structure.num_atoms), chains=list(r.predicted_structure.chains))
               for i, r in enumerate(inference_results)]
    facts = dict(n_samples=S, n_tokens=int(np.asarray(pk["example"]["seq_length"])), model_id_source=model_id_source)
    return rfs, pk["fold_input"], samples, facts


def postprocess_one(name, work_dir, out_dir, RA):
    """One item: every seed-<s>/{batch.pkl,result.npz} -> the fork's outputs in out_dir; returns the report row fields."""
    dirs = seed_dirs(work_dir)
    missing = [f"seed-{s}" for s, d in dirs if not os.path.isfile(os.path.join(d, "result.npz"))]
    if missing:
        raise FileNotFoundError(f"{work_dir}: result.npz missing for {missing}")
    conversions = []
    all_rfs, samples, facts, fold_input = [], [], {}, None
    for seed, d in dirs:
        rfs, fold_input, srows, facts = results_for_seed(seed, d, RA, conversions)
        all_rfs.append(rfs); samples.extend(srows)
    os.makedirs(out_dir, exist_ok=True)
    RA.write_fold_input_json(fold_input, out_dir)
    RA.write_outputs(all_inference_results=all_rfs, output_dir=out_dir, job_name=name, compress_large_output_files=False, save_terms_of_use=True)
    ranking = [dict(seed=s["seed"], sample=s["sample"], ranking_score=s["ranking_score"]) for s in samples]
    top = max(ranking, key=lambda r: r["ranking_score"])
    return dict(seeds=[s for s, _ in dirs], job_name=name, sanitised_name=fold_input.sanitised_name(), name_matches_json=(name == fold_input.sanitised_name()),
                n_tokens=facts["n_tokens"], n_samples_per_seed=facts["n_samples"], n_atoms=samples[0]["num_atoms"], samples=samples, ranking=ranking, top=top,
                conversions=sorted(set(conversions)), model_id_source=facts["model_id_source"], files=file_rows(out_dir))


def run(items, report_path, repo_dir, follow=None):
    """Post-process every item (total accounting: each row ok or failed with its named reason); returns the report dict. ``follow`` = the pred's
    work root: the streamed chain (pipeline_stream.await_results per item; withdrawn items leave no row; write_behind = the census)."""
    rows = []
    RA = None
    ra_error = None
    try:
        RA = load_run_alphafold(repo_dir)
    except Exception as e:  # every item then fails with this named reason
        ra_error = f"{type(e).__name__}: {e}"
    wb = {"streamed": int(bool(follow)), "items": 0, "early": 0, "withdrawn": 0, "wait_s": 0.0}
    if follow:                                                       # this process runs beside the wrapper, and ends with it
        PS.exit_with_parent()
    for name, work_dir, out_dir in items:
        row = dict(name=name, work_dir=work_dir, out_dir=out_dir, ok=False)
        if follow:                                                   # the model process runs beside this one: take the item when its results are handed over
            w = PS.await_results(work_dir, follow)
            if w["state"] != "ready":                                # never handed a result (featurise / every forward failed): not this process's item, exactly as in the sequential chain
                wb["withdrawn"] += 1
                print(json.dumps({"name": name, "withdrawn": w.get("reason"), "follow_wait_s": w["wait_s"]}), flush=True)
                continue
            row.update(follow_wait_s=w["wait_s"], early=w["early"]); wb["items"] += 1; wb["early"] += int(w["early"]); wb["wait_s"] = round(wb["wait_s"] + w["wait_s"], 3)
        try:
            if RA is None:
                raise RuntimeError(f"run_alphafold.py import failed: {ra_error}")
            row.update(postprocess_one(name, work_dir, out_dir, RA))
            row["ok"] = True
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
            row["traceback"] = traceback.format_exc()[-4000:]
        print(json.dumps({k: v for k, v in row.items() if k not in ("traceback", "files")}), flush=True)
        rows.append(row)
    report = dict(items=rows, ok=all(r["ok"] for r in rows), repo_dir=repo_dir, write_behind=wb)
    if report_path:
        os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=1)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--item", action="append", required=True, help="NAME=<work_dir>=<out_dir> (repeatable)")
    ap.add_argument("--report", required=True, help="report json path")
    ap.add_argument("--repo_dir", required=True, help="the fork checkout holding run_alphafold.py (cli.py passes stock/PINS.json image.jax_repo)")
    ap.add_argument("--opt_core", default=None, help="the directory holding the opt_core package (cli.py passes the core it imported); the file hasher is the core's")
    ap.add_argument("--follow", default=None, help="the pred's work root when the wrapper runs the chain streamed (package lever write_behind): each item is written as soon as the model process hands over its results (pipeline_stream), while the next item runs")
    args = ap.parse_args(argv)
    global SHA256
    SHA256 = file_hasher(args.opt_core)
    report = run([parse_item(s) for s in args.item], args.report, args.repo_dir, follow=args.follow)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
