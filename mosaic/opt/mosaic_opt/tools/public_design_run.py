#!/usr/bin/env python
"""
public_design_run.py — ONE binder-hallucination design in ONE fresh process, stock recipe, optional kit levers. The target is the PUBLIC example
input unless a FASTA is given; every recipe call is tools/recipe.py's (the one module the package imports too).

Recipe = escalante-bio/mosaic examples/boltz_notebook.py @ 70fec525 (nothing reduced; tools/recipe.py carries its literals with their source lines):
  target   = barstar (PDB 1BRS chain D SEQRES, 89 aa) or --target-fasta's one record (a multi-record FASTA is refused by name, `fasta_multi_record`,
             unless --first-record takes record 1: named on an `INPUT target_fasta records=<n> used=1` line and in results.json); --target-copies
             chains of it; the target's MSA from a precomputed alignment (--msa PATH: boltz `msa: <path>`, the notebook's use_msa=True with the fetch
             done ahead of time) or single-sequence (`msa: empty`) when none is given — NO MSA server call either way (a chain asking for an MSA
             without a file is refused)
  binder   = poly-X, length L (default 80)  -> 89·copies + L tokens on the public target (169 at L 80, "1:1"; 258 with --target-copies 2)
  loss     = 2*BinderTargetContact + WithinBinderContact + 5*InverseFoldingSequenceRecovery(ProteinMPNN v_48_020, T=0.01)
             -> Boltz2Loss(recycling_steps=1, sampling_steps=25, deterministic=True); --epitope 12,15,40-48 (an input option, every arm: 1-based
             residue positions along the target sequence, applied on every copy) makes the first term upstream's own
             BinderTargetContact(epitope_idx=[...]) at the same weight — named on an `INPUT epitope …` line and recorded in results.json
  x0       = softmax(0.5 * gumbel(key(seed), (L,20)));  stage1 = simplex_APGM(75 steps, stepsize 0.1, momentum 0, key fold_in(key(seed),1));
  stage2   = simplex_APGM(50 steps, stepsize 0.5, scale 1.5, momentum 0, key fold_in(key(seed),2)) from stage-1 best;  refold = model_output(x2, key(0)).
  (The stock notebook seeds with np.random.randint / key=None; explicit keys are the only deviation, needed for replayability.)

Levers (all default OFF = STOCK arm):
  --levers WORD[+ID[=SPEC],..]  a tier row's per-step levers: WORD = a served mode word (exact | fast | big), `+ID…` extra levers /
                                settings (a development request); = mosaic_opt.levers.install(WORD, plus=…) once after the stack import, before any trace
  --weights torch|fastinit      P2: Boltz2() stock torch load  vs  mosaic.fast.fastload.load_stock_fast_init() (identical parameters; sha256 of all leaves recorded)
  env JAX_COMPILATION_CACHE_DIR + XLA_FLAGS=--xla_gpu_{dump,load}_autotune_results_{to,from}=FILE     P1 (set by the caller's environment row, recorded here)
  --features-in NPZ --features-sha SHA   P3: load frozen features instead of featurizing (the npz is
                                          written by --features-out of a stock arm of the same shape)
Report-only lines: `[run] kit_modules source=mosaic.fast|kit_src …` (which home the fastload / numstate helpers loaded from — the kit venv's
mosaic.fast or the kit's mosaic_fast/ copy — the same bytes in either home), `INPUT …` (tools/recipe.py input_lines: the input options that narrow
what the recipe reads — `--first-record`, `--epitope` — once the inputs resolve) and `PEAK item=<tag> peak_bytes_in_use_gib=<x> …` (jax's
device-memory peak of the process) after the design.
Outputs: <out>/<tag>/results.json (manifest: versions, gpu, host class, precision statement, P1 keys, timings; run: 125 per-step losses,
         sha256 of x0/x1/best1/x2/best2 PSSMs, final sequences, refold ipTM/pLDDT/PAE, sha256 of refold arrays), pssm + refold npz.
"""
import argparse, hashlib, json, os, re, subprocess, sys, time, traceback
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--tag", required=True)
p.add_argument("--out", default="out")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--binder-length", type=int, default=80)
p.add_argument("--target-copies", type=int, default=1)
p.add_argument("--weights", default="torch", choices=["torch", "fastinit"])
p.add_argument("--levers", default="", help="WORD[+ID[=SPEC][,ID[=SPEC]...]]: the per-step levers of a served mode word (mosaic_opt.levers.install(WORD, plus=...), the ONE installer); empty = stock's step, nothing imported")
p.add_argument("--features-in", default=None)
p.add_argument("--features-sha", default=None)
p.add_argument("--features-out", default=None)
p.add_argument("--steps1", type=int, default=75)
p.add_argument("--steps2", type=int, default=50)
p.add_argument("--target-fasta", default=None, help="the target = the ONE record of this FASTA (default: the public example target barstar 1BRS:D); a multi-record file is refused unless --first-record")
p.add_argument("--first-record", action="store_true", help="design against record 1 of a multi-record --target-fasta (the record count and the record used are printed on an INPUT line and recorded)")
p.add_argument("--epitope", default=None, help="residue positions on the target the binder should contact, 1-based along the target sequence as given, comma-separated with ranges (12,15,40-48), applied on every --target-copies copy: the loss's BinderTargetContact term gets upstream's epitope_idx (every arm; absent = the notebook's unrestricted term)")
p.add_argument("--msa", default=None, help="the target chain's precomputed alignment — boltz's processed MSA .csv (the file its own server route writes for the chain) or a raw .a3m — applied to every copy; absent = single-sequence")
args = p.parse_args()
OUT = Path(args.out) / args.tag; OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import recipe  # noqa: E402  tools/recipe.py: the recipe (the notebook's literals and calls, once)
L = args.binder_length
try:                                                          # the inputs resolved through the recipe; a refusal (fasta_no_record, msa_not_found, msa_public_target, …) is a NAMED result, never a bare traceback
    TARGET = recipe.target(fasta=args.target_fasta, copies=args.target_copies, msa=args.msa, first_record=args.first_record)
    EPITOPE = recipe.epitope(args.epitope, TARGET) if args.epitope is not None else None      # the --epitope input option resolved on the target (positions checked against its length; upstream's index list for every copy)
    STAGES = recipe.stages(args.steps1, args.steps2)
except recipe.RecipeError as _e:
    json.dump({"manifest": {"tag": args.tag, "args": vars(args)}, "run": None, "status": "error", "error": f"inputs_refused: {_e}"}, open(OUT / "results.json", "w"), indent=2, default=str)
    print(f"[run] {args.tag} inputs_refused: {_e}", flush=True); print(f"[run] {args.tag} done in {time.time() - T0:.0f}s status=error", flush=True)
    sys.exit(1)
for _line in recipe.input_lines(TARGET, EPITOPE):             # the input options that narrow what the recipe reads, named once (record 1 of a multi-record FASTA; the epitope positions): never silent
    print(_line, flush=True)


def sha256_file(pth, block=1 << 24):
    h = hashlib.sha256()
    with open(pth, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def sha16(arr):
    import numpy as _np
    return hashlib.sha256(_np.ascontiguousarray(_np.asarray(arr, dtype=_np.float32)).tobytes()).hexdigest()[:16]


def host_class():
    info = {"vendor": "", "model_name": "", "family": "", "model": ""}; flags = set()
    try:
        for line in open("/proc/cpuinfo"):
            k = line.split(":", 1)[0].strip()
            v = line.split(":", 1)[1].strip() if ":" in line else ""
            if k == "vendor_id" and not info["vendor"]: info["vendor"] = v
            elif k == "model name" and not info["model_name"]: info["model_name"] = v
            elif k == "cpu family" and not info["family"]: info["family"] = v
            elif k == "model" and not info["model"]: info["model"] = v
            elif k == "flags" and not flags: flags = set(v.split())
    except Exception:
        pass
    isa = [f for f in ["avx512f", "avx512_fp16", "amx_tile", "amx_bf16", "amx_int8", "avx512_bf16", "avx_vnni"] if f in flags]
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:
        smi = f"nvidia-smi unavailable ({e})"
    return {"gpu_nvidia_smi": smi, "cpu": info, "isa": "+".join(isa) if isa else "no-avx512", "ncpu": os.cpu_count(),
            "region": next((v for k, v in sorted(os.environ.items()) if k.endswith("_REGION") and v), os.environ.get("CLOUD_REGION", "")),
            "cloud": next((v for k, v in sorted(os.environ.items()) if k.endswith("_CLOUD_PROVIDER") and v), "")}


def identity_key():
    fl = os.environ.get("XLA_FLAGS", ""); info = {"xla_flags": fl, "jax_compilation_cache_dir": os.environ.get("JAX_COMPILATION_CACHE_DIR", "")}
    for k in ["--xla_gpu_dump_autotune_results_to", "--xla_gpu_load_autotune_results_from"]:
        m = re.search(k + r"=(\S+)", fl)
        if m:
            pth = m.group(1); info[k.strip("-")] = pth
            if os.path.exists(pth): info[k.strip("-") + "_sha256"] = sha256_file(pth)
    cd = info["jax_compilation_cache_dir"]
    if cd and os.path.isdir(cd):
        names = sorted(os.listdir(cd)); info["jax_cache_n_entries"] = len(names)
        info["jax_cache_listing_sha16"] = hashlib.sha256("\n".join(names).encode()).hexdigest()[:16]
    return info


manifest = {"tag": args.tag, "args": vars(args),
            "recipe": (f"examples/boltz_notebook.py ({STAGES[0]['n_steps']}+{STAGES[1]['n_steps']} {recipe.STAGE_CALL}, Boltz2Loss recycling {recipe.BOLTZ2_LOSS['recycling_steps']} / "
                       f"sampling {recipe.BOLTZ2_LOSS['sampling_steps']} / deterministic, MPNN {recipe.MPNN_WEIGHTS} T={recipe.MPNN_TEMP})"),
            "target": {k: TARGET[k] for k in ("name", "id", "pdb", "chain", "length", "seq_sha256", "copies", "use_msa", "source", "fasta", "fasta_sha256", "fasta_records", "fasta_record_used",
                                             "first_record", "msa", "msa_sha256", "tokens_target")},
            "epitope": EPITOPE,                               # None = the notebook's unrestricted BinderTargetContact term; else {spec, residues (1-based), target_length, copies, idx (upstream's epitope_idx), convention}
            "stages": STAGES, "binder_length": L, "host_class": host_class(), "identity_key_pre": identity_key(), "pid": os.getpid()}
results = {"manifest": manifest, "run": None, "status": "started"}


def dump():
    results["t_process_total_s"] = time.time() - T0
    json.dump(results, open(OUT / "results.json", "w"), indent=1, default=lambda o: float(o) if hasattr(o, "item") else str(o))


dump()
try:
    t = time.time()
    import jax, jax.numpy as jnp, numpy as np, equinox as eqx
    import importlib.metadata as md
    from mosaic.common import TOKENS
    from mosaic.losses.structure_prediction import IPTMLoss
    STACK = recipe.import_stack()                             # the stack imported BEFORE the numeric-state snapshot below: its observation window opens on the loaded stack (jax, equinox, mosaic's optimizers / losses / Boltz-2 / MPNN and, through them, boltz / torch / joltz)
    LEVERS = recipe.install_levers(args.levers)               # --levers WORD[+ID...]: the per-step levers through mosaic_opt.levers.install(WORD, ...) (the ONE installer) after the stack import and BEFORE the numeric-state snapshot, any trace and the model load; "" = stock's step (nothing imported)
    manifest["levers_at_install"] = recipe.levers_record(); manifest["levers"] = dict(manifest["levers_at_install"]); manifest["levers_label"] = LEVERS["label"] if LEVERS else None; manifest["levers_off"] = list(LEVERS.get("levers_off") or []) if LEVERS else []   # the ablation switch (MODEL_OPT_LEVERS_OFF): the per-step levers skipped by name in this process
    fastload, numstate, KIT_MODULES = recipe.kit_modules()   # mosaic.fast's copies when the package carries the kit's files (the kit venv), else tools/ (an install without them: the stock venv); which home loaded is a named line
    print(f"[run] kit_modules source={KIT_MODULES['source']} files={KIT_MODULES['files']}", flush=True)
    manifest["kit_modules"] = KIT_MODULES
    manifest["t_import_s"] = time.time() - t
    manifest["versions"] = {"jax": jax.__version__, "jaxlib": md.version("jaxlib"), "jax-cuda12-plugin": (md.version("jax-cuda12-plugin") if any(d.metadata["Name"].lower() == "jax-cuda12-plugin" for d in md.distributions()) else "n/a"),
                            "equinox": eqx.__version__, "torch": md.version("torch"), "numpy": np.__version__,
                            "mosaic": md.version("mosaic") if any(d.metadata["Name"].lower() == "mosaic" for d in md.distributions()) else "source", }
    for dist in ("mosaic", "joltz", "boltz"):
        try:
            txt = md.distribution(dist).read_text("direct_url.json")
            if txt: manifest["versions"][dist + "_direct_url"] = json.loads(txt)
        except Exception:
            pass
    manifest["jax_devices"] = [str(d) for d in jax.devices()]; manifest["jax_backend"] = jax.default_backend()
    assert jax.default_backend() == "gpu", f"JAX is not using the GPU (backend={jax.default_backend()}); check jax/jaxlib/jax-cuda12-plugin versions are all equal"
    NS0 = numstate.snapshot()
    manifest["numeric_state_S0"] = NS0
    manifest["precision_statement"] = {"jax_default_matmul_precision": str(jax.config.jax_default_matmul_precision), "jax_enable_x64": bool(jax.config.jax_enable_x64),
                                       "dtype": "float32 params/activations (stock)", "xla_gpu_deterministic_ops": "--xla_gpu_deterministic_ops=true" in os.environ.get("XLA_FLAGS", ""),
                                       "joltz_deterministic": True, "note": "kit levers never change precision, dtypes or XLA numerics flags"}
    # ---- model load (P2 arm or stock)
    t = time.time()
    model, manifest["load_path"] = recipe.load_model(args.weights)
    manifest["t_load_boltz2_s"] = time.time() - t
    manifest["param_fingerprint"] = fastload.fingerprint(model)
    ck = Path(os.environ.get("MOSAIC_CACHE_DIR", "~/.cache/mosaic")).expanduser() / "boltz" / "boltz2_conf.ckpt"
    manifest["boltz2_ckpt_sha256"] = sha256_file(ck) if ck.exists() else "n/a"
    t = time.time(); mpnn = recipe.load_mpnn(); manifest["t_load_mpnn_s"] = time.time() - t
    try:
        import mosaic.proteinmpnn as _pm
        manifest["mpnn_weight_sha256"] = {f.name: sha256_file(f) for f in (Path(_pm.__file__).parent / "weights").glob("*.pt")}
    except Exception as e:
        manifest["mpnn_weight_sha256"] = f"n/a {e}"
    # ---- features (the target featurized in process, with its staged MSA or single-sequence) or the frozen npz of an earlier arm of this run
    t = time.time()
    if args.features_in:
        features, got = recipe.load_frozen_features(args.features_in, args.features_sha)
        manifest["features"] = {"source": f"frozen npz {args.features_in}", "sha256": got}
    else:
        features, _writer = recipe.featurize(model, L, TARGET)
        features = recipe.as_jax(features)
        manifest["features"] = {"source": f"in-process boltz featurization of {recipe.target_words(TARGET)} ({recipe.msa_words(TARGET)}; no server)"}
        if args.features_out:
            fo, manifest["features"]["sha256"] = recipe.save_features(args.features_out, features)
            manifest["features"]["frozen_out"] = fo
    manifest["features"]["ref_pos_sha16"] = sha16(features["ref_pos"]); manifest["features"]["n_keys"] = len(features)
    manifest["features"]["msa_rows"] = int(features["msa"].shape[1]); manifest["t_features_s"] = time.time() - t
    n_tok = int(features["res_type"].shape[1]); manifest["n_tokens"] = n_tok; manifest["n_atoms_padded"] = int(features["atom_pad_mask"].shape[-1])
    assert n_tok == recipe.tokens(TARGET, L), f"token count {n_tok} != target residues x copies + L = {recipe.tokens(TARGET, L)} (recipe.tokens)"
    loss = recipe.build_loss(model, mpnn, dict(features), epitope_idx=EPITOPE["idx"] if EPITOPE else None)   # --epitope: upstream's BinderTargetContact(epitope_idx=...) at the notebook's weight; absent: the notebook's term unchanged
    dump()
    # ---- design
    seed = args.seed; rec = {"seed": seed}
    x0 = recipe.x0(seed, L); rec["x0_sha16"] = sha16(x0)
    (s1, s2) = recipe.design(loss, x0, seed, STAGES, trajectory_fn=recipe.trajectory_record)     # the notebook's chaining, once (recipe.design): stage 2 starts from stage 1's best
    x1, best1, tr1, x2, best2, tr2 = s1["x"], s1["best"], s1["trajectory"], s2["x"], s2["best"], s2["trajectory"]
    rec["t_stage1_s"] = s1["wall_s"]; rec["t_stage2_s"] = s2["wall_s"]
    times = [r["time"] for r in tr1 + tr2]
    rec.update({"t_design_opt_s": rec["t_stage1_s"] + rec["t_stage2_s"], "t_first_iter_s": times[0], "t_iter_stage1_steady_s": float(np.mean(times[2:args.steps1])),
                "t_iter_stage2_first_s": times[args.steps1], "t_iter_stage2_steady_s": float(np.mean(times[args.steps1 + 2:])),
                "loss_traj": [r["loss"] for r in tr1 + tr2], "nnz_last": tr2[-1]["nnz"]})
    X1, B1, X2, B2 = (np.asarray(a) for a in (x1, best1, x2, best2))
    for nm, arr in [("x1", X1), ("best1", B1), ("x2", X2), ("best2", B2)]: rec[f"{nm}_sha16"] = sha16(arr)
    rec["seq_stage2_x"] = "".join(TOKENS[i] for i in X2.argmax(-1)); rec["seq_stage2_best"] = "".join(TOKENS[i] for i in B2.argmax(-1))
    rec["loss_traj_sha256"] = hashlib.sha256(np.asarray(rec["loss_traj"], dtype=np.float64).tobytes()).hexdigest()
    np.savez(OUT / f"pssm_seed{seed}.npz", x0=np.asarray(x0), x1=X1, best1=B1, x2=X2, best2=B2, loss_traj=np.array(rec["loss_traj"]))
    t = time.time()
    o = recipe.refold(model, X2, features); jax.block_until_ready(o.pae)
    rec["t_refold_s"] = time.time() - t
    rec["refold_iptm"] = float(-IPTMLoss()(jnp.asarray(X2), o, key=jax.random.key(recipe.REFOLD_KEY))[0])
    rec["refold_plddt_binder_mean"] = float(np.asarray(o.plddt)[:L].mean()); rec["refold_pae_bt_mean"] = float(np.asarray(o.pae)[:L, L:].mean())
    coords = np.asarray(o.structure_coordinates)
    rec["refold_coords_sha16"] = sha16(coords); rec["refold_pae_sha16"] = sha16(o.pae); rec["refold_plddt_sha16"] = sha16(o.plddt)
    np.savez(OUT / f"refold_seed{seed}.npz", pae=np.asarray(o.pae), plddt=np.asarray(o.plddt), coords=coords)
    rec["numeric_state_unchanged"] = (numstate.diff(NS0, numstate.snapshot()) == {})
    manifest["levers"] = recipe.levers_finalize(args.tag)     # the per-step levers AFTER every phase ran: census lines, fail-closed gates (a raise fails the arm by name), final served / fallback facts — installed is not the same as ran; this is the record results.json carries
    manifest["peak"] = recipe.peak_memory(); print(recipe.peak_line(args.tag, manifest["peak"]), flush=True)
    results["run"] = rec; results["manifest"]["identity_key_post"] = identity_key(); results["status"] = "ok"
    print(f"[run] {args.tag}: tokens {n_tok} load {manifest['t_load_boltz2_s']:.1f}s feat {manifest['t_features_s']:.1f}s first_step {rec['t_first_iter_s']:.1f}s "
          f"steady {rec['t_iter_stage1_steady_s']:.3f}s/step opt {rec['t_design_opt_s']:.1f}s refold {rec.get('t_refold_s', float('nan')):.1f}s "
          f"loss[0]={rec['loss_traj'][0]!r} loss[-1]={rec['loss_traj'][-1]!r} best2={rec['best2_sha16']} iptm={rec.get('refold_iptm')} seq={rec['seq_stage2_best']}", flush=True)
except Exception:
    results["status"] = "error"; results["error"] = traceback.format_exc(); print(results["error"], flush=True)
finally:
    dump(); print(f"[run] {args.tag} done in {time.time() - T0:.0f}s status={results['status']}", flush=True)
sys.exit(0 if results["status"] == "ok" else 1)
