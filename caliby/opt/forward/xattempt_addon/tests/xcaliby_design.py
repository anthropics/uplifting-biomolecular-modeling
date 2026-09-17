#!/usr/bin/env python3
"""xcaliby_design.py — minimal CLI over caliby's public Python API (clean_pdbs -> load_model -> [seed] -> sample, or with --ensemble:
generate_ensembles -> ensemble_sample), the tree's design writer (the package's `design` verb runs it) and a plain design driver.

Every upstream keyword is an option of the same name and is forwarded ONLY when given, so a keyword not given runs at caliby's own default
(caliby/api.py, caliby/configs/seq_des/inference.yaml): load_model(model_name, device, sampling_cfg_path); sample / ensemble_sample(
num_seqs_per_pdb, batch_size, omit_aas, temperature, num_workers, verbose, sampling_overrides, pos_constraint_df from --pos_constraint_csv,
use_primary_res_type); clean_pdbs(num_workers = --clean_workers); generate_ensembles(num_samples_per_pdb, batch_size = --pp_batch_size,
sampling_yaml_path). One `model.sample` call over all cleaned inputs; seeded when --seed is given (Lightning's seed_everything; caliby's
API sets no seed); `--det 1` is caliby's scripts' deterministic recipe (the seed plus torch.backends.cudnn.deterministic=True /
benchmark=False, caliby/eval/sampling/seq_des.py:22-24).
`--ensemble`: the ensemble-conditioned protocol of caliby's scripts — Protpardelle-1c partial-diffusion conformers per input
(generate_ensembles; seeded --seed+i per input when --seed is given), then one ensemble_sample per input over the members
seq_des_ensemble.py selects: the input structure first (--include_primary_conformer, default true), then the generated conformers in
natural order, --max_num_conformers members in all (default 32; caliby/eval/eval_utils/eval_setup_utils.py process_conformer_dirs);
a positional-constraint row for an input is expanded to its conformers (get_ensemble_constraint_df).

Outputs in --out_dir: seq_des_outputs.csv (the outputs table exactly as caliby's seq_des.py / seq_des_ensemble.py write it:
pd.DataFrame(outputs).to_csv(index=False)), raw/samples/*.cif (caliby output; raw/<stem>/samples/ per input with --ensemble),
cleaned/*.cif, ensembles/<stem>/ (with --ensemble), timing.json (clean_s, model_load_s, design_call_s, total_s, torch/GPU info, the
options given, env switches in effect).

usage: MODEL_PARAMS_DIR=<weights root> python xcaliby_design.py --inputs a.pdb b.cif ... --out_dir OUT [--seed S] [--det 0|1]
       [--model_name caliby] [--num_seqs_per_pdb 1] [--batch_size 4] [--omit_aas C,G] [--temperature 0.01] [--num_workers 2]
       [--verbose true|false] [--sampling_overrides KEY=VALUE ...] [--pos_constraint_csv F] [--clean_workers 1] [--device cuda]
       [--sampling_cfg_path Y] [--ensemble [--num_samples_per_pdb 32] [--pp_batch_size 8] [--sampling_yaml_path Y]
       [--max_num_conformers 32] [--include_primary_conformer true|false] [--use_primary_res_type true|false]]
"""
import argparse, json, os, sys, time

MAX_NUM_CONFORMERS = 32                                                # caliby/configs/eval/sampling/seq_des_ensemble.yaml max_num_conformers


def bool_word(s):
    v = str(s).strip().lower()
    if v in ("true", "1", "yes"):
        return True
    if v in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"expected true|false, got {s!r}")


def given(ns, **names):
    """{upstream keyword: value} for the options actually given (None = not given -> caliby's default applies)."""
    return {kw: getattr(ns, opt) for opt, kw in names.items() if getattr(ns, opt) is not None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True, help="PDB/CIF complexes (or single chains)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=None, help="given: seed_everything(seed) before sampling and seed+i per input for conformer generation; absent: caliby's unseeded call")
    ap.add_argument("--det", type=int, default=0, choices=[0, 1], help="1: --seed (required) + cuDNN deterministic flags (caliby's scripts' recipe)")
    # load_model
    ap.add_argument("--model_name", default=None, help="load_model(model_name=): registered name or .ckpt path (caliby's default: caliby)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--sampling_cfg_path", default=None)
    # sample / ensemble_sample
    ap.add_argument("--num_seqs_per_pdb", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--omit_aas", default=None, help="comma-separated one-letter codes, e.g. C or C,G")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--verbose", type=bool_word, default=None)
    ap.add_argument("--sampling_overrides", nargs="+", default=None, metavar="KEY=VALUE")
    ap.add_argument("--pos_constraint_csv", default=None)
    ap.add_argument("--use_primary_res_type", type=bool_word, default=None)
    # clean_pdbs
    ap.add_argument("--clean_workers", type=int, default=None, help="clean_pdbs(num_workers=); >1 with CALIBY_X_CLEAN=loader is the add-on's L-B path")
    # generate_ensembles + member selection
    ap.add_argument("--ensemble", action="store_true", help="ensemble-conditioned design (generate_ensembles per input, then ensemble_sample)")
    ap.add_argument("--num_samples_per_pdb", type=int, default=None)
    ap.add_argument("--pp_batch_size", type=int, default=None, help="generate_ensembles(batch_size=)")
    ap.add_argument("--sampling_yaml_path", default=None)
    ap.add_argument("--max_num_conformers", type=int, default=None)
    ap.add_argument("--include_primary_conformer", type=bool_word, default=None)
    a = ap.parse_args()
    if a.det and a.seed is None:
        sys.exit("--det 1 needs --seed")
    os.makedirs(a.out_dir, exist_ok=True)
    t_all = time.time()
    import torch
    import lightning as L
    import pandas as pd
    from caliby import clean_pdbs, load_model
    if a.det:                                                          # caliby/eval/sampling/seq_des.py:23-24
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    t0 = time.time()
    cleaned = clean_pdbs(sorted(a.inputs), out_dir=os.path.join(a.out_dir, "cleaned"), **given(a, clean_workers="num_workers"))
    clean_s = time.time() - t0
    t0 = time.time()
    model = load_model(**given(a, model_name="model_name", device="device", sampling_cfg_path="sampling_cfg_path"))
    load_s = time.time() - t0
    skw = given(a, num_seqs_per_pdb="num_seqs_per_pdb", batch_size="batch_size", temperature="temperature", num_workers="num_workers", verbose="verbose")
    if a.omit_aas is not None:
        skw["omit_aas"] = [x for x in a.omit_aas.split(",") if x] or None
    if a.sampling_overrides:
        from omegaconf import OmegaConf                                # caliby's own config library: KEY=VALUE words -> the nested dict sample() takes
        skw["sampling_overrides"] = OmegaConf.to_container(OmegaConf.from_dotlist(list(a.sampling_overrides)), resolve=True)
    cons = pd.read_csv(a.pos_constraint_csv) if a.pos_constraint_csv else None   # caliby/eval/sampling/seq_des.py:41-44
    ens_s, ens_n, conf_hex = 0.0, 0, None
    if a.ensemble:
        import hashlib
        from natsort import natsorted                                  # caliby's own ordering of conformer files (eval_setup_utils.process_conformer_dirs)
        from caliby import generate_ensembles
        from caliby.eval.eval_utils.eval_setup_utils import get_ensemble_constraint_df
        gkw = given(a, num_samples_per_pdb="num_samples_per_pdb", sampling_yaml_path="sampling_yaml_path")
        n_samp = gkw.get("num_samples_per_pdb", 32)                    # caliby/api.py generate_ensembles default
        if a.pp_batch_size is not None:
            gkw["batch_size"] = min(a.pp_batch_size, n_samp)
        elif n_samp < 8:
            gkw["batch_size"] = n_samp                                 # caliby's default batch (8) capped at the conformer count
        max_n = a.max_num_conformers if a.max_num_conformers is not None else MAX_NUM_CONFORMERS
        primary = True if a.include_primary_conformer is None else a.include_primary_conformer
        mappings, conf_sha = {}, hashlib.sha256()
        t0 = time.time()
        for i, c in enumerate(cleaned):
            stem = os.path.splitext(os.path.basename(c))[0]
            seed_kw = {"seed": a.seed + i} if a.seed is not None else {}
            conf = generate_ensembles([c], out_dir=os.path.join(a.out_dir, "ensembles", stem), **gkw, **seed_kw)
            samples = natsorted(conf.get(stem) or next(iter(conf.values())))
            for p in samples:
                conf_sha.update(open(p, "rb").read())
            members = ([c] + samples[:max_n - 1]) if primary else samples[:max_n]   # process_conformer_dirs' rule
            mappings[stem] = {stem: members}
            ens_n = max(ens_n, len(members))
        ens_s = time.time() - t0
        conf_hex = conf_sha.hexdigest()
        ekw = dict(skw, **given(a, use_primary_res_type="use_primary_res_type"))
        if a.seed is not None:
            L.seed_everything(a.seed)
        t0 = time.time()
        res = {}
        for stem, mapping in mappings.items():
            pc = get_ensemble_constraint_df(cons, mapping) if cons is not None and (cons["pdb_key"] == stem).any() else None   # seq_des_ensemble.py: the input's row expanded to its conformers
            r = model.ensemble_sample(mapping, out_dir=os.path.join(a.out_dir, "raw", stem), pos_constraint_df=pc, **ekw)
            for k, v in r.items():                                     # the calls' outputs concatenated, every key caliby returns (one table like seq_des_ensemble.py's)
                res.setdefault(k, []).extend(list(v))
        design_s = time.time() - t0
    else:
        if a.seed is not None:
            L.seed_everything(a.seed)
        t0 = time.time()
        res = model.sample(cleaned, out_dir=os.path.join(a.out_dir, "raw"), pos_constraint_df=cons, **skw)
        design_s = time.time() - t0
    pd.DataFrame(res).to_csv(os.path.join(a.out_dir, "seq_des_outputs.csv"), index=False)   # caliby/eval/sampling/seq_des.py:63-64 / seq_des_ensemble.py:77-78, verbatim
    opts = {k: v for k, v in vars(a).items() if k not in ("inputs", "out_dir") and v is not None and v is not False}
    info = {"n_inputs": len(cleaned), "n_designs": len(res["seq"]), "clean_s": round(clean_s, 2), "model_load_s": round(load_s, 2),
            "design_call_s": round(design_s, 2), "ensemble_s": round(ens_s, 2), "ensemble_n": ens_n, "conformers_sha256": conf_hex,
            "total_s": round(time.time() - t_all, 2), "seed": a.seed, "det": a.det, "options": opts,
            "torch": torch.__version__, "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "env": {k: os.environ.get(k, "") for k in ("CALIBY_FAST_SAMPLER", "CALIBY_FAST_POTTS_PARAMS", "CALIBY_X_SPARSE_EXACT", "CALIBY_X_LCP", "CALIBY_X_CLEAN", "CALIBY_X_BG_CIF",
                                                        "CALIBY_X_TIED_DET", "CALIBY_X_ENS_WORKERS", "CALIBY_X_PP_CACHE", "CALIBY_X_PP_NOSYNC", "CALIBY_X_PP_FASTPDB", "CALIBY_X_MULTISEQ", "CALIBY_X_CIF_WORKERS")}}
    json.dump(info, open(os.path.join(a.out_dir, "timing.json"), "w"), indent=1)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
