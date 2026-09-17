"""The stock caller — mode ``off``: the upstream Python API, nothing from the kit on the path.

``pred --mode off`` execs this file ONCE per invocation in a clean subprocess that loops the plan's items (``python -s -m chai1_opt.stock_fold --plan <json> ...``, every
kit and recipe variable stripped from the environment). It imports only ``chai_lab``, torch, numpy and the package's own I/O
modules (outputs.py, det.py — which import nothing from the kit); the kit's ``kit/`` directory is never on ``sys.path`` here, and the
process proves it before importing torch (``env_proof``): no forbidden environment variable, no kit module loaded, no kit directory on
the path — printed as one ``[chai1-opt stock] ENV-CLEAN ok: ...`` line (a violation is named and nothing folds). A seed-fold that fails is
named with its error (``SEED FAIL <key> <seed> <error>``) before its ``SEED`` status line; the pass ends ``DONE <ok>/<n>``.

The stock call itself is the block marked ``# --- the stock call`` below: ``chai_lab.chai1.run_inference`` at the plan's settings
(one timing-only wrapper is installed around ``chai_lab.chai1.run_folding_on_context`` for the FORWARD line — ``forward_timer``: synchronize,
perf_counter, arguments and result untouched; the allocator's peak counters restart at fold entry, read after each seed-fold for the row's
``max_mem_alloc_gb``)
— the plan's fold keywords (settings.py: stock's own run_inference flags, stock's defaults unless a flag was given; ``--device`` as given,
else cuda:0), one call per seed, seeds in sequence, each into an empty ``<out_dir>/<tag>/<key>/seed_<s>/`` (an item with no seed named: one
unseeded call, ``seed=None`` — stock sets no seed — into ``<key>/seed_none/``) (upstream asserts the directory is empty, ``chai1.py:505-508``).
The call is a PASS-THROUGH to upstream: before the first call the plan's keywords are compared with the INSTALLED signature's defaults
(``defaults_check``) and the pass is refused (exit 2) unless they differ from those defaults in exactly the knobs the caller gave a
non-default value (the plan's ``non_default``) — so without fold flags the only arguments that differ from ``run_inference``'s defaults are
the inputs (``fasta_file``, ``msa_directory``), the output directory and the seed; never a silent override (the comparison is written to the
proof file as ``defaults_check``). ``device="cuda:0"`` is upstream's own resolution of its ``None`` default (``chai1.py:510``). Upstream's files stay where it wrote them and
nothing is added beside them. ``det`` = 1 applies the deterministic recipe (det.py) —
``CUBLAS_WORKSPACE_CONFIG`` exported before ``import torch``, the other statements after — a stock-side switch, never a mode. This is
the only stock caller in the tree (``run.sh pred --mode off`` and ``pred --mode off`` both reach it).

Exit code: 0 when every seed-fold succeeded, 2 otherwise (the kit's own convention, ``kit/stock_fold.py:70``).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

T0 = time.time()
FORBIDDEN_PREFIXES = ("CHAI1_OPT",)                                   # the package's own switches
FORBIDDEN_EXACT = ("CHAI_DETERMINISTIC", "CHAI_JIT_PROFILING_OFF", "CUBLAS_WORKSPACE_CONFIG")   # the kit's recipe / switches
ALLOWED_CHAI = ("CHAI_DOWNLOADS_DIR",)                                # upstream's own data-path variable
KIT_MODULES = ("chai_proto", "chai_worker")
KIT_DIR_MARK = os.path.join("forward", "fast_inference")


def env_proof(environ=None, path=None, modules=None) -> dict:
    """Prove the process is clean BEFORE torch is imported. Raises RuntimeError naming the first violation."""
    environ = os.environ if environ is None else environ
    path = sys.path if path is None else path
    modules = sys.modules if modules is None else modules
    bad_env = sorted(k for k in environ if k.startswith(FORBIDDEN_PREFIXES) or k in FORBIDDEN_EXACT
                     or (k.startswith("CHAI_") and k not in ALLOWED_CHAI))
    bad_mod = sorted(m for m in modules if m.split(".")[0] in KIT_MODULES)
    bad_path = sorted(p for p in path if KIT_DIR_MARK in os.path.normpath(p))
    proof = {"schema": "chai1_opt.stock_env_proof/1", "clean": not (bad_env or bad_mod or bad_path), "forbidden_env_present": bad_env,
             "kit_modules_loaded": bad_mod, "kit_dirs_on_path": bad_path, "torch_imported_before_proof": "torch" in modules,
             "chai_lab_imported_before_proof": "chai_lab" in modules, "CHAI_DOWNLOADS_DIR": environ.get("CHAI_DOWNLOADS_DIR"),
             "python": sys.version.split()[0], "argv": list(sys.argv)}
    proof["core"] = _core_proof(environ, path, modules)
    if proof["core"] is not None and not proof["core"]["ok"]:
        proof["clean"] = False
    if not proof["clean"]:
        core_why = (proof["core"] or {}).get("violations")
        raise RuntimeError(f"stock process is not clean: env={bad_env} modules={bad_mod} path={bad_path}" + (f" core={core_why}" if core_why else ""))
    return proof


def _core_proof(environ, path, modules):
    """The shared core's clean-process proof (``opt_core.stock_proof.env_proof``: the same names, plus armed autoload finders, a kit
    sitecustomize, torch before the proof, core modules beyond the proof itself) — ``{"ok", "violations", "proof"}``; None when the core
    does not resolve in this process (the kit's own proof above stands alone; the fact is recorded)."""
    try:
        from . import _core
        _core.ensure_importable()
        from opt_core import stock_proof as _sp
    except Exception as e:  # noqa: BLE001
        return {"ok": True, "unavailable": repr(e), "violations": None, "proof": None}
    kit_dirs = [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "forward")]   # every carried kit lives under opt/forward/
    p = _sp.env_proof(env_absent=list(FORBIDDEN_PREFIXES) + list(FORBIDDEN_EXACT), kit_dirs=[os.path.normpath(d) for d in kit_dirs],
                      module_prefixes=list(KIT_MODULES), environ=environ, modules=modules, path=list(path))
    return {"ok": bool(p["ok"]), "violations": None if p["ok"] else _sp.violations_sentence(p), "proof": p}


def clean_environment(environ=None) -> dict:
    """The environment ``pred --mode off`` hands this process: every forbidden name stripped (the proof then holds by construction)."""
    environ = dict(os.environ if environ is None else environ)
    for k in list(environ):
        if k.startswith(FORBIDDEN_PREFIXES) or k in FORBIDDEN_EXACT or (k.startswith("CHAI_") and k not in ALLOWED_CHAI):
            environ.pop(k)
    return environ


def msa_directory_for(fasta: str, msa_dir):
    """chai-lab's own form for the item: ``msa_directory`` = the run's directory when at least one chain of the FASTA has its
    ``.aligned.pqt`` there (chai-lab's naming, ``expected_basename``; chains without one then get upstream's single-sequence context,
    ``data/dataset/msas/load.py:48-57``), else None — upstream's default, the empty MSA context for every chain (``chai1.py:394-398``).
    Returns (msa_directory | None, the basenames present, the chain count)."""
    from pathlib import Path
    from chai_lab.data.dataset.inference_dataset import read_inputs
    seqs = [inp.sequence for inp in read_inputs(Path(fasta))]
    if not msa_dir:
        return None, [], len(seqs)
    from chai_lab.data.parsing.msas.aligned_pqt import expected_basename
    present = sorted({expected_basename(q) for q in seqs if (Path(msa_dir) / expected_basename(q)).is_file()})
    return (Path(msa_dir) if present else None), present, len(seqs)


def pqt_rows(path) -> int:
    """Row count of one ``.aligned.pqt`` (parquet metadata; pandas when pyarrow is absent)."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        import pandas as pd
        return int(len(pd.read_parquet(path)))
    return int(pq.ParquetFile(path).metadata.num_rows)


def msa_depths(fasta: str, msa_dir):
    """Per chain of the FASTA (in order): the row count of its ``.aligned.pqt`` (chai-lab's naming) found in ``msa_dir``, 0 when absent.
    Returns (n_chains, [depth, ...])."""
    from pathlib import Path
    from chai_lab.data.dataset.inference_dataset import read_inputs
    seqs = [inp.sequence for inp in read_inputs(Path(fasta))]
    if not msa_dir:
        return len(seqs), [0] * len(seqs)
    from chai_lab.data.parsing.msas.aligned_pqt import expected_basename
    depths = []
    for q in seqs:
        p = Path(msa_dir) / expected_basename(q)
        depths.append(pqt_rows(p) if p.is_file() else 0)
    return len(seqs), depths


def defaults_check(fold: dict, run_inference) -> dict:
    """The pass-through proof: every keyword of ``fold`` that ``run_inference``'s signature defaults (``inspect.signature``) compared with the
    value the plan passes; ``{"checked": n, "mismatch": {name: {"passed", "default"}}, "not_in_signature": [...]}``."""
    import inspect
    params = inspect.signature(run_inference).parameters
    out = {"checked": 0, "mismatch": {}, "not_in_signature": []}
    for k, v in fold.items():
        if k not in params or params[k].default is inspect.Parameter.empty:
            out["not_in_signature"].append(k); continue
        out["checked"] += 1
        if params[k].default != v:
            out["mismatch"][k] = {"passed": v, "default": params[k].default}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -s -m chai1_opt.stock_fold")
    ap.add_argument("--plan", required=True, help="inputs.to_plan JSON (every item of the invocation)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tag", default="off")
    a = ap.parse_args(argv)
    plan = json.load(open(a.plan, "r", encoding="utf-8"))
    tag_dir = os.path.join(os.path.abspath(a.out_dir), a.tag)
    os.makedirs(tag_dir, exist_ok=True)
    proof = env_proof()                                                                   # before torch: the inherited environment is the one proved
    proof["det"] = int(plan.get("det", 0))
    proof["run_kwargs"] = dict(plan["fold"])
    proof["settings_non_default"] = dict(plan.get("non_default") or {})
    from . import det as _det, outputs as _out, report as _rep
    sys.stderr.write(f"{_rep.STOCK_PREFIX} ENV-CLEAN ok: env={len(os.environ)} forbidden=none kit_modules=none kit_dirs=none "
                     f"torch_before_proof={proof['torch_imported_before_proof']} det={proof['det']}\n"); sys.stderr.flush()
    lv = _det.level(plan.get("det", 0))
    for k, v in _det.env_before_torch(lv).items():
        os.environ[k] = v
    import torch                                                                          # noqa: F401
    det_state = _det.apply(lv, torch)
    from pathlib import Path
    import chai_lab
    from chai_lab.chai1 import run_inference
    import chai_lab.chai1 as _chai1_mod
    from . import forward_timer as _ft
    _ft.install(_chai1_mod, _rep.STOCK_PREFIX)                                             # timing only: the FORWARD line around run_folding_on_context
    fold = dict(plan["fold"])
    proof.update(det_applied=det_state, chai_lab_version=getattr(chai_lab, "__version__", None), torch_version=torch.__version__,
                 defaults_check=defaults_check(fold, run_inference))
    dc = proof["defaults_check"]
    expected = dict(plan.get("non_default") or {})                                        # the knobs the caller gave a non-default value: the only allowed departures from the installed signature's defaults
    got = {k: m["passed"] for k, m in dc["mismatch"].items()}
    if got != expected or dc["not_in_signature"]:                                        # the pass-through gate: stock's defaults with exactly the flags given, or no fold
        sys.stderr.write(f"{_rep.STOCK_PREFIX} refusing: the plan's run_inference keywords are not the installed signature's defaults with exactly the "
                         f"flags given {expected} on this install: {dc}\n"); sys.stderr.flush()
        return 2
    pass_kw = dict(fold, device=plan.get("device") or "cuda:0")                          # the resolved keywords every run_inference call of this pass receives; --device as given, else cuda:0 = what stock resolves None to, chai1.py:510 (+ per item: fasta, out dir, msa_directory, seed)
    pass_kw.update({k: Path(pass_kw[k]) for k in ("constraint_path", "template_hits_path") if pass_kw.get(k)})   # run_inference's Path-typed keywords
    sys.stderr.write(_rep.settings_line(_rep.STOCK_PREFIX, pass_kw) + "\n"); sys.stderr.flush()
    _rep.print_ready("stock", T0, prefix=_rep.STOCK_PREFIX)
    msa_dir = plan.get("msa_dir")
    rows = []
    for item in plan["items"]:                                                            # every item of the invocation, in order, in this one process
        key = item["key"]
        try:
            msa_directory, present, n_chains = msa_directory_for(item["fasta"], msa_dir)
            depth_chains, depths = msa_depths(item["fasta"], msa_dir)
            item_error = None
        except Exception as e:  # noqa: BLE001  — an item whose inputs cannot be read is that item's failed rows, never the end of the pass
            msa_directory, present, n_chains, depth_chains, depths, item_error = None, [], None, 0, [], f"inputs: {e!r}"
        sys.stderr.write(f"{_rep.STOCK_PREFIX} MSA form={'directory' if msa_directory else 'none'} item={key} aligned_pqt={len(present)}/{n_chains} "
                         f"msa_dir={msa_dir or 'none'}\n")
        sys.stderr.write(_rep.msa_depth_line(_rep.STOCK_PREFIX, key, depth_chains, depths, msa_dir) + "\n"); sys.stderr.flush()
        for s in (item["seeds"] or [None]):                                              # no seed named: ONE unseeded call, stock's own behaviour (run_inference(seed=None) sets no seed, chai1.py:570-572), under seed_none/
            od = Path(tag_dir) / key / ("seed_none" if s is None else f"seed_{s}")
            row = {"key": key, "id": item.get("id"), "seed": (None if s is None else int(s)), "tag": a.tag, "settings": plan.get("non_default") or {}, "det": det_state,
                   "msa_form": "directory" if msa_directory else "none", "msa_dir": msa_dir, "aligned_pqt_present": present, "n_chains": n_chains,
                   "status": "fail", "t_start": time.time()}
            t0 = time.perf_counter()
            try:
                if item_error is not None:
                    raise RuntimeError(item_error)
                if od.exists() and any(od.iterdir()):
                    raise RuntimeError(f"{od} exists and is not empty (upstream requires an empty output directory)")
                od.mkdir(parents=True, exist_ok=True)
                # --- the stock call: upstream's run_inference, one call per seed, at the plan's fold keywords
                cand = run_inference(fasta_file=Path(item["fasta"]), output_dir=od, msa_directory=msa_directory, seed=(None if s is None else int(s)), **pass_kw)
                torch.cuda.synchronize()
                row["wall_s"] = time.perf_counter() - t0
                row["forward"] = _ft.last(1)                                              # the fold's own cuda-synced wall inside this call (FORWARD line): item, out, tokens, crop, forward_s
                row["forward_s"] = row["forward"][0]["forward_s"] if row["forward"] else None
                row["n_samples"] = _out.n_samples(cand)
                row["files"] = sorted(os.listdir(od))                                     # upstream's own files, left where it wrote them
                row["status"] = "ok"
            except Exception as e:  # noqa: BLE001
                row["wall_s"] = time.perf_counter() - t0
                row["error"] = (repr(e) + " | " + traceback.format_exc()[-1500:])[:2000]
                sys.stderr.write(f"{_rep.STOCK_PREFIX} SEED FAIL {key} {'none' if s is None else s} {row['error']}\n")   # a failed seed-fold is named with its error; the SEED line below carries the status
            peak = _ft.read_peak()                                                        # the allocator's peak since this seed-fold's fold entry (forward_timer resets the counters at fold entry)
            row["max_mem_alloc_gb"] = round(peak["alloc_gib"], 2) if peak is not None else None
            rows.append(row)
            sys.stderr.write(f"{_rep.STOCK_PREFIX} SEED {key} {'none' if s is None else s} {row['status']} wall={row['wall_s']:.1f}s\n"); sys.stderr.flush()
    n_ok = sum(r["status"] == "ok" for r in rows)
    sys.stderr.write(f"{_rep.STOCK_PREFIX} DONE {n_ok}/{len(rows)} seed-folds ok wall={time.time() - T0:.1f}s\n")
    return 0 if n_ok == len(rows) else 2


if __name__ == "__main__":
    sys.exit(main())
