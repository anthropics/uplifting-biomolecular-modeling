"""The stock route: `protenix pred` as pinned, in a process that proves nothing of the kit is on it — the shared core's clean-process
contract (opt_core.stock_proof) with this kit's names.

`cli._run_stock` (for `--mode off`, and nothing else) launches `python -s -m protenix_v1_opt.stock_pred [--det] --proof-json <path>
--env-absent <prefixes> --kit-dirs <kit dir> --module-prefixes <kit modules> -- <protenix pred args>` (opt_core.stock_proof.stock_command +
this module's one extra option `--det`). Before protenix is imported the child runs the core's proof (stock_proof.env_proof: no environment
name under the stock pin's must-be-absent prefixes — the caller strips them —, no kit module loaded, no kit directory on sys.path, no
autoload finder armed, no kit sitecustomize, no torch yet, no core module beyond the proof machinery), writes it to `--proof-json`
(stock_proof.write_proof) and prints it as one line (`[protenix-v1-opt stock] PROOF {...}`); it exits 3 without running anything when the
proof fails (`NOT STOCK: <stock_proof.violations_sentence>`). Otherwise it runs the stock CLI in-process (`runner.batch_inference.protenix_cli`),
adds the after-call scan (stock_proof.after_call_check: kit modules loaded by the call) to the proof file, and exits with the CLI's code.
`--det` = the deterministic recipe on the stock arm: the kit's two detpatch files as protenix.utils.{det_segment_reduce,scatter_utils} + the
recipe's variables (det.py) — installed after the proof, recorded in the proof as "det". The per-item PHASE timing line (phase.py:
the runner's predict and the three Protenix stage methods wrapped, timing only) is installed before the call and recorded as "phase";
det.py and phase.py are the only kit bytes in this process.
This module imports nothing that reads the kit (no `protenix_v1_opt.kit`, no PINS reader): FORBIDDEN_PREFIXES / KIT_MODULES are the
same values as PINS.json / the carried kit's module names (locked equal by the package tests) and travel to the child on its command line.
"""
from __future__ import annotations

import json
import sys

from opt_core import stock_proof as SP

PREFIX = "[protenix-v1-opt stock]"
MODULE = "protenix_v1_opt.stock_pred"
FORBIDDEN_PREFIXES = ("PROTENIX_V1_OPT", "PTX_", "FPF_", "INFOPT_", "PROTENIX_DET_SCATTER")   # == PINS.json stock_environment.must_be_absent_prefixes
KIT_MODULES = ("levers_ptx1", "ptx1_", "fpf_trimul_v4", "fpf_trimul", "infopt_graphs", "dit_hoist",
               "fastln_prebuilt", "lnl_fused", "flash_triattn", "biascache_static", "apb_ptx1",
               "trunk2_ptx1", "protenix_fpf_msa", "ditfast_ptx1", "protenix_fpf_ditfast")   # every top-level module name the kit loads: the names under its sys.path roots (test_stock_route locks the list against the tree), the core kernels imported by name, levers_ptx1's biascache_static alias
DET_FLAG = "--det"


def command(python: str, *, proof_json: str, kit_dirs, stock_args, det: bool) -> list:
    """The child's argv (stock_proof.stock_command + `--det` first among the own options when the recipe is requested)."""
    cmd = SP.stock_command(python, MODULE, proof_json=proof_json, env_absent=FORBIDDEN_PREFIXES, kit_dirs=list(kit_dirs), args=list(stock_args),
                           module_prefixes=KIT_MODULES)
    if det:
        cmd.insert(cmd.index("--proof-json"), DET_FLAG)
    return cmd


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    own, _ = SP.split_argv(argv)
    det = DET_FLAG in own
    if det:
        argv.remove(DET_FLAG)
    opts, stock_args = SP.parse_stock_argv(argv, prog=f"python -s -m {MODULE}")
    proof = SP.env_proof(env_absent=opts.env_absent, kit_dirs=opts.kit_dirs, module_prefixes=opts.module_prefixes)
    proof["det"] = None
    if det and proof["ok"]:
        from protenix_v1_opt import det as D                        # the recipe's installer (det.py reads the kit's two detpatch files; named in the proof)
        try:
            proof["det"] = D.install()
        except Exception as e:
            proof["ok"] = False
            proof["det"] = {"installed": False, "reason": repr(e)}
    SP.write_proof(opts.proof_json, proof)
    print(f"{PREFIX} PROOF {json.dumps(proof, sort_keys=True, default=str)}", file=sys.stderr, flush=True)
    if not proof["ok"]:
        why = SP.violations_sentence(proof) if proof.get("det") is None or (proof["det"] or {}).get("installed", True) else f"det recipe: {proof['det'].get('reason')}"
        print(f"{PREFIX} NOT STOCK: {why}; nothing ran", file=sys.stderr, flush=True)
        return SP.EXIT_NOT_STOCK
    from runner.batch_inference import protenix_cli
    from protenix_v1_opt import phase as PH                        # the per-item PHASE line (phase.py: timing only, the stock methods wrapped per item; named in the proof)
    proof["phase"] = PH.install(PREFIX)
    SP.write_proof(opts.proof_json, proof)
    rc = 0
    try:
        protenix_cli.main(args=["pred"] + list(stock_args), prog_name="protenix", standalone_mode=False)
    except SystemExit as e:
        rc = int(e.code or 0) if isinstance(e.code, int) or e.code is None else 1
    proof.update(SP.after_call_check(opts.kit_dirs, opts.module_prefixes, lever_modules=("levers_ptx1",)))
    SP.write_proof(opts.proof_json, proof)
    print(f"{PREFIX} AFTER kit_modules_loaded_after={','.join(proof['kit_modules_loaded_after']) or 'none'} lever_modules_imported={','.join(proof['lever_modules_imported']) or 'none'} after_ok={proof['after_ok']}",
          file=sys.stderr, flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
