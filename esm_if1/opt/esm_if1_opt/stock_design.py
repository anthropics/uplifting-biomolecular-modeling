"""The proven stock child: ``python -s -m esm_if1_opt.stock_design --proof-json <path> --env-absent <prefixes> --kit-dirs <dirs> -- <script>
<script arguments>`` — the command ``design --mode off`` launches once per structure (``opt_core.stock_proof.stock_command``) in an
environment with every name under the must-be-absent prefixes stripped.

Before importing torch or esm, this process proves what it is (``opt_core.stock_proof.env_proof``: no forbidden environment name, no kit
lever module loaded or reachable, no autoload finder armed, the kit's sitecustomize not this process's, torch not yet imported, no core
module beyond the proof machinery), records which data-path variables are present, writes the proof to ``--proof-json``
(``<out>/stock_env_proof.json``) and prints ONE line: ``[esm_if1-opt] ENV-CLEAN ok (proof: … torch_preloaded=False; present: …)`` — or
``[esm_if1-opt] NOT STOCK: <violations>`` and exit 3 with nothing run. A leading ``--upstream-fix <paths>`` pair after ``--`` (present only
when the run named ``--upstream-fix <ID>``) is applied next (``upstream_fix.apply_files``: one ``UPSTREAM-FIX <ID> applied`` line each). Then upstream's example script — the next argument,
``stock/src/examples/inverse_folding/sample_sequences.py`` of the tree, byte-identical to the pinned commit — runs in this very process as
``__main__`` with the remaining arguments as its ``sys.argv`` (``runpy.run_path``): its own argument parser, loader, sampling loop, prints and
output file, nothing of the kit between it and the model. When it returns, two report lines follow (``lines.Clocks.finish``: PEAK — the
allocator's high-water mark — and KERNELS, read from the torch the script imported; nothing is computed); a script that raises ends this
process with the interpreter's own traceback and status. At module level this file imports only the standard library,
``opt_core.stock_proof`` and the package's inert ``__init__`` (the tag).
"""
import os
import runpy
import sys

from opt_core import stock_proof as _proof

from . import TAG                                       # the package __init__ is inert (no core, no sibling import): safe before the proof

PREFIX = "[" + TAG + "]"
DATA_PATH_ENV = ("TORCH_HOME", "ESM_IF1_TIMING_JSONL", "MODEL_OPT", "MODEL_OPT_STATE", "ESM_IF1_WEIGHTS", "CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF")   # recorded as present, never judged


def env_clean_line(proof):
    """``[esm_if1-opt] ENV-CLEAN ok (proof: <core clean sentence> torch_preloaded=<bool>; present: <names|none>)`` — or ``NOT STOCK: <violations>``."""
    if not proof.get("ok"):
        return "%s NOT STOCK: %s" % (PREFIX, _proof.violations_sentence(proof))
    return "%s ENV-CLEAN ok (proof: %s torch_preloaded=%s; present: %s)" % (
        PREFIX, _proof.clean_sentence(proof), proof["torch_loaded_before_proof"], ",".join(proof.get("present") or []) or "none")


def main(argv=None):
    opts, script_argv = _proof.parse_stock_argv(sys.argv[1:] if argv is None else argv, prog="python -s -m esm_if1_opt.stock_design")
    proof = _proof.env_proof(env_absent=opts.env_absent, kit_dirs=opts.kit_dirs, module_prefixes=opts.module_prefixes, det_exception=opts.det_exception)
    proof["present"] = sorted(k for k in os.environ if k in DATA_PATH_ENV)
    proof["sys_path"] = [e for e in sys.path[1:] if e]
    _proof.write_proof(opts.proof_json, proof)
    sys.stderr.write(env_clean_line(proof) + "\n")
    sys.stderr.flush()
    if not proof["ok"]:
        return _proof.EXIT_NOT_STOCK
    from . import upstream_fix                            # standard library only; loads nothing unless the run named a fix
    fixes, script_argv = upstream_fix.split_argv(script_argv)
    if fixes:
        upstream_fix.apply_files(fixes)                   # after the proof, before the script: UPSTREAM-FIX <ID> applied
    if not script_argv or not os.path.isfile(script_argv[0]):
        sys.stderr.write("%s stock_design: the first argument after -- must be upstream's script (got %r)\n" % (PREFIX, script_argv[:1]))
        return 2
    sys.argv = list(script_argv)                         # the script's own argv: sample_sequences.py <pdbfile> --chain … --outpath …
    runpy.run_path(script_argv[0], run_name="__main__")   # upstream's example script as shipped, in this proven process
    sys.stdout.flush()
    from . import lines                                   # after the script: two report lines read from the torch it imported (PEAK, KERNELS)
    lines.Clocks("stock").finish(batch=1, inference_mode=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
