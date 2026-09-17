"""An on-disk stand-in for the stock ``protenix`` command line (``runner.batch_inference:protenix_cli``) for the clean subprocess of
``pred --mode off``: a click group shaped like the stock one whose ``pred`` writes ``<out_dir>/stock_stub_run.json`` — the argv it got,
the kit-named environment variables, the kit modules loaded, sys.path and the sitecustomize of the process that ran it."""
import os
import textwrap

STOCK_INFERENCE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "stock", "src", "runner", "inference.py"))


def pinned_update_inference_configs():
    """``update_inference_configs`` of the pinned ``runner/inference.py`` (stock/src), compiled from that source alone: the protenix-v2 size
    assertion and the runner's skip_amp policy by token count, as shipped — a test that needs the stock policy calls this, never a copy."""
    import ast
    src = open(STOCK_INFERENCE, encoding="utf-8").read()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "update_inference_configs")
    ns = {"Any": object}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), STOCK_INFERENCE, "exec"), ns)     # noqa: S102 — the pinned function's own statements
    return ns["update_inference_configs"]

RUN_RECORD = "stock_stub_run.json"
PROOF_NAME = "stock_env_proof.json"                                  # cli.STOCK_PROOF_NAME: the proof the stock subprocess writes (a temporary directory of the call)
KIT_MARKER_MODULE = "ptx_drop_bond_mask"                                 # a kit-named module (a name in stock_pred.KIT_MODULE_PREFIXES) the stub imports when told to (--input IMPORT_KIT)
KIT_ENV_PREFIXES = ("PTX_", "FPF_", "INFOPT_", "PF_", "PROTENIX_OPT", "CUEQ_TRITON_CACHE_DIR")
KIT_MODULE_PREFIXES = ("ptx_", "fpf", "infopt_graphs")

# The stock `pred` options both stubs declare (stock/src/runner/batch_inference.py:600-690: spelling, type and default as pinned) —
# the caller's input / out_dir, then the stock knobs a caller passes through. Both stubs parse with exactly this table, so a token
# the stock parser would reject fails here too; a stub's recorded ``params`` is the full parsed set (STOCK_PRED_DEFAULTS + the argv's).
PRED_OPTIONS = (
    (("-i", "--input"), str, None), (("-o", "--out_dir"), str, "./output"), (("-s", "--seeds"), str, "101"),
    (("-c", "--cycle"), int, 10), (("-p", "--step"), int, 200), (("-e", "--sample"), int, 5), (("-d", "--dtype"), str, "bf16"),
    (("-n", "--model_name"), str, "protenix_base_default_v1.0.0"), (("--use_msa",), bool, True),
    (("--trimul_kernel",), str, "cuequivariance"), (("--triatt_kernel",), str, "cuequivariance"),
    (("--use_template",), bool, False), (("--use_rna_msa",), bool, False), (("--need_atom_confidence",), bool, False),
)
STOCK_PRED_DEFAULTS = {names[-1].lstrip("-"): default for names, _, default in PRED_OPTIONS if default is not None}
SHORT_ENV = "STOCK_STUB_SHORT"                                       # <n>: the stub leaves the last n expected files unwritten (an incomplete run)


def write_outputs(params: dict) -> list:
    """Write the stock dumper's output layout for ``params`` (every entry of the input × seeds × samples: the CIF and its summary
    JSON under <out_dir>/<name>/seed_<seed>/predictions/), minus the last $STOCK_STUB_SHORT files; returns the files written."""
    from protenix_opt import outputs
    exp = outputs.expected(params)
    short = int(os.environ.get(SHORT_ENV, "0") or 0)
    files = exp["files"][: len(exp["files"]) - short] if short else exp["files"]
    for rel in files:
        p = os.path.join(params["out_dir"], rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("{}\n" if rel.endswith(".json") else "data_stub\n")
    return files


def input_json(path: str, names=("p1",)) -> str:
    """A stock-shaped input file: a list of entries with ``name``; returns ``path``."""
    import json
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([{"name": n, "sequences": []} for n in names], fh)
    return path


def pred_command(body):
    """``body(**params)`` as the stub ``pred`` click command declaring PRED_OPTIONS (help names ``-h/--help`` like the stock group)."""
    import click
    cmd = body
    for names, typ, default in reversed(PRED_OPTIONS):
        cmd = click.option(*names, type=typ, default=default, required=default is None)(cmd)
    return click.command(context_settings=dict(help_option_names=["-h", "--help"]))(cmd)


_OPTIONS_TEXT = "\n".join(f"    @click.option({', '.join(repr(n) for n in names)}, type={typ.__name__}, default={default!r}"
                          f"{', required=True' if default is None else ''})" for names, typ, default in PRED_OPTIONS)

_BATCH_INFERENCE = textwrap.dedent(f'''
    import json, os, sys
    import click

    @click.group(context_settings=dict(help_option_names=["-h", "--help"]))
    def protenix_cli():
        pass

    @click.command(context_settings=dict(help_option_names=["-h", "--help"]))
{_OPTIONS_TEXT}
    def predict(**params):
        input, out_dir = params["input"], params["out_dir"]
        os.makedirs(out_dir, exist_ok=True)
        proof_path = ""                                                   # the proof the wrapper (protenix_opt.stock_pred) was told to write: --proof-json <path> on this process's own command line
        try:
            argv0 = open(f"/proc/{{os.getpid()}}/cmdline", "rb").read().split(bytes([0]))
            argv0 = [a.decode() for a in argv0]
        except OSError:
            argv0 = list(sys.argv)
        if "--proof-json" in argv0:
            proof_path = argv0[argv0.index("--proof-json") + 1]
        proof_at_call = None
        if proof_path and os.path.isfile(proof_path):
            with open(proof_path, encoding="utf-8") as fh:
                proof_at_call = {{"keys": sorted(json.load(fh))}}
        if input == "IMPORT_KIT":
            import {KIT_MARKER_MODULE}  # noqa: F401
        sc = sys.modules.get("sitecustomize")
        rec = {{"argv": sys.argv[1:], "params": params,
               "env_kit_names": sorted(k for k in os.environ if k.startswith({KIT_ENV_PREFIXES!r})),
               "modules_kit": sorted(m for m in sys.modules if m.startswith({KIT_MODULE_PREFIXES!r})),
               "sys_path": list(sys.path), "pythonpath": os.environ.get("PYTHONPATH"),
               "sitecustomize": getattr(sc, "__file__", None) if sc is not None else None,
               "no_user_site": bool(sys.flags.no_user_site), "torch_loaded": "torch" in sys.modules, "pid": os.getpid(),
               "proof_at_call": proof_at_call}}
        with open(os.path.join(out_dir, {RUN_RECORD!r}), "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=1)
        if input == "FAIL":
            sys.exit(5)
        from protenix_opt.tests._stock_stub import write_outputs
        write_outputs(params)

    protenix_cli.add_command(predict, name="pred")
''')


def write_stub_runner(root: str) -> str:
    """Write the ``runner`` package under ``root`` and return ``root`` (a sys.path / PYTHONPATH entry)."""
    pkg = os.path.join(root, "runner")
    os.makedirs(pkg, exist_ok=True)
    with open(os.path.join(pkg, "__init__.py"), "w", encoding="utf-8") as fh:
        fh.write("")
    with open(os.path.join(pkg, "batch_inference.py"), "w", encoding="utf-8") as fh:
        fh.write(_BATCH_INFERENCE)
    with open(os.path.join(root, KIT_MARKER_MODULE + ".py"), "w", encoding="utf-8") as fh:
        fh.write("MARKER = True\n")
    return root
