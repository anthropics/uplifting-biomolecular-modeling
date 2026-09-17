"""The stock caller (`pred --mode off`), its proof, and the one launcher every `pred` uses.

`off` = `colabfold_batch` — the wheel's console script (`colabfold.batch:main`) resolved BESIDE this interpreter
(`<dirname(sys.executable)>/colabfold_batch`, its shebang naming this interpreter: the script's process is the one whose site-packages carry
this package's .pth) — with the caller's colabfold_batch options (verbatim), `--data <parameters root>` when they carry none, and the
caller's two positionals as written, launched in a cleaned environment: every variable with a must-be-absent prefix stripped (stock/PINS.json
stock_proof.must_be_absent_prefixes — the package's switch COLABFOLD_OPT* and the kit's AF_PALLAS_ATTN*) and every PYTHONPATH entry that
is or carries a kit directory removed. When no script sits beside the interpreter, or the script's shebang names something else (another
interpreter; pip's `#!/bin/sh` wrapper for a long interpreter path; `#!/usr/bin/env python`), the launch is the import form on the
interpreter itself, `python -c "import sys; from colabfold.batch import main; sys.exit(main())"` — the console script's own body (the
wheel's entry point `colabfold_batch = colabfold.batch:main`; the module's `__main__` guard calls the same `main`, colabfold/batch.py:2254-2255),
the same process, on which the .pth finder fires (launch_form; the reason recorded in the proof). LaunchError — named, never a silent other
interpreter — only when neither form exists. Never `python -m colabfold.batch`: runpy executes the module as __main__ through the loader's
get_code, not exec_module, and the finder never fires. The proof printed before the launch (`[colabfold-opt stock] STOCK ...`): the
stripped-prefix scan of the launched environment is empty and no kit directory is on PYTHONPATH. Nothing from the kit is imported: with COLABFOLD_OPT absent the .pth installs no finder (_autoload.py).

`fast` = the same launcher with COLABFOLD_OPT=fast exported into the child's environment — every
AF_PALLAS_ATTN* variable of the caller stripped first, so the kit sees only the switch the mode exports in the model process — and the
launch's id (COLABFOLD_OPT_LAUNCH_ID, manifest.ENV_LAUNCH_ID) that the model process's manifest records and `pred`'s verdict requires, and
the launch's work directory (COLABFOLD_OPT_WORK_DIR, manifest.ENV_WORK_DIR) where that manifest is written:
the env route, the one activation implementation (stack.py), in the process that runs the model.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from typing import Callable, List, Optional

from opt_core import process as _core_process

from . import ablation as _ablation, manifest as _manifest, modes as _modes, report as _report
from .stack import KIT_MODULE_FILE, PACKAGE_ENV_PREFIXES, kit_home, pins

STOCK_CLI = "colabfold_batch"                                             # stock/PINS.json stock_proof.cli; the wheel's console script
IMPORT_FORM = "import sys; from colabfold.batch import main; sys.exit(main())"   # the console script's body (entry point colabfold.batch:main)
IMPORT_FORM_NAME = "import-form"                                          # the STOCK line's token for it (whitespace-free; the argv is in the proof file)
FORMS = (STOCK_CLI, IMPORT_FORM_NAME)


class LaunchError(RuntimeError):
    """No launch form for the stock command line on this interpreter (named; never a silent other interpreter)."""


def script_interpreter(path: str) -> Optional[str]:
    """The interpreter a console script's shebang names (`#!<path> [flags]`), None when the file has no `#!` line. A pip wrapper
    (`#!/bin/sh` plus an `exec <python>` line, written for a long interpreter path) or `#!/usr/bin/env python` names the wrapper, not
    the interpreter."""
    with open(path, "rb") as fh:
        first = fh.readline(512)
    if not first.startswith(b"#!"):
        return None
    return first[2:].decode("utf-8", "replace").strip().split()[0] if first[2:].strip() else None


def importable(name: str) -> bool:
    """Whether `name` (a top-level module) is importable on this interpreter: found on sys.path, or already imported."""
    if name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def launch_form(executable: Optional[str] = None) -> dict:
    """The stock command line's launch form on this interpreter: {argv, form, script, shebang, reason}.
    `form` = STOCK_CLI when the console script sits beside the interpreter (`<dirname(executable)>/colabfold_batch`) and its shebang names
    that interpreter — the process whose site-packages carry this package's .pth; else IMPORT_FORM_NAME: `python -c "<IMPORT_FORM>"` on the
    interpreter itself — the script's body in the same process (only sys.argv[0] differs), on which the finder fires — taken when no script
    sits beside the interpreter or the script's shebang names something else (another interpreter, a `/bin/sh` wrapper, `env`), with the
    reason recorded. LaunchError (named; rc 3 in `pred`) only when neither form exists: no usable script and no importable `colabfold` on
    this interpreter (probed when `executable` is this process's interpreter; another interpreter's imports are not probed)."""
    py = executable or sys.executable
    script = os.path.join(os.path.dirname(py), STOCK_CLI)
    d = {"argv": None, "form": None, "script": script if os.path.isfile(script) else None, "shebang": None, "reason": None}
    if d["script"]:
        d["shebang"] = script_interpreter(script)
        if d["shebang"] is not None and os.path.realpath(d["shebang"]) == os.path.realpath(py):
            d.update(argv=[script], form=STOCK_CLI, reason=f"the console script beside the interpreter, its shebang naming it ({d['shebang']})")
            return d
        why = f"{script} has the shebang {d['shebang']!r}, not this interpreter ({py})"
    else:
        why = f"no {STOCK_CLI} beside {py}"
    if os.path.realpath(py) == os.path.realpath(sys.executable) and not importable("colabfold"):
        raise LaunchError(f"no launch form for the stock command line: {why}, and colabfold is not importable on {py} "
                          f"(pip install colabfold[alphafold]==1.6.1 on it)")
    d.update(argv=[py, "-c", IMPORT_FORM], form=IMPORT_FORM_NAME, reason=f"{why}: the import form runs the script's body on this interpreter")
    return d


def cli_argv(executable: Optional[str] = None) -> List[str]:
    """launch_form(executable)["argv"]."""
    return launch_form(executable)["argv"]


def cli_name(argv: List[str]) -> str:
    return STOCK_CLI if os.path.basename(argv[0]) == STOCK_CLI else IMPORT_FORM_NAME


def compose(options: Optional[List[str]], input_path: str, results: str, data_dir: Optional[str] = None, dashdash: bool = False) -> List[str]:
    """argv of the model process: the stock entry, the caller's colabfold_batch options verbatim (none = upstream's defaults), `--data <dir>`
    when the options do not carry it (the parameters root from COLABFOLD_OPT_DATA_DIR), then colabfold_batch's two positionals as the
    caller wrote them (after a `--` when the caller used one)."""
    argv = cli_argv() + list(options or [])
    if data_dir:
        argv += ["--data", data_dir]
    if dashdash:
        argv.append("--")
    argv += [input_path, results]
    return argv

def _prefixes() -> tuple:
    return tuple(pins().get("stock_proof", {}).get("must_be_absent_prefixes", list(PACKAGE_ENV_PREFIXES)))


def _is_kit_dir(p: str) -> bool:
    """A PYTHONPATH entry that is a kit's importable directory (carries af2_pallas_attn.py) or a kit root (carries
    af2_pallas_flash/af2_pallas_attn.py) — any copy, not only the package's own."""
    return os.path.isfile(os.path.join(p, _modes.KIT_MODULE + ".py")) or os.path.isfile(os.path.join(p, KIT_MODULE_FILE))


def _kit_dirs_on(pythonpath: Optional[str]) -> List[str]:
    try:
        kit = os.path.realpath(kit_home())
    except FileNotFoundError:
        kit = None
    out = []
    for p in (pythonpath or "").split(os.pathsep):
        if p and ((kit and os.path.realpath(p).startswith(kit)) or _is_kit_dir(p)):
            out.append(p)
    return out


def stock_env(environ: Optional[dict] = None) -> dict:
    """The stock process's environment: the caller's minus the must-be-absent prefixes, minus kit directories on PYTHONPATH."""
    environ = dict(os.environ if environ is None else environ)
    prefixes = _prefixes()
    env = {k: v for k, v in environ.items() if not k.startswith(prefixes) and k != _ablation.ENV}   # + the ablation switch (MODEL_OPT_LEVERS_OFF, refused under off before any launch): the stock child never carries it
    if "PYTHONPATH" in env:
        bad = set(_kit_dirs_on(env["PYTHONPATH"]))
        keep = [p for p in env["PYTHONPATH"].split(os.pathsep) if p not in bad]
        if keep:
            env["PYTHONPATH"] = os.pathsep.join(keep)
        else:
            env.pop("PYTHONPATH")
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def fast_env(mode: str, environ: Optional[dict] = None, launch_id: Optional[str] = None, work_dir: Optional[str] = None) -> dict:
    """The kit-mode process's environment: the caller's minus each of the kit's variables (AF_PALLAS_ATTN*: the model process sees only the switch
    the mode exports, stack.activate), with the package's switch exported (the env route), the launch id the model process's manifest
    must carry (manifest.ENV_LAUNCH_ID; the verdict requires it), the launch's work directory (manifest.ENV_WORK_DIR: where that manifest is
    written; absent: none is written)."""
    env = {k: v for k, v in dict(os.environ if environ is None else environ).items() if not k.startswith(_modes.KIT_SWITCH)}
    env[_modes.ENV] = mode
    if launch_id:
        env[_manifest.ENV_LAUNCH_ID] = launch_id
    env.pop(_manifest.ENV_WORK_DIR, None)
    if work_dir:
        env[_manifest.ENV_WORK_DIR] = work_dir
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env

def proof(env: dict, argv: List[str]) -> dict:
    prefixes = _prefixes()
    present = sorted(k for k in env if k.startswith(prefixes) or k == _ablation.ENV)
    kit_dirs = _kit_dirs_on(env.get("PYTHONPATH"))
    return {"cli": cli_name(argv), "argv0": argv[0], "launch": launch_form(), "prefixes": list(prefixes), "env_present": present,
            "kit_dirs": kit_dirs, "ok": not present and not kit_dirs}


def proof_line(p: dict) -> str:
    return _report.stock_line(p)


def run_logged(argv: List[str], env: dict, cwd: str, on_line: Callable[[str], None]) -> int:
    """Run argv to its own exit, stream every output line to on_line; the return code — opt_core.process.run_logged (undecodable bytes are replaced)."""
    return _core_process.run_logged(argv, timeout_s=None, on_line=on_line, env=env, cwd=cwd).rc
