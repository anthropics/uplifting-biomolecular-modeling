"""The stock route: xfold's own CLI, ``run_alphafold.py`` as shipped (the pinned archive's pristine bytes, ``stock/xfold-22bdeed.tar.gz``),
at its shipped defaults — the wrapper adds the inputs, the output directory and the parameters file, nothing else. Two facts of this stack
are named on the line the wrapper prints and in the record, never hidden: (1) pristine xfold refuses the OpenFold3 checkpoint
(``xfold/params.py:747-748``: the layout flag ``xfold.of3.OF3`` must be set before the model is built) and DeepMind's own parameters are not
in the zoo, so the CLI runs on the kit's ported ``xfold`` package (``opt/forward/af3t/af3_torch/xfold/``, the kit's port) with
``of3.OF3 = True`` set by the launcher (``stock_launch.py``) before ``main``; (2) ``--run_data_pipeline=false``: the inputs carry their MSAs and
templates and the image has no sequence databases; (3) the declared patches the carried
``xfold`` package holds (``stock/PINS.json`` ``kit_patches``: today ``AF3_WEIGHT_PORT-04-equivalent`` — empty-template
restype GAP under the OpenFold3 weights — or ``none``) are named as ``kit_patches=`` on the line. Every other flag is the CLI's own
default (``--fastnn`` True: triton layer norm / attention / gated linear unit; ``--num_diffusion_samples`` 5; the model's own recycles).
The interpreter is the composed stock venv (``stock_venv.py``; ``AF3_TORCH_STOCK_PY``, no default; one another account owns or could replace is
refused by name, ``interpreter_refusal``). Outputs are the CLI's own files under
``<output_dir>/<sanitised name>/`` (``ranking_scores.csv`` unprefixed — the CLI's spelling, kept).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat as stat_mode
import tarfile
import time
from typing import List, Optional

from opt_core.gates import sha256_file

from . import stack
from .report import emit, line

ENV_STOCK_PY = "AF3_TORCH_STOCK_PY"
ARCHIVE_MEMBER = "run_alphafold.py"          # the CLI entry inside the archive's one root directory
LAUNCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_launch.py")   # the launcher script (runs on the stock venv; imports nothing of the package)
LAUNCHER_COMPAT = "of3.OF3=True,loaders-as-lists,cached_ccd-as-Ccd"                     # the launcher's named adaptations (stock_launch.py ADAPTATIONS)
PASS_THROUGH_FLAGS = ("--run_data_pipeline=false",)   # the one non-default, named above
# The CLI's own per-seed clock, teed. run_alphafold.py prints, per fold input, `Processing fold input <name>` and, per seed, `Running model
# inference for seed <s> took  <t> seconds.` (its predict_structure: torch.cuda.synchronize() on both sides of ModelRunner.run_inference —
# the model forward of ONE seed over the CLI's samples; featurisation and structure extraction are timed on their own lines). The wrapper
# tees that clock to ONE kit line per (fold, seed): `[af3-torch-opt] STOCK-ITEM name=<fold name> seed=<s> inference_s=<t>` — a timing
# print only; the CLI's transcript stays whole in _stock_cli/stock_cli.log. The two patterns are the archive's format strings
# (tests/test_stock_route.py locks them against the pinned run_alphafold.py source).
UPSTREAM_FOLD_RX = re.compile(r"^Processing fold input (?P<name>.+?)\s*$")
UPSTREAM_SEED_RX = re.compile(r"^Running model inference for seed (?P<seed>-?\d+) took\s+(?P<t>\d+(?:\.\d+)?) seconds\.\s*$")


def kit_patches() -> str:
    """The declared patches of the carried xfold package the CLI runs on, by id (stock/PINS.json kit_patches whose `touches` lie under
    af3_torch/xfold/ — a patch on the kernel adapters is not on the CLI's path), or 'none'."""
    return ",".join(p["id"] for p in stack.pins()["kit_patches"]["patches"] if any(t.startswith("af3_torch/xfold/") for t in p["touches"])) or "none"


def stock_item_teer(emit_line=None):
    """The per-line callback for the CLI's transcript (opt_core.process run_step on_line): remembers the fold input the CLI is processing
    and, on the CLI's per-seed inference clock line, emits ONE `STOCK-ITEM name=<fold> seed=<s> inference_s=<t>` kit line (emit_line
    defaults to this module's emit ∘ line). Returns the callback; its `.items` list holds every (name, seed, inference_s) teed."""
    state = {"name": None}
    items = []

    def on_line(raw: str) -> None:
        text = raw.rstrip("\n")
        m = UPSTREAM_FOLD_RX.match(text)
        if m:
            state["name"] = m.group("name"); return
        m = UPSTREAM_SEED_RX.match(text)
        if m:
            rec = (state["name"] if state["name"] is not None else "UNKNOWN", int(m.group("seed")), float(m.group("t")))
            items.append(rec)
            (emit_line or (lambda **kv: emit(line("STOCK-ITEM", **kv))))(name=rec[0], seed=rec[1], inference_s=m.group("t"))

    on_line.items = items
    return on_line


def stock_python() -> Optional[str]:
    return os.environ.get(ENV_STOCK_PY) or None


def interpreter_refusal(stock_py: str, uid: Optional[int] = None, lstat=os.lstat, stat=os.stat) -> Optional[str]:
    """None when ``stock_py`` may run as the stock interpreter; else why not, by name. The variable has no default: it names the composed venv
    the user built. The entry, the file it resolves to, the directory holding it and the venv directory above that must each belong to this
    uid or to root, and neither directory may be writable by others -- an interpreter another account could have placed or replaced there is
    not run. uid 0 accepts any owner (a container's root over a bind-mounted venv)."""
    uid = os.geteuid() if uid is None else uid
    entry = os.path.abspath(stock_py); bindir = os.path.dirname(entry); venv = os.path.dirname(bindir)
    for p, st in ((entry, lstat(entry)), (entry, stat(entry)), (bindir, lstat(bindir)), (venv, lstat(venv))):
        if uid != 0 and st.st_uid not in (uid, 0):
            return (f"{ENV_STOCK_PY}={stock_py}: {p} belongs to uid {st.st_uid}, not to this user (uid {uid}) or root -- not run; "
                    f"fix: a stock venv of your own (python -m af3_torch_opt.stock_venv --dest DIR; {ENV_STOCK_PY}=DIR/bin/python), or chown it")
        if p != entry and stat_mode.S_ISDIR(st.st_mode) and st.st_mode & 0o002:
            return f"{ENV_STOCK_PY}={stock_py}: {p} is writable by others (mode {st.st_mode & 0o7777:04o}) -- not run; fix: chmod o-w {p}"
    return None


def venv_record(stock_py: str) -> dict:
    """The composed venv's own record (stock_venv.py writes <venv>/stock_venv.json): strategy, compat modules, preflight; {} for another interpreter."""
    p = os.path.join(os.path.dirname(os.path.dirname(stock_py)), "stock_venv.json")
    return json.load(open(p, encoding="utf-8")) if os.path.isfile(p) else {}


def extract_cli(work: str) -> dict:
    """run_alphafold.py extracted from the pinned archive (stock/PINS.json upstream.archive; the upstream commit is named in the
    tar member's own path) into <work>/; returns {path, archive, member, sha256, bytes}."""
    up = stack.pins()["upstream"]; arch = up["archive"]
    src = os.path.join(stack.home(), "stock", arch["file"])
    if not os.path.isfile(src):        # a tree that arrived without the archive: `run.sh install` fetches it (stock/fetch_upstream.py) — named, not a bare traceback
        raise FileNotFoundError(f"stock/{arch['file']} is not in this tree — `run.sh install` makes it present (stock/fetch_upstream.py fetches "
                                f"{up['repo']}/archive/{up['commit']}.tar.gz); the stock route reads xfold's own {ARCHIVE_MEMBER} out of it")
    os.makedirs(work, exist_ok=True)
    with tarfile.open(src) as t:
        members = [m for m in t.getmembers() if m.name.count("/") == 1 and m.name.endswith("/" + ARCHIVE_MEMBER)]
        if len(members) != 1:
            raise RuntimeError(f"{src}: expected one <root>/{ARCHIVE_MEMBER}, found {[m.name for m in members]}")
        data = t.extractfile(members[0]).read()
    path = os.path.join(work, ARCHIVE_MEMBER)
    open(path, "wb").write(data)
    return {"path": path, "archive": arch["file"], "member": members[0].name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def argv_for(stock_py: str, cli_path: str, json_path: Optional[str], input_dir: Optional[str], output_dir: str) -> List[str]:
    """The pass-through command for ONE CLI process: interpreter, the launcher script, the CLI, then ONLY the input (one ``--json_path`` — the
    CLI's flag is one string, a repeated flag keeps the last value silently — or one ``--input_dir``), the output dir, the parameters file and
    the named non-default."""
    if bool(json_path) == bool(input_dir):
        raise ValueError("exactly one of json_path / input_dir")
    argv = [stock_py, LAUNCH, cli_path]
    argv += ["--json_path", json_path] if json_path else ["--input_dir", input_dir]
    argv += ["--output_dir", output_dir, "--model_dir", stack.checkpoint_path()]     # the RESOLVED checkpoint file (stack.resolve_checkpoint: the pinned name, else the dir's one parameters file)
    argv += list(PASS_THROUGH_FLAGS)
    return argv


def env_for() -> dict:
    """The model-process environment (the deployment variables, no AF3_TORCH_OPT*, the cache dirs) plus the ported package on the path."""
    env = stack.model_process_env()
    env["PYTHONPATH"] = os.path.join(stack.kit_home(), "af3_torch")      # the kit's xfold/ (the port): the only package the launcher imports before the CLI
    return env


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--json_path", action="append", default=[], help="a fold input (repeatable: one CLI process per input — xfold's --json_path is one string)")
    p.add_argument("--input_dir", default=None, help="a directory of fold inputs (one CLI process; not with --json_path)")
    p.add_argument("--output_dir", required=True)


def run(a, run_step) -> int:
    """The verb: gates (interpreters, parameters, kit dirs), the CLI extracted and sha-gated, ONE CLI PROCESS PER INPUT (``--json_path``
    repeated = that many processes, in order; ``--input_dir`` = one process over the directory; the two are not mixed — the CLI reads one
    ``--json_path`` string and would keep only the last one, silently); the wrapper's lines: STOCK-CLI (what runs, its named facts) once,
    the COMMAND / STEP pair per process, one STOCK-ITEM line per (fold input, seed) teed from the CLI's own per-seed inference clock
    (stock_item_teer), DONE / FAILED once with the first failing rc."""
    if not a.json_path and not a.input_dir:
        emit(line("USAGE", error="no input: --json_path or --input_dir")); return 2
    if a.json_path and a.input_dir:
        emit(line("USAGE", error="--json_path and --input_dir are not mixed: one process per --json_path, or one --input_dir")); return 2
    stock_py = stock_python()
    why = [w for w in stack.gates() if not w.startswith(("AF3_TORCH_PY=", "AF3_TORCH_JAX_PY="))]   # the kit's interpreters are not this route's
    if not stock_py or not os.access(stock_py, os.X_OK):
        why.append(f"{ENV_STOCK_PY}={stock_py or '<unset>'} is not an executable interpreter (no default: the composed stock venv, python -m af3_torch_opt.stock_venv --dest DIR; {ENV_STOCK_PY}=DIR/bin/python)")
    else:
        refusal = interpreter_refusal(stock_py)                              # by name: an interpreter another account owns or could replace is not run
        if refusal:
            why.append(refusal)
    if why:
        emit(line("NOT ACTIVE", reason="; ".join(why))); return 3
    out = os.path.abspath(a.output_dir); os.makedirs(out, exist_ok=True)
    work = os.path.join(out, "_stock_cli")
    cli = extract_cli(work)
    runs = [(os.path.abspath(p), None) for p in a.json_path] or [(None, os.path.abspath(a.input_dir))]
    argvs = [argv_for(stock_py, cli["path"], jp, d, out) for jp, d in runs]
    vr = venv_record(stock_py)
    weights = stack.weights_record(stack.checkpoint_path())                 # the checkpoint's digest against the pins: pinned, or ONE UNPINNED line and the CLI runs
    emit(line("STOCK-CLI", entry=cli["member"], cli_sha256=cli["sha256"], interpreter=stock_py, of3_layout_flag="set-by-launcher", launcher_sha256=sha256_file(LAUNCH),
              non_default=",".join(PASS_THROUGH_FLAGS), params=os.path.basename(argvs[0][argvs[0].index("--model_dir") + 1]), weights=stack.weights_word(weights), xfold_package=os.path.join(stack.kit_home(), "af3_torch", "xfold"),
              kit_patches=kit_patches(), launcher_compat=LAUNCHER_COMPAT, venv_strategy=vr.get("strategy", "?"), venv_compat=",".join(vr.get("compat_modules") or []) or "none", n_processes=len(argvs)))
    stack.warn_unpinned(weights)                                             # the ONE UNPINNED line (once per process; never silenced)
    t0 = time.time(); rc = 0
    for i, argv in enumerate(argvs):
        rec = run_step("stock_cli", argv, env_for(), os.path.join(work, "stock_cli.log" if len(argvs) == 1 else f"stock_cli.{i}.log"), None,
                       on_line=stock_item_teer())                              # the CLI's per-seed clock teed as STOCK-ITEM lines; the transcript stays whole in the log
        if not rec["ok"] and rc == 0:
            rc = rec["rc"]
    emit(line("DONE" if rc == 0 else "FAILED", step="stock_cli", rc=rc, n_processes=len(argvs), wall_s=round(time.time() - t0, 1), output_dir=out))
    return 0 if rc == 0 else 1
