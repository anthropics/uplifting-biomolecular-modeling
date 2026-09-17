"""Where the kit and the pinned stock are, and how each is checked before anything runs.

The kit is ``opt/datapath/pipeline_tf`` (the frozen dir ``v17/`` = the entry script generated from the stock bytes + ``kitlib/``;
``build.py`` documents that generation; ``tests/`` hold the levers to the stock functions' bytes). Its entry script
``v17/borzoi_sad.py`` keeps the stock CLI and resolves ``kitlib`` beside itself, so nothing here is imported at run time — the package
only locates (the frozen dir and its entry present) and execs it with its one composition (``KIT_COMPOSITION``: the kit's forward
call switched on — the traced graph from the second call, the copy-free return). The git commit identifies the kit: this module checks presence and shape, not bytes. The stock entry is
the stock checkout's ``$BORZOI_DIR/src/scripts/borzoi_sad.py`` (else the ``borzoi_sad.py`` on PATH), checked against ``stock/PINS.json`` (the
installed checkout is unpacked from ``stock/``'s archives; the pin says which bytes the entry must have).

Nothing of the kit's vocabulary is transcribed: the writer's name, its tier words and its statistics set are read from
``v17/kitlib/writer.py`` by AST (``writer_table``); the frozen-dir name is checked against ``pipeline_tf/__init__.py``'s ``FROZEN_DIRS``.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
from typing import Dict, Optional

KIT_RELPATH = os.path.join("opt", "datapath", "pipeline_tf")     # under the model directory (borzoi/)
FROZEN = "v17"                                                   # the frozen dir
KIT_NAME = f"pipeline_tf.{FROZEN}"                               # the kit's own stamp name (v17/borzoi_sad.py `_KIT_STAMP["kit"]`)
KIT_COMPOSITION = {"KIT_FWD": "1"}                               # the kit switch the package sets in its kit mode: the kit's forward call (v17/kitlib/forward.py: call 1 eager,
                                                                 # calls 2.. one traced graph, XLA off; the copy-free return — the same float32 values as the stock call)
ENTRY = "borzoi_sad.py"                                          # the entry of the `sad` verb (the stock script's name)
SED_ENTRY = "borzoi_sed.py"                                      # the stock's second program: no program-specific levers; under BORZOI_OPT it gets the library levers like any program
STOCK_SCRIPTS_REL = os.path.join("src", "scripts")               # the stock checkout's script dir ($BORZOI_DIR/src/scripts)
PINS_RELPATH = os.path.join("stock", "PINS.json")


class KitError(RuntimeError):
    """The kit or the pinned stock is not where it should be, or the stock's bytes are not the pinned bytes."""


def sha256_file(path: str, bufsize: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def model_root() -> str:
    """The ``borzoi/`` directory: ``$MODEL_OPT`` when set (run.sh / configs export it), else the tree this package is installed from
    (editable install: ``<borzoi>/opt/borzoi_opt/kit.py``)."""
    env = os.environ.get("MODEL_OPT")
    if env:
        return os.path.abspath(env)
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def kit_dir(root: Optional[str] = None) -> str:
    return os.path.join(root or model_root(), KIT_RELPATH)


def frozen_name() -> str:
    """The frozen dir."""
    return FROZEN


def kit_name() -> str:
    return KIT_NAME


def frozen_dir(root: Optional[str] = None) -> str:
    return os.path.join(kit_dir(root), FROZEN)


def kit_entry(script: str = ENTRY, root: Optional[str] = None) -> str:
    return os.path.join(frozen_dir(root), script)


def kit_tests_dir(root: Optional[str] = None) -> str:
    return os.path.join(kit_dir(root), "tests")


def declared_frozen_dirs(kd: str) -> Optional[tuple]:
    """``FROZEN_DIRS`` of ``pipeline_tf/__init__.py`` by AST (None when the file or the name is absent)."""
    p = os.path.join(kd, "__init__.py")
    if not os.path.isfile(p):
        return None
    for node in ast.parse(open(p, encoding="utf-8").read()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "FROZEN_DIRS" for t in node.targets):
            v = ast.literal_eval(node.value)
            return tuple(v) if isinstance(v, (tuple, list)) else None
    return None


def check_kit(root: Optional[str] = None) -> dict:
    """The kit tree: the frozen dir and its entry present (the git commit identifies the tree — this checks presence and shape,
    not bytes). Raises nothing; ``ok`` says whether the kit may run and ``reason`` names why not."""
    kd, fd = kit_dir(root), frozen_dir(root)
    if not os.path.isdir(kd):
        return {"ok": False, "dir": kd, "frozen": FROZEN, "reason": f"kit directory not found: {kd} (MODEL_OPT={os.environ.get('MODEL_OPT')!r})"}
    entry = os.path.join(fd, ENTRY)
    reasons = []
    if not os.path.isdir(fd):
        reasons.append(f"frozen dir not found: {fd}")
    elif not os.path.isfile(entry):
        reasons.append(f"kit entry not found: {entry}")
    return {"ok": not reasons, "dir": kd, "frozen": FROZEN, "frozen_dir": fd, "entry": entry,
            "declared_frozen_dirs": declared_frozen_dirs(kd), "reason": "; ".join(reasons) if reasons else None}


_ENV_READ = re.compile(r"""(?:environ\.get|environ\[|getenv)\(?\s*['"](KIT_[A-Z0-9_]+)['"]""")


def kit_env_names(root: Optional[str] = None) -> tuple:
    """The kit's OWN switches: every ``KIT_*`` name the frozen dir's code reads from the environment (``os.environ.get("KIT_…")`` /
    ``environ["KIT_…"]`` / ``getenv``) over ``v17/*.py`` and ``v17/kitlib/*.py``, sorted. A ``KIT_``-prefixed name the kit never reads
    (another tool's bookkeeping variable) is not a switch and is not refused."""
    fd = frozen_dir(root)
    names = set()
    for d in (fd, os.path.join(fd, "kitlib")):
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith(".py"):
                names.update(_ENV_READ.findall(open(os.path.join(d, f), encoding="utf-8").read()))
    return tuple(sorted(names))


def writer_table(root: Optional[str] = None) -> dict:
    """The kit's own writer, read from ``v17/kitlib/writer.py`` by AST: ``{"writer": WRITER, "tier": TIER, "stats": WRITER_STATS, "file"}``.
    Never transcribed into this package."""
    p = os.path.join(frozen_dir(root), "kitlib", "writer.py")
    tree = ast.parse(open(p, encoding="utf-8").read())
    consts: Dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in ("WRITER", "TIER", "WRITER_STATS"):
                    consts[t.id] = ast.literal_eval(node.value)
    missing = [k for k in ("WRITER", "TIER", "WRITER_STATS") if k not in consts]
    if missing or not isinstance(consts.get("WRITER"), str) or not isinstance(consts.get("TIER"), str):
        raise KitError(f"{p}: writer table not readable (missing {missing}) — the kit's writer.py changed shape")
    return {"writer": consts["WRITER"], "tier": consts["TIER"], "stats": tuple(consts["WRITER_STATS"]), "file": p}

def pins(root: Optional[str] = None) -> dict:
    p = os.path.join(root or model_root(), PINS_RELPATH)
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def stock_script_pin(script: str = ENTRY, root: Optional[str] = None) -> Optional[dict]:
    """``stock/PINS.json`` "stock_scripts"[<script>] -> {"sha256", "bytes", "path"} (None when the pin file has no entry)."""
    try:
        return dict(pins(root)["stock_scripts"][script])
    except (OSError, KeyError, TypeError, ValueError):
        return None


def stock_entry(script: str = ENTRY, environ=None) -> Optional[str]:
    """The stock script: ``$BORZOI_DIR/src/scripts/<script>`` when BORZOI_DIR is set and the file exists, else the one on PATH."""
    environ = os.environ if environ is None else environ
    bd = environ.get("BORZOI_DIR")
    if bd:
        p = os.path.join(bd, STOCK_SCRIPTS_REL, script)
        if os.path.isfile(p):
            return os.path.abspath(p)
    w = shutil.which(script, path=environ.get("PATH"))
    return os.path.abspath(w) if w else None


def check_stock(script: str = ENTRY, root: Optional[str] = None, environ=None) -> dict:
    """The pinned stock entry: present, and its sha256 == the pin's. ``pinned`` False names why."""
    entry = stock_entry(script, environ)
    pin = stock_script_pin(script, root)
    if entry is None:
        return {"ok": False, "entry": None, "pinned": False, "sha256": None, "pin": pin,
                "reason": f"stock {script} not found (BORZOI_DIR unset or {script} not on PATH)"}
    got = sha256_file(entry)
    if pin is None:
        return {"ok": False, "entry": entry, "pinned": False, "sha256": got, "pin": None, "reason": f"stock/PINS.json has no stock_scripts[{script!r}] pin"}
    ok = got == pin.get("sha256")
    return {"ok": ok, "entry": entry, "pinned": ok, "sha256": got, "pin": pin,
            "reason": None if ok else f"{entry} sha256 {got[:12]}… != the pin {str(pin.get('sha256'))[:12]}… (not the pinned stock)"}


def is_kit_entry(path: str, script: str = ENTRY, root: Optional[str] = None) -> bool:
    """Whether ``path`` is the kit's own entry: the tree's own file, or a copy of the frozen dir (a ``kitlib/`` sibling, its bytes a
    live match against the tree's own current entry — never a stored digest)."""
    if not path:
        return False
    p = os.path.realpath(path)
    try:
        if os.path.samefile(p, kit_entry(script, root)):
            return True
    except OSError:
        pass
    d = os.path.dirname(p)
    if os.path.basename(p) == script and os.path.isdir(os.path.join(d, "kitlib")):
        try:
            return sha256_file(p) == sha256_file(kit_entry(script, root))
        except OSError:
            return False
    return False
