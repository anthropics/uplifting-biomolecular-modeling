"""``install``: the one model-specific step — the add-on's ``install.sh`` applied ONCE into the patched interpreter, and both
interpreters' tree states asserted by sha.

The kit installs by overwriting five ``rf3`` files in the site-packages of whatever ``python`` is on PATH;
a tree with those bytes is not stock (``graph_flags.py`` defaults ``RF3_CUDAGRAPH`` to ``1``). The tree therefore keeps two
interpreters: the pristine one (``stock/venv``, ``ROSETTAFOLD3_OPT_STOCK_PYTHON``) that this command only ever READS, and the patched
one (``opt/venv``, ``ROSETTAFOLD3_OPT_PYTHON``) that receives the kit bytes. There is no uninstall on the stock path: ``off`` never
depends on restoring files.

``--make-venv``: creates the patched interpreter from the pristine one when it does not exist yet — ``<stock python> -m venv
--system-site-packages opt/venv`` plus a copy of the ``rf3`` and ``foundry`` package directories into the venv's own site-packages
(a venv's site-packages precede the system's on ``sys.path``, so ``install.sh`` run with the venv's ``python`` on PATH overwrites the
venv's copy and leaves the pristine tree untouched; the dist metadata stays shared, so the pin check reads the same
``direct_url.json`` in both; a stock interpreter that is itself a venv has its site-packages written into the new venv's
``parent_venv.pth``, since ``--system-site-packages`` chains to that venv's base and not to it; when no ``setuptools`` is reachable, the
lock's ``setuptools`` and ``wheel`` are pip-installed into the venv — the build backend of run.sh's editable installs). From-scratch deployments instead run the pin's install twice; the result is the same two
states. The kit's own install-state line is printed verbatim and recorded.

``--no-addon``: every step but the add-on — the patched interpreter (made when ``--make-venv`` asks), the core, both tree states read —
and the patched interpreter's ``rf3`` tree left exactly as found (pristine after ``--make-venv``), said so on one line. ``environment/Dockerfile``
builds this way, so the five files always come from the kit tree that later runs ``install`` inside the container, never from the image.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from typing import Optional

from . import _core
from . import report as _report
from . import stack as _stack
from . import tree as _tree

_gates = _core.load("gates")

PREFIX = _report.PREFIX + " install"


def _venv_site_packages(python: str) -> str:
    code = "import sysconfig; print(sysconfig.get_paths()['purelib'])"
    return subprocess.run([python, "-c", code], capture_output=True, text=True, timeout=60, check=True).stdout.strip()


def _setuptools_pin() -> str:
    """``setuptools==<v>`` as ``environment/requirements.lock`` pins it (bare ``setuptools`` when the lock has no such line)."""
    lock = os.path.join(os.path.dirname(_stack.opt_root()), "environment", "requirements.lock")
    try:
        with open(lock) as fh:
            for line in fh:
                if line.startswith("setuptools=="):
                    return line.strip()
    except OSError:
        pass
    return "setuptools"


def make_venv(stock_python: str, venv_dir: str, lists: _tree.Digests) -> dict:
    if os.path.exists(os.path.join(venv_dir, "bin", "python")):
        raise RuntimeError(f"{venv_dir} exists already; remove it or point {_stack.ENV_PYTHON} at it")
    ts = _tree.state_of(stock_python, lists)
    _tree.expect(ts, "stock")
    foundry_dir = _tree.locate_of(stock_python, "foundry")
    subprocess.run([stock_python, "-m", "venv", "--system-site-packages", venv_dir], check=True, timeout=600)
    vpy = os.path.join(venv_dir, "bin", "python")
    vsp = _venv_site_packages(vpy)
    parent = json.loads(subprocess.run([stock_python, "-c", "import json, site, sys; print(json.dumps([sys.prefix != sys.base_prefix] + site.getsitepackages()))"],
                                       capture_output=True, text=True, timeout=60, check=True).stdout.strip().splitlines()[-1])
    if parent[0]:                                             # the stock interpreter is itself a venv: --system-site-packages chains to ITS base, not to it,
        with open(os.path.join(vsp, "parent_venv.pth"), "w") as fh:   # so its site-packages (the pinned stack) go on the new venv's path, after the venv's own
            fh.write("\n".join(parent[1:]) + "\n")
    if subprocess.run([vpy, "-c", "import setuptools.build_meta"], capture_output=True, timeout=120).returncode != 0:   # no setuptools reachable:
        subprocess.run([vpy, "-m", "pip", "install", "-q", "--no-deps", _setuptools_pin(), "wheel"], check=True, timeout=600)  # seed the build backend of run.sh's editable installs
    for pkg, base in (("rf3", ts.site_packages), ("foundry", foundry_dir)):
        src = os.path.join(base, pkg)
        dst = os.path.join(vsp, pkg)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
    ts2 = _tree.state_of(vpy, lists)
    vfd = _tree.locate_of(vpy, "foundry")
    if ts2.site_packages != vsp or vfd != vsp:
        raise RuntimeError(f"the venv resolves rf3 from {ts2.site_packages} and foundry from {vfd}, not from its own site-packages {vsp}")
    _tree.expect(ts2, "stock")
    return {"python": vpy, "site_packages": vsp, "copied_from": ts.site_packages, "state": ts2.state}


def apply_kit(opt_python: str, kit_home: str, log_path: Optional[str] = None) -> dict:
    """``bash install.sh --check || bash install.sh`` with the patched interpreter's bin first on PATH (the kit locates
    site-packages with ``python`` from PATH)."""
    env = dict(os.environ)
    env["PATH"] = os.path.dirname(os.path.abspath(opt_python)) + os.pathsep + env.get("PATH", "")
    env.pop("ROSETTAFOLD3_OPT", None)
    out = {"check": None, "apply": None, "kit_line": None}
    r = subprocess.run(["bash", os.path.join(kit_home, "install.sh"), "--check"], capture_output=True, text=True, env=env, cwd=kit_home, timeout=600)
    out["check"] = {"rc": r.returncode, "out": (r.stdout + r.stderr).strip()[-600:]}
    if r.returncode == 0:
        out["kit_line"] = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else None
        out["already"] = True
    else:
        r2 = subprocess.run(["bash", os.path.join(kit_home, "install.sh")], capture_output=True, text=True, env=env, cwd=kit_home, timeout=600)
        out["apply"] = {"rc": r2.returncode, "out": (r2.stdout + r2.stderr).strip()[-800:]}
        out["already"] = False
        if r2.returncode != 0:
            raise RuntimeError(f"install.sh failed (rc={r2.returncode}): {(r2.stdout + r2.stderr).strip()[-600:]}")
        out["kit_line"] = r2.stdout.strip().splitlines()[-1] if r2.stdout.strip() else None
    if log_path:
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(str(out) + "\n")
    return out


def install_core(opt_python: str) -> dict:
    """The pinned core (opt/pyproject.toml [tool.opt_core]) into the patched interpreter — ``pip install -e <pin path>`` unless it already
    imports an opt_core from that path at or above the pinned version — checked by a probe of ``opt_core.gates.imported_core()`` run inside
    that interpreter (this install-time check imports opt_core to ask it; the activation gate's own ``_core_gate.gate`` is a different,
    self-contained check that must work even before any core is installed)."""
    pinned = _gates.core_pin(_core.PYPROJECT)
    root = _core.pin_path()
    probe = [opt_python, "-c", "import json; from opt_core import gates; print(json.dumps(gates.imported_core()))"]

    def found():
        r = subprocess.run(probe, capture_output=True, text=True, timeout=120)
        return json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip().startswith("{") else None

    def is_the_pin(core):
        return (core is not None and os.path.realpath(core["package_dir"]) == os.path.realpath(os.path.join(root, "opt_core"))
                and _gates.version_tuple(core["version"]) >= _gates.version_tuple(pinned["version"]))
    before = found()
    state = "present"
    if not is_the_pin(before):
        r = subprocess.run([opt_python, "-m", "pip", "install", "-q", "--no-build-isolation", "--no-deps", "-e", root], capture_output=True, text=True, timeout=900)
        if r.returncode != 0:
            raise RuntimeError(f"pip install -e {root} into {opt_python} failed (rc={r.returncode}): {(r.stdout + r.stderr).strip()[-600:]}")
        state = "installed" if before is None else "replaced"
    after = found()
    if not is_the_pin(after):
        raise RuntimeError(f"{opt_python} carries opt_core {after and after['version']} at {after and after['package_dir']}, "
                           f"not the pin >= {pinned['version']} at {root} ({_core.PYPROJECT})")
    return {"state": state, "path": root, "version": after["version"], "package_dir": after["package_dir"],
            "line": f"{state}: opt_core {after['version']} ({root}) on {opt_python}"}


def run(*, make_venv_flag: bool = False, addon: bool = True, stock_python: Optional[str] = None, opt_python: Optional[str] = None, log_path: Optional[str] = None) -> dict:
    kh = _stack.kit_home()
    lists = _stack.tree_digests()
    spy = stock_python or _stack.stock_python()
    if not os.path.exists(spy):
        raise RuntimeError(f"stock interpreter not found: {spy} (set {_stack.ENV_STOCK_PYTHON} or create stock/venv)")
    res = {"stock_python": spy, "kit_home": kh}
    ts_stock = _tree.state_of(spy, lists)
    _tree.expect(ts_stock, "stock")
    res["stock"] = {"state": ts_stock.state, "line": ts_stock.line()}
    print(f"{PREFIX} stock interpreter {spy}: {ts_stock.line()}", file=sys.stderr, flush=True)
    opy = opt_python or (os.environ.get(_stack.ENV_PYTHON) or os.path.join(_stack.opt_root(), "venv", "bin", "python"))
    if make_venv_flag and not os.path.exists(opy):
        res["make_venv"] = make_venv(spy, os.path.dirname(os.path.dirname(opy)), lists)
        print(f"{PREFIX} created {opy} from {spy} (state {res['make_venv']['state']})", file=sys.stderr, flush=True)
    if not os.path.exists(opy):
        raise RuntimeError(f"patched interpreter not found: {opy} (rosettafold3-opt install --make-venv, or set {_stack.ENV_PYTHON})")
    ts_before = _tree.state_of(opy, lists)
    if os.path.realpath(ts_before.site_packages) == os.path.realpath(ts_stock.site_packages):
        raise RuntimeError(f"the patched interpreter resolves rf3 from the stock interpreter's site-packages ({ts_stock.site_packages}): "
                           "refused (two interpreters, never one switched in place)")
    _tree.expect(ts_before, "stock", "patched")
    res["opt_before"] = {"state": ts_before.state, "line": ts_before.line()}
    res["core"] = install_core(opy)
    print(f"{PREFIX} core {res['core']['line']}", file=sys.stderr, flush=True)
    if addon:
        res["kit"] = apply_kit(opy, kh, log_path=log_path)
        ts_after = _tree.state_of(opy, lists)
        _tree.expect(ts_after, "patched")
    else:                                                                    # --no-addon: install.sh is not run; the tree must read exactly as it did before
        res["kit"] = {"check": None, "apply": None, "kit_line": None, "already": None, "skipped": True}
        ts_after = _tree.state_of(opy, lists)
        _tree.expect(ts_after, ts_before.state)
        print(f"{PREFIX} add-on NOT applied (--no-addon): {opy} {ts_after.line()} — `rosettafold3-opt install` (run.sh install) applies the five rf3 files", file=sys.stderr, flush=True)
    res["opt_python"] = opy
    res["opt"] = {"state": ts_after.state, "line": ts_after.line()}
    ts_stock2 = _tree.state_of(spy, lists)
    _tree.expect(ts_stock2, "stock")
    if res["kit"].get("kit_line"):
        print(res["kit"]["kit_line"], file=sys.stderr, flush=True)
    print(f"{PREFIX} patched interpreter {opy}: {ts_after.line()}", file=sys.stderr, flush=True)
    res["status"] = "PASS"
    return res


def summary_line(res: dict) -> str:
    kit = "addon=not_applied(--no-addon)" if res["kit"].get("skipped") else f"kit_already_installed={res['kit'].get('already')}"
    return f"{PREFIX} {res['status']} stock={res['stock']['state']} opt={res['opt']['state']} {kit}"
