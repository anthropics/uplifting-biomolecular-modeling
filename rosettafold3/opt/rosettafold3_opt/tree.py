"""Tree states: which bytes an interpreter's ``rf3`` package carries, decided by sha256 against the trees this checkout holds.

The kit installs by overwriting five files under site-packages (``install.sh``); nothing in this package decides a mode
without first proving the interpreter's tree state. The reference digests are computed live (``digests``): the ``patched`` bytes are
the add-on's own ``patched/<file>``, the pristine bytes are ``stock/src/<repository path>`` (the pinned upstream's files, unpacked).

States (``STATES``): ``stock`` = the 4 rf3 files pristine, ``graph_flags.py`` absent; ``patched`` = the 5 rf3 files == the add-on's
``patched/``; anything else is ``unknown`` and every command refuses on it. The stock route (``off``) runs only on ``stock``; the kit modes only on
``patched`` (the default of ``RF3_CUDAGRAPH`` flips to ``1`` in the patched ``graph_flags.py``, so a stock pass on a patched tree would
not be stock).
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import _core

_gates = _core.load("gates")                       # the core's file hasher (one hasher in the tree)

RF3_FILES = ("rf3/graph_flags.py", "rf3/diffusion_samplers/inference_sampler.py", "rf3/model/RF3_structure.py",
             "rf3/model/layers/af3_diffusion_transformer.py", "rf3/loss/loss.py")            # the five files install.sh overlays, in its order
STATES = ("stock", "patched")
PATCHED_DIR = "patched"                                                                      # <add-on>/patched/<installed path>


def stock_src_path(rel: str) -> str:
    """The pinned upstream's repository path of an installed ``rf3/…`` file: under ``models/rf3/src/``."""
    return os.path.join("models", "rf3", "src", rel)


def sha256_file(path: str) -> Optional[str]:
    """The core's file hasher (``opt_core.gates.sha256_file``); None for a file that is not there (a tree state counts present files)."""
    if not os.path.isfile(path):
        return None
    return _gates.sha256_file(path)


@dataclass(frozen=True)
class Digests:
    """The reference digests a tree is classified against, each computed from the file named in ``sources``."""
    patched: Dict[str, str]            # the 5 rf3 files as the add-on carries them
    stock: Dict[str, str]              # the 4 rf3 files as the pinned upstream ships them (graph_flags.py is absent upstream)
    sources: Dict[str, str]            # installed path or set name -> the file the digest came from


def digests(kit_home: str, stock_src: str) -> Digests:
    """Hash the add-on's ``patched/`` files and the pinned upstream's files under ``stock_src``. A reference file that is missing raises
    (the checkout is incomplete)."""
    def need(path: str) -> str:
        d = sha256_file(path)
        if d is None:
            raise FileNotFoundError(f"tree reference file missing: {path}")
        return d
    sources: Dict[str, str] = {}
    patched = {}
    for rel in RF3_FILES:
        p = os.path.join(kit_home, PATCHED_DIR, rel); patched[rel] = need(p); sources[f"patched:{rel}"] = p
    stock = {}
    for rel in RF3_FILES[1:]:
        p = os.path.join(stock_src, stock_src_path(rel)); stock[rel] = need(p); sources[f"stock:{rel}"] = p
    return Digests(patched, stock, sources)


@dataclass
class TreeState:
    site_packages: str                          # the directory that holds rf3/
    state: str                                  # one of STATES or "unknown"
    files: Dict[str, str] = field(default_factory=dict)      # relpath -> class: patched|stock|absent|unknown
    shas: Dict[str, Optional[str]] = field(default_factory=dict)
    unknown: List[str] = field(default_factory=list)

    @property
    def rf3_count(self) -> str:
        """``patched(5/5)`` / ``stock(5/5)``: the 5 rf3 files in the state's class (graph_flags.py absent counts for stock)."""
        want = "patched" if self.state == "patched" else "stock"
        n = sum(1 for f in RF3_FILES if self.files.get(f) == want or (want == "stock" and f == RF3_FILES[0] and self.files.get(f) == "absent"))
        return f"{self.state}({n}/{len(RF3_FILES)})"

    def line(self) -> str:
        return f"tree={self.rf3_count} site-packages={self.site_packages}"


def classify(site_packages: str, lists: Digests) -> TreeState:
    """Sha every file of the set — ``rf3/`` under ``site_packages`` — and name the state."""
    files: Dict[str, str] = {}
    shas: Dict[str, Optional[str]] = {}
    for f in RF3_FILES:
        s = sha256_file(os.path.join(site_packages, f))
        shas[f] = s
        if s is None:
            files[f] = "absent"
        elif s == lists.patched.get(f):
            files[f] = "patched"
        elif s == lists.stock.get(f):
            files[f] = "stock"
        else:
            files[f] = "unknown"
    rf3 = [files[f] for f in RF3_FILES]
    if rf3[0] == "absent" and all(c == "stock" for c in rf3[1:]):
        state = "stock"
    elif all(c == "patched" for c in rf3):
        state = "patched"
    else:
        state = "unknown"
    unknown = [f for f, c in files.items() if c == "unknown"]
    return TreeState(site_packages, state, files, shas, unknown)


_LOCATE = ("import importlib.util, os, sys; s = importlib.util.find_spec(sys.argv[1]); "
           "print(os.path.dirname(os.path.dirname(os.path.abspath(s.origin))) if s and s.origin else 'ABSENT')")   # install.sh's own locator, without importing the package


def locate_of(python: str, package: str = "rf3", timeout: float = 120) -> str:
    """The directory that holds ``<package>/`` (``rf3`` by default; ``foundry`` for the venv derivation) for interpreter ``python``,
    without importing it (``importlib.util.find_spec`` only). A git or archive install puts both under one site-packages; an editable
    checkout at the pin puts ``rf3`` under ``models/rf3/src`` and ``foundry`` under ``src``."""
    r = subprocess.run([python, "-c", _LOCATE, package], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"{python} cannot locate {package}: {(r.stderr or r.stdout).strip()[-400:]}")
    lines = [ln.strip() for ln in r.stdout.strip().splitlines() if ln.strip()]
    d = lines[-1] if lines else "ABSENT"
    if d == "ABSENT":
        raise RuntimeError(f"{python}: the {package} package is not installed (stock/PINS.json names the pin to install)")
    return d


def this_site_packages() -> str:
    """The rf3 location of the running interpreter (no import of rf3)."""
    import importlib.util
    s = importlib.util.find_spec("rf3")
    if s is None or not s.origin:
        raise RuntimeError("the rf3 package is not installed in this interpreter (stock/PINS.json names the pin to install)")
    return os.path.dirname(os.path.dirname(os.path.abspath(s.origin)))


def state_of(python: Optional[str], lists: Digests) -> TreeState:
    return classify(this_site_packages() if python is None else locate_of(python), lists)


def expect(ts: TreeState, *states: str) -> None:
    """Raise with a named reason unless ``ts.state`` is one of ``states``."""
    if ts.state not in states:
        detail = ", ".join(f"{f}={c}" for f, c in ts.files.items())
        raise RuntimeError(f"tree state is {ts.state!r} (expected {'|'.join(states)}) in {ts.site_packages}: {detail}")
