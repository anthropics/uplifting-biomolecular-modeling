"""``stock/PINS.json`` of the kit tree: the stock's package pins (evo2, vtx and their archives), the two stacks' library pins (with and without
Transformer Engine), the checkpoints, the listed GPUs. The tree is ``EVO2_OPT_HOME`` when set (run.sh exports it), else the tree this
package is installed editable from (``opt/..``)."""
import json
import os
from importlib import metadata as _md
from typing import Optional

ENV_HOME = "EVO2_OPT_HOME"
PINS_RELPATH = os.path.join("stock", "PINS.json")
ROUTE_RELPATH = os.path.join("route", "evo2_route.py")


def package_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def tree_root() -> str:
    v = os.environ.get(ENV_HOME)
    if v:
        return os.path.abspath(v)
    return os.path.abspath(os.path.join(package_dir(), "..", ".."))


def pins_path() -> str:
    return os.path.join(tree_root(), PINS_RELPATH)


def load(path: Optional[str] = None) -> dict:
    p = path or pins_path()
    if not os.path.isfile(p):
        raise FileNotFoundError(f"stock/PINS.json is missing: {p}")
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def dist_version(name: str) -> Optional[str]:
    """The installed distribution's version by metadata (nothing imported); None when absent."""
    for n in (name, name.replace("_", "-")):
        try:
            return _md.version(n)
        except _md.PackageNotFoundError:
            continue
    return None


def te_version() -> Optional[str]:
    for n in ("transformer_engine", "transformer-engine", "transformer_engine_cu12"):
        v = dist_version(n)
        if v:
            return v
    return None


def stack_of(doc: dict, te_present: bool) -> str:
    """The stack id whose Transformer Engine pin matches this process: ``present`` -> the stack pinning a version, ``absent`` -> the TE-free one."""
    for sid, st in (doc.get("stacks") or {}).items():
        want = (st.get("pins") or {}).get("transformer_engine")
        if te_present and want not in (None, "absent") or (not te_present and want == "absent"):
            return sid
    return "unpinned"
