"""The kit's core pin gate, run BEFORE any ``opt_core`` import: is an ``opt_core`` of at least the pinned version importable?

Every documented entry of a kit executes ``gate(__file__)`` as its first statement. The gate reads the kit's pin — ``[tool.opt_core]``
``path`` / ``version`` in the kit's ``pyproject.toml``, ``version`` being the MINIMUM core version the kit needs — locates ``opt_core`` with
``importlib.util.find_spec`` WITHOUT importing it, reads the ``__version__`` literal of the located package, and compares. A core that is
absent or older than the pin is one line on stderr and exit status 3, never a raw ``ModuleNotFoundError`` from a sub-module::

    [<tag>] NOT ACTIVE: reason=core_missing:opt_core (pinned >= v<want> at <path>; nothing importable as opt_core on sys.path)
    [<tag>] NOT ACTIVE: reason=core_mismatch: opt_core pinned >= v<want> at <path>, installed v<have> at <dir>
    [<tag>] NOT ACTIVE: reason=core_pin_unreadable: <why>

On a pass it returns the facts ``{"pinned": {"path", "version", "pyproject"}, "installed": {"package_dir", "root", "version"}, "tag"}``;
run records carry the installed version. The kit carries this file inside its package (``<engine>/opt/<package>/_core_gate.py``) because it
must run while the core may be absent; standard library only, imports nothing of the core, touches no ``sys.path``.
"""
import importlib.util
import json
import os
import re
import sys

EXIT_NOT_ACTIVE = 3
PIN_TABLE = "opt_core"                       # [tool.opt_core] in the kit's pyproject.toml
PIN_KEYS = ("path", "version")
_PIN_LINE = re.compile(r'^\s*([A-Za-z0-9_]+)\s*=\s*"([^"]*)"\s*(?:#.*)?$')


class CoreGateRefused(SystemExit):
    """``SystemExit(3)`` carrying the printed line (``.line``) and the reason word (``.reason``)."""

    def __init__(self, line, reason, code=EXIT_NOT_ACTIVE):
        super().__init__(code)
        self.line, self.reason = line, reason


def version_tuple(text):
    """``"0.5.17.4"`` -> ``(0, 5, 17, 4)``; a non-numeric component counts as -1 (older than any release)."""
    return tuple(int(p) if p.isdigit() else -1 for p in str(text or "").strip().split("."))


def read_table(pyproject_path, table):
    """The string keys of one TOML table (``tomllib`` when the interpreter has it, else a reader of that table's ``key = "value"`` lines)."""
    try:
        import tomllib                                       # noqa: PLC0415
    except ModuleNotFoundError:
        tomllib = None
    if tomllib is not None:
        with open(pyproject_path, "rb") as fh:
            node = tomllib.load(fh)
        for part in table.split("."):
            node = node.get(part, {}) if isinstance(node, dict) else {}
        return {k: v for k, v in node.items() if isinstance(v, str)} if isinstance(node, dict) else {}
    out, inside = {}, False
    with open(pyproject_path, "r", encoding="utf-8") as fh:
        for ln in fh:
            s = ln.split("#", 1)[0].strip()
            if s.startswith("["):
                inside = s.replace(" ", "") == "[" + table + "]"
                continue
            m = _PIN_LINE.match(ln) if inside else None
            if m:
                out[m.group(1)] = m.group(2)
    return out


def find_pyproject(anchor, max_up=3):
    """The nearest ``pyproject.toml`` carrying ``[tool.opt_core]`` at or above ``anchor``'s directory (None when there is none)."""
    d = os.path.abspath(anchor if os.path.isdir(anchor) else os.path.dirname(anchor))
    for _ in range(max_up + 1):
        pp = os.path.join(d, "pyproject.toml")
        if os.path.isfile(pp) and read_table(pp, "tool." + PIN_TABLE):
            return pp
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


def kit_tag(pyproject_path, fallback):
    """The kit's tag for its lines: ``[project] name``, else ``fallback``."""
    return (read_table(pyproject_path, "project").get("name") if pyproject_path else None) or fallback


def installed_core():
    """The ``opt_core`` this interpreter would import, located WITHOUT importing it: None when nothing is importable under that name, else
    ``{"package_dir", "root", "version"}`` (``version``: the ``__version__`` literal of its ``__init__.py``, None when there is none)."""
    try:
        spec = importlib.util.find_spec("opt_core")
    except (ImportError, ValueError):
        spec = None
    if spec is None or not spec.origin or not os.path.isfile(spec.origin):
        return None
    package_dir = os.path.dirname(os.path.abspath(spec.origin))
    version = None
    try:
        with open(spec.origin, "r", encoding="utf-8") as fh:
            m = re.search(r'^__version__\s*=\s*"([^"]+)"', fh.read(), re.M)
        version = m.group(1) if m else None
    except OSError:
        pass
    return {"package_dir": package_dir, "root": os.path.dirname(package_dir), "version": version}


def _refuse(tag, reason, detail, stream):
    line = "[%s] NOT ACTIVE: reason=%s%s" % (tag, reason, detail)
    (stream or sys.stderr).write(line + "\n")
    (stream or sys.stderr).flush()
    raise CoreGateRefused(line, reason.split(":", 1)[0])


def gate(anchor, tag=None, stream=None):
    """Refuse by name (one ``NOT ACTIVE`` line, ``SystemExit(3)``) unless an ``opt_core`` of at least the pinned version is importable;
    return the facts on a pass. ``anchor``: a file or directory of the kit (``__file__`` of the caller); ``tag``: the kit's line tag
    (default: its ``[project] name``)."""
    pp = find_pyproject(anchor)
    here = os.path.abspath(anchor) if os.path.isdir(anchor) else os.path.dirname(os.path.abspath(anchor))
    tag = tag or kit_tag(pp, os.path.basename(here))
    if pp is None:
        _refuse(tag, "core_pin_unreadable", ": no pyproject.toml with [tool.%s] at or above %s" % (PIN_TABLE, here), stream)
    pin = read_table(pp, "tool." + PIN_TABLE)
    missing = [k for k in PIN_KEYS if not pin.get(k)]
    if missing:
        _refuse(tag, "core_pin_unreadable", ": %s [tool.%s] lacks %s" % (pp, PIN_TABLE, ", ".join(missing)), stream)
    want_v, pin_path = pin["version"], pin["path"]
    core = installed_core()
    if core is None:
        _refuse(tag, "core_missing:opt_core", " (pinned >= v%s at %s; nothing importable as opt_core on sys.path)" % (want_v, pin_path), stream)
    if core["version"] is None or version_tuple(core["version"]) < version_tuple(want_v):
        _refuse(tag, "core_mismatch", ": opt_core pinned >= v%s at %s, installed v%s at %s" % (want_v, pin_path, core["version"] or "?", core["root"]), stream)
    return {"pinned": {"path": pin_path, "version": want_v, "pyproject": pp}, "installed": core, "tag": tag}


if __name__ == "__main__":                               # python _core_gate.py [<anchor>]  — prints the facts or the NOT ACTIVE line (exit 3)
    sys.stdout.write(json.dumps(gate(sys.argv[1] if len(sys.argv) > 1 else os.getcwd()), indent=1) + "\n")
