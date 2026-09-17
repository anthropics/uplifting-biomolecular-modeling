"""Mode tables and env-script transcription.

Contract. Mode names are promises with fixed meanings — ``off`` (stock: no environment set, no lever applied), ``exact`` (byte-identical
to stock under the kit's deterministic recipe; absent when no such lever set exists), ``fast`` (best speed within the panel band) — plus
extra modes a kit names by their testable property. A :class:`ModeTable` validates a kit's table; :func:`mode_argument` is the one
precedence rule (command line > environment > default). An :class:`EnvScript` is a line-by-line transcription of a kit's own activation
script: each call mirrors one statement (forced export, ``${VAR:-default}``, unset, path prepend) and :meth:`EnvScript.apply` turns it
into a :class:`Resolution` — what the mode means on this box — without touching the process environment. :func:`install_env` applies a
resolution and returns what it replaced. :func:`source_env_sh` runs the real script in a clean subshell so a kit's conformance test can
compare the transcription with the script variable by variable. The core adds no lever and reorders nothing: the values are the kit's own.
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, MutableMapping, Optional, Sequence

OFF, EXACT, FAST = "off", "exact", "fast"
STANDARD_MODES = (OFF, EXACT, FAST)
_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


class ModeError(ValueError):
    """A mode name outside the kit's table (the message names the table)."""


class UnsupportedMode(ModeError):
    """A mode this engine does not have (a standard name the kit's table lacks, e.g. ``exact`` on an engine with no exact lever set) or an
    unknown one — the ONE refusal type of a mode argument anywhere in the kit (a :class:`ModeError`, hence a ``ValueError``)."""


def levers_label(levers) -> str:
    """``a+b+c`` — the lever set of a line (``none`` when empty): the ``levers=`` value of the kit's ACTIVE lines."""
    return "+".join(str(x) for x in levers) if levers else "none"


def literal_assignment(path: str, name: str):
    """The value of the top-level ``<name> = <literal>`` assignment of a Python source file, parsed WITHOUT importing it (``ast.literal_eval``:
    a kit reads an engine file's table on an interpreter that cannot import the engine). A missing assignment or a non-literal value is a
    named failure (``ValueError`` naming the file and the name), never a fallback; the caller checks the shape it needs."""
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else ([node.target] if isinstance(node, ast.AnnAssign) and node.value is not None else [])
        if any(isinstance(t, ast.Name) and t.id == name for t in targets):
            try:
                return ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError) as e:
                raise ValueError(f"{path}: top-level {name} is not a literal ({e}) — the file changed shape") from None
    raise ValueError(f"{path}: no top-level {name} assignment — the file changed shape")


@dataclass(frozen=True)
class ModeTable:
    """The kit's mode names in order (unique, ``[a-z][a-z0-9_]*``, containing ``off``), its default, and — when the kit's own usage
    text for an unknown mode differs from the core's — ``unknown_message(value) -> str`` composing the kit's bytes."""

    modes: tuple
    default: str
    unknown_message: Optional[Callable[[str], str]] = None

    def __post_init__(self):
        modes = tuple(self.modes)
        if len(modes) != len(set(modes)):
            raise ValueError(f"duplicate mode names in {modes}")
        for m in modes:
            if not isinstance(m, str) or not _NAME.match(m):
                raise ValueError(f"mode name {m!r} is not [a-z][a-z0-9_]*")
        if OFF not in modes:
            raise ValueError(f"the mode table {modes} lacks 'off'")
        if self.default not in modes:
            raise ValueError(f"default mode {self.default!r} is not in {modes}")
        object.__setattr__(self, "modes", modes)

    def check(self, name: Optional[str]) -> str:
        """The normalised mode name (stripped, lowercased); ``None``/empty means the default; unknown raises :class:`ModeError`."""
        if name is None or not str(name).strip():
            return self.default
        m = str(name).strip().lower()
        if m not in self.modes:
            raise ModeError(self.unknown_message(name) if self.unknown_message else f"unknown mode {name!r} (expected {'|'.join(self.modes)})")
        return m

    @property
    def kit_modes(self) -> tuple:
        """The modes that apply levers (every mode but ``off``)."""
        return tuple(m for m in self.modes if m != OFF)


def mode_argument(cli: Optional[str], environ_value: Optional[str], table: ModeTable) -> str:
    """The mode of a run: the command-line value when given, else the environment variable's, else the table's default."""
    if cli is not None and str(cli).strip():
        return table.check(cli)
    if environ_value is not None and str(environ_value).strip():
        return table.check(environ_value)
    return table.default


@dataclass
class Resolution:
    """What a mode means on this box before anything is applied. ``env`` holds the exports in order (forced and defaulted values
    after :meth:`EnvScript.apply`), ``unset`` the names removed, ``pythonpath`` the entries in order, ``notes`` the named events
    (a pre-set value kept, a probe branch taken), ``refuse`` the would-refuse reason of a dry run, ``extra`` the kit's own fields."""

    mode: str
    env: dict = field(default_factory=dict)
    unset: tuple = ()
    pythonpath: tuple = ()
    notes: list = field(default_factory=list)
    refuse: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"mode": self.mode, "env": dict(self.env), "unset": list(self.unset), "pythonpath": list(self.pythonpath),
                "notes": list(self.notes), "refuse": self.refuse, "extra": dict(self.extra)}


class EnvScript:
    """A transcription of an activation script, one call per statement, in the script's order. ``line`` cites the statement mirrored."""

    def __init__(self, mode: str):
        self.mode = mode
        self._ops: list = []

    def export(self, name: str, value, *, line=None) -> "EnvScript":
        """``VAR=value`` — forced: replaces a pre-set value."""
        self._ops.append(("export", name, str(value), line))
        return self

    def default(self, name: str, value, *, keep_empty: bool = False, line=None) -> "EnvScript":
        """``VAR=${VAR:-value}`` — a pre-set, non-empty value is kept (recorded in the notes). ``keep_empty=True`` mirrors ``${VAR-value}``:
        a pre-set EMPTY value is kept too (only an unset variable takes the default)."""
        self._ops.append(("default", name, (str(value), bool(keep_empty)), line))
        return self

    def unset(self, name: str, *, line=None) -> "EnvScript":
        self._ops.append(("unset", name, None, line))
        return self

    def prepend_path(self, name: str, entries: Iterable[str], *, if_set: bool = False, line=None) -> "EnvScript":
        """``NAME="<entries>:$NAME"`` — the literal bash idiom: entries first, then the pre-set value, joined by ``os.pathsep`` (an empty
        pre-set value leaves the trailing separator, as bash does). ``if_set=True`` mirrors ``NAME="<entries>${NAME:+:$NAME}"`` (no
        separator when the value is empty). Entries are never de-duplicated: the string is what the script would export."""
        self._ops.append(("prepend", name, (tuple(entries), bool(if_set)), line))
        return self

    @property
    def statements(self) -> list:
        return list(self._ops)

    def apply(self, environ: Mapping[str, str]) -> Resolution:
        """Pure: the resolution of this script over ``environ`` (the caller's environment, read only)."""
        env: dict = {}
        unset: list = []
        notes: list = []
        pythonpath: tuple = ()

        def current(name):
            if name in env:
                return env[name]
            if name in unset:
                return None
            return environ.get(name)

        for op, name, value, line in self._ops:
            where = f" ({line})" if line else ""
            if op == "export":
                env[name] = value
                if name in unset:
                    unset.remove(name)
            elif op == "default":
                value, keep_empty = value
                pre = current(name)
                if pre or (keep_empty and pre is not None):
                    env[name] = pre
                    notes.append(f"{name} pre-set to {pre!r}: kept{where}")
                else:
                    env[name] = value
            elif op == "unset":
                env.pop(name, None)
                if name not in unset:
                    unset.append(name)
            elif op == "prepend":
                entries, if_set = value
                pre = current(name) or ""
                head = os.pathsep.join(entries)
                env[name] = head + (os.pathsep + pre if (pre or not if_set) else "")
                if name in unset:
                    unset.remove(name)
        if "PYTHONPATH" in env:
            pythonpath = tuple(e for e in env["PYTHONPATH"].split(os.pathsep) if e)
        return Resolution(mode=self.mode, env=env, unset=tuple(unset), pythonpath=pythonpath, notes=notes)


def install_env(res: Resolution, environ: Optional[MutableMapping[str, str]] = None) -> dict:
    """Apply a resolution's exports and unsets to ``environ`` (the process environment by default). Returns the previous values
    (``None`` for a name that was absent) so a caller can restore or record them."""
    environ = os.environ if environ is None else environ
    before: dict = {}
    for name in res.unset:
        before[name] = environ.get(name)
        environ.pop(name, None)
    for name, value in res.env.items():
        before.setdefault(name, environ.get(name))
        environ[name] = value
    return before


def source_env_sh(path: str, preset: Optional[Mapping[str, str]] = None, *, bash: str = "bash", timeout: float = 60,
                  args: Sequence[str] = ()) -> dict:
    """Test helper: source ``path`` in a clean subshell (``env -i`` plus ``preset``, plus ``PATH``) and return the resulting
    environment as a dict (``env -0`` parsed). ``args`` are the script's positional parameters. A failing script raises."""
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}
    env.update(preset or {})
    script = 'source "$1"; shift; env -0'
    proc = subprocess.run([bash, "-c", script, "source_env_sh", os.path.abspath(path), *args], env=env, capture_output=True,
                          timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"source {path} failed rc={proc.returncode}: {proc.stderr.decode(errors='replace')[-2000:]}")
    out: dict = {}
    for item in proc.stdout.split(b"\0"):
        if not item:
            continue
        k, _, v = item.decode(errors="replace").partition("=")
        out[k] = v
    for k in ("_", "SHLVL", "PWD", "OLDPWD"):
        out.pop(k, None)
    return out
