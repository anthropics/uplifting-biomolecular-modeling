"""The ``BORZOI_OPT`` route: ``BORZOI_OPT=<mode> borzoi_sad.py <arguments>`` activates the mode without a code change.

``borzoi_opt_autoload.pth`` (the guarded import line ``opt/_build_backend.py`` generates) imports this module at interpreter start
(site processing; ``sys.argv`` and ``sys.orig_argv`` are already set — site runs after the interpreter's config is applied), and the
import runs :func:`install` once, at the bottom of this module; a later import finds the module cached and decides nothing again.
With ``BORZOI_OPT`` unset, empty or ``off``, ``install`` does nothing and imports nothing else. With the kit mode set, the running
script decides:

* the pinned STOCK ``borzoi_sad.py`` (its sha256 == ``stock/PINS.json``'s): the mode is resolved and gated (modes.resolve), the SWAP and
  ACTIVE lines are printed, and the process re-execs the kit's entry ``v17/borzoi_sad.py`` with the same interpreter flags and arguments
  and the mode's environment (the composition; ``BORZOI_OPT`` removed, so the lines print once) — the kit is an entry-script swap, so this IS the
  activation; nothing of the stock ran;
* the KIT's own entry (invoked directly under the switch): the mode is resolved and gated, the ACTIVE line is printed (route=env) and the
  composition it names (``KIT_FWD=1``) is installed in this process's environment before the entry's ``main()`` reads it — the swapped
  child runs without the switch (its composition came with the exec), so the hook does nothing there;
* any other program — upstream's other scripts (``borzoi_sed.py``, ``borzoi_satg_*``, ``borzoi_test_*`` …), a modified copy of
  ``borzoi_sad.py``, the user's own code, ``-m``/``-c``: the LIBRARY route. Nothing happens at start-up; a finder waits for the stock library
  and, when ``baskerville.dna`` / ``baskerville.seqnn`` are imported, installs the kit's levers on them for this process — the LUT one-hot on
  ``dna.dna_1hot`` and the kit's forward call on ``SeqNN.__call__`` (call 1 eager, one trace, the graph from call 2, the copy-free return;
  ``v17/kitlib/forward.py install_class``) — prints one ``ACTIVE mode=<m> route=hook program=<p> levers=…`` line then and one ``EXIT …
  route=hook …`` line with the forward's counters at exit. The program's own code runs unmodified; its outputs are the stock's bytes; the
  ``borzoi_sad.py``-specific levers (the pipelined host post) are the swap's alone. ``borzoi-opt`` itself (``python -m borzoi_opt``)
  strips the variable from every subprocess it starts, so the hook never fires twice for one run.

Every decision is printed on stderr with the ``[borzoi-opt]`` prefix. The env route runs production settings (``--det 1`` is
``borzoi-opt sad``'s). It ends at the exec: the kit's process owns the exit and its own ``KIT_STAMP`` line on stdout is the record of what
applied — a partial activation (a lever that fell back, by that stamp) sets no exit here; ``borzoi-opt sad`` is the gated form (exit 3 on a
partial run unless ``--allow-partial`` is recorded).
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import os
import sys

from . import ENV                                                # the switch's name only: kit / modes / report are imported below, once a kit mode is actually set

EXIT_NOT_ACTIVE = 3
TRIGGER = "baskerville"
LIBRARY_TARGETS = {"baskerville.dna": "onehot_lut", "baskerville.seqnn": "graph_forward,copy_free_forward"}   # module imported -> the levers installed on it
LIBRARY_LEVERS = ("graph_forward", "copy_free_forward", "onehot_lut")
_STATE = {"armed": False, "decision": None, "installed": {}, "announced": False}


def _main_script(argv=None) -> str:
    argv = sys.argv if argv is None else argv
    a0 = argv[0] if argv else ""
    if not a0 or a0 in ("-m", "-c", "-") or not os.path.isfile(a0):
        return ""
    return os.path.realpath(a0)


def _exec_argv(kit_entry: str, argv=None, orig=None) -> list:
    """The command line to re-exec: sys.orig_argv with the script path replaced by the kit entry (interpreter flags kept), else
    [sys.executable, kit_entry] + the script arguments."""
    argv = sys.argv if argv is None else argv
    orig = getattr(sys, "orig_argv", None) if orig is None else orig
    if orig and argv and argv[0] in orig:
        i = orig.index(argv[0])
        return [sys.executable] + list(orig[1:i]) + [kit_entry] + list(orig[i + 1:])
    return [sys.executable, kit_entry] + list(argv[1:])


def decide(environ=None, argv=None) -> dict:
    """What the hook does for this process (pure: no exec, no exit, no print): ``{"action": none|swap|active|refuse, "mode", "reason",
    "report", "script"}``."""
    environ = os.environ if environ is None else environ
    mode = (environ.get(ENV) or "").strip().lower()
    if not mode or mode == "off":
        return {"action": "none", "mode": mode or None, "reason": "switch unset/off" if mode != "off" else "mode off: stock runs as is", "script": None}
    from . import kit, modes                                       # a kit mode is set: the decision needs the mode table and the kit's pins
    if mode not in modes.MODES:
        return {"action": "refuse", "mode": mode, "reason": f"unknown mode {mode!r} in {modes.ENV} (expected {'|'.join(modes.MODES)})", "script": None}
    script = _main_script(argv)
    base = os.path.basename(script) if script else ""
    argv_ = sys.argv if argv is None else argv
    program = script or (argv_[0] if argv_ else "")
    if base != kit.ENTRY:
        return {"action": "engage", "mode": mode, "reason": f"{program or 'this program'} is not the documented {kit.ENTRY}: the library route (levers on {', '.join(LIBRARY_TARGETS)} when imported)", "script": script, "program": program}
    if kit.is_kit_entry(script):
        rep = modes.resolve(mode, environ=environ, route="env", read_gpu=False)
        return {"action": "active", "mode": mode, "reason": rep.get("reason"), "report": rep, "script": script}
    stock = kit.check_stock(kit.ENTRY, None, environ)
    if not stock["ok"] or os.path.realpath(stock["entry"]) != script:
        sha = kit.sha256_file(script) if os.path.isfile(script) else None
        pin = (kit.stock_script_pin(kit.ENTRY) or {}).get("sha256")
        if sha and pin and sha == pin:
            pass                                                   # the stock bytes, found at another path than BORZOI_DIR/PATH: still the pinned stock
        else:                                                      # a modified borzoi_sad.py: the user's program — it runs as written, with the library levers (no swap: the swap would drop the edits)
            return {"action": "engage", "mode": mode, "reason": f"{script} is not the pinned stock {kit.ENTRY} (sha256 {str(sha)[:12]}… vs pin {str(pin)[:12]}…): it runs as written with the library levers; the {kit.ENTRY}-specific levers apply to the pinned script only", "script": script, "program": script}
    rep = modes.resolve(mode, environ=environ, route="env", read_gpu=False)
    if not rep["ok"]:
        return {"action": "refuse", "mode": mode, "reason": rep["reason"], "report": rep, "script": script}
    return {"action": "swap", "mode": mode, "reason": None, "report": rep, "script": script, "kit_entry": rep["entry"]}


class _EngageOnImport(importlib.abc.MetaPathFinder):
    """The library route's finder: when ``baskerville.dna`` / ``baskerville.seqnn`` are imported, the real loader runs and the kit's lever is
    installed on the fresh module (``kitlib.onehot.install`` / ``kitlib.forward.install_class``); the first installation resolves and gates the
    mode and prints the ACTIVE line (a mode that cannot run prints NOT ACTIVE and exits 3 there, before the program computed anything)."""

    def __init__(self, mode: str, program: str):
        self.mode, self.program, self._busy = mode, program, set()

    def find_spec(self, name, path=None, target=None):
        if name not in LIBRARY_TARGETS or name in self._busy or name in _STATE["installed"]:
            return None
        self._busy.add(name)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            self._busy.discard(name)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _PatchingLoader(spec.loader, name, self)
        return spec

    def installed(self, name, module):
        from . import kit, modes, report
        if not _STATE["announced"]:
            rep = modes.resolve(self.mode, route="hook", read_gpu=False)
            _STATE["report"] = rep
            if not rep["ok"]:
                report.log(report.activation_line(rep))
                _exit(EXIT_NOT_ACTIVE)
            report.log(report.hook_active_line(rep, self.program, LIBRARY_LEVERS))
            _STATE["announced"] = True
            import atexit
            atexit.register(_hook_exit, self.mode, self.program)
        kitlib = _kitlib(kit.frozen_dir())
        if name == "baskerville.dna":
            _STATE["installed"][name] = importlib.import_module("kitlib.onehot").install()
        elif name == "baskerville.seqnn":
            _STATE["installed"][name] = importlib.import_module("kitlib.forward").install_class(module.SeqNN)


class _PatchingLoader(importlib.abc.Loader):
    """Runs the real loader, then hands the executed module to the finder's ``installed``."""

    def __init__(self, real, name, finder):
        self.real, self.name, self.finder = real, name, finder

    def create_module(self, spec):
        return self.real.create_module(spec)

    def exec_module(self, module):
        self.real.exec_module(module)
        self.finder.installed(self.name, module)


def engage(mode: str, program: str, rep=None) -> list:
    """The library route from Python (modes.enable): announce with ``rep`` if given, install the levers on whatever of the stock library is
    already imported, arm the finder for the rest. Returns the LIBRARY_TARGETS installed so far."""
    finder = next((f for f in sys.meta_path if isinstance(f, _EngageOnImport)), None)
    if finder is None:
        finder = _EngageOnImport(mode, program)
        sys.meta_path.insert(0, finder)
        _STATE["armed"] = True
    if rep is not None and not _STATE["announced"]:
        from . import report
        _STATE["report"] = rep
        rep["route"] = "hook"
        report.log(report.hook_active_line(rep, program, LIBRARY_LEVERS))
        _STATE["announced"] = True
        import atexit
        atexit.register(_hook_exit, mode, program)
    for name in LIBRARY_TARGETS:
        if name in sys.modules and name not in _STATE["installed"]:
            finder.installed(name, sys.modules[name])
    return sorted(_STATE["installed"])


def _kitlib(frozen_dir: str):
    """``kitlib`` (the frozen dir's package) importable by name in this process without putting the frozen dir on sys.path."""
    if "kitlib" in sys.modules:
        return sys.modules["kitlib"]
    init = os.path.join(frozen_dir, "kitlib", "__init__.py")
    spec = importlib.util.spec_from_file_location("kitlib", init, submodule_search_locations=[os.path.dirname(init)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["kitlib"] = mod
    spec.loader.exec_module(mod)
    return mod


def _hook_exit(mode: str, program: str) -> None:
    from . import report
    inst = _STATE["installed"]
    report.log(report.hook_exit_line(mode, program, inst.get("baskerville.seqnn"), inst.get("baskerville.dna")))


def _exit(code: int) -> None:
    """Exit from site's .pth processing with the code as is: a SystemExit raised there is printed as a traceback by the interpreter and
    becomes exit 1; os._exit after flushing keeps the named code (3 = not active)."""
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def install(environ=None, argv=None, do_exec=True) -> dict:
    d = decide(environ, argv)
    _STATE["decision"] = d
    if d["action"] == "none":
        return d
    from . import modes, report                                    # a kit mode is set (refuse / engage / active / swap)
    if d["action"] == "refuse":
        report.log(f"{report.PREFIX} NOT ACTIVE: {d['reason']} (mode={d['mode']})")
        if do_exec:
            _exit(EXIT_NOT_ACTIVE)
        return d
    if d["action"] == "engage":                                     # the library route: nothing now; the levers land when the stock library is imported
        if not _STATE["armed"]:
            sys.meta_path.insert(0, _EngageOnImport(d["mode"], d.get("program") or ""))
            _STATE["armed"] = True
        return d
    if d["action"] == "active":                                     # this process IS the kit entry: the composition the ACTIVE line prints is installed here,
        rep = d["report"]                                           # in-process, before the entry's main() — the kit reads KIT_FWD at forward install (v17/kitlib/forward.py install), after the model load
        report.log(report.activation_line(rep))
        if not rep["ok"]:
            if do_exec:
                _exit(EXIT_NOT_ACTIVE)
            return d
        target = os.environ if environ is None else environ
        target.update(rep["env"])
        d["env"] = {k: target[k] for k in sorted(rep["env"])}
        return d
    # swap
    rep = d["report"]
    env = dict(os.environ if environ is None else environ)
    env.pop(modes.ENV, None)                                       # the kit entry runs without the switch: one ACTIVE line per run (printed below, before the exec)
    env.update(rep["env"])                                        # the composition (KIT_FWD=1) — nothing else; the kit prints its KIT_STAMP line itself
    cmd = _exec_argv(d["kit_entry"], argv)
    d["exec"] = cmd
    d["env"] = {k: env[k] for k in sorted(rep["env"])}
    report.log(report.swap_line(d["mode"], d["script"], d["kit_entry"]))
    report.log(report.activation_line(rep))
    if do_exec:
        sys.stdout.flush()
        sys.stderr.flush()
        os.execve(cmd[0], cmd, env)
    return d


# The hook itself: the .pth's import of this module IS the start-up decision (one install() per process). With BORZOI_OPT unset or off
# nothing is decided and nothing else runs; a refusal (unknown mode) ends the process here (exit 3); the pinned stock script is swapped by
# exec; any other program gets the library levers when it imports the stock library.
STARTUP = install()
