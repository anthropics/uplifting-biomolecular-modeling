"""Lazy autoload: the meta-path finder a kit's ``<pkg>_autoload.pth`` installs at interpreter start.

Contract. A kit describes itself in an :class:`AutoloadSpec` and calls ``install(spec)`` from its ``_autoload.py`` (the module the
.pth imports). With ``spec.env`` unset or ``off`` nothing is installed. A selection the spec does not declare (a mode outside
``spec.modes``, a variant outside ``spec.variants``) is a REFUSAL, never a note: ``install`` installs a finder whose trigger prints the
kit's own NOT ACTIVE line and exits ``spec.exit_not_active`` (default, ``on_unknown="refuse_at_trigger"``: the kit's exit code is honoured
and a process that never imports the engine is untouched); ``on_unknown="exit_now"`` prints the line, flushes both streams and
``os._exit(exit_not_active)`` at once — the form for a kit whose bytes refuse at .pth time with the kit's exit code (no
atexit handlers run, buffered stdout beyond the flush is lost: the .pth runs before anything of the program); ``on_unknown="exit"``
prints the line and ``sys.exit`` — from a .pth line the interpreter reports status 1 whatever the code (a SystemExit during ``site`` is
an init failure); ``on_unknown="note"`` (print and install nothing) is an explicit opt-in a kit names, and it is the only path on which
stock runs under a set variable. Values are matched against the declared names verbatim; the mode is
case-folded only when the spec says so (``fold_mode``) and only onto a declared mode. Otherwise a :class:`Finder` sits first on
``sys.meta_path`` and waits for the first import of one of ``spec.triggers``; it lets the trigger's own module body run, removes itself,
and calls ``<spec.package>.enable(mode[, variant], strict=True, trigger=<name>)``. ``strict=True`` asks the kit to RAISE its
``ActivationError`` instead of returning an inactive report, and a kit raises it for exactly the refusals of an activation: the stock
engine's source at another commit / version than the tree pins, an ``opt_core`` below the kit's minimum, a lever of the mode that cannot
run on this box (``opt_core.gates`` outcome (b)). Uncertainty about the environment — a framework wheel at another patch level, a card no
engine was measured on, an unreadable pin, a cold cache — is words on the kit's ACTIVE line (``opt_core.gates`` outcome (a)) and never an
``ActivationError``: the finder sees ``enable()`` return and the program continues, active. When the kit raises its ``ActivationError``
(the kit has printed its NOT ACTIVE line) the process exits ``spec.exit_not_active``; an exception carrying ``cannot_run = True``
(``opt_core.gates.is_cannot_run``) becomes ``[tag] NOT ACTIVE: mode <mode> refused at <trigger>: <Type>: <message>``; any OTHER exception
from ``enable()`` (or from importing the kit package) becomes ``[tag] NOT ACTIVE: enable() raised at <trigger>: <Type>: <message>`` —
each plus the same exit, never a traceback, and stock never runs silently under ``spec.env``. The finder
disarms only when it actually serves the trigger's import (the loader path): a bare ``importlib.util.find_spec`` probe leaves it armed,
so the import that follows a probe is still hooked. Until the trigger nothing else is imported: this module imports only ``os`` and
``sys``; the kit package is imported only inside ``_fire``.

The kit's ``_autoload.py`` reads its variable FIRST and imports this module only when the variable names a selection: the .pth line runs
in every interpreter on the box, including the stock arm's processes, and those must load nothing of the core. The kit keeps its own
import-free copy of the exit code for the one refusal it prints itself — the core not importable under a set variable::

    import os, sys
    ENV, EXIT_NOT_ACTIVE = "ACME_OPT", 3
    FINDER = None
    if (os.environ.get(ENV) or "").strip().lower() not in ("", "off"):
        try:
            from opt_core.autoload import AutoloadSpec, install
        except ImportError:
            sys.stderr.write("[acme-opt] NOT ACTIVE: opt_core not importable (install the core beside this kit)\\n")
            sys.exit(EXIT_NOT_ACTIVE)
        FINDER = install(AutoloadSpec(env=ENV, package="acme_opt", tag="acme-opt", triggers=("vendor.models.acme",),
                                      modes=("exact", "fast", "off"), exit_not_active=EXIT_NOT_ACTIVE))

:func:`patch_attr_at_import` is the smaller sibling for ONE stock attribute: ``patch_attr_at_import(module, "Class.method",
make_wrapper, tag=...)`` patches now when the module is imported, else at its import (body first); armed → installed; a missing
attribute is a named refusal (:class:`PatchError` now, the kit's NOT ACTIVE line + exit at import time); the original is kept and a
re-patch wraps the original, never the wrapper; :meth:`AttrPatch.pairs` / :meth:`AttrPatch.record` are its evidence.
"""
import os
import sys

ON_UNKNOWN = ("refuse_at_trigger", "exit_now", "exit", "note")


class AutoloadSpec:
    """What the finder needs to know about one kit — every value is the kit's own.

    env: the mode variable (read once at interpreter start; stripped; case-folded onto a declared mode only when ``fold_mode``).
    package: the kit package whose ``enable(...)`` / ``ActivationError`` the finder uses (imported only when it fires).
    tag: the ``[tag]`` prefix of the NOT ACTIVE line the default ``line_of`` composes.
    triggers: fully-qualified module names; the finder fires after the first one's body has executed.
    modes: the kit's mode names including ``off`` (restated in the spec: nothing else is imported at start).
    variant_env: when set, its stripped value (or None) is passed VERBATIM (never case-folded) as the second positional argument of
        ``enable`` — a kit whose own hook folded the variant folds it in ``enable``, or declares the folded names in ``variants``.
    variants: when declared, the variant value must be one of these names verbatim — anything else is a refusal.
    accept: optional predicate ``(fullname, spec) -> bool`` — a generic trigger name is accepted only when the predicate says the
        found module is the kit's family; refused triggers are ignored, the finder stays armed.
    exit_not_active: the exit code of a refusal (the kit's EXIT_NOT_ACTIVE).
    on_unknown: ``"refuse_at_trigger"`` (default: the trigger refuses with the kit's line and exit code) · ``"exit_now"`` (refuse at
        once with the kit's exit code: flush + ``os._exit``, no atexit) · ``"exit"`` (``sys.exit`` at once; from a .pth line the
        interpreter's status is 1) · ``"note"`` (print, install nothing: the kit's explicit opt-in).
    line_of: ``(what, value) -> str`` composes the kit's own NOT ACTIVE line for an unknown selection (``what`` = "mode" | "variant");
        default: ``[tag] NOT ACTIVE: unknown <ENV>=<value!r> (expected a|b|c)``.
    fold_mode: lowercase the mode value before matching (the usual form); False matches the mode verbatim.
    pre_body: optional ``(trigger_fullname, selection) -> None`` the finder calls exactly once, immediately BEFORE the trigger module's
        own body executes (``enable`` runs after the body) — for what a stock module reads AT IMPORT (an environment variable read at
        module level). ``selection`` is the very dict (``{"mode", "variant"}``) whose values ``enable`` then receives, so the two cannot
        disagree. It lives in the kit's ``_autoload.py`` (os/sys only; it never imports the engine); when it raises, the finder prints
        ``[tag] NOT ACTIVE: pre-body hook failed before <trigger>: <exc!r>`` and exits ``exit_not_active`` (fail-closed, as a refused
        selection). It is not called on the refusal path, nor when the variable is unset or ``off``, nor after ``disarm``.
    """

    __slots__ = ("env", "package", "tag", "triggers", "modes", "variant_env", "variants", "accept", "exit_not_active", "on_unknown",
                 "line_of", "fold_mode", "pre_body")

    def __init__(self, env, package, tag, triggers, modes, variant_env=None, accept=None, exit_not_active=3, *, variants=None,
                 on_unknown="refuse_at_trigger", line_of=None, fold_mode=True, pre_body=None):
        if not env or not package or not tag:
            raise ValueError("AutoloadSpec needs env, package and tag")
        if not triggers:
            raise ValueError("AutoloadSpec needs at least one trigger module")
        if "off" not in modes:
            raise ValueError("AutoloadSpec.modes must contain 'off'")
        if on_unknown not in ON_UNKNOWN:
            raise ValueError(f"AutoloadSpec.on_unknown must be one of {ON_UNKNOWN}, not {on_unknown!r}")
        if variants is not None and not variant_env:
            raise ValueError("AutoloadSpec.variants needs variant_env")
        self.env = env
        self.package = package
        self.tag = tag
        self.triggers = tuple(triggers)
        self.modes = tuple(modes)
        self.variant_env = variant_env
        self.variants = None if variants is None else tuple(variants)
        self.accept = accept
        self.exit_not_active = int(exit_not_active)
        self.on_unknown = on_unknown
        self.line_of = line_of
        self.fold_mode = bool(fold_mode)
        if pre_body is not None and not callable(pre_body):
            raise ValueError("AutoloadSpec.pre_body must be callable: (trigger_fullname, selection) -> None")
        self.pre_body = pre_body

    def unknown_line(self, what, value):
        """The kit's NOT ACTIVE line for an unknown selection (the kit's ``line_of`` when given)."""
        if self.line_of is not None:
            return self.line_of(what, value)
        if what == "mode":
            return not_active_line(self, f"unknown {self.env}={value!r} (expected {'|'.join(self.modes)})")
        return not_active_line(self, f"unknown {self.variant_env}={value!r} (expected {'|'.join(self.variants or ())})")


class _Loader:
    """The trigger's loader with ``exec_module`` followed by the activation; everything else delegates. A fresh wrapper per spec
    (a loader shared between specs — a zip importer — is never patched in place)."""

    def __init__(self, loader, finder):
        self._loader = loader
        self._finder = finder

    def __getattr__(self, name):
        return getattr(self._loader, name)

    def create_module(self, spec):
        create = getattr(self._loader, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module):
        name = module.__name__
        self._finder._before_body(name)                # the kit's pre-body hook (once; refusals and disarmed finders skip it)
        self._loader.exec_module(module)               # the trigger package's own body first, then the activation
        self._finder._fire(name)
        for other in [f for f in list(sys.meta_path) if isinstance(f, AttrPatch) and f is not self._finder]:
            if other.state == "armed" and other.target == name:
                other._fire(name)                      # every other patch armed on this module installs at the same import


def _real_spec(hook, fullname, path, target):
    """The module spec the interpreter's OWN meta-path finders give for ``fullname`` (None when none does or it has no loader). Every
    hook of this module (``hook`` itself, other :class:`Finder` / :class:`AttrPatch` instances) is skipped: the first armed hook serves the
    import and its loader fires the other hooks armed on the same module (:meth:`_Loader.exec_module`), so any number of patches on one
    not-yet-imported module install at its import."""
    for finder in sys.meta_path:
        if finder is hook or isinstance(finder, (Finder, AttrPatch)):
            continue
        try:
            spec = finder.find_spec(fullname, path, target)
        except Exception:  # noqa: BLE001
            spec = None
        if spec is not None:
            return spec if spec.loader is not None else None
    return None


class Finder:
    """Duck-typed meta-path finder (no importlib.abc import at interpreter start). Fires once — when it serves a trigger's import —
    and removes itself; a ``find_spec`` probe alone leaves it armed."""

    def __init__(self, spec, selection, refuse=None):
        self.spec = spec
        self.selection = selection                     # {"mode": ..., "variant": ...}
        self.refuse = refuse                           # a NOT ACTIVE line: the trigger refuses with it instead of activating
        self.armed = True
        self.fired = None
        self.pre_fired = None                          # the trigger whose import ran the pre-body hook

    @property
    def mode(self):
        return self.selection["mode"]

    def find_spec(self, fullname, path=None, target=None):
        if not self.armed or fullname not in self.spec.triggers:
            return None
        spec = _real_spec(self, fullname, path, target)
        if spec is None:
            return None
        if self.spec.accept is not None and not self.spec.accept(fullname, spec):
            return None
        spec.loader = _Loader(spec.loader, self)       # the finder stays armed until this loader runs
        return spec

    def remove(self):
        try:
            sys.meta_path.remove(self)
        except ValueError:
            pass

    def _before_body(self, trigger):
        """Run the spec's ``pre_body(trigger, selection)`` once, before the trigger's body — only on the activation path."""
        if self.spec.pre_body is None or self.refuse is not None or not self.armed or self.fired is not None or self.pre_fired is not None:
            return
        self.pre_fired = trigger
        try:
            self.spec.pre_body(trigger, self.selection)
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(not_active_line(self.spec, f"pre-body hook failed before {trigger}: {e!r}") + "\n")
            sys.stderr.flush()
            sys.exit(self.spec.exit_not_active)

    def _fire(self, trigger):
        if self.fired is not None or not self.armed:   # a reload through a stale wrapper, or disarmed meanwhile: the body ran, nothing more
            return
        self.armed = False
        self.fired = trigger
        self.remove()
        if self.refuse is not None:
            sys.stderr.write(self.refuse + "\n")
            sys.stderr.flush()
            sys.exit(self.spec.exit_not_active)
        import importlib
        try:
            pkg = importlib.import_module(self.spec.package)
        except Exception as e:                         # noqa: BLE001 — a kit package that cannot be imported is a refusal by name, never a traceback
            self._refuse_naming(f"{self.spec.package} could not be imported at {trigger}: {type(e).__name__}: {e}")
        args = [self.selection["mode"]]
        if self.spec.variant_env:
            args.append(self.selection["variant"])
        refused = getattr(pkg, "ActivationError", None)
        try:
            pkg.enable(*args, strict=True, trigger=trigger)   # returns = active (its words printed on the kit's ACTIVE line); raises = a refusal
        except Exception as e:                         # noqa: BLE001
            if refused is not None and isinstance(e, refused):
                # the kit has printed `[tag] NOT ACTIVE: <reason>`; the process stops here rather than running stock silently
                sys.exit(self.spec.exit_not_active)
            if getattr(e, "cannot_run", False):
                # a lever of the mode cannot run on this box (opt_core.gates outcome (b)): the MODE refuses by name, nothing continues on stock
                self._refuse_naming(f"mode {self.selection['mode']} refused at {trigger}: {type(e).__name__}: {e}")
            # anything else enable() raises (an ImportError / AttributeError on a core or library that lacks a name, a RuntimeError …)
            # is the same refusal WITH THE EXCEPTION NAMED — one line, the kit's exit code, no traceback, stock never runs silently
            self._refuse_naming(f"enable() raised at {trigger}: {type(e).__name__}: {e}")

    def _refuse_naming(self, reason):
        sys.stderr.write(not_active_line(self.spec, reason) + "\n")
        sys.stderr.flush()
        sys.exit(self.spec.exit_not_active)


class PatchError(RuntimeError):
    """The attribute to patch is not there (the stock module changed shape) or the kit's wrapper factory raised: a named refusal."""


_PATCHES = {}                                          # (target module, attr) -> AttrPatch: one patch per site per process


class AttrPatch:
    """One stock attribute (``attr``: a module-level name or a dotted ``Class.method``) of ``target`` replaced by
    ``make_wrapper(original)`` — now when the module is imported, else at its import (a meta-path hook that lets the module's body run,
    then patches). States: ``armed`` (waiting for the import) → ``installed``; ``disarmed`` (the kit withdrew it before the import).
    Fail-closed: a missing attribute or a raising factory is a :class:`PatchError` when patching now, and the kit's
    ``[tag] NOT ACTIVE: …`` line plus ``exit_not_active`` when it happens inside the import (stock never runs silently under a planned
    patch). The original is kept (:attr:`original`); patching the same site again wraps the ORIGINAL with the new factory, never the wrapper.
    Any number of patches may be armed on one not-yet-imported module: the first hook serves the import and fires the others."""

    def __init__(self, target, attr, make_wrapper, *, tag, name=None, exit_not_active=3):
        self.target, self.attr, self.make_wrapper = target, attr, make_wrapper
        self.tag, self.name, self.exit_not_active = tag, name or attr, exit_not_active
        self.state = "armed"
        self.error = None
        self.original = None
        self.wrapper = None
        self._hooked = False

    # -- the meta-path hook (duck-typed; reuses _Loader: body first, then _fire)
    def find_spec(self, fullname, path=None, target=None):
        if self.state != "armed" or fullname != self.target:
            return None
        spec = _real_spec(self, fullname, path, target)
        if spec is None:
            return None
        spec.loader = _Loader(spec.loader, self)
        return spec

    def _before_body(self, name):
        return None

    def _fire(self, name):
        self._unhook()
        if self.state != "armed":
            return
        try:
            self._apply(sys.modules[name])
        except PatchError as e:
            sys.stderr.write(str(e) + "\n")
            sys.stderr.flush()
            sys.exit(self.exit_not_active)

    def _unhook(self):
        if self._hooked:
            try:
                sys.meta_path.remove(self)
            except ValueError:
                pass
            self._hooked = False

    # -- the patch
    def _resolve(self, module):
        owner, obj = None, module
        for part in self.attr.split("."):
            owner = obj
            obj = getattr(obj, part, _MISSING)
            if obj is _MISSING:
                raise PatchError(f"[{self.tag}] NOT ACTIVE: {self.name} needs {self.target}.{self.attr}, not found")
        return owner, self.attr.split(".")[-1], obj

    def _apply(self, module):
        owner, leaf, current = self._resolve(module)
        original = self.original if self.original is not None else getattr(current, "_opt_core_original", current)
        try:
            wrapper = self.make_wrapper(original)
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"
            raise PatchError(f"[{self.tag}] NOT ACTIVE: {self.name}: the wrapper of {self.target}.{self.attr} could not be built: {e!r}") from e
        try:
            wrapper._opt_core_original = original     # a later patch of the same site finds the stock attribute through the wrapper
        except (AttributeError, TypeError):
            pass
        setattr(owner, leaf, wrapper)
        self.original, self.wrapper, self.state, self.error = original, wrapper, "installed", None

    def install(self):
        """Patch now when ``target`` is imported, else arm the hook (first on ``sys.meta_path``). Idempotent; returns the state."""
        if self.state == "installed":
            return self.state
        self.state = "armed"
        m = sys.modules.get(self.target)
        if m is not None:
            self._unhook()
            self._apply(m)
        elif not self._hooked:
            sys.meta_path.insert(0, self)
            self._hooked = True
        return self.state

    def repatch(self, make_wrapper):
        """Switch the wrapper factory: re-wraps the ORIGINAL now when installed, else takes effect at the import."""
        self.make_wrapper = make_wrapper
        if self.state == "installed":
            self._apply(sys.modules[self.target])
        return self.state

    def disarm(self):
        """Withdraw an armed patch before the import (True when one was armed); an installed patch stays (:meth:`restore` undoes it)."""
        if self.state != "armed":
            return False
        self._unhook()
        self.state = "disarmed"
        return True

    def restore(self):
        """Put the original attribute back (tests; a kit never un-patches a live model)."""
        if self.state == "installed":
            owner, leaf, _ = self._resolve(sys.modules[self.target])
            setattr(owner, leaf, self.original)
            self.state = "disarmed"

    def record(self):
        """Plain data for the kit's report: ``{name, target, attr, state, error}``."""
        return {"name": self.name, "target": self.target, "attr": self.attr, "state": self.state, "error": self.error}

    def pairs(self):
        """Evidence pairs for the lever's LEVER line: ``patch=<state> site=<target>:<attr>``."""
        return [("patch", self.state), ("site", f"{self.target}:{self.attr}")]


_MISSING = object()


def patch_attr_at_import(target_module, attr, make_wrapper, *, tag, name=None, exit_not_active=3):
    """Replace ``<target_module>.<attr>`` (``attr`` may be ``Class.method``) by ``make_wrapper(original)`` — now when the module is in
    ``sys.modules``, else when it is first imported (its body runs first). One :class:`AttrPatch` per site per process: a second call
    for the same site returns the same object, re-wrapping the original when the factory differs. See :class:`AttrPatch` for the
    states and the fail-closed rule."""
    key = (target_module, attr)
    p = _PATCHES.get(key)
    if p is None:
        p = _PATCHES[key] = AttrPatch(target_module, attr, make_wrapper, tag=tag, name=name, exit_not_active=exit_not_active)
        p.install()
    elif p.make_wrapper is not make_wrapper or p.state == "disarmed":
        if p.state == "disarmed":
            p.state = "armed"
        p.make_wrapper = make_wrapper
        p.install() if p.state != "installed" else p.repatch(make_wrapper)
    return p


def not_active_line(spec, reason):
    return f"[{spec.tag}] NOT ACTIVE: {reason}"


def installed(spec):
    """The finder installed for ``spec`` (by package), or None."""
    for f in sys.meta_path:
        if isinstance(f, Finder) and f.spec.package == spec.package:
            return f
    return None


def selection(spec, environ=None):
    """Read the kit's selection from ``environ``: ``(mode, variant, refusal_line)``. ``mode`` is None when the variable is unset or
    ``off``; ``refusal_line`` is the kit's NOT ACTIVE line when the mode or the variant is not a declared name (matched verbatim; the
    mode case-folded onto a declared mode only when ``spec.fold_mode``)."""
    environ = os.environ if environ is None else environ
    raw = (environ.get(spec.env) or "").strip()
    if not raw or raw == "off" or (spec.fold_mode and raw.lower() == "off"):
        return None, None, None
    mode = raw
    if mode not in spec.modes and spec.fold_mode and raw.lower() in spec.modes:
        mode = raw.lower()
    if mode not in spec.modes:
        return None, None, spec.unknown_line("mode", raw)
    variant = None
    if spec.variant_env:
        variant = (environ.get(spec.variant_env) or "").strip() or None
        if variant is not None and spec.variants is not None and variant not in spec.variants:
            return mode, None, spec.unknown_line("variant", variant)
    return mode, variant, None


def install(spec, environ=None):
    """Install the finder for ``spec`` (idempotent per package). Returns the finder, or None when nothing is to be done (variable unset
    or ``off``; ``on_unknown="note"`` after printing). An unknown selection refuses: ``"refuse_at_trigger"`` (default) installs a finder
    whose trigger prints the kit's line and exits ``spec.exit_not_active``; ``"exit_now"`` prints the line, flushes and calls ``os._exit`` with
    the kit's code here; ``"exit"`` prints the line and calls ``sys.exit`` here."""
    mode, variant, refusal = selection(spec, environ)
    if refusal is not None:
        if spec.on_unknown == "refuse_at_trigger":
            f = installed(spec)
            if f is not None:
                return f
            f = Finder(spec, {"mode": mode, "variant": variant}, refuse=refusal)
            sys.meta_path.insert(0, f)
            return f
        sys.stderr.write(refusal + "\n")
        sys.stderr.flush()
        if spec.on_unknown == "exit_now":
            try:
                sys.stdout.flush()
            except Exception:  # noqa: BLE001
                pass
            os._exit(spec.exit_not_active)
        if spec.on_unknown == "exit":
            sys.exit(spec.exit_not_active)
        return None                                    # "note": the kit opted into stock under the variable, and said so
    if mode is None:
        return None
    f = installed(spec)
    if f is not None:
        return f
    f = Finder(spec, {"mode": mode, "variant": variant})
    sys.meta_path.insert(0, f)
    return f


def disarm(spec):
    """Remove the finder for ``spec`` before importing the stock command line (``pred --help``, the stock route). True when one was armed."""
    f = installed(spec)
    if f is None:
        return False
    f.armed = False
    f.remove()
    return True
