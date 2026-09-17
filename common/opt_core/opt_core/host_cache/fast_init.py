"""Fast checkpoint load: the random parameter initialisers a model's constructors run are no-ops while a checkpoint load that
overwrites those parameters executes.

Contract. :func:`suppressed` is a context manager over the kit's own namespaces: for every ``module`` given (a module object, a class,
or any object with attributes) and every name in ``names`` present on it, the attribute is replaced by a counting no-op for the duration
of the block and restored on exit — on the error path too. The no-op returns its first argument (the tensor, for the ``init_(tensor, …)``
convention) and draws from no RNG. The kit lists EVERY namespace that holds a reference: the defining module and each module that imported
the function by name (``from … import trunc_normal_init_`` binds a separate reference). Nothing is imported here — torch, scipy and the
engine stay the kit's.

The block is only sound when the load restores every parameter the skipped initialisers would have set, so the kit closes it with
:func:`check_loaded`: given the load's result (torch's ``load_state_dict`` return value or any object with ``missing_keys`` /
``unexpected_keys``, or the key lists themselves) it raises :class:`FastInitError` naming the missing keys — fail-closed, never a warning.
Unexpected keys are the checkpoint's business and allowed unless ``allow_unexpected=False``.

RNG contract (why outputs stay byte-identical, to be re-established per engine by its equality row): a skipped initialiser does not
consume the RNG draws the running one does (numpy / scipy global state for ``trunc_normal_init_``; torch's generator for ``torch.nn.init.*``).
Outputs are unchanged exactly when the engine seeds its generators AFTER model construction, or when nothing downstream reads the
generator the initialiser advanced. The default ``names`` is the one initialiser two kits of this tree skip with test records
(``trunc_normal_init_``: CPU scipy truncnorm, the dominant construction cost); a kit that also skips torch's own inits says so in its list.

Evidence. ``record.fields()`` → ``{"fast_init": "skipped:<calls>"}`` after :func:`check_loaded` (``unchecked:<calls>`` before it;
``"fast_init": "off"`` from :func:`off_fields` when the lever is not selected) for the kit's ACTIVE line via :func:`opt_core.report.kv`; ``record.as_dict()`` is the manifest block.
"""
from __future__ import annotations

import contextlib
from typing import Iterable, Iterator, Optional, Sequence

DEFAULT_NAMES = ("trunc_normal_init_",)


class FastInitError(RuntimeError):
    """The checkpoint load left parameters unrestored while their initialisers were suppressed (or the record was checked twice)."""


class Record:
    """What a :func:`suppressed` block did: which (namespace, name) sites were patched and how often each name was called."""

    def __init__(self, names: Sequence[str]):
        self.names = tuple(names)
        self.sites = []                    # "<namespace>.<name>" per patched reference
        self.calls = {n: 0 for n in self.names}
        self.restored = False
        self.checked = False
        self.missing_keys = []
        self.unexpected_keys = []

    @property
    def skipped(self) -> int:
        return sum(self.calls.values())

    def fields(self) -> dict:
        """``fast_init=skipped:<calls>`` once :func:`check_loaded` passed on this record; ``fast_init=unchecked:<calls>`` before — a line
        that says ``unchecked`` in a run is a kit defect legible from the log."""
        return {"fast_init": f"{'skipped' if self.checked else 'unchecked'}:{self.skipped}"}

    def as_dict(self) -> dict:
        return {"names": list(self.names), "sites": list(self.sites), "calls": dict(self.calls), "skipped": self.skipped,
                "restored": self.restored, "checked": self.checked, "missing_keys": list(self.missing_keys),
                "n_unexpected_keys": len(self.unexpected_keys)}


def off_fields() -> dict:
    """The evidence field of the unselected lever (the line names the lever either way)."""
    return {"fast_init": "off"}


def _namespace_label(ns) -> str:
    return getattr(ns, "__name__", None) or type(ns).__name__


@contextlib.contextmanager
def suppressed(*modules, names: Iterable[str] = DEFAULT_NAMES, record: Optional[Record] = None) -> Iterator[Record]:
    """Patch ``names`` on every namespace in ``modules`` with counting no-ops for the duration of the block (module contract)."""
    names = tuple(names)
    if not names:
        raise ValueError("fast_init.suppressed: no initialiser names given")
    rec = record if record is not None else Record(names)
    saved = []                                                          # (namespace, name, original)

    def make_noop(name):
        def _noop(*args, **kwargs):
            rec.calls[name] = rec.calls.get(name, 0) + 1
            return args[0] if args else None
        _noop.__name__ = name
        _noop.__qualname__ = f"fast_init_noop.{name}"
        _noop.fast_init_noop = True
        return _noop

    try:
        for ns in modules:
            if ns is None:
                continue
            for name in names:
                if hasattr(ns, name):
                    original = getattr(ns, name)
                    if getattr(original, "fast_init_noop", False):      # the same namespace listed twice: patch once
                        continue
                    saved.append((ns, name, original))
                    setattr(ns, name, make_noop(name))
                    rec.sites.append(f"{_namespace_label(ns)}.{name}")
        if not saved:
            raise FastInitError(f"fast_init.suppressed: none of {list(names)} found on the {len(modules)} namespace(s) given — "
                                f"the kit's site list is stale")
        yield rec
    finally:
        for ns, name, original in reversed(saved):
            setattr(ns, name, original)
        rec.restored = True


def check_loaded(result=None, *, record: Optional[Record] = None, missing_keys: Optional[Iterable[str]] = None,
                 unexpected_keys: Optional[Iterable[str]] = None, allow_unexpected: bool = True,
                 ignore_missing: Iterable[str] = ()) -> Record:
    """Refuse a load that left parameters unrestored. ``result`` is torch's ``load_state_dict`` return (or anything with
    ``missing_keys`` / ``unexpected_keys``); the key lists may be passed directly instead. ``ignore_missing`` names keys the kit
    proves are not parameters an initialiser touches (e.g. registered buffers rebuilt at load) — listed by the kit, never guessed here."""
    rec = record if record is not None else Record(DEFAULT_NAMES)
    if rec.checked:
        raise FastInitError("fast_init.check_loaded: record already checked")
    if missing_keys is None and not hasattr(result, "missing_keys"):
        raise FastInitError("fast_init.check_loaded: nothing to check — pass the load's result (an object with missing_keys) or "
                            "missing_keys=[...]; an unchecked fast-init load is refused, not assumed complete")
    missing = list(missing_keys) if missing_keys is not None else list(getattr(result, "missing_keys") or [])
    unexpected = list(unexpected_keys) if unexpected_keys is not None else list(getattr(result, "unexpected_keys", []) or [])
    ignored = set(ignore_missing)
    missing = [k for k in missing if k not in ignored]
    rec.missing_keys, rec.unexpected_keys, rec.checked = missing, unexpected, True
    if missing:
        head = ", ".join(missing[:8]) + (f", … (+{len(missing) - 8})" if len(missing) > 8 else "")
        raise FastInitError(f"fast_init: checkpoint load left {len(missing)} parameter(s) unrestored while their initialisers were "
                            f"suppressed: {head}")
    if unexpected and not allow_unexpected:
        raise FastInitError(f"fast_init: checkpoint carries {len(unexpected)} unexpected key(s): {', '.join(unexpected[:8])}")
    return rec
