"""RNG discipline around captured and batched sampling steps: the same draws as the stock loop, engine-free.

Contract. One state API: every rule takes GENERATOR OBJECTS (``torch.Generator``); ``None`` in a generator list means the default CUDA
generator of the current (or given) device, fetched lazily by :func:`default_generator` — ``torch.cuda.get_rng_state(dev)`` is that
object's ``get_state()``, so a kit that saves the device state and a kit that saves a generator's state follow the same rule here.

(1) Warm-up does not move the stream. :func:`generator_checkpoint` snapshots ``get_state()`` of every generator it is given and restores
    them on exit: the warm-up a CUDA-graph capture requires consumes random draws the stock loop would have made.
(2) The capture contract is declared, not implied — ``RNG_NONE`` (no draw inside the captured region), ``RNG_OUTSIDE`` (every draw stays
    outside the graph in stock order), ``RNG_INSIDE`` (draws inside; every generator the region draws from is registered with the graph
    BEFORE capture so each replay advances its Philox seed/offset exactly as the eager call does). :func:`register_generators` is that
    registration, explicit for every generator including the default one; on a torch whose ``CUDAGraph`` lacks ``register_generator_state``
    the DEFAULT generator is recorded as auto-registered (torch registers it at capture begin) and an explicit non-default generator is a
    named refusal (:class:`RngContractUnsupported`), never an unregistered capture. :func:`capture_contract_gate` refuses a capture
    whose region holds modules in training mode under any contract but ``RNG_INSIDE`` (dropout drawn inside an unregistered graph does
    not reproduce the eager stream) and returns an :class:`opt_core.gates.Gate`. :func:`offsets_moved` reports which generators a callable
    advanced — the check of an ``RNG_NONE`` / ``RNG_OUTSIDE`` claim where it is made.
(3) Per-item stream positioning for batched or padded designs. Philox offsets advance by a per-call amount that depends on the op and the
    operand shape, so it is MEASURED, never assumed: :func:`philox_increment` runs one stock draw on a fresh probe generator and reads the
    offset it consumed. :func:`fork_at_offsets` builds one generator per item at ``(seed, offset_i)`` (items drawing inside one captured
    step each own a generator object — registration needs distinct objects); :func:`reposition` sets an item's generator to
    ``base + n_calls × increment`` after a padded replay drew more than the item's stock run would have. :class:`StreamCursors` is the
    same contract over ONE generator and K saved states the kit supplies (items drawn eagerly one at a time from a shared generator, each
    from the state its own stock run would have started at): ``with cursors.item(i):`` swaps item ``i``'s state in and saves it back on
    exit, so item ``i`` consumes its own stream whatever the interleaving.
:class:`RngLedger` counts checkpoints, restores, registrations (explicit and default-auto), forks, repositions, cursor swaps and refusals for the kit's ONE activation
line. torch is imported inside the functions that construct or fetch generators; everything else is duck-typed on the generator object.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, asdict
from typing import Any, Callable, Iterable, List, Optional, Sequence

RNG_NONE, RNG_OUTSIDE, RNG_INSIDE = "rng-none", "rng-outside", "rng-inside"
CONTRACTS = (RNG_NONE, RNG_OUTSIDE, RNG_INSIDE)


class RngContractUnsupported(RuntimeError):
    """The running torch cannot honour the declared contract (no generator registration, no Philox offset API) — capture refused by name."""


@dataclass
class RngLedger:
    checkpoints: int = 0
    restores: int = 0
    registered: int = 0
    registered_default_auto: int = 0
    forks: int = 0
    repositions: int = 0
    cursor_swaps: int = 0
    increments_measured: int = 0
    refusals: int = 0

    def evidence_fields(self) -> dict:
        return asdict(self)


_LEDGER = RngLedger()          # the process ledger used when a call names none


def ledger() -> RngLedger:
    """The process-wide ledger (what a kit prints when it did not keep its own)."""
    return _LEDGER


# ---------------------------------------------------------------------------------------------------------------- generators
def default_generator(device: Any = None):
    """The default CUDA generator object of ``device`` (a ``torch.device``, an index, or None = current device). Imports torch."""
    import torch
    if device is None:
        idx = torch.cuda.current_device()
    elif isinstance(device, int):
        idx = device
    else:
        dev = torch.device(device)
        if dev.type != "cuda":
            raise RngContractUnsupported(f"default_generator: device {dev} is not a CUDA device (the CPU default generator is torch.default_generator)")
        idx = dev.index if dev.index is not None else torch.cuda.current_device()
    return torch.cuda.default_generators[idx]


def _resolve(generators: Iterable, device: Any = None) -> list:
    out = []
    for g in generators:
        out.append(default_generator(device) if g is None else g)
    return out


@contextlib.contextmanager
def generator_checkpoint(generators: Iterable, device: Any = None, ledger: Optional[RngLedger] = None):
    """``with generator_checkpoint([g1, g2, None]): warm_up()`` — every listed generator (``None`` = the device default) is restored to
    its entry state on exit, whatever the body drew. Yields the resolved generator list."""
    led = ledger or _LEDGER
    gens = _resolve(generators, device)
    states = [g.get_state() for g in gens]
    led.checkpoints += 1
    try:
        yield gens
    finally:
        for g, st in zip(gens, states):
            g.set_state(st)
        led.restores += 1


def register_generators(graph: Any, generators: Iterable, device: Any = None, ledger: Optional[RngLedger] = None) -> list:
    """Register every generator the captured region draws from with ``graph`` (``graph.register_generator_state(g)``) BEFORE capture —
    explicit for the default generator too. On a graph object without the method: a default generator (``None`` or the device default
    object) is recorded as ``registered_default_auto`` (torch registers the default CUDA generator itself at capture begin); an explicit
    non-default generator → :class:`RngContractUnsupported` (its draws inside the graph would not advance as the eager call does)."""
    led = ledger or _LEDGER
    raw = list(generators)
    gens = _resolve(raw, device)
    reg = getattr(graph, "register_generator_state", None)
    if reg is None:
        explicit = [g for given, g in zip(raw, gens) if given is not None and not _is_default_generator(g)]
        if explicit:
            led.refusals += 1
            raise RngContractUnsupported(f"this torch's CUDAGraph has no register_generator_state: {len(explicit)} explicit generator(s) drawn "
                                         "inside a captured region cannot advance as the eager call does — rng-inside capture refused")
        led.registered_default_auto += len(gens)
        return gens
    for g in gens:
        reg(g)
        led.registered += 1
    return gens


def _is_default_generator(g) -> bool:
    """True when ``g`` is one of torch's default CUDA generator objects (torch consulted only if already imported)."""
    import sys
    torch = sys.modules.get("torch")
    if torch is None:
        return False
    try:
        return any(g is d for d in torch.cuda.default_generators)
    except Exception:  # noqa: BLE001 — no CUDA: no default CUDA generators
        return False


def train_mode_modules(modules: Iterable, limit: Optional[int] = None) -> List[str]:
    """Names of modules in training mode within ``modules`` (each walked through ``named_modules()`` when it has one)."""
    out: List[str] = []
    for m in modules:
        walker = getattr(m, "named_modules", None)
        items = walker() if callable(walker) else [(getattr(m, "__class__", type(m)).__name__, m)]
        for name, sub in items:
            if getattr(sub, "training", False):
                out.append(name or sub.__class__.__name__)
                if limit is not None and len(out) >= limit:
                    return out
    return out


def capture_contract_gate(contract: str, modules: Iterable = (), name: str = "capture_contract", ledger: Optional[RngLedger] = None):
    """The gate a capture passes before warm-up: the contract is one of :data:`CONTRACTS`; modules in training mode inside the region are
    refused unless the contract is ``RNG_INSIDE``. Returns :class:`opt_core.gates.Gate` (details: contract, train-mode module names)."""
    from opt_core.gates import Gate
    led = ledger or _LEDGER
    if contract not in CONTRACTS:
        led.refusals += 1
        return Gate(name=name, ok=False, reason=f"capture contract {contract!r} is not one of {', '.join(CONTRACTS)}")
    training = train_mode_modules(modules, limit=64)
    details = {"contract": contract, "train_mode_modules": training[:8], "n_train_mode_modules": len(training)}
    if training and contract != RNG_INSIDE:
        led.refusals += 1
        shown = ", ".join(training[:5]) + ("…" if len(training) > 5 else "")
        return Gate(name=name, ok=False, details=details,
                    reason=f"capture region holds {len(training)} module(s) in training mode ({shown}) under contract {contract}: "
                           f"draws inside an unregistered graph do not reproduce the eager stream — declare {RNG_INSIDE} and register the generators, or eval()")
    return Gate(name=name, ok=True, details=details)


def offsets_moved(generators: Iterable, fn: Callable[[], Any], device: Any = None):
    """Run ``fn()`` and report which listed generators it advanced: ``(result, [(index, before, after), ...])`` over the generators whose
    Philox offset (``get_offset()``) or, lacking that, whose state bytes changed. The check of an rng-none / rng-outside claim."""
    gens = _resolve(generators, device)
    before = [_position(g) for g in gens]
    result = fn()
    after = [_position(g) for g in gens]
    moved = [(i, b, a) for i, (b, a) in enumerate(zip(before, after)) if not _same_position(b, a)]
    return result, moved


def _position(g):
    get_offset = getattr(g, "get_offset", None)
    if callable(get_offset):
        try:
            return ("offset", int(get_offset()))
        except Exception:  # noqa: BLE001 — CPU generators raise on get_offset
            pass
    return ("state", _state_bytes(g.get_state()))


def _state_bytes(s) -> bytes:
    if hasattr(s, "numpy"):                       # a torch ByteTensor state
        t = s.cpu() if hasattr(s, "cpu") else s
        return t.numpy().tobytes()
    if isinstance(s, (bytes, bytearray)):
        return bytes(s)
    return repr(s).encode("utf-8")


def _same_position(a, b) -> bool:
    return a == b


# ---------------------------------------------------------------------------------------------------------------- Philox positioning
def positioned_generator(device: Any, seed: int, offset: int = 0):
    """A generator on ``device`` at ``(seed, offset)``: ``manual_seed(seed)`` then ``set_offset(offset)``. Imports torch. A device whose
    generator has no Philox offset (CPU) with ``offset != 0`` → :class:`RngContractUnsupported`."""
    import torch
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    if offset:
        if not callable(getattr(g, "set_offset", None)):
            raise RngContractUnsupported(f"positioned_generator: generator on {device} has no Philox offset (set_offset) — cannot position at {offset}")
        g.set_offset(int(offset))
    return g


def philox_increment(draw: Callable[[Any], Any], device: Any, seed: int = 0, ledger: Optional[RngLedger] = None) -> int:
    """The Philox offset one UNIT of the stock loop consumes: ``draw(g)`` is called once with a fresh generator positioned at ``(seed, 0)``
    and must perform the stock draw SEQUENCE of one unit (one step, or one item's step: the same ops, operand shapes and dtypes, in stock
    order, with ``generator=g``); the returned offset is what each unit advances a stream by. Offsets depend on the op AND the operand
    shape — measure every distinct unit, assume nothing."""
    led = ledger or _LEDGER
    g = positioned_generator(device, seed, 0)
    if not callable(getattr(g, "get_offset", None)):
        led.refusals += 1
        raise RngContractUnsupported(f"philox_increment: generator on {device} has no get_offset — offsets cannot be measured on this device")
    draw(g)
    led.increments_measured += 1
    return int(g.get_offset())


def fork_at_offsets(seed: int, offsets: Sequence[int], device: Any, ledger: Optional[RngLedger] = None) -> list:
    """One generator per item, item ``i`` at ``(seed, offsets[i])`` — the position item ``i``'s first draw has in the stock stream."""
    led = ledger or _LEDGER
    gens = [positioned_generator(device, seed, int(off)) for off in offsets]
    led.forks += len(gens)
    return gens


def reposition(generator: Any, base_offset: int, n_calls: int, increment: int, ledger: Optional[RngLedger] = None) -> int:
    """Set ``generator`` to ``base_offset + n_calls × increment`` (where the item's stock run of ``n_calls`` draws would have left it) and
    return that offset — used after a padded/batched replay advanced the item's generator by more calls than its stock run makes."""
    led = ledger or _LEDGER
    if not callable(getattr(generator, "set_offset", None)):
        led.refusals += 1
        raise RngContractUnsupported("reposition: generator has no set_offset (not a CUDA Philox generator)")
    target = int(base_offset) + int(n_calls) * int(increment)
    generator.set_offset(target)
    led.repositions += 1
    return target


class StreamCursors:
    """K logical streams over ONE generator: item ``i`` starts from ``initial_states[i]`` — the state item ``i``'s own stock run would have
    started from (the kit derives it: a per-item seed, or the shared stream advanced to the item's position) — and ``with cursors.item(i):
    draw(...)`` swaps item ``i``'s saved state in, lets the body draw, and saves the advanced state back; the generator's own state is restored
    after each swap so code between items sees the stream it would have seen. ``states()`` returns the K saved states (e.g. to carry into
    the next batch). There is no default: K copies of one state would hand K items the same stream."""

    def __init__(self, generator: Any, initial_states: Sequence[Any], device: Any = None, ledger: Optional[RngLedger] = None):
        if not len(initial_states):
            raise ValueError("StreamCursors needs one initial state per item")
        self._gen = default_generator(device) if generator is None else generator
        self._led = ledger or _LEDGER
        self._states = [_clone_state(st) for st in initial_states]

    def __len__(self) -> int:
        return len(self._states)

    @contextlib.contextmanager
    def item(self, i: int):
        outer = self._gen.get_state()
        self._gen.set_state(self._states[i])
        self._led.cursor_swaps += 1
        try:
            yield self._gen
        finally:
            self._states[i] = _clone_state(self._gen.get_state())
            self._gen.set_state(outer)

    def states(self) -> list:
        return [_clone_state(st) for st in self._states]


def _clone_state(s):
    clone = getattr(s, "clone", None)
    return clone() if callable(clone) else s
