"""Length-keyed caches that hold ONE length at a time (no torch): the vortex-kernels route's spectrum caches are keyed (layer, L, device); an entry
written for a new L — or hold(L) before the first write — evicts every registered store's entries of the previous length, so scoring many distinct
lengths keeps one length's spectra resident (one refill per change of length; a change of batch keeps them)."""

_REGISTRY: list = []


class OneLength(dict):
    """A dict whose keys are tuples carrying the length at ``l_index``; all registered stores share one resident length."""

    resident = None            # the length every registered store currently holds (class-wide)
    evictions = 0              # changes of length that released entries (class-wide count)

    def __init__(self, name: str, l_index: int = 1):
        super().__init__()
        self.name, self.l_index = name, l_index
        _REGISTRY.append(self)

    def __setitem__(self, key, value):
        hold(int(key[self.l_index]))
        super().__setitem__(key, value)


def hold(L: int) -> bool:
    """Make L the resident length: True iff entries of another length were released."""
    L = int(L)
    if OneLength.resident == L:
        return False
    released = any(len(s) for s in _REGISTRY)
    for s in _REGISTRY:
        dict.clear(s)
    if released:
        OneLength.evictions += 1
    OneLength.resident = L
    return released


def clear_all() -> None:
    for s in _REGISTRY:
        dict.clear(s)
    OneLength.resident = None


def resident_length():
    return OneLength.resident
