"""opt_core.kernels.trimul._pystack — the large-frame trampoline now lives at :mod:`opt_core._pystack` (every provider face and
``opt_core.warm_imports`` use it); this module keeps the names the trimul face and its tests import."""
from opt_core._pystack import PAD_SLOTS, _TRAMPOLINE, _trampoline, padded_call, call_cost_here, import_padded, guarded_import, sweep_orphans  # noqa: F401
