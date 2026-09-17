"""Class records — the ONE rule deciding whether a record serves a device, pure (no torch): read by the wrapper's
assert_device_class and by the package's file reader (flashzoi_opt.modes.kit_class_record).

A record `{"class", "sm": [major, minor], "memory_mib", "device_names": [...], "kit": <fz_exact.py sha256>, ...}` serves a device for a kit when
its `kit` is that kit's fz_exact digest AND the device is of the record's class BY CAPABILITY — compute capability == `sm`; the record's
`memory_mib` is documentation of the part the record was cut on, never a key (an 80 GB part's record serves the 40 GB / 94 GB / 141 GB parts of
the same capability: the same kernels run there; memory_note words the difference for the apply line) — or, where the caller cannot state the
capability (no CUDA in the process), by its name being one of `device_names` (the names are documentation of the class first, a key only then). The kit's PINNED classes (PINS['device_classes'] /
PINS['device_names']) are served by the same rule (pinned_serves)."""

MEMORY_TOL_MIB = 2048   # the memory difference above which memory_note names the device's memory beside the row's (torch's total_memory and nvidia-smi's memory.total of one card differ by < 1 GiB); NOT a key


def _cc_tuple(cc):
    try:
        return tuple(int(x) for x in (cc.split(".") if isinstance(cc, str) else cc))
    except (TypeError, ValueError, AttributeError):
        return None


def class_serves(entry: dict, cc=None, memory_mib=None) -> bool:
    """Whether a device of compute capability `cc` is of the class `entry` = {"sm": [major, minor], "memory_mib": …} (PINS['device_classes'] rows
    and class build records alike): sm equal. `memory_mib` is accepted for the callers' signature and NOT compared — the class is the capability
    (the row's memory is the part it was measured on; memory_note words a difference)."""
    sm = tuple(int(x) for x in ((entry or {}).get("sm") or ()))
    return cc is not None and bool(sm) and _cc_tuple(cc) == sm


def class_row(rows, cc=None, memory_mib=None):
    """The row of `rows` (PINS['device_classes'] / records) that words a device's class: among the rows of its capability (class_serves), the one
    whose `memory_mib` is within MEMORY_TOL_MIB of the device's (the part the row was measured on — h100 vs h200 are one capability, two measured
    rows), else the first row of the capability (a part of a memory size no row was cut on: served, memory_note words it); None when no row has
    the capability."""
    of_cc = [r for r in (rows or ()) if class_serves(r, cc, memory_mib)]
    near = [r for r in of_cc if memory_note(r, memory_mib) is None and r.get("memory_mib") is not None and memory_mib is not None]
    return (near or of_cc or [None])[0]


def memory_note(entry: dict, memory_mib=None):
    """None when the device's total memory is within MEMORY_TOL_MIB of the row's `memory_mib` (or either is unstated); else the one clause the
    apply line carries — the device is served all the same."""
    mm = (entry or {}).get("memory_mib")
    if mm is None or memory_mib is None or abs(int(memory_mib) - int(mm)) <= MEMORY_TOL_MIB:
        return None
    return f"this device reports {int(memory_mib)} MiB, the class row {entry.get('class') or list(entry.get('sm') or [])} was cut on a {int(mm)} MiB part (memory is not a key: served by capability)"


def record_serves(rec: dict, kit_digest, device_name: str, cc=None, memory_mib=None) -> bool:
    if not isinstance(rec, dict) or rec.get("kit") != kit_digest:
        return False
    if class_serves(rec, cc, memory_mib):
        return True
    return device_name in (rec.get("device_names") or ())


def pinned_serves(PINS: dict, device_name: str, cc=None, memory_mib=None) -> bool:
    """Whether the kit's PINNED classes serve the device: by capability (PINS['device_classes'] rows) or by name (PINS['device_names'])."""
    return any(class_serves(c, cc, memory_mib) for c in (PINS.get("device_classes") or ())) or device_name in (PINS.get("device_names") or ())
