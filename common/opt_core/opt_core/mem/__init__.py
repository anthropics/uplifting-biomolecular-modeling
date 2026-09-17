"""opt_core.mem — the ``big`` memory mode: the levers that let an engine fold bigger inputs, applied by name and recorded in full.

``big`` is a release mode beside ``off`` / ``exact`` / ``fast``: it composes on the kit's ``fast`` mode (``exact`` where no fast
exists) and adds memory levers — host offload, chunking, checkpointing, allocator policy, sub-batching, memory-lean kernels,
engine settings. Its guarantee is "runs bigger, within band": every lever carries an exactness label (``bitwise`` / ``band`` /
``measured``) with its reason, and the record names the labels per lever and for the mode. Nothing in this package names an engine:
a kit's adapter (``<engine>/opt/<pkg>/big.py``) composes the mode into its table, names its hook points per lever, and applies.

    from opt_core import modes
    from opt_core import mem

    TABLE, LINE = mem.compose_big(modes.ModeTable(("off", "exact", "fast"), "fast"), levers=("pair_offload", "chunk_pair_transition"),
                                    drop=("graph_sampler",))                       # big = fast + [levers] - [drop]
    ctx = mem.Ctx(prefix="ACME", tag="acme-opt", framework="torch",
                  hooks={"pair_offload": {"module": trunk, "tensor": "z"}, "chunk_pair_transition": {"module": trunk.transition, "dim": 1}})
    record = mem.apply(LINE, ctx, line=args.big_line, switches=switches,           # the kit's parsed selector / lever switches / opt-out;
                       allow_partial=args.allow_partial)                            # strict: a refusal raises mem.Refused(record)
    report.log_once(rep, record.active_line("acme-opt"))                            # [acme-opt] ACTIVE mode=big:pair_offload,… base=fast exact=…
    record.attach(kit_fields)                                                       # opt_manifest.json kit.big
    ...
    record.unit_begin(item_id); ...; record.unit_end()                              # the census delimiters (levers mark themselves)
    verdict = record.exit_gate(rc)                                                  # fail-closed: refused / partial -> exit 3 unless allow_partial (the kit's --allow-partial)

Modules (each bound on first access — importing this package imports only the primitives):

    mode        apply / census / undo / Refused: the mode's application contract over the registry, the record and the composition
    registry    Lever / LEVERS / register / discover; Ctx; Refusal / refuse / RefusalError; the flag grammar and the selection
    record      AppliedRecord: applied / refused / checked, the mode line, the manifest block, the census, the exit gate
    compose     compose_big / BigLine / base_for on top of opt_core.modes.ModeTable
    allocator   the device allocator's policy as a lever (expandable segments, graph pools, cache release, peak counters); the allocator variables recorded
    offload     host offload of named tensors / modules with streamed transfers
    chunk       chunked / streamed forms of the pair, triangle, head and MSA-row operations
    ckpt        block / cycle checkpointing, diffusion-sample chunking, seed batching caps
    jax_mem     the jax-side levers (sub-batching, bucket policy, memory flags)
    peak        the peak-memory probe beside each pass

Inputs: ``<PREFIX>_OPT_MODE=big`` selects the mode through the kit's own autoload / mode plumbing; everything else the kit parses from
its own flags or variables and PASSES — ``line=`` (a declared line name or ``auto``), ``switches={lever: on/off}`` (a line lever off, a
registered lever on), ``ctx.settings={lever: {setting: value}}`` (a lever's declared settings), ``allow_partial=`` (the census opt-out,
its ``--allow-partial``). The core reads no environment variable for any of them; a switch naming no lever refuses by name.

Primitives (engine-free mechanisms a lever or a kit adapter composes; standard library at import, torch imported inside the call):

    primitives      MemLeverRefused, row_blocks, Ledger (per-process counters), evidence_line, env_int / env_flag, parse_levers — re-exported here
    patchset        PatchSet: attribute patches applied and restored as one set (framework-free)
    torch_rowchunk  row-chunked evaluation of row-independent pair sub-modules into ONE preallocated output, the pair-shape guard, dead-tensor release
    torch_alloc     the CUDA caching-allocator policy of a process (PYTORCH_CUDA_ALLOC_CONF): export-at-activation with named refusals, read-back, env row
    torch_hostpair  a pinned-host mirror of a pair tensor served to the device in row blocks, a pinned buffer pool, the layer_norm row-split guard
    graph_gate      the token-gated capture decision of a CUDA-graph / static-arena lever and its line fragment
    budget          byte budgets: rows_within, nbytes, device_free_bytes (allocator-aware)
    ngpu            the ``--n_gpu P`` resource axis: visible-device count, the by-name refusal when fewer than P are visible, the worker launch contract
"""
from __future__ import annotations

from .. import lazy_getattr
from .primitives import Ledger, MemLeverRefused, env_flag, env_int, evidence_line, parse_levers, row_blocks   # the primitives' names, re-exported

LAZY_EXPORTS = {                                                                  # public name -> the sub-module that defines it (bound on first access)
    **dict.fromkeys(("apply", "census", "undo", "Refused"), "mode"),
    **dict.fromkeys(("compose_big", "BigLine", "base_for", "BASE_RULE", "select_line", "DEFAULT_LINE", "AUTO"), "compose"),
    **dict.fromkeys(("AppliedRecord", "UnitRecord", "compose_exact", "PROCESS_UNIT"), "record"),
    **dict.fromkeys(("Ctx", "Lever", "LEVERS", "register", "unregister", "get", "names", "table", "discover", "selection", "Selection", "Applied",
                     "Refusal", "RefusalError", "HookMissing", "RegistryError", "off_ref", "setting_ref", "switch_value", "framework", "NAME_RE",
                     "refusal_from", "refuse", "BIG", "FAMILIES", "EXACT_LABELS", "FRAMEWORKS", "LEVER_MODULES", "OPT_OUT", "SCOPES"), "registry"),
}

__all__ = ["MemLeverRefused", "row_blocks", "Ledger", "evidence_line", "env_int", "env_flag", "parse_levers",
           "apply", "refuse", "census", "undo", "Refused", "compose_big", "BigLine", "base_for", "BASE_RULE", "AppliedRecord",
           "UnitRecord", "compose_exact", "PROCESS_UNIT", "Ctx", "Lever", "LEVERS", "register", "unregister", "get", "names", "table",
           "discover", "selection", "Selection", "Applied", "Refusal", "RefusalError", "HookMissing", "RegistryError", "off_ref", "setting_ref", "switch_value",
           "framework", "NAME_RE",
           "refusal_from", "select_line", "DEFAULT_LINE", "AUTO", "BIG", "FAMILIES", "EXACT_LABELS", "FRAMEWORKS", "LEVER_MODULES",
           "OPT_OUT", "SCOPES"]

__getattr__ = lazy_getattr(__name__, LAZY_EXPORTS)   # the names above and every sub-module by name, on first access
