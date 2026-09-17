# opt_core.host_cache — host-side pipeline primitives

The work around the network that kits repeat: writing outputs off the critical path, skipping parameter initialisation a checkpoint
load overwrites, running many items in one resident process with every item accounted for. Pure standard library; the kit passes its
own objects in (writer function, engine modules, per-item callables). No primitive prints: each hands back `key=value` evidence fields
the kit appends to its own ACTIVE / summary line with `opt_core.report.kv`, and a dict for `opt_manifest.json`.

| module | primitive | the kit supplies | evidence fields (`fields()`) | manifest block | fail-loud rule |
|---|---|---|---|---|---|
| `bg_writer` | `BackgroundWriter(mode, workers)` — `submit(fn, *args)`, `write_bytes(path, data)`, `join()`; modes `off` (inline, stock behaviour) · `thread` · `fork` (lazy pool, workers never touch the device) · `spawn`; in every background mode `submit` pickles `(fn, args)` before returning = a snapshot, so a caller may reuse its buffers | the stock writer function (module-level) and the object it writes; the mode/worker count from its own switch | `bg_writer=<mode>:<workers> bg_written=<written>/<submitted>` | `census()`: submitted, written, failed, errors, complete | `join()` is mandatory (context manager joins on the error path too); a worker failure is counted, named, and re-raised by `join()` after the census; an unpicklable argument is a `TypeError` at `submit`; a writer left unjoined at exit is drained by the exit guard (the files land), ONE line `[opt_core.host_cache] BG-WRITER NOT JOINED … failed=F` is printed, and a lost write forces exit status `EXIT_UNJOINED_LOSS` (70); `written < submitted` is the kit's `incomplete` clause |
| `fast_init` | `suppressed(*namespaces, names=("trunc_normal_init_",))` context manager; `check_loaded(load_result, record=rec)` | every namespace holding a reference to the initialisers (defining module + each `from … import` site); the load call inside the block; the load's result | `fast_init=skipped:<calls>` after `check_loaded`, `fast_init=unchecked:<calls>` before it (a defect legible from the log), `fast_init=off` via `off_fields()` when unselected | `record.as_dict()`: names, sites, calls, missing_keys | a stale site list (no name found) raises; `check_loaded` raises `FastInitError` on any missing key and when given nothing to check — a parameter whose initialiser was skipped and the checkpoint did not restore is never tolerated; namespaces are restored on every exit path |
| `resident` | `run_items(items, run_one, before_item=…, after_item=…, journal=…)` → `Census`; `read_journal(path)` | what an item is, the per-item reseed/reset in `before_item`, per-item finalisation in `after_item` (its failure is the item's failure), the model kept across items, output paths | `<name>_items=<ok>/<requested> <name>_failed=<ids or none>` | `census.as_dict()`: n_items, n_items_complete, items_complete, item_failed[{id,status,reason}] | every item is `ok` · `failed` (reason `Type: message`, traceback on stderr) · `not_run` (after `stop_on_error`) · `interrupted`; `census.incomplete()` is the clause for `opt_core.report.verdict`; the journal is fsynced per item and a torn last line reads back as `interrupted`, never dropped |

## Exactness

`bg_writer` and `resident` change no arithmetic: the writer runs the kit's own function on the same values (byte-identical files), and
the item loop calls the kit's own per-item code. What a kit must prove on its equality row is its own part: that its snapshot handed to
the writer is complete, and that item *i* of a resident run is seeded and reset exactly as a fresh process at item *i* (the reseed rule is
the kit's, in `before_item`). `fast_init` changes which RNG draws happen during model construction (module docstring, "RNG contract"):
byte-identical outputs hold exactly when the engine seeds after construction or nothing reads the advanced generator — each adopting
kit's equality row is the proof, and `check_loaded` is the guard that the parameters themselves are the checkpoint's.

## Adoption pattern (kit side)

```python
from opt_core import report
from opt_core.host_cache import bg_writer, fast_init, resident

# 1. output writer — the switch, its name and default are the kit's mode-table business
writer = bg_writer.BackgroundWriter(mode=os.environ.get("<KIT>_BG_WRITE", "off"), workers=4, name="pdb")
...
writer.submit(stock_write_fn, path, snapshot)          # instead of stock_write_fn(path, obj)
...
census = writer.join()                                 # before the manifest is written; census -> manifest["kit"]["bg_writer"]
line_fields.update(writer.fields())

# 2. checkpoint load
import engine.layers.initialize as init_mod, engine.layers.linear as linear_mod   # every reference site, listed by the kit
with fast_init.suppressed(init_mod, linear_mod) as rec:
    result = model.load_state_dict(state, strict=False)   # or the engine's own load call
fast_init.check_loaded(result, record=rec)             # raises on missing keys
line_fields.update(rec.fields())

# 3. resident item loop
census = resident.run_items(range(n_designs), design_one, before_item=lambda i, _: reseed(seed + i),
                            journal=os.path.join(out_dir, "journal.jsonl"), name="design")
v = report.verdict(rc, activation_report, allow_partial, incomplete=census.incomplete())
```

The ACTIVE line stays one per arm per process and keeps the kit's grammar (`opt_core.report`: `[<tag>] <VERB> key=value ...`); these
fields are appended to it, e.g. `[<tag>] ACTIVE mode=exact levers=… bg_writer=fork:4 fast_init=skipped:212` and, at the end of the
pass, `[<tag>] DONE design_items=8/8 design_failed=none bg_written=8/8`.
