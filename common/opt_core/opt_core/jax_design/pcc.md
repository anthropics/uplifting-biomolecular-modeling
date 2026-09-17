# opt_core.jax_design.pcc — persistent compilation cache + XLA autotune pin (mechanism `graph.xla_jit_and_cache`, numerics class exact)

A JAX process compiles its jitted step on first use and re-runs XLA autotuning at every fresh compile, so two fresh processes hold
slightly different executables and one seed gives two trajectories. `pcc` puts two settings in force before the first jax
computation: the persistent compilation cache (`JAX_COMPILATION_CACHE_DIR` + store-everything thresholds) and the autotune pin
(`--xla_gpu_dump_autotune_results_to` in the ONE populating process, `--xla_gpu_load_autotune_results_from` in every other one,
APPENDED to `XLA_FLAGS`). Effect: every process after the populating one skips the compile and runs the populating process's
executable (bit-exact across processes and hosts of one GPU type).

Warm once, then run the fleet: the kit's `warm` verb is the populating process (`autotune="dump"`); every design process asks for
`autotune="auto"`, which loads the results. When they are absent the tier decides (`exact=`, a keyword of every call; default `True`
because this pin IS the cross-process bitwise mechanism of an exact JAX line): under `exact=True` a refusal by name — a process never
dumps implicitly (a fleet of cold processes each dumping its own results would hold executables that differ box to box), the kit's
census counts dumpers per key and gates on exactly one; under `exact=False` (a tolerance-class line) the process proceeds COLD — no
autotune flag, the record's `autotune` is `cold` and its `words` carry `autotune=cold` for the activation line, exit 0. `pcc.env(dir)` /
`pcc.plan(dir)` with `autotune="auto"` therefore RAISE on an unpopulated cache under exact by design, so a kit evaluates them lazily —
inside the `exact` row's activation, at the moment the row is applied — never at mode-table construction or module import (else
`--mode off`, `check` and `--help` would die on a cold box). The environment's refusals (`PccError` with `cannot_run = True`,
`opt_core.gates.is_cannot_run`) are the MODE's refusal by name in the kit; a usage error (an unknown mode word) is a plain `PccError`.

| function | contract |
|---|---|
| `key(jax_version=None, jaxlib_version=None, plugin_version=None, gpu_name=None, plugin_dist="jax-cuda12-plugin", exact=True)` | `jax<v>-jaxlib<v>-<plugin label><v>-<gpu slug>`, e.g. `jax0.4.30-jaxlib0.4.30-cuda12plugin0.4.30-nvidia-h100-80gb-hbm3`; `exact=True` raises `PccError` naming the missing part (jax, jaxlib, plugin, GPU name) — never a shared `unknown` bucket: a wrong key is a silent cache collision; `exact=False` = `key_facts(...)["key"]` |
| `key_facts(...)` | the tolerance-class run path: `{"key", "parts", "unknown", "word"}` — a part no metadata / probe gives takes a live module's `__version__`, else the ISOLATED part `unknown<token>` (`opt_core.jit_cache.isolation_token`: this process alone, a cold compile nobody reuses); `word` = `cache_key=unknown(<parts>)` for the activation line |
| `cache_dir(root, key, *parts)` | `<root>/<key>/<parts…>` (default `xla`); a kit with one cache per input shape passes the shape as a part; a pre-set directory is kept or replaced by `opt_core.jit_cache.keep_or_key` |
| `plan(cache_dir, autotune="auto", environ=None, exact=True)` | the record both routes share: `cache_dir`, `autotune` as resolved (`load`∣`dump`∣`off`∣`cold`), `autotune_file`, `exact`, `words`, `exports`; pure |
| `env(cache_dir, autotune="auto", environ=None, exact=True)` | `plan(...)["exports"]`: the variables a mode table exports — the route of every command-line process |
| `enable(cache_dir, autotune="auto", environ=None, exact=True)` | the in-process route for a resident driver that did not start under the mode table's exports: writes the same variables and mirrors them into a live `jax.config`; a recorded no-op (`via=already`) when the same settings are in force; refuses by name on a conflicting setting (proceeding would run on another cache directory / autotune file than the record names) and when a live jax's backends are initialised (`XLA_FLAGS` is read once at initialisation); a backend state this jax cannot report (`unknown`) applies the settings and adds the word `backend_state=unknown` |
| `already_enabled(environ=None)` / `in_force(environ=None)` | the never-re-apply rule: what is already in force, as a reason / as fields |
| `xla_flags_append(existing, *flags)` | the only writer of `XLA_FLAGS`: appends, never drops or rewrites a flag |
| `identity_key(cache_dir)` | sha256 of the autotune file's bytes + sha256 of the sorted names of the cache entries: equal keys and equal seeds ⇒ equal trajectories |
| `evidence_fields(record)` | `jax_cache=… autotune=load∣dump∣off∣cold via=… autotune_sha256=… cache_entries=…` from the route's record for the kit's one activation line (`opt_core.report.kv`; the record's `words` appended with `opt_core.report.with_words`); `None` prints `jax_cache=off` |

Kit side (thin adapter, inside the kit): `configs/<gpu>.env` exports `MODEL_OPT_STACK_KEY=$(python -c 'from opt_core.jax_design import pcc; print(pcc.key())')`
and the cache root; the kit's `warm` verb runs one design under `pcc.env(dir, autotune="dump")`; the kit's mode table lists
`pcc.env(dir)`'s variables in its `exact` row; a resident driver calls `pcc.enable(dir)` once before importing jax's backends; every arm prints
`report.prefix(tag) + "APPLIED " + report.kv(**pcc.evidence_fields(record))` in the kit's own words.
