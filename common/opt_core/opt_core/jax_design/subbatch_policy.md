# `opt_core.jax_design.subbatch_policy` — the attention sub-batch decision (engine-free)

Standard library only; nothing under this package imports jax. A kit's adapter reads its own values (token count,
`jax.devices()[0].memory_stats()["bytes_limit"]`, its measured peak points) and calls the primitive; the primitive returns a record whose
`fields()` dict the kit renders into its ONE activation-evidence line with `opt_core.report.kv` (grammar `[<tag>] <VERB> key=value ...`) and whose
`as_dict()` goes into the kit's `opt_manifest.json`.

| module | mechanism | public surface | record → line fragment |
|---|---|---|---|
| `subbatch_policy` | attn.subbatch (the row-chunk of an engine's chunked attention, a configuration value fixed at trace time — the kit cites its engine's reading site; decided per traced executable) | `parse_request(text, default)` → `None` (unchunked) · `int` · `'auto'` · `'stock'`; `choose(tokens=, stock_value= (required: the engine's chunk, cited by the kit), requested=, device_bytes=, peak_estimator=, stock_rule=, when_device_unknown=)` → `SubbatchDecision{value, source, tokens, stock_value, estimated_bytes, device_bytes}`; `QuadraticPeak(a, b, c, unit, margin)` / `fit_quadratic(points)` (the adapter's measured fit); `threshold_rule(N, chunk)` (a stock size rule as a callable) | dict `{subbatch: none\|n, subbatch_source: requested\|auto:fits\|auto:exceeds\|auto:no_device\|stock[, subbatch_est_gb, subbatch_dev_gb]}` → `report.kv` |auto:fits\|auto:exceeds\|auto:no_device\|stock> [subbatch_est_gb=… subbatch_dev_gb=…]` |

Decision table of `choose` (every row is a named `source`; there is no silent branch):

| `requested` | device size known | outcome (`value`) | `source` |
|---|---|---|---|
| `None` or an int | — | as requested | `requested` |
| `'stock'` | — | `stock_rule(tokens)` | `stock` |
| `'auto'` | yes, estimate ≤ device | `None` (unchunked) | `auto:fits` |
| `'auto'` | yes, estimate > device | `stock_value` | `auto:exceeds` |
| `'auto'` | no | `stock_value` if `when_device_unknown='stock'` (default), `None` if `'unchunked'` | `auto:no_device` |

Class of the lever that uses it is the ENGINE's: where the decision changes the traced program (chunked ↔ unchunked) the accumulation is
re-associated — fast-class, judged by the engine's tier-2 form; where it reproduces the engine's stock rule the program is unchanged.
A design kit typically decides twice per model build: the forward-only executable with `requested='stock'` (predict numerics do not move)
and the differentiated executable with the kit variable's request (default `'auto'`).

Tests: `tests/test_jax_design_subbatch_policy.py` (parser words, every source, the fragment bytes, the fit, import hygiene in a clean
interpreter).
