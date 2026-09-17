# OB0-001 — declared templates dropped silently when a structure fetch fails (openfold3 0.5.0)

Doc-only entry: nothing here patches upstream and no `--upstream-fix` code exists for it. The kit's loud behaviour is the template guard
(`opt/openfold3_ob0_opt/templ_census.py`, every route): a query that DECLARED templates and reached the model with an empty featurised
template stack prints `[openfold3_ob0-opt] TEMPLATES DROPPED: query=<q> declared chains_templated=<n> real_slots=0 slots=<T> (<reason>)`,
counts `fallbacks=templates_dropped:<reason>=<k>` in the exit census and exits 5 after the outputs are written — `--allow-template-drop`
is the caller's recorded opt-out (exit 0, the census word kept).

WHAT UPSTREAM 0.5.0 DOES. `run_openfold predict --use-templates true` on a query whose chains name a template alignment file, without
`template_preprocessor_settings.structure_directory` and with `fetch_missing_structures` at its default `True`
(openfold3/core/data/pipelines/preprocessing/template.py:1683; the structure directory then defaults to a fresh temporary directory,
:1713-1723): for each alignment hit whose structure file is absent the preprocessor fetches it from RCSB (:2198-2230). The alignment's first
row is the query pseudo-hit ('query'), so the very first fetch targets a non-existent entry. A `RequestError` is caught there and the hit is
skipped by name (:2220-2230) — but on a host without network access the fetch raises a `ConnectionError`-class exception that is NOT a
`RequestError`; it propagates to `preprocess_templates`' per-query catch-all (:2036-2062), which logs the exception and ABANDONS THE WHOLE
ALIGNMENT of that query chain: 'Preprocessing templates: 0/1', no real hit is loaded, the featuriser fills the template slots with empty
(all-masked) entries and the model predicts untemplated. Exit status 0; the outputs equal those of the same query with its template keys
removed.

SYMPTOM. The run log shows the preprocessor's exception trace for the 'query' row followed by 'Preprocessing templates: 0/1' and no
'Loading template structure' line; the prediction completes; on a kit route the exact line's own census reads 'templ distinct census: …
identical dummy templates: untemplated input'. Before the guard the kit's `TEMPLATES DECLARED … chains_templated=1 files_resolved=1/1`
line was the only trace and the exit census said `fallbacks=none`.

CONSEQUENCE FOR CALLERS. Offline hosts must hand upstream the structures: `template_preprocessor_settings: {structure_directory: <dir
with the hits' mmCIF files>, structure_file_format: cif, fetch_missing_structures: false}` in the runner yaml (the kit composes a caller's
yaml under every route's configuration and keeps these keys). With that block the 'query' pseudo-hit is skipped by name, the real hits load
from the directory ('Loading template structure …'), and the guard stays silent.

WHY NO PATCH. The behaviour is upstream's error handling in its preprocessing pipeline, outside the kit's optimisation surface; the correct
caller-side configuration exists and the guard makes the failure mode impossible to miss. A fix belongs upstream (catch the fetch's
connection errors per hit like `RequestError`, or never fetch the 'query' pseudo-row).
