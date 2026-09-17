# OF3-003 — declared templates dropped silently when a structure fetch fails (openfold3 0.4.1)

Doc-only entry: nothing here patches upstream and no `--upstream-fix` code exists for it. The package's loud behaviour is the template guard
(`opt/openfold3_opt/templ_census.py`, every route): a query that DECLARED templates and reached the model with an empty featurised
template stack prints `[openfold3-opt] TEMPLATES DROPPED: query=<q> declared chains_templated=<n> real_slots=0 slots=<T> (<reason>)`,
counts `fallbacks=templates_dropped:<reason>=<k>` in the exit census and exits 5 after the outputs are written — `--allow-template-drop`
is the caller's recorded opt-out (exit 0, the census word kept); the success case is loud too — one
`[openfold3-opt] TEMPLATES FEATURISED query=<q> real_slots=<r>/<T>` line per templated query that kept real slots. The guard judges only
calls that run with templates ENABLED (`--use-templates true`, upstream's default): under `--use-templates false` a query's template paths
are no workload — one `[openfold3-opt] TEMPLATES DISABLED (--use-templates false): queries_with_template_paths=<n>` info line, nothing
judged, exit rule unaffected.

AT THE PINNED 0.4.1 THE GUARD FIRES ON EVERY `--use-templates true` CALL OF A TEMPLATED QUERY WITHOUT `--upstream-fix OF3-001`: upstream 0.4.1 parses templates and never
consumes them at inference (OF3-001), so a templated query reaches the model with an empty stack on every route and the guard names it
(`dummy_only`). A caller who wants 0.4.1's untemplated behaviour on a templated query passes `--allow-template-drop`; a caller who wants
the templates consumed passes `--use-templates true --upstream-fix OF3-001`. What follows is the second way templates can go missing, under
OF3-001, and why it is silent in upstream.

WHAT UPSTREAM 0.4.1 DOES. `run_openfold predict --use-templates true` on a query whose chains name a template alignment file, without
`template_preprocessor_settings.structure_directory` and with `fetch_missing_structures` at its default `True`
(openfold3/core/data/pipelines/preprocessing/template.py:1574; the structure directory then defaults to `<output>/template_structures`,
:1611-1613): for each alignment hit whose structure file is absent the preprocessor fetches it from RCSB (:1921-1931). The alignment's first
row is the query pseudo-hit ('query'), so the very first fetch targets a non-existent entry. A `RequestError` is caught there and the hit is
skipped by name (:1931-1936) — but on a host without network access the fetch raises a `ConnectionError`-class exception that is NOT a
`RequestError`; it propagates to `preprocess_templates`' per-query catch-all (:1787-1800), which prints the exception and ABANDONS THE WHOLE
ALIGNMENT of that query chain: 'Preprocessing templates: 0/1', no real hit is loaded, the featuriser fills the template slots with empty
(all-masked) entries and the model predicts untemplated. Exit status 0; the outputs equal those of the same query with its template keys
removed.

SYMPTOM. The run log shows the preprocessor's exception text for the 'query' row followed by 'Preprocessing templates: 0/1' and no
'Loading template structure' line; the prediction completes; the featurizer's census under OF3-001 reads `TEMPLATES chain=<id> real=0/<N>`.
Before the guard the exit census said nothing about it.

CONSEQUENCE FOR CALLERS. Offline hosts must hand upstream the structures: `template_preprocessor_settings: {structure_directory: <dir
with the hits' structure files>, structure_file_format: cif|npz, fetch_missing_structures: false}` in the runner yaml (a caller's
`--runner-yaml` is composed under every route's configuration and keeps these keys). With that block the 'query' pseudo-hit is skipped by
name, the real hits load from the directory, and the guard stays silent.

WHY NO PATCH. The behaviour is upstream's error handling in its preprocessing pipeline, outside this package's optimisation surface; the
correct caller-side configuration exists and the guard makes the failure mode impossible to miss. A fix belongs upstream (catch the fetch's
connection errors per hit like `RequestError`, or never fetch the 'query' pseudo-row); later upstream releases still behave this way.
