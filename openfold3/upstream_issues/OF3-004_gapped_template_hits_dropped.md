# OF3-004 — template hits with a gapped alignment are dropped at inference (openfold3 0.4.1; behind OF3-001)

Fixed together with OF3-001 (`--upstream-fix OF3-001`): the code path is reachable only once OF3-001 hands a query's preprocessed template
entries to the model, so the two travel under one flag; without the flag nothing here runs and stock is byte-unchanged.

WHAT UPSTREAM 0.4.1 DOES. The predict-mode template preprocessor (`core/data/pipelines/preprocessing/template.py`, the branch `run_openfold
predict --use-templates true` runs) stores, per hit, an `idx_map` — one row per alignment column, `[query residue index, template residue
index]` — in which a column that is a gap on either side carries `-1` (`core/data/io/sequence/template.py calculate_ids_hit`: "-1 for template
positions aligned to gaps in the query", and the converse). The featurizer's residue mapper (`core/data/primitives/structure/template.py
map_token_pos_to_template_residues`) is written for the TRAINING cache's map, whose rows are residue-residue pairs only
(`core/data/primitives/sequence/template.py create_residue_idx_map`: "columns where the template sequence is not a gap"): it selects the query
tokens whose residue index appears in column 0 and the template residues whose index appears in column 1 and requires the two counts to agree;
a `-1` row makes them differ, the "Skip if still misaligned" branch yields no template slice, and the hit's slot stays empty. Every hit whose
alignment to the query contains a gap — an insertion or a deletion anywhere in the aligned span — is therefore dropped without a word; only
gap-free hits are featurised.

SYMPTOM. Under `--upstream-fix OF3-001` a templated query whose hits are all gapped prints `UPSTREAM-FIX OF3-001 chain=<id> template_ids=<n>
sampled=<n>` (the hits were sampled) and then the guard's `TEMPLATES DROPPED: query=<q> … real_slots=0` (exit 5); a query with a gap-free hit
shows that hit featurised and the gapped ones missing (`TEMPLATES chain=<id> real=<r>/<k>` with r < k).

REPRODUCTION. A 2 × 306-residue query with four PDB hits per chain (135–176 template-gap and 0–115 query-gap columns each): 4 sampled, 0
featurised; with the gap rows removed from the sampled entries the same four hits featurise (aligned tokens per slot 314 / 238 / 232 / 450 of
612) and the structures parse unchanged; a 61-residue query with one gap-free hit featurises identically with and without the removal.

WHAT THE FIX CHANGES. `upstream_issues/OF3-001_templates_not_consumed.py drop_gap_rows`: each sampled entry's `idx_map` is cut to the rows
whose two indices are both residues before the entry reaches the featurizer (the mapper's documented input; residue pairs untouched, so a
gap-free hit is byte-unchanged); an entry with no residue pair left is removed from the sample. The chain's line counts the rows:
`[openfold3-opt] UPSTREAM-FIX OF3-001 chain=<id> template_ids=<n> sampled=<k> gap_rows_dropped=<g>`.

WHY NOT UPSTREAM'S OWN PATH. At 0.4.1 upstream never reaches the mapper with a predict-mode entry (OF3-001), so the defect is invisible there;
later upstream releases rewrote the inference template path. The kit pins 0.4.1 and fixes the entry it hands over rather than editing vendored code.
