# inputs/targets/ — the three public structures the warm-up tasks condition on

| file | wwPDB entry | bytes | source URL |
|---|---|---|---|
| `5o45.cif` | 5O45 — human PD-L1 in complex with an inhibitor | 498156 | https://files.rcsb.org/download/5O45.cif |
| `1tnf.cif` | 1TNF — tumour necrosis factor-alpha | 383234 | https://files.rcsb.org/download/1TNF.cif |
| `3di3.cif` | 3DI3 — interleukin-7 / IL-7 receptor complex | 312813 | https://files.rcsb.org/download/3DI3.cif |

`../tasks_3targets.json` names these files relative to the kit directory (`opt/forward/hoist/inputs/targets/<id>.cif`, so `pxdesign infer -i` and
`run.sh design --tasks` take it as is from `pxdesign/`); `run.sh warm` stages a copy with absolute paths (`pxdesign_opt/warm.py` stage_tasks)
and designs against the first task (5O45, chain A 1-116, binder 75).
`SOURCES.json` lists entry, URL, size, title and deposition date per file. PDB archive data are free of all copyright restrictions (CC0 1.0).
