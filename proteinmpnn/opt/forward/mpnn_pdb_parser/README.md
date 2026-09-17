# mpnn_pdb_parser — the read-once PDB parser of the ProteinMPNN exact line

`kit/fast_parse.py` is `helper_scripts/parse_multiple_chains.py` of dauparas/ProteinMPNN @ `8907e667` with one change: each PDB file is
read once and its ATOM lines are bucketed by chain letter (the stock helper re-reads the file for every one of its 352 candidate chain
letters). The per-line logic, the entry order and the output are the stock helper's: `parsed.jsonl` is byte-identical.

```bash
python kit/fast_parse.py --input_path <pdb dir>/ --output_path parsed.jsonl     # the stock helper's own two arguments (+ --ca_only)
```

`--mode exact` runs it in place of the stock helper whenever the input is a PDB directory (a `parsed.jsonl` the caller gives is read as
given); `--mode off` runs the stock helper. The exact line's worker, `mpnn_worker2.py`, is staged into `kit/` beside it at run time from
`../mpnn_exact_worker/addon/`; the worker file itself lives only in `../mpnn_exact_worker/`.

`fast_parse.py` derives from ProteinMPNN (MIT) — `NOTICE.md`, `licenses/`.
