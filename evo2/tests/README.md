# evo2 — tests

The kit's tests are the package's CPU suite under `opt/evo2_opt/tests/` (no torch, no GPU, no upstream packages needed):

    cd opt && python -m unittest discover -s evo2_opt/tests -t .

They cover the autoload hook, the activation gate, the route driver's argument handling, the kit's call-surface gate,
`stock/PINS.json` reading, and the host-side shape and reduction rules the kernels rely on (`opt/evo2_opt/tests/README.md`
lists them). The levers' bit-identity with stock is a GPU property and is not asserted there: on the pinned stacks
`bash run.sh score --mode off` and `--mode exact` over the same FASTA file write byte-identical `scores.jsonl`.
