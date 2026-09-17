# pair_fused cells — where `triatt_block` and `pair_transition` get their kernel settings

The two fused pair levers run opt_core's `attn/pair_fused` kernels (`fpf_triatt_pro`, `fpf_triatt_epi`, `fpf_transition`). Their cell rows live in
opt_core's table, `common/opt_core/opt_core/attn/pair_fused_cells.json`, resolved once at activation by `opt_core.attn.pair_fused.preflight`
(`hooks/pair_cells.py`); this kit ships no copy. The kit's three shapes and the rows that serve them:

| lever | piece | key | H100 (cc 9.0) row | A100 (cc 8.0) row |
|---|---|---|---|---|
| `triatt_block` | fpf prologue | (c_z 128, 4 heads, 32) | `9.0\|*` | `8.0\|*` |
| `triatt_block` | fpf epilogue | (c_z 128, 4 heads, 32) | `9.0\|*` | `8.0\|*` |
| `pair_transition` | fpf transition | (c_z 128, hidden 512) | `9.0\|*` | `8.0\|*` |

What served prints on the LEVER line: `cells=<cc>|<triton or *> cell_rows=<row ids>` (a table row), `cells=safe settings=safe:no_cell:<shape>` (a
capability without rows: the core's safe settings, engaged), or `cells=none` with every call counted `no-cell:…` (a key the core serves with nothing:
the stock statements run, exit 0). `run.sh check --mode fast` prints the `KERNEL fpf_triatt_pro | fpf_triatt_epi | fpf_transition` route lines.
