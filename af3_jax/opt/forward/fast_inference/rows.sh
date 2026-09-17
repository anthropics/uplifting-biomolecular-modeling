# The base kit's command-line rows, read as TEXT by the af3_jax_opt package (opt/af3_jax_opt/modes.py kit_rows) — never executed:
# COMMON = the flags every route shares, FAST = the row levers' flags (L1 featurisation prefetch, WRITER output writer thread; on exact, fast and big). Shell syntax with the variables
# left unexpanded: the package fills $CACHE and $IN; the $MODEL placeholder (the model flags) is the variant's weights flags plus whatever
# stock flags the caller states, composed by the package (stock_pred.compose).
COMMON="--norun_data_pipeline --cache_dir=$CACHE --input_dir=$IN $MODEL"
FAST="--featurisation_workers=3 --featurisation_prefetch=4 --output_writer"
