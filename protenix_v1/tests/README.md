# Tests

- `opt/protenix_v1_opt/tests/` — the package's own CPU tests (no GPU).
- `opt/forward/v05_addon/lib/kit112_src/infopt_graphs/tests/` — the carried graph-capture library's own GPU tests.

## Running the package tests

Most tests run standalone; a few need the pinned stock package importable to exercise real behavior instead
of skipping (the frozen-weights gate's cache checks, the row-sharded trunk's CPU tests). Install it once, `--no-deps` — its own
dependency list pulls in CUDA-specific packages this suite never needs:

    pip install --no-deps stock/protenix-1.1.0-py3-none-any.whl

`protenix` itself imports fine without torch. If you're in a fully isolated venv (not one that already has
torch), either install torch too or create the venv with `--system-site-packages` so it inherits torch from
whatever environment already has it — some of the carried kit's own vendored tests (`opt/forward/v05_addon/
lib/kit112_src/`) import torch unconditionally at collection time and will error the whole run otherwise.

Then, from `opt/`:

    PYTHONPATH="$(pwd):$(pwd)/../../common/opt_core" python -m pytest -q protenix_v1_opt/tests
