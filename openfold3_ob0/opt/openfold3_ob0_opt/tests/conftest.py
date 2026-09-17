"""Session setup for the kit's tests: when the openfold3 wheel is installed it is imported FIRST (its attention primitive included), so every
test that shadows openfold3 modules with stand-ins (``test_tp_rowpair_census_cpu``) runs against a session that already holds the real package —
the order a GPU image produces — and a stand-in that leaked past its test would fail the dense-reference tests after it by name.

A session without a CUDA device runs the tp line's triangle-attention lever under the core's opt-out ``ROWPAIR_TRIATT_CORE=torch`` (the carried
flash kernel needs a CUDA device; without the opt-out the core refuses the lever by name — test_tp_kernels_cpu asserts that refusal and manages the
variable itself): every CPU rank then runs OpenFold3's torch attention statement, counted as ``fallback_by=kernel_torch``. The triangle-multiplication
provider needs no opt-out at these sizes (every unit is below its pair-size gate: ``below_gate``)."""
import os
try:                                                                    # absent wheel: the openfold3-dependent tests skip by name as before
    import openfold3.core.model.primitives.attention  # noqa: F401
    import openfold3.projects.of3_all_atom.model  # noqa: F401
except Exception:                                                       # noqa: BLE001
    pass

try:
    import torch as _torch
    if not _torch.cuda.is_available():
        os.environ.setdefault("ROWPAIR_TRIATT_CORE", "torch")
except Exception:                                                       # noqa: BLE001
    pass
