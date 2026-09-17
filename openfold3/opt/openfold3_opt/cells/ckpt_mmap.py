"""Cell `ckpt_mmap` — this kit's binding of `of3_ckptmmap` (exact class, in bytes): the engine's checkpoint read
(`openfold3.core.utils.checkpoint_loading_utils.load_checkpoint`: `torch.load(path)` — 2.3 GB of CPU-tagged fp32 tensors unpickled into fresh host
memory before the model can load them) made through the same call with `mmap=True`: the storages memory-mapped instead
of copied, the same dict of the same tensors handed to the engine's `get_state_dict_from_checkpoint` / `load_state_dict`. Switch
`OPENFOLD3_OPT_CKPT_MMAP=1` (every kit line but `off` exports it). Installed by the package at activation (openfold3_opt.stack.activate,
modes.ACTIVATION_CELLS): the checkpoint read is engine plumbing every line shares."""
from . import of3_ckptmmap as _core

ENV = "OPENFOLD3_OPT_CKPT_MMAP"

_core.configure(ENV=ENV, PREFIX="[openfold3-opt/ckpt_mmap]", M_CKPT="openfold3.core.utils.checkpoint_loading_utils",
                REBIND=("openfold3.entry_points.experiment_runner",),
                DIGESTS={"fff61c486d8fb129": "v041"})           # OpenFold3 0.4.1 core/utils/checkpoint_loading_utils.py:44-51 (torch.load(ckpt_path))

STATE = _core.STATE
VALUES = _core.VALUES
requested, install, uninstall, census_line, serving = _core.requested, _core.install, _core.uninstall, _core.census_line, _core.serving
