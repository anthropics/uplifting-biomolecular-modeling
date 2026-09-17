"""The `ckpt_mmap` lever — this kit's binding of `cells/of3_ckptmmap.py` (exact class, in bytes): the engine's checkpoint read
(`openfold3.core.utils.checkpoint_loading_utils.load_checkpoint`: `torch.load(path, map_location='cpu', weights_only=False)` — the fp32
tensors unpickled into fresh host memory in every process before the model can load them) made through the same call with `mmap=True`:
the storages memory-mapped instead of copied, the same dict of the same tensors handed to the engine's `get_state_dict_from_checkpoint` /
`load_state_dict` — without the copy. Switch `OPENFOLD3_OB0_OPT_CKPT_MMAP=1` (every kit line but `off` exports it). Installed by the package's activation in
every active process (stack.ACTIVATION_MODULES; registry kit `package`). Marker `[openfold3_ob0-opt/ckpt_mmap] installed`; exit census
`[openfold3_ob0-opt/ckpt_mmap] LEVER name=ckpt_mmap state=on|armed|refused|off loads=<n> mmap=<n> fallback=<n> load_s=<s>`."""
from .cells import of3_ckptmmap as _core

ENV = "OPENFOLD3_OB0_OPT_CKPT_MMAP"
TARGET = "openfold3.core.utils.checkpoint_loading_utils"

_core.configure(ENV=ENV, PREFIX="[openfold3_ob0-opt/ckpt_mmap]", M_CKPT=TARGET, REBIND=("openfold3.entry_points.experiment_runner",),
                DIGESTS={"60872dea77deaa3b": "v050"})           # OpenFold3 0.5.0 core/utils/checkpoint_loading_utils.py:44-51 (torch.load(path, map_location='cpu', weights_only=False))

STATE = _core.STATE
VALUES = _core.VALUES
requested, install, uninstall, census_line, serving = _core.requested, _core.install, _core.uninstall, _core.census_line, _core.serving
