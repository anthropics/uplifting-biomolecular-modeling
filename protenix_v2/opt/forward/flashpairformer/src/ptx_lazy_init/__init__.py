"""ptx_lazy_init — skip the dead random parameter init during model construction (the checkpoint load overwrites every tensor). See README.md here. The bundle wires it as a DEFAULT-OFF opt-in (PTX_LAZY_INIT=1|recheck) via src/sitecustomize.py -> install() before any InferenceRunner is built."""
from .ptx_lazy_init import *          # noqa: F401,F403
from .ptx_lazy_init import install, skip_random_init  # noqa: F401
