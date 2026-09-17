import os
import jax  # noqa: F401
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"
from colabdesign.af.model import mk_af_model  # noqa: E402
mk_afdesign_model = mk_af_model


def clear_mem():
    pass
