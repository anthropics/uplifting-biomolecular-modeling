"""opt_core.oom.is_oom — the one out-of-memory classifier (core + kits): six true cases, three false, cause/context walk, no GPU."""
import subprocess
import sys

import pytest

from opt_core.oom import is_oom


class OutOfMemoryError(RuntimeError):      # the class name torch >= 2.5 uses (torch.OutOfMemoryError; torch.cuda.OutOfMemoryError is the same object)
    pass


class XlaRuntimeError(RuntimeError):       # jaxlib's class name
    pass


class JaxRuntimeError(RuntimeError):       # jax >= 0.8's class name (jax.errors.JaxRuntimeError; XlaRuntimeError is an alias name, not in the MRO)
    pass


class ResourceExhaustedError(Exception):   # TensorFlow's class name
    pass


def test_true_cases():
    assert is_oom(OutOfMemoryError("CUDA out of memory. Tried to allocate 4.00 GiB"))                                   # 1 torch-style class (by name)
    assert is_oom(RuntimeError("CUDA out of memory. Tried to allocate 20.00 MiB (GPU 0; 79.10 GiB total capacity)"))   # 2 older torch RuntimeError
    assert is_oom(RuntimeError("CUDA error: out of memory"))                                                            # 3 driver-side wording
    assert is_oom(XlaRuntimeError("RESOURCE_EXHAUSTED: Out of memory while trying to allocate 17179869184 bytes."))     # 4 JAX
    assert is_oom(JaxRuntimeError("RESOURCE_EXHAUSTED: Out of memory while trying to allocate 17179869184 bytes."))     # 4b JAX >= 0.8
    assert is_oom(MemoryError())                                                                                         # 5 host
    assert is_oom(ResourceExhaustedError("OOM when allocating tensor"))                                                 # 5b TensorFlow, by class name
    try:                                                                                                                  # 6 an OOM re-raised as something else (one level of __cause__ / __context__)
        try:
            raise OutOfMemoryError("CUDA out of memory")
        except OutOfMemoryError as e:
            raise ValueError("kernel wrapper failed") from e
    except ValueError as wrapped:
        assert is_oom(wrapped)
    try:
        try:
            raise MemoryError()
        except MemoryError:
            raise KeyError("implicit context")
    except KeyError as ctx:
        assert is_oom(ctx)


def test_false_cases():
    assert not is_oom(RuntimeError("Triton Error [CUDA]: invalid argument"))
    assert not is_oom(XlaRuntimeError("INTERNAL: ptxas exited with non-zero error code"))
    assert not is_oom(JaxRuntimeError("INTERNAL: ptxas exited with non-zero error code"))
    assert not is_oom(ValueError("shape mismatch"))


def test_torch_types_when_torch_is_imported():
    torch = pytest.importorskip("torch")
    kinds = {getattr(torch, "OutOfMemoryError", None), getattr(torch.cuda, "OutOfMemoryError", None)} - {None}
    assert kinds
    for k in kinds:
        assert is_oom(k("CUDA out of memory"))


def test_stdlib_only_at_import():
    code = "import sys; import opt_core.oom; assert 'torch' not in sys.modules and 'jax' not in sys.modules, sorted(m for m in sys.modules if m in ('torch','jax')); print('ok')"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr[-400:]


def test_both_trimul_reroute_sites_ask_is_oom():
    import os
    import opt_core
    core = os.path.dirname(opt_core.__file__)
    lever = open(os.path.join(core, "trimul.py")).read(); cell = open(os.path.join(core, "kernels", "fpf_trimul_v4", "trimul.py")).read()
    assert "from .oom import is_oom" in lever and "if is_oom(e):" in lever
    assert "from opt_core.oom import is_oom" in cell and "is_oom(e)" in cell
    assert "OutOfMemoryError" not in lever.split('"""', 2)[2] and "_OOM" not in cell        # one definition: no second classifier in either file


def test_is_oom_text_shares_the_vocabulary():
    from opt_core.oom import is_oom_text
    assert is_oom_text("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB")
    assert is_oom_text("RuntimeError: CUDA error: out of memory") and is_oom_text("XlaRuntimeError: RESOURCE_EXHAUSTED: Out of memory")
    assert is_oom_text("tensorflow ResourceExhaustedError ...") and is_oom_text("MemoryError")
    assert not is_oom_text("RuntimeError: CUDA error: an illegal memory access was encountered") and not is_oom_text("")
