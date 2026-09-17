"""opt_core.seq.numerics: the environment half without torch; the torch half (a façade over opt_core.precision.policy) against a
stand-in torch module (no GPU, no real torch)."""
import sys
import types

import pytest

from opt_core.gates import Gate
from opt_core.precision import FrameworkMissing
from opt_core.seq import numerics


class _Matmul:
    """cuda.matmul stand-in: ``allow_tf32`` coupled to the process matmul precision word, as in torch."""

    def __init__(self, state):
        object.__setattr__(self, "_s", state)

    @property
    def allow_tf32(self):
        if self._s["refuse_old_api"]:
            raise RuntimeError("mixed old and fp32_precision APIs")
        return self._s["precision"] != "highest"

    @allow_tf32.setter
    def allow_tf32(self, v):
        self._s["precision"] = "high" if v else "highest"
        self._s["fp32_precision_mm"] = "tf32" if v else "ieee"

    @property
    def fp32_precision(self):
        return self._s["fp32_precision_mm"]


class _Cudnn:
    def __init__(self, state):
        object.__setattr__(self, "_s", state)
        object.__setattr__(self, "benchmark", False)
        object.__setattr__(self, "deterministic", False)

    @property
    def allow_tf32(self):
        if self._s["refuse_old_api"]:
            raise RuntimeError("mixed old and fp32_precision APIs")
        return self._s["cudnn_tf32"]

    @allow_tf32.setter
    def allow_tf32(self, v):
        self._s["cudnn_tf32"] = bool(v)

    @property
    def conv(self):
        return types.SimpleNamespace(fp32_precision="tf32" if self._s["cudnn_tf32"] else "ieee")


def fake_torch(*, precision="highest", cudnn_tf32=True, cuda_available=False, cuda_initialised=False, refuse_old_api=False):
    """A stand-in exposing exactly the surface policy/recipe read and set (no GPU, no real torch)."""
    s = {"precision": precision, "fp32_precision_mm": "ieee" if precision == "highest" else "tf32", "cudnn_tf32": cudnn_tf32,
         "refuse_old_api": refuse_old_api, "det": False, "warn_only": False, "seed": None, "seed_all": None}
    t = types.ModuleType("torch")
    t._state = s
    t.backends = types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=_Matmul(s)), cudnn=_Cudnn(s))

    def get_precision():
        if s["refuse_old_api"]:
            raise RuntimeError("mixed old and fp32_precision APIs")
        return s["precision"]

    def set_precision(p):
        s["precision"] = p
        s["fp32_precision_mm"] = "ieee" if p == "highest" else "tf32"
    t.get_float32_matmul_precision = get_precision
    t.set_float32_matmul_precision = set_precision
    t.are_deterministic_algorithms_enabled = lambda: s["det"]
    t.is_deterministic_algorithms_warn_only_enabled = lambda: s["warn_only"]

    def use_det(mode, warn_only=False):
        s["det"] = bool(mode)
        s["warn_only"] = bool(warn_only)
    t.use_deterministic_algorithms = use_det
    t.is_autocast_enabled = lambda device_type=None: False
    t.get_autocast_dtype = lambda device_type: "torch.float16"

    def manual_seed(v):
        s["seed"] = v
    t.manual_seed = manual_seed

    def seed_all(v):
        s["seed_all"] = v
    t.cuda = types.SimpleNamespace(is_available=lambda: cuda_available, is_initialized=lambda: cuda_initialised, manual_seed_all=seed_all)
    return t


@pytest.fixture
def torch_in_modules():
    saved = sys.modules.get("torch")

    def install(**kw):
        t = fake_torch(**kw)
        sys.modules["torch"] = t
        return t
    yield install
    if saved is None:
        sys.modules.pop("torch", None)
    else:
        sys.modules["torch"] = saved


def test_override_env_presence_any_value():
    assert numerics.tf32_override_env({}) == []
    assert numerics.tf32_override_env({"NVIDIA_TF32_OVERRIDE": "0"}) == ["NVIDIA_TF32_OVERRIDE"]
    env = {"TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "1", "NVIDIA_TF32_OVERRIDE": "1", "NVIDIA_TF32_OVERRIDES": "x"}
    assert numerics.tf32_override_env(env) == ["NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"]


def test_refusal_sentence_and_gate():
    s = numerics.tf32_override_refusal(["NVIDIA_TF32_OVERRIDE"])
    assert s.startswith("TF32 override variables set in the environment ['NVIDIA_TF32_OVERRIDE']: refused") and s.endswith("unset them")
    g = numerics.tf32_override_gate({})
    assert isinstance(g, Gate) and g.ok and g.name == "tf32_override_env"
    g = numerics.tf32_override_gate({"TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "0"}, name="env")
    assert not g.ok and g.name == "env" and g.reason == numerics.tf32_override_refusal(["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"])
    assert g.details["hits"] == ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"]


def test_env_half_does_not_import_torch():
    saved = sys.modules.pop("torch", None)
    try:
        numerics.tf32_override_gate({"NVIDIA_TF32_OVERRIDE": "1"})
        assert "torch" not in sys.modules
        assert numerics.readback_if_loaded() is None
    finally:
        if saved is not None:
            sys.modules["torch"] = saved


def test_classify_words_in_order():
    prod = dict(matmul_allow_tf32=False, cudnn_allow_tf32=True, cudnn_benchmark=False, cudnn_deterministic=False,
                deterministic_algorithms=False, float32_matmul_precision="highest")
    assert numerics.classify(prod) == "prod"
    assert numerics.classify(dict(prod, deterministic_algorithms=True, matmul_allow_tf32=True)) == "det"          # det wins
    assert numerics.classify(dict(prod, cudnn_deterministic=True)) == "det"
    assert numerics.classify(dict(prod, matmul_allow_tf32=True, cudnn_benchmark=True)) == "tf32-matmul"           # before timing run
    assert numerics.classify(dict(prod, float32_matmul_precision="high")) == "tf32-matmul"
    assert numerics.classify(dict(prod, cudnn_benchmark=True, cudnn_allow_tf32=False)) == "prod+benchmark"          # before fp32-conv
    assert numerics.classify(dict(prod, cudnn_allow_tf32=False)) == "fp32-conv"
    assert set(numerics.CLASSES) == {"det", "tf32-matmul", "prod+benchmark", "fp32-conv", "prod"}
    assert numerics.line_fields(prod) == {"numerics": "prod"} and numerics.line_fields(None) == {"numerics": "unread"}


def test_readback_is_the_policy_signature_projected(torch_in_modules):
    from opt_core.precision import policy
    t = torch_in_modules(precision="high", cudnn_tf32=True)
    t.backends.cudnn.benchmark = True
    rb = numerics.readback({"CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    assert tuple(rb) == numerics.READBACK_KEYS
    sig = dict(policy.numerics_signature(t))
    for field, key in numerics.SIGNATURE_TO_READBACK.items():                     # one producer: every shared fact equals policy's
        assert rb[key] == (str(sig[field]) if key == "float32_matmul_precision" else bool(sig[field]))
    assert rb["matmul_allow_tf32"] is True and rb["cudnn_benchmark"] is True and rb["cudnn_deterministic"] is False
    assert rb["float32_matmul_precision"] == "high" and rb["autocast_gpu_dtype"] == "torch.float16"
    assert rb["cublas_workspace_config"] == ":4096:8" and rb["deterministic_algorithms_warn_only"] is False
    assert numerics.readback_if_loaded({}) == dict(rb, cublas_workspace_config=None)
    assert numerics.classify(rb) == "tf32-matmul"


def test_scoped_tf32_sets_only_the_named_switches_and_restores_on_every_exit(torch_in_modules):
    t = torch_in_modules(precision="highest", cudnn_tf32=False)
    with numerics.scoped_tf32(matmul=True, cudnn=True) as found:
        assert found == {"matmul_allow_tf32": False, "cudnn_allow_tf32": False}
        assert t.backends.cuda.matmul.allow_tf32 is True and t.backends.cudnn.allow_tf32 is True and t.get_float32_matmul_precision() == "high"
    assert t.backends.cuda.matmul.allow_tf32 is False and t.backends.cudnn.allow_tf32 is False and t.get_float32_matmul_precision() == "highest"
    with numerics.scoped_tf32(matmul=True) as found:                            # cudnn unnamed: never written
        assert found == {"matmul_allow_tf32": False} and t.backends.cuda.matmul.allow_tf32 is True and t.backends.cudnn.allow_tf32 is False
    assert t.backends.cuda.matmul.allow_tf32 is False
    with numerics.scoped_tf32() as found:                                       # nothing named: nothing touched
        assert found == {}
    with pytest.raises(KeyError):
        with numerics.scoped_tf32(matmul=True, cudnn=False):
            assert t.backends.cuda.matmul.allow_tf32 is True and t.backends.cudnn.allow_tf32 is False
            raise KeyError("inside")
    assert t.backends.cuda.matmul.allow_tf32 is False and t.backends.cudnn.allow_tf32 is False
    t.set_float32_matmul_precision("medium")                                    # a coarser precision found is put back as found
    with numerics.scoped_tf32(matmul=True):
        assert t.get_float32_matmul_precision() == "high"
    assert t.get_float32_matmul_precision() == "medium"
    t.set_float32_matmul_precision("highest")
    before = numerics.readback()
    with numerics.scoped_tf32(matmul=True, cudnn=True):
        pass
    numerics.assert_unchanged(before, numerics.readback())


def test_assert_unchanged_names_what_moved():
    numerics.assert_unchanged({"a": 1, "b": 2}, {"a": 1, "b": 2, "c": 3})
    with pytest.raises(numerics.NumericsResidue) as e:
        numerics.assert_unchanged({"a": 1, "b": 2}, {"a": 1, "b": 5}, what="flags")
    assert "flags changed across the call" in str(e.value) and "'b': (2, 5)" in str(e.value)


def test_torch_unavailable_is_the_precision_refusal(monkeypatch):
    assert numerics.TorchUnavailable is FrameworkMissing
    monkeypatch.setitem(sys.modules, "torch", None)          # an import of torch now raises ImportError
    with pytest.raises(numerics.TorchUnavailable) as e:
        numerics.readback()
    assert e.value.record()["event"] == "framework_missing"
