"""opt_core.precision.policy — the TF32 library-override environment gate (standard library only; no torch)."""
from __future__ import annotations

from opt_core.gates import Gate
from opt_core.precision import policy


def test_override_env_presence_any_value():
    assert policy.tf32_override_env({}) == []
    assert policy.tf32_override_env({"NVIDIA_TF32_OVERRIDE": "0"}) == ["NVIDIA_TF32_OVERRIDE"]          # =0 moves a class too
    env = {"TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "1", "NVIDIA_TF32_OVERRIDE": "1", "NVIDIA_TF32_OVERRIDES": "x"}
    assert policy.tf32_override_env(env) == ["NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"]
    assert policy.TF32_OVERRIDE_ENV == ("NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE")


def test_refusal_sentence_and_gate():
    s = policy.tf32_override_refusal(["NVIDIA_TF32_OVERRIDE"])
    assert s.startswith("TF32 override variables set in the environment ['NVIDIA_TF32_OVERRIDE']: refused") and s.endswith("unset them")
    g = policy.tf32_override_gate({})
    assert isinstance(g, Gate) and g.ok and g.name == "tf32_override_env" and g.details["hits"] == []
    g = policy.tf32_override_gate({"TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "0"}, name="env")
    assert not g.ok and g.name == "env" and g.reason == policy.tf32_override_refusal(["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"])
    assert g.details == {"hits": ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"], "checked": ["NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"]}
