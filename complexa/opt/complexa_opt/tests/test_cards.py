"""The two deployment files and the JIT-cache key. `stack.jit_key` is the core's key rule over the running stack (torch<version>-cu<CUDA>-sm<cc>
from distribution metadata and nvidia-smi; the word `unknown`, named on stderr, when a part cannot be established — never a refusal). configs/h100.env
keys the shared JIT root by that key when MODEL_OPT_JIT_ROOT is set — torch2.7.0-cu126-sm90 on an H100 at the pin, a pre-set MODEL_OPT_JIT_KEY kept, an
underivable key the directory `unknown` (return code 0) — and exports nothing of it when no root is named. configs/a100.env sets the A100's target word and sources
configs/h100.env: the same rule yields torch2.7.0-cu126-sm80 there. The stack is faked for the sourcing tests: a `torch` dist-info first on the
child's path and an `nvidia-smi` on its PATH, so the answers do not depend on the test box."""
import os
import subprocess
import sys

from complexa_opt import stack
from opt_core import gates, jit_cache

TREE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))     # complexa/
CONFIGS = os.path.join(TREE, "configs")
H100_SMI = "NVIDIA H100 80GB HBM3, 9.0, 81559"
A100_SMI = "NVIDIA A100-SXM4-80GB, 8.0, 81920"


def _code_lines(text):
    return [ln.split("#", 1)[0] for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]


def test_jit_key_is_the_cores_rule_over_the_running_stack(monkeypatch, capsys):
    versions = {"torch": "2.7.0+cu126"}
    monkeypatch.setattr(gates, "dist_version", lambda name: versions.get(name))
    monkeypatch.setattr(gates, "nvidia_smi_probe", lambda *a, **k: {"name": "NVIDIA A100-SXM4-80GB", "cc": "8.0", "sm": "sm80", "memory_mib": 81920, "probe": "nvidia-smi"})
    assert stack.jit_key() == "torch2.7.0-cu126-sm80" == jit_cache.key()
    monkeypatch.setattr(gates, "nvidia_smi_probe", lambda *a, **k: {"name": "NVIDIA H100 80GB HBM3", "cc": "9.0", "sm": "sm90", "memory_mib": 81559, "probe": "nvidia-smi"})
    assert stack.jit_key() == "torch2.7.0-cu126-sm90"
    monkeypatch.setattr(gates, "nvidia_smi_probe", lambda *a, **k: {"name": None, "cc": None, "sm": None, "memory_mib": None, "probe": "nvidia-smi unavailable (FileNotFoundError)"})
    assert stack.jit_key() == "unknown"                                    # no GPU visible: the word `unknown`, the missing part named on stderr — never a guessed suffix, never a refusal
    assert "JIT-CACHE key=unknown: cache key: cannot establish cc" in capsys.readouterr().err


def test_one_file_per_card_and_the_a100_file_is_the_h100_file_with_its_word():
    assert sorted(os.listdir(CONFIGS)) == ["a100.env", "h100.env", "h200.env"]
    a100 = open(os.path.join(CONFIGS, "a100.env"), encoding="utf-8").read()
    h100 = open(os.path.join(CONFIGS, "h100.env"), encoding="utf-8").read()
    code = _code_lines(a100)
    assert [ln.strip() for ln in code] == ["export MODEL_OPT_TARGET_GPU=${MODEL_OPT_TARGET_GPU:-A100}", '. "$(dirname "${BASH_SOURCE[0]}")/h100.env"'], code   # the word, then the H100 file — the `.` line last, so its refusals are this file's
    hcode = "\n".join(_code_lines(h100))
    assert 'MODEL_OPT_JIT_KEY=$(python -c "from complexa_opt import stack; print(stack.jit_key())"' in hcode
    assert "sm90" not in hcode and "sm80" not in hcode                        # the arch suffix is derived, never written
    assert not [ln for ln in _code_lines(h100) + code if "COMPLEXA_OPT" in ln or "++" in ln]   # a deployment file selects no mode and sets no sampling value


def _fake_stack(tmp_path, smi_line):
    """A child environment whose stack answers are pinned: `python` = this interpreter, a torch 2.7.0+cu126 dist-info first on its path, and an
    nvidia-smi printing `smi_line` (None: one that fails, as on a box with no GPU)."""
    bindir, site = tmp_path / "bin", tmp_path / "site"
    bindir.mkdir(exist_ok=True); site.mkdir(exist_ok=True)
    py = bindir / "python"
    py.write_text(f"#!/bin/bash\nexec {os.path.abspath(sys.executable)} \"$@\"\n"); py.chmod(0o755)
    smi = bindir / "nvidia-smi"
    smi.write_text("#!/bin/bash\nexit 9\n" if smi_line is None else f"#!/bin/bash\necho '{smi_line}'\n"); smi.chmod(0o755)
    di = site / "torch-2.7.0+cu126.dist-info"
    di.mkdir(exist_ok=True)
    (di / "METADATA").write_text("Metadata-Version: 2.1\nName: torch\nVersion: 2.7.0+cu126\n")
    pypath = os.pathsep.join([str(site)] + [p for p in sys.path if p])
    return {"PATH": f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}", "HOME": str(tmp_path), "MODEL_OPT": TREE, "PYTHONPATH": pypath}


def _source(name, env, **extra):
    """`source configs/<name>.env` in a bash child; echoes what the target word and the JIT block set. (returncode, stdout, stderr)."""
    cmd = f'source {os.path.join(CONFIGS, name + ".env")} || exit $?; echo "G=$MODEL_OPT_TARGET_GPU X=${{TORCH_EXTENSIONS_DIR:-}} T=${{TRITON_CACHE_DIR:-}} K=${{MODEL_OPT_JIT_KEY:-}}"'
    r = subprocess.run(["bash", "-c", cmd], env=dict(env, **extra), capture_output=True, text=True, timeout=120)
    return r.returncode, r.stdout, r.stderr


def test_h100_file_keys_the_jit_root_by_the_running_stack(tmp_path):
    env = _fake_stack(tmp_path, H100_SMI)
    rc, out, err = _source("h100", env, MODEL_OPT_JIT_ROOT="/jit")
    assert rc == 0, err[-600:]
    assert "G=H100 X=/jit/torch2.7.0-cu126-sm90/torch_ext T=/jit/torch2.7.0-cu126-sm90/triton K=torch2.7.0-cu126-sm90" in out
    assert "NOT ACTIVE" not in err
    rc, out, err = _source("h100", env, MODEL_OPT_JIT_ROOT="/jit", MODEL_OPT_JIT_KEY="mykey")             # a pre-set key is kept, nothing derived
    assert (rc, "G=H100 X=/jit/mykey/torch_ext T=/jit/mykey/triton K=mykey" in out) == (0, True), (out, err[-600:])
    rc, out, err = _source("h100", _fake_stack(tmp_path, None))                                             # no root named: nothing keyed, no GPU needed
    assert (rc, "G=H100 X= T= K=" in out) == (0, True), (out, err[-600:])


def test_a100_file_is_the_same_rule_with_its_word(tmp_path):
    env = _fake_stack(tmp_path, A100_SMI)
    rc, out, err = _source("a100", env, MODEL_OPT_JIT_ROOT="/jit")
    assert rc == 0, err[-600:]
    assert "G=A100 X=/jit/torch2.7.0-cu126-sm80/torch_ext T=/jit/torch2.7.0-cu126-sm80/triton K=torch2.7.0-cu126-sm80" in out
    rc, out, err = _source("a100", env)
    assert (rc, "G=A100 X= T= K=" in out) == (0, True), (out, err[-600:])
    rc, out, err = _source("a100", env, MODEL_OPT_TARGET_GPU="A100-80GB")                                   # a launcher's own word is kept
    assert "G=A100-80GB " in out


def test_a_key_that_cannot_be_derived_is_the_word_unknown_named_not_refused_on_either_card(tmp_path):
    env = _fake_stack(tmp_path, None)                                                                        # no GPU visible to nvidia-smi: environment uncertainty is named, never a reason to refuse
    for name, word in (("h100", "H100"), ("a100", "A100")):
        rc, out, err = _source(name, env, MODEL_OPT_JIT_ROOT="/jit")
        assert rc == 0, (name, out, err[-600:])
        assert f"G={word} X=/jit/unknown/torch_ext T=/jit/unknown/triton K=unknown" in out, (name, out)   # keyed under <root>/unknown/, a directory no pinned stack shares; the file goes on
        assert "[complexa-opt] JIT-CACHE key=unknown: cache key: cannot establish cc" in err                # the core's own words for the missing part
        assert "NOT ACTIVE" not in err
