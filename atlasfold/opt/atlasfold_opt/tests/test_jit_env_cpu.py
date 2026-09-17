"""jit_env — the first-use compile caches keyed by the running stack under MODEL_OPT_JIT_ROOT (CPU: a dict environment, tmp dirs)."""
import os

from atlasfold_opt import cli


def test_no_root_touches_nothing():
    env = {"TRITON_CACHE_DIR": "/somewhere"}
    assert cli.jit_env(env) is None and env == {"TRITON_CACHE_DIR": "/somewhere"}


def test_root_keys_unset_caches_and_keeps_preset_dirs(tmp_path):
    kept = tmp_path / "mine"; kept.mkdir()
    env = {cli.JIT_ROOT_ENV: str(tmp_path / "jit"), cli.JIT_KEY_ENV: "torch9.9.9-cu130-sm90", "TRITON_CACHE_DIR": str(kept)}
    note = cli.jit_env(env)
    assert note.startswith(f"jit_root={tmp_path / 'jit'} cache_key=torch9.9.9-cu130-sm90 ")
    assert env["TRITON_CACHE_DIR"] == str(kept) and "TRITON_CACHE_DIR=kept" in note                     # a pre-set existing directory is the caller's
    for var, sub in (("TORCHINDUCTOR_CACHE_DIR", "inductor"), ("TORCH_EXTENSIONS_DIR", "torch_extensions")):
        assert env[var] == os.path.join(str(tmp_path / "jit"), "torch9.9.9-cu130-sm90", sub) and os.path.isdir(env[var]) and f"{var}=keyed" in note
    assert env[cli.JIT_KEY_ENV] == "torch9.9.9-cu130-sm90"


def test_key_resolves_without_torch_import_when_not_given(tmp_path):
    env = {cli.JIT_ROOT_ENV: str(tmp_path)}
    note = cli.jit_env(env)
    assert note.startswith(f"jit_root={tmp_path} cache_key=torch") and env[cli.JIT_KEY_ENV].startswith("torch")
    assert env["TRITON_CACHE_DIR"].startswith(os.path.join(str(tmp_path), env[cli.JIT_KEY_ENV]))
