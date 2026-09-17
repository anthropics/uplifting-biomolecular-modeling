"""The box facts: weights staging into torch.hub's cache (a symlink, never a gate, never read), the cache word, the constants other modules
hold copies of, absolute TORCH_HOME."""
import os

import esm_if1_opt
from esm_if1_opt import batched, det, lines, modes, stack


def test_stage_torch_home_links_and_never_reads(tmp_path, monkeypatch):
    th = tmp_path / "th"
    monkeypatch.setenv("TORCH_HOME", str(th))
    monkeypatch.delenv("ESM_IF1_WEIGHTS", raising=False)
    r = stack.stage_torch_home()
    assert r == {"file": stack.WEIGHTS_FILE, "torch_home": str(th), "source": "torch.hub", "hub_entry": "absent"} and stack.cache_word() == "na"
    w = tmp_path / "w.pt"; w.write_bytes(b"not a checkpoint")                       # content is never read
    monkeypatch.setenv("ESM_IF1_WEIGHTS", str(w))
    r = stack.stage_torch_home()
    entry = th / "hub" / "checkpoints" / stack.WEIGHTS_FILE
    assert r["hub_entry"] == "linked" and entry.is_symlink() and os.path.realpath(entry) == os.path.realpath(w) and stack.cache_word() == "hit"
    assert stack.stage_torch_home()["hub_entry"] == "linked"                          # idempotent
    other = tmp_path / "other.pt"; other.write_bytes(b"x")
    monkeypatch.setenv("ESM_IF1_WEIGHTS", str(other))
    assert stack.stage_torch_home()["hub_entry"] == "linked" and os.path.realpath(entry) == os.path.realpath(other)   # relinked to the named file
    os.unlink(entry); entry.write_bytes(b"regular")
    assert stack.stage_torch_home()["hub_entry"] == "kept" and stack.cache_word() == "miss"                            # a regular file there is kept, and named
    monkeypatch.setenv("ESM_IF1_WEIGHTS", str(tmp_path / "missing.pt"))
    r = stack.stage_torch_home()
    assert r["hub_entry"] == "present" and r["source"].endswith("(not a file)") and stack.cache_word() == "na"


def test_relative_torch_home_is_made_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TORCH_HOME", "rel/th")
    assert stack.torch_home() == os.path.join(str(tmp_path), "rel", "th")
    monkeypatch.delenv("TORCH_HOME")
    monkeypatch.setenv("MODEL_OPT_STATE", "state")
    assert stack.torch_home() == os.path.join(str(tmp_path), "state", "torch_home")


def test_constants_held_equal_across_the_standalone_modules():
    assert lines.UPSTREAM_COMMIT == stack.UPSTREAM_COMMIT and lines.PREFIX == "[" + esm_if1_opt.TAG + "]" == "[esm_if1-opt]"
    assert lines.ROUTES == (modes.STOCK, modes.KIT) and lines.TIMING_ENV == stack.TIMING_ENV == "ESM_IF1_TIMING_JSONL"
    assert batched.DEFAULT_SEED == det.DEFAULT_SEED == 37 and batched.DEFAULT_BATCH == 64
    assert stack.MUST_BE_ABSENT_PREFIXES[0] == modes.ENV_MODE == "ESM_IF1_OPT"
