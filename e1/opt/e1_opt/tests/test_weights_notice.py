"""Weights are digested, worded and run — never refused for their digest. ``pins.assert_weights(size)`` takes the sha256 (and size) of
``model.safetensors`` in the pinned snapshot once per activation and words it against the pin: the pinned digest is ``tested``
(``[e1-opt] weights=<repo>@<rev> sha256=<12> (pinned)`` on stderr), any other digest is ``untested`` (``[e1-opt] weights sha256=<12>
NOT PINNED — the kit's numbers apply to the pinned weights only (...)``) and both run; only a snapshot with no
``model.safetensors`` refuses, by name (``PinDrift``). Three cases on the mocked box (a placeholder checkpoint in a temp HF cache, the pins
module's table monkeypatched; no GPU, no network, no real weights)."""
import os

import pytest

from e1_opt import cli, stack
from e1_opt.tests import _stubs


def _score(capsys, box, tmp_path):
    items = _stubs.write_items(str(tmp_path / "in"), names=("x",))
    out = str(tmp_path / "out")
    rc = cli.main(["score", "--variant", box["variant"], "--mode", "off", "--parent-path", os.path.join(items, "x", "parent.fasta"), "--mutants-path", os.path.join(items, "x", "mutants.fasta"), "--output-path", os.path.join(out, "x", "scores.csv")])
    cap = capsys.readouterr()
    return rc, cap.out.splitlines(), cap.err.splitlines(), stack.status()          # the activation report of the run (its pins record: the weights word)


def test_unknown_digest_runs_with_the_notice(monkeypatch, tmp_path, capsys):
    box = _stubs.setup_box(monkeypatch, tmp_path)                       # the placeholder checkpoint, the weights table NOT patched to it: an unknown digest
    pins, v = box["pins"], box["variant"]
    monkeypatch.setitem(pins.KERNEL, "layer_norm_py_sha256", box["caches"]["kernel_sha256"])   # the hub kernel placeholder is the box's, not the subject
    rec = pins.assert_weights(v)                                         # no exception: the record carries the digest, the pin and the word
    assert rec["word"] == pins.WEIGHTS_WORDS[1] == "not-pinned"
    assert (rec["sha256"], rec["bytes"]) == (box["caches"]["weights_sha256"], box["caches"]["weights_bytes"]) and rec["pinned_sha256"] == pins.WEIGHTS[v]["sha256"] != rec["sha256"]
    line = pins.weights_words(rec)
    assert line == (f"weights sha256={rec['sha256'][:12]} NOT PINNED — the kit's numbers apply to the pinned weights only "
                    f"({v}: {rec['sha256']} ({rec['bytes']} B) != pin {rec['pinned_sha256']} ({rec['pinned_bytes']} B))")
    rc, out, err, rep = _score(capsys, box, tmp_path)                    # the wrapper command end to end on those weights: it runs
    assert rc == 0, (out, err)
    assert [ln for ln in err if pins.WEIGHTS_NOTICE in ln] == [f"[e1-opt] {line}"]          # said once, on stderr
    assert rep["pins"]["refusals"] == [] and rep["pins"]["checks"]["weights"]["detail"]["word"] == "not-pinned"
    assert rep["pins"]["checks"]["weights"]["detail"]["sha256"] == rec["sha256"]              # the digest is the record either way


def test_pinned_digest_says_pinned(monkeypatch, tmp_path, capsys):
    box = _stubs.setup_box(monkeypatch, tmp_path)
    _stubs.patch_pins_for_stub(monkeypatch, box["pins"], box["variant"], box["caches"])       # the pins table names the placeholder's digest: pinned
    pins, v = box["pins"], box["variant"]
    rec = pins.assert_weights(v)
    assert rec["word"] == pins.WEIGHTS_WORDS[0] == "pinned" and rec["sha256"] == rec["pinned_sha256"]
    assert pins.weights_words(rec) == f"weights={pins.WEIGHTS[v]['repo']}@{pins.WEIGHTS[v]['rev'][:12]} sha256={rec['sha256'][:12]} (pinned)"
    rc, out, err, rep = _score(capsys, box, tmp_path)
    assert rc == 0, (out, err)
    assert err[0] == f"[e1-opt] {pins.weights_words(rec)}" and not any(pins.WEIGHTS_NOTICE in ln for ln in err)
    assert rep["pins"]["checks"]["weights"]["detail"]["word"] == "pinned"


def test_missing_checkpoint_is_refused_by_name(monkeypatch, tmp_path, capsys):
    box = _stubs.setup_box(monkeypatch, tmp_path)
    _stubs.patch_pins_for_stub(monkeypatch, box["pins"], box["variant"], box["caches"])
    pins, v = box["pins"], box["variant"]
    empty = tmp_path / "empty_snapshot"
    empty.mkdir()
    with pytest.raises(pins.PinDrift) as ei:
        pins.assert_weights(v, str(empty))
    assert str(ei.value) == f"{v}: no model.safetensors at {empty}"
    monkeypatch.setattr(pins, "weights_snapshot", lambda size, hf_home=None: str(empty))     # the activation's own lookup lands on the empty snapshot
    monkeypatch.setattr(stack.registry, "snapshot_dir", lambda variant, p: str(empty), raising=False)
    rc, out, err, rep = _score(capsys, box, tmp_path)
    assert rc == 3 and out[0].startswith("[e1-opt] NOT ACTIVE: weights: ") and f"{v}: no model.safetensors at {empty}" in out[0], (out, err)   # a missing weights file IS a refusal (nothing can run), by name
    assert rep["pins"]["checks"]["weights"]["ok"] is False and not any("weights" in ln for ln in err)
