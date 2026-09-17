"""R-TP-3 (no size-gated replication under ``--mode big --n_gpu P``): the boltz2 TP adapter has NO threshold below / above which a pair
statement runs replicated or gathered instead of sharded — every pair stack, the pair init, the template / MSA modules and the distogram run
row-sharded at every N. The thresholds that exist in this kit are chunking / kernel-route levers that act INSIDE a rank's rows (they compose
with sharding and never replicate): boltz's ``const.chunk_size_threshold`` (384: chunked MSA / triangle-attention statements),
``BOLTZ_XL_MIN_TOKENS`` (the memory line's row-chunk levers), ``BOLTZ_TRIATTN_MIN_TOKENS`` (the flash triangle attention route). This test is
the grep-able guard: the adapter source carries no size gate, and ``SIZE_GATES`` (the one census list this file checks a value for) is the
empty tuple."""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, "..", "rowpair.py"), encoding="utf-8").read()

FORBIDDEN = [r"REPLICATE_BELOW", r"GATHER_BELOW", r"TP_ABOVE", r"MIN_TOKENS", r"if\s+N\s*[<>]=?\s*\d", r"shape\[1\]\s*[<>]=?\s*\d{3,}"]


def test_adapter_has_no_size_gated_replicated_path():
    hits = {pat: re.findall(pat, SRC) for pat in FORBIDDEN}
    hits = {k: v for k, v in hits.items() if v}
    assert not hits, f"size gate tokens in boltz2_opt/rowpair.py: {hits} (R-TP-3: a speed-gated replicated path may only exist as a separately named mode)"


def test_boltz2_has_no_size_gated_replication_fallback():
    """SIZE_GATES must be the empty tuple: boltz2 forces the sharded path at every size -- never a silent size-based fallback to
    replication (a size gate here would be the same defect class as a silent drop)."""
    from boltz2_opt import rowpair
    assert rowpair.SIZE_GATES == (), rowpair.SIZE_GATES


def test_the_memory_reach_probe_shape_is_two_boltz_predict_options():
    """The memory-reach probe's degraded shape (1 recycle, 2 denoising steps) is given as `boltz predict`'s own options like any other call
    (no preset): the batch carries them, the ACTIVE line names them, the KERNELS census says settings=flags."""
    from boltz2_opt import settings, stack
    eff = settings.worker_settings("big", {"recycling_steps": 1, "sampling_steps": 2})
    assert (eff["recycling_steps"], eff["sampling_steps"], eff["diffusion_samples"]) == (1, 2, 1) and settings.settings_tokens(eff) == "recycling_steps=1 sampling_steps=2"
    cmd = stack.worker_command("b.json", "big", 8, settings.settings_word({"recycling_steps": 1, "sampling_steps": 2}))
    assert cmd[cmd.index("--kernels-settings") + 1] == "flags" and "--attach" in cmd and "settings" not in cmd[cmd.index("--attach") + 1].split(","), "no settings attachment: the batch carries the values"
    env = stack.child_env("big", base={}, n_gpu=8)
    assert env["BOLTZ_TP"] == "8" and not any(k.startswith(("BOLTZ_RECYCLING", "BOLTZ_SAMPLING")) for k in env)


def test_unbound_head_is_refused_by_name():
    from boltz2_opt import rowpair
    saved = dict(rowpair.HEADS)
    try:
        rowpair.HEADS["confidence_rows"] = None
        import pytest
        with pytest.raises(rowpair.Refused, match="heads seam confidence_rows unbound"):
            rowpair._head("confidence_rows")
    finally:
        rowpair.HEADS.update(saved)
