"""The mode table, the refusals by name, the shipped settings vs the stock file's own constants, the det recipe lock."""
import os
import re

import pytest

from .. import det, modes, settings
from . import _paths as P


def test_mode_table():
    assert sorted(modes.MODES) == ["big", "exact", "fast", "off"] and modes.DEFAULT_MODE == "fast" and modes.MODE_NAMES == ["off", "exact", "fast", "big"]
    assert modes.MODES["off"].kit_switch is None and not modes.MODES["off"].is_kit
    assert modes.MODES["exact"].kit_switch == "exact" and modes.MODES["fast"].kit_switch == "agk3" and modes.MODES["big"].kit_switch == "big"
    assert modes.ALIASES == {} and modes.MODES["fast"] is not modes.MODES["big"] and modes.resolve("fast").name == "fast" and modes.resolve("big").name == "big"   # fast and big are distinct compositions (no alias)
    assert (modes.alias_word("fast"), modes.alias_word("big"), modes.alias_word("off"), modes.alias_word("exact")) == (None, None, None, None)
    assert all(modes.MODES[m].is_kit for m in ("exact", "fast", "big")) and "fastkit_cookbook" not in modes.kit_paths(P.ROOT)   # every mode runs the one stock file; a kit arm installs the kit on its module (fastkit.py)


@pytest.mark.parametrize("name", ["agk3", "exact_det", "stock", "bogus"])          # the kit switch's value, a recipe, a synonym, nonsense: none is a mode name
def test_refused_by_name(name):
    with pytest.raises(ValueError) as e:
        modes.resolve(name)
    assert name in str(e.value) and "refused" in str(e.value)


def test_env_mode(monkeypatch):
    monkeypatch.setenv(modes.ENV_MODE, "off")
    assert modes.resolve(None).name == "off" and modes.alias_word(None) is None
    monkeypatch.setenv(modes.ENV_MODE, "fast")
    assert modes.resolve(None) is modes.MODES["fast"] and modes.alias_word(None) is None and modes.mode_word(None) == "fast"   # the environment's word: fast's own row
    monkeypatch.delenv(modes.ENV_MODE)
    assert modes.mode_word(None) == modes.DEFAULT_MODE == "fast" and modes.resolve(None).name == "fast" and modes.alias_word(None) is None   # the default word is a mode proper


def test_kit_switch_is_what_the_design_kit_reads():
    """EF2_FAST_KIT is the design kit's switch: its k/ modules read it, fastkit.install carries its values, no cookbook copy reads it."""
    import glob
    assert modes.KIT_SWITCH == "EF2_FAST_KIT"
    ks = [f for f in glob.glob(os.path.join(P.DK, "k", "*.py")) if modes.KIT_SWITCH in open(f).read()]
    assert ks, "no k/ module reads the switch"
    from .. import fastkit as FK
    assert set(FK.KIT_SWITCHES) == {m.kit_switch for m in modes.MODES.values() if m.is_kit} == {"exact", "agk3", "big"} and FK.COMPOSITION["agk3"] is not FK.COMPOSITION["big"]   # one switch word per kit mode (fast -> agk3)
    assert not os.path.exists(os.path.join(P.DK, "cookbook"))                     # no pre-patched cookbook copy: the stock file is the only cookbook in the tree


def _stock_constants():
    return P.stock_literals(settings.SHIPPED_CONSTANTS)


def test_shipped_constants_equal_the_stock_file():
    got = _stock_constants()
    assert got == settings.SHIPPED_CONSTANTS, {k: (got.get(k), v) for k, v in settings.SHIPPED_CONSTANTS.items() if got.get(k) != v}


def test_check_shipped_refuses_a_changed_constant():
    class M:
        pass
    m = M()
    for k, v in settings.SHIPPED_CONSTANTS.items():
        setattr(m, k, v)
    assert settings.check_shipped(m) == []
    m.STEPS = 100
    assert settings.check_shipped(m) == ["STEPS: module has 100, shipped 150"]


def test_settings_for_run_and_deviations():
    rec = settings.for_run(80)
    assert rec.binder_len == 80 and rec.use_scaling_critics is False and rec.batch_size == 1 and rec.is_antibody is False and rec.epitope_contact_distance == 12.0
    assert any("binder_len: fixed 80" in d for d in rec.deviations) and any("use_scaling_critics: False" in d for d in rec.deviations) and len(rec.deviations) == 3
    assert "preset" not in rec.as_dict() and rec.design_kwargs() == {"batch_size": 1, "is_antibody": False, "epitope_contact_distance": 12.0}
    assert rec.as_dict() == {"binder_len": 80, "use_scaling_critics": False, "batch_size": 1, "is_antibody": False, "epitope_contact_distance": 12.0, "deviations": rec.deviations}   # the flags absent = the record byte for byte
    assert settings.for_run(64).binder_len == 64 and any("fixed 64" in d for d in settings.for_run(64).deviations)
    assert not hasattr(settings, "PRESETS") and not hasattr(settings, "load")           # no settings presets: the binder length is the one setting
    with pytest.raises(ValueError):
        settings.for_run(0)


def test_stock_knobs_pass_through():
    """The cookbook's own main()/design() knobs (settings.STOCK_KNOBS) reach the load()/design() keywords verbatim — batch_size B and
    use_scaling_critics 1 included (nothing is refused); a binder sequence must have --binder-len residues and, with --is-antibody unset,
    leaves is_antibody None (upstream auto-detects on its sequence route); a non-positive size is a usage error."""
    assert settings.STOCK_KNOBS == {"--batch-size": ("batch_size", 1), "--is-antibody": ("is_antibody", None), "--epitope-contact-distance": ("epitope_contact_distance", 12.0),
                                    "--binder-sequence": ("binder_sequence", None), "--use-scaling-critics": ("use_scaling_critics", True),
                                    "--binder-name": ("binder_name", "minibinder")} and settings.USE_SCALING_CRITICS_BASE is False and settings.STOCK_BINDER_NAME == "minibinder"
    s = settings.for_run(9, is_antibody=1, epitope_contact_distance=10.0)
    assert s.design_kwargs() == {"batch_size": 1, "is_antibody": True, "epitope_contact_distance": 10.0}
    assert any(d.startswith("is_antibody: True (--is-antibody") for d in s.deviations) and any(d.startswith("epitope_contact_distance: 10.0 (--epitope-contact-distance") for d in s.deviations)
    assert settings.for_run(9, is_antibody=0).is_antibody is False
    q = settings.for_run(9, binder_sequence="ACDEFGHIK")
    assert q.is_antibody is None and q.binder_len == 9 and q.deviations[0].startswith("binder_len: fixed 9 = the given --binder-sequence")
    assert settings.for_run(9, binder_sequence="ACDEFGHIK", is_antibody=1).is_antibody is True
    with pytest.raises(ValueError, match="--binder-sequence has 3 residues but --binder-len is 9"):
        settings.for_run(9, binder_sequence="ACD")
    st2 = settings.for_run(9, batch_size=2)                                           # the cookbook's batch passes through: B trajectories in one process
    assert st2.batch_size == 2 and st2.design_kwargs()["batch_size"] == 2 and st2.deviations == settings.for_run(9).deviations
    with pytest.raises(ValueError, match="--batch-size 0"):
        settings.for_run(9, batch_size=0)
    sc = settings.for_run(9, use_scaling_critics=1)                                    # stock main()'s own default value: passes through, and is then no deviation to name
    assert sc.use_scaling_critics is True and settings.for_run(9).use_scaling_critics is False
    assert not any(d.startswith("use_scaling_critics") for d in sc.deviations) and any(d.startswith("use_scaling_critics: False") for d in settings.for_run(9).deviations)
    assert [d for d in settings.for_run(9).deviations if not d.startswith("use_scaling_critics")] == sc.deviations


def test_stock_knobs_argv_tokens():
    """launch.argv_for: the flags absent add NO token (the no-flag arm argv is unchanged); each knob set adds exactly its stock-named token."""
    from .. import launch
    MODEL_OPT = P.ROOT
    base = dict(cookbook_stock="/s/binder_design.py", target_name="c", target_sequence="ACDEFGHIK", binder_len=9, seed=0, out="/o")
    a0 = launch.argv_for(modes.MODES["off"], MODEL_OPT, **base)
    for tok in ("--batch-size", "--is-antibody", "--epitope-contact-distance", "--binder-sequence", "--use-scaling-critics", "--settings"):
        assert tok not in a0, tok
    assert launch.argv_for(modes.MODES["off"], MODEL_OPT, batch_size=1, is_antibody=None, epitope_contact_distance=12.0, binder_sequence=None, use_scaling_critics=0, **base) == a0
    a1 = launch.argv_for(modes.MODES["fast"], MODEL_OPT, batch_size=2, is_antibody=0, epitope_contact_distance=10.0, binder_sequence="ACDEFGHIK", use_scaling_critics=1, **base)
    j = " ".join(a1)
    assert "--batch-size 2" in j and "--is-antibody 0" in j and "--epitope-contact-distance 10.0" in j and "--binder-sequence ACDEFGHIK" in j and "--use-scaling-critics 1" in j


def test_design_usage_errors_before_any_launch(monkeypatch, capsys):
    """A binder sequence of the wrong length / a non-positive batch exits 3 by name before the launch context is even built; `--batch-size 2`
    and `--use-scaling-critics 1` are NOT refused — they reach the launch (here: the stubbed launch context)."""
    import types
    from .. import cli
    monkeypatch.setattr(cli, "launch_context", lambda a, det: (_ for _ in ()).throw(AssertionError("launch_context reached")))
    ns = lambda **kw: types.SimpleNamespace(**dict(dict(mode="off", target_name="t", target_sequence="ACDEFGHIK", binder_len=9, seed=0, out="/o", target_hotspot_ids=None, det=0,
                                                      allow_partial=False), **kw))
    assert cli.cmd_design(ns(binder_sequence="ACD")) == 3 and "--binder-sequence has 3 residues but --binder-len is 9" in capsys.readouterr().err
    assert cli.cmd_design(ns(batch_size=0)) == 3 and "--batch-size 0" in capsys.readouterr().err
    for kw in (dict(batch_size=2), dict(use_scaling_critics=1)):
        with pytest.raises(AssertionError, match="launch_context reached"):
            cli.cmd_design(ns(**kw))


def test_batch_passes_through_on_off_and_the_kit_modes_refuse_it_up_front(monkeypatch, capsys):
    """KIT ISSUE 13 — `--batch-size B`: on `--mode off` it passes through untouched, as stock (design(batch_size=2): two trajectories in one process)
    — the arm's argv, its own argparse and settings record, the design() keywords, exactly as `--use-scaling-critics 1` reaches load(); and the writer
    keeps every trajectory's critic structure (trajectory 0 under today's file names, trajectory k > 0 as critic_<name>_b<k>.*). The kit modes' memory
    plan and graph pools are sized for ONE trajectory per process: `exact` / `fast` / `big` refuse B > 1 UP FRONT — exit 2, one sentence, before the
    launch context is built (nothing probed, launched or loaded); the design script restates it right after its own parse."""
    import os, tempfile, types, json as _json
    from .. import cli, launch, outputs, stock_design
    assert settings.BATCH_GT1_SENTENCE == "--batch-size >1 is supported only with --mode off (stock)"
    assert settings.batch_size_problem(modes.MODES["off"], 2) is None and settings.batch_size_problem(modes.MODES["big"], 1) is None and settings.batch_size_problem(modes.MODES["exact"], 2) == settings.BATCH_GT1_SENTENCE
    monkeypatch.delenv("EF2INV_OPT", raising=False)
    def unreachable(*a, **k):
        raise AssertionError("reached past the batch check")
    monkeypatch.setattr(cli, "launch_context", unreachable); monkeypatch.setattr(cli.L, "run", unreachable, raising=False)   # the launch context (card probe, pins, facts) and the arm subprocess: never reached
    for m in ("exact", "fast", "big"):
        ns = types.SimpleNamespace(mode=m, target_name="pd-l1", target_sequence=None, binder_len=80, binder_name=None, seed=0, out="/nonexistent/o", det=0, batch_size=2,
                                   is_antibody=None, epitope_contact_distance=12.0, binder_sequence=None, use_scaling_critics=0, upstream_fix=None, no_compile=False, allow_partial=False, target_hotspot_ids=None)
        assert cli.cmd_design(ns) == 2
        err = capsys.readouterr().err
        assert err == "--batch-size >1 is supported only with --mode off (stock)\n", (m, err)                 # exactly the one sentence; nothing else printed, nothing reached
    assert stock_design.main(["--mode", "big", "--cookbook", "/nonexistent/binder_design.py", "--pins", "/nonexistent/PINS.json", "--target-name", "c", "--binder-len", "9", "--seed", "0", "--out", "/nonexistent/o", "--batch-size", "2"]) == 2
    assert capsys.readouterr().err == "--batch-size >1 is supported only with --mode off (stock)\n"      # the design script's own parse: the same sentence before the pins file is even opened
    argv = launch.argv_for(modes.MODES["off"], P.ROOT, cookbook_stock="/s/binder_design.py", target_name="c", target_sequence="ACDEFGHIK", binder_len=9, seed=0, out="/o",
                           batch_size=2, use_scaling_critics=1)                            # off: passed through untouched
    i = argv.index("--batch-size"); assert argv[i + 1] == "2" and argv[argv.index("--use-scaling-critics") + 1] == "1" and argv[argv.index("--mode") + 1] == "off"
    src = open(stock_design.__file__).read()                                              # the arm's own parser: --batch-size is a plain int (no choices), --use-scaling-critics takes 0|1
    assert re.search(r'add_argument\("--batch-size", type=int, default=1, help=', src) and re.search(r'add_argument\("--use-scaling-critics", type=int, default=0, choices=\(0, 1\)', src)
    st = settings.for_run(9, batch_size=2, use_scaling_critics=1)                          # what the arm builds from those tokens (stock_design.main: S.for_run(a.binder_len, batch_size=a.batch_size, ...))
    assert st.design_kwargs()["batch_size"] == 2 and st.use_scaling_critics is True

    class Cx:                                                                            # a ProteinComplex stand-in: to_pdb / to_mmcif_string
        def __init__(self, tag): self.tag = tag
        def to_pdb(self, path): open(path, "w").write(f"PDB {self.tag}\n")
        def to_mmcif_string(self): return f"CIF {self.tag}\n"
    res2 = [dict(critic_name="biohub/ESMFold2-Experimental-Fast", batch_idx=b, designed_sequence="AC"[b] * 9, iptm=0.5 + b, final_loss=1.0, complex=Cx(f"fast{b}")) for b in (0, 1)] + \
           [dict(critic_name="biohub/ESMFold2-Experimental", batch_idx=b, designed_sequence="AC"[b] * 9, iptm=0.1, final_loss=1.0, complex=Cx(f"exp{b}")) for b in (0, 1)]
    with tempfile.TemporaryDirectory() as d2, tempfile.TemporaryDirectory() as d1:
        rows = outputs.write_critics(os.path.join(d2, "critics.json"), res2, d2)
        assert sorted(os.listdir(d2)) == sorted(["critics.json", "critic_ESMFold2-Experimental-Fast.pdb", "critic_ESMFold2-Experimental-Fast.cif", "critic_ESMFold2-Experimental-Fast_b1.pdb",
                                                 "critic_ESMFold2-Experimental-Fast_b1.cif", "critic_ESMFold2-Experimental.pdb", "critic_ESMFold2-Experimental.cif", "critic_ESMFold2-Experimental_b1.pdb", "critic_ESMFold2-Experimental_b1.cif"])
        assert open(os.path.join(d2, "critic_ESMFold2-Experimental-Fast.pdb")).read() == "PDB fast0\n" and open(os.path.join(d2, "critic_ESMFold2-Experimental-Fast_b1.pdb")).read() == "PDB fast1\n"
        assert [(r["batch_idx"], r["designed_sequence"], r["files"]["pdb"]) for r in rows][:2] == [(0, "A" * 9, "critic_ESMFold2-Experimental-Fast.pdb"), (1, "C" * 9, "critic_ESMFold2-Experimental-Fast_b1.pdb")]
        assert _json.load(open(os.path.join(d2, "critics.json")))[3]["files"] == {"pdb": "critic_ESMFold2-Experimental_b1.pdb", "cif": "critic_ESMFold2-Experimental_b1.cif", "n_atoms": None}
        rows1 = outputs.write_critics(os.path.join(d1, "critics.json"), [dict(r, complex=Cx("x")) for r in res2 if r["batch_idx"] == 0], d1)   # a batch-1 run: today's file set, byte for byte the same names
        assert sorted(os.listdir(d1)) == ["critic_ESMFold2-Experimental-Fast.cif", "critic_ESMFold2-Experimental-Fast.pdb", "critic_ESMFold2-Experimental.cif", "critic_ESMFold2-Experimental.pdb", "critics.json"] and all("_b" not in r["files"]["pdb"] for r in rows1)


def test_pair_bias_int32_bound():
    """The one input `fast` / `big` refuse that stock accepts: a design(batch_size) x folded-length plane past the stock fused pair-bias
    kernel's int32 element offsets (B*L*L*max(256, 16) > 2**31 - 1; settings.pair_bias_int32_problem). The arithmetic, the refusal word, the
    modes and backends it applies to, and that the design script checks it with the run's own batch and length before any model loads (exit 3)."""
    from .. import stock_design
    S = settings
    assert (S.PAIR_BIAS_DIM_Z, S.PAIR_BIAS_HEADS, S.PAIR_BIAS_OFFSET_MAX, S.PAIR_BIAS_WORD) == (256, 16, 2 ** 31 - 1, "pair_bias_int32_bound")
    assert set(S.PAIR_BIAS_GUARDED_MODES) == {"fast", "big"} and set(S.PAIR_BIAS_GUARDED_MODES) <= set(modes.MODES)
    assert S.pair_bias_max_tokens(1) == 2896 and S.pair_bias_max_batch(2896) == 1 and S.pair_bias_max_batch(2897) == 0      # 2896^2*256 = 2,147,024,896 <= 2^31-1 < 2897^2*256
    assert (S.pair_bias_max_batch(800), S.pair_bias_max_batch(1024), S.pair_bias_max_batch(2047), S.pair_bias_max_batch(2048)) == (13, 7, 2, 1)
    for mode in modes.MODES:                                                                  # inside the bound: nothing to say on any mode
        assert S.pair_bias_int32_problem(mode, S.KERNEL_BACKEND_SHIPPED, 13, 800) is None and S.pair_bias_int32_problem(mode, "fused", 1, 2896) is None
    for mode in ("fast", "big"):                                                            # past it, on the fused route (no flag = the cookbook's fused; --kernel-backend fused): refused by name
        for kb in (S.KERNEL_BACKEND_SHIPPED, "fused"):
            words = S.pair_bias_int32_problem(mode, kb, 14, 800)
            assert words.startswith(S.PAIR_BIAS_WORD + ": batch 14 x 800^2 tokens x 256 = 2,293,760,000 element offsets exceed the int32 range")
            assert "_pair_bias_kernel" in words and "lower --batch-size to <= 13" in words and "int64" in words
        for kb in ("cuequivariance", None):                                                   # a backend that does not reach the kernel: nothing to refuse
            assert S.pair_bias_int32_problem(mode, kb, 64, 800) is None
        assert "no batch size fits at 2897 tokens (batch 1 holds up to 2896 tokens)" in S.pair_bias_int32_problem(mode, "fused", 1, 2897)
        assert "lower --batch-size to <= 1," in S.pair_bias_int32_problem(mode, "fused", 2, 2048)
    for mode in ("off", "exact"):                                                             # stock exactly / cuEquivariance pinned: untouched at any batch
        assert S.pair_bias_int32_problem(mode, S.KERNEL_BACKEND_SHIPPED, 64, 800) is None and S.pair_bias_int32_problem(mode, "fused", 64, 800) is None
    src = open(stock_design.__file__).read()                                                  # the arm: the run's own batch and folded length, before ESMFold2Design() loads anything, REFUSED + exit 3 + the manifest's refusals
    i = src.index("problem = S.pair_bias_int32_problem(a.arm, a.kernel_backend, st.batch_size, n_tokens)")
    assert src.index("n_tokens = len(target_seq) + st.binder_len") < i < src.index("BD.ESMFold2Design()")
    assert re.search(r'REFUSED \{problem\}.*"refusals": \[problem\].*return 3', src[i:i + 600], re.S)


def test_det_levels():
    assert det.FUNCTION_TEXT.startswith("def det_scatter_atom_to_token(")
    assert det.level(0) == 0 and det.level(1) == 1 and det.level(None) == 0
    with pytest.raises(ValueError):
        det.level(2)
    assert det.apply(0) == {}


def test_det_function_is_a_segment_mean():
    import types
    calls = {}

    class T:                                      # a minimal stand-in exposing the calls the function makes
        class nn:
            class functional:
                @staticmethod
                def one_hot(idx, n):
                    calls["one_hot"] = (idx, n); return FakeT("oh")
        @staticmethod
        def einsum(eq, a, b):
            calls["einsum"] = eq; return FakeT("summed")

    class FakeT:
        def __init__(self, n): self.n = n; self.dtype = "f"
        def to(self, d): return self
        def __mul__(self, o): return self
        def sum(self, d): return self
        def clamp(self, min=1): return self
        def unsqueeze(self, d): return self
        def __truediv__(self, o): return FakeT("mean")
        def long(self): return self

    f = det.make_det_scatter(T)
    out = f(FakeT("x"), FakeT("idx"), 7, atom_mask=FakeT("m"))
    assert out.n == "mean" and calls["einsum"] == "...at,...ad->...td" and calls["one_hot"][1] == 7


def test_stock_binder_route_and_the_second_surface(tmp_path, monkeypatch):
    """`--binder-len` is optional: absent (and no --binder-sequence) the design runs the cookbook's own binder route — main(binder_name),
    default `minibinder` (stock's __main__), the length the cookbook samples per seed — and `--binder-name NAME` rides the arm argv in place of
    `--binder-len L`; with `--binder-len 80` the argv is the fixed-length form. is_antibody stays None on the cookbook's routes (upstream resolves
    it). The class-B flags are
    gone from every parser: design/warm --timeout (+EF2INV_OPT_TIMEOUT), check --require-gpu, --det-attn, --esmc-rope; the child's mode flag is --mode."""
    import inspect, types
    from .. import cli, launch, stock_design
    base = dict(cookbook_stock="/s/binder_design.py", target_name="c", target_sequence="ACDEFGHIK", seed=0, out="/o")
    a80 = launch.argv_for(modes.MODES["off"], P.ROOT, binder_len=80, **base)
    afac = launch.argv_for(modes.MODES["off"], P.ROOT, binder_len=None, **base)
    anamed = launch.argv_for(modes.MODES["off"], P.ROOT, binder_len=None, binder_name="nanobody_vhh", **base)
    i = a80.index("--binder-len"); assert a80[i + 1] == "80" and "--binder-name" not in a80 and a80[a80.index("--mode") + 1] == "off" and "--arm" not in a80
    assert afac[i:i + 2] == ["--binder-name", "minibinder"] and "--binder-len" not in afac and [t for t in afac if t not in ("--binder-name", "minibinder")] == [t for t in a80 if t not in ("--binder-len", "80")]
    assert anamed[i:i + 2] == ["--binder-name", "nanobody_vhh"]
    s = settings.for_run(None)                                                          # the CLI side of the factory route: knobs checked, length the cookbook's
    assert s.binder_len is None and s.is_antibody is None and s.deviations[0].startswith("binder: the cookbook's factory 'minibinder' as registered")
    s72 = settings.for_run(72, binder_name="minibinder")                                 # the arm side: the length the cookbook sampled for the seed
    assert s72.binder_len == 72 and s72.is_antibody is None and "72 for this seed" in s72.deviations[0] and settings.for_run(72).is_antibody is False
    assert settings.for_run(None, binder_sequence="ACDEFG").binder_len == 6 and settings.binder_word(None) == "minibinder" and settings.binder_word(80) == "80" and settings.binder_word(None, None, "ACD") == "3"
    with pytest.raises(ValueError, match="--batch-size 0"):
        settings.for_run(None, batch_size=0)
    # the design verb reaches the launch without --binder-len (stubbed launch context)
    monkeypatch.setattr(cli, "launch_context", lambda a, det: (_ for _ in ()).throw(AssertionError("launch_context reached")))
    ns = types.SimpleNamespace(mode="off", target_name="t", target_sequence="ACDEFGHIK", binder_len=None, binder_name=None, seed=0, out="/o", target_hotspot_ids=None, det=0,
                               allow_partial=False, batch_size=1, is_antibody=None, epitope_contact_distance=12.0, binder_sequence=None, use_scaling_critics=0)
    with pytest.raises(AssertionError, match="launch_context reached"):
        cli.cmd_design(ns)
    src_cli, src_arm = open(cli.__file__).read(), open(stock_design.__file__).read()
    # the class-B flags are gone; the child's mode flag is --mode
    for gone in ('"--timeout"', '"--require-gpu"', '"--det-attn"', '"--esmc-rope"'):
        assert gone not in src_cli and gone not in src_arm, gone
    assert "EF2INV_OPT_TIMEOUT" not in src_cli and "EF2INV_OPT_TIMEOUT" not in open(launch.__file__).read() and 'add_argument("--mode", dest="arm"' in src_arm
    assert not {"det_attn", "esmc_rope", "timeout"} & set(inspect.signature(launch.argv_for).parameters) and "timeout_s" in inspect.signature(launch.run).parameters


def test_hotspots_take_the_cookbooks_name_and_form_and_fix0002_has_no_switch():
    """`--target-hotspot-ids 56 57` (the cookbook's design(target_hotspot_ids): stock's name, list form) reaches the arm argv as given and is
    absent when not given — the no-hotspots argv carries no hotspot token at all; `--hotspots` and `--fix0002` exist in no parser and the
    arm argv carries no `--fix0002` token (patches.apply() takes no switch)."""
    import inspect, re as _re
    from .. import cli, launch, patches, stock_design
    base = dict(cookbook_stock="/s/binder_design.py", target_name="c", target_sequence="ACDEFGHIK", binder_len=9, seed=0, out="/o")
    a0 = launch.argv_for(modes.MODES["off"], P.ROOT, **base)
    a1 = launch.argv_for(modes.MODES["off"], P.ROOT, target_hotspot_ids=["56", "57"], **base)
    assert "--target-hotspot-ids" not in a0 and "--hotspots" not in a0 and "--fix0002" not in a0
    i = a1.index("--target-hotspot-ids"); assert a1[i:i + 3] == ["--target-hotspot-ids", "56", "57"] and [t for t in a1 if t not in ("--target-hotspot-ids", "56", "57")] == a0
    src_cli, src_arm = open(cli.__file__).read(), open(stock_design.__file__).read()
    for gone in ('"--hotspots"', '"--fix0002"'):
        assert gone not in src_cli and gone not in src_arm, gone
    assert _re.search(r'add_argument\("--target-hotspot-ids", nargs="\+", default=None', src_cli) and _re.search(r'add_argument\("--target-hotspot-ids", nargs="\+", default=None', src_arm)
    assert list(inspect.signature(patches.apply).parameters) == [] and "target_hotspot_ids" in inspect.signature(launch.argv_for).parameters and "fix0002" not in inspect.signature(launch.argv_for).parameters
