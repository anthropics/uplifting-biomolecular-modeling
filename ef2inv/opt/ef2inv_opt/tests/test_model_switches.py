"""The two model switches (modes.py values per mode; settings grammar; stock_design.apply_model_switches — every loaded model, read back; launch.argv_for;
cli.effective_switches; manifest.switch_records). Stock = the cookbook + pair stack unchunked + cuEquivariance kernel backend: `off` and `exact`
carry chunk None + "cuequivariance" and their arm argv says so; `fast` / `big` carry shipped and their argv is the one they always had. A user
`--chunk-size` / `--kernel-backend` replaces the mode's value for one run: opt_manifest.json `overrides`, and for the backend ONE NOTE line saying
what the value reaches under grad (fused: no-grad folds only; cuequivariance: NOT grad-gated; None: reference); the mode's own values are silent."""
import json
import os
import types

import pytest

from .. import cli, launch as L, manifest as M, modes as MD, report as R, settings as S, stock_design as SD
from . import _paths as P


def test_the_mode_table_carries_the_stock_definition():
    """off makes no setter call (the cookbook as shipped); STOCK_* = the stock arm's two documented settings, pinned by exact; fast / big no call."""
    assert (MD.STOCK_CHUNK_SIZE, MD.STOCK_KERNEL_BACKEND) == (None, "cuequivariance")
    assert (MD.MODES["off"].chunk_size, MD.MODES["off"].kernel_backend) == ("shipped", "shipped")
    assert (MD.MODES["exact"].chunk_size, MD.MODES["exact"].kernel_backend) == (None, "cuequivariance")
    for m in ("fast", "big"):
        assert (MD.MODES[m].chunk_size, MD.MODES[m].kernel_backend) == ("shipped", "shipped")


def test_grammar_is_the_forks_literals():
    assert S.KERNEL_BACKENDS == ("fused", "cuequivariance", None)
    assert S.parse_kernel_backend("fused") == "fused" and S.parse_kernel_backend("cuequivariance") == "cuequivariance"
    assert S.parse_kernel_backend("None") is None and S.parse_kernel_backend("none") is None
    for bad in ("cueq", "torch", "reference", "", "FUSED", "shipped"):                          # `shipped` is no flag value: omitting the flag = no call
        with pytest.raises(ValueError) as e:
            S.parse_kernel_backend(bad)
        assert repr(bad) in str(e.value) and "fused | cuequivariance | None" in str(e.value)          # refused by name, the legal words listed
    assert S.parse_chunk_size("none") is None and S.parse_chunk_size("64") == 64
    for bad in ("0", "-4", "abc", "", "64.5", "shipped"):
        with pytest.raises(ValueError):
            S.parse_chunk_size(bad)
    assert S.kernel_backend_word(None) == "None" and {"chunk_size", "kernel_backend"} <= set(S.SHIPPED_CALLS)


def test_note_grammar_and_the_grad_facts():
    """Every value the flag's parser returns — the fork's three literals — has its NOTE (printed for a user override of a kit mode's value):
    a flag naming any legal word resolves a line, never a KeyError."""
    for v in S.KERNEL_BACKENDS:
        line = R.note_line(*S.kernel_backend_note(v))
        assert line.startswith(f"NOTE --kernel-backend {S.kernel_backend_word(v)} — ") and line.endswith("; proceeding") and "REFUSED" not in line
    assert S.KERNEL_BACKEND_WORDS == ("fused", "cuequivariance", "None", "none")
    for w in S.KERNEL_BACKEND_WORDS:
        assert S.parse_kernel_backend(w) in S.KERNEL_BACKEND_NOTES, w                              # the notes table covers the parser's whole range
    assert "no-grad folds" in S.KERNEL_BACKEND_NOTES["fused"] and "85-92" in S.KERNEL_BACKEND_NOTES["fused"]
    assert "NOT grad-gated" in S.KERNEL_BACKEND_NOTES["cuequivariance"] and "95-96" in S.KERNEL_BACKEND_NOTES["cuequivariance"]
    assert "reference modules in every fold" in S.KERNEL_BACKEND_NOTES[None]
    assert S.SHIPPED_CALLS["kernel_backend"].startswith("fused when TRITON_KERNELS_AVAILABLE")      # what plain off keeps: the loader's selection, l.1247-1252 of the stock file
    src = open(P.STOCK_FILE).read().splitlines()[1246:1252]
    assert [s.strip() for s in src] == ["kernel_backend = None", "if TRITON_KERNELS_AVAILABLE:", "kernel_backend = BACKEND_FUSED", "elif CUE_AVAILABLE:",
                                        "kernel_backend = BACKEND_CUEQ", "model.set_kernel_backend(kernel_backend)"]


class _Mod:
    """A torch-module-like node: named_modules() walks the tree; the fork's switch attributes live in the instance dict where the fork keeps them."""
    def __init__(self, **attrs):
        self.__dict__.update(attrs); self._children = {}
    def add(self, name, mod):
        self._children[name] = mod; setattr(self, name, mod); return mod
    def named_modules(self, prefix=""):
        yield prefix, self
        for n, c in self._children.items():
            yield from c.named_modules(f"{prefix}.{n}" if prefix else n)


def _fork_model(n_blocks=2, msa_encoder=False, broken=None):
    """The shape of ESMFold2ExperimentalModel for the two setters (modeling_esmfold2_experimental.py:577-588): folding_trunk.blocks[i] =
    PairUpdateBlock(_kernel_backend) with the tri_mul_out/in modules (_use_kernels, _chunk_size) and pair_transition (_kernel_backend, _chunk_size);
    confidence_head.folding_trunk likewise; structure_head…attn (_kernel_backend); an optional msa_encoder whose triangle modules the model setters do NOT
    reach (hero critic 3). The cookbook's own call left every reached module on "fused". `broken` names a module path the setter skips."""
    m = _Mod(); m.calls = []
    def trunk():
        t = _Mod(); blocks = t.add("blocks", _Mod())
        for b in range(n_blocks):
            blk = blocks.add(str(b), _Mod(_kernel_backend="fused"))
            for tm in ("tri_mul_out", "tri_mul_in"):
                blk.add(tm, _Mod()).add("_engine", _Mod(_use_kernels=False, _chunk_size=64))
            blk.add("pair_transition", _Mod(_kernel_backend="fused", _chunk_size=64))
        return t
    m.add("folding_trunk", trunk()); m.add("confidence_head", _Mod()).add("folding_trunk", trunk())
    attn = m.add("structure_head", _Mod()).add("diffusion_module", _Mod()).add("token_transformer", _Mod()).add("attn_blocks", _Mod())
    for a in range(2):
        attn.add(str(a), _Mod(_kernel_backend="fused"))
    if msa_encoder:
        blk = m.add("msa_encoder", _Mod()).add("blocks", _Mod()).add("0", _Mod())
        for tm in ("tri_mul_out", "tri_mul_in"):
            blk.add(tm, _Mod()).add("_engine", _Mod(_use_kernels=False, _chunk_size=64))
        blk.add("outer_product_mean", _Mod(_chunk_size=None))
    def set_kernel_backend(v):
        m.calls.append(("set_kernel_backend", v))
        for root in (m.folding_trunk, m.confidence_head.folding_trunk, m.structure_head):
            for path, mod in root.named_modules():
                if broken and path.endswith(broken):
                    continue
                if "_kernel_backend" in vars(mod): mod._kernel_backend = v
                if "_use_kernels" in vars(mod): mod._use_kernels = (v == "cuequivariance")
    def set_chunk_size(v):
        m.calls.append(("set_chunk_size", v))
        for root in (m.folding_trunk, m.confidence_head.folding_trunk):
            for path, mod in root.named_modules():
                if broken and path.endswith(broken):
                    continue
                if "_chunk_size" in vars(mod): mod._chunk_size = v
    m.set_kernel_backend = set_kernel_backend; m.set_chunk_size = set_chunk_size
    return m


NAMES = ("ESMFold2-Experimental-Fast", "ESMFold2-Experimental-Fast-Cutoff2025")        # the cookbook loads the SAME two repos as inversion models and as hero critics


def _app(**kw):
    """The loaded cookbook app's model containers: inversion_models and hf_critic_models under the same names (distinct objects), plus the
    attributes that are not ESMFold2 models (the shared ESMC, strings)."""
    return types.SimpleNamespace(inversion_models={n: _fork_model(**kw) for n in NAMES},
                                 hf_critic_models={NAMES[0]: _fork_model(**kw), NAMES[1]: _fork_model(msa_encoder=True, **kw)},
                                 esmc=object(), device="cuda", _fast_kit_tokens=None)


def _all_models(app):
    return [("inversion_models", n, m) for n, m in app.inversion_models.items()] + [("hf_critic_models", n, m) for n, m in app.hf_critic_models.items()]


def test_model_handles_enumerate_every_loaded_model_never_a_by_name_merge():
    app = _app()
    handles = SD.model_handles(app)
    assert len(handles) == 4 and len({id(m) for _, m in handles}) == 4
    assert sorted(l for l, _ in handles) == sorted([f"hf_critic_models:{n}" for n in NAMES] + [f"inversion_models:{n}" for n in NAMES])
    shared = _fork_model(); app2 = types.SimpleNamespace(inversion_models={"A": shared}, hf_critic_models={"A": shared})
    assert SD.model_handles(app2) == [("hf_critic_models:A=inversion_models:A", shared)]                       # one object under two labels: once, both labels
    assert SD.model_handles(types.SimpleNamespace(esmc=object())) == []


def test_shipped_touches_nothing(capsys):
    app = _app()
    assert SD.apply_model_switches(app, "shipped", "shipped", "fast") == {}
    assert all(m.calls == [] for _, _, m in _all_models(app)) and "MODEL-SWITCH" not in capsys.readouterr().err


def test_the_stock_definition_reaches_every_loaded_model_and_is_read_back(capsys):
    """off and exact: EVERY model handle the design step constructs — both inversion models AND both critics, same names — receives
    set_chunk_size(None) and set_kernel_backend('cuequivariance'); every reached module then reads cuequivariance / _use_kernels True / chunk None;
    the MODEL-SWITCH lines come after the read-back and name the four models; the unreached msa_encoder of a critic is outside the setters' scope."""
    for arm in ("off", "exact"):
        app = _app()
        assert SD.apply_model_switches(app, None, "cuequivariance", arm) == {"chunk_size": None, "kernel_backend": "cuequivariance"}
        for attr, name, m in _all_models(app):
            assert m.calls == [("set_chunk_size", None), ("set_kernel_backend", "cuequivariance")], (attr, name, m.calls)
            for root in (m.folding_trunk, m.confidence_head.folding_trunk, m.structure_head):
                for path, mod in root.named_modules():
                    if "_kernel_backend" in vars(mod): assert mod._kernel_backend == "cuequivariance", (attr, name, path)
                    if "_use_kernels" in vars(mod): assert mod._use_kernels is True and mod._chunk_size is None, (attr, name, path)
        err = capsys.readouterr().err
        lines = [ln for ln in err.splitlines() if "MODEL-SWITCH" in ln]
        assert len(lines) == 2 and lines[0].startswith(f"[ef2inv-opt {arm}] MODEL-SWITCH chunk_size=None ") and lines[1].startswith(f"[ef2inv-opt {arm}] MODEL-SWITCH kernel_backend=cuequivariance ")
        for ln in lines:
            assert " models=4 names=hf_critic_models:ESMFold2-Experimental-Fast,hf_critic_models:ESMFold2-Experimental-Fast-Cutoff2025,inversion_models:ESMFold2-Experimental-Fast,inversion_models:ESMFold2-Experimental-Fast-Cutoff2025 readback=" in ln
        assert "NOT ACTIVE" not in err
    for v in ("fused", None):                                                                    # the other literals reach the same four models and read back
        app = _app(); assert SD.apply_model_switches(app, "shipped", v, "off") == {"kernel_backend": v}
        assert all(m.calls == [("set_kernel_backend", v)] for _, _, m in _all_models(app))
    app = _app(); assert SD.apply_model_switches(app, 32, "shipped", "off") == {"chunk_size": 32}
    assert all(m.calls == [("set_chunk_size", 32)] for _, _, m in _all_models(app))


def test_a_switch_that_did_not_take_effect_raises_by_name(capsys):
    """A model whose setter leaves one module on the old value: ModelSwitchError naming the model label, the module path, the attribute, got and
    expected — for the backend (_kernel_backend / _use_kernels) and for the chunk; nothing is printed as MODEL-SWITCH."""
    app = _app(); app.inversion_models[NAMES[1]] = _fork_model(broken="blocks.1.pair_transition")
    with pytest.raises(SD.ModelSwitchError) as e:
        SD.apply_model_switches(app, None, "cuequivariance", "off")
    msg = str(e.value)
    assert msg.startswith("model switch not applied: ") and f"inversion_models:{NAMES[1]} folding_trunk.blocks.1.pair_transition _kernel_backend='fused' expected='cuequivariance'" in msg
    assert f"inversion_models:{NAMES[1]} folding_trunk.blocks.1.pair_transition _chunk_size=64 expected=None" in msg and "MODEL-SWITCH" not in capsys.readouterr().err
    app = _app(); app.hf_critic_models[NAMES[0]] = _fork_model(broken="blocks.0.tri_mul_in._engine")
    with pytest.raises(SD.ModelSwitchError) as e:
        SD.apply_model_switches(app, "shipped", "cuequivariance", "exact")
    assert f"hf_critic_models:{NAMES[0]} folding_trunk.blocks.0.tri_mul_in._engine _use_kernels=False expected=True" in str(e.value)
    class _Opaque:                                                                               # a model carrying none of the fork's attributes: nothing to read back = refused, never trusted
        calls = []
        def set_kernel_backend(self, v): pass
        def set_chunk_size(self, v): pass
    with pytest.raises(SD.ModelSwitchError) as e:
        SD.apply_model_switches(types.SimpleNamespace(inversion_models={"x": _Opaque()}), None, "cuequivariance", "off")
    assert "inversion_models:x no module carries the fork's switch attributes" in str(e.value)
    with pytest.raises(SD.ModelSwitchError):
        SD.apply_model_switches(types.SimpleNamespace(esmc=object()), None, "cuequivariance", "off")


def test_fast_and_big_put_the_hero_critics_on_the_stock_arms_pair():
    """The mode table's critic scope: fast / big carry stock's none + cuequivariance for the HERO CRITICS only (their every-model switches stay
    shipped: the inversion models keep the cookbook's choices under the kit's kernels); off and exact carry no critic scope (off: stock as shipped;
    exact: its every-model pins already cover the critics). Mode.critic_switches resolves what the critics fold with, per key: an every-model value
    in force (exact's pins, the stock arm's flags, a user flag) wins with scope "all", else the critic value with scope "critics", else shipped."""
    for m in ("fast", "big"):
        assert (MD.MODES[m].critic_chunk_size, MD.MODES[m].critic_kernel_backend) == (MD.STOCK_CHUNK_SIZE, MD.STOCK_KERNEL_BACKEND) == (None, "cuequivariance")
        assert MD.CRITIC_SWITCH_WORDS in MD.MODES[m].levers and "hero critics on the stock arm's switches" in MD.MODES[m].what      # the ACTIVE line names it
        assert MD.MODES[m].critic_switches() == {"chunk_size": None, "kernel_backend": "cuequivariance", "scope": {"chunk_size": "critics", "kernel_backend": "critics"}}
        assert MD.MODES[m].critic_switches(32, "shipped") == {"chunk_size": 32, "kernel_backend": "cuequivariance", "scope": {"chunk_size": "all", "kernel_backend": "critics"}}   # a user --chunk-size covers the critics
        assert MD.MODES[m].critic_switches("shipped", "fused")["kernel_backend"] == "fused"                                        # a user --kernel-backend wins over the critic default, by name
    for m in ("off", "exact"):
        assert (MD.MODES[m].critic_chunk_size, MD.MODES[m].critic_kernel_backend) == ("shipped", "shipped")
    assert MD.MODES["off"].critic_switches() == {"chunk_size": "shipped", "kernel_backend": "shipped", "scope": {"chunk_size": "shipped", "kernel_backend": "shipped"}}
    ex = MD.MODES["exact"]
    assert ex.critic_switches(ex.chunk_size, ex.kernel_backend) == {"chunk_size": None, "kernel_backend": "cuequivariance", "scope": {"chunk_size": "all", "kernel_backend": "all"}}
    assert MD.CRITIC_MODELS_ATTR == "hf_critic_models"
    src = open(P.STOCK_FILE).read()
    assert "self.hf_critic_models: dict[str, Any] = {}" in src and "for name in self.hero_critic_hf_paths:" in src               # the cookbook's own container of the hero critics, loaded apart from the inversion models (l.1304-1308)


def test_the_critic_scope_reaches_the_hero_critics_only_and_is_read_back(capsys):
    """apply_model_switches(critic_chunk_size=, critic_kernel_backend=): the two setters on app.hf_critic_models only, read back module by module,
    one MODEL-SWITCH … scope=critics line each; the inversion models untouched; run.json model_switches["critics"] = the critics' effective pair +
    who set each key. An every-model value makes the critic value yield (scope=all on that key, no critic-scope call for it)."""
    app = _app()
    got = SD.apply_model_switches(app, "shipped", "shipped", "fast", critic_chunk_size=None, critic_kernel_backend="cuequivariance")
    assert got == {"critics": {"chunk_size": None, "kernel_backend": "cuequivariance", "scope": {"chunk_size": "critics", "kernel_backend": "critics"}}}
    for n, m in app.inversion_models.items():
        assert m.calls == [], n                                                                              # the inversion models: no call (the kit's enable sets their chunk; their backend stays the loader's)
    for n, m in app.hf_critic_models.items():
        assert m.calls == [("set_chunk_size", None), ("set_kernel_backend", "cuequivariance")], n
        assert all(vars(mod)["_kernel_backend"] == "cuequivariance" for _, mod in m.folding_trunk.named_modules() if "_kernel_backend" in vars(mod))
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 2 and all(" scope=critics" in ln and " models=2 names=hf_critic_models:" in ln and "on the hero critic models (the stock arm's setting" in ln for ln in err), err
    assert err[0].startswith("[ef2inv-opt fast] MODEL-SWITCH chunk_size=None ") and err[1].startswith("[ef2inv-opt fast] MODEL-SWITCH kernel_backend=cuequivariance ")
    # a user --chunk-size 32 on big: every model takes 32 (scope=all), the critics still get the backend from the critic scope
    app = _app()
    got = SD.apply_model_switches(app, 32, "shipped", "big", critic_chunk_size=None, critic_kernel_backend="cuequivariance")
    assert got == {"chunk_size": 32, "critics": {"chunk_size": 32, "kernel_backend": "cuequivariance", "scope": {"chunk_size": "all", "kernel_backend": "critics"}}}
    for n, m in app.inversion_models.items():
        assert m.calls == [("set_chunk_size", 32)], n
    for n, m in app.hf_critic_models.items():
        assert m.calls == [("set_chunk_size", 32), ("set_kernel_backend", "cuequivariance")], n
    err = capsys.readouterr().err
    assert err.count(" scope=all") == 1 and err.count(" scope=critics") == 1
    # exact's pins (every model): the critic scope has nothing left to do and records nothing of its own
    app = _app()
    assert SD.apply_model_switches(app, None, "cuequivariance", "exact", critic_chunk_size=None, critic_kernel_backend="cuequivariance") == {"chunk_size": None, "kernel_backend": "cuequivariance"}
    assert " scope=critics" not in capsys.readouterr().err
    # a critic whose setter skips a module is refused by name, as the every-model scope is
    app = _app(); app.hf_critic_models[NAMES[0]] = _fork_model(broken="pair_transition")
    with pytest.raises(SD.ModelSwitchError) as e:
        SD.apply_model_switches(app, "shipped", "shipped", "fast", critic_chunk_size=None, critic_kernel_backend="cuequivariance")
    assert f"hf_critic_models:{NAMES[0]} folding_trunk.blocks.0.pair_transition _kernel_backend='fused' expected='cuequivariance'" in str(e.value)
    with pytest.raises(SD.ModelSwitchError) as e:                                                          # no hero critic loaded: nothing to put on the stock pair — refused, never skipped
        SD.apply_model_switches(types.SimpleNamespace(inversion_models={"a": _fork_model()}), "shipped", "shipped", "big", critic_chunk_size=None, critic_kernel_backend="cuequivariance")
    assert "holds no hero critic model under hf_critic_models" in str(e.value)


def test_the_arm_exits_3_by_name_when_a_switch_does_not_take_effect(monkeypatch, capsys, tmp_path):
    """stock_design.main maps ModelSwitchError to `[ef2inv-opt <arm>] NOT ACTIVE: model switch not applied: …` and exit 3 (source-level contract:
    the only caller wraps apply_model_switches in that handler)."""
    import inspect
    src = inspect.getsource(SD.main)
    assert "apply_model_switches(app, a.chunk_size, a.kernel_backend, a.arm, critic_chunk_size=cmode.critic_chunk_size, critic_kernel_backend=cmode.critic_kernel_backend)" in src   # every model on the arm's values, then the hero critics on the mode's critic scope (cmode = the mode, or the mode without its critic scope when `critic_switches` is ablated)
    assert "except ModelSwitchError as e:" in src and 'NOT ACTIVE: {e}' in src and "return 3" in src.split("except ModelSwitchError as e:")[1][:200]


def test_effective_switches_the_modes_values_unless_the_user_passes_a_flag():
    """off: no call unless a flag names upstream's setter — a SETTING, never an override (no user record); exact: its own none + cuequivariance,
    restated pass silently, a differing flag is a user override every model takes exactly as the stock arm would (recorded, worded, never
    refused: the one exact lever with nothing to attach to steps aside by name at enable); fast / big: no call, a differing flag is a user override."""
    U = cli.UNSET
    assert cli.effective_switches(MD.MODES["off"]) == ({"chunk_size": "shipped", "kernel_backend": "shipped"}, {})                 # plain off = the cookbook as shipped
    assert cli.effective_switches(MD.MODES["off"], None, "cuequivariance") == ({"chunk_size": None, "kernel_backend": "cuequivariance"}, {})   # the stock arm's settings: not an override
    assert cli.effective_switches(MD.MODES["off"], 64, U) == ({"chunk_size": 64, "kernel_backend": "shipped"}, {})
    assert cli.effective_switches(MD.MODES["off"], U, None) == ({"chunk_size": "shipped", "kernel_backend": None}, {})
    assert cli.effective_switches(MD.MODES["exact"], U, U) == ({"chunk_size": None, "kernel_backend": "cuequivariance"}, {})
    assert cli.effective_switches(MD.MODES["exact"], None, "cuequivariance") == ({"chunk_size": None, "kernel_backend": "cuequivariance"}, {})   # exact's own pins restated: silent
    assert cli.effective_switches(MD.MODES["exact"], 64, U) == ({"chunk_size": 64, "kernel_backend": "cuequivariance"}, {"chunk_size": 64})            # exact accepts what stock accepts: the user's chunk on every model, its own backend kept
    assert cli.effective_switches(MD.MODES["exact"], U, "fused") == ({"chunk_size": None, "kernel_backend": "fused"}, {"kernel_backend": "fused"})
    assert cli.effective_switches(MD.MODES["exact"], U, None) == ({"chunk_size": None, "kernel_backend": None}, {"kernel_backend": None})
    assert cli.effective_switches(MD.MODES["exact"], 32, "cuequivariance") == ({"chunk_size": 32, "kernel_backend": "cuequivariance"}, {"chunk_size": 32})   # the backend restated is no override
    assert not hasattr(cli, "ModeContradiction") and S.user_switches(MD.MODES["exact"], 64, "shipped") == {"chunk_size": 64} and S.user_switches(MD.MODES["off"], 64, None) == {}
    what, words = S.exact_switch_note("kernel_backend", None)
    assert what == "--mode exact with --kernel-backend None" and "LEVER name=cueq_tiles state=stepped_aside reason=user_kernel_backend:None" in words
    what, words = S.exact_switch_note("chunk_size", 64)
    assert what == "--mode exact with --chunk-size 64" and "set_chunk_size(64) on every loaded model" in words and "plan_row=<row>(unchunked-priced)" in words
    assert cli.effective_switches(MD.MODES["fast"]) == ({"chunk_size": "shipped", "kernel_backend": "shipped"}, {})
    assert cli.effective_switches(MD.MODES["fast"], None, "cuequivariance") == ({"chunk_size": None, "kernel_backend": "cuequivariance"}, {"chunk_size": None, "kernel_backend": "cuequivariance"})
    assert cli.effective_switches(MD.MODES["big"], 64, U) == ({"chunk_size": 64, "kernel_backend": "shipped"}, {"chunk_size": 64})


def test_argv_carries_the_stock_definition_on_off_and_exact_and_nothing_on_fast():
    """The arm argv: the stock arm (`--mode off --chunk-size none --kernel-backend cuequivariance`) and exact carry `--chunk-size none
    --kernel-backend cuequivariance`; plain off (the cookbook as shipped), fast and big carry no setter token."""
    kw = dict(cookbook_stock="/x/binder_design.py", target_name="c", target_sequence="ACDEFGHIK", binder_len=80, seed=0, out="/o")
    stock_eff, _ = cli.effective_switches(MD.MODES["off"], None, "cuequivariance"); exact_eff, _ = cli.effective_switches(MD.MODES["exact"])
    for m, eff in (("off", stock_eff), ("exact", exact_eff)):
        argv = L.argv_for(MD.MODES[m], P.ROOT, chunk_size=eff["chunk_size"], kernel_backend=eff["kernel_backend"], **kw)
        assert argv[argv.index("--chunk-size") + 1] == "none" and argv[argv.index("--kernel-backend") + 1] == "cuequivariance" and "--tag" not in argv and argv[argv.index("--target-name") + 1] == "c"
    for m in ("off", "fast", "big"):
        eff, _ = cli.effective_switches(MD.MODES[m])
        argv = L.argv_for(MD.MODES[m], P.ROOT, chunk_size=eff["chunk_size"], kernel_backend=eff["kernel_backend"], **kw)
        assert "--chunk-size" not in argv and "--kernel-backend" not in argv and argv == L.argv_for(MD.MODES[m], P.ROOT, **kw)    # no setter token
    sw = L.argv_for(MD.MODES["off"], P.ROOT, chunk_size=64, kernel_backend=None, **kw)
    assert sw[sw.index("--chunk-size") + 1] == "64" and sw[sw.index("--kernel-backend") + 1] == "None"


def _sd_parse(extra):
    import argparse
    seen = {}
    real = argparse.ArgumentParser.parse_args
    def grab(self, argv=None, namespace=None):
        ns = real(self, argv, namespace); seen["ns"] = ns; raise SystemExit(0)
    argparse.ArgumentParser.parse_args = grab
    try:
        with pytest.raises(SystemExit):
            SD.main(["--mode", "off", "--cookbook", "x", "--pins", "p", "--target-name", "c", "--target-sequence", "ACDE", "--binder-len", "80", "--seed", "0", "--out", "o"] + extra)
    finally:
        argparse.ArgumentParser.parse_args = real
    return seen.get("ns")


def test_the_arm_reads_back_the_api_values():
    for v in S.KERNEL_BACKENDS:
        ns = _sd_parse(["--kernel-backend", S.kernel_backend_word(v), "--chunk-size", "none"])
        assert ns.kernel_backend == v and ns.chunk_size is None
    ns = _sd_parse([]); assert ns is not None and not hasattr(ns, "chunk_size") and not hasattr(ns, "kernel_backend")   # NO token (plain off / fast / big argv) PARSES (no string default for argparse to type-check) …
    SD.absent_setters(ns); assert ns.chunk_size == "shipped" and ns.kernel_backend == "shipped"                      # … and main's fill names it: no setter call
    ns = SD.absent_setters(_sd_parse(["--chunk-size", "64"])); assert ns.chunk_size == 64 and ns.kernel_backend == "shipped"
    import inspect; assert "absent_setters(ap.parse_args(argv))" in inspect.getsource(SD.main)
    for bad in (["--kernel-backend", "cueq"], ["--chunk-size", "shipped"], ["--kernel-backend", "shipped"]):   # `shipped` is no argv value any more
        with pytest.raises(SystemExit) as e:
            SD.main(["--mode", "off", "--cookbook", "x", "--pins", "p", "--target-name", "c", "--target-sequence", "ACDE", "--binder-len", "80", "--seed", "0", "--out", "o", *bad])
        assert e.value.code == 2


def test_cli_refuses_an_unknown_backend_word_by_name(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["design", "--mode", "off", "--target-name", "t", "--target-sequence", "ACDEFGHIK", "--binder-len", "80", "--seed", "0", "--out", "o", "--kernel-backend", "torch"])
    err = capsys.readouterr().err
    assert e.value.code == 2 and "'torch'" in err and "fused | cuequivariance | None" in err
    with pytest.raises(SystemExit) as e:
        cli.main(["design", "--mode", "off", "--target-name", "t", "--target-sequence", "ACDEFGHIK", "--binder-len", "80", "--seed", "0", "--out", "o", "--chunk-size", "half"])
    assert e.value.code == 2 and "'half'" in capsys.readouterr().err
    for flag in ("--chunk-size", "--kernel-backend"):                                           # the retired pseudo-value: a usage error naming it
        with pytest.raises(SystemExit) as e:
            cli.main(["design", "--mode", "off", "--target-name", "t", "--target-sequence", "ACDEFGHIK", "--binder-len", "80", "--seed", "0", "--out", "o", flag, "shipped"])
        assert e.value.code == 2 and "'shipped'" in capsys.readouterr().err


def _fake_torch(name):
    m = types.ModuleType("torch"); m.__version__ = "2.11.0+cu128"
    m.cuda = types.SimpleNamespace(is_available=lambda: True, get_device_name=lambda i=0: name); m.version = types.SimpleNamespace(cuda="12.8")
    return m


def _design_rig(monkeypatch, tmp_path):
    import sys
    monkeypatch.setenv("MODEL_OPT", cli._model_opt())
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout=json.dumps({"ok": True, "facts": {"esm_version": "3.4.0"}}), stderr="", returncode=0))
    monkeypatch.setattr(cli.MD, "jit_cache_key", lambda: "torch2.11.0-cu128-sm90")
    monkeypatch.setattr(cli, "_cookbook_stock", lambda: P.STOCK_FILE)
    monkeypatch.setattr(cli, "CORE_FACTS", {"pinned": "stub", "installed": "stub"})
    runs, writes = [], []
    monkeypatch.setattr(cli.L, "run", lambda argv, env, log, timeout=None: runs.append((argv, env)) or {"exit_code": 0, "wall_s": 1.0})
    monkeypatch.setattr(cli.M, "write", lambda *a, **k: writes.append(k) or {"missing_outputs": [], "refusals": [], "evidence": "stock"})
    tgt = tmp_path / "target.fasta"; tgt.write_text(">t\nACDEFGHIK\n")
    def ns(**kw):
        d = dict(mode="off", target_name="t", target_sequence="ACDEFGHIK", binder_len=80, seed=0, out=str(tmp_path / "d"), target_hotspot_ids=None, det=0,
                 allow_partial=False, chunk_size=cli.UNSET, kernel_backend=cli.UNSET); d.update(kw)
        return types.SimpleNamespace(**d)
    def card(name, mib):
        monkeypatch.setitem(sys.modules, "torch", _fake_torch(name))
        monkeypatch.setattr(cli.HW, "card", lambda probe=None: (name, mib, f"{name}, {mib}"))
    return ns, card, runs, writes


def _notes(capsys):
    return [ln for ln in capsys.readouterr().err.splitlines() if ln.startswith("[ef2inv-opt] NOTE ")]


def test_design_off_runs_the_stock_definition_silently_and_alike_on_any_card(monkeypatch, tmp_path, capsys):
    """The stock arm `design --mode off --chunk-size none --kernel-backend cuequivariance`: the argv carries chunk none + cuequivariance, NO NOTE
    about them (a setting on off, not an override; the card NOTE only off the pinned card), the manifest gets the effective switches and no user
    override; the same argv and env on the pinned card and on another card. Plain `--mode off` (the cookbook as shipped) and fast carry no token."""
    ns, card, runs, writes = _design_rig(monkeypatch, tmp_path)
    notes = []
    for c in (("NVIDIA H100 80GB HBM3", 81559), ("NVIDIA A100-SXM4-80GB", 81920)):
        card(*c); assert cli.cmd_design(ns(chunk_size=None, kernel_backend="cuequivariance")) == 0; notes.append(_notes(capsys))
    assert notes[0] == [] and len(notes[1]) == 1 and "hardware" in notes[1][0]
    assert runs[0] == runs[1] and runs[0][0][runs[0][0].index("--chunk-size") + 1] == "none" and runs[0][0][runs[0][0].index("--kernel-backend") + 1] == "cuequivariance"
    assert writes[0]["switches"] == {"chunk_size": None, "kernel_backend": "cuequivariance"} and writes[0]["user_overrides"] == {}
    for kw in ({}, {"mode": "fast"}):                                                            # plain off = the cookbook as shipped; fast = its own composition: no setter token, no NOTE
        runs.clear(); writes.clear(); card("NVIDIA H100 80GB HBM3", 81559)
        assert cli.cmd_design(ns(**kw)) == 0 and _notes(capsys) == []
        assert "--chunk-size" not in runs[0][0] and "--kernel-backend" not in runs[0][0] and writes[0]["switches"] == {"chunk_size": "shipped", "kernel_backend": "shipped"} and writes[0]["user_overrides"] == {}


def test_a_user_backend_override_notes_once_and_is_recorded(monkeypatch, tmp_path, capsys):
    """On a kit mode (fast) a flag differing from the mode's value is a user override: ONE NOTE for the backend, the tokens on the argv, recorded
    under user_overrides. On off the same flags are SETTINGS: applied, silent, no user record. Under exact a differing flag is a user override
    too — the tokens on the argv, recorded, ONE NOTE per switch of what the exact lever set does with it (plus the backend's grad-facts NOTE), the
    launch reached, never refused; exact's own values restated pass silently."""
    ns, card, runs, writes = _design_rig(monkeypatch, tmp_path); card("NVIDIA H100 80GB HBM3", 81559)
    assert cli.cmd_design(ns(mode="fast", kernel_backend="fused", chunk_size=64)) == 0
    notes = _notes(capsys)
    assert len(notes) == 1 and notes[0].startswith("[ef2inv-opt] NOTE --kernel-backend fused — applies to the models' no-grad folds") and notes[0].endswith("; proceeding")
    assert runs[0][0][runs[0][0].index("--kernel-backend") + 1] == "fused" and runs[0][0][runs[0][0].index("--chunk-size") + 1] == "64"
    assert writes[0]["user_overrides"] == {"kernel_backend": "fused", "chunk_size": 64} and writes[0]["switches"] == {"chunk_size": 64, "kernel_backend": "fused"}
    runs.clear(); writes.clear()
    assert cli.cmd_design(ns(mode="fast", kernel_backend=None)) == 0
    notes = _notes(capsys); assert len(notes) == 1 and notes[0].startswith("[ef2inv-opt] NOTE --kernel-backend None — the reference modules in every fold")
    assert runs[0][0][runs[0][0].index("--kernel-backend") + 1] == "None" and writes[0]["user_overrides"] == {"kernel_backend": None}
    runs.clear(); writes.clear()
    assert cli.cmd_design(ns(kernel_backend="fused", chunk_size=64)) == 0 and _notes(capsys) == []            # off: settings, silent, no user record
    assert runs[0][0][runs[0][0].index("--kernel-backend") + 1] == "fused" and writes[0]["user_overrides"] == {} and writes[0]["switches"] == {"chunk_size": 64, "kernel_backend": "fused"}
    runs.clear(); writes.clear(); capsys.readouterr()
    assert cli.cmd_design(ns(mode="exact", chunk_size=64)) == 0                                  # exact + the user's chunk: launched with the user's value on the argv, exact's own backend beside it
    notes = _notes(capsys)
    assert len(notes) == 1 and notes[0].startswith("[ef2inv-opt] NOTE --mode exact with --chunk-size 64 — stock's set_chunk_size(64) on every loaded model") and notes[0].endswith("; proceeding") and "REFUSED" not in notes[0]
    assert runs[0][0][runs[0][0].index("--chunk-size") + 1] == "64" and runs[0][0][runs[0][0].index("--kernel-backend") + 1] == "cuequivariance"
    assert writes[0]["user_overrides"] == {"chunk_size": 64} and writes[0]["switches"] == {"chunk_size": 64, "kernel_backend": "cuequivariance"}
    for kb, word in ((None, "None"), ("fused", "fused")):                                        # exact + the user's backend: the grad-facts NOTE and exact's NOTE (the tile table steps aside by name), launched
        runs.clear(); writes.clear()
        assert cli.cmd_design(ns(mode="exact", kernel_backend=kb)) == 0
        notes = _notes(capsys)
        assert len(notes) == 2 and notes[0].startswith(f"[ef2inv-opt] NOTE --kernel-backend {word} — ") and notes[1].startswith(f"[ef2inv-opt] NOTE --mode exact with --kernel-backend {word} — stock's set_kernel_backend({word}) on every loaded model"), notes
        assert f"LEVER name=cueq_tiles state=stepped_aside reason=user_kernel_backend:{word}" in notes[1] and all("REFUSED" not in n for n in notes)
        assert runs[0][0][runs[0][0].index("--kernel-backend") + 1] == word and runs[0][0][runs[0][0].index("--chunk-size") + 1] == "none" and writes[0]["user_overrides"] == {"kernel_backend": kb}
    runs.clear(); writes.clear()
    assert cli.cmd_design(ns(mode="exact", chunk_size=None, kernel_backend="cuequivariance")) == 0 and _notes(capsys) == [] and writes[0]["user_overrides"] == {}   # exact's own values restated: silent


ARM_LINES = {                                                                                   #  a fixture of five design command lines: the stock arm S = off + upstream's two
    "stock": "--mode off --chunk-size none --kernel-backend cuequivariance",                     #  documented speed settings; `default` = the cookbook exactly as shipped (plain
    "default": "--mode off",                                                                     #  off, no setter call); exact / fast / big = `run.sh design --mode <m>`;
    "exact": "--mode exact",                                                                     #  retyped here — the kit tree reads no other tree
    "fast": "--mode fast",
    "big": "--mode big",
}
ARM_LINES_BEFORE = {"stock": "--mode off", "default": "--mode off --chunk-size shipped --kernel-backend shipped"}   # the two lines' earlier spellings: the arm argv each produces now must be the argv its earlier spelling produced


def test_every_arm_line_reaches_its_launch_through_the_cli_and_default_notes_the_shipped_backend(monkeypatch, tmp_path, capsys):
    """Each line through `cli.main` (argparse types = settings.parse_*, then cmd_design): exit 0 and one launch, NO NOTE on any line; stock and
    exact carry `--chunk-size none --kernel-backend cuequivariance` on the arm argv; default (plain off), fast and big carry no setter token.
    BYTE-IDENTITY with the earlier spellings: the stock line's arm argv, launch switches and manifest switch records equal those the old `--mode off`
    produced (chunk None + cuequivariance as off's own values, no user override); the default line's arm argv and run.json model_switches ({})
    equal the old `--mode off --chunk-size shipped --kernel-backend shipped`'s (which additionally recorded the two `shipped` words as user
    overrides + one NOTE — plain off has nothing to override). The retired `shipped` word is a usage error (exit 2)."""
    ns, card, runs, writes = _design_rig(monkeypatch, tmp_path); card("NVIDIA H100 80GB HBM3", 81559)
    got = {}
    for arm, line in ARM_LINES.items():
        runs.clear(); writes.clear()
        assert cli.main(["design", *line.split(), "--target-name", "t", "--target-sequence", "ACDEFGHIK", "--binder-len", "80", "--seed", "0", "--out", str(tmp_path / f"d_{arm}")]) == 0, arm
        notes = _notes(capsys); argv = runs[0][0]; got[arm] = (argv, dict(writes[0]))
        assert len(runs) == 1 and len(writes) == 1 and notes == [], arm
        if arm in ("stock", "exact"):
            assert argv[argv.index("--chunk-size") + 1] == "none" and argv[argv.index("--kernel-backend") + 1] == "cuequivariance", arm
            assert writes[0]["switches"] == {"chunk_size": None, "kernel_backend": "cuequivariance"} and writes[0]["user_overrides"] == {}, arm
        else:
            assert "--chunk-size" not in argv and "--kernel-backend" not in argv, arm
            assert argv[argv.index("--mode") + 1] == ("off" if arm == "default" else MD.MODES[arm].name) and (arm != "default" or argv[argv.index("--cookbook") + 1] == P.STOCK_FILE)   # the arm process is given the resolved mode (`fast` -> big)
            assert writes[0]["switches"] == {"chunk_size": "shipped", "kernel_backend": "shipped"} and writes[0]["user_overrides"] == {}, arm
    # the stock arm's records = what the old `--mode off` (off's own values None + cuequivariance, no user flag) produced, word for word
    eff, overrides, words = M.switch_records(MD.MODES["off"], {"model_switches": {"chunk_size": None, "kernel_backend": "cuequivariance"}, "overrides": {}}, got["stock"][1]["switches"], got["stock"][1]["user_overrides"])
    assert eff == {"chunk_size": None, "kernel_backend": "cuequivariance"} and overrides is None
    assert words == ["model switch chunk_size=None (the off mode's value; stock = chunk None + cuequivariance backend)", "model switch kernel_backend='cuequivariance' (the off mode's value; stock = chunk None + cuequivariance backend)"]
    s_argv = got["stock"][0]; d_argv = got["default"][0]
    assert [t.replace("d_stock", "d_default") for t in s_argv if t not in ("--chunk-size", "none", "--kernel-backend", "cuequivariance")] == d_argv   # the two off lines differ by exactly the two setter tokens (and the out dir)
    eff, overrides, words = M.switch_records(MD.MODES["off"], {"model_switches": {}, "overrides": {}}, got["default"][1]["switches"], got["default"][1]["user_overrides"])
    assert eff == {"chunk_size": "shipped", "kernel_backend": "shipped"} and overrides is None and words == []
    for arm, line in ARM_LINES_BEFORE.items():                                                   # the earlier spellings: `--mode off` alone is now the default arm; `shipped` is refused at parse
        if "shipped" in line:
            with pytest.raises(SystemExit) as e:
                cli.main(["design", *line.split(), "--target-name", "t", "--target-sequence", "ACDEFGHIK", "--binder-len", "80", "--seed", "0", "--out", str(tmp_path / f"b_{arm}")])
            assert e.value.code == 2 and "'shipped'" in capsys.readouterr().err


def test_manifest_records_the_effective_switches_and_user_overrides_only(tmp_path):
    d = tmp_path / "design"; d.mkdir()
    run = {"settings": {"deviations": ["binder_len: fixed 80"]}, "patches": [], "det": {"level": 0}, "overrides": {}, "model_switches": {"chunk_size": None, "kernel_backend": "cuequivariance"}, "launch": {}}
    (d / "run.json").write_text(json.dumps(run))
    man = M.write(str(d), MD.MODES["off"], P.ROOT, P.PINS, P.STOCK_FILE, {"exit_code": 0}, "torch2.11.0-cu128-sm90", switches={"chunk_size": None, "kernel_backend": "cuequivariance"}, user_overrides={})
    assert man["model_switches"] == {"chunk_size": None, "kernel_backend": "cuequivariance"} and man["overrides"] is None                     # the stock arm: settings recorded, no override
    assert any(x.startswith("model switch chunk_size=None (the off mode's value") for x in man["deviations"]) and any(x.startswith("model switch kernel_backend='cuequivariance' (the off mode's value") for x in man["deviations"])
    on_disk = json.load(open(d / "opt_manifest.json")); assert on_disk["model_switches"] == {"chunk_size": None, "kernel_backend": "cuequivariance"} and on_disk["overrides"] is None
    run["model_switches"] = {"chunk_size": 64, "kernel_backend": "fused"}; run["overrides"] = {}; (d / "run.json").write_text(json.dumps(run))
    man = M.write(str(d), MD.MODES["fast"], P.ROOT, P.PINS, P.STOCK_FILE, {"exit_code": 0}, "torch2.11.0-cu128-sm90", switches={"chunk_size": 64, "kernel_backend": "fused"}, user_overrides={"chunk_size": 64, "kernel_backend": "fused"})
    assert man["model_switches"] == {"chunk_size": 64, "kernel_backend": "fused"} and man["overrides"] == {"chunk_size": 64, "kernel_backend": "fused"}     # a kit mode's user override
    assert any(x.startswith("override kernel_backend='fused' (user flag; the mode's value is 'shipped')") for x in man["deviations"])
    d2 = tmp_path / "fast"; d2.mkdir(); (d2 / "run.json").write_text(json.dumps({"settings": {"deviations": []}, "patches": [], "det": {"level": 0}, "overrides": {}, "model_switches": {}}))
    for m in ("fast", "off"):                                                                     # fast, and plain off: no setter called, nothing to record but `shipped`
        man = M.write(str(d2), MD.MODES[m], P.ROOT, P.PINS, P.STOCK_FILE, {"exit_code": 0}, "torch2.11.0-cu128-sm90", switches={"chunk_size": "shipped", "kernel_backend": "shipped"}, user_overrides={})
        assert man["model_switches"] == {"chunk_size": "shipped", "kernel_backend": "shipped"} and man["overrides"] is None and not any("model switch" in x or x.startswith("override") for x in man["deviations"])


def test_warm_carries_the_modes_switches_too(monkeypatch, tmp_path, capsys):
    """`warm` takes the same two setters: `warm --mode off --chunk-size none --kernel-backend cuequivariance` launches the arm with the stock arm's
    settings (stock_design applies and reads them back before the warm folds); plain `warm --mode off` and `warm --mode fast` carry none;
    `warm --mode exact --chunk-size 64` carries the user's chunk with exact's NOTE, as design does."""
    runs = []
    monkeypatch.setenv("MODEL_OPT", cli._model_opt())
    monkeypatch.setattr(cli.HW, "card", lambda probe=None: ("NVIDIA H100 80GB HBM3", 81559, "NVIDIA H100 80GB HBM3, 81559"))
    monkeypatch.setattr(cli.L, "run", lambda argv, env, log, timeout=None: runs.append(argv) or {"exit_code": 0})
    monkeypatch.setattr(cli, "_cookbook_stock", lambda: P.STOCK_FILE)
    tgt = tmp_path / "target.fasta"; tgt.write_text(">t\nACDEFGHIK\n")
    w = lambda m, **kw: types.SimpleNamespace(**dict(dict(mode=m, target_name="t", target_sequence="ACDEFGHIK", binder_len=80, out=str(tmp_path / f"warm_{m}_{len(runs)}")), **kw))
    assert cli.cmd_warm(w("off", chunk_size=None, kernel_backend="cuequivariance")) == 0 and cli.cmd_warm(w("off")) == 0 and cli.cmd_warm(w("fast")) == 0 and cli.cmd_warm(w("exact")) == 0
    assert runs[0][runs[0].index("--chunk-size") + 1] == "none" and runs[0][runs[0].index("--kernel-backend") + 1] == "cuequivariance" and runs[0][-1] == "--warm-only"
    assert "--chunk-size" not in runs[1] and "--kernel-backend" not in runs[1] and runs[1][-1] == "--warm-only"
    assert "--chunk-size" not in runs[2] and "--kernel-backend" not in runs[2] and runs[2][-1] == "--warm-only"
    assert runs[3][runs[3].index("--chunk-size") + 1] == "none" and runs[3][runs[3].index("--kernel-backend") + 1] == "cuequivariance"        # exact's pins ride the argv
    capsys.readouterr()
    assert cli.cmd_warm(w("exact", chunk_size=64)) == 0 and len(runs) == 5 and runs[4][runs[4].index("--chunk-size") + 1] == "64" and runs[4][runs[4].index("--kernel-backend") + 1] == "cuequivariance"
    err = capsys.readouterr().err
    assert "NOTE --mode exact with --chunk-size 64 — stock's set_chunk_size(64) on every loaded model" in err and "REFUSED" not in err


def test_manifest_deviations_over_the_stock_arms_patch_records(tmp_path):
    """The stock arm's three patch records as stock_design writes them (the bug-0002 fix, the design kit not installed, the RoPE pin)
    format into opt_manifest.json deviations; stock_patch names the bug-0002 record only."""
    import json as _json
    from .. import det as D, fastkit as FK, patches as PT
    d = tmp_path / "off"; d.mkdir()
    patches = [{"name": "upstream_bug_0002_transition_addmm_dtype", "applied": True, "source": PT.FIX_SOURCE if hasattr(PT, "FIX_SOURCE") else "port", "function_sha256": "ab" * 32, "target": "T"},
               FK.record_off(),
               PT.pin_esmc_rope(module=type("M", (), {PT.ESMC_ROPE_FLAG: True})())]
    (d / "run.json").write_text(_json.dumps({"exit_code": 0, "patches": patches, "det": {"level": 1, "applied": D.LEVEL_WHAT[1]}, "attention": {"words": {"esmc_rope": "torch"}}}))
    man = M.write(str(d), MD.MODES["off"], P.ROOT, P.PINS, P.STOCK_FILE, {"exit_code": 0}, "torch2.11.0-cu128-sm90", switches={"chunk_size": None, "kernel_backend": "cuequivariance"}, user_overrides={})
    devs = man["deviations"]
    assert any(x.startswith("patch upstream_bug_0002_transition_addmm_dtype: applied=True") for x in devs) and any(x.startswith(f"patch {FK.NAME}: applied=False (") for x in devs)
    assert any(x.startswith("patch esmc_rope_pin: applied=True impl=torch (") for x in devs)
    assert man["stock_patch"] == "upstream_bug_0002" and man["attention"] == {"words": {"esmc_rope": "torch"}}
