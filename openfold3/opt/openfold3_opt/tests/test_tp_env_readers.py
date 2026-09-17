"""The tp line's switch surface is READ, name by name: every ``OF3TP_*`` switch the line exports (``modes.LINES[("big", "tp")].env``) reaches
the name its statement reads — a core lever through ``tp_rowpair.env.ENV_MAP`` (``export_core_env`` writes the ``ROWPAIR_*`` name, and the
line's constants ``env.LINE_CONSTANTS``); every ``OF3TP_*`` literal the rank-side modules spell is in the table; the of3tp add-on's switches with no
tp-line statement (``env.NOT_CARRIED``) are refused by name with their reason. A switch that is exported or accepted but read by nothing (a silent
no-op) fails here.

Also the LayerNorm launch guard (``model.install_ln_guard``): with ``ROWPAIR_LN_GUARD_ELEMS`` forced small on CPU,
OpenFold3's ``LayerNorm`` primitive and ``torch.nn.LayerNorm`` run through the core's ``shard.ln_rows_guarded`` in row blocks and return the
whole launch's bytes (``torch.equal``) on shard-shaped operands (the aspect classes of the add-on's ``OF3TP_LN2p32_shardshape_test.py``:
``[rows, N, 128|64]`` pair / template shards, whose B200 log holds row blocks bitwise-equal to the whole launch below 2^32 elements and wrong rows
beyond), on the MSA representation ``[1, S, N, c]`` and a 2-D operand; the census counts the launches it split. Needs torch + openfold3 for the
guard items (skipped by name otherwise); the table items need neither."""
import os
import re

import pytest

from openfold3_opt import modes
from openfold3_opt.tp_rowpair import env

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)                                                                  # opt/openfold3_opt
CORE = os.path.normpath(os.path.join(PKG, "..", "..", "..", "common", "opt_core"))               # the release tree's core beside the kit (opt/pyproject.toml [tool.opt_core] path)
TP_LINE_ENV = dict(modes.LINES[("big", "tp")].env)
RANK_SIDE = sorted([os.path.join(PKG, "tp_rowpair", f) for f in os.listdir(os.path.join(PKG, "tp_rowpair")) if f.endswith(".py")]
                   + [os.path.join(PKG, "tp_rowpair", "hook", "sitecustomize.py"), os.path.join(PKG, "sample_loop.py"), os.path.join(PKG, "tp.py"), os.path.join(PKG, "cli.py")])
READ_FORMS = re.compile(r"environ|getenv|\.get\(|_flag01\(|env_int\(|env_flag\(|^\s*[A-Z_]*ENV[A-Z_]*\s*[:=]", re.M)


def _text(rel):
    with open(os.path.join(PKG, rel), encoding="utf-8") as fh:
        return fh.read()


def test_line_exports_reach_their_readers():
    """Every OF3TP_* word of the tp line's env is a mapped size parameter and lands on its reader's name; the line's constants are written beside
    them: ROWPAIR_FREE_ZTRUNK=1 / ROWPAIR_CONF_PARK_ZTRUNK=1 (the names ``tp_rowpair.confidence`` and the core's ``heads`` read) among them."""
    e = {k: v for k, v in TP_LINE_ENV.items() if k.startswith("OF3TP_")}
    assert e, "the tp line exports no OF3TP_* word"
    written = env.export_core_env(dict(e))
    for k, v in e.items():
        assert k in env.ENV_MAP, f"{k}: exported by the tp line, not in tp_rowpair/env.py ENV_MAP"
        assert written.get(env.ENV_MAP[k]) == v.strip(), (k, env.ENV_MAP[k], written.get(env.ENV_MAP[k]))
    for k, v in env.LINE_CONSTANTS.items():
        assert written[k] == v, k
    assert written["ROWPAIR_FREE_ZTRUNK"] == "1" and written["ROWPAIR_CONF_PARK_ZTRUNK"] == "1"
    assert env.export_core_env({"ROWPAIR_CONF_PARK_ZTRUNK": "0"})["ROWPAIR_CONF_PARK_ZTRUNK"] == "1"   # a constant of the line: an inherited value does not select another placement
    core_heads = open(os.path.join(CORE, "opt_core", "mem", "rowpair", "heads.py"), encoding="utf-8").read()   # the reader of the two names: the pinned core's heads.ZTrunkPlan
    assert 'ENV_FREE_ZTRUNK = "ROWPAIR_FREE_ZTRUNK"' in core_heads and 'ENV_CONF_PARK_ZTRUNK = "ROWPAIR_CONF_PARK_ZTRUNK"' in core_heads
    assert "ZTrunkPlan" in _text("tp_rowpair/confidence.py")


def test_line_constants_are_core_names_no_alias_maps_onto():
    assert env.LINE_CONSTANTS and all(k.startswith("ROWPAIR_") for k in env.LINE_CONSTANTS)
    assert not set(env.LINE_CONSTANTS) & set(env.ENV_MAP.values()), set(env.LINE_CONSTANTS) & set(env.ENV_MAP.values())


def test_every_rank_side_literal_is_in_a_table():
    """An OF3TP_* literal in a rank-side module that no table lists is either an undeclared switch (check_known would refuse a user setting it)
    or a dead read of a refused name."""
    bad = []
    for path in RANK_SIDE:
        if os.path.basename(path) == "env.py" and os.path.dirname(path).endswith("tp_rowpair"):
            continue
        with open(path, encoding="utf-8") as fh:
            for name in sorted(set(re.findall(r'"(OF3TP_[A-Z0-9_]+)"', fh.read()))):
                if name in env.NOT_CARRIED:
                    bad.append(f"{os.path.relpath(path, PKG)} reads {name}, which env.NOT_CARRIED refuses (a dead read)")
                elif not env.known(name):
                    bad.append(f"{os.path.relpath(path, PKG)} reads {name}, which tp_rowpair/env.py ENV_MAP does not list")
    assert not bad, "\n".join(bad)


def test_tables_are_disjoint_and_core_names_are_rowpair():
    assert not set(env.ENV_MAP) & set(env.NOT_CARRIED), set(env.ENV_MAP) & set(env.NOT_CARRIED)
    assert all(v.startswith("ROWPAIR_") for v in env.ENV_MAP.values())
    for k in ("OF3TP_LN_GUARD_ELEMS", "OF3TP_TEMPL_NODEDUPE", "OF3TP_PARK_STRICT", "OF3TP_TRIATT_QBLOCK", "OF3TP_TRIMUL_SUB"):
        assert env.ENV_MAP[k] == "ROWPAIR_" + k[len("OF3TP_"):], k
    for gone in ("OF3TP_MODE", "OF3TP_MSA_HOST", "OF3TP_CONF_PARK_ZTRUNK", "OF3TP_SAMPLER_HOOK", "OF3TP_CKPT_DIR", "OF3TP_SAMPLE_CHUNK", "OF3TP_TRIATT_KERNEL"):
        assert not env.known(gone), gone                                                    # the line's constants have no switch


def test_not_carried_and_unknown_names_are_refused_by_name():
    for name, why in env.NOT_CARRIED.items():
        with pytest.raises(env.UnknownSwitch) as ei:
            env.check_known({name: "1"})
        assert name in str(ei.value) and why[:40] in str(ei.value)
    with pytest.raises(env.UnknownSwitch, match="OF3TP_NOT_A_SWITCH"):
        env.export_core_env({"OF3TP_NOT_A_SWITCH": "1"})
    assert env.check_known({"OF3TP_HOST_FEATS": "", "OF3TP_CHUNK": "64"}) == ["OF3TP_CHUNK"]                       # an empty value is unset; a known name passes


# ----------------------------------------------------------------------------------------------------------------- the guard itself (torch + openfold3)
try:
    import torch
    from openfold3.core.model.primitives.normalization import LayerNorm as OF3LayerNorm   # noqa: F401
    import opt_core.mem.rowpair.shard  # noqa: F401
    HAVE_STACK = True
except Exception:                                                                            # noqa: BLE001
    HAVE_STACK = False
needs_stack = pytest.mark.skipif(not HAVE_STACK, reason="needs torch + openfold3 + opt_core (the LayerNorm guard runs OpenFold3's LayerNorm primitive)")

SHARD_SHAPES = [(61, 487, 128), (61, 487, 64), (34, 273, 128), (34, 273, 64), (30, 243, 128), (15, 122, 128)]   # [rows, N, C]: OF3TP_LN2p32_shardshape_test.py's aspect classes at 1/64 scale
OTHER_SHAPES = [(1, 40, 97, 64), (1, 1, 33, 211, 64), (777, 48), (5, 9, 32)]                                   # m [1, S, N, c], a slot slab [1, 1, R, N, c_t], a 2-D operand, a small 3-D one


@needs_stack
@pytest.mark.parametrize("limit", ["4096", "0"])
def test_ln_guard_forced_row_blocks_are_the_whole_launch_bytes(monkeypatch, limit):
    from opt_core.mem.patchset import PatchSet
    from openfold3.core.model.primitives.normalization import LayerNorm as OF3LayerNorm
    from openfold3_opt.tp_rowpair import model as M
    monkeypatch.setenv("ROWPAIR_LN_GUARD_ELEMS", limit)
    g = torch.Generator().manual_seed(0)
    cases = []
    for shape in SHARD_SHAPES + OTHER_SHAPES:
        C = shape[-1]
        for mod in (OF3LayerNorm(C), torch.nn.LayerNorm(C)):
            with torch.no_grad():
                for p in mod.parameters():
                    p.copy_(torch.randn(p.shape, generator=g))
            x = torch.randn(*shape, generator=g)
            with torch.no_grad():
                ref = mod(x)                                                                 # the whole launch, stock forward
            cases.append((mod, x, ref))
    patches = PatchSet("tp_test_ln_guard")
    monkeypatch.setattr(M, "PATCHES", patches)
    monkeypatch.setattr(M, "LN_GUARD", {"state": "not_installed", "launches": 0, "split": 0, "unsplittable": 0, "shapes": []})
    try:
        state = M.install_ln_guard()
        assert state.startswith("installed:") and set(patches.names()) == {"LayerNorm.forward"} | ({"LayerNorm.forward"}) and len(patches) == 2
        for mod, x, ref in cases:
            with torch.no_grad():
                got = mod(x)
            assert got.shape == ref.shape and torch.equal(got, ref), (type(mod).__name__, tuple(x.shape), float((got - ref).abs().max()))
        fields = M.census_fields(None)
        assert fields["ln_guard"] == state and fields["ln_guard_launches"] == len(cases)
        big = sum(1 for _, x, _ in cases if int(limit) and x.numel() >= int(limit))
        assert fields["ln_guard_split"] == big and (big > 0) == (limit != "0"), fields
    finally:
        patches.restore()
    with torch.no_grad():                                                                     # restored: the stock forwards again
        for mod, x, ref in cases[:2]:
            assert torch.equal(mod(x), ref)


@needs_stack
def test_ln_guard_sites_forced_row_blocks(monkeypatch):
    """The two adapter sites' call form (``ln_rows_guarded(module, x, rows_dim=-3)``) on their operand shapes, forced to block: whole-launch bytes."""
    from openfold3.core.model.primitives.normalization import LayerNorm as OF3LayerNorm
    from opt_core.mem.rowpair.shard import ln_rows_guarded
    monkeypatch.setenv("ROWPAIR_LN_GUARD_ELEMS", "2048")
    g = torch.Generator().manual_seed(1)
    for shape in ((1, 40, 97, 64), (1, 33, 211, 64)):                                          # m [1, S, N, c_m]; u4 [1, n_loc, N, c_t]
        ln = OF3LayerNorm(shape[-1])
        calls = [0]
        stock = ln.forward

        def counted(x, _f=stock):
            calls[0] += 1
            return _f(x)

        x = torch.randn(*shape, generator=g)
        with torch.no_grad():
            ref = stock(x)
            got = ln_rows_guarded(counted, x, rows_dim=-3)
        assert calls[0] > 1 and torch.equal(got, ref), (shape, calls[0])
