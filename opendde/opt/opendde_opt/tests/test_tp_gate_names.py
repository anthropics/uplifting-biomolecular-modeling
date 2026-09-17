"""`--mode big --n_gpu P` never switches to a replicated / faster path by size (R-TP-3): every gate the adapter and its line own is FORCED to
the sharded value here, grep-ably. The gate table lives in the kit README's TP section; this file is its executable form. No torch needed.

No statement of the `--n_gpu P>1` path (or of the modules that arm it) references a name its module does not define: every function and
class body of the listed modules is compiled to a symbol table and each IMPLICIT-GLOBAL reference must resolve in the imported module's
namespace or in builtins — a dropped function-local import (a `torch.` statement in a wrapper that only a GPU rank executes) fails HERE, on
CPU, without running the wrapper. pyflakes, when installed, must agree (zero 'undefined name')."""
import builtins
import importlib
import symtable

import pytest

from opendde_opt import modes, tp


def test_big_tp_line_carries_no_offload_or_xl_unit():
    """Under `--n_gpu P>1` every pair track — pair init, trunk, template, MSA module, structural stage, diffusion conditioning, heads — is
    ROW-SHARDED; the offload unit's host-streamed stages and the XL unit are alternatives to row sharding on ONE card, not layers, and the line
    carries neither: no ODDE_OFFLOAD export, none of the unit's registry rows planned, no CUDA-graph switch."""
    line = tp.register_line()
    assert line.name == modes.BIG_TP_LINE == "BIG_TP"
    assert modes.BIG_TP_LINE not in modes.BIG_OFFLOAD_STAGES and modes.BIG_TP_LINE not in modes.BIG_OFFLOAD_LINES
    assert "ODDE_OFFLOAD" not in line.exports and not any(k.startswith("ODDE_OFFLOAD") for k in line.exports)
    assert line.exports["TORCH_NCCL_AVOID_RECORD_STREAMS"] == "1" == modes.TP_EXPORTS["TORCH_NCCL_AVOID_RECORD_STREAMS"]   # collective operands reusable at once in the ranks
    assert line.exports["ROWPAIR_TRANSPOSE_INPLACE"] == "1" == modes.TP_EXPORTS["ROWPAIR_TRANSPOSE_INPLACE"]   # the ending transpose inside the shard's storage
    assert line.exports[tp.ENV_STRUCT_PAIR_DTYPE] == "bf16" and modes.TP_STRUCT_BF16_LEVER in line.levers   # struct_pair_bf16: on by default on the line, named on ACTIVE (levers + exports)
    assert line.exports[tp.ENV_PARK_ZINIT] == "1"                                # z_init parked on the host during the trunk (core ShardPark); =0 by a caller keeps it on the card (named)
    assert line.exports[tp.ENV_PARK_ZRES] == "1"                                 # the residue z shard parked during the structural stage + roll-out (RowShard.park, core ShardPark)
    assert line.exports[tp.ENV_STRUCT_TRIMUL_RB] == "128"                         # the structural refiner's streamed tri-mult slabs: 128 rows, P-invariant (never budgeted from free bytes)
    assert line.exports["ROWPAIR_NCCL_TIMEOUT_S"] == "600"                       # fail-fast: the core's collective timeout for the ranks (a dead peer never holds the others 30 min)
    assert not (set(line.levers) & set(modes._OFFLOAD_LEVERS)) and not (set(line.levers) & set(modes._XL_LEVERS)), line.levers
    assert modes.TP_LEVER in line.levers and "no_dit_hoist" in line.levers and "sample_chunk" in line.levers
    assert line.path_order == modes.KIT_PATH_ORDER
    assert line.tier == "tier2"                                                # named by guarantee: never 'exact'


def test_a_pair_stack_call_shape_the_adapter_does_not_shard_REFUSES_by_default(monkeypatch):
    from opt_core.mem.rowpair import RowpairRefused
    monkeypatch.delenv(tp.ENV_ALLOW_UNSHARDED, raising=False)
    before = tp.STATS["unsharded"]
    with pytest.raises(RowpairRefused, match=tp.ENV_ALLOW_UNSHARDED):
        tp._note_unsharded("pair tensor (2, 8, 8, 4): a leading batch > 1")
    assert tp.STATS["unsharded"] == before                                     # nothing ran unsharded
    monkeypatch.setenv(tp.ENV_ALLOW_UNSHARDED, "1")                            # the opt-in: a NAMED event + census count, never silent
    tp._note_unsharded("pair tensor (2, 8, 8, 4): a leading batch > 1")
    assert tp.STATS["unsharded"] == before + 1
    tp.STATS["unsharded"] = before
    tp.STATS["unsharded_reasons"].clear()


def test_adapter_gates_default_to_the_sharded_statement(monkeypatch):
    for k in (tp.ENV_MSA_TRANS_SHARD, tp.ENV_CONF_FINISH, tp.ENV_ALLOW_UNSHARDED):
        monkeypatch.delenv(k, raising=False)
    assert tp._msa_trans_shard() is True                                       # MSA transition token-sharded (M = S·R + one all-gather), not the whole-m statement per rank
    monkeypatch.delenv("ODDE_SERVED_DETERMINISTIC", raising=False)
    assert tp._noise_sync() == "bcast"                                         # det 0: replicated draws / per-call diffusion state = rank 0's, broadcast (identical by construction)
    monkeypatch.setenv("ODDE_SERVED_DETERMINISTIC", "1")
    assert tp._noise_sync() == "guard"                                         # det 1: strict bitwise guard (a mismatch is a defect, refused by name)
    monkeypatch.delenv("ODDE_SERVED_DETERMINISTIC", raising=False)
    assert tp._m_layout() == "token_sharded"                                   # m [S, N, c_m] token-sharded by default (this rank's token block); replicated = the named whole-m form
    monkeypatch.setenv(tp.ENV_MSA_M_LAYOUT, "replicated")
    assert tp._m_layout() == "replicated"
    monkeypatch.delenv(tp.ENV_MSA_M_LAYOUT)
    assert tp._conf_finish() == "exact"                                        # the confidence reducer's finish: exact (never the [N, N, bins] all-gather; 'rowsum' = reordered, opt-in)
    monkeypatch.setenv(tp.ENV_MSA_TRANS_SHARD, "2")
    from opt_core.mem.rowpair import RowpairRefused
    with pytest.raises(RowpairRefused, match=tp.ENV_MSA_TRANS_SHARD):          # malformed values refuse by name (no silent default)
        tp._msa_trans_shard()
    monkeypatch.setenv(tp.ENV_CONF_FINISH, "allgather")
    with pytest.raises(RowpairRefused, match=tp.ENV_CONF_FINISH):
        tp._conf_finish()
    monkeypatch.setenv(tp.ENV_STRUCT_PAIR_DTYPE, "fp16")
    with pytest.raises(RowpairRefused, match=tp.ENV_STRUCT_PAIR_DTYPE):           # only bf16 | fp32
        tp._struct_pair_dtype()
    monkeypatch.delenv(tp.ENV_STRUCT_PAIR_DTYPE)
    assert tp._struct_pair_dtype() == "fp32"                                     # unset (n_gpu=1, or a caller's plain env): fp32


def test_gate_names_are_grep_able_module_constants():
    """The gate table's names are the adapter's constants (the table greps clean against the source)."""
    assert (tp.ENV_ALLOW_UNSHARDED, tp.ENV_MSA_TRANS_SHARD, tp.ENV_CONF_FINISH, tp.ENV_CONF_ROWS, tp.ENV_PWA_SCHUNK, tp.ENV_RING_ROWS, tp.ENV_ATTN_ROWS, tp.ENV_LAYOUT_B, tp.ENV_MSA_M_LAYOUT, tp.ENV_NOISE_SYNC) == (
        "ROWPAIR_ALLOW_UNSHARDED", "ROWPAIR_MSA_TRANS_SHARD", "ROWPAIR_CONF_FINISH", "ROWPAIR_CONF_ROWS", "ROWPAIR_PWA_SCHUNK", "ROWPAIR_HOSTGATHER_ROWS", "ROWPAIR_ATTN_ROWS",
        "ROWPAIR_LAYOUT_B", "ROWPAIR_MSA_M_LAYOUT", "ROWPAIR_DIFF_NOISE_SYNC")


def test_kit_schedule_census_carries_every_word(monkeypatch):
    """`_record_schedule` fills STATS['schedule'] with exactly KIT_SCHEDULE_KEYS (a word swallowed by an edit fails here by name); every
    per-rank-derived block size the adapter owns (layout_B, pwa_schunk, hostgather_rows, conf_rows) is among them with its source."""
    from opt_core.mem.rowpair import dist as _dist
    for k in (tp.ENV_LAYOUT_B, tp.ENV_PWA_SCHUNK, tp.ENV_RING_ROWS, tp.ENV_CONF_ROWS, tp.ENV_NOISE_SYNC, tp.ENV_MSA_M_LAYOUT, tp.ENV_MSA_TRANS_SHARD, tp.ENV_CONF_FINISH):
        monkeypatch.delenv(k, raising=False)
    lay = _dist.Layout.auto(256, 2, 0, lever=tp.LEVER)
    tp._record_schedule(lay)
    got = tp.STATS["schedule"]
    assert tuple(got) == tp.KIT_SCHEDULE_KEYS, (sorted(set(tp.KIT_SCHEDULE_KEYS) ^ set(got)))
    for k in ("layout_B", "layout_B_source", "pwa_schunk", "pwa_schunk_source", "hostgather_rows", "hostgather_rows_source", "conf_rows", "conf_rows_source",
              "msa_m", "msa_transition", "conf_logits", "diff_noise", "contact_probs", "struct_stage", "dit_queries", "atom_band", "trunk_init", "input_feats", "inputs", "diffz", "struct_pair_dtype", "zinit", "recycle", "template", "zres", "template_rows_after_trunk", "struct_trimul_rb", "nccl_timeout_s", "transpose_inplace"):
        assert k in got, k
    assert got["conf_rows"] == 128 and got["conf_rows_source"] == "fixed" and got["layout_B"] == 128 and got["diff_noise"].startswith("bcast_rank0_state")


def test_struct_trimul_rb_is_explicit_and_refuses_by_name(monkeypatch):
    """The structural refiner's tri-mult slab rows: ODDE_TP_STRUCT_TRIMUL_RB (multiple of 8) reaches the core as trimul_update_'s own RB= kwarg
    through pairstack.bind's trimul_kw; 0 = the core's budgeted RB; malformed values refuse by name."""
    from opt_core.mem.rowpair import RowpairRefused
    monkeypatch.setenv(tp.ENV_STRUCT_TRIMUL_RB, "128")
    assert tp._struct_trimul_rb() == 128
    for bad in ("12x", "4", "100"):
        monkeypatch.setenv(tp.ENV_STRUCT_TRIMUL_RB, bad)
        with pytest.raises(RowpairRefused, match=tp.ENV_STRUCT_TRIMUL_RB):
            tp._struct_trimul_rb()
    monkeypatch.setenv(tp.ENV_STRUCT_TRIMUL_RB, "0")
    assert tp._struct_trimul_rb() is None


def test_refiner_bind_carries_rb(monkeypatch):
    """_pair_fns(trimul_rb=N) hands RB=N to the core's tri-mult driver (PairBlockFns trimul_kw); without it the kwarg is absent (budgeted)."""
    import types
    from opt_core.mem.rowpair import pairstack as _ps
    seen = {}
    monkeypatch.setattr(_ps, "bind", lambda **kw: seen.setdefault("kw", kw) or types.SimpleNamespace(**kw))
    monkeypatch.setattr(tp, "_trimul_fns", lambda m: None); monkeypatch.setattr(tp, "_triatt_fns", lambda *a, **k: None)
    monkeypatch.setattr(tp, "_trimul_inplace_chunk", lambda m: 64); monkeypatch.setattr(tp, "_attn_rows", lambda c, lay: 32)
    blk = types.SimpleNamespace(training=False, c_s=0, tri_mul_out=None, tri_mul_in=None, tri_att_start=None, tri_att_end=None, pair_transition=lambda x: x)
    tp._pair_fns(blk, None, triangle_attention="torch", trimul_rb=128)
    assert seen["kw"]["trimul_kw"] == {"inplace_chunk": 64, "RB": 128}
    seen.clear()
    tp._pair_fns(blk, None, triangle_attention="torch")
    assert seen["kw"]["trimul_kw"] == {"inplace_chunk": 64}


MODULES = ("opendde_opt.tp", "opendde_opt.tp_feats", "opendde_opt.tp_struct", "opendde_opt.tp_diffusion", "opendde_opt.modes", "opendde_opt.registry", "opendde_opt.ran",
           "opendde_opt.report", "opendde_opt.stack", "opendde_opt.cli", "opendde_opt.big", "opendde_opt.settings")


def _implicit_global_refs(table, out):
    for child in table.get_children():
        for sym in child.get_symbols():
            if sym.is_referenced() and sym.is_global() and not sym.is_declared_global() and not sym.is_assigned() and not sym.is_imported():
                out.setdefault(sym.get_name(), []).append(f"{child.get_type()} {child.get_name()} (line {child.get_lineno()})")
            elif sym.is_referenced() and sym.is_declared_global():
                out.setdefault(sym.get_name(), []).append(f"{child.get_type()} {child.get_name()} (line {child.get_lineno()}, declared global)")
        _implicit_global_refs(child, out)
    return out


@pytest.mark.parametrize("modname", MODULES)
def test_every_global_reference_resolves(modname):
    pytest.importorskip("torch")                                               # the modules import torch lazily; the kit's CPU image has it
    mod = importlib.import_module(modname)
    src = open(mod.__file__, encoding="utf-8").read()
    refs = _implicit_global_refs(symtable.symtable(src, mod.__file__, "exec"), {})
    known = set(vars(mod)) | set(dir(builtins))
    missing = {name: sites for name, sites in refs.items() if name not in known}
    assert not missing, f"{modname}: names referenced in a function/class body but defined nowhere in the module: {missing}"


def test_pyflakes_reports_no_undefined_name():
    api = pytest.importorskip("pyflakes.api", reason="pyflakes is not installed on this image: the symtable scan above is the check as shipped")
    from pyflakes import reporter
    import io
    bad = []
    for modname in MODULES:
        mod = importlib.import_module(modname)
        out, err = io.StringIO(), io.StringIO()
        api.checkPath(mod.__file__, reporter.Reporter(out, err))
        bad += [l for l in out.getvalue().splitlines() if "undefined name" in l]
    assert not bad, bad
