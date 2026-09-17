"""MODEL_OPT_LEVERS_OFF — the ablation door (modes.apply_levers_off): every exact / fast lever leaves its line singly (graphs_strict only with
cuda_graphs, by name), its switches unset and required unset, the pair cells by value, the dtype levers by their fp32 word, a hook nothing arms any more with it; unknown or stuck
names are refused by name listing the line's levers; unset or empty is the line as written byte for byte; mode off ignores it; the ACTIVE /
DRY-RUN / exit lines and the LEVER lines name what was left off."""
import os

import pytest

from openfold3_ob0_opt import modes, report, stack
from openfold3_ob0_opt.registry import LEVERS
from openfold3_ob0_opt.tests import _stubs

HOME = _stubs.tree_home()
OFF = modes.ENV_LEVERS_OFF
CASES = {("exact", 400): {}, ("exact", 612): {}, ("fast", 612): {}, ("big", 2565): {}, ("big", 1012): {}}


def _same(a, b):
    return (a.exports, a.unsets, a.hooks, a.levers, a.notes, a.conflicts, a.levers_off, modes.describe_line(a)) == \
           (b.exports, b.unsets, b.hooks, b.levers, b.notes, b.conflicts, b.levers_off, modes.describe_line(b))


def test_unset_or_empty_is_the_line_as_written_byte_for_byte():
    for (mode, n) in CASES:
        base = modes.resolve(mode, HOME, environ={}, n_tokens=n)
        assert base.levers_off == [] and not any(OFF in x for x in base.notes)
        for raw in ("", " ", ",", " , "):
            assert _same(base, modes.resolve(mode, HOME, environ={OFF: raw}, n_tokens=n)), (mode, n, raw)
    assert modes.levers_off_names({OFF: " a, b ,,a "}) == ("a", "b") and modes.levers_off_names({}) == ()


@pytest.mark.parametrize("mode,line,n_tokens", [("exact", "cueq", 400), ("fast", None, 612)])
def test_every_lever_of_the_line_leaves_singly(mode, line, n_tokens):
    ln = modes.LINES[(mode, line)]
    base = modes.resolve(mode, HOME, environ={}, n_tokens=n_tokens)
    for name in ln.levers:
        r = modes.resolve(mode, HOME, environ={OFF: name}, n_tokens=n_tokens)
        if name in modes.LEVERS_OFF_STUCK:                                                   # graphs_strict: by name, with the reason and the lever to switch instead
            assert r.conflicts and f"lever {name} of the" in r.conflicts[0] and "cuda_graphs" in r.conflicts[0] and r.levers_off == [], name
            continue
        assert r.conflicts == [], (name, r.conflicts)
        gone = [name] + [t for t in modes.LEVERS_OFF_TAKES.get(name, ()) if t in ln.levers]
        assert r.levers_off == [l for l in ln.levers if l in gone], (name, r.levers_off)
        assert [l for l in r.levers if l in gone] == [] and [l for l in base.levers if l not in gone and l not in r.levers] == [], name
        assert len(r.notes) == 1 and r.notes[0].startswith(f"{OFF}: {','.join(r.levers_off)} left off"), r.notes
        if name in modes.DTYPE_LEVER_WORDS:                                                 # a dtype lever leaves by its word: upstream's fp32, source=mode (conf_dtype takes z_dtype along)
            assert all(r.exports[modes.DTYPE_LEVER_WORDS[g]] == modes.DTYPE_OFF_WORD for g in gone) and set(base.exports) == set(r.exports), name
            assert {k: v for k, v in base.exports.items() if k not in modes.DTYPE_KNOB_ENVS} == {k: v for k, v in r.exports.items() if k not in modes.DTYPE_KNOB_ENVS}, name
            assert (r.conf_dtype, r.z_dtype) == (("conf_dtype=fp32 source=mode" if "conf_dtype" in gone else base.conf_dtype), "z_dtype=fp32 source=mode"), (name, r.conf_dtype, r.z_dtype)
            continue
        if name in modes.PAIR_CELL_LEVERS:                                                  # a pair cell leaves OPENFOLD3_OB0_OPT_PAIR's value; the family stays for the others
            left = [p for p in modes.PAIR_CELL_LEVERS if p in ln.levers and p != name]
            taken = {k for t in gone[1:] for k in modes.LEVER_SWITCHES[t]}                    # triatt_block takes triatt_provider along: the block's core word leaves with it
            assert r.exports[modes.PAIR_ENVS[0]] == ":".join(left) and set(base.exports) - taken == set(r.exports), name
            assert all(k in r.unsets for k in taken), (name, taken)
            continue
        keys = [k for g in gone for k in modes.LEVER_SWITCHES[g]]
        assert keys and all(k not in r.exports for k in keys), (name, keys)
        assert all((k in r.unsets) != (k in modes.LEVERS_OFF_KEPT_SETTABLE) for k in keys if k in base.exports or k in modes.LEVER_SWITCHES[name]), (name, r.unsets)
        assert {k: v for k, v in base.exports.items() if k not in keys} == {k: v for k, v in r.exports.items() if k not in keys}, name   # nothing else moves
        for k in keys:                                                                      # the env-override refusal stays: a caller cannot re-arm the lever behind the mode
            if k in modes.LEVERS_OFF_KEPT_SETTABLE:
                continue
            c = modes.resolve(mode, HOME, environ={OFF: name, k: base.exports.get(k, "1")}, n_tokens=n_tokens).conflicts
            assert any(k in x for x in c), (name, k, c)


def test_the_last_pair_cell_takes_the_family_and_all_cells_take_the_hook():
    r = modes.resolve("fast", HOME, environ={OFF: "trimul_v4,triatt_block,pair_transition"}, n_tokens=612)
    assert r.conflicts == [] and all(k not in r.exports for k in modes.PAIR_ENVS) and modes.PAIR_ENVS[0] in r.unsets
    assert r.hooks == ["cells", "trunk_kernels", "fast_inference"]                                       # other cells still arm the hook
    cells = [l for l in modes.LINES[("exact", "cueq")].levers if (modes.LEVER_SWITCHES.get(l) or ("",))[0].startswith("OPENFOLD3_OB0_OPT_")]
    assert cells == ["atom_hoist", "castcache", "exactln", "apb_hoist", "triatt_exact", "trimul_exact", "transition_exact", "post_release", "postfwd_mem", "loader_workers", "fastjson", "writer_overlap", "hostfeat", "ckpt_mmap"]
    hook_cells = [l for l in cells if modes.LEVER_SWITCHES[l] not in modes.ACTIVATION_FAMILIES]                # the activation-installed lever (fastjson) arms no hook: the six hook-installed cells off already free the cells hook
    r = modes.resolve("exact", HOME, environ={OFF: ",".join(hook_cells)}, n_tokens=400)
    assert r.conflicts == [] and r.hooks == ["trunk_kernels", "fast_inference"] and "fastjson" in r.levers and r.exports.get("OPENFOLD3_OB0_OPT_FASTJSON") == "1"
    r = modes.resolve("exact", HOME, environ={OFF: ",".join(cells)}, n_tokens=400)
    assert r.conflicts == [] and r.hooks == ["trunk_kernels", "fast_inference"] and os.path.basename(os.path.dirname(r.entry_hook)) == "of3t_hook"
    assert modes.PAIRFUSED_CHAIN_ENV not in r.exports and r.exports[modes.KIT_LEVERS_ENV] == modes.hook_dir(HOME, "fast_inference")
    assert modes.describe_line(r).split(" @ ")[1] == "of3t_hook(trunk_kernels)>of3_levers(fast_inference)"
    r = modes.resolve("fast", HOME, environ={OFF: "fast_init,cuda_graphs"}, n_tokens=612)                 # nothing arms of3_levers any more: it leaves the chain, unnamed by OF3T_KIT_LEVERS
    assert r.conflicts == [] and r.hooks == ["cells", "trunk_kernels"] and modes.KIT_LEVERS_ENV not in r.exports
    assert r.levers_off == ["fast_init", "cuda_graphs", "graphs_strict"] and "graphs_strict taken along" in r.notes[0]
    assert r.size_gate is None and not modes.graphed_call("fast", None, 612, {OFF: "cuda_graphs"}) and modes.graphed_call("fast", None, 612, {})
    assert not modes.graphed_call("exact", "cueq", 400, {OFF: "cuda_graphs"}) and modes.graphed_call("exact", "cueq", 400, {})


def test_unknown_and_stuck_names_are_refused_by_name_with_the_lines_levers():
    r = modes.resolve("exact", HOME, environ={OFF: "trunk_graph"}, n_tokens=400)                          # a fast lever, not the exact line's
    assert len(r.conflicts) == 1 and r.conflicts[0].startswith(f"{OFF}='trunk_graph' names trunk_graph: not a lever of the exact/cueq line (its levers: fast_init,")
    assert r.levers_off == [] and r.exports == modes.resolve("exact", HOME, environ={}, n_tokens=400).exports   # nothing rewritten on a refusal
    r = modes.resolve("fast", HOME, environ={OFF: "atom_window,nope,castcache,zzz"}, n_tokens=612)
    assert len(r.conflicts) == 1 and "names nope,zzz: not a lever of the fast line" in r.conflicts[0]
    rep = stack.activate("fast", dry_run=True, home=HOME, environ={OFF: "nope", "PATH": os.environ.get("PATH", "")}, log=False)
    assert rep["active"] is False and "names nope: not a lever of the fast line" in rep["reason"], rep.get("reason")
    rank = {"OF3TP_RANK": "0", "OF3TP_WORLD": "2"}
    for name in ("tp_shard_s", "sample_loop", "tp_triatt", "tp_trimul"):
        c = modes.resolve("big", HOME, environ={**rank, OFF: name}, n_tokens=2565, n_gpu="2").conflicts
        assert len(c) == 1 and f"lever {name} of the big/tp line cannot be switched off on its own" in c[0], c
    r = modes.resolve("big", HOME, environ={**rank, OFF: "fast_init"}, n_tokens=2565, n_gpu="2")           # the tp line's one switchable lever: of3_levers leaves the chain with it
    assert r.conflicts == [] and r.levers_off == ["fast_init"] and r.hooks == ["cells", "tp_rowpair"] and "OF3_FAST_INIT" not in r.exports
    for name in ("trimul_hostsnap", "templ_host", "host_pool", "bigln_guard"):
        c = modes.resolve("big", HOME, environ={OFF: name}, n_tokens=2565).conflicts
        assert len(c) == 1 and "the big/resident line is applied as composed" in c[0], c
    r = modes.resolve("big", HOME, environ={OFF: "conf_chunked"}, n_tokens=2565)                           # the scorer takes the row-block head along; the port runs its stock scorer
    assert r.conflicts == [] and r.levers_off == ["conf_chunked", "confhead"] and r.hooks == ["offload", "cells", "trunk_kernels", "fast_inference"]   # the port's hook leads without the head's
    assert r.exports["OF3O_CONF_MODE"] == "stock" and "OF3O_CONF_TM_BACKEND" not in r.exports and modes.ENV_CONFHEAD not in r.exports and r.conf_gate is None
    r = modes.resolve("big", HOME, environ={OFF: "confhead,alloc_expandable"}, n_tokens=2565)
    assert r.conflicts == [] and r.hooks == ["offload", "cells", "trunk_kernels", "fast_inference"] and "PYTORCH_CUDA_ALLOC_CONF" not in r.exports and "PYTORCH_CUDA_ALLOC_CONF" not in r.unsets
    assert r.exports["OF3O_CONF_MODE"] == "chunked" and r.conf_gate and "confhead" not in r.levers and "conf_chunked" in r.levers


def test_mode_off_ignores_it_with_a_note():
    r = modes.resolve("off", HOME, environ={OFF: "cuda_graphs"})
    assert r.conflicts == [] and r.levers_off == [] and r.notes == [modes.levers_off_ignored({OFF: "cuda_graphs"})] and "ignored: mode off runs stock" in r.notes[0]
    rep = stack.activate("off", dry_run=True, home=HOME, environ={OFF: "cuda_graphs", "PATH": os.environ.get("PATH", "")}, log=False)
    assert rep["mode"] == "off" and any("ignored: mode off runs stock" in n for n in rep["notes"])
    assert modes.levers_off_ignored({}) is None


def test_the_lines_name_what_was_left_off():
    res = modes.resolve("fast", HOME, environ={OFF: "atom_window,castcache"}, n_tokens=612)
    rep = {"active": True, "mode": "fast", "line": None, "line_spelling": modes.describe_line(res), "levers_requested": list(res.levers), "levers_off": list(res.levers_off),
           "levers_applied": [], "levers_unavailable": [], "levers_pending": list(res.levers), "hooks": list(res.hooks), "notes": list(res.notes), "n_gpu": 1}
    line = report.activation_line(rep)
    assert " levers_off=atom_window,castcache" in line and "OPENFOLD3_OB0_OPT_ATOM_WINDOW" not in line and "OPENFOLD3_OB0_OPT_CASTCACHE" not in line
    assert line.index(" levers_off=") > line.index(" hooks=")                                              # after the fields the activation tables read
    assert " levers_off=atom_window,castcache" in report.exit_tally_line(rep).splitlines()[0]
    got = {l.split("name=")[1].split(" ")[0]: l for l in report.lever_lines(rep)}
    assert " state=off reason=levers_off " in got["atom_window"] + " " and " state=off reason=levers_off" in got["castcache"]
    assert "reason=levers_off" not in got["trunk_graph"] and " state=off reason=not_in_line" in got["confhead"]
    assert report.levers_off_field({}) == "" and " levers_off=" not in report.activation_line(dict(rep, levers_off=[]))
    drep = dict(rep, active=False, dry_run=True, env=dict(res.exports))
    assert " levers_off=atom_window,castcache" in report.activation_line(drep)
    assert all(name in LEVERS for name in modes.LEVER_SWITCHES) and all(name in LEVERS for name in modes.LEVERS_OFF_STUCK)


def test_a_process_the_run_starts_inherits_what_was_left_off():
    r = modes.resolve("fast", HOME, environ={OFF: "atom_window,conf_dtype"}, n_tokens=612)
    assert r.levers_off == ["atom_window", "conf_dtype", "z_dtype"] and r.conflicts == []
    child = modes.resolve("fast", HOME, environ={modes.ENV_RESOLVED: modes.freeze(r), OFF: "nope"}, n_tokens=None)   # the variable is read once, in the primary process
    assert child.inherited and child.levers_off == r.levers_off and child.exports == r.exports and child.conflicts == []
