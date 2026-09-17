"""`boltz predict`'s own options — recycling_steps, sampling_steps, diffusion_samples, max_parallel_samples, step_scale, write_full_pae,
write_full_pde, output_format, num_workers, override, seed, no_kernels — are inputs of `pred` on every mode with upstream's names, semantics
and defaults (stock/PINS.json cli_defaults = boltz/main.py). Not given = not passed to stock / upstream's default on the worker. No presets."""
import json
import os

import pytest

from .. import manifest as mf
from .. import cli, modes, report as rep, settings, stack, worker
from . import _stubs


def test_write_batch_carries_the_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(stack, "cache_dir", lambda: "/cache")
    items = worker.items_from_yamls(_stubs.write_yamls(str(tmp_path), ("a",)))
    b = json.load(open(stack.write_batch(str(tmp_path), "t", items, [0], str(tmp_path / "o"))))
    assert b["predict"] == {"recycling_steps": 3, "sampling_steps": 200, "diffusion_samples": 1, "max_parallel_samples": 5} and b["write_full_pae"] is False and b["step_scale"] == 1.5, "nothing given: upstream's defaults"
    eff = settings.worker_settings("exact", {"recycling_steps": 10, "diffusion_samples": 5, "write_full_pae": True, "write_full_pde": True, "output_format": "pdb", "step_scale": 1.638})
    b = json.load(open(stack.write_batch(str(tmp_path), "u", items, [0], str(tmp_path / "o"), settings=eff)))
    assert b["predict"] == {"recycling_steps": 10, "sampling_steps": 200, "diffusion_samples": 5, "max_parallel_samples": 5}
    assert (b["write_full_pae"], b["write_full_pde"], b["output_format"], b["step_scale"]) == (True, True, "pdb", 1.638)


def test_the_worker_route_takes_the_options_into_the_batch_the_active_line_and_the_manifest(tmp_path, monkeypatch, capsys):
    _stubs.gated_ok(monkeypatch); _stubs.install_fake_stage(monkeypatch)
    ys = _stubs.write_yamls(str(tmp_path), ("a",))
    rc = worker.run("fast", ys, str(tmp_path / "o"), [7], settings={"recycling_steps": 10, "diffusion_samples": 5, "write_full_pae": True})
    text = capsys.readouterr().out; assert rc == 0, text
    b = json.load(open([os.path.join(dp, f) for dp, _, fn in os.walk(tmp_path / "o" / "_kit") for f in fn if f.startswith("batch_")][0]))
    assert b["predict"] == {"recycling_steps": 10, "sampling_steps": 200, "diffusion_samples": 5, "max_parallel_samples": 5} and b["seeds"] == [7] and b["write_full_pae"] is True
    active = [l for l in text.splitlines() if l.startswith("[boltz2-opt] ACTIVE ")][0]
    assert active.endswith(" recycling_steps=10 diffusion_samples=5 write_full_pae=1"), active            # only the options away from upstream's defaults are named
    m = mf.LAST
    assert {k: m["settings"][k] for k in ("recycling_steps", "sampling_steps", "diffusion_samples", "max_parallel_samples", "write_full_pae", "output_format", "kernels", "seeds", "given")} == \
           {"recycling_steps": 10, "sampling_steps": 200, "diffusion_samples": 5, "max_parallel_samples": 5, "write_full_pae": True, "output_format": "mmcif", "kernels": "on", "seeds": [7], "given": ["recycling_steps", "diffusion_samples", "write_full_pae"]}
    census = m["command"][m["command"].index("--kernels-settings") + 1]
    assert census == "flags" and m["report"]["settings"]["recycling_steps"] == 10
    rep.reset_tally()
    worker.run("fast", ys, str(tmp_path / "o2"), [0])                                                    # nothing given: the ACTIVE line carries no settings words (the default line)
    active = [l for l in capsys.readouterr().out.splitlines() if l.startswith("[boltz2-opt] ACTIVE ")][0]
    assert "recycling_steps" not in active and "preset" not in active, active
    m = mf.LAST
    assert m["command"][m["command"].index("--kernels-settings") + 1] == "defaults" and m["settings"]["given"] == [] and m["settings"]["write_full_pae"] is False
    rep.reset_tally()
    rc = worker.run("fast", ys, str(tmp_path / "o_no_kernels"), [0], settings={"no_kernels": True})    # the one option the worker line cannot serve: refused BY NAME, nothing staged
    out = capsys.readouterr().out
    assert rc == rep.EXIT_USAGE and "NOT ACTIVE: settings: not served on the worker route (--mode fast): --no_kernels: " in out and not os.path.isdir(tmp_path / "o_no_kernels" / "_kit"), out
    rc = worker.run("fast", ys, str(tmp_path / "o4"), [0], settings={"sampling_steps": 0})
    assert rc == rep.EXIT_USAGE and "--sampling_steps 0 is below 1" in capsys.readouterr().out


def test_cli_plumbs_seed_and_the_options_on_the_worker_route(tmp_path, monkeypatch, capsys):
    _stubs.gated_ok(monkeypatch); _stubs.install_fake_stage(monkeypatch)
    calls = []
    real = worker.run
    monkeypatch.setattr(worker, "run", lambda *a, **k: calls.append((a, k)) or real(*a, **k))
    y = _stubs.write_yamls(str(tmp_path), ("a",))[0]
    rc = cli.main(["pred", "--mode", "fast", "--input", y, "--out_dir", str(tmp_path / "o"), "--seed", "101", "--recycling_steps", "10", "--diffusion_samples", "5", "--write_full_pae"])
    assert rc == 0, capsys.readouterr().out
    (args, kw), = calls
    assert args[3] == [101] and kw["settings"]["recycling_steps"] == 10 and kw["settings"]["diffusion_samples"] == 5 and kw["settings"]["write_full_pae"] is True and kw["settings"]["sampling_steps"] is None
    assert kw["settings"]["seed"] == 101 and kw["settings"]["no_kernels"] is False
    assert cli.main(["pred", "--mode", "fast", "--input", y, "--out_dir", str(tmp_path / "o2"), "--seed", "1", "--seeds", "1,2"]) == rep.EXIT_USAGE
    assert "give --seed or --seeds, not both" in capsys.readouterr().err
    assert cli.main(["pred", "--mode", "fast", "--input", y, "--out_dir", str(tmp_path / "o3"), "--no_kernels"]) == rep.EXIT_USAGE
    assert "not served on the worker route (--mode fast): --no_kernels: " in capsys.readouterr().out


def test_cli_plumbs_the_options_on_the_stock_route_in_the_callers_order_and_refuses_seeds(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("BOLTZ_CACHE", str(tmp_path / "nocache"))                                       # the cache check refuses before any launch: enough to see the usage errors first
    y = _stubs.write_yamls(str(tmp_path), ("a",))[0]
    assert cli.main(["pred", "--mode", "off", "--input", y, "--out_dir", str(tmp_path / "o"), "--seeds", "1,2"]) == rep.EXIT_USAGE
    assert "--seeds belongs to the worker route" in capsys.readouterr().err
    assert cli.main(["pred", "--mode", "off", "--input", y, "--out_dir", str(tmp_path / "o"), "--seed", "3", "--diffusion_samples", "0"]) == rep.EXIT_USAGE
    assert "--diffusion_samples 0 is below 1" in capsys.readouterr().err
    argv, _env = cli.stock_command(y, str(tmp_path / "o"), {"recycling_steps": 10, "diffusion_samples": 5, "seed": 3, "write_full_pae": True}, ["--", "--use_potentials"], ["seed", "write_full_pae", "diffusion_samples"])
    assert argv[argv.index("--") + 1:] == ["predict", y, "--out_dir", str(tmp_path / "o"), "--seed", "3", "--write_full_pae", "--diffusion_samples", "5", "--recycling_steps", "10", "--use_potentials"], "the caller's order, then declaration order, then the arguments after --"
    assert cli.knob_order(["--mode", "off", "--seed", "3", "--input", "x", "--no_kernels", "--recycling_steps=2", "--", "--seed", "9"]) == ["seed", "no_kernels", "recycling_steps"]


def test_upstream_sample_chunks_is_torch_chunk_of_upstreams_request():
    """boltz diffusionv2.py: arange(D).chunk(D % M + 1) — torch.chunk semantics, table-checked (incl. the quirks: D=10, M=5 is ONE chunk; D=3, M=5 is three)."""
    table = {(1, 5): [1], (2, 5): [1, 1], (3, 5): [1, 1, 1], (4, 5): [1, 1, 1, 1], (5, 5): [5], (6, 5): [3, 3], (7, 5): [3, 3, 1], (10, 5): [10],
             (3, 2): [2, 1], (4, 2): [4], (8, 8): [8], (25, 5): [25], (9, 4): [5, 4], (11, 3): [4, 4, 3]}
    assert {k: settings.upstream_sample_chunks(*k) for k in table} == table


def test_distinct_sample_chunk_sizes_take_the_single_shape_sampler_levers_off_by_name_never_a_refusal():
    """A kit accepts everything stock accepts (0.3.21): a (--diffusion_samples D, --max_parallel_samples M) whose upstream chunks
    (arange(D).chunk(D % M + 1)) have DISTINCT sizes is served on every row — the single-shape sampler levers the row carries
    (settings.SINGLE_SHAPE_LEVERS: the roll-out graph / the graphed sampler / the DiT hoist capture one chunk shape per prediction) step aside
    BY NAME for that run with their riders (`run_off=<lever>:distinct_sample_chunks`, the stock eager loop serves), never a refusal; equal-size
    chunks — one chunk of any size (D=5 or 10 at M=5, D=10 at M=2: one chunk by upstream's rule) or several equal ones (D=3 -> [1,1,1], D=6 ->
    [3,3] at M=5) — keep every lever on; big carries none of those levers, nothing leaves its row."""
    rule = modes.RUN_DROPS["distinct_sample_chunks"]
    assert set(rule["modes"]) == {"exact", "fast", "big"} and set(rule["levers"]) == set(settings.SINGLE_SHAPE_LEVERS) and callable(rule["when"])
    for mode in ("exact", "fast"):
        carried = [l for l in settings.SINGLE_SHAPE_LEVERS if l in modes.resolve(mode)["levers"]]
        assert {"rollout", "dit_hoist"} <= set(carried)
        for D, M in ((3, 2), (5, 2), (7, 5), (9, 4), (9, 5), (11, 3), (13, 5), (14, 5)):
            assert len(set(settings.upstream_sample_chunks(D, M))) > 1, (D, M)
            eff = settings.worker_settings(mode, {"diffusion_samples": D, "max_parallel_samples": M})          # no exception: the run is served
            assert (eff["diffusion_samples"], eff["max_parallel_samples"]) == (D, M)
            assert all(eff["run_off"].get(l) == "distinct_sample_chunks" for l in carried), (mode, D, M, eff["run_off"])
            assert eff["run_off"] == modes.run_drops_for(mode, {"diffusion_samples": D, "max_parallel_samples": M}) and set(eff["run_off"].values()) == {"distinct_sample_chunks"}
            modes.set_run_drops(eff["run_off"])
            try:
                row = modes.resolve(mode)
                assert not set(carried) & set(row["levers"]) and not set(eff["run_off"]) & set(row["levers"]) and eff["run_off"].items() <= row["run_off"].items()   # the levers and their riders left the row for this run (a row may name further conditional riders, e.g. the key gather without the hoist on fast)
                line = rep.active_line({"mode": mode, "route": "worker", "levers_applied": row["levers"], "levers_fallback": [], "worker": "w", "kernels": "on", "run_off": row["run_off"]})
                assert "run_off=" in line and "rollout:distinct_sample_chunks" in line and "dit_hoist:distinct_sample_chunks" in line, line
            finally:
                modes.set_run_drops(None)
        for D, M in ((1, 5), (1, 2), (2, 5), (3, 5), (4, 5), (5, 5), (6, 5), (10, 5), (10, 2), (4, 2), (8, 8)):
            assert len(set(settings.upstream_sample_chunks(D, M))) == 1, (D, M)
            eff = settings.worker_settings(mode, {"diffusion_samples": D, "max_parallel_samples": M})
            assert (eff["diffusion_samples"], eff["max_parallel_samples"]) == (D, M) and eff["run_off"] == {}, (mode, D, M, eff["run_off"])   # one chunk shape: every lever stays on
    eff = settings.worker_settings("big", {"diffusion_samples": 7, "max_parallel_samples": 5})
    assert eff["diffusion_samples"] == 7 and eff["run_off"] == {}, \
        "big serves distinct chunk sizes as before: it carries no single-shape sampler lever (off by rule, 0.3.17), the stock eager loop serves every size and nothing leaves for the run"
    assert settings.worker_settings("big", {"diffusion_samples": 6, "max_parallel_samples": 5})["run_off"] == {} and settings.worker_settings("big", {"use_potentials": True})["run_off"] == {}


def test_expected_files_follow_the_output_format():
    assert worker.expected_files("a", False) == ["a_model_0.cif"] and worker.expected_files("a", True, "pdb") == ["a_model_0.pdb", "affinity_a.json"]


def test_evidence_expects_the_graph_sampler_to_replay_all_but_the_first_of_the_calls_sampling_steps():
    """The roll-out captures once per prediction shape and replays every step after the first: S-1 replays per prediction at the call's
    --sampling_steps S (upstream's 200 when the call names none) — fewer is a problem, by name."""
    from . import _faster
    row = dict(stack.modes.env_row("exact"))
    def wlog(n_replay, items=1):
        log = {"env": row, "events": [], "attach": {},
               "per_item": [{"name": "a", "seed": s, "graph_sampler_mode": "off", "graph_n_replay": 0, "hoist_hoist_level": 2, "hoist_captured_with_cache": 0, "hoist_capture_stock_fallbacks": 0} for s in range(items)]}
        log.update(_faster.reports(row, items, replay=n_replay))
        return log
    def rollout_problems(n_replay, S):
        _, problems = stack.evidence("exact", wlog(n_replay), "", attached=False, sampling_steps=S)
        return [p for p in problems if p.startswith("roll-out:")]
    assert rollout_problems(199, None) == [] == rollout_problems(199, 200) == rollout_problems(9, 10) == rollout_problems(1, 2)
    assert rollout_problems(9, 200) and "replays=9 for 1 item(s)" in rollout_problems(9, 200)[0] and "199 replays" in rollout_problems(9, 200)[0]
    assert rollout_problems(9, None), "upstream's default S=200 when the call names none"
