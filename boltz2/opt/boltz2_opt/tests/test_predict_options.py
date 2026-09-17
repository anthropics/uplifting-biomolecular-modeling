"""boltz predict's remaining options are inputs of `pred` on every mode, upstream's names / kinds / defaults verbatim: --checkpoint,
--use_msa_server / --msa_server_url / --msa_pairing_strategy, --max_msa_seqs, --preprocessing-threads, --method, --write_embeddings, the
affinity options, --subsample_msa / --num_subsampled_msa, --num_workers, --use_potentials. On `off` they are the stock command's arguments in
the caller's order; on the worker route each reaches the ONE site the persistent worker states upstream's default at (bz_worker_lev*.py):
process_inputs' kwargs, MSAModuleArgs, the steering switches, the data module's override_method, the writer, affinity_leg.request, the
checkpoint, the DataLoader's num_workers. `--use_potentials` on exact / fast takes the graphed sampler and the DiT hoist OFF by name for that
run (two LEVER … state=off reason=use_potentials lines); nothing given = the lines as tabled, byte for byte."""
import json
import os
import re

import pytest

from .. import manifest as mf
from .. import cli, modes, report as rep, settings, stack, worker
from . import _stubs

BASES = ("bz_worker_lev.py", "bz_worker_levf2.py")


def _ckpt(tmp_path, name="my.ckpt"):
    p = tmp_path / name; p.write_text("x"); return str(p)


def test_the_stock_command_carries_every_option_verbatim_in_the_callers_order(tmp_path, monkeypatch):
    monkeypatch.setenv("BOLTZ_CACHE", "/cache")
    y = _stubs.write_yamls(str(tmp_path), ("a",))[0]; ck = _ckpt(tmp_path); ack = _ckpt(tmp_path, "aff.ckpt")
    argv = ["pred", "--mode", "off", "--input", y, "--out_dir", str(tmp_path / "o"), "--use_msa_server", "--msa_server_url", "http://msa", "--msa_pairing_strategy", "complete",
            "--preprocessing-threads", "4", "--checkpoint", ck, "--method", "x-ray diffraction", "--write_embeddings", "--subsample_msa", "--num_subsampled_msa", "12", "--max_msa_seqs", "64",
            "--use_potentials", "--affinity_mw_correction", "--sampling_steps_affinity", "7", "--diffusion_samples_affinity", "2", "--affinity_checkpoint", ack, "--num_workers", "3"]
    a = cli.build_parser().parse_args(argv); a.knob_order = cli.knob_order(argv)
    cmd, _env = cli.stock_command(a.input[0], a.out_dir, cli.knob_flags(a), [], a.knob_order, None)
    tail = cmd[cmd.index("predict") + 2:]
    assert tail[:2] == ["--out_dir", str(tmp_path / "o")]
    assert tail[2:] == ["--use_msa_server", "--msa_server_url", "http://msa", "--msa_pairing_strategy", "complete", "--preprocessing-threads", "4", "--checkpoint", ck, "--method", "x-ray diffraction",
                        "--write_embeddings", "--subsample_msa", "--num_subsampled_msa", "12", "--max_msa_seqs", "64", "--use_potentials", "--affinity_mw_correction",
                        "--sampling_steps_affinity", "7", "--diffusion_samples_affinity", "2", "--affinity_checkpoint", ack, "--num_workers", "3"], tail
    with pytest.raises(ValueError, match="--checkpoint '/nope.ckpt': no such file or directory"):        # click.Path(exists=True): refused before anything runs, as boltz does
        settings.given({"checkpoint": "/nope.ckpt"})
    try:
        from boltz.data import const as _const                                                            # where boltz is importable, --method is checked against boltz's own list with main.py's message
    except ImportError:
        _const = None
    if _const is not None:
        with pytest.raises(ValueError, match=r"Method x-ray not supported\. Supported: \["):
            settings.given({"method": "x-ray"})
        assert settings.given({"method": "X-RAY DIFFRACTION"}) == {"method": "X-RAY DIFFRACTION"}
    assert settings.stock_argv({}) == [] and settings.flag("preprocessing_threads") == "--preprocessing-threads" and cli.knob_order(["--preprocessing-threads", "2"]) == ["preprocessing_threads"]
    main = open(os.path.join(stack.tree_dir(), "stock", "src", "boltz", "main.py")).read()
    for k in settings.KNOB_NAMES:                                                                         # every knob is a boltz predict option, spelled as main.py spells it
        assert f'"{settings.flag(k)}",' in main, k
    d = settings.stock_defaults()
    assert (d["msa_server_url"], d["msa_pairing_strategy"], d["num_subsampled_msa"], d["sampling_steps_affinity"], d["diffusion_samples_affinity"], d["max_msa_seqs"]) == ("https://api.colabfold.com", "greedy", 1024, 200, 5, 8192)
    for k in ("use_msa_server", "use_potentials", "affinity_mw_correction", "subsample_msa", "write_embeddings"):
        assert re.search(r'"--%s",(?:(?!@click).)*?is_flag=True' % k, main, re.S) and d[k] is False, k


def test_the_worker_route_forwards_each_option_to_the_workers_one_site(tmp_path, monkeypatch, capsys):
    _stubs.gated_ok(monkeypatch); _stubs.install_fake_stage(monkeypatch)
    seen = []
    real = stack.worker_command
    monkeypatch.setattr(stack, "worker_command", lambda *a, **k: seen.append((a, k)) or real(*a, **k))
    ys = _stubs.write_yamls(str(tmp_path), ("a",)); ck = _ckpt(tmp_path); ack = _ckpt(tmp_path, "aff.ckpt")
    opts = {"use_msa_server": True, "msa_server_url": "http://msa", "msa_pairing_strategy": "complete", "preprocessing_threads": 4, "max_msa_seqs": 64, "checkpoint": ck, "method": "x-ray diffraction",
            "write_embeddings": True, "subsample_msa": True, "num_subsampled_msa": 12, "affinity_mw_correction": True, "sampling_steps_affinity": 7, "diffusion_samples_affinity": 2,
            "affinity_checkpoint": ack, "num_workers": 3}
    rc = worker.run("fast", ys, str(tmp_path / "o"), [0], settings=opts)
    text = capsys.readouterr().out; assert rc == 0, text
    b = json.load(open([os.path.join(dp, f) for dp, _, fn in os.walk(tmp_path / "o" / "_kit") for f in fn if f.startswith("batch_")][0]))
    assert b["checkpoint"] == os.path.abspath(ck), "the caller's --checkpoint is the worker's CKPT (Boltz2.load_from_checkpoint)"
    assert b["options"] == {"method": "x-ray diffraction", "write_embeddings": True, "use_msa_server": True, "msa_server_url": "http://msa", "msa_pairing_strategy": "complete",
                            "preprocessing_threads": 4, "max_msa_seqs": 64, "subsample_msa": True, "num_subsampled_msa": 12,
                            "affinity": {"affinity_mw_correction": True, "sampling_steps_affinity": 7, "diffusion_samples_affinity": 2, "affinity_checkpoint": os.path.abspath(ack)}}, b["options"]
    nw = lambda c: c[1]["num_workers"] if "num_workers" in c[1] else (c[0][4] if len(c[0]) > 4 else None)   # noqa: E731
    n1 = len(seen); assert n1 >= 1 and all(nw(c) == 3 for c in seen), "the caller's --num_workers is the worker command's (its DataLoader's)"
    cmd = stack.worker_command("b.json", "big", 1, "flags", 3)
    assert cmd[cmd.index("--num_workers") + 1] == "3" and stack.worker_command("b.json", "big")[stack.worker_command("b.json", "big").index("--num_workers") + 1] == modes.WORKER_ARGS["num_workers"]
    active = [l for l in text.splitlines() if l.startswith("[boltz2-opt] ACTIVE ")][0]
    for word in ("num_workers=3", "checkpoint=", "method=x-ray diffraction", "subsample_msa=1", "num_subsampled_msa=12", "sampling_steps_affinity=7", "max_msa_seqs=64", "write_embeddings=1"):
        assert word in active, (word, active)
    assert " state=off " not in text
    m = mf.LAST
    assert m["settings"]["options"]["affinity_mw_correction"] is True and m["settings"]["run_off"] == {} and m["settings"]["num_workers"] == 3
    rep.reset_tally(); seen.clear()
    rc = worker.run("fast", ys, str(tmp_path / "o2"), [0])                                                 # nothing given: options empty, the command's num_workers is the line's own
    b = json.load(open([os.path.join(dp, f) for dp, _, fn in os.walk(tmp_path / "o2" / "_kit") for f in fn if f.startswith("batch_")][0]))
    assert rc == 0 and b["options"] == {"affinity": {}} and b["checkpoint"].endswith("boltz2_conf.ckpt") and seen and all(nw(c) is None for c in seen)
    big = settings.worker_settings("big", {"use_potentials": True})                                     # big carries the roll-out and the hoist up to its ceiling (0.3.13): --use_potentials takes them off by name as on fast; its fused step stays (eager, as before); the switch reaches the worker
    assert big["run_off"] == {} and big["options"] == {"use_potentials": True}   # big carries no roll-out / hoist (off by rule, 0.3.17): the option is forwarded, nothing further leaves


@pytest.mark.parametrize("base", BASES)
def test_the_worker_states_each_option_once_at_upstreams_site(base):
    src = open(stack.kit_path(f"forward/trunk_levers/src/{base}")).read()
    assert 'OPTS = B.get("options") or {}' in src
    assert 'MSAModuleArgs(subsample_msa=bool(OPTS.get("subsample_msa", False)), num_subsampled_msa=int(OPTS.get("num_subsampled_msa", 1024)), use_paired_feature=True)' in src
    assert 'steering_args.fk_steering = steering_args.physical_guidance_update = bool(OPTS.get("use_potentials", False))' in src   # main.py:1310-1311
    assert src.count('override_method=OPTS.get("method")') == 2 and "override_method=None" not in src
    assert 'write_embeddings=bool(OPTS.get("write_embeddings", False))' in src and '**(OPTS.get("affinity") or {})' in src and 'CKPT = B["checkpoint"]' in src
    i = src.index("PROCESS_INPUTS_KW = dict("); j = src.index("\n", src.index("\n", i) + 1)
    for OPTS, want in (({}, dict(use_msa_server=False, msa_server_url="https://api.colabfold.com", msa_pairing_strategy="greedy", preprocessing_threads=1, max_msa_seqs=8192, boltz2=True)),
                       ({"use_msa_server": True, "max_msa_seqs": 64, "method": "nmr"}, dict(use_msa_server=True, msa_server_url="https://api.colabfold.com", msa_pairing_strategy="greedy", preprocessing_threads=1, max_msa_seqs=64, boltz2=True))):
        ns = {"OPTS": OPTS}; exec(src[i:j], ns)
        assert ns["PROCESS_INPUTS_KW"] == want, (base, OPTS, ns["PROCESS_INPUTS_KW"])                  # upstream's process_inputs arguments: the CLI defaults, or the caller's


def test_use_potentials_on_exact_and_fast_takes_the_graphed_sampler_and_the_hoist_off_by_name(tmp_path, monkeypatch, capsys):
    assert modes.run_drops_for("exact", {"use_potentials": True}) == {"rollout": "use_potentials", "align_aligncap": "use_potentials", "dit_hoist": "use_potentials", "dit_par": "use_potentials", "dit_mask": "use_potentials", "dit_sba": "use_potentials", "dit_smx": "use_potentials", "dit_glue": "use_potentials"}
    assert modes.run_drops_for("fast", {"use_potentials": True}) == {"rollout": "use_potentials", "align_jacobi64": "use_potentials", "dit_fused": "use_potentials", "dit_hoist": "use_potentials"}, "fast: the fused bf16 step and the in-graph Kabsch ride the roll-out and leave with it"
    assert modes.run_drops_for("big", {"use_potentials": True}) == {}, "big carries no roll-out, rider or hoist (off by rule, 0.3.17): nothing further leaves; formerly (0.3.15): (served from the eager loop, NEEDS_IN)"
    assert {} == modes.run_drops_for("exact", {}) == modes.run_drops_for("exact", {"use_potentials": False}) == modes.run_drops_for("big", {})
    tabled = modes.resolve("exact")
    _stubs.gated_ok(monkeypatch); _stubs.install_fake_stage(monkeypatch)
    ys = _stubs.write_yamls(str(tmp_path), ("a",))
    try:
        rc = worker.run("exact", ys, str(tmp_path / "o"), [0], settings={"use_potentials": True})
        text = capsys.readouterr().out; assert rc == 0, text
        row = modes.resolve("exact")                                                                       # this process's row for the run: the two levers and their switches left it, named
        OFF = {"rollout": "use_potentials", "align_aligncap": "use_potentials", "dit_hoist": "use_potentials", "dit_par": "use_potentials", "dit_mask": "use_potentials", "dit_sba": "use_potentials", "dit_smx": "use_potentials", "dit_glue": "use_potentials"}   # the roll-out (and the alignment seam that rides it), the hoist, and the schedule levers that read the hoist's cache
        assert row["run_off"] == OFF and not set(OFF) & set(row["levers"])
        assert "BOLTZ_SAMPLER_ROLLOUT" not in row["env"] and "BOLTZ_SAMPLER_ALIGN" not in row["env"] and "BOLTZ_DIT_HOIST" not in row["env"] and "BOLTZ_DIT_EXACT" not in row["env"] and "sampler" not in row["attach"] and "ditexact" not in row["attach"] and row["env"]["BOLTZ_LEVERS"] == tabled["env"]["BOLTZ_LEVERS"], "every trunk lever stays on"
        lines = [l for l in text.splitlines() if l.startswith("[boltz2-opt] LEVER ")]
        off = [l for l in lines if " state=off " in l]
        assert [l.split()[2] for l in off] == [f"name={n}" for n in OFF] and all(" reason=use_potentials " in l for l in off) and "[boltz2-opt] LEVER name=dit_hoist state=off reason=use_potentials impl=forward/dit_hoist/src/boltz_dit_hoist.py origin=kit strategy=LOCAL.step_invariant_hoist" in off, off
        assert [l.split()[2] for l in lines if " state=on " in l] == [f"name={n}" for n in tabled["levers"] if n not in OFF], "the other levers report engaged"
        active = [l for l in text.splitlines() if l.startswith("[boltz2-opt] ACTIVE ")][0]
        assert "graph_sampler" not in active and "use_potentials=1" in active
        b = json.load(open([os.path.join(dp, f) for dp, _, fn in os.walk(tmp_path / "o" / "_kit") for f in fn if f.startswith("batch_")][0]))
        m = mf.LAST
        assert b["options"]["use_potentials"] is True and m["settings"]["run_off"] == modes.run_drops_for("exact", {"use_potentials": True})
        assert "BOLTZ_GRAPH_DIFFUSION" not in m["report"]["env"] and m["report"]["levers_applied"] == row["levers"]
        rep.reset_tally()
        rc = worker.run("exact", ys, str(tmp_path / "o2"), [0])                                            # the next run without the option: the row as tabled again (the statement is per run)
        assert rc == 0 and modes.resolve("exact") == tabled and " state=off " not in capsys.readouterr().out
    finally:
        modes.set_run_drops(None)
    eff = settings.worker_settings("exact", {"diffusion_samples": 3, "max_parallel_samples": 2})                # sample chunks of distinct sizes ([2, 1]): the single-shape sampler levers
    assert {l: eff["run_off"].get(l) for l in ("rollout", "dit_hoist")} == {"rollout": "distinct_sample_chunks", "dit_hoist": "distinct_sample_chunks"}   # step aside BY NAME for that run (never a refusal, 0.3.21)
    assert settings.worker_settings("exact", {"diffusion_samples": 3, "max_parallel_samples": 2, "use_potentials": True})["run_off"]["rollout"] == "use_potentials", "the first option that takes a lever off names it"
