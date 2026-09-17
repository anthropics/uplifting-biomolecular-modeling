"""The command line without a GPU or the stack: the kit's OWN words malformed exit 2 (one named line), upstream's arguments after `--` are
never judged here (they reach the deployment gates untouched), every mode — kit modes included — stops at the first deployment gate on a
box without the checkout and weights (exit 3, never a traceback), `check` prints one DRY-RUN line whose kit-mode form names the levers and
the hook's state."""
import json
import os
import subprocess
import sys

import pytest

from complexa_opt import report


def run(args, env=None):
    e = dict(os.environ)
    e.pop("COMPLEXA_OPT", None)
    e.update(env or {})
    r = subprocess.run([sys.executable, "-m", "complexa_opt"] + args, capture_output=True, text=True, env=e)
    return r.returncode, r.stderr


def entry_file(tmp_path):
    pdb = tmp_path / "target.pdb"
    pdb.write_text("ATOM\n")
    p = tmp_path / "e.json"
    p.write_text(json.dumps({"t1": {"source": "custom", "target_filename": "t1", "target_path": str(pdb), "target_input": "A1-10", "hotspot_residues": ["A5"], "binder_length": [80, 80], "pdb_id": None}}))
    return str(p)


def test_the_kits_own_words_malformed_exit_2_with_one_named_line(tmp_path):
    e = entry_file(tmp_path)
    cases = [
        ["design", "--mode", "turbo", "--input", e, "--out", str(tmp_path / "o")],
        ["design", "--mode", "default", "--input", e, "--out", str(tmp_path / "o")],                                   # `default` is not a mode word
        ["design", "--mode", "off", "--settings", "upstream", "--input", e, "--out", str(tmp_path / "o")],             # no settings presets: an unknown option
        ["design", "--mode", "off", "--designs", "4", "--seed", "5", "--input", e, "--out", str(tmp_path / "o")],      # upstream's knobs are Hydra overrides after `--`, not options
        ["design", "--mode", "off", "--input", e, "--out", str(tmp_path / "o"), "--batch", "4"],
        ["design", "--mode", "off", "--input", e, "--out", str(tmp_path / "o"), "--det", "1"],                         # no deterministic-recipe switch: upstream seeds itself (an unknown option)
        ["design", "--mode", "off", "--input", e, "--out", str(tmp_path / "o"), "--run-name", "p1"],                    # the run name is upstream's own ++run_name= after `--`
        ["design", "--mode", "off", "--input", e, "--out", str(tmp_path / "o"), "--json"],                             # retired options are unknown options
        ["design", "--mode", "off", "--input", e, "--out", str(tmp_path / "o"), "--no-weights-sha"],
        ["design", "--mode", "off", "--input", str(tmp_path / "missing.json"), "--out", str(tmp_path / "o")],
        ["design", "--mode", "off", "--input", e],                                                                     # --out is the one required option
        ["check", "--mode", "off", "--json"],
        ["frobnicate"],
        ["--mode", "off"],                                                                                              # no command
    ]
    for args in cases:
        rc, err = run(args)
        assert rc == report.EXIT_USAGE == 2, (args, err)
        assert err.count("[complexa-opt] USAGE refused: ") == 1 and "Traceback" not in err, (args, err)
    rc, err = run(["design", "--mode", "off", "--input", e, "--out", str(tmp_path / "o")], env={"COMPLEXA_OPT": "exact"})
    assert rc == 2 and "disagrees" in err


def test_design_without_weights_or_checkout_is_not_active(tmp_path):
    e = entry_file(tmp_path)
    base = ["design", "--mode", "off", "--input", e, "--out", str(tmp_path / "o"), "--", "++run_name=p1"]
    rc, err = run(base, env={"CKPT_PATH": ""})
    assert rc == 3 and "CKPT_PATH is not set" in err, err
    rc, err = run(base, env={"CKPT_PATH": str(tmp_path)})
    assert rc == 3 and "UNPINNED" in err and "weights" in err, err


def test_upstreams_arguments_after_dashdash_are_never_judged_here(tmp_path):
    """Whatever follows `--` — Hydra overrides for ANY key (also the ones the package spells itself: run name, task, checkpoints, scope, job
    split), upstream's generate options, or plain nonsense — passes the command line untouched; the run then stops at the first deployment
    gate (no checkpoint on a CPU test box: exit 3, never 2), which proves nothing before the launch judged the tokens. So does a run without
    --input (optional)."""
    e = entry_file(tmp_path)
    tails = [["++generation.dataloader.dataset.nres.nsamples=32", "++seed=7", "++generation.dataloader.batch_size=32"],
             ["++run_name=x", "++generation.task_name=02_PDL1", "++job_id=1", "++gen_njobs=2", "++root_path=/elsewhere"],
             ["++generation.search.algorithm=single-pass", "++generation.reward_model=null", "++ckpt_path=/x", "++ckpt_name=y"],
             ["--job-id", "3", "notanoverride", "++generation.dataloader.dataset.nres.nsamples=0", "++seed=notint", "~seed"]]
    for i, tail in enumerate(tails):
        for head in (["--input", e], []):
            rc, err = run(["design", "--mode", "off", "--out", str(tmp_path / f"o{i}")] + head + ["--"] + tail, env={"CKPT_PATH": str(tmp_path)})
            assert rc == 3 and "USAGE" not in err and "WEIGHTS dir=" in err and "Traceback" not in err, (head, tail, err)


def test_check_reports_on_one_dry_run_line(tmp_path):
    rc, err = run(["check", "--mode", "off"], env={"CKPT_PATH": str(tmp_path)})
    dry = [l for l in err.splitlines() if " DRY-RUN " in l]
    assert rc == 3 and len(dry) == 1 and " mode=off route=stock " in dry[0] and " weights=UNPINNED " in dry[0], err
    assert "[complexa-opt] WEIGHTS dir=" in err


@pytest.mark.parametrize("mode", ["exact", "fast", "big"])
def test_kit_modes_stop_at_the_first_deployment_gate_like_stock(mode, tmp_path):
    """A kit mode is served: on a box without weights it stops where `off` stops (WEIGHTS unpinned, exit 3) — never a tier refusal, never usage."""
    e = entry_file(tmp_path)
    rc, err = run(["design", "--mode", mode, "--input", e, "--out", str(tmp_path / "o"), "--", "++run_name=p1"], env={"CKPT_PATH": str(tmp_path)})
    assert rc == 3 and "UNPINNED" in err and "ships no" not in err and "USAGE" not in err and "Traceback" not in err, err
    rc, err = run(["design", "--input", e, "--out", str(tmp_path / "o2")], env={"CKPT_PATH": str(tmp_path), "COMPLEXA_OPT": mode})
    assert rc == 3 and "UNPINNED" in err and "USAGE" not in err, err
    rc, err = run(["design", "--mode", mode])                                                                        # argparse judges the rest of the line: --out is required
    assert rc == 2 and "USAGE refused" in err, err


def test_check_names_levers_and_hook_state_for_a_kit_mode(tmp_path):
    """`check --mode big` on a CPU test box: one DRY-RUN line with route=kit, the mode's lever set, hook=missing (this interpreter has no
    complexa_opt_autoload.pth in its site directory) among the reasons — exit 3, the line still printed whole."""
    rc, err = run(["check", "--mode", "big"], env={"CKPT_PATH": str(tmp_path)})
    dry = [l for l in err.splitlines() if " DRY-RUN " in l]
    assert rc == 3 and len(dry) == 1, err
    assert " mode=big route=kit " in dry[0] and " levers=onehot_f32,target_hoist,pair_assembly,loop_desync,pair_bias_rows " in dry[0] and " hook=" in dry[0] and " ok=False" in dry[0], dry[0]


def test_no_mode_given_asks_for_the_default(tmp_path):
    """The default mode is fast: a run naming no mode is a fast run (here it stops at the weights gate like any run on a CPU box)."""
    e = entry_file(tmp_path)
    rc, err = run(["design", "--input", e, "--out", str(tmp_path / "o")], env={"CKPT_PATH": str(tmp_path)})
    assert rc == 3 and "(mode=fast)" in err and "USAGE" not in err, err
    rc, err = run(["check"], env={"CKPT_PATH": str(tmp_path)})
    assert rc == 3 and " mode=fast route=kit " in err, err
    rc, err = run(["design", "--help"])
    assert rc == 0                                                                                                           # help is not a run: nothing is resolved or refused


def test_hydra_overrides_among_the_flags_join_the_tail_in_order(tmp_path):
    """`complexa generate <config> k=v …` takes overrides positionally; so does this command line: override tokens before `--` move behind it."""
    from complexa_opt import cli
    assert cli.split_overrides(["design", "--mode", "off", "++generation.args.nsteps=5", "--out", "o", "--", "++seed=1"]) == ["design", "--mode", "off", "--out", "o", "--", "++generation.args.nsteps=5", "++seed=1"]
    assert cli.split_overrides(["design", "++a=1", "~b", "--out", "o"]) == ["design", "--out", "o", "--", "++a=1", "~b"]
    assert cli.split_overrides(["design", "--mode", "off", "--out", "o", "--", "x=1"]) == ["design", "--mode", "off", "--out", "o", "--", "x=1"]      # nothing before `--`: untouched
    assert cli.split_overrides(["check", "--mode", "exact"]) == ["check", "--mode", "exact"]
    e = entry_file(tmp_path)
    rc, err = run(["design", "--mode", "off", "++generation.args.nsteps=5", "--input", e, "--out", str(tmp_path / "o"), "--", "++run_name=p1"], env={"CKPT_PATH": str(tmp_path)})
    assert rc == 3 and "USAGE" not in err and "UNPINNED" in err, err                                             # reached the deployment gate: the token was not judged here
