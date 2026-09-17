"""`warm`: the kit's one-time kernel and cache setup ahead of the first pred — argument parsing, the mode list, the synthetic inputs, the pred
it composes (the stub interpreters of conftest stand in for the venvs), the WARM line's words, the cache census / new-files bookkeeping, the
temporary directory's removal, run.sh's dispatch. CPU only."""
import json
import os
import re
import subprocess
import tempfile

import pytest

from af3_torch_opt import cli, modes, stack

from .conftest import HOME, stub_calls

WARM_RE = re.compile(r"^\[af3-torch-opt\] WARM mode=(\S+) sizes=(\S+) inputs=(\S+) cache_root=(\S+) cache_before=(\d+)files/([\d.]+)MB cache_after=(\d+)files/([\d.]+)MB new_files=(-?\d+) seconds=([\d.]+) rc=(\d+)$")


def warm_lines(err: str):
    return [ln for ln in err.splitlines() if ln.startswith("[af3-torch-opt] WARM ")]


# ---- parsing ------------------------------------------------------------------------------------------------------------------------------
def test_parser_takes_mode_all_positional_and_flag_inputs():
    a = cli.build_parser().parse_args(["warm"])
    assert a.cmd == "warm" and a.mode is None and a.inputs == [] and a.json_path is None and a.input_dir is None and a.fn is cli.cmd_warm
    a = cli.build_parser().parse_args(["warm", "--mode", "all", "x.json", "--json_path", "y.json", "--input_dir", "d"])
    assert a.mode == "all" and a.inputs == ["x.json"] and a.json_path == ["y.json"] and a.input_dir == "d"
    with pytest.raises(SystemExit):                                                # warm has no --n_gpu / --output_dir: one GPU, a temporary output dir of its own
        cli.build_parser().parse_args(["warm", "--n_gpu", "2"])


def test_warm_modes(monkeypatch):
    monkeypatch.delenv(modes.ENV_MODE, raising=False)
    assert cli.warm_modes("all") == list(modes.MODES) == ["off", "exact", "fast", "big"]
    assert cli.warm_modes(None) == [modes.DEFAULT_MODE]                            # no --mode: the mode a pred without --mode runs
    monkeypatch.setenv(modes.ENV_MODE, "exact")
    assert cli.warm_modes(None) == ["exact"]                                       # AF3_TORCH_OPT names it, as for pred
    for m in modes.MODES:
        assert cli.warm_modes(m) == [m]
    with pytest.raises(modes.UnsupportedMode):
        cli.warm_modes("bogus")


def test_unknown_mode_is_usage(box, capsys):
    assert cli.main(["warm", "--mode", "bogus"]) == 2
    assert "warm: unknown mode 'bogus'" in capsys.readouterr().err and stub_calls(box) == []   # refused before anything is written or launched


def test_missing_named_input_is_usage(box, capsys):
    assert cli.main(["warm", "--mode", "fast", str(box["tmp"] / "nope.json")]) == 2
    assert "warm: input not found" in capsys.readouterr().err and stub_calls(box) == []


# ---- the synthetic inputs -----------------------------------------------------------------------------------------------------------------
def test_synthetic_inputs_are_single_chain_query_only_at_the_tile_lengths(tmp_path):
    assert cli.WARM_SIZES == (448, 832, 1216)
    from opt_core import shape_policy as sp                                        # the sizes ARE the padded lengths of the 400 / 800 / 1200-token rungs under fast / big, and tile multiples themselves
    assert [sp.padded_len(n, "kernel_tile") for n in (400, 800, 1200)] == list(cli.WARM_SIZES) and all(sp.padded_len(n, "kernel_tile") == n for n in cli.WARM_SIZES)
    paths = cli.warm_inputs(str(tmp_path / "in"), sizes=(1216, 448, 832, 448))     # any order, duplicates folded: ascending, one file per size
    assert [os.path.basename(p) for p in paths] == ["warm0448.json", "warm0832.json", "warm1216.json"]
    example = json.load(open(os.path.join(HOME, "inputs", "1BRS.json"), encoding="utf-8"))
    assert cli.WARM_RESIDUES == example["sequences"][0]["protein"]["sequence"]        # the example's first chain, repeated to length
    for p, n in zip(paths, (448, 832, 1216)):
        fi = json.load(open(p, encoding="utf-8"))
        assert fi["name"] == f"warm{n:04d}" == cli.item_name(p) and fi["dialect"] == example["dialect"] and fi["version"] == example["version"] and fi["modelSeeds"] == [1]
        (entry,) = fi["sequences"]; ch = entry["protein"]
        assert set(ch) == set(example["sequences"][0]["protein"]) and ch["id"] == "A" and ch["modifications"] == [] and ch["templates"] == [] and ch["pairedMsa"] == ""
        assert len(ch["sequence"]) == n and set(ch["sequence"]) <= set(cli.WARM_RESIDUES) and ch["sequence"].startswith(cli.WARM_RESIDUES)
        assert ch["unpairedMsa"] == f">query\n{ch['sequence']}\n"                    # query-only MSA: nothing for the data pipeline to search
        assert cli.templates_declared(p) == 0
    assert cli.warm_sequence(5) == cli.WARM_RESIDUES[:5] and len(cli.warm_sequence(1000)) == 1000
    with pytest.raises(ValueError):
        cli.warm_sequence(0)


def test_cache_census_counts_files_and_bytes(tmp_path):
    assert cli.cache_census(None) == (0, 0) and cli.cache_census(str(tmp_path / "absent")) == (0, 0)
    root = tmp_path / "cache"; (root / "triton" / "k").mkdir(parents=True); (root / "inductor").mkdir()
    (root / "triton" / "k" / "a.cubin").write_bytes(b"x" * 1500000); (root / "inductor" / "b").write_bytes(b"yy"); (root / "weights_digests.json").write_text("{}")
    assert cli.cache_census(str(root)) == (3, 1500004)
    assert cli.cache_word((3, 1500004)) == "3files/1.5MB" and cli.cache_word((0, 0)) == "0files/0.0MB"


def test_sizes_word_and_rc_rule():
    assert cli.warm_sizes_word({"items": [{"bucket": 256}]}, [448, 832, 1216]) == "448,832,1216"                  # synthetic: the constructed token counts
    assert cli.warm_sizes_word({"items": [{"bucket": 448, "n_tokens": 440}, {"bucket": 832}]}, None) == "448,832"   # named: the padded lengths that ran
    assert cli.warm_sizes_word({"items": [{"bucket": None, "n_tokens": None}]}, None) == "named" and cli.warm_sizes_word(None, None) == "named"
    assert cli.warm_rc([]) == 0 and cli.warm_rc([0, 0]) == 0 and cli.warm_rc([0, 3]) == 3 and cli.warm_rc([3, 1, 2]) == 1 and cli.warm_rc([0, 2]) == 2 and cli.warm_rc([0, 7]) == 7


def test_pred_args_are_the_pred_parsers_own_with_the_knobs_turned_down(tmp_path):
    a = cli.warm_pred_args("fast", ["/i/warm0448.json", "/i/warm0832.json"], str(tmp_path / "o"))
    assert a.cmd == "pred" and a.fn is cli.cmd_pred and a.mode == "fast" and a.json_path == ["/i/warm0448.json", "/i/warm0832.json"] and a.output_dir == str(tmp_path / "o")
    assert a.run_data_pipeline is False and a.run_inference is True                # the inputs carry their MSAs: nothing searched, no network
    assert a.num_recycles == cli.WARM_KNOBS["num_recycles"] == 1 and a.diffusion_steps == cli.WARM_KNOBS["diffusion_steps"] == 8
    assert a.num_diffusion_samples == modes.STOCK_NUM_DIFFUSION_SAMPLES and a.n_gpu == 1 and a.fastnn is None and not a.no_compile and not a.allow_partial and not a.quiet
    assert cli.warm_pred_args("exact", [], str(tmp_path / "o"), quiet=True).quiet


# ---- the verb end to end on the stub box ---------------------------------------------------------------------------------------------------
def test_warm_fast_runs_one_short_pred_and_prints_one_warm_line(box, capsys, monkeypatch):
    (box["tmp"] / "t").mkdir(); monkeypatch.setattr(tempfile, "tempdir", str(box["tmp"] / "t"))   # where tempfile.mkdtemp puts warm's directory in this process
    rc = cli.main(["warm", "--mode", "fast"])
    assert rc == 0
    err = capsys.readouterr().err
    lines = warm_lines(err)
    assert len(lines) == 1, err
    m = WARM_RE.match(lines[0]); assert m, lines[0]
    mode, sizes, source, root, nb, _, na, _, new, secs, prc = m.groups()
    assert mode == "fast" and sizes == "448,832,1216" and source == "synthetic" and root == stack.cache_root() == str(box["tmp"] / "cache") and prc == "0"
    assert int(na) - int(nb) == int(new) >= 0 and float(secs) >= 0
    assert err.index("[af3-torch-opt] ACTIVE mode=fast") < err.index("[af3-torch-opt] DONE ") < err.index("[af3-torch-opt] WARM ")   # the pred's own lines first, the WARM line after its DONE
    assert "[af3-torch-opt] SETTINGS num_recycles=1 num_diffusion_samples=5 diffusion_steps=8" in err   # the reduced protocol is named, as for any pred
    calls = stub_calls(box)
    feats = [c for c in calls if len(c) > 1 and os.path.basename(c[1]) == "featurise.py"]; (fwd,) = [c for c in calls if len(c) > 1 and os.path.basename(c[1]) == "forward.py"]
    assert feats and all(c[c.index("--run_data_pipeline") + 1] == "0" and "--repo_dir" not in c for c in feats)   # no data pipeline, no fork repo needed for it
    items = sorted(x.split("=")[0] for c in feats for i, x in enumerate(c) if c[i - 1] == "--item")
    assert items == ["warm0448", "warm0832", "warm1216"]
    assert fwd[fwd.index("--num_recycles") + 1] == "1" and fwd[fwd.index("--diffusion_steps") + 1] == "8" and fwd[fwd.index("--num_samples") + 1] == "5"
    assert fwd[fwd.index("--levers") + 1] == ",".join(modes.resolve("fast")["levers"])          # the mode's whole selection: every compiled / captured lever is reached
    srcs = [x.split("=")[1] for c in feats for i, x in enumerate(c) if c[i - 1] == "--item"]
    tmp_root = os.path.dirname(os.path.dirname(srcs[0]))
    assert os.path.basename(tmp_root).startswith("af3_torch_warm_") and os.path.dirname(tmp_root) == str(box["tmp"] / "t")
    assert not os.path.exists(tmp_root) and os.listdir(str(box["tmp"] / "t")) == []             # warm's temporary directory (inputs + outputs) is gone
    M = cli.last_run()
    assert M["ok"] and [it["name"] for it in M["items"]] == ["warm0448", "warm0832", "warm1216"] and M["activation"]["mode"] == "fast"


def test_second_warm_adds_no_cache_files(box, capsys):
    """Idempotence bookkeeping: the first warm on a fresh root may add the weights digest memo; a second one over the same root adds nothing
    (new_files=0) — on a GPU the same census words say whether the compile caches were already warm."""
    assert cli.main(["warm", "--mode", "exact"]) == 0
    first = WARM_RE.match(warm_lines(capsys.readouterr().err)[0]); assert first
    assert cli.main(["warm", "--mode", "exact"]) == 0
    second = WARM_RE.match(warm_lines(capsys.readouterr().err)[0]); assert second
    assert int(second.group(9)) == 0 and second.group(5) == second.group(7) == first.group(7)   # before == after == the first run's after


def test_warm_all_runs_every_mode_in_table_order(box, capsys):
    rc = cli.main(["warm", "--mode", "all", "--quiet"])
    err = capsys.readouterr().err
    got = [WARM_RE.match(ln) for ln in warm_lines(err)]
    assert all(got), err
    assert [g.group(1) for g in got] == list(modes.MODES) and all(g.group(11) == "0" for g in got) and rc == 0
    assert "[af3-torch-opt] ACTIVE " not in err                                   # --quiet reaches the preds
    fwds = [c for c in stub_calls(box) if len(c) > 1 and os.path.basename(c[1]) == "forward.py"]
    assert len(fwds) == len(modes.MODES)                                           # one model process per mode, the three sizes in each
    assert [("--fastnn" in c) for c in fwds] == [m in modes.FASTNN_MODES for m in modes.MODES]   # off / exact on xfold's kernels, fast / big on the kit's: each mode's own paths


def test_named_inputs_replace_the_synthetic_ones(box, capsys):
    d = box["tmp"] / "mine"; d.mkdir()
    p = d / "seq7.json"; p.write_text(json.dumps({"name": "seq7", "sequences": [], "modelSeeds": [3]}))
    q = d / "dir" ; q.mkdir(); (q / "b2.json").write_text(json.dumps({"name": "b2", "sequences": [], "modelSeeds": [1]}))
    rc = cli.main(["warm", "--mode", "exact", str(p), "--input_dir", str(q)])
    assert rc == 0
    (ln,) = warm_lines(capsys.readouterr().err); m = WARM_RE.match(ln); assert m
    assert m.group(3) == "named" and m.group(2) == "256,256"                       # the stub featuriser's bucket per named input, in input order
    feats = [c for c in stub_calls(box) if len(c) > 1 and os.path.basename(c[1]) == "featurise.py"]
    items = sorted(x.split("=")[0] + "@" + x.split("=")[1] for c in feats for i, x in enumerate(c) if c[i - 1] == "--item")
    assert items == [f"b2@{q / 'b2.json'}", f"seq7@{p}"] and p.exists() and (q / "b2.json").exists()   # the user's files are read in place and left alone


def test_a_failed_pred_is_warms_rc_and_the_temp_dir_still_goes(box, capsys, monkeypatch):
    (box["tmp"] / "t").mkdir(); monkeypatch.setattr(tempfile, "tempdir", str(box["tmp"] / "t"))
    monkeypatch.setenv("STUB_FAIL", "forward:warm0832")
    rc = cli.main(["warm", "--mode", "fast"])
    (ln,) = warm_lines(capsys.readouterr().err); m = WARM_RE.match(ln); assert m
    assert rc == 1 and m.group(11) == "1" and os.listdir(str(box["tmp"] / "t")) == []


# ---- run.sh dispatches it ------------------------------------------------------------------------------------------------------------------
def test_run_sh_lists_and_dispatches_warm():
    src = open(os.path.join(HOME, "run.sh"), encoding="utf-8").read()
    assert "#   run.sh warm    [--config h100] [--mode M|all]" in src and 'case "$CMD" in pred|check|stock|install|warm)' in src
    usage = subprocess.run(["bash", os.path.join(HOME, "run.sh")], capture_output=True, text=True)
    assert usage.returncode == 2 and "run.sh warm" in usage.stderr and "Exit codes: 0 ok, 1 failed, 2 usage, 3 not active" in usage.stderr   # the usage block spans the whole header, the new line included
    assert '[ "$CMD" = warm ] && [ "$MODE" = all ]' in src                          # warm --mode all is no disagreement with a set AF3_TORCH_OPT
