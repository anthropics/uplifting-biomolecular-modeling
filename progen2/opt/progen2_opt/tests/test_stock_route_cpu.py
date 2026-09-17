"""The stock route on a fake stock (CPU) through the `score` / `sample` verbs: ONE clean stock process per call with the flags as given (its
streams this process's own), or one per item of an --input job (items/<id>/block.txt, nothing else); the contract lines; `--device cpu` under
exact (refused by name, exit 3); the pins gate (a stack off the pins is a note, a stock off the pinned commit THE refusal on every mode)."""
import json
import os

from .conftest import run_package


def _fake_stock_with_scripts(fake_stock):
    ll = ('import argparse, sys\n'
          'ap = argparse.ArgumentParser(); ap.add_argument("--model"); ap.add_argument("--context", default="1ABC2")\n'
          'for k in ("--device", "--rng-seed", "--rng-deterministic", "--fp16", "--sanity"): ap.add_argument(k)\n'
          'a = ap.parse_args()\n'
          'print("loading parameters took 0.10s")\nprint(f"ll_sum={-1.5}")\nprint(f"ll_mean={-0.5}")\nprint("done.")\n')
    sa = ('import argparse, sys\n'
          'ap = argparse.ArgumentParser()\n'
          'for k in ("--model", "--context", "--t", "--p", "--rng-seed", "--num-samples", "--max-length", "--fp16", "--sanity"): ap.add_argument(k)\n'
          'a = ap.parse_args()\nprint("loading parameters took 0.10s")\nprint("sampling")\nprint(a.context)\nprint()\nprint(0)\nprint("1MSEQ2")\nprint("sampling took 0.01s")\nprint("done.")\n')
    (fake_stock / "likelihood.py").write_text(ll)
    (fake_stock / "sample.py").write_text(sa)
    (fake_stock / "tokenizer.json").write_text("{}")
    (fake_stock / "requirements.txt").write_text("")
    (fake_stock / "models" / "progen" / "configuration_progen.py").write_text("")
    return fake_stock



def _weights(tmp_path):
    weights = tmp_path / "weights" / "progen2-small"; weights.mkdir(parents=True); (weights / "pytorch_model.bin").write_bytes(b"x"); (weights / "config.json").write_text("{}")


def _items(tmp_path, name, rows):
    p = tmp_path / name
    p.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    return p


def test_score_off_outputs(fake_stock, tmp_path):
    sd = _fake_stock_with_scripts(fake_stock); _weights(tmp_path)
    out = tmp_path / "pass" / "s"
    items = _items(tmp_path, "items.jsonl", [{"item_id": "ubq", "context": "1MQIF2"}])
    env = {"PROGEN2_STOCK_DIR": str(sd), "PROGEN2_WEIGHTS": str(tmp_path / "weights"), "PROGEN2_RUN_DIR": str(tmp_path / "run"), "PROGEN2_OPT": "exact", "CUBLAS_WORKSPACE_CONFIG": ":16:8"}
    cmd = ["score", "--model", "progen2-small", "--mode", "off", "--input", str(items), "--out_dir", str(out)]
    rc, _, r = run_package(cmd, env, gate_ok=True)                                # gate_ok: the fake stock differs from the pins (conftest)
    assert rc == 0, r.stderr[-2000:]                                              # a stale PROGEN2_OPT in the caller's shell selects nothing (and is stripped from the child as a PROGEN2_ name)
    err = r.stderr
    assert "[progen2-opt] NOT ACTIVE: mode off: stock in a clean subprocess (mode=off variant=small)" in err
    assert "[progen2-opt stock] score progen2-small pid=" in err and "[progen2-opt stock] ENV-CLEAN ok stripped=" in err and "PROGEN2_OPT" in err.split("stripped=")[1].split()[0].split(",")
    assert "CUBLAS_WORKSPACE_CONFIG" not in err.split("stripped=")[1].split()[0]       # the caller's own variables reach the stock child: only the kit-side names go
    assert "[progen2-opt] ready variant=small t=0.10s" in err and "[progen2-opt] item ubq " in err
    assert "[progen2-opt] EXIT mode=off route=score items=1 kit_modules=none" in err
    assert sorted(os.listdir(out)) == ["items"] and os.listdir(out / "items" / "ubq") == ["block.txt"]      # the job's only output
    assert open(out / "items" / "ubq" / "block.txt").read() == "ll_sum=-1.5\nll_mean=-0.5\n"


def test_score_off_single_call_is_the_stock_process(fake_stock, tmp_path):
    """No --input: ONE stock process with the flags as given; its stdout is ours verbatim (the stock's whole output, not a cut block); no file."""
    sd = _fake_stock_with_scripts(fake_stock); _weights(tmp_path)
    env = {"PROGEN2_STOCK_DIR": str(sd), "PROGEN2_WEIGHTS": str(tmp_path / "weights"), "PROGEN2_RUN_DIR": str(tmp_path / "run")}
    rc, _, r = run_package(["score", "--model", "progen2-small", "--mode", "off", "--context", "1MQIF2"], env, gate_ok=True)
    assert rc == 0, r.stderr[-2000:]
    assert r.stdout == "loading parameters took 0.10s\nll_sum=-1.5\nll_mean=-0.5\ndone.\n", r.stdout          # the stock script's own stdout, untouched
    assert "[progen2-opt stock] ENV-CLEAN ok" in r.stderr and "[progen2-opt] item item0000 " in r.stderr and "EXIT mode=off route=score items=1 kit_modules=none" in r.stderr
    assert not (tmp_path / "run").exists() or not any(n.endswith(".json") and n.startswith("stock_env") for n in os.listdir(tmp_path))   # nothing written by the package
    (sd / "likelihood.py").write_text('import sys\nprint("stock says no", file=sys.stderr)\nsys.exit(7)\n')      # the stock's own exit code is returned
    rc, _, r = run_package(["score", "--model", "progen2-small", "--mode", "off"], env, gate_ok=True)
    assert rc == 7 and "stock says no" in r.stderr and "item item0000" in r.stderr and "FAILED rc=7" in r.stderr, r.stderr[-800:]


def test_sample_off_items_carry_the_stock_arguments(fake_stock, tmp_path):
    sd = _fake_stock_with_scripts(fake_stock); _weights(tmp_path)
    (sd / "sample.py").write_text((sd / "sample.py").read_text().replace('print("sampling")', 'print("ARGV", " ".join(sys.argv[1:]), file=sys.stderr)\nprint("sampling")'))
    out = tmp_path / "pass" / "g"
    env = {"PROGEN2_STOCK_DIR": str(sd), "PROGEN2_WEIGHTS": str(tmp_path / "weights"), "PROGEN2_RUN_DIR": str(tmp_path / "run")}
    items = _items(tmp_path, "gen.jsonl", [{"item_id": "ctx1_L256", "context": "1", "max_length": 256, "num_samples": 1, "t": 0.2, "p": 0.95, "rng_seed": 42},
                                          {"item_id": "second", "context": "1M", "max_length": 64, "rng_seed": 7}])
    rc, _, r = run_package(["sample", "--model", "progen2-small", "--mode", "off", "--fp16", "true", "--input", str(items), "--out_dir", str(out)], env, gate_ok=True)
    assert rc == 0, r.stderr[-2000:]
    assert open(out / "items" / "ctx1_L256" / "block.txt").read() == "1\n\n0\n1MSEQ2\ndone.\n" and open(out / "items" / "second" / "block.txt").read() == "1M\n\n0\n1MSEQ2\ndone.\n"
    assert sorted(os.listdir(out)) == ["items"] and sorted(os.listdir(out / "items" / "second")) == ["block.txt"]
    assert "[progen2-opt] item ctx1_L256 " in r.stderr and "[progen2-opt] item second " in r.stderr and "EXIT mode=off route=sample items=2 kit_modules=none" in r.stderr
    noid = _items(tmp_path, "noid.jsonl", [{"context": "1", "max_length": 32}])                       # item_id is optional: the item is numbered
    rc, _, r = run_package(["sample", "--model", "progen2-small", "--mode", "off", "--input", str(noid), "--out_dir", str(tmp_path / "p3")], env, gate_ok=True)
    assert rc == 0 and (tmp_path / "p3" / "items" / "item0000" / "block.txt").is_file(), r.stderr[-1200:]
    rc, _, r = run_package(["sample", "--model", "progen2-small", "--mode", "off", "--max-length", "64", "--num-samples", "2", "--rng-seed", "7", "--context", "1", "--sanity", "False"], env, gate_ok=True)
    assert rc == 0 and r.stdout.endswith("sampling\n1\n\n0\n1MSEQ2\nsampling took 0.01s\ndone.\n"), (r.stdout, r.stderr[-800:])      # one call: the stock's stdout verbatim
    assert "ARGV --model progen2-small --max-length 64 --num-samples 2 --context 1 --sanity False --rng-seed 7" in r.stderr or "ARGV --model progen2-small" in r.stderr, r.stderr
    argv_line = next(l for l in r.stderr.splitlines() if l.startswith("ARGV "))
    assert argv_line.split()[1:] == ["--model", "progen2-small", "--sanity", "False", "--rng-seed", "7", "--max-length", "64", "--num-samples", "2", "--context", "1"], argv_line   # --model, the job-level flags, then the per-item ones, each value as typed


def test_default_mode_is_exact_and_the_cpu_device_is_refused_by_name(fake_stock, tmp_path):
    """No --mode: the package default (exact). `--device cpu` under it is refused BY NAME, exit 3, nothing run: the levers run on CUDA and a
    mode is all of its levers (`--mode off --device cpu` runs the stock on the cpu — the line says so); a box without a CUDA device is refused
    the same way by the box's name; the stock route under --mode off takes the flag untouched."""
    from progen2_opt import stack
    sd = _fake_stock_with_scripts(fake_stock); _weights(tmp_path)
    base = {"PROGEN2_STOCK_DIR": str(sd), "PROGEN2_WEIGHTS": str(tmp_path / "weights"), "PROGEN2_RUN_DIR": str(tmp_path / "run")}
    items = _items(tmp_path, "items.jsonl", [{"item_id": "ubq", "context": "1MQIF2"}])
    cmd = ["score", "--model", "progen2-small", "--input", str(items), "--out_dir", str(tmp_path / "p" / "s")]
    rc, _, r = run_package(cmd + ["--mode", "off", "--device", "cpu"], base, gate_ok=True)
    assert rc == 0 and "EXIT mode=off route=score items=1 kit_modules=none" in r.stderr, r.stderr[-800:]          # the stock route: the flag is the stock script's
    assert open(tmp_path / "p" / "s" / "items" / "ubq" / "block.txt").read() == "ll_sum=-1.5\nll_mean=-0.5\n"
    rc, _, r = run_package(cmd + ["--device", "cpu", "--out_dir", str(tmp_path / "p" / "e")], base, gate_ok=True)
    assert rc == 3, r.stderr[-1200:]
    assert (f"[progen2-opt] NOT ACTIVE: {stack.DEVICE_CPU_REFUSAL.format(device='cpu')} (mode=exact variant=small route=score); {stack.CPU_ESCAPE}") in r.stderr.splitlines(), r.stderr[-1200:]
    assert "ENV-CLEAN" not in r.stderr and "stock main continued" not in r.stdout and not (tmp_path / "p" / "e").exists()   # nothing ran, nothing written
    assert stack.CPU_ESCAPE == "exit 3 (`--mode off --device cpu` runs the stock on the cpu)" and "the levers run on CUDA" in stack.DEVICE_CPU_REFUSAL
    if not stack.gpu_info().get("name"):                                          # this box has no CUDA device: the default device is refused by the box's name
        rc, _, r = run_package(cmd, base, gate_ok=True)
        assert rc == 3 and f"[progen2-opt] NOT ACTIVE: {stack.NO_CUDA_REFUSAL} (mode=exact variant=small route=score); {stack.CPU_ESCAPE}" in r.stderr.splitlines(), r.stderr[-1200:]


def test_pins_gate_notes_a_stack_off_the_pins_and_refuses_a_stock_off_the_pinned_commit(fake_stock, tmp_path, monkeypatch):
    """The gate unpatched. A stack (interpreter / torch / transformers / tokenizers) off stock/PINS.json `pins` is a NOTE — the stack line's
    words, never a refusal: the levers were tested on the pinned stack and engage wherever their mechanism applies. A stock dir whose files
    are not the pinned commit's bytes (stock/PINS.json stock_files.files digests, hashed by stock/check_pins.py's own check) is THE refusal, by name
    and digest, on every mode (it changes what stock means) — exit 3, no override."""
    from progen2_opt import stack
    p = stack.pins()
    monkeypatch.setattr(stack, "stack_versions", lambda: {"python": "3.13.1", "torch": "9.9.9", "transformers": "0.0.1", "tokenizers": "0.0.2"})
    ok, detail, why = stack.pins_gate(p)                                          # the tree's own stock: the pinned bytes
    assert ok and why is None, why
    assert detail["mismatch"] == ["python have 3.13.1 want 3.9", "torch have 9.9.9 want 2.8.0+cu128", "transformers have 0.0.1 want 4.16.2", "tokenizers have 0.0.2 want 0.10.3"], detail["mismatch"]
    assert stack.STACK_NOTE.format(detail="; ".join(detail["mismatch"])) == ("stack differs from the pinned one (python have 3.13.1 want 3.9; torch have 9.9.9 want 2.8.0+cu128; "
                                                                              "transformers have 0.0.1 want 4.16.2; tokenizers have 0.0.2 want 0.10.3): the levers were tested on the pinned stack")
    assert set(detail["files"]) == set(stack.stock_file_pins(p)) and all(v == "ok" for v in detail["files"].values()) and 0.0 <= detail["hash_ms"] < 2000.0
    sd = _fake_stock_with_scripts(fake_stock)
    monkeypatch.setenv("PROGEN2_STOCK_DIR", str(sd))
    ok, detail, why = stack.pins_gate(p)                                          # the fake stock: its files are not the pinned commit's
    assert not ok and why.startswith("stock files differ from the pinned commit (sample.py: ") and "!= 6451430c" in why and why.endswith(") — this is not the stock the modes are defined against"), why
    assert detail["files"]["sample.py"].endswith("!= 6451430c") and detail["files"]["likelihood.py"] != "ok"
    monkeypatch.undo()
    env = {"PROGEN2_STOCK_DIR": str(sd), "PROGEN2_WEIGHTS": str(tmp_path / "weights"), "PROGEN2_RUN_DIR": str(tmp_path / "run")}
    items = _items(tmp_path, "items.jsonl", [{"item_id": "ubq", "context": "1MQIF2"}])
    for mode in ("off", "exact"):                                                 # through the verbs, unpatched: every mode refuses it, exit 3, the files named, the restore named
        rc, out, _ = run_package(["score", "--model", "progen2-small", "--mode", mode, "--input", str(items), "--out_dir", str(tmp_path / "p" / mode)], env)
        assert rc == 3 and "[progen2-opt] NOT ACTIVE: stock files differ from the pinned commit (sample.py: " in out and f"(mode={mode} variant=small route=score); " + stack.STOCK_FILES_ESCAPE in out, out[-1200:]
        assert not (tmp_path / "p" / mode / "items" / "ubq").exists()               # nothing ran


def test_run_verb_is_gone():
    rc, out, _ = run_package(["run", "--model", "progen2-small", "--mode", "off", "--route", "score", "--item", "u", "--out_dir", "o"])
    assert rc == 2 and "unknown command 'run'" in out, out[-400:]
