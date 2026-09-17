"""Mode off is stock: the launch strips every kit variable and cleans PYTHONPATH; the stock runner proves its own environment before
importing anything upstream (no forbidden variable, no kit module, no kit directory), prints the two contract lines, and then runs the
upstream CLI's __main__ block — with the stub upstream the whole path runs on CPU end to end; an injected variable makes it refuse (3)."""
import os
import subprocess
import sys

from e1_opt import report, stack, stock_score
from e1_opt.tests import _stubs


def test_clean_env_strips_every_prefix_and_kit_paths(tmp_path):
    polluted = {"E1_OPT": "exact", "E1_VARIANT": "300m", "E1_OPT_DET": "1", "E1_KIT_WARMUP": "0", "MODEL_OPT": "/x", "MODEL_OPT_STACK_KEY": "k",
                "MODEL_OPT_TARGET_GPU": "H100", "HF_HOME": "/hf", "TRITON_CACHE_DIR": "/jit", "PATH": "/bin",
                "PYTHONPATH": os.pathsep.join(["/pkg", "/x/e1/opt/forward", "/y/engines/e1/kits", "/root"])}
    env, unset = stock_score.clean_env(polluted)
    assert unset == sorted(["E1_OPT", "E1_VARIANT", "E1_OPT_DET", "E1_KIT_WARMUP", "MODEL_OPT", "MODEL_OPT_STACK_KEY", "MODEL_OPT_TARGET_GPU"])
    assert not any(k.startswith(stack.MUST_BE_ABSENT_PREFIXES) for k in env)
    assert env["HF_HOME"] == "/hf" and env["TRITON_CACHE_DIR"] == "/jit" and env["PYTHONPATH"] == os.pathsep.join(["/pkg", "/root"])
    env, _ = stock_score.clean_env({"PYTHONPATH": "/x/opt/forward"})
    assert "PYTHONPATH" not in env


def test_env_proof_detects_each_violation():
    ok = stock_score.env_proof({"HF_HOME": "/hf"}, {"os": None, "E1": None, "e1_opt": None}, ["/site", "/pkg"])
    assert ok["ok"] and ok["kit_modules"] == [] and ok["kit_dirs"] == [] and ok["present"] == []
    assert stock_score.env_proof({"E1_OPT_DET": "1"}, {}, [])["present"] == ["E1_OPT_DET"]
    assert stock_score.env_proof({}, {"engines": None, "engines.e1.kits.pins": None, "e1_opt.kit_score": None}, [])["kit_modules"] == \
        ["engines", "engines.e1.kits.pins"]
    assert stock_score.env_proof({}, {}, ["/a/e1/opt/forward", "/b"])["kit_dirs"] == ["/a/e1/opt/forward"]
    assert stock_score.env_check_lines({"E1_OPT": "exact", "HF_HOME": "/hf", "MODEL_OPT": "/m"}) == ["E1_OPT=exact", "MODEL_OPT=/m"]
    assert stock_score.env_check_lines({"HF_HOME": "/hf"}) == []
    line = stock_score.env_clean_line(ok)
    from e1_opt import report
    assert report.RE_ENV_CLEAN.match(line)


def test_stock_command_form():
    cmd = stock_score.command("/usr/bin/python3", "300m", 1, "/t/pins.py", ["--model-name", "/snap", "--parent-path", "p", "--mutants-path", "m", "--output-path", "o"])
    assert cmd[:4] == ["/usr/bin/python3", "-s", "-m", "e1_opt.stock_score"] and "--det" in cmd and cmd[cmd.index("--") + 1:] == \
        ["--model-name", "/snap", "--parent-path", "p", "--mutants-path", "m", "--output-path", "o"]


def _run_stock(box, tmp_path, extra_env=None, det=0):
    items = _stubs.write_items(str(tmp_path / "in"), names=("k1",))
    it = os.path.join(items, "k1")
    out = tmp_path / "out"
    out.mkdir()
    cmd = stock_score.command(sys.executable, box["variant"], det, stack.pins_path(),
                              ["--model-name", box["caches"]["snapshot"], "--parent-path", os.path.join(it, "parent.fasta"),
                               "--mutants-path", os.path.join(it, "mutants.fasta"), "--output-path", str(out / "scores.csv")])
    env = _stubs.child_env(box, **(extra_env or {}))
    env.pop("MODEL_OPT", None)
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120)
    return p, out / "scores.csv"


def test_stock_runner_end_to_end_proves_env_and_runs_the_upstream_main(monkeypatch, tmp_path):
    box = _stubs.setup_box(monkeypatch, tmp_path)
    p, csv = _run_stock(box, tmp_path)
    lines = p.stdout.splitlines()
    assert p.returncode == 0, p.stdout + p.stderr
    assert lines[0] == "[e1-opt stock] ENV-CLEAN ok: absent=E1_OPT*,E1_KIT*,E1_VARIANT,MODEL_OPT* kit_modules=none kit_dirs=none", lines
    assert lines[1] == "STOCK_ENV_CHECK:" and len(lines) == 3, lines            # nothing after the header: the environment holds no kit variable; then the KERNELS proof
    k = report.RE_KERNELS.match(lines[2])                                      # the proof read the stub upstream's BOUND objects at the tool's first scorer construction
    assert k and k.group("runner", "route", "site", "upstream_says") == ("stock", "stock", "stock", "True") and "require=" not in lines[2], lines[2]
    assert k.group("flash_attn") == f"engaged:flash_attn@{box['pins'].STACK['flash_attn']}" and k.group("hub_layernorm") == f"engaged:triton_layer_norm@{box['pins'].KERNEL['rev'][:8]}" \
        and k.group("flex_attention") == f"engaged:torch.compile@{box['pins'].STACK['torch_version_str']}", lines[2]
    assert csv.is_file() and csv.read_text().splitlines()[0] == "id,context_id,score" and len(csv.read_text().splitlines()) == 4


def test_stock_runner_refuses_an_injected_kit_variable(monkeypatch, tmp_path):
    box = _stubs.setup_box(monkeypatch, tmp_path)
    p, csv = _run_stock(box, tmp_path, extra_env={"E1_OPT": "exact", "E1_KIT_WARMUP": "0"})
    assert p.returncode == 3 and p.stdout.splitlines()[0].startswith("[e1-opt stock] ENV-CLEAN FAILED: present=E1_KIT_WARMUP,E1_OPT")
    assert "E1_OPT=exact" in p.stdout.splitlines() and not csv.exists()


def test_stock_runner_refuses_a_kit_dir_on_pythonpath(monkeypatch, tmp_path):
    box = _stubs.setup_box(monkeypatch, tmp_path)
    p, csv = _run_stock(box, tmp_path, extra_env={"PYTHONPATH": box["site"] + os.pathsep + _stubs.OPT_DIR + os.pathsep + stack.forward_root()})
    assert p.returncode == 3 and "kit_dirs=" in p.stdout.splitlines()[0] and "forward" in p.stdout.splitlines()[0]


def test_arm_pin_applies_the_kits_pin_once_at_scorer_construction(monkeypatch, tmp_path):
    box = _stubs.setup_box(monkeypatch, tmp_path)
    calls = []

    class FakePins:
        @staticmethod
        def apply_autotune_pin(size):
            calls.append(size)
            return {"num_warps": 0}
    import E1.scorer as SC
    orig = SC.E1Scorer.__init__
    try:
        stock_score.arm_pin(FakePins, box["variant"])
        from E1.modeling import E1ForMaskedLM
        m = E1ForMaskedLM.from_pretrained("x")
        SC.E1Scorer(m, method="masked_marginal")
        SC.E1Scorer(m, method="masked_marginal")
        assert calls == [box["variant"]]
    finally:
        SC.E1Scorer.__init__ = orig


def test_load_pins_by_path_binds_no_engines_module(monkeypatch, tmp_path):
    _stubs.tree_or_skip()
    before = {m for m in __import__("sys").modules if m.startswith(("engines", "compare"))}
    pins = stock_score.load_pins_by_path(stack.pins_path())
    assert hasattr(pins, "apply_autotune_pin") and hasattr(pins, "DET_RECIPE")
    after = {m for m in __import__("sys").modules if m.startswith(("engines", "compare"))}
    assert after == before
