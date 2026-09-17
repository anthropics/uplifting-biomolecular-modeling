"""The command layer and the process forms without a GPU: the three verbs (design | check | warm) and their exit codes (0 ok, 1 failed,
2 usage, 3 not active); the mode/env disagreement rule; `check` (dry run) lines; a kit mode without `--seed` is not active before
configure; `--pack` is not an option; the runner / stock-step command lines and environments built from the
mode table and the shared core's stock-proof contract; the lever evidence parsed from the kit modules' own lines; the exit rule
(`exit_for`: the accelerator census is a record, never an exit); `warm` follows `design`."""
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

from opt_core import stock_proof

from boltzgen_opt import census, cli, codes, design, kernels, manifest, modes, report, stack, stock_design
from boltzgen_opt.tests import _stubs

HOME = stack.opt_home()
RESOLUTION_ON = {"use_kernels": True, "cc": [9, 0], "line": "Using kernels: True [device capability: (9, 0)]"}
RESOLUTION_OFF = {"use_kernels": False, "cc": [9, 0], "line": "Using kernels: False [device capability: (9, 0)]"}


def run_cli(args, env):
    r = subprocess.run([sys.executable, "-m", "boltzgen_opt"] + args, env=env, capture_output=True, text=True, timeout=300)
    return r.returncode, r.stdout, r.stderr


class TestCli(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="bgopt_cli_")
        cls.site = _stubs.materialize(cls.tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def dry(self, mode, **env):
        """stack.activate(mode, dry_run=True) in a fresh interpreter on the stubs: the report `check` formats its DRY-RUN line from."""
        code = "import json, sys; from boltzgen_opt import stack; print('REPORT ' + json.dumps(stack.activate(sys.argv[1], dry_run=True), default=str))"
        rc, so, se = _stubs.run_py(code, _stubs.clean_env(self.site, **env), args=[mode])
        self.assertEqual(rc, 0, se)
        return next(json.loads(ln[7:]) for ln in so.splitlines() if ln.startswith("REPORT ")), se

    def test_usage_unknown_and_the_verb_set(self):
        self.assertEqual(cli.main([]), cli.EXIT_USAGE)
        self.assertEqual(cli.main(["--help"]), cli.EXIT_OK)
        self.assertEqual(cli.main(["nope"]), cli.EXIT_USAGE)
        self.assertEqual(set(cli.COMMANDS), {"design", "check", "warm"})            # the package's verbs (run.sh adds `install`)
        for word in ("serve", "selftest"):                                          # anything outside the three is an unknown command: usage, exit 2
            rc, so, se = run_cli([word, "--mode", "exact"], _stubs.clean_env(self.site))
            self.assertEqual(rc, cli.EXIT_USAGE, se)
            self.assertIn(f"unknown command {word!r}", se)

    def test_exit_codes_are_0_1_2_3(self):
        self.assertEqual((codes.EXIT_OK, codes.EXIT_FAIL, codes.EXIT_USAGE, codes.EXIT_NOT_ACTIVE), (0, 1, 2, 3))
        self.assertEqual((cli.EXIT_OK, cli.EXIT_FAIL, cli.EXIT_USAGE, cli.EXIT_NOT_ACTIVE), (0, 1, 2, 3))
        for name in ("EXIT_OOM", "EXIT_KERNELS", "BIG_OOM_CHILD_RC"):                # no fourth condition: an OOM is the run's failure (1), the census is a record
            self.assertFalse(hasattr(codes, name), name)
            self.assertFalse(hasattr(cli, name), name)
        public = sorted(k for k, v in vars(codes).items() if k.startswith("EXIT_"))
        self.assertEqual(public, ["EXIT_FAIL", "EXIT_NOT_ACTIVE", "EXIT_OK", "EXIT_USAGE"])

    def test_mode_env_disagreement_and_fast(self):
        rc, so, se = run_cli(["check", "--mode", "off"], _stubs.clean_env(self.site, BOLTZGEN_OPT="exact"))
        self.assertEqual(rc, 2)
        self.assertIn("--mode off disagrees with BOLTZGEN_OPT=exact", se)
        rc, so, se = run_cli(["check", "--mode", "fast"], _stubs.clean_env(self.site))
        self.assertEqual(rc, 0, se)
        self.assertIn("[boltzgen-opt] DRY-RUN mode=fast form=inproc switches=BG_GRAPH=graph,XA_FAST_INIT=1,XA_HOIST=1,HL_ASYNC_WRITER=1,FL_COND_DEDUP=1,FL_ATTN_BF16=1,FL_ATTN_BACKEND=cudnn,FL_DIT_FUSED=1 ", se)
        rc, so, se = run_cli(["check", "--mode", "nope"], _stubs.clean_env(self.site))
        self.assertEqual(rc, 2)
        rc, so, se = run_cli(["check", "--mode", "exact"], _stubs.clean_env(self.site, BOLTZGEN_OPT="exact"))
        self.assertEqual(rc, 0, se)

    def test_check_takes_only_mode(self):
        """`check [--mode M]` is the whole surface: any other option is a usage error (exit 2), never silently ignored."""
        for extra in (["--json"], ["--variant", "tz"], ["--settings", "upstream"], ["--quiet"]):
            rc, so, se = run_cli(["check", "--mode", "exact", *extra], _stubs.clean_env(self.site))
            self.assertEqual(rc, 2, (extra, se[-300:]))
            self.assertIn("unrecognized arguments", se)

    def test_check_dry_run_lines(self):
        rc, so, se = run_cli(["check"], _stubs.clean_env(self.site))                       # no --mode, no BOLTZGEN_OPT: the package default
        self.assertEqual(rc, 0, se)
        self.assertIn("[boltzgen-opt] DRY-RUN mode=fast form=inproc switches=BG_GRAPH=graph,XA_FAST_INIT=1,XA_HOIST=1,HL_ASYNC_WRITER=1,FL_COND_DEDUP=1,FL_ATTN_BF16=1,FL_ATTN_BACKEND=cudnn,FL_DIT_FUSED=1 ", se)
        rep, _ = self.dry("fast")
        self.assertEqual((rep["mode"], rep["imports"][-3:]), ("fast", ["fl_levers", "sz_levers", "hl_levers"]))
        rc, so, se = run_cli(["check", "--mode", "exact"], _stubs.clean_env(self.site))
        self.assertEqual(rc, 0, se)
        self.assertIn("[boltzgen-opt] DRY-RUN mode=exact form=inproc switches=BG_GRAPH=graph,XA_FAST_INIT=1,XA_HOIST=1,HL_ASYNC_WRITER=1 boltzgen=0.3.2 gpu=NVIDIA H100 80GB HBM3(81559MiB)", se)
        self.assertEqual(so, "")                                                            # nothing on stdout: the report is the stderr line (no --json form)
        rep, _ = self.dry("exact")
        self.assertEqual((rep["active"], rep["dry_run"], rep["gpu_gate"], rep["would_export"]), (False, True, "ok", {"BG_GRAPH": "graph", "XA_FAST_INIT": "1", "XA_HOIST": "1", "HL_ASYNC_WRITER": "1"}))
        self.assertEqual(rep["imports"], ["bg_hook", "xa_fastinit", "xa_hoist", "sz_levers", "hl_levers"])   # the vendor imports, then the mode's own: the out-of-memory trace (sz_levers, every kit mode) and the background writer
        self.assertNotIn("variant", rep); self.assertNotIn("worker", rep)
        rc, so, se = run_cli(["check"], _stubs.clean_env(self.site, gpu="Tesla V100-SXM2-32GB,32768,7.0"))   # below the tested floor (8.0): admitted, the NOTE says so — the card is never a refusal
        self.assertEqual(rc, 0, se)
        self.assertIn("capability sm_70 is below the tested floor 8.0; proceeding", se); self.assertNotIn("would-refuse", se)
        rc, so, se = run_cli(["check"], _stubs.clean_env(self.site, gpu=""))                                  # no GPU visible: the one card condition a kit mode cannot activate on
        self.assertEqual(rc, 3)
        self.assertIn("would-refuse: no GPU visible", se)
        rc, so, se = run_cli(["check", "--mode", "exact"], _stubs.clean_env(self.site, gpu="NVIDIA A100-SXM4-80GB,81920,8.0"))   # another card at the floor: admitted, one NOTE line, gpu_gate noted
        self.assertEqual(rc, 0, se)
        self.assertIn("[boltzgen-opt] NOTE gpu=NVIDIA A100-SXM4-80GB mem=81920MiB differs from the pinned card NVIDIA H100 80GB HBM3 (81559MiB, stock/PINS.json); proceeding (capability sm_80)", se.splitlines())
        self.assertIn("[boltzgen-opt] DRY-RUN mode=exact form=inproc switches=BG_GRAPH=graph,XA_FAST_INIT=1,XA_HOIST=1,HL_ASYNC_WRITER=1 boltzgen=0.3.2 gpu=NVIDIA A100-SXM4-80GB(81920MiB)", se)
        rep, _ = self.dry("exact", gpu="NVIDIA A100-SXM4-80GB,81920,8.0")
        self.assertEqual(rep["gpu_gate"], "noted")
        rc, so, se = run_cli(["check", "--mode", "off"], _stubs.clean_env(self.site, gpu=""))
        self.assertEqual(rc, 0)
        self.assertIn("DRY-RUN mode=off", se)

    def test_check_big(self):
        rc, so, se = run_cli(["check", "--mode", "big"], _stubs.clean_env(self.site))
        self.assertEqual(rc, 0, se)
        self.assertIn("[boltzgen-opt] DRY-RUN mode=big form=inproc switches=BG_GRAPH=off,XA_FAST_INIT=1,XA_HOIST=0,SZ_TD_CHUNK=64,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,HL_ASYNC_WRITER=1 ", se)
        rep, _ = self.dry("big")
        self.assertEqual(rep["would_export"], {"BG_GRAPH": "off", "XA_FAST_INIT": "1", "XA_HOIST": "0", "SZ_TD_CHUNK": "64",
                                                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "HL_ASYNC_WRITER": "1"})
        self.assertEqual(rep["imports"], ["bg_hook", "xa_fastinit", "xa_hoist", "sz_levers", "hl_levers"])
        self.assertEqual(rep["levers_planned"], ["inproc", "fastinit", "td_chunk", "async_writer"])
        self.assertNotIn("variant", rep)

    def test_a_kit_mode_without_seed_is_not_active_before_configure(self):
        """`design --mode exact` (fast, big alike) without `--seed`: the kits' runner is seeded by construction and no seed is invented —
        NOT ACTIVE naming `--seed`, exit 3, the manifest's reason, and nothing configured; `--mode off` without a seed is upstream's own
        unseeded run (test_stock_route)."""
        spec = os.path.join(self.tmp, "s_seed.yaml"); open(spec, "w").write("entities: []\n")
        for mode in ("exact", "fast", "big"):
            out = os.path.join(self.tmp, f"noseed_{mode}")
            rc, so, se = run_cli(["design", "--mode", mode, spec, "--output", out, "--num_designs", "1"], _stubs.clean_env(self.site))
            self.assertEqual(rc, 3, se)
            self.assertIn(f"[boltzgen-opt] NOT ACTIVE: mode {mode} runs the kits' seeded runner and needs --seed N (step i runs with N + i); --mode off runs upstream unseeded; exit 3", se.splitlines())
            self.assertEqual(cli.seed_refusal(mode), f"mode {mode} runs the kits' seeded runner and needs --seed N (step i runs with N + i); --mode off runs upstream unseeded; exit 3")
            self.assertFalse(os.path.exists(os.path.join(out, "steps.yaml")))              # refused before configure
            self.assertFalse(os.path.exists(os.path.join(out, "opt_configure.log")))
            man = manifest.read(out)
            self.assertEqual((man["active"], man["exit_code"], man["mode"], man["command"]), (False, 3, mode, "design"))
            self.assertEqual(man["activation_report"]["reason"], cli.seed_refusal(mode))
        out = os.path.join(self.tmp, "noseed_default")                                # no --mode, no BOLTZGEN_OPT: the default mode (fast) — the same refusal, naming fast
        rc, so, se = run_cli(["design", spec, "--output", out, "--num_designs", "1"], _stubs.clean_env(self.site))
        self.assertEqual(rc, 3, se)
        self.assertIn(f"[boltzgen-opt] NOT ACTIVE: {cli.seed_refusal(modes.DEFAULT_MODE)}", se.splitlines())
        self.assertEqual((modes.DEFAULT_MODE, manifest.read(out)["mode"]), ("fast", "fast"))

    def test_design_refuses_without_a_gpu_before_configure(self):
        out = os.path.join(self.tmp, "wrong")
        spec = os.path.join(self.tmp, "s.yaml"); open(spec, "w").write("entities: []\n")
        rc, so, se = run_cli(["design", "--mode", "exact", spec, "--output", out, "--seed", "1", "--num_designs", "1"], _stubs.clean_env(self.site, gpu=""))   # no GPU visible
        self.assertEqual(rc, 3, se)
        self.assertIn("NOT ACTIVE: no GPU visible", se)
        self.assertFalse(os.path.exists(os.path.join(out, "steps.yaml")))
        man = manifest.read(out)
        self.assertEqual((man["active"], man["exit_code"], man["mode"]), (False, 3, "exact"))

    def test_designs_surface_is_its_own_options_plus_upstreams(self):
        """`design` defines <spec>, --output, --mode, --seed, --cache, --devices; upstream's `run` options (cli.UPSTREAM_OPTIONS)
        are accepted as given; an option neither defines — `--variant`, `--quiet`, `--pack`, yesterday's `--spec`/`--out` — is a usage error
        (exit 2) naming it before anything runs; a kit mode asked for `--devices 2` is not active by name (the kits' runner drives one GPU)."""
        spec = os.path.join(self.tmp, "s_extra.yaml"); open(spec, "w").write("entities: []\n")
        for extra in (["--variant", "tz"], ["--quiet"], ["--pack", "2"], ["--out", os.path.join(self.tmp, "o")], ["--spec", spec], ["--settings", "upstream"]):
            out = os.path.join(self.tmp, "extra_" + extra[0].lstrip("-"))
            rc, so, se = run_cli(["design", "--mode", "off", spec, "--output", out, *extra], _stubs.clean_env(self.site))
            self.assertEqual(rc, 2, (extra, se[-300:])); self.assertIn(f"design: no such option {extra[0]} ", se)
            self.assertFalse(os.path.exists(out), extra)
        rc, so, se = run_cli(["design", "--mode", "off", spec, "--output", os.path.join(self.tmp, "nosteps"), "--steps"], _stubs.clean_env(self.site))
        self.assertEqual(rc, 2); self.assertIn("upstream's option --steps takes one or more values", se)
        out = os.path.join(self.tmp, "extra_devices")
        rc, so, se = run_cli(["design", "--mode", "exact", spec, "--output", out, "--seed", "1", "--devices", "2"], _stubs.clean_env(self.site))
        self.assertEqual(rc, 3, se[-300:]); self.assertIn("NOT ACTIVE: mode exact runs the kits' runner on one GPU and was asked for --devices 2", se)
        self.assertFalse(os.path.exists(os.path.join(out, "steps.yaml")))
        rc, so, se = run_cli(["design", "--help"], _stubs.clean_env(self.site))
        self.assertEqual(rc, 0)
        own = sorted(set(re.findall(r"^  (--[a-z-]+)", so, re.M)))
        self.assertEqual(own, ["--cache", "--devices", "--mode", "--output", "--seed"], so)

    def test_upstream_option_table_is_the_vendored_wheels(self):
        """cli.UPSTREAM_OPTIONS (name -> arity) is upstream 0.3.2's `run`/`configure` option set, read from the vendored wheel's parser source:
        every option of add_configure_arguments, add_models_download_options and the configure parser's --steps, less the three `design`
        reads itself (--output, --cache, --devices); arity "+" where upstream says nargs="+", 0 for store_true flags, else 1."""
        import zipfile
        whl = os.path.join(stack.tree_home(), stack.pins()["wheel"]["file"])                  # stock/PINS.json names the vendored wheel
        src = zipfile.ZipFile(whl).read("boltzgen/cli/boltzgen.py").decode()
        def block(name):
            i = src.index(f"def {name}("); j = src.find("\ndef ", i + 5); return src[i:j]
        want = {}
        for fn in ("add_configure_arguments", "add_models_download_options", "build_configure_parser"):
            for m in re.finditer(r'add_argument\(\s*"(--[\w-]+)"(.*?)\)\s*\n', block(fn), re.S):
                rest = m.group(2)
                want[m.group(1)] = "+" if 'nargs="+"' in rest else (0 if "store_true" in rest else 1)
        for k in cli.KIT_HANDLED_UPSTREAM_OPTIONS:
            want.pop(k, None)
        self.assertEqual(cli.UPSTREAM_OPTIONS, want)
        self.assertFalse(hasattr(modes, "VARIANTS")); self.assertFalse(hasattr(modes, "ENV_VARIANT")); self.assertFalse(hasattr(modes, "route_refusal"))
        pkg = os.path.dirname(os.path.abspath(cli.__file__))
        self.assertEqual(sorted(f[:-3] for f in os.listdir(pkg) if f.endswith(".py")),   # the package's modules, exactly: no settings / serve / self-test module
                         ["__init__", "__main__", "_autoload", "_core_gate", "census", "cli", "codes", "design", "kernels", "manifest", "modes", "registry", "report", "stack", "stock_design", "warm", "weights"])   # weights: the install step's --weights DIR (run.sh install), not a verb
        self.assertFalse(os.path.isdir(os.path.join(HOME, "serving")))                   # no worker kit, no packed form


class TestProcessForms(unittest.TestCase):
    def setUp(self):
        self.res = modes.resolve("exact", HOME)
        self.tmp = tempfile.mkdtemp(prefix="bgopt_forms_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_runner_line(self):
        cmd = design.runner_cmd(self.res, "/r/job", 5)
        self.assertEqual(cmd, [sys.executable, os.path.join(HOME, "forward", "xattempt_addon", "src", "xa_run.py"), "/r/job", "5"])
        self.assertEqual(design.runner_cmd(self.res, "/r/job", 5, ["design", "inverse_folding"])[-1], "design,inverse_folding")
        fast = modes.resolve("fast", HOME)                                          # every kit mode launches the same runner
        self.assertEqual(design.runner_cmd(fast, "/r/job", 5), cmd)
        self.assertFalse(hasattr(design, "packer_cmd")); self.assertFalse(hasattr(design, "run_packed"))   # no packed form

    def test_stock_step_command_is_the_cores_contract_plus_the_census_option(self):
        """The stock child's argv IS opt_core.stock_proof.stock_command(...) with `-s -m <module>` spelled `-S -s -c <preamble>` (this tree's
        package and core first on sys.path, imported before site: design.STOCK_PREAMBLE) — nothing of the kit inserted (no census payload: a
        stock process runs upstream alone after its proof); the stock arguments are `<seed|none> <config>`."""
        se = stack.pins()["stock_environment"]
        for seed, word in ((5, "5"), (None, "none")):
            cmd = design.stock_step_cmd(seed, "/r/c.yaml", "/r/proof.json")
            want = stock_proof.stock_command(sys.executable, "boltzgen_opt.stock_design", proof_json="/r/proof.json",
                                             env_absent=list(se["must_be_absent_prefixes"]) + list(se["must_be_absent_names"]),
                                             kit_dirs=[stack.kit_dir(k) for k in modes.KITS], args=[word, "/r/c.yaml"], module_prefixes=stack.KIT_MODULE_PREFIXES)
            self.assertEqual(want[1:4], ["-s", "-m", "boltzgen_opt.stock_design"])
            self.assertEqual(cmd, want[:1] + ["-S", "-s", "-c", design.stock_preamble()] + want[4:])   # the contract's line, its module started by the preamble (design.STOCK_PREAMBLE: this tree's copies before site)
            self.assertNotIn("--kernels", cmd)

    def test_kit_env_is_the_mode_tables(self):
        env, dropped = design.kit_env(self.res, "/r/job", {"PATH": "/bin", "XA_FAST_INIT": "0", "BOLTZGEN_OPT": "exact", "HOME": "/h", "SZ_TD_CHUNK": "8", "FL_COND_DEDUP": "1", "HL_WRITER_WORKERS": "9"})
        self.assertEqual(dropped, ["FL_COND_DEDUP", "HL_WRITER_WORKERS", "SZ_TD_CHUNK", "XA_FAST_INIT"])   # every caller-set kit switch (BG_/XA_/SZ_/FL_/HL_) is dropped and reported; exact hands the mode over, one-shot, rather than dropping it
        self.assertEqual({k: env[k] for k in ("BG_GRAPH", "XA_FAST_INIT", "XA_HOIST", "HF_HUB_OFFLINE", "PYTHONUNBUFFERED")}, {"BG_GRAPH": "graph", "XA_FAST_INIT": "1", "XA_HOIST": "1", "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1"})
        self.assertEqual(env["BG_TIMING_FILE"], os.path.join("/r/job", "opt_timing.jsonl"))
        self.assertEqual(env["PYTHONPATH"].split(os.pathsep), [stack.kit_src(modes.KIT_XATTEMPT), stack.kit_src(modes.KIT_PARTNER), stack.kit_src(modes.KIT_SIZE_LEVERS), stack.kit_src(modes.KIT_HOST_LEVERS)] + stack.package_roots())
        self.assertEqual({k: env.get(k) for k in ("BOLTZGEN_OPT", "BOLTZGEN_OPT_HANDOVER")}, {"BOLTZGEN_OPT": "exact", "BOLTZGEN_OPT_HANDOVER": "1"})
        self.assertNotIn("BOLTZGEN_OPT_VARIANT", env)
        p = census.parse_payload(env[census.ENV_KERNELS])                          # the census armed in the child: the mode, its route, what the route expects, the job's item; no seed word for an unseeded call
        self.assertEqual((p["mode"], p["route"], p["expect"], p["item"], "seed" in p, "settings" in p), ("exact", "exact", "on", "job", False, False))   # a bare caller (no expectation handed in) arms the child for kernels on
        env, _ = design.kit_env(self.res, "/r/job", {"PATH": "/bin"}, exp=design.kernel_expectation("exact", "/r/job"), seed=4)   # the running verbs hand the run's expectation in: no configure log under /r/job -> upstream printed no resolution -> unknown
        p = census.parse_payload(env[census.ENV_KERNELS])
        self.assertEqual((p["expect"], p["seed"]), ("unknown", "4"))
        env, _ = design.kit_env(self.res, "/r/job", {"PATH": "/bin"}, exp={"expect": "on", "words": {}}, seed=5)
        self.assertEqual((census.parse_payload(env[census.ENV_KERNELS])["expect"], census.parse_payload(env[census.ENV_KERNELS])["seed"]), ("on", "5"))
        env, _ = design.kit_env(self.res, "/r/job", {"PATH": "/bin"}, arm=False)
        self.assertNotIn(census.ENV_KERNELS, env)                                   # a CPU-steps-only run arms nothing
        senv, sdropped = design.stock_env({"PATH": "/bin", "XA_FAST_INIT": "0", "BOLTZGEN_OPT": "off", "PYTHONPATH": "/k", "CUDA_MPS_PIPE_DIRECTORY": "/p", "BOLTZGEN_OPT_HOME": "/h", "BOLTZGEN_OPT_KERNELS": "x"})
        self.assertEqual(sdropped, ["BOLTZGEN_OPT", "BOLTZGEN_OPT_HOME", "BOLTZGEN_OPT_KERNELS", "PYTHONPATH", "XA_FAST_INIT"])   # the names of stock/PINS.json read as prefixes, as the child's proof reads them: nothing under BOLTZGEN_OPT* reaches the stock arm
        self.assertEqual(senv["HF_HUB_OFFLINE"], "1")
        self.assertNotIn("BOLTZGEN_OPT_HOME", senv); self.assertEqual((senv["PATH"], senv["CUDA_MPS_PIPE_DIRECTORY"]), ("/bin", "/p"))   # a caller's own CUDA MPS setting is not a kit name: it reaches the stock arm as it reaches upstream
        se = stack.pins()["stock_environment"]
        self.assertEqual(design.stock_env({"XA_A": "1", "PYTHONPATH": "/k", "KEEP": "1"})[0],
                         dict(stock_proof.strip_env({"XA_A": "1", "PYTHONPATH": "/k", "KEEP": "1"}, list(se["must_be_absent_prefixes"]) + list(se["must_be_absent_names"]))[0], HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1"))   # the core's strip over the PINS lists, plus the two names the stock arm always carries
        self.assertEqual(set(se["must_be_absent_prefixes"]), set(modes.KIT_SWITCH_PREFIXES))   # the stock arm's must-be-absent prefixes are the kits' switch prefixes (modes.py): one table, no drift
        self.assertEqual(set(se["kit_module_prefixes"]) - {"sitecustomize"}, set(stack.KIT_MODULE_PREFIXES))
        self.assertEqual(set(se["must_be_absent_names"]), {modes.ENV, "PYTHONPATH", census.ENV_KERNELS})

    def test_configure_arguments_are_the_callers_verbatim(self):
        """No presets: `--cache $BOLTZGEN_CACHE` only when the caller named none, then the caller's own configure arguments as given."""
        self.assertEqual(design.configure_args("/hf", ["--protocol", "protein-anything", "--use_kernels", "false"]), ["--cache", "/hf", "--protocol", "protein-anything", "--use_kernels", "false"])
        self.assertEqual(design.configure_args("/hf", ["--cache", "/mine", "--budget", "5"]), ["--cache", "/mine", "--budget", "5"])
        self.assertEqual(design.configure_args(None, []), [])
        cmd = design.configure_cmd("/s.yaml", "/r/job", ["--config", "design", "compile_pairformer=true"], "/hf")
        self.assertEqual(cmd, [sys.executable, "-m", "boltzgen.cli.boltzgen", "configure", "/s.yaml", "--output", "/r/job", "--cache", "/hf", "--config", "design", "compile_pairformer=true"])
        self.assertFalse(hasattr(design, "SpecRefused")); self.assertFalse(hasattr(design, "refuse_unconsumed_spec_keys"))   # a spec's keys are upstream's to read: nothing of the spec is refused here

    def test_evidence_from_the_kits_own_lines(self):
        run_dir = os.path.join(self.tmp, "job")
        os.makedirs(run_dir)
        log = os.path.join(run_dir, "opt_run.log")
        open(log, "w").write("[bg_graph_patch] mode=graph\n[xa_fastinit] coverage OK: all 1234 parameters (+5 persistent buffers) present\n[xa_hoist] installed (with partner graph-sampler integration)\n"
                             "inproc step design wall_s=10.0 seed=0\n[xa_run] xa_fastinit stats: {'enabled': True}\n[xa_run] xa_hoist stats: {'enabled': True, 'builds': 2}\n")
        json.dump({"design": {"wall_s": 10.0, "seed": 0}}, open(os.path.join(run_dir, "inproc_times.json"), "w"))
        open(os.path.join(run_dir, "opt_timing.jsonl"), "w").write(json.dumps({"pid": 1, "acc": {"seed_fix_active": [1.0, 1]}}) + "\n")
        ev = design.evidence(self.res, run_dir, log)
        self.assertEqual(sorted(ev["levers_applied"]), sorted(["fastinit", "hoist", "graph_sampler", "inproc"]))
        self.assertEqual(ev["levers_fallback"], ["async_writer"])                # no async_writer marker line in this synthetic log
        self.assertEqual(ev["kit_stats_lines"]["xa_hoist"], "{'enabled': True, 'builds': 2}")
        self.assertTrue(ev["partial"])                                           # async_writer's own fallback makes the run partial
        self.assertEqual(set(ev["kernels"]), {"ok", "findings", "lines", "words", "expected", "peak"})   # the census account rides the evidence: a record (`findings`), never an exit
        self.assertEqual((ev["kernels"]["ok"], len(ev["kernels"]["findings"])), (False, 1))              # one model process ran and this synthetic log has no KERNELS line: a finding
        open(log, "w").write("[xa_fastinit] disabled for this step (num_workers=0): featurizer RNG would share the main-process numpy stream\n[xa_hoist] DISABLED: no partner\n")
        os.remove(os.path.join(run_dir, "inproc_times.json")); os.remove(os.path.join(run_dir, "opt_timing.jsonl"))
        ev = design.evidence(self.res, run_dir, log)
        self.assertEqual(ev["levers_applied"], [])
        self.assertEqual(sorted(ev["levers_fallback"]), sorted(["async_writer", "fastinit", "hoist", "graph_sampler", "inproc"]))
        self.assertIn("DISABLED", ev["fallback_reasons"]["hoist"])
        self.assertTrue(ev["partial"])

    def test_graph_capture_failure_is_a_named_fallback(self):
        """bg_graph_patch.py l.152-154: a CUDA-graph capture error switches the sampler to eager predraw for the rest of the process, after
        its `mode=graph` line — the lever is a fallback by name (partial), never a silent pass."""
        run_dir = os.path.join(self.tmp, "job_capfail")
        os.makedirs(run_dir)
        log = os.path.join(run_dir, "opt_run.log")
        open(log, "w").write("[bg_graph_patch] mode=graph\n[xa_fastinit] coverage OK: all 10 parameters\n[xa_hoist] installed\n"
                             "[bg_graph_patch] CAPTURE FAILED -> predraw mode for the rest of the process:\nTraceback (most recent call last):\nRuntimeError: capture\n")
        json.dump({"design": {"wall_s": 10.0, "seed": 0}}, open(os.path.join(run_dir, "inproc_times.json"), "w"))
        open(os.path.join(run_dir, "opt_timing.jsonl"), "w").write(json.dumps({"pid": 1, "acc": {"seed_fix_active": [1.0, 1]}}) + "\n")
        ev = design.evidence(self.res, run_dir, log)
        self.assertEqual(ev["levers_fallback"], ["graph_sampler", "async_writer"])   # no async_writer marker line in this synthetic log either
        self.assertTrue(ev["fallback_reasons"]["graph_sampler"].startswith("[bg_graph_patch] CAPTURE FAILED"))
        self.assertTrue(ev["partial"])

    def test_process_report_shape(self):
        run = {"rc": 0, "levers_applied": ["fastinit"], "levers_fallback": ["hoist"], "fallback_reasons": {"hoist": "x"}, "partial": True, "dropped_env": [], "gpu_gate": "ok",
               "kernels": {"ok": False, "findings": ["route=exact step=design cueq_triatt=absent:x (expected engaged)"], "lines": [], "words": {}, "expected": {}}, "route": "exact"}
        rep = design.process_report(self.res, run, {"name": "NVIDIA H100 80GB HBM3", "memory_total_mib": 81559, "compute_cap": "9.0"})
        self.assertEqual((rep["active"], rep["form"], rep["mode"], rep["partial"]), (True, "process", "exact", True))
        self.assertEqual(rep["stack_key"], stack.stack_key({"compute_cap": "9.0"}))
        self.assertRegex(rep["stack_key"], r"^torch[0-9][0-9.]+-cu\w+-sm90$")
        self.assertEqual((rep["kernels"]["route"], rep["kernels"]["findings"]), ("exact", run["kernels"]["findings"]))   # the account rides the report, route attached
        self.assertNotIn("oom", rep)                                             # a run that never traced an OOM carries no such key
        self.assertTrue(design.process_report(self.res, dict(run, oom=True), None)["oom"])
        off = design.process_report(modes.resolve("off", HOME), {"rc": 0, "stock_env_proof": ["p"]}, None)
        self.assertEqual((off["active"], off["switches"], off["stock_env_proof"]), (False, "stock", ["p"]))
        self.assertTrue(off["reason"].startswith("mode off: stock"))

    def test_stack_key_is_the_installed_torch_and_the_visible_gpu(self):
        """The JIT cache key of the configs is keyed by the torch that is installed (metadata version + CUDA tag, no import) and the
        card that is visible; the pinned stack stands in only for what is absent."""
        import importlib.metadata as md
        code = "from boltzgen_opt.stack import stack_key; import sys; print(stack_key(), stack_key({'compute_cap': '8.0'}), 'torch' in sys.modules)"
        fake = os.path.join(self.tmp, "fake_dists")
        for name, version_py in (("torch-9.9.9+cu777.dist-info", None), ("torch-9.9.9.dist-info", "__version__ = '9.9.9'\ncuda = '12.8'\n")):
            d = os.path.join(fake, name.split(".dist-info")[0]); os.makedirs(os.path.join(d, name), exist_ok=True)
            open(os.path.join(d, name, "METADATA"), "w").write("Metadata-Version: 2.1\nName: torch\nVersion: %s\n" % name[len("torch-"):-len(".dist-info")])
            if version_py:
                os.makedirs(os.path.join(d, "torch"), exist_ok=True); open(os.path.join(d, "torch", "version.py"), "w").write(version_py)
        site = _stubs.materialize(os.path.join(self.tmp, "stack_key_site"))
        for sub, key in (("torch-9.9.9+cu777", "torch9.9.9-cu777"), ("torch-9.9.9", "torch9.9.9-cu128")):
            env = _stubs.clean_env(site)
            env["PYTHONPATH"] = os.path.join(fake, sub) + os.pathsep + env["PYTHONPATH"]         # the fake distribution is the first `torch` metadata found
            rc, so, se = _stubs.run_py(code, env)
            self.assertEqual(rc, 0, se)
            self.assertEqual(so.split(), [key + "-sm90", key + "-sm80", "False"], sub)            # the seam's card has no compute capability: the pinned card's; a card's own wins; no torch import
        rc, so, se = _stubs.run_py(code, _stubs.clean_env(site))
        self.assertEqual(rc, 0, se)
        self.assertTrue(so.startswith("torch" + md.version("torch").partition("+")[0] + "-cu"), so)  # this interpreter's own torch


class TestExitFor(unittest.TestCase):
    """exit_for()'s table, direct and fast (no subprocess): the child's rc 3 is 3; any other non-zero rc is 1; a partial activation (a
    lever of the mode could not run) is 3 with one NOT ACTIVE line — a mode is all of its levers, no opt-out; else 0. The accelerator
    census (rep["kernels"]) and an OOM trace (rep["oom"]) are records and never change the exit."""

    def _quiet(self, fn):
        import contextlib, io
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = fn()
        return rc, err.getvalue()

    def test_the_table(self):
        self.assertEqual(cli.exit_for(0, {}), cli.EXIT_OK)
        self.assertEqual(cli.exit_for(codes.EXIT_NOT_ACTIVE, {}), cli.EXIT_NOT_ACTIVE)   # a child that refused its own activation exits 3 itself: 3
        for bad_rc in (1, 2, 4, 5, 13, 42, 137):
            self.assertEqual(cli.exit_for(bad_rc, {}), cli.EXIT_FAIL, f"rc={bad_rc}")
            rc, _ = self._quiet(lambda: cli.exit_for(bad_rc, {"partial": True, "levers_fallback": ["hoist"]}))
            self.assertEqual(rc, cli.EXIT_FAIL, f"rc={bad_rc}")                         # the run failed: 1, whatever the lever evidence says

    def test_partial_is_3_with_one_line_and_no_opt_out(self):
        rep = {"partial": True, "levers_fallback": ["hoist"], "fallback_reasons": {"hoist": "why"}}
        rc, err = self._quiet(lambda: cli.exit_for(0, dict(rep), where="in the run: the outputs under /o are not the mode's"))
        self.assertEqual(rc, cli.EXIT_NOT_ACTIVE)
        self.assertEqual(err.strip(), "[boltzgen-opt] NOT ACTIVE: partial activation — hoist fell back (hoist: why) in the run: the outputs under /o are not the mode's; a mode is all of its levers: exit 3")
        self.assertEqual(cli.IN_RUN_WHERE.format(out="/o"), "in the run: the outputs under /o are not the mode's")
        self.assertNotIn("allow_partial", rep)
        import inspect
        self.assertEqual(list(inspect.signature(cli.exit_for).parameters), ["child_rc", "rep", "where"])   # no `allowed`: there is no opt-out

    def test_the_census_and_an_oom_trace_are_records_not_exits(self):
        finding = {"kernels": {"ok": False, "findings": ["route=exact step=design cueq_triatt=absent:x (expected engaged)"], "route": "exact"}}
        self.assertEqual(cli.exit_for(0, dict(finding)), cli.EXIT_OK)          # a kernels finding never changes the exit
        self.assertEqual(cli.exit_for(1, dict(finding)), cli.EXIT_FAIL)
        self.assertEqual(cli.exit_for(3, dict(finding)), cli.EXIT_NOT_ACTIVE)
        self.assertEqual(cli.exit_for(0, {"oom": True}), cli.EXIT_OK)          # the trace printed its line and re-raised; upstream skipped the batch as stock does: the run's own rc rules
        self.assertEqual(cli.exit_for(1, {"oom": True}), cli.EXIT_FAIL)
        self.assertEqual(cli.exit_for(42, {"oom": True}), cli.EXIT_FAIL)       # no special child rc any more: a generic failure

    def test_stack_levers_from_lines_detects_the_sz_oom_line(self):
        res = modes.Resolution(mode="big", kits=(), env={}, imports=(), levers=(), runner=None, notes=[])
        lines_with_oom = ['[sz] {"event": "oom", "where": "boltz.py:1:forward", "msg": "CUDA out of memory.", "max_alloc_GB": 79.0}']
        self.assertTrue(stack.levers_from_lines(lines_with_oom, res)["oom"])
        self.assertFalse(stack.levers_from_lines(["ordinary line", "[sz] not an oom event"], res)["oom"])
        import dataclasses
        self.assertEqual([f.name for f in dataclasses.fields(modes.Resolution)], ["mode", "kits", "env", "imports", "levers", "runner", "notes"])   # no worker, no variant, no packer


WHOLE = {"form": "process", "arm": "kit", "rc": 0, "wall_s": 1.0, "dropped_env": [], "pythonpath": "", "exported": {"BG_GRAPH": "graph", "XA_FAST_INIT": "1", "XA_HOIST": "1"},
         "levers_applied": ["fastinit", "hoist", "graph_sampler", "inproc"], "levers_fallback": [], "fallback_reasons": {}, "partial": False,
         "kernels": {"ok": True, "findings": [], "lines": [], "words": {}, "expected": {}, "peak": []}, "route": "exact"}
PARTIAL = dict(WHOLE, levers_applied=["fastinit", "graph_sampler", "inproc"], levers_fallback=["hoist"], fallback_reasons={"hoist": "[xa_hoist] DISABLED: no partner"}, partial=True)
FAILED = dict(WHOLE, rc=1)
FINDING = dict(WHOLE, kernels={"ok": False, "findings": ["route=exact step=design cueq_triatt=fallback:reference@s_qo=128:1[served=0,byrule=1,fallback=1] (expected engaged)"], "lines": [], "words": {}, "expected": {}, "peak": []})


def cli_with_fake_run(args, env, run, resolution=RESOLUTION_ON):
    """cli.main(args) in a fresh interpreter with the children replaced: design.run_kit / design.run_stock return ``run`` (the evidence a
    real child leaves; its ``log`` is ``<run_dir>/opt_run.log``, which the caller may have written), configure is a no-op returning
    upstream's resolution line as ``resolution`` (the caller may have laid out ``steps.yaml`` / ``config/`` itself)."""
    code = f"""
import json, sys
from boltzgen_opt import cli, design
RUN, RES = json.loads({json.dumps(run)!r}), json.loads({json.dumps(resolution)!r})
design.configure = lambda spec, run_dir, user_args, cache, log_path=None: {{"cmd": ["configure", spec, "--output", run_dir, *user_args], "kernels_resolution": RES}}
def _run_kit(res, run_dir, seed, echo=True, steps=None, exp=None):
    print("RUNKIT " + json.dumps({{"seed": seed, "expect": (exp or {{}}).get("expect"), "words": (exp or {{}}).get("words")}}))
    return dict(RUN, log=run_dir + "/opt_run.log")
def _run_stock(run_dir, seed, echo=True):
    print("RUNSTOCK " + json.dumps({{"seed": seed}}))
    return dict(RUN, log=run_dir + "/opt_run.log")
design.run_kit, design.run_stock = _run_kit, _run_stock
raise SystemExit(cli.main({args!r}))
"""
    return _stubs.run_py(code, env)


STOCK_RUN = {"form": "process", "arm": "stock", "steps": [], "rc": 0, "wall_s": 1.0, "dropped_env": [], "stock_env_proof": [], "kernels": None, "route": "off"}


def lay_out_run(run_dir, *, multiplicity=3, diffusion_samples=4, specs=1, written=None, stale=0, reuse=False, log_lines=(), step="design",
                steps=("design", "inverse_folding", "folding"), relative_output=False):
    """A configured run directory as upstream's `configure` and its design step leave it: ``steps.yaml``, the resolved ``config/<step>.yaml``
    of every step (the design-generating one reads the spec: FromYamlDataModule; the others a design directory), ``written`` design CIFs
    (default: all requested) newer than ``steps.yaml`` plus ``stale`` older ones and one ``_native.cif`` / ``.npz`` beside them, and
    ``opt_run.log`` holding ``log_lines``. Returns (requested, design_dir)."""
    import yaml
    os.makedirs(os.path.join(run_dir, "config"), exist_ok=True)
    design_dir = os.path.join(run_dir, "intermediate_designs")
    os.makedirs(design_dir, exist_ok=True)
    requested = specs * multiplicity * diffusion_samples
    written = requested if written is None else written
    t_cfg = time.time() - 100                                                   # steps.yaml: written by configure, before any design
    for name in steps:
        if name == step:
            cfg = {"_target_": "boltzgen.task.predict.predict.Predict",
                   "data": {"_target_": "boltzgen.task.predict.data_from_yaml.FromYamlDataModule",
                            "cfg": {"yaml_path": [f"/s/spec{i}.yaml" for i in range(specs)], "multiplicity": multiplicity, "skip_existing": reuse,
                                    "output_dir": "${output}", "diffusion_samples": "${diffusion_samples}"}, "num_workers": 1},
                   "writer": {"_target_": "boltzgen.task.predict.writer.DesignWriter", "output_dir": "${output}"},
                   "output": "intermediate_designs" if relative_output else design_dir, "diffusion_samples": diffusion_samples}
            if relative_output:
                cfg["output"] = os.path.relpath(design_dir, os.path.dirname(os.path.abspath(run_dir)))
        else:
            cfg = {"_target_": "boltzgen.task.predict.predict.Predict",
                   "data": {"_target_": "boltzgen.task.predict.data_from_generated.FromGeneratedDataModule", "design_dir": design_dir, "cfg": {"multiplicity": 2}},
                   "output": design_dir + "_inverse_folded"}
        with open(os.path.join(run_dir, "config", f"{name}.yaml"), "w") as fh:
            yaml.safe_dump(cfg, fh)
    with open(os.path.join(run_dir, "steps.yaml"), "w") as fh:
        yaml.safe_dump({"steps": [{"name": n, "config_file": f"config/{n}.yaml"} for n in steps]}, fh)
    os.utime(os.path.join(run_dir, "steps.yaml"), (t_cfg, t_cfg))
    width = len(str(max(requested - 1, 1)))
    for i in range(written + stale):
        f = os.path.join(design_dir, f"spec0_{i:0{width}d}.cif")
        open(f, "w").write("data_x\n")
        if i >= written:                                                        # the stale ones: older than this run's steps.yaml
            os.utime(f, (t_cfg - 50, t_cfg - 50))
    open(os.path.join(design_dir, "spec0_0_native.cif"), "w").write("data_native\n")   # the writer's copy of the input structure: not a design
    open(os.path.join(design_dir, "spec0_0.npz"), "w").write("")
    with open(os.path.join(run_dir, "opt_run.log"), "w") as fh:
        fh.write("".join(ln + "\n" for ln in log_lines))
    return requested, design_dir


OOM_LINE = "| WARNING: ran out of memory, skipping batch"                     # upstream's own words (stock boltz.py predict_step)
FEATURIZER_LINE = "WARNING: Skipping batch. Exception for spec0"


class TestDesignCensus(unittest.TestCase):
    """The design census every mode ends with (design.designs_census → the one `DESIGNS` line, report.designs_line, and the manifest's
    `designs`): requested from the design-generating step's resolved config (specs × multiplicity × diffusion_samples — upstream's own
    numbers), produced by counting the design files this run wrote, the batches upstream's own handlers skipped counted from its own two
    lines in the run log. Upstream skips an out-of-memory batch and its step still exits 0: the census is where the shortfall is stated."""
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="bgopt_census_")
        cls.site = _stubs.materialize(cls.tmp)
        cls.spec = os.path.join(cls.tmp, "s.yaml"); open(cls.spec, "w").write("entities: []\n")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_requested_is_the_resolved_configs_count_and_produced_the_files_this_run_wrote(self):
        run_dir = os.path.join(self.tmp, "r_whole")
        requested, design_dir = lay_out_run(run_dir, multiplicity=3, diffusion_samples=4)      # --num_designs 10 --diffusion_batch_size 4: upstream writes 12
        c = design.designs_census(run_dir, [])
        self.assertEqual((c["step"], c["requested"], c["produced"], c["oom_skipped"], c["featurizer_skipped"], c["stale"], c["reuse"]), ("design", 12, 12, 0, 0, 0, False))
        self.assertEqual((c["specs"], c["multiplicity"], c["diffusion_samples"], c["output_dir"], c["config"]), (1, 3, 4, design_dir, os.path.join(run_dir, "config", "design.yaml")))
        self.assertEqual(report.designs_line(c), "[boltzgen-opt] DESIGNS step=design requested=12 produced=12 oom_skipped=0 featurizer_skipped=0")
        run_dir = os.path.join(self.tmp, "r_short")                            # one batch of 4 ran out of memory, one design's featurizer raised: 7 written, 2 stale files of an earlier run not rewritten
        lay_out_run(run_dir, multiplicity=3, diffusion_samples=4, written=7, stale=2, log_lines=["Predicting DataLoader 0:  33%|###3      | 1/3\r" + OOM_LINE, FEATURIZER_LINE, "unrelated " + OOM_LINE[2:]])
        c = design.designs_census(run_dir, design._lines(os.path.join(run_dir, "opt_run.log")))
        self.assertEqual((c["requested"], c["produced"], c["oom_skipped"], c["featurizer_skipped"], c["stale"]), (12, 7, 1, 1, 2))   # the OOM line counts behind a progress-bar fragment; a line without upstream's `| WARNING:` head does not
        self.assertEqual(report.designs_line(c), "[boltzgen-opt] DESIGNS step=design requested=12 produced=7 oom_skipped=1 featurizer_skipped=1 stale=2")
        run_dir = os.path.join(self.tmp, "r_reuse")                            # upstream's --reuse keeps existing designs and writes the rest: produced counts them all
        lay_out_run(run_dir, multiplicity=2, diffusion_samples=1, written=1, stale=1, reuse=True)
        c = design.designs_census(run_dir, [])
        self.assertEqual((c["requested"], c["produced"], c["stale"], c["reuse"]), (2, 2, 0, True))
        self.assertEqual(report.designs_line(c), "[boltzgen-opt] DESIGNS step=design requested=2 produced=2 oom_skipped=0 featurizer_skipped=0 reuse=true")
        run_dir = os.path.join(self.tmp, "r_rel")                              # a relative `output` is configure's own: resolved against the run directory's parent, its working directory
        lay_out_run(run_dir, multiplicity=1, diffusion_samples=2, relative_output=True)
        self.assertEqual(design.designs_census(run_dir, [])["produced"], 2)
        run_dir = os.path.join(self.tmp, "r_two_specs")
        lay_out_run(run_dir, specs=2, multiplicity=2, diffusion_samples=3, written=5)
        self.assertEqual((design.designs_census(run_dir, [])["requested"], design.designs_census(run_dir, [])["produced"]), (12, 5))

    def test_a_run_without_a_design_generating_step_says_so(self):
        run_dir = os.path.join(self.tmp, "r_fold_only")
        lay_out_run(run_dir, steps=("folding",), log_lines=[OOM_LINE])                     # `--steps folding`: configure lists only that step; the run refolds existing designs, generates none
        c = design.designs_census(run_dir, [OOM_LINE])
        self.assertEqual((c["step"], c["requested"], c["produced"], c["oom_skipped"]), (None, None, None, 1))
        self.assertEqual(report.designs_line(c), "[boltzgen-opt] DESIGNS step=none requested=n/a produced=n/a oom_skipped=1 featurizer_skipped=0")
        self.assertEqual(design.designs_census(os.path.join(self.tmp, "never_configured"), [])["step"], None)   # no steps.yaml (configure never ran): no request, no crash
        run_dir = os.path.join(self.tmp, "r_ifold_only")                                    # --only_inverse_fold: inverse_folding reads the spec and generates the designs (one per batch)
        lay_out_run(run_dir, step="inverse_folding", steps=("inverse_folding", "folding"), multiplicity=5, diffusion_samples=1)
        c = design.designs_census(run_dir, [])
        self.assertEqual((c["step"], c["requested"], c["produced"]), ("inverse_folding", 5, 5))

    def test_design_prints_the_line_and_records_the_census_in_every_mode(self):
        """`design` on a kit mode and on `off` alike: the DESIGNS line on stderr after the run, `designs` / `oom_skipped` / the activation
        report's `oom` in opt_manifest.json — and the exit code is the run's own (0: upstream skipped the batch and exited 0)."""
        for mode, run, tag in (("exact", WHOLE, "RUNKIT"), ("off", STOCK_RUN, "RUNSTOCK")):
            out = os.path.join(self.tmp, "d_" + mode)
            lay_out_run(out, multiplicity=2, diffusion_samples=1, written=1, log_lines=[OOM_LINE])
            rc, so, se = cli_with_fake_run(["design", "--mode", mode, self.spec, "--output", out, "--seed", "0", "--num_designs", "2"], _stubs.clean_env(self.site), run)
            self.assertEqual(rc, 0, se)
            self.assertIn(tag + " ", so)
            self.assertIn("[boltzgen-opt] DESIGNS step=design requested=2 produced=1 oom_skipped=1 featurizer_skipped=0", se.splitlines(), (mode, se))
            man = manifest.read(out)
            self.assertEqual((man["designs"]["requested"], man["designs"]["produced"], man["designs"]["oom_skipped"], man["activation_report"].get("oom"), man["exit_code"]),
                             (2, 1, 1, True, 0), mode)
            self.assertEqual(("oom_skipped" in man, "designs" in man["activation_report"], "designs" in man["run"]), (False, False, False), mode)   # the census has one place in the manifest: `designs`
        out = os.path.join(self.tmp, "d_clean")                                                 # a complete run: no `oom` key in the report, oom_skipped 0
        lay_out_run(out, multiplicity=2, diffusion_samples=1)
        rc, so, se = cli_with_fake_run(["design", "--mode", "exact", self.spec, "--output", out, "--seed", "0", "--num_designs", "2"], _stubs.clean_env(self.site), WHOLE)
        self.assertEqual(rc, 0, se)
        self.assertIn("[boltzgen-opt] DESIGNS step=design requested=2 produced=2 oom_skipped=0 featurizer_skipped=0", se.splitlines(), se)
        man = manifest.read(out)
        self.assertEqual((man["designs"]["oom_skipped"], "oom" in man["activation_report"], man["designs"]["stale"]), (0, False, 0))
        i_designs, i_active = se.find("] DESIGNS "), se.find("] ACTIVE mode=exact")
        self.assertTrue(0 < i_designs < i_active, se)                                          # the census first, then the activation line

    def test_upstreams_skip_lines_are_read_as_it_prints_them(self):
        """The two counted lines are upstream's own prints, held byte-literal against the pinned wheel's source."""
        import zipfile
        whl = os.path.join(os.path.dirname(stack.pins_path()), os.path.basename(stack.pins()["wheel"]["file"]))   # stock/<wheel>, beside PINS.json
        with zipfile.ZipFile(whl) as z:
            src = z.read("boltzgen/model/models/boltz.py").decode()
        self.assertIn('print("| WARNING: ran out of memory, skipping batch")', src)
        self.assertIn("""print(f"WARNING: Skipping batch. Exception for {batch['id'][0]}")""", src)
        self.assertRegex("| WARNING: ran out of memory, skipping batch", design.UPSTREAM_OOM_SKIP)
        self.assertRegex("WARNING: Skipping batch. Exception for 7xyz_3", design.UPSTREAM_FEATURIZER_SKIP)
        self.assertNotRegex("WARNING: Skipping batch. Skip was set true for x", design.UPSTREAM_FEATURIZER_SKIP)   # upstream's --reuse skip of an existing design is not a failure


class TestExitRule(unittest.TestCase):
    """One exit code per condition on the running verbs: a partial activation (a lever of the mode could not run) is recorded and exits 3 —
    a mode is all of its levers, there is no opt-out —, a failed run is 1, a kernels finding is recorded and exits 0; `warm` follows `design`."""
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="bgopt_exit_")
        cls.site = _stubs.materialize(cls.tmp)
        cls.spec = os.path.join(cls.tmp, "s.yaml"); open(cls.spec, "w").write("entities: []\n")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _cli(self, args, env, run, resolution=RESOLUTION_ON):
        return cli_with_fake_run(args, env, run, resolution)

    @staticmethod
    def _runkit(so):
        return [json.loads(ln[7:]) for ln in so.splitlines() if ln.startswith("RUNKIT ")]

    def test_design_partial_is_exit_3_and_recorded(self):
        out = os.path.join(self.tmp, "d_partial")
        rc, so, se = self._cli(["design", "--mode", "exact", self.spec, "--output", out, "--seed", "0", "--num_designs", "1"], _stubs.clean_env(self.site), PARTIAL)
        self.assertEqual(rc, 3, se)                                              # a mode is all of its levers: refused by name after the run
        self.assertIn("ACTIVE mode=exact form=process", se); self.assertIn(" partial=hoist", se); self.assertIn(" fallbacks=hoist", se)
        self.assertIn(f"[boltzgen-opt] NOT ACTIVE: partial activation — hoist fell back (hoist: [xa_hoist] DISABLED: no partner) in the run: the outputs under {out} are not the mode's; a mode is all of its levers: exit 3", se.splitlines())
        man = manifest.read(out)
        self.assertEqual((man["active"], man["exit_code"], man["partial"], man["levers_fallback"], man["seed"]), (True, 3, ["hoist"], ["hoist"], 0))
        self.assertNotIn("allow_partial", man)
        self.assertEqual(self._runkit(so), [{"seed": 0, "expect": "on", "words": {"cueq_triatt": "engaged", "cueq_trimul": "engaged"}}])   # the child got the caller's seed and the route's expectation
        rc, so, se = self._cli(["design", "--mode", "exact", self.spec, "--output", out + "_a", "--seed", "0", "--num_designs", "1", "--allow-partial"], _stubs.clean_env(self.site), PARTIAL)
        self.assertEqual(rc, 2, se)                                              # yesterday's opt-out is no option: nothing runs under a mode's name with a subset of its levers
        rc, so, se = self._cli(["design", "--mode", "exact", self.spec, "--output", out + "_e", "--seed", "0", "--num_designs", "1"], _stubs.clean_env(self.site, BOLTZGEN_OPT_ALLOW_PARTIAL="1"), PARTIAL)
        self.assertEqual(rc, 3, se)                                              # nor is its env form
        out3 = os.path.join(self.tmp, "d_whole")                                # a whole activation: no partial record, exit 0
        rc, so, se = self._cli(["design", "--mode", "exact", self.spec, "--output", out3, "--seed", "0", "--num_designs", "1"], _stubs.clean_env(self.site), WHOLE)
        self.assertEqual(rc, 0, se); self.assertNotIn("partial=", se); self.assertNotIn("NOT ACTIVE", se)
        self.assertEqual(manifest.read(out3)["partial"], [])

    def test_design_failed_run_is_exit_1_never_partial(self):
        out = os.path.join(self.tmp, "d_failed")
        rc, so, se = self._cli(["design", "--mode", "exact", self.spec, "--output", out, "--seed", "0", "--num_designs", "1"], _stubs.clean_env(self.site), FAILED)
        self.assertEqual(rc, 1, se)
        self.assertEqual((manifest.read(out)["exit_code"], manifest.read(out)["partial"]), (1, []))
        rc, so, se = self._cli(["design", "--mode", "exact", self.spec, "--output", out, "--seed", "0", "--num_designs", "1"], _stubs.clean_env(self.site), dict(PARTIAL, rc=1))
        self.assertEqual(rc, 1, se)                                          # the run failed: 1, whatever the lever evidence says
        rc, so, se = self._cli(["design", "--mode", "exact", self.spec, "--output", out + "_rc3", "--seed", "0", "--num_designs", "1"], _stubs.clean_env(self.site), dict(WHOLE, rc=3))
        self.assertEqual(rc, 3, se)                                          # a child that refused its own activation (rc 3) is 3, not 1
        self.assertEqual(manifest.read(out + "_rc3")["exit_code"], 3)

    def test_a_kernels_finding_is_recorded_and_never_an_exit(self):
        """The accelerator census is a record: a run whose census found a reference-path call outside the library's rules (or an absent
        library, a missing line) exits with the run's own code — 0 here — and the finding is in opt_manifest.json under the activation
        report's `kernels` (`ok` false, `findings`), never a fourth exit code."""
        out = os.path.join(self.tmp, "d_finding")
        rc, so, se = self._cli(["design", "--mode", "exact", self.spec, "--output", out, "--seed", "0", "--num_designs", "1"], _stubs.clean_env(self.site), FINDING)
        self.assertEqual(rc, 0, se)
        self.assertNotIn("REFUSED", se); self.assertNotIn("NOT ACTIVE", se)
        k = manifest.read(out)["activation_report"]["kernels"]
        self.assertEqual((k["ok"], k["findings"], k["route"]), (False, FINDING["kernels"]["findings"], "exact"))
        self.assertNotIn("refusals", k)
        rc, so, se = self._cli(["design", "--mode", "exact", self.spec, "--output", out + "_p", "--seed", "0", "--num_designs", "1"], _stubs.clean_env(self.site), dict(PARTIAL, kernels=FINDING["kernels"]))
        self.assertEqual(rc, 3, se)                                          # partial stays 3: the finding neither raises nor lowers the exit

    def test_use_kernels_false_is_the_callers_switch_on_a_kit_mode(self):
        """`-- --use_kernels false` (upstream's own configure switch) on a kit mode: passed to configure verbatim, the route then expects no
        library call (`off:use_kernels=false`, words off-by-route), the child runs, exit 0 — no refusal on any route."""
        out = os.path.join(self.tmp, "d_nokernels")
        rc, so, se = self._cli(["design", "--mode", "exact", self.spec, "--output", out, "--seed", "4", "--num_designs", "1", "--", "--use_kernels", "false"],
                               _stubs.clean_env(self.site), WHOLE, resolution=RESOLUTION_OFF)
        self.assertEqual(rc, 0, se)
        self.assertNotIn("REFUSED", se); self.assertNotIn("NOT ACTIVE", se)
        self.assertEqual(self._runkit(so), [{"seed": 4, "expect": "off:use_kernels=false", "words": {"cueq_triatt": "off-by-route:use_kernels=False(use_kernels=false)", "cueq_trimul": "off-by-route:use_kernels=False(use_kernels=false)"}}])
        man = manifest.read(out)
        self.assertEqual(man["exit_code"], 0)
        self.assertEqual(man["configure"]["cmd"][-2:], ["--use_kernels", "false"])       # the caller's words reached configure as given
        for mode in modes.MODES:                                                          # the expectation itself, every route: a record with no refusal in it
            e = kernels.expectation(mode, {"use_kernels": False, "cc": (9, 0), "line": "Using kernels: False [device capability: (9, 0)]"})
            self.assertEqual((e["expect"], set(e["words"].values()), "refusal" in e), ("off:use_kernels=false", {"off-by-route:use_kernels=False(use_kernels=false)"}, False), mode)
        self.assertFalse(hasattr(kernels, "KernelsRefused")); self.assertFalse(hasattr(kernels, "refusal_line"))

    def _warm(self, name, args, env_more, run):
        """`warm` in a fresh interpreter with the child replaced; its outputs go to a temporary directory under TMPDIR (removed after a PASS),
        so TMPDIR points at a directory of this test and the manifest is read from there."""
        tmpdir = os.path.join(self.tmp, "warmtmp_" + name); os.makedirs(tmpdir)
        rc, so, se = self._cli(["warm", *args], _stubs.clean_env(self.site, TMPDIR=tmpdir, **env_more), run)
        mans = glob.glob(os.path.join(tmpdir, "boltzgen_opt_warm_*", "warm", manifest.FILENAME))
        return rc, so, se, mans

    def test_warm_follows_design(self):
        rc, so, se, mans = self._warm("partial", ["--mode", "exact"], {}, PARTIAL)
        self.assertEqual(rc, 3, se)                                              # a mode is all of its levers: warm refuses by name like design
        self.assertIn("warm mode=exact: NOT ACTIVE rc=0", se); self.assertIn(" partial=hoist", se)
        self.assertIn("[boltzgen-opt] NOT ACTIVE: partial activation — hoist fell back (hoist: [xa_hoist] DISABLED: no partner); a mode is all of its levers: exit 3", se.splitlines())
        self.assertEqual(len(mans), 1, mans)                                    # not a PASS: the run directory is kept
        man = manifest.read(mans[0])
        self.assertEqual((man["command"], man["exit_code"], man["partial"], man["seed"]), ("warm", 3, ["hoist"], 0)); self.assertNotIn("allow_partial", man)
        self.assertEqual(self._runkit(so)[0]["seed"], 0)                       # warm's own seed (warm.SEED); no --seed option
        rc, so, se, mans = self._warm("failed", ["--mode", "exact"], {}, FAILED)
        self.assertEqual(rc, 1, se); self.assertIn("warm mode=exact: FAIL rc=1", se)
        self.assertEqual(manifest.read(mans[0])["exit_code"], 1)
        for extra in (["--out", os.path.join(self.tmp, "w")], ["--keep"], ["--quiet"], ["--json"], ["--seed", "1"], ["--cache", "/hf"], ["--allow-partial"]):   # warm takes --mode only
            rc, so, se, mans = self._warm("extra_" + extra[0].lstrip("-"), ["--mode", "exact", *extra], {}, WHOLE)
            self.assertEqual(rc, 2, (extra, se[-300:]))
            self.assertEqual(mans, [])

    def test_the_partial_refusal_line_is_literal(self):
        """The grammar of the one partial line — a refusal —, byte-literal around this engine's own <detail> (one formatter, report.py)."""
        line = report.partial_exit_line(["hoist"], {"hoist": "why"})
        self.assertEqual(line, "[boltzgen-opt] NOT ACTIVE: partial activation — hoist fell back (hoist: why); a mode is all of its levers: exit 3")
        self.assertEqual(report.partial_exit_line(["a", "b"], None, where=cli.IN_RUN_WHERE.format(out="/o")),
                         "[boltzgen-opt] NOT ACTIVE: partial activation — a,b fell back in the run: the outputs under /o are not the mode's; a mode is all of its levers: exit 3")
        self.assertEqual(report.activation_line({"active": False, "reason": report.partial_reason(["hoist"], {"hoist": "why"})}), line)
        self.assertTrue(line.startswith(f"{report.PREFIX} {report.NOT_ACTIVE}: partial activation — ")); self.assertTrue(line.endswith(f"exit {codes.EXIT_NOT_ACTIVE}"))
        self.assertFalse(hasattr(report, "partial_allowed_line"))                       # no opt-out form: nothing runs under a mode's name with a subset of its levers

    def test_activation_line_names_the_partial_levers(self):
        rep = {"active": True, "mode": "exact", "form": "process", "switches": "x", "levers_applied": ["fastinit"], "levers_fallback": ["hoist"], "partial": True}
        self.assertIn(" partial=hoist", report.activation_line(rep)); self.assertTrue(report.activation_line(rep).endswith(" fallbacks=hoist"))
        self.assertNotIn("allow_partial", report.activation_line(dict(rep, allow_partial=True)))   # no such field any more: nothing prints it
        self.assertNotIn("gpu_gate=", report.activation_line(dict(rep, gpu_gate="noted")))          # the card is the NOTE line's, never the ACTIVE line's
        self.assertNotIn("partial=", report.activation_line(dict(rep, partial=False, levers_fallback=[])))
        self.assertEqual(stack.partial_levers(rep), ["hoist"]); self.assertEqual(stack.partial_levers(dict(rep, partial=False)), [])
        for gone in ("allow_partial_env", "ENV_ALLOW_PARTIAL", "ENV_FORCE"):
            self.assertFalse(hasattr(stack, gone), gone)


if __name__ == "__main__":
    unittest.main()
