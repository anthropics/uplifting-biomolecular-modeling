"""The command line: usage and exit codes, the env-vs-CLI rule, check's DRY-RUN line, the stock route end to end against a stub checkout
(a git checkout of stub scripts standing in for the upstream ones: they record the environment they see and write the stock output layout;
the route runs the checkout's own script where it is), the ENV-CLEAN proof, opt_manifest.json, the CMD line (one per process launched, both
routes), the EXIT tally, the stock options passed through (verbatim on the child's argv, upstream's defaults when absent; a lever, a non-stock
option or a package-supplied option refused by name, nothing launched; an option mode exact cannot serve refused by name, exit 3, nothing launched),
the kit route's staged launch against a stub worker, the ``--hybrid_gemm 0`` opt-out, the weights rule end to end (a second weight file selected by
--model_name runs on both routes with the NOT PINNED line and its record in the manifest; upstream's own weights selectors pass through; the pinned
digest is named as pinned; an absent file is refused by name), and run.sh in a subprocess. The pin check is stubbed where a test does not say
otherwise (the weights part of it is stock/check_pins.py's own where a test says so); no torch, no GPU."""
import contextlib
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
import stat
from unittest import mock

from proteinmpnn_opt import cli, inputs, kit_run, manifest, report, stack, stock_run

sys.path.insert(0, os.path.join(stack.tree_home(), "stock"))
import check_pins  # noqa: E402  (stock/check_pins.py, standard library only: the one producer of the weights record and its line)

PINNED = {"pinned": True, "findings": [], "detail": {"head": "8907e6671bfbfc92303b5f79c4b5e6ce47cdef57", "commit": "26ec57ac976ade5379920dbd43c7f97a91cf82de"}}
KIT = ["--save_score", "1", "--save_probs", "1"]                                # upstream's two write gates, open: scores/ and probs/ written on either route (the worker takes them as upstream does; closed or absent, no arrays)

STUB_PARSER = r'''
import json, sys
a = dict(zip(sys.argv[1::2], sys.argv[2::2]))
inp = a.get("--input_path") or [x for x in sys.argv if x.startswith("--input_path=")][0].split("=", 1)[1]
out = a.get("--output_path") or [x for x in sys.argv if x.startswith("--output_path=")][0].split("=", 1)[1]
import os
with open(out, "w") as fh:
    for f in sorted(os.listdir(inp), reverse=True):                                    # a listing order that is not name order, as a file system's may be
        if f.endswith(".pdb"):
            n = f[:-4]
            fh.write(json.dumps({"name": n, "seq_chain_A": "MKV", "seq_chain_B": "GLW", "num_of_chains": 2}) + "\n")
'''
STUB_RUN = r'''
import json, os, sys
a = dict(zip(sys.argv[1::2], sys.argv[2::2]))
out = a["--out_folder"]
os.makedirs(os.path.join(out, "seqs"), exist_ok=True)
for gate, sub in (("--save_score", "scores"), ("--save_probs", "probs")):                     # upstream's write gates: the folder and the arrays only when open
    if a.get(gate) == "1": os.makedirs(os.path.join(out, sub), exist_ok=True)
rows = [json.loads(line) for line in open(a["--jsonl_path"])] if "--jsonl_path" in a else [{"name": os.path.basename(a["--pdb_path"])[:-4]}]   # --pdb_path: upstream's single-PDB input
for r in rows:
    with open(os.path.join(out, "seqs", r["name"] + ".fa"), "w") as fh:
        fh.write(">%s, seed=%s, git_hash=stub\nMKVGLW\n" % (r["name"], a.get("--seed")))
    if a.get("--save_score") == "1": open(os.path.join(out, "scores", r["name"] + ".npz"), "wb").write(b"NPZ")
    if a.get("--save_probs") == "1": open(os.path.join(out, "probs", r["name"] + ".npz"), "wb").write(b"NPZ")
json.dump({"env": {k: v for k, v in os.environ.items() if k.startswith(("PROTEINMPNN", "MPNN", "PYTHON"))}, "argv": sys.argv[1:], "cwd": os.getcwd()},
          open(os.path.join(out, "stub_seen.json"), "w"))
print("stub protein_mpnn_run.py ran")
'''
STUB_WORKER = r'''
import json, os, sys
a = {}
i = 1
while i < len(sys.argv):
    if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--"):
        a[sys.argv[i]] = sys.argv[i + 1]; i += 2
    else:
        a[sys.argv[i]] = True; i += 1
out = a["--out_folder"]
if os.environ.get("STUB_OOM"):                                                                   # the worker dies on an out-of-memory: torch's line, exit 1, no output
    sys.stderr.write("Traceback (most recent call last):\n  ...\ntorch.OutOfMemoryError: CUDA out of memory (mock)\n"); sys.exit(1)
hyb = bool(a.get("--hybrid_gemm")); hyb_pass = hyb and os.environ.get("STUB_PROBE", "PASS") == "PASS"          # the worker's on-device probe: STUB_PROBE=FAIL refuses the lever on this device
probe = {"verdict": ("PROBE PASS -> using hybrid (stub)" if hyb_pass else "PROBE FAIL -> stock-shape GEMMs (stub)") if hyb else "not requested -> stock-shape GEMMs (exact by construction)", "device": "stub"}
if os.environ.get("STUB_TRITON") == "0" and (a.get("--x_all") or a.get("--fused_draw")):      # no Triton on this stack: the draw kernel cannot run — refused by name before any input is read, rc 3 (the worker's rule)
    fd = {"verdict": "PROBE FAIL (no Triton) -> the fused draw kernel cannot run here"}
    print("fused_draw: %s" % fd["verdict"]); print("fused_draw: REFUSED -> (stub)")
    print(json.dumps({"fused_draw_probe": fd, "refused": "fused_draw: triton unavailable"})); sys.exit(3)
if hyb and not hyb_pass:                                                                         # the line is all of its levers: refused by name before any output, rc 3 (the worker's rule)
    print("hybrid_gemm: PROBE FAIL -> stock-shape GEMMs (stub)"); print("hybrid_gemm: REFUSED -> (stub)")
    print(json.dumps({"hybrid_gemm_probe": probe, "refused": "hybrid_gemm: PROBE FAIL"})); sys.exit(3)
os.makedirs(os.path.join(out, "seqs"), exist_ok=True)
if a.get("--inputs_ready"):                                                                      # started beside the parse step: the inputs are read once the driver says they are written
    import time
    t0 = time.time()
    while not os.path.exists(a["--inputs_ready"]) and time.time() - t0 < 60: time.sleep(0.005)
names = [json.loads(line)["name"] for line in open(a["--jsonl_path"])]
for name in names[:len(names) - int(os.environ.get("STUB_SHORT", "0"))]:
    open(os.path.join(out, "seqs", name + ".fa"), "w").write(">%s, seed=%s\nMKVGLW\n" % (name, a.get("--seed")))
chains = json.loads(open(a["--chain_id_jsonl"]).readline()) if a.get("--chain_id_jsonl") else "absent"          # line one, as protein_mpnn_run.py / the worker read it
json.dump({"flags": a, "chains": chains, "env": {k: v for k, v in os.environ.items() if k.startswith(("PROTEINMPNN", "MPNN"))}}, open(os.path.join(out, "stub_seen.json"), "w"))
print("DEVICE CELL: cpu n/a torch stub cuda None")
# the worker's own end-of-run record: its flags after `--x_all` (all on), its CPU rule (the CUDA-graph levers off unless STUB_CUDA), STUB_DROP=<lever,...> off
levers = {k: bool(a.get("--x_all")) or bool(a.get("--" + k)) for k in ("chunk_gemm", "cache_enc_ctx", "graph_rng", "single_graph", "fused_draw", "analytic_offsets", "stock_shape_enc")}
if not os.environ.get("STUB_CUDA"):
    levers["graph_rng"] = levers["single_graph"] = levers["fused_draw"] = False
for k in [x for x in os.environ.get("STUB_DROP", "").split(",") if x]:
    levers[k] = False
rec = {"mode": a.get("--mode"), "bb_batch": int(a.get("--bb_batch", 1)), "graph": False, "hybrid_gemm": hyb, "hybrid_gemm_active": hyb_pass,
       "hybrid_gemm_probe": probe, "n_fa": len(os.listdir(os.path.join(out, "seqs"))), "expected_fa": len(names), **levers,
       "lowmem": os.path.basename(sys.argv[0]) == "mpnn_worker2_lowmem.py"}   # the derived executable's record field (stage.lowmem_worker_source)
if not os.environ.get("STUB_NO_RECORD"):
    print(json.dumps(rec))
'''

H100 = {"name": "NVIDIA H100 80GB HBM3", "mem_mib": 81559, "cc": "9.0", "sm": "sm_90", "count": 1}


def stub_stage_patch():
    """stage_base with the stub worker and parser written over the staged copies (the tree's own files are never touched)."""
    real_stage = kit_run._stage.stage_base

    def stub_stage(stage_dir):
        paths = real_stage(stage_dir)
        open(paths["worker"], "w").write(STUB_WORKER); open(paths["fast_parse"], "w").write(STUB_PARSER)
        open(paths["worker_lowmem"], "w").write(STUB_WORKER)               # the derived executable the base variants' exact line runs
        return paths
    return mock.patch.object(kit_run._stage, "stage_base", stub_stage)


def git_checkout(path):
    """Make ``path`` a git checkout with one commit (protein_mpnn_run.py writes the checkout's HEAD into its .fa headers; check reports it); returns HEAD.
    Skips the test (not a failure) when no ``git`` binary is on PATH -- this fabricates an *external* checkout to exercise the pin, it
    is not exercising this repository's own git commit, and a git-less test sandbox is a real, nameable environment gap rather than
    a code defect."""
    if shutil.which("git") is None:
        raise unittest.SkipTest("git binary not on PATH: cannot fabricate the stub ProteinMPNN checkout this class's setUp needs")
    ident = ["-c", "user.name=stub", "-c", "user.email=stub@example.invalid"]
    subprocess.run(["git", "-C", path, "init", "-q"], check=True, capture_output=True)
    subprocess.run(["git", "-C", path, *ident, "commit", "-q", "--allow-empty", "-m", "stub checkout"], check=True, capture_output=True)
    return subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def parsed_names(path):
    """The entry names of a parsed.jsonl, in file order."""
    return [json.loads(ln)["name"] for ln in open(path, encoding="utf-8") if ln.strip()]


def cmd_lines(text):
    """The CMD lines of a captured stderr, parsed (report.CMD_LINE_RE)."""
    return [re.search(report.CMD_LINE_RE, ln).groupdict() for ln in text.splitlines() if re.search(report.CMD_LINE_RE, ln)]


def weights_rule_pins(variant, model_name=None, weights_file=None):
    """The pin check of a stub checkout: the checkout taken as present, the weights part stock/check_pins.py's own
    (check_weights on the file the pass loads: ``weights_file`` as the CLI resolved it from the pass's own selectors (settings.weights_path), else
    check_pins.weights_file from MPNN_DIR and the model_name)."""
    pins = check_pins.load_pins()
    bad, record = check_pins.check_weights(pins, weights_file or check_pins.weights_file(pins, variant, mpnn_dir=os.environ.get(stack.ENV_MPNN_DIR), model_name=model_name))
    return {"pinned": not bad, "findings": bad, "detail": {**PINNED["detail"], "weights": record}}


def _run(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = cli.main(argv)
    return rc, out.getvalue(), err.getvalue()


class Base(unittest.TestCase):
    def setUp(self):
        stack.reset_for_tests()
        self._env = dict(os.environ)
        for k in stack.PACKAGE_ENV + (stack.ENV_MPNN_DIR, stack.ENV_TARGET_GPU, stack.ENV_TARGET_GPU_MEM):
            os.environ.pop(k, None)
        self.tmp = tempfile.mkdtemp()
        self.mpnn = os.path.join(self.tmp, "ProteinMPNN")
        os.makedirs(os.path.join(self.mpnn, "helper_scripts")); os.makedirs(os.path.join(self.mpnn, "soluble_model_weights")); os.makedirs(os.path.join(self.mpnn, "vanilla_model_weights"), exist_ok=True)
        open(os.path.join(self.mpnn, "helper_scripts", "parse_multiple_chains.py"), "w").write(STUB_PARSER)
        open(os.path.join(self.mpnn, "protein_mpnn_run.py"), "w").write(STUB_RUN)
        open(os.path.join(self.mpnn, "protein_mpnn_utils.py"), "w").write("# stub\n")
        self.head = git_checkout(self.mpnn)                                       # a checkout with its .git: protein_mpnn_run.py names its HEAD in the .fa headers
        self.pdbs = os.path.join(self.tmp, "pdbs"); os.makedirs(self.pdbs)
        for n in ("a1", "b2"):
            open(os.path.join(self.pdbs, n + ".pdb"), "w").write("ATOM\n")
        os.environ[stack.ENV_MPNN_DIR] = self.mpnn
        self.p_pins = mock.patch.object(stack, "check_pins", return_value=dict(PINNED)); self.p_pins.start()
        self.p_gpu = mock.patch.object(stack, "gpu_info", return_value=None); self.p_gpu.start()

    def tearDown(self):
        self.p_pins.stop(); self.p_gpu.stop()
        os.environ.clear(); os.environ.update(self._env)
        shutil.rmtree(self.tmp, ignore_errors=True)
        stack.reset_for_tests()


class TestUsage(Base):
    def test_usage_and_unknown(self):
        self.assertEqual(_run([])[0], report.EXIT_USAGE)
        self.assertEqual(_run(["--help"])[0], report.EXIT_OK)
        self.assertEqual(_run(["frobnicate"])[0], report.EXIT_USAGE)
        rc, _, err = _run(["version"])                                           # no version pseudo-verb: the tree's git commit names the kit
        self.assertEqual(rc, report.EXIT_USAGE); self.assertIn("unknown command 'version'", err)

    def test_check_dry_run_line(self):
        rc, out, err = _run(["check", "--mode", "exact", "--variant", "soluble"])
        self.assertEqual(rc, report.EXIT_NOT_ACTIVE, err)                   # no GPU on this box: the CUDA-graph levers cannot run — the mode is all of its levers, refused by name (the code design would give)
        self.assertIn("[proteinmpnn-opt] DRY-RUN mode=exact variant=soluble route=worker proteinmpnn=8907e667", err)
        self.assertIn("bb_batch=16", err); self.assertIn("probe=measured at launch partial=graph_rng,single_graph,fused_draw reason=mode exact is all of its levers and graph_rng,single_graph,fused_draw cannot run on this box (no CUDA device", err)
        self.assertNotIn("NOT ACTIVE: partial activation", err)                   # the reason names it; no second line
        with mock.patch.dict(os.environ, {stack.ENV_ALLOW_PARTIAL: "1"}):        # the environment spelling of the opt-out: the code design would give
            rc, out, err = _run(["check", "--mode", "exact", "--variant", "soluble"])
        self.assertEqual(rc, 0, err); self.assertNotIn("NOT ACTIVE", err)
        self.assertIn("[proteinmpnn-opt] PARTIAL allowed: graph_rng,single_graph,fused_draw: switched off by the worker's own CPU rule (no CUDA device) "
                      "(mode=exact variant=soluble) (--allow-partial, recorded)\n", err)
        with mock.patch.object(stack, "gpu_info", return_value=H100):
            rc, out, err = _run(["check", "--mode", "exact", "--variant", "soluble"])
        self.assertEqual(rc, 0, err); self.assertNotIn("partial=", err); self.assertNotIn("partial activation", err); self.assertNotIn("opted_out=", err)
        self.assertIn("hybrid_gemm", err.split(" levers=")[1].split(" ")[0])           # the probe-gated lever is requested by default
        with mock.patch.object(stack, "gpu_info", return_value=H100):
            rc, out, err = _run(["check", "--mode", "exact", "--variant", "soluble", "--hybrid_gemm", "0"])   # the lever's opt-out: left out of the set by name
        self.assertEqual(rc, 0, err); self.assertIn(" opted_out=hybrid_gemm", err); self.assertNotIn("hybrid_gemm", err.split(" levers=")[1].split(" ")[0])
        rc, out, err = _run(["check", "--mode", "exact", "--variant", "soluble", "--json"])
        self.assertEqual(json.loads(out)["opted_out"], [])
        rc, _, err = _run(["check", "--mode", "fast"])                          # no fast tier ships: refused by name before anything resolves, 3
        self.assertEqual(rc, report.EXIT_NOT_ACTIVE); self.assertIn("[proteinmpnn-opt] ERROR: proteinmpnn ships no fast tier: select --mode exact\n", err)
        self.assertNotIn("ACTIVE mode", err)
        rc, _, err = _run(["check", "--mode", "turbo"])                         # an unknown name is a usage error, 2
        self.assertEqual(rc, report.EXIT_USAGE); self.assertIn("unknown mode 'turbo': choose one of exact|off", err)
        rc, _, err = _run(["check", "--mode", "off"])
        self.assertEqual(rc, report.EXIT_USAGE)
        rc, _, err = _run(["check", "--mode", "exact", "--variant", "ligandmpnn"])       # no such variant: the two weight sets are the variants
        self.assertEqual(rc, report.EXIT_USAGE); self.assertIn("unknown variant 'ligandmpnn'", err)

    def test_env_disagreement_and_off_for_kit_only_commands(self):
        os.environ[stack.ENV_MODE] = "exact"
        rc, _, err = _run(["check", "--mode", "off"])
        self.assertEqual(rc, report.EXIT_USAGE); self.assertIn("disagrees", err)
        rc, _, err = _run(["warm", "--mode", "off"])
        self.assertEqual(rc, report.EXIT_USAGE)
        del os.environ[stack.ENV_MODE]
        rc, _, err = _run(["warm", "--mode", "fast"])
        self.assertEqual(rc, report.EXIT_NOT_ACTIVE); self.assertIn("proteinmpnn ships no fast tier: select --mode exact", err)
        rc, _, err = _run(["design", "--input", self.pdbs, "--out", os.path.join(self.tmp, "out_nomode")])   # no --mode, no $PROTEINMPNN_OPT: no default mode on this engine — a usage error, one line naming the modes served, nothing written
        self.assertEqual(rc, report.EXIT_USAGE); self.assertEqual(err.strip().splitlines(), ["[proteinmpnn-opt] ERROR: no mode named: --mode off|exact (or PROTEINMPNN_OPT) — proteinmpnn ships no fast tier, so it has no default mode"])
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "out_nomode"))); self.assertEqual(_run(["check"])[0], report.EXIT_USAGE); self.assertEqual(_run(["warm"])[0], report.EXIT_USAGE)
        os.environ[stack.ENV_MODE] = "fast"                                     # the environment's name is refused like the option's, before any work
        out = os.path.join(self.tmp, "out_fast")
        rc, _, err = _run(["design", "--input", self.pdbs, "--out", out])
        self.assertEqual(rc, report.EXIT_NOT_ACTIVE); self.assertIn("proteinmpnn ships no fast tier", err); self.assertFalse(os.path.exists(out))
        del os.environ[stack.ENV_MODE]

    def test_mode_help_lists_the_shipped_modes_only(self):
        p = cli.argparse.ArgumentParser(prog="x"); cli._common(p)
        text = p.format_help()
        self.assertIn("exact|off (else $PROTEINMPNN_OPT; no default:", text); self.assertNotIn("fast|", text); self.assertNotIn("|fast", text); self.assertIn("--det {0,1}", text)   # fast is never a choice; --det is on the surface

    def test_design_option_refusals(self):
        """What no route takes: a token upstream's argparse does not define, a kit lever, an input / output the package renders (exit 2, nothing launched)."""
        out = os.path.join(self.tmp, "out")
        for toks, word in ((["--num_seq", "8"], "--num_seq is not an option of the stock command line"), (["--settings", "x"], "--settings is not an option"),
                           (["--frobnicate"], "--frobnicate is not an option"), (["--x_all"], "--x_all is a kit lever"), (["--jsonl_path", "p"], "one of them, given once (got --jsonl_path and --input)")):
            for mode in ("off", "exact"):
                rc, _, err = _run(["design", "--mode", mode, "--input", self.pdbs, "--out", out] + toks)
                self.assertEqual(rc, report.EXIT_USAGE, (mode, toks)); self.assertIn(word, err); self.assertFalse(os.path.exists(out))
        for gone in (["--parser", "kit"], ["--quiet"], ["--json"]):                            # not options of design: upstream's argparse does not define them either
            rc, _, err = _run(["design", "--mode", "off", "--input", self.pdbs, "--out", out] + gone)
            self.assertEqual(rc, report.EXIT_USAGE, gone); self.assertIn(f"{gone[0]} is not an option of the stock command line", err)

    def test_exact_refuses_by_name_what_its_worker_cannot_serve(self):
        """A kit-mode pass with an option mode exact cannot serve is refused by name before any process: one NOT ACTIVE line naming each option with its
        mechanism, --mode off named as the stock path, the EXIT tally saying nothing was launched, exit 3, no design outputs; the manifest records the refusal."""
        cases = ((["--seed", "3", *KIT, "--ca_only"], ["--ca_only"]),
                 (["--score_only", "1", "--path_to_fasta", "x.fa"], ["--score_only=1"]),                                                # --path_to_fasta: read by the refused pass only, inert here
                 (["--seed", "3", "--tied_positions_jsonl", "t.jsonl", "--suppress_print", "1"], ["--tied_positions_jsonl=t.jsonl"]),   # --suppress_print: inert, never a reason
                 (["--conditional_probs_only", "1", "--unconditional_probs_only", "1"], ["--conditional_probs_only=1", "--unconditional_probs_only=1"]),
                 (["--backbone_noise", "0.2"], ["--backbone_noise=0.2"]),
                 (["--num_seq_per_target", "2", "--batch_size", "4"], ["--num_seq_per_target=2<--batch_size=4"]))
        for given, refused in cases:
            stack.reset_for_tests()
            out = os.path.join(self.tmp, "out_refused"); shutil.rmtree(out, ignore_errors=True)
            with mock.patch.object(kit_run, "run", side_effect=AssertionError("the kit route ran a pass its worker does not serve")), \
                 mock.patch.object(stock_run, "run", side_effect=AssertionError("a kit-mode pass ran the stock route under the mode's name")):
                rc, _, err = _run(["design", "--mode", "exact", "--input", self.pdbs, "--out", out] + given)
            self.assertEqual(rc, report.EXIT_NOT_ACTIVE, (given, err))
            na = [ln for ln in err.splitlines() if ln.startswith("[proteinmpnn-opt] NOT ACTIVE: cannot serve ")]
            self.assertEqual(len(na), 1, err)
            for token in refused:
                self.assertIn(token + " (", na[0])                                      # each option named, its mechanism in the parenthesis after it
            self.assertIn("under mode exact — refused by name, nothing launched (exit 3): --mode off runs the stock command line with these options (mode=exact variant=vanilla)", na[0])
            self.assertNotIn("] ACTIVE ", err); self.assertNotIn("] CMD ", err); self.assertNotIn("ENV-CLEAN", err)
            self.assertRegex(err, r"EXIT pid=\d+ nothing launched: no stock or kit process ran in this command")
            self.assertFalse(os.path.exists(os.path.join(out, "seqs"))); self.assertFalse(os.path.exists(os.path.join(out, "stub_seen.json")))
            man = manifest.read(out)
            self.assertEqual((man["active"], man["exit_code"], man["mode"], man["route"]), (False, 3, "exact", None)); self.assertEqual(man["refused"], refused)
            self.assertTrue(man["reason"].startswith("cannot serve " + refused[0] + " ("), man["reason"])
        stack.reset_for_tests()
        out = os.path.join(self.tmp, "out_off_serves"); shutil.rmtree(out, ignore_errors=True)
        rc, _, err = _run(["design", "--mode", "off", "--input", self.pdbs, "--out", out, "--ca_only", "--backbone_noise", "0.2"])   # the stock path named on the line runs them
        self.assertEqual(rc, 0, err)
        seen = json.load(open(os.path.join(out, "stub_seen.json")))
        self.assertEqual(seen["argv"][seen["argv"].index("--path_to_model_weights") + 2:], ["--ca_only", "--backbone_noise", "0.2"])

    def test_exact_serves_the_design_pass_options_on_its_worker(self):
        """The design pass's own options reach the kit worker verbatim under mode exact — upstream's default --seed 0 (the worker draws the seed as upstream
        draws it), several batches and temperatures, the residue dictionaries and the PSSM settings; --suppress_print is dropped (inert); one PDB file is
        parsed by the kit's parser from a staged directory holding it alone and its chains are assigned as protein_mpnn_run.py assigns a --pdb_path input's."""
        os.environ[stack.ENV_ALLOW_PARTIAL] = "1"                              # no GPU here: the CUDA-graph levers recorded as partial, the pass runs
        out = os.path.join(self.tmp, "out_serves")
        given = ["--num_seq_per_target", "4", "--batch_size", "2", "--sampling_temp", "0.1 0.2", "--omit_AA_jsonl", "o.jsonl", "--bias_AA_jsonl", "b.jsonl",
                 "--bias_by_res_jsonl", "r.jsonl", "--pssm_jsonl", "p.jsonl", "--pssm_multi", "0.3", "--pssm_bias_flag", "1", "--suppress_print", "1", "--path_to_model_weights", "", *KIT]
        with stub_stage_patch():
            rc, _, se = _run(["design", "--mode", "exact", "--input", self.pdbs, "--out", out] + given)
        self.assertEqual(rc, 0, se)
        self.assertNotIn("NOT ACTIVE", se)
        f = json.load(open(os.path.join(out, "stub_seen.json")))["flags"]
        self.assertEqual((f["--seed"], f["--num_seq_per_target"], f["--batch_size"], f["--sampling_temp"]), ("0", "4", "2", "0.1 0.2"))   # upstream's default seed 0 as given: the worker draws it as upstream does
        self.assertEqual((f["--omit_AA_jsonl"], f["--bias_AA_jsonl"], f["--bias_by_res_jsonl"], f["--pssm_jsonl"], f["--pssm_multi"], f["--pssm_bias_flag"]), ("o.jsonl", "b.jsonl", "r.jsonl", "p.jsonl", "0.3", "1"))
        self.assertNotIn("--suppress_print", f)                                # upstream's console verbosity: nothing the kit route computes
        self.assertTrue(f["--path_to_model_weights"].endswith("/vanilla_model_weights"), f["--path_to_model_weights"])   # given empty: upstream's own default directory
        man = manifest.read(out)
        self.assertEqual((man["route"], man["mode"], man["refused"], man["stock_args"]), ("worker", "exact", [], given))
        one = os.path.join(self.tmp, "one.pdb"); open(one, "w").write("ATOM\n")                            # a single PDB: protein_mpnn_run.py's --pdb_path
        for chains_given, assigned in (([], [["A", "B"], []]), (["--pdb_path_chains", "B"], [["B"], ["A"]]), (["--pdb_path_chains", "B A"], [["B", "A"], []])):
            stack.reset_for_tests()
            out = os.path.join(self.tmp, "out_one"); shutil.rmtree(out, ignore_errors=True)
            with stub_stage_patch():
                rc, _, err = _run(["design", "--mode", "exact", "--input", one, "--out", out, "--seed", "3"] + chains_given + KIT)
            self.assertEqual(rc, 0, err)
            seen = json.load(open(os.path.join(out, "stub_seen.json")))
            self.assertEqual(seen["chains"], {"one": assigned})                                             # protein_mpnn_run.py's assignment for a --pdb_path input: designed = --pdb_path_chains else all, fixed = the rest
            self.assertNotIn("--pdb_path_chains", seen["flags"]); self.assertNotIn("--pdb_path", seen["flags"])   # applied by the kit, not options of the worker
            self.assertEqual(parsed_names(os.path.join(out, "parsed.jsonl")), ["one"])                        # the kit parser's one entry, beside the outputs as for a directory
            cl = cmd_lines(err)
            self.assertEqual(len(cl), 2); self.assertIn("fast_parse.py", cl[0]["argv"]); self.assertIn("one_pdb", cl[0]["argv"])   # the parse step on the staged one-file directory, then the worker
            man = manifest.read(out)
            self.assertEqual((man["inputs"]["parser"], man["inputs"]["assigned"], man["inputs"]["n_items"], man["route"]), ("kit", None, 1, "worker"))


class TestStockRoute(Base):
    def _python_without_site(self):
        """A launcher for the proof child that sees no site-packages and no user site: the core is importable there only from the location env_proof hands it."""
        w = os.path.join(self.tmp, "python_no_site.sh")
        open(w, "w").write("#!/bin/sh\nexec %s -S -s \"$@\"\n" % sys.executable)
        os.chmod(w, os.stat(w).st_mode | stat.S_IXUSR)
        return w

    def test_proof_child_is_handed_the_core_explicitly(self):
        """The child imports opt_core.stock_proof from core_root() (put on its path for the one import, taken off before sys.path is read), not from
        whatever the cleaned environment happens to hold: no site, empty PYTHONPATH, and the proof still reports — ok, and free of the core's directory."""
        os.environ["PYTHONPATH"] = stack.kit_home()                                   # a kit path: clean_env strips it, the child's PYTHONPATH is empty
        env, removed = stock_run.clean_env()
        self.assertNotIn("PYTHONPATH", env); self.assertEqual(removed["pythonpath"], [stack.kit_home()])
        proof = stock_run.env_proof(env, python=self._python_without_site())
        self.assertTrue(proof["ok"], proof); self.assertEqual(proof["present"], []); self.assertEqual(proof["kit_dirs_on_path"], [])
        self.assertEqual(proof["core_root"], stock_run.core_root()); self.assertNotIn(stock_run.core_root(), proof["sys_path"])
        self.assertEqual(proof["core_modules_loaded"], [])                             # opt_core.stock_proof itself is the core's one exempt module

    def test_proof_child_that_cannot_report_is_a_stock_error(self):
        os.environ["PYTHONPATH"] = stack.kit_home()                                   # stripped: the child's PYTHONPATH is empty
        env, _ = stock_run.clean_env()
        with mock.patch.object(stock_run, "core_root", return_value=os.path.join(self.tmp, "no_core_here")):
            with self.assertRaises(stock_run.StockError) as cm:
                stock_run.env_proof(env, python=self._python_without_site())            # no site, no PYTHONPATH, a core root that holds no core
        self.assertIn("the proof child could not report (rc 1:", str(cm.exception)); self.assertIn("No module named 'opt_core'", str(cm.exception))
        with self.assertRaises(stock_run.StockError) as cm:
            stock_run.env_proof(env, python=os.path.join(self.tmp, "no_python_here"))
        self.assertIn("the proof child could not report", str(cm.exception))

    def test_off_is_fail_closed_on_an_unproven_environment(self):
        """StockError from the proof -> the design command exits 1 with its ERROR line; the stock command never starts (no stub_seen.json, no manifest claiming rc 0)."""
        os.environ[stack.ENV_MODE] = "off"
        out = os.path.join(self.tmp, "out_off_unproven")
        with mock.patch.object(stock_run, "env_proof", side_effect=stock_run.StockError("stock env proof: the proof child could not report (rc 1: boom)")):
            rc, so, se = _run(["design", "--input", self.pdbs, "--out", out, "--seed", "7"])
        self.assertEqual(rc, report.EXIT_FAIL, se)
        self.assertIn("[proteinmpnn-opt] ERROR: stock env proof: the proof child could not report (rc 1: boom)", se)
        self.assertNotIn("ENV-CLEAN ok", se)
        self.assertFalse(os.path.exists(os.path.join(out, "stub_seen.json")))

    def test_off_runs_the_stub_stock_line_in_a_clean_env(self):
        os.environ[stack.ENV_MODE] = "off"; os.environ["PROTEINMPNN_OPT_SOMETHING"] = "1"; os.environ["SELFTEST_SEEDS"] = "3"
        os.environ["PYTHONPATH"] = os.pathsep.join([stack.kit_home(), "/usr/lib/keepme"])
        out = os.path.join(self.tmp, "out_off")
        rc, so, se = _run(["design", "--input", self.pdbs, "--out", out, "--seed", "7"] + KIT)
        self.assertEqual(rc, 0, se)
        self.assertIn("[proteinmpnn-opt] NOT ACTIVE: mode off (stock route) (mode=off variant=vanilla) stock_args=--seed 7 --save_score 1 --save_probs 1", se)   # no --variant: upstream's own default weight set
        self.assertIn("[proteinmpnn-opt] ENV-CLEAN ok", se)
        self.assertRegex(se, r"\[proteinmpnn-opt\] EXIT pid=\d+ mode=off variant=vanilla route=stock rc=0 inputs=2 n_seqs=2 n_scores=2 n_probs=2")
        self.assertNotIn(" probe=", [ln for ln in se.splitlines() if "] EXIT pid=" in ln][0])            # the stock route requests no probe-gated lever: no probe= field
        seen = json.load(open(os.path.join(out, "stub_seen.json")))
        self.assertNotIn("PROTEINMPNN_OPT", seen["env"]); self.assertNotIn("PROTEINMPNN_OPT_SOMETHING", seen["env"]); self.assertNotIn("SELFTEST_SEEDS", seen["env"])
        self.assertEqual(seen["env"]["MPNN_DIR"], self.mpnn)
        self.assertEqual(seen["env"].get("PYTHONPATH"), "/usr/lib/keepme")
        self.assertEqual(seen["env"]["PYTHONDONTWRITEBYTECODE"], "1")
        argv = seen["argv"]
        self.assertEqual(argv[argv.index("--seed") + 1], "7")
        self.assertEqual(argv[argv.index("--path_to_model_weights") + 1], os.path.join(self.mpnn, "vanilla_model_weights"))   # protein_mpnn_run.py's own default directory, rendered
        self.assertEqual([a for a in argv if a.startswith("--")], ["--jsonl_path", "--out_folder", "--path_to_model_weights", "--seed", "--save_score", "--save_probs"])   # no --chain_id_jsonl without the caller's (upstream's default chain handling: every chain designed); the inputs, the output, the weights directory, the one option given — upstream's defaults for the rest
        self.assertEqual([f for f in os.listdir(out) if f.endswith(".jsonl")], ["parsed.jsonl"]); self.assertIsNone(manifest.read(out)["inputs"]["assigned"])   # no chain assignment file of the package's
        proof = json.load(open(os.path.join(out, stock_run.PROOF_FILE)))
        self.assertEqual(proof["present"], []); self.assertEqual(proof["kit_dirs_on_path"], [])
        self.assertIn("PROTEINMPNN_OPT", proof["removed"]["variables"]); self.assertIn("SELFTEST_SEEDS", proof["removed"]["variables"])
        self.assertEqual(proof["removed"]["pythonpath"], [stack.kit_home()])
        man = manifest.read(out)
        self.assertEqual((man["mode"], man["variant"], man["route"], man["exit_code"]), ("off", "vanilla", "stock", 0))
        self.assertFalse(man["active"]); self.assertEqual(man["outputs"]["counts"], {"seqs": 2, "scores": 2, "probs": 2})
        self.assertEqual(man["stock_args"], ["--seed", "7"] + KIT); self.assertEqual(man["inputs"]["parser"], "stock"); self.assertEqual(man["inputs"]["n_items"], 2)
        self.assertEqual(man["stock"]["env_proof"]["present"], []); self.assertEqual(man["outputs"]["listing"], ["seqs/a1.fa", "seqs/b2.fa", "scores/a1.npz", "scores/b2.npz", "probs/a1.npz", "probs/b2.npz"])   # the file listing: names, no digests
        self.assertEqual(man["env"]["PROTEINMPNN_OPT"], "off"); self.assertNotIn("staged", man["stock"]); self.assertNotIn("sha256", json.dumps(man["inputs"])); self.assertNotIn("fa_headers", man["outputs"])
        self.assertEqual(parsed_names(man["inputs"]["parsed"]), ["b2", "a1"]); self.assertNotIn("parsed_order", man["inputs"])   # the stub helper lists b2 first: the script reads the helper's own order, as upstream's user does
        self.assertEqual(seen["cwd"], os.getcwd())                                      # the child's cwd is the caller's
        self.assertEqual(sorted(os.listdir(out)), ["opt_manifest.json", "parsed.jsonl", "probs", "scores", "seqs", stock_run.PROOF_FILE, "stub_seen.json"])   # no stage, no logs: upstream's layout, the two chain files, the manifest and the proof
        self.assertEqual([c["step"] for c in man["stock"]["commands"]], ["parse", "design"])
        design = man["stock"]["commands"][1]["cmd"]
        self.assertEqual(design[1], os.path.join(self.mpnn, "protein_mpnn_run.py")); self.assertEqual(design[2:], argv)   # the checkout's own script, where it is; the stub saw exactly that argv
        self.assertEqual(man["stock"]["commands"][0]["cmd"][1], os.path.join(self.mpnn, "helper_scripts", "parse_multiple_chains.py"))
        cmds = cmd_lines(se)                                                             # one CMD line per process launched: the parse step, then the design
        self.assertEqual(len(cmds), 2, se); self.assertEqual([(c["route"], c["variant"]) for c in cmds], [("stock", "vanilla")] * 2)
        self.assertEqual(shlex.split(cmds[1]["argv"]), design); self.assertIn("--seed 7", cmds[1]["argv"]); self.assertIn("parse_multiple_chains.py", cmds[0]["argv"])
        self.assertLess(se.index("] ENV-CLEAN"), se.index("] CMD ")); self.assertLess(se.index("] CMD "), se.index("] EXIT "))

    def test_off_runs_a_checkout_without_git(self):
        """A checkout without .git runs as upstream runs it (git_hash=unknown in the headers); the checkout's commit is check's report, never design's gate."""
        shutil.rmtree(os.path.join(self.mpnn, ".git"))
        out = os.path.join(self.tmp, "out_nogit")
        rc, _, se = _run(["design", "--mode", "off", "--input", self.pdbs, "--out", out])
        self.assertEqual(rc, 0, se); self.assertNotIn("Traceback", se)
        self.assertTrue(os.path.exists(os.path.join(out, "stub_seen.json"))); self.assertEqual(len(cmd_lines(se)), 2)

    def test_off_with_parsed_jsonl_and_chain_id_jsonl(self):
        out = os.path.join(self.tmp, "out_p")
        parsed = os.path.join(self.tmp, "p.jsonl")
        open(parsed, "w").write(json.dumps({"name": "x", "seq_chain_A": "M", "seq_chain_C": "G"}) + "\n")
        cid = os.path.join(self.tmp, "cid.jsonl"); open(cid, "w").write(json.dumps({"x": [["A"], ["C"]]}) + "\n")
        rc, _, se = _run(["design", "--mode", "off", "--input", parsed, "--out", out, "--chain_id_jsonl", cid])
        self.assertEqual(rc, 0, se)
        seen = json.load(open(os.path.join(out, "stub_seen.json")))
        self.assertEqual(seen["argv"][seen["argv"].index("--chain_id_jsonl") + 1], cid)                 # the caller's --chain_id_jsonl, verbatim
        man = manifest.read(out)
        self.assertEqual((man["inputs"]["parsed"], man["inputs"]["parser"], man["inputs"]["assigned"]), (parsed, None, cid)); self.assertEqual(len(cmd_lines(se)), 1)   # the caller's files, read as given: no parse step

    def test_off_without_checkout_is_not_active(self):
        os.environ[stack.ENV_MPNN_DIR] = os.path.join(self.tmp, "nowhere")
        out = os.path.join(self.tmp, "out_x")
        with mock.patch.object(stack, "check_pins", return_value={"pinned": False, "findings": ["MPNN_DIR is not a directory"], "detail": {}}):
            rc, _, se = _run(["design", "--mode", "off", "--input", self.pdbs, "--out", out])
        self.assertEqual(rc, report.EXIT_NOT_ACTIVE); self.assertIn("stock pin", se)
        self.assertIn("EXIT pid=", se); self.assertIn("nothing launched", se)


class TestKitRoute(Base):
    def test_exact_stages_and_launches_the_worker(self):
        out = os.path.join(self.tmp, "out_exact")
        real_stage = kit_run._stage.stage_base

        def stub_stage(stage_dir):
            paths = real_stage(stage_dir)
            open(paths["worker"], "w").write(STUB_WORKER)          # the staged copy is disposable: the stub replaces the worker there only
            open(paths["worker_lowmem"], "w").write(STUB_WORKER)   # and the derived executable the base variants' exact line runs
            open(paths["fast_parse"], "w").write(STUB_PARSER)
            return paths
        argv = ["design", "--mode", "exact", "--variant", "vanilla", "--input", self.pdbs, "--out", out, "--seed", "11"] + KIT
        with mock.patch.object(kit_run._stage, "stage_base", stub_stage):
            rc, so, se = _run(argv)                                             # no GPU on this box: the CUDA-graph levers cannot run — the mode is all of its levers: refused by name, nothing launches
        self.assertEqual(rc, report.EXIT_NOT_ACTIVE, se)
        self.assertIn("[proteinmpnn-opt] ERROR: mode exact is all of its levers and graph_rng,single_graph,fused_draw cannot run on this box (no CUDA device: the worker's CUDA-graph levers) — "
                      "--mode off runs the stock command line here; --allow-partial (or PROTEINMPNN_OPT_ALLOW_PARTIAL=1) runs the levers that can, recorded\n", se)
        self.assertNotIn("] ACTIVE ", se); self.assertNotIn("] CMD ", se); self.assertIn("nothing launched", se)   # no worker, no parse step
        man = manifest.read(out)
        self.assertEqual((man["exit_code"], man["active"]), (3, False)); self.assertIn("all of its levers", man["reason"])
        self.assertFalse(os.path.exists(os.path.join(out, "seqs")))
        shutil.rmtree(out); stack.reset_for_tests()
        with mock.patch.object(kit_run._stage, "stage_base", stub_stage):
            rc, so, se = _run(argv + ["--allow-partial"])                        # recorded, the pass proceeds, the exit is its own
        self.assertEqual(rc, 0, se)
        self.assertIn("[proteinmpnn-opt] ACTIVE mode=exact variant=vanilla proteinmpnn=8907e667 gpu=none levers=", se)
        self.assertIn("stream,sort_by_length,chunk_gemm,cache_enc_ctx,graph_rng,single_graph,fused_draw,analytic_offsets,stock_shape_enc,hybrid_gemm,lowmem,bb_batch=16 fallbacks=none partial=graph_rng,single_graph,fused_draw allow_partial=yes", se)
        self.assertRegex(se, r"EXIT pid=\d+ mode=exact variant=vanilla route=worker rc=0 inputs=2 n_seqs=2 wall_s=[0-9.]+ probe=PASS partial=graph_rng,single_graph,fused_draw allow_partial=yes exit=0")
        self.assertIn("[proteinmpnn-opt] PARTIAL allowed: graph_rng,single_graph,fused_draw: switched off by the worker's own CPU rule (no CUDA device) "
                      "(mode=exact variant=vanilla) (--allow-partial, recorded)\n", se)
        self.assertNotIn("NOT ACTIVE", se)
        seen = json.load(open(os.path.join(out, "stub_seen.json")))
        f = seen["flags"]
        self.assertEqual((f["--mode"], f["--bb_batch"], f.get("--sort_by_length"), f.get("--x_all")), ("stream", "16", True, True))
        self.assertEqual(f["--seed"], "11"); self.assertEqual(f["--path_to_model_weights"], os.path.join(self.mpnn, "vanilla_model_weights"))
        self.assertEqual(os.path.basename(f["--chain_id_jsonl"]), inputs.UNASSIGNED); self.assertIsNone(seen["chains"])   # no --chain_id_jsonl given: the worker is handed the staged `null` line = protein_mpnn_run.py's chain_id_dict None (every chain designed, none fixed); gone with the stage
        self.assertFalse(os.path.exists(f["--chain_id_jsonl"]))
        man = manifest.read(out)
        self.assertIsNone(man["inputs"]["assigned"])
        self.assertEqual(parsed_names(man["inputs"]["parsed"]), ["b2", "a1"]); self.assertNotIn("parsed_order", man["inputs"])   # the kit parser lists b2 first, as the stock helper does on this box: the worker reads that order
        cmds = cmd_lines(se)                                                             # one CMD line per process launched: the parse step, then the worker's whole argv
        self.assertEqual(len(cmds), 2, se); self.assertEqual([(c["route"], c["variant"]) for c in cmds], [("kit", "vanilla")] * 2); cmds = cmds[1:]
        kit_cmd = shlex.split(cmds[0]["argv"])
        self.assertEqual(os.path.basename(kit_cmd[1]), "mpnn_worker2_lowmem.py"); self.assertIn("--seed 11", cmds[0]["argv"]); self.assertIn("--x_all", kit_cmd)
        self.assertEqual(kit_cmd, [c for c in man["kit"]["commands"] if c["step"] == "design"][0]["cmd"])
        self.assertLess(se.index("] ACTIVE "), se.index("] CMD ")); self.assertLess(se.index("] CMD "), se.index("] EXIT "))
        self.assertNotIn("--backbone_noise", f)                                          # the worker's built-in option never reaches its argv
        self.assertEqual((f["--save_score"], f["--save_probs"]), ("1", "1"))                # upstream's write gates reach it verbatim (its own options, upstream's meaning and default)
        self.assertEqual((f["--num_seq_per_target"], f["--batch_size"], f["--sampling_temp"], f["--omit_AAs"], f["--model_name"]), ("1", "1", "0.1", "X", "v_48_020"))   # not given: upstream's defaults, handed explicitly
        self.assertEqual(man["stock_args"], ["--seed", "11"] + KIT)
        self.assertEqual(seen["env"]["MPNN_DIR"], self.mpnn); self.assertNotIn("PROTEINMPNN_OPT", seen["env"])
        self.assertFalse(os.path.exists(os.path.join(out, "_stage"))); self.assertIn(os.sep + "mpnn_pdb_parser" + os.sep + "kit" + os.sep, kit_cmd[1])   # staged in a temporary directory, removed when the pass ends
        self.assertEqual(sorted(os.listdir(out)), ["opt_manifest.json", "parsed.jsonl", "seqs", "stub_seen.json"])   # no logs, no stage, no chain file of the package's beside the outputs
        man = manifest.read(out)
        self.assertTrue(man["active"]); self.assertEqual(man["applied"], "launched"); self.assertEqual(man["route"], "worker")
        self.assertEqual((man["exit_code"], man["partial"], man["allow_partial"]), (0, ["graph_rng", "single_graph", "fused_draw"], True))
        self.assertEqual(man["levers_applied"], ["stream", "sort_by_length", "chunk_gemm", "cache_enc_ctx", "analytic_offsets", "stock_shape_enc", "hybrid_gemm", "lowmem"])
        self.assertEqual(man["kit"]["worker_record"]["mode"], "stream"); self.assertEqual(man["env"][stack.ENV_ALLOW_PARTIAL], None)
        self.assertEqual(man["kit"]["lines"]["device_cell"], ["DEVICE CELL: cpu n/a torch stub cuda None"])
        self.assertEqual(man["inputs"]["parser"], "kit"); self.assertEqual(man["line"], "kit/mpnn_worker2.py <stock args> --mode stream --bb_batch 16 --sort_by_length --x_all --hybrid_gemm (run as kit/mpnn_worker2_lowmem.py: + the low-memory featuriser and decoding-order mask, lowmem.py)")
        self.assertEqual(man["kit"]["lowmem"], {"executable": "mpnn_worker2_lowmem.py"}); self.assertEqual((man["opted_out"], man["refused"]), ([], []))
        self.assertIsNone(man["stock"])
        # the tree's own worker is untouched by the stub
        n_ok, bad = stack.check_kits()
        self.assertEqual(bad, [])


    def test_fused_draw_refusal_refuses_the_job_by_name(self):
        """On a stack without Triton the worker's draw kernel cannot run: the worker prints `fused_draw: REFUSED`, emits its refusal record and exits 3 before
        reading any input; design relays it as the NOT ACTIVE line naming the lever and --mode off, gated names fused_draw (not another lever), exit 3."""
        out = os.path.join(self.tmp, "out_no_triton")
        argv = ["design", "--mode", "exact", "--variant", "soluble", "--input", self.pdbs, "--out", out, "--seed", "11"] + KIT
        with stub_stage_patch(), mock.patch.dict(os.environ, {"STUB_CUDA": "1", "STUB_TRITON": "0"}), mock.patch.object(stack, "gpu_info", return_value=H100):
            rc, so, se = _run(argv)
        self.assertEqual(rc, report.EXIT_NOT_ACTIVE, se)
        self.assertIn("fused_draw: REFUSED", so)
        self.assertIn("[proteinmpnn-opt] NOT ACTIVE: mode exact refused by name on this device: fused_draw: triton unavailable — the lever cannot engage bit-identically here "
                      "and the line is all of its levers; nothing was designed (--mode off runs the stock command line) (mode=exact variant=soluble)", se)
        self.assertRegex(se, r"EXIT pid=\d+ mode=exact variant=soluble route=worker rc=3 inputs=2 wall_s=[0-9.]+ probe=unobserved exit=3")   # hybrid_gemm requested, never probed: named, not guessed
        man = manifest.read(out)
        self.assertEqual((man["exit_code"], man["active"], man["partial"], man["gated"]), (3, False, [], {"fused_draw": "fused_draw probe: FAIL"}))
        self.assertEqual(man["kit"]["worker_record"]["refused"], "fused_draw: triton unavailable"); self.assertEqual(man["outputs"]["counts"], {})

    def test_note_lines_name_what_the_options_imply_on_both_routes(self):
        """--num_seq_per_target that is not a whole number of --batch_size batches designs whole batches only (upstream's rule, both routes): one NOTE line
        with the counts and the manifest's designs_per_target; a --chain_id_jsonl beside a single-PDB input is not read (upstream's rule): one NOTE line."""
        os.environ[stack.ENV_ALLOW_PARTIAL] = "1"                              # no GPU here: the CUDA-graph levers recorded as partial, the exact pass runs
        note = "[proteinmpnn-opt] NOTE designs per target requested=6 produced=4 (upstream's rule: whole batches of 4)"
        for mode in ("off", "exact"):
            stack.reset_for_tests()
            out = os.path.join(self.tmp, "out_note_" + mode); ctx = stub_stage_patch() if mode == "exact" else contextlib.nullcontext()
            with ctx:
                rc, _, se = _run(["design", "--mode", mode, "--input", self.pdbs, "--out", out, "--seed", "3", "--num_seq_per_target", "6", "--batch_size", "4"])
            self.assertEqual(rc, 0, se); self.assertEqual(se.count(note), 1, se); self.assertLess(se.index("] NOTE "), se.index("] CMD "))
            man = manifest.read(out)
            self.assertEqual((man["designs_per_target"], man["notes"]), ({"requested": 6, "batch_size": 4, "produced": 4}, [note[len("[proteinmpnn-opt] NOTE "):]]))
            stack.reset_for_tests()
            with ctx if mode == "off" else stub_stage_patch():
                rc, _, se = _run(["design", "--mode", mode, "--input", self.pdbs, "--out", out + "_whole", "--seed", "3", "--num_seq_per_target", "8", "--batch_size", "4"])
            self.assertEqual(rc, 0, se); self.assertNotIn("] NOTE ", se); self.assertEqual(manifest.read(out + "_whole")["notes"], [])
        one = os.path.join(self.tmp, "lone.pdb"); open(one, "w").write("ATOM\n"); chains = os.path.join(self.tmp, "chains.jsonl"); open(chains, "w").write('{"lone": [["A"], ["B"]]}\n')
        for mode in ("off", "exact"):
            stack.reset_for_tests()
            out = os.path.join(self.tmp, "out_lone_" + mode)
            with stub_stage_patch() if mode == "exact" else contextlib.nullcontext():
                rc, _, se = _run(["design", "--mode", mode, "--input", one, "--out", out, "--seed", "3", "--chain_id_jsonl", chains])
            self.assertEqual(rc, 0, se)
            self.assertEqual(se.count(f"[proteinmpnn-opt] NOTE --chain_id_jsonl {chains} not read for a single-PDB input (protein_mpnn_run.py assigns a --pdb_path input's chains from --pdb_path_chains, else designs every chain)"), 1, se)

    def test_probe_fail_refuses_the_job_by_name(self):
        """The exact line requests the worker's probe-gated --hybrid_gemm; a probe FAIL means the lever cannot engage bit-identically on this device, and the
        line is all of its levers: the worker refuses the job by name before any output (rc 3, `refused` in its record), design exits 3 naming
        --hybrid_gemm 0 (the line without the lever, by name); --allow-partial does not apply (nothing partial ran)."""
        out = os.path.join(self.tmp, "out_probe_fail")
        argv = ["design", "--mode", "exact", "--variant", "soluble", "--input", self.pdbs, "--out", out, "--seed", "11"] + KIT
        for extra in ([], ["--allow-partial"]):
            stack.reset_for_tests()
            with stub_stage_patch(), mock.patch.dict(os.environ, {"STUB_CUDA": "1", "STUB_PROBE": "FAIL"}), mock.patch.object(stack, "gpu_info", return_value=H100):
                rc, so, se = _run(argv + extra)
            self.assertEqual(rc, report.EXIT_NOT_ACTIVE, se)
            self.assertIn("hybrid_gemm: REFUSED", so)                                                             # the worker's own line, relayed
            self.assertIn("[proteinmpnn-opt] NOT ACTIVE: mode exact refused by name on this device: hybrid_gemm: PROBE FAIL — the lever cannot engage bit-identically here "
                          "and the line is all of its levers; nothing was designed (--hybrid_gemm 0 runs the line without the lever, by name) (mode=exact variant=soluble)", se)
            self.assertRegex(se, r"EXIT pid=\d+ mode=exact variant=soluble route=worker rc=3 inputs=2 wall_s=[0-9.]+ probe=FAIL%s exit=3" % (" allow_partial=yes" if extra else ""))
            self.assertNotIn(" partial=", se); self.assertNotIn("PARTIAL allowed", se); self.assertNotIn("NOT ACTIVE: partial", se)
            man = manifest.read(out)
            self.assertEqual((man["exit_code"], man["active"], man["partial"], man["gated"]), (3, False, [], {"hybrid_gemm": "hybrid_gemm probe: FAIL"}))
            self.assertEqual((man["probe"]["requested"], man["probe"]["verdict"], man["probe"]["worker"]["verdict"]), (True, "FAIL", "PROBE FAIL -> stock-shape GEMMs (stub)"))   # the worker's word folded into the report, its cell kept whole
            self.assertEqual(man["kit"]["worker_record"]["refused"], "hybrid_gemm: PROBE FAIL"); self.assertEqual(man["outputs"]["counts"], {})
            self.assertNotIn("probe=PASS", se); self.assertEqual(se.count("probe=FAIL"), 1)
            shutil.rmtree(out)
        stack.reset_for_tests()
        with stub_stage_patch(), mock.patch.dict(os.environ, {"STUB_CUDA": "1"}), mock.patch.object(stack, "gpu_info", return_value=H100):
            rc, so, se = _run(["design", "--mode", "exact", "--variant", "soluble", "--input", self.pdbs, "--out", out + "_pass", "--seed", "11"] + KIT)
        self.assertEqual(rc, 0, se)                                                # probe PASS: applied, complete, exit 0
        man = manifest.read(out + "_pass")
        self.assertIn("hybrid_gemm", man["levers_applied"]); self.assertEqual(man["partial"], []); self.assertEqual(man["gated"], {})
        self.assertEqual(man["probe"]["verdict"], "PASS"); self.assertRegex(se, r"EXIT pid=\d+ mode=exact variant=soluble route=worker rc=0 inputs=2 n_seqs=2 wall_s=[0-9.]+ probe=PASS exit=0")
        stack.reset_for_tests()
        with stub_stage_patch(), mock.patch.dict(os.environ, {"STUB_CUDA": "1", "STUB_PROBE": "FAIL"}), mock.patch.object(stack, "gpu_info", return_value=H100):
            rc, so, se = _run(argv + ["--hybrid_gemm", "0"])                         # the opt-out: the lever is not requested, so a box whose probe would refuse it runs the mode complete, exit 0
        self.assertEqual(rc, 0, se)
        man = manifest.read(out)
        self.assertEqual((man["opted_out"], man["partial"], man["gated"], man["probe"]["requested"]), (["hybrid_gemm"], [], {}, False))
        self.assertNotIn("hybrid_gemm", man["levers_applied"]); self.assertNotIn("--hybrid_gemm", json.load(open(os.path.join(out, "stub_seen.json")))["flags"])   # the worker is not handed the flag
        self.assertIn(" opted_out=hybrid_gemm", se); self.assertRegex(se, r"EXIT pid=\d+ mode=exact variant=soluble route=worker rc=0 inputs=2 n_seqs=2 wall_s=[0-9.]+ exit=0"); self.assertNotIn(" probe=", se.splitlines()[-1])   # no probe-gated lever requested: no probe= field

    def _stub_stage(self):
        return stub_stage_patch()

    def _design(self, out, *more, env=None):
        stack.reset_for_tests()                                              # one launch per process: a fresh process per design run
        with mock.patch.dict(os.environ, {"STUB_CUDA": "1", **(env or {})}), self._stub_stage():     # the stub worker sees the mocked GPU
            return _run(["design", "--mode", "exact", "--variant", "soluble", "--input", self.pdbs, "--out", out, "--seed", "11", *KIT, *more])

    def test_a_lever_the_worker_ran_without_is_partial_by_name(self):
        """The worker's own end-of-run record is the evidence: a requested lever it shows off is `partial` (3), on a GPU box too."""
        out = os.path.join(self.tmp, "out_drop")
        with mock.patch.object(stack, "gpu_info", return_value=H100):
            rc, _, se = self._design(out, env={"STUB_DROP": "chunk_gemm"})
            self.assertEqual(rc, report.EXIT_NOT_ACTIVE, se)
            man = manifest.read(out)
            self.assertEqual((man["partial"], man["levers_unavailable"], man["levers_fallback"]), (["chunk_gemm"], [], ["chunk_gemm"]))
            self.assertIn("partial activation: chunk_gemm", man["reason"]); self.assertIn(" partial=chunk_gemm exit=3", se)
            self.assertIn("[proteinmpnn-opt] NOT ACTIVE: partial activation — chunk_gemm: the worker's end-of-run record shows the lever off "
                          "(mode=exact variant=soluble); exit 3 (--allow-partial records and proceeds)\n", se)
            shutil.rmtree(out)
            rc, _, se = self._design(out, "--allow-partial", env={"STUB_DROP": "chunk_gemm"})
            self.assertEqual(rc, 0, se)
            man = manifest.read(out)
            self.assertEqual((man["exit_code"], man["partial"], man["allow_partial"]), (0, ["chunk_gemm"], True))
            shutil.rmtree(out)
            rc, _, se = self._design(out)                                     # nothing dropped on a GPU box: not partial, no flag needed
            self.assertEqual(rc, 0, se); self.assertNotIn("partial", se)
            man = manifest.read(out)
            self.assertEqual((man["partial"], man["levers_fallback"], man["levers_unobserved"]), ([], [], ["sort_by_length"]))

    def test_the_worker_starts_beside_the_parse_step(self):
        """A PDB directory: the parse step's and the worker's CMD lines are both printed before either runs, the worker carries --inputs_ready <stage>/inputs.ready
        and reads the parsed jsonl once the driver has created it (the stub waits for it as the worker does); the outputs are the same as ever."""
        out = os.path.join(self.tmp, "out_beside")
        with mock.patch.object(stack, "gpu_info", return_value=H100):
            rc, _, se = self._design(out)
            self.assertEqual(rc, 0, se)
            cmds = [l for l in se.splitlines() if l.startswith("[proteinmpnn-opt] CMD route=kit")]
            self.assertEqual(len(cmds), 2); self.assertIn("fast_parse.py", cmds[0]); self.assertIn("mpnn_worker2_lowmem.py", cmds[1]); self.assertIn("--inputs_ready", cmds[1])
            seen = json.load(open(os.path.join(out, "stub_seen.json")))
            self.assertTrue(seen["flags"]["--inputs_ready"].endswith("/" + kit_run.INPUTS_READY))
            self.assertEqual(sorted(os.listdir(os.path.join(out, "seqs"))), sorted(n[:-4] + ".fa" for n in os.listdir(self.pdbs) if n.endswith(".pdb")))

    def test_no_worker_record_is_missing_evidence_not_partial(self):
        out = os.path.join(self.tmp, "out_norec")
        with mock.patch.object(stack, "gpu_info", return_value=H100):
            rc, _, se = self._design(out, "--allow-partial", env={"STUB_NO_RECORD": "1"})
        self.assertEqual(rc, report.EXIT_NOT_ACTIVE, se)                    # whatever the switch: the levers cannot be judged
        man = manifest.read(out)
        self.assertEqual(man["partial"], []); self.assertTrue(man["partial_detection"].startswith("none ("))
        self.assertIn("no end-of-run record", man["reason"])

    def test_a_worker_out_of_memory_is_exit_1_and_nothing_reruns(self):
        """OOM propagates; no fallback is applied. The package applies no lever in-process and imports no torch: the worker that runs out of memory
        (the stub: torch's traceback line, exit 1) is design's exit 1 with the worker's stderr kept — one launch, no stock rerun, no lever-less retry."""
        out = os.path.join(self.tmp, "out_oom")
        with mock.patch.object(stack, "gpu_info", return_value=H100), mock.patch.object(stock_run, "run", side_effect=AssertionError("the stock route reran")):
            rc, so, se = self._design(out, env={"STUB_OOM": "1"})
        self.assertEqual(rc, report.EXIT_FAIL, se)
        self.assertIn("torch.OutOfMemoryError: CUDA out of memory (mock)", se); self.assertFalse(os.path.exists(os.path.join(out, "kit_stderr.log")))   # the worker's stderr is the caller's; no log file
        man = manifest.read(out)
        self.assertEqual((man["exit_code"], man["outputs"]["counts"].get("seqs", 0), man["levers_fallback"], man["active"]), (1, 0, [], True))
        self.assertEqual([c["rc"] for c in man["kit"]["commands"] if c["step"] == "design"], [1])          # the one launch, its exit code the record
        self.assertRegex(se, r"EXIT pid=\d+ .* rc=1 .*exit=1")
        self.assertFalse(os.path.exists(os.path.join(out, "stub_seen.json")))                                  # nothing ran past the death

    def test_outputs_short_of_the_inputs_are_incomplete(self):
        out = os.path.join(self.tmp, "out_short")
        with mock.patch.object(stack, "gpu_info", return_value=H100):
            rc, _, se = self._design(out, env={"STUB_SHORT": "1"})
        self.assertEqual(rc, report.EXIT_FAIL, se)
        man = manifest.read(out)
        self.assertEqual((man["incomplete"], man["partial"], man["active"]), ("1/2", [], True))
        self.assertRegex(se, r"EXIT pid=\d+ .* rc=0 inputs=2 n_seqs=1 wall_s=[0-9.]+ probe=PASS incomplete=1/2 exit=1")

    def test_warm_follows_design(self):
        from proteinmpnn_opt import warm
        seen = []

        def fake_design(argv, echo=True):
            seen.append(list(argv)); self.assertTrue(echo)                     # warm relays the pass's lines as design does
            os.makedirs(os.path.join(argv[argv.index("--out") + 1], "seqs"), exist_ok=True)
            return report.EXIT_NOT_ACTIVE if "--allow-partial" not in argv and "--hybrid_gemm" not in argv else 0
        with mock.patch.object(warm, "public_input", return_value=self.pdbs), mock.patch.object(cli, "cmd_design", fake_design):
            self.assertEqual(_run(["warm", "--mode", "exact", "--variant", "soluble"])[0], report.EXIT_NOT_ACTIVE)
            self.assertNotIn("--allow-partial", seen[-1]); self.assertNotIn("--hybrid_gemm", seen[-1])
            self.assertEqual(seen[-1][seen[-1].index("--out") + 2:], warm.WARM_STOCK_ARGS)               # warm's own stock options close the argv
            rc, out, _ = _run(["warm", "--mode", "exact", "--variant", "soluble", "--allow-partial"])
            self.assertIn("--allow-partial", seen[-1]); self.assertEqual(rc, report.EXIT_FAIL)     # rc 0 with no .fa: FAIL, design's own verdict
            _run(["warm", "--mode", "exact", "--variant", "soluble", "--hybrid_gemm", "0"])          # the opt-out is warm's too: forwarded as design's --hybrid_gemm 0
            i = seen[-1].index("--hybrid_gemm"); self.assertEqual(seen[-1][i:i + 2], ["--hybrid_gemm", "0"])

    def test_exact_refuses_lever_tokens_before_anything_is_staged(self):
        out = os.path.join(self.tmp, "out_l")
        rc, _, se = _run(["design", "--mode", "exact", "--input", self.pdbs, "--out", out, "--seed", "11", *KIT,
                          "--chunk_gemm", "--bb_batch", "32"])
        self.assertEqual(rc, report.EXIT_USAGE)
        self.assertIn("--chunk_gemm is a kit lever", se)
        self.assertFalse(os.path.exists(os.path.join(out, "_stage"))); self.assertFalse(os.path.exists(os.path.join(out, "stub_seen.json")))
        for extra in (["--frobnicate"], ["--dtype", "bf16"], ["--jsonl_path", "x"], ["--", "--max_length", "5"]):
            rc, _, se = _run(["design", "--mode", "exact", "--input", self.pdbs, "--out", out, "--seed", "11", *KIT] + extra)
            self.assertEqual(rc, report.EXIT_USAGE, extra); self.assertIn(extra[0], se); self.assertFalse(os.path.exists(out))

    def test_exact_passes_a_stock_option_the_worker_takes_and_reports_it(self):
        out = os.path.join(self.tmp, "out_e")
        real_stage = kit_run._stage.stage_base

        def stub_stage(stage_dir):
            paths = real_stage(stage_dir)
            open(paths["worker"], "w").write(STUB_WORKER); open(paths["worker_lowmem"], "w").write(STUB_WORKER); open(paths["fast_parse"], "w").write(STUB_PARSER)
            return paths
        os.environ[stack.ENV_ALLOW_PARTIAL] = "1"                              # the environment spelling of --allow-partial (no GPU here)
        with mock.patch.object(kit_run._stage, "stage_base", stub_stage):
            rc, _, se = _run(["design", "--mode", "exact", "--input", self.pdbs, "--out", out, "--seed", "11", "--max_length", "500", *KIT])
        self.assertEqual(rc, 0, se)
        self.assertRegex(se, r"\[proteinmpnn-opt\] ACTIVE mode=exact variant=vanilla .* partial=graph_rng,single_graph,fused_draw allow_partial=yes stock_args=--seed 11 --max_length 500 --save_score 1 --save_probs 1\n")
        f = json.load(open(os.path.join(out, "stub_seen.json")))["flags"]
        self.assertEqual(f["--max_length"], "500"); self.assertEqual((f["--mode"], f["--bb_batch"], f.get("--x_all")), ("stream", "16", True))
        man = manifest.read(out)
        self.assertEqual(man["stock_args"], ["--seed", "11", "--max_length", "500"] + KIT); self.assertEqual(man["activation_report"]["stock_args"], man["stock_args"])
        self.assertTrue(man["allow_partial"]); self.assertEqual(man["env"][stack.ENV_ALLOW_PARTIAL], "1")
        self.assertNotIn(stack.ENV_ALLOW_PARTIAL, json.load(open(os.path.join(out, "stub_seen.json")))["env"])   # a package name: never reaches the worker
        self.assertNotIn("sha256", man["inputs"]); self.assertEqual((man["inputs"]["parser"], man["inputs"]["n_items"]), ("kit", 2))
        self.assertIsInstance(man["numpy"], str); self.assertIn("torch", man)

    def test_upstreams_own_input_and_output_names_run_as_given(self):
        """--jsonl_path / --pdb_path / --out_folder are read as protein_mpnn_run.py reads them: the same child command line as --input / --out."""
        out = os.path.join(self.tmp, "out_names"); parsed = os.path.join(self.tmp, "given.jsonl")
        with open(parsed, "w") as fh:
            for n in ("a1", "b2"):
                fh.write(json.dumps({"name": n, "seq": "ACD", "seq_chain_A": "ACD", "num_of_chains": 1}) + "\n")
        rc, _, se = _run(["design", "--mode", "off", "--jsonl_path", parsed, "--out_folder", out, "--seed", "7"])
        self.assertEqual(rc, 0, se)
        argv = json.load(open(os.path.join(out, "stub_seen.json")))["argv"]
        self.assertEqual([a for a in argv if a.startswith("--")], ["--jsonl_path", "--out_folder", "--path_to_model_weights", "--seed"])
        self.assertEqual((argv[argv.index("--jsonl_path") + 1], argv[argv.index("--out_folder") + 1]), (parsed, out))   # the caller's file as given (no parse step), the caller's folder
        self.assertEqual(manifest.read(out)["inputs"]["parser"], None)
        one = os.path.join(self.tmp, "one_given.pdb"); open(one, "w").write("ATOM\n"); out1 = os.path.join(self.tmp, "out_pdb")
        rc, _, se = _run(["design", "--mode", "off", "--pdb_path", one, "--out_folder", out1, "--seed", "3", "--pdb_path_chains", "A"] + KIT)   # upstream's single-PDB form on the stock route: its own --pdb_path, --pdb_path_chains verbatim
        self.assertEqual(rc, 0, se)
        argv = json.load(open(os.path.join(out1, "stub_seen.json")))["argv"]
        self.assertEqual(argv[argv.index("--pdb_path") + 1], one); self.assertEqual(argv[argv.index("--pdb_path_chains") + 1], "A")
        self.assertEqual([a for a in argv if a.startswith("--")], ["--pdb_path", "--out_folder", "--path_to_model_weights", "--seed", "--pdb_path_chains", "--save_score", "--save_probs"])   # no parsed.jsonl, no chain file: upstream parses and assigns
        shutil.rmtree(out1); stack.reset_for_tests(); os.environ[stack.ENV_ALLOW_PARTIAL] = "1"   # no GPU here: the CUDA-graph levers recorded as partial, the pass runs
        with stub_stage_patch():
            rc, _, se = _run(["design", "--mode", "exact", "--pdb_path", one, "--out_folder", out1, "--seed", "3", "--pdb_path_chains", "A"] + KIT)   # the same form on the kit route: parsed by the kit, the chains assigned as upstream assigns them
        self.assertEqual(rc, 0, se)
        self.assertEqual(json.load(open(os.path.join(out1, "stub_seen.json")))["chains"], {"one_given": [["A"], ["B"]]})
        rc, _, se = _run(["design", "--mode", "off", "--pdb_path", parsed, "--out_folder", out1])
        self.assertEqual(rc, report.EXIT_USAGE); self.assertIn("not a .pdb file (a parsed jsonl is --jsonl_path <file>", se)

    def test_bb_batch_is_the_worker_lines_backbone_batch(self):
        """--bb_batch K replaces the mode line's value in the worker's argv and on the lines; the stock route has none (usage); K is a whole number."""
        out = os.path.join(self.tmp, "out_bb")
        real_stage = kit_run._stage.stage_base

        def stub_stage(stage_dir):
            paths = real_stage(stage_dir)
            open(paths["worker"], "w").write(STUB_WORKER); open(paths["worker_lowmem"], "w").write(STUB_WORKER); open(paths["fast_parse"], "w").write(STUB_PARSER)
            return paths
        os.environ[stack.ENV_ALLOW_PARTIAL] = "1"                                  # CPU box: the CUDA-graph levers are unavailable, recorded
        with mock.patch.object(kit_run._stage, "stage_base", side_effect=stub_stage):
            rc, _, se = _run(["design", "--mode", "exact", "--bb_batch", "4", "--input", self.pdbs, "--out", out, "--seed", "11"] + KIT)
        self.assertEqual(rc, 0, se)
        self.assertRegex(se, r"\[proteinmpnn-opt\] ACTIVE mode=exact variant=vanilla .*levers=\S*,bb_batch=4 ")
        f = json.load(open(os.path.join(out, "stub_seen.json")))["flags"]
        self.assertEqual((f["--mode"], f["--bb_batch"], f.get("--x_all")), ("stream", "4", True))
        man = manifest.read(out)
        self.assertEqual(man["activation_report"]["bb_batch"], 4); self.assertIn("--bb_batch 4", " ".join(man["activation_report"]["flags"]))
        self.assertNotIn("--bb_batch", man["stock_args"])                            # a kit option: never among the stock options, never on the stock route's argv
        rc, _, se = _run(["design", "--mode", "off", "--bb_batch", "4", "--input", self.pdbs, "--out", out + "_off", "--seed", "11"])
        self.assertEqual(rc, report.EXIT_USAGE); self.assertIn("--bb_batch 4: mode off runs the stock command line, one backbone per forward pass", se)
        self.assertFalse(os.path.exists(out + "_off"))
        for bad in ("0", "-2", "x", "1.5"):                                        # argparse's own refusal: usage, exit 2
            err = io.StringIO()
            with self.assertRaises(SystemExit) as cm, contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                cli.main(["check", "--mode", "exact", "--bb_batch", bad])
            self.assertEqual(cm.exception.code, report.EXIT_USAGE, bad); self.assertIn("--bb_batch", err.getvalue())
        rc, _, se = _run(["check", "--mode", "exact", "--variant", "soluble", "--bb_batch", "1"])
        self.assertIn("[proteinmpnn-opt] DRY-RUN mode=exact variant=soluble route=worker proteinmpnn=8907e667", se); self.assertIn(",bb_batch=1 probe=", se)

    def test_off_serves_every_stock_option_verbatim(self):
        out = os.path.join(self.tmp, "out_o")
        rc, _, se = _run(["design", "--mode", "off", "--input", self.pdbs, "--out", out, "--out_folder", "x"])     # one output folder, two spellings: usage
        self.assertEqual(rc, report.EXIT_USAGE); self.assertIn("an output folder: --out_folder <dir> (upstream's own) or --out <dir> — one of them, given once (got --out_folder and --out)", se)
        rc, _, se = _run(["design", "--mode", "off", "--out", out, "--seed", "1"])                                    # no input at all: usage, the spellings named
        self.assertEqual(rc, report.EXIT_USAGE); self.assertIn("design takes an input: --jsonl_path <parsed.jsonl> | --pdb_path <file.pdb>", se)
        rc, _, se = _run(["design", "--mode", "off", "--jsonl_path", self.pdbs, "--out_folder", out])                 # upstream's --jsonl_path names a parsed jsonl, not a directory
        self.assertEqual(rc, report.EXIT_USAGE); self.assertIn("not a parsed jsonl (a directory of PDB files is --input <dir>", se)
        rc, _, se = _run(["design", "--mode", "off", "--input", self.pdbs, "--out", out, "--x_all"])
        self.assertEqual(rc, report.EXIT_USAGE); self.assertIn("kit lever", se)
        self.assertFalse(os.path.exists(out))
        given = ["--suppress_print", "1", "--seed", "0", "--ca_only", "--sampling_temp", "0.1 0.2", "--num_seq_per_target", "3", "--save_score", "0"]   # values the kit worker does not serve: the stock command line serves them on either mode
        rc, _, se = _run(["design", "--mode", "off", "--input", self.pdbs, "--out", out] + given)
        self.assertEqual(rc, 0, se)
        self.assertIn("NOT ACTIVE: mode off (stock route) (mode=off variant=vanilla) stock_args=--suppress_print 1 --seed 0 --ca_only --sampling_temp 0.1 0.2 --num_seq_per_target 3 --save_score 0\n", se)
        argv = json.load(open(os.path.join(out, "stub_seen.json")))["argv"]
        self.assertEqual(argv[argv.index("--path_to_model_weights") + 2:], given)                      # verbatim, in the order given, after the inputs the package renders
        self.assertEqual(manifest.read(out)["stock_args"], given)
        out2 = os.path.join(self.tmp, "out_o2")
        rc, _, se = _run(["design", "--mode", "off", "--variant", "vanilla", "--input", self.pdbs, "--out", out2])   # no option at all: nothing but the inputs reaches the script
        self.assertEqual(rc, 0, se)
        argv = json.load(open(os.path.join(out2, "stub_seen.json")))["argv"]
        self.assertEqual([a for a in argv if a.startswith("--")], ["--jsonl_path", "--out_folder", "--path_to_model_weights"])   # no --chain_id_jsonl either: upstream's own default chain handling
        self.assertEqual(argv[argv.index("--path_to_model_weights") + 1], os.path.join(self.mpnn, "vanilla_model_weights")); self.assertEqual(manifest.read(out2)["stock_args"], [])
        self.assertNotIn(" stock_args=", se)
        out3 = os.path.join(self.tmp, "out_o3")                                                           # upstream's own weights selectors pass through: the kit renders none of its own; --det 1 is inert (nothing reaches the child)
        rc, _, se = _run(["design", "--mode", "off", "--variant", "vanilla", "--det", "1", "--input", self.pdbs, "--out", out3, "--use_soluble_model", "--seed", "5"])
        self.assertEqual(rc, 0, se)
        argv = json.load(open(os.path.join(out3, "stub_seen.json")))["argv"]
        self.assertEqual([a for a in argv if a.startswith("--")], ["--jsonl_path", "--out_folder", "--use_soluble_model", "--seed"]); self.assertNotIn("--path_to_model_weights", argv)
        self.assertNotIn("--det", argv); self.assertEqual(manifest.read(out3)["stock_args"], ["--use_soluble_model", "--seed", "5"])   # --det is the kit's inert flag, not a stock option: absent from the child's argv and from stock_args


class TestWeightsRule(Base):
    """A user's own weights file always runs: the file the pass loads is digested and named on the line after the activation line — pinned for a
    stock/PINS.json digest, NOT PINNED for any other bytes, the exit code unchanged — and recorded in opt_manifest.json; only an absent file is
    refused, by name (3). ``--model_name`` selects the file (base variants)."""

    def setUp(self):
        super().setUp()
        self.p_pins.stop()
        self.p_pins = mock.patch.object(stack, "check_pins", side_effect=weights_rule_pins); self.p_pins.start()
        self.wdir = os.path.join(self.mpnn, "vanilla_model_weights"); os.makedirs(self.wdir, exist_ok=True)
        self.ckpt = os.path.join(self.wdir, "pmft_v1.pt")
        open(self.ckpt, "wb").write(b"a fine-tuned checkpoint: not the repository bytes")
        open(os.path.join(self.wdir, "v_48_020.pt"), "wb").write(b"stub repository weights")
        self.sha = stack.sha256(self.ckpt)
        self.not_pinned = f"[proteinmpnn-opt] weights sha256={self.sha[:12]} NOT PINNED — proceeding (stock/PINS.json names the tested weights)\n"

    def test_exact_runs_a_second_weight_file_not_pinned(self):
        out = os.path.join(self.tmp, "out_ft")
        with mock.patch.object(stack, "gpu_info", return_value=H100), mock.patch.dict(os.environ, {"STUB_CUDA": "1"}), stub_stage_patch():
            rc, _, se = _run(["design", "--mode", "exact", "--variant", "vanilla", "--model_name", "pmft_v1", "--input", self.pdbs, "--out", out, "--seed", "11"] + KIT)
        self.assertEqual(rc, 0, se)
        self.assertIn("[proteinmpnn-opt] ACTIVE mode=exact variant=vanilla proteinmpnn=8907e667 ", se); self.assertIn(self.not_pinned, se)
        self.assertEqual(se.count("NOT PINNED"), 1); self.assertLess(se.index("] ACTIVE "), se.index("NOT PINNED")); self.assertNotIn("NOT ACTIVE", se)
        f = json.load(open(os.path.join(out, "stub_seen.json")))["flags"]
        self.assertEqual((f["--path_to_model_weights"], f["--model_name"]), (self.wdir, "pmft_v1"))            # the worker loads that file
        man = manifest.read(out)
        self.assertEqual(man["pins"]["detail"]["weights"], {"file": self.ckpt, "sha256": self.sha, "pinned": None, "verdict": "not_pinned", "line": self.not_pinned[len("[proteinmpnn-opt] "):-1]})
        self.assertEqual((man["exit_code"], man["active"], man["stock_args"][:2], man["pins"]["pinned"]), (0, True, ["--model_name", "pmft_v1"], True))
        self.assertEqual(man["activation_report"]["weights"], man["pins"]["detail"]["weights"])            # status()'s view of the same record

    def test_off_runs_it_too(self):
        out = os.path.join(self.tmp, "out_ft_off")
        rc, _, se = _run(["design", "--mode", "off", "--variant", "vanilla", "--model_name", "pmft_v1", "--input", self.pdbs, "--out", out])
        self.assertEqual(rc, 0, se)
        self.assertIn("[proteinmpnn-opt] NOT ACTIVE: mode off (stock route) (mode=off variant=vanilla) stock_args=--model_name pmft_v1\n" + self.not_pinned, se)   # the weights line follows the route's line
        argv = json.load(open(os.path.join(out, "stub_seen.json")))["argv"]
        self.assertEqual((argv[argv.index("--path_to_model_weights") + 1], argv[argv.index("--model_name") + 1]), (self.wdir, "pmft_v1"))
        man = manifest.read(out)
        self.assertEqual((man["pins"]["detail"]["weights"]["verdict"], man["pins"]["detail"]["weights"]["sha256"], man["exit_code"]), ("not_pinned", self.sha, 0))

    def test_a_fine_tune_outside_the_checkout_runs_on_both_routes(self):
        """--path_to_model_weights <dir> --model_name <name>: the file exists only there (the flag's primary use); the pin check digests THAT file, so no
        route refuses it, and the weights line / manifest name it."""
        mine = os.path.join(self.tmp, "finetunes"); os.makedirs(mine); ft = os.path.join(mine, "only_here.pt"); open(ft, "wb").write(b"fine-tuned bytes")
        for mode, extra in (("off", []), ("exact", ["--seed", "11"] + KIT)):
            out = os.path.join(self.tmp, f"out_ft2_{mode}"); stack.reset_for_tests()
            with mock.patch.object(stack, "gpu_info", return_value=H100), mock.patch.dict(os.environ, {"STUB_CUDA": "1"}), stub_stage_patch():
                rc, _, se = _run(["design", "--mode", mode, "--variant", "vanilla", "--input", self.pdbs, "--out", out, "--path_to_model_weights", mine, "--model_name", "only_here"] + extra)
            self.assertEqual(rc, 0, (mode, se)); self.assertNotIn("NOT ACTIVE: stock pin", se)
            self.assertIn(f"[proteinmpnn-opt] weights sha256={stack.sha256(ft)[:12]} NOT PINNED", se)
            self.assertEqual(manifest.read(out)["pins"]["detail"]["weights"]["file"], ft)

    def test_the_pinned_digest_is_named_as_pinned(self):
        vanilla = check_pins.load_pins()["weights"]["vanilla"]["sha256"]
        with mock.patch.object(stack, "gpu_info", return_value=H100), mock.patch.object(check_pins, "sha256", return_value=vanilla):     # the repository bytes, by digest
            rc, _, se = _run(["check", "--mode", "exact", "--variant", "vanilla"])
        self.assertEqual(rc, 0, se)
        self.assertIn("[proteinmpnn-opt] weights=vanilla_model_weights/v_48_020.pt sha256=c9cb4a671d79 (pinned)\n", se); self.assertNotIn("NOT PINNED", se)

    def test_an_absent_weights_file_is_refused_by_name(self):
        out = os.path.join(self.tmp, "out_absent")
        absent = os.path.join(self.wdir, "no_such.pt")
        for mode in ("exact", "off"):
            stack.reset_for_tests()
            rc, _, se = _run(["design", "--mode", mode, "--variant", "vanilla", "--model_name", "no_such", "--input", self.pdbs, "--out", out, "--seed", "11"] + KIT)
            self.assertEqual(rc, report.EXIT_NOT_ACTIVE, (mode, se))
            self.assertIn(f"stock pin: {absent}: the weights file is absent", se); self.assertNotIn("NOT PINNED", se)
            self.assertIn("nothing launched", se); self.assertFalse(os.path.exists(os.path.join(out, "stub_seen.json")))

    def test_upstreams_weights_selectors_pass_through(self):
        """--path_to_model_weights / --use_soluble_model are upstream's: the worker is handed the directory upstream's own rule gives (protein_mpnn_run.py:35-47),
        the kit's variant default stands aside; run.py has no --model_name."""
        out = os.path.join(self.tmp, "out_mn")
        other = os.path.join(self.tmp, "my_weights"); os.makedirs(other); open(os.path.join(other, "v_48_020.pt"), "wb").write(b"my own weights")
        with mock.patch.object(stack, "gpu_info", return_value=H100), mock.patch.dict(os.environ, {"STUB_CUDA": "1"}), stub_stage_patch():
            rc, _, se = _run(["design", "--mode", "exact", "--variant", "vanilla", "--input", self.pdbs, "--out", out, "--path_to_model_weights", other, "--seed", "11"] + KIT)
        self.assertEqual(rc, 0, se)
        f = json.load(open(os.path.join(out, "stub_seen.json")))["flags"]
        self.assertEqual(f["--path_to_model_weights"], other); self.assertEqual(list(f).count("--path_to_model_weights"), 1)   # the pass's directory, rendered once
        self.assertIn(" stock_args=--path_to_model_weights ", se)
        w = manifest.read(out)["pins"]["detail"]["weights"]
        self.assertEqual((w["file"], w["sha256"]), (os.path.join(other, "v_48_020.pt"), stack.sha256(os.path.join(other, "v_48_020.pt"))))   # the file digested and named is the one the pass loads
        shutil.rmtree(out); stack.reset_for_tests(); os.makedirs(os.path.join(self.mpnn, "soluble_model_weights"), exist_ok=True); open(os.path.join(self.mpnn, "soluble_model_weights", "v_48_020.pt"), "wb").write(b"s")
        with mock.patch.object(stack, "gpu_info", return_value=H100), mock.patch.dict(os.environ, {"STUB_CUDA": "1"}), stub_stage_patch():
            rc, _, se = _run(["design", "--mode", "exact", "--variant", "vanilla", "--input", self.pdbs, "--out", out, "--use_soluble_model", "--seed", "11"] + KIT)
        self.assertEqual(rc, 0, se)
        f = json.load(open(os.path.join(out, "stub_seen.json")))["flags"]
        self.assertEqual(f["--path_to_model_weights"], os.path.join(self.mpnn, "soluble_model_weights")); self.assertNotIn("--use_soluble_model", f)   # upstream's rule for the flag; the worker has no such flag of its own
        self.assertEqual(manifest.read(out)["pins"]["detail"]["weights"]["file"], os.path.join(self.mpnn, "soluble_model_weights", "v_48_020.pt"))   # variant vanilla, soluble weights: the record names what loads


class TestRunSh(Base):
    """run.sh is a thin wrapper: names are validated by the package, the env-vs-CLI rule names the real source, the exit codes are the package's."""

    def setUp(self):
        super().setUp()
        self.tree = stack.tree_home()
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
        env["PATH"] = os.pathsep.join([os.path.dirname(sys.executable), env.get("PATH", "")])
        for k in stack.PACKAGE_ENV + (stack.ENV_TARGET_GPU, stack.ENV_TARGET_GPU_MEM):
            env.pop(k, None)
        self.env = env

    def _sh(self, *args, **envx):
        env = dict(self.env); env.update(envx)
        return subprocess.run(["bash", os.path.join(self.tree, "run.sh"), *args], capture_output=True, text=True, env=env, timeout=120)

    def test_names_are_the_packages(self):
        r = self._sh("check", "--mode", "turbo"); self.assertEqual(r.returncode, report.EXIT_USAGE); self.assertIn("unknown mode 'turbo'", r.stderr)
        r = self._sh("check", "--variant", "membrane"); self.assertEqual(r.returncode, report.EXIT_USAGE); self.assertIn("unknown variant 'membrane'", r.stderr)
        r = self._sh("check", "--mode", "off"); self.assertEqual(r.returncode, report.EXIT_USAGE); self.assertIn("stock route is a design pass only", r.stderr)
        r = self._sh("frobnicate"); self.assertEqual(r.returncode, report.EXIT_USAGE)
        r = self._sh("check", "--mode", "fast", "--variant", "vanilla"); self.assertEqual(r.returncode, report.EXIT_NOT_ACTIVE); self.assertIn("proteinmpnn ships no fast tier: select --mode exact", r.stderr)

    def test_disagreement_names_the_source(self):
        r = self._sh("check", "--config", "h100", "--mode", "exact", PROTEINMPNN_OPT="off", MPNN_DIR=self.mpnn)
        self.assertEqual(r.returncode, report.EXIT_USAGE)
        self.assertIn("PROTEINMPNN_OPT=off from the environment", r.stderr)             # h100.env sets no mode: the environment is the source
        r = self._sh("check", "--config", "h100", "--variant", "vanilla", PROTEINMPNN_VARIANT="soluble", MPNN_DIR=self.mpnn)
        self.assertEqual(r.returncode, report.EXIT_USAGE); self.assertIn("PROTEINMPNN_VARIANT=soluble from the environment", r.stderr)

    def test_check_reaches_the_package(self):
        r = self._sh("check", "--mode", "exact", "--variant", "soluble", MPNN_DIR=self.mpnn)      # a stub checkout: the pin gate refuses, the DRY-RUN line prints
        self.assertEqual(r.returncode, report.EXIT_NOT_ACTIVE, r.stderr)
        self.assertIn("[proteinmpnn-opt] DRY-RUN mode=exact variant=soluble route=worker", r.stderr); self.assertIn("stock pin", r.stderr)


class TestReportLines(unittest.TestCase):
    def test_cmd_line_round_trips_through_its_regex(self):
        """report.cmd_line / report.CMD_LINE_RE are a producer/consumer pair: a reader parses this line back out of a run's
        stdout (behind a timestamp), argv round-tripping through shlex."""
        argv = ["/usr/bin/python", "/out/a b/_stage/stock/protein_mpnn_run.py", "--num_seq_per_target", "8"]
        line = report.cmd_line("stock", "soluble", argv)
        m = re.search(report.CMD_LINE_RE, "12.5 " + line)
        self.assertEqual((m.group("route"), m.group("variant"), shlex.split(m.group("argv"))), ("stock", "soluble", argv))
        self.assertEqual(line, "[proteinmpnn-opt] CMD route=stock variant=soluble argv=/usr/bin/python '/out/a b/_stage/stock/protein_mpnn_run.py' --num_seq_per_target 8")
        self.assertIsNone(re.search(report.CMD_LINE_RE, "[proteinmpnn-opt] EXIT pid=1 mode=off rc=0"))

    def test_probe_word_reads_the_workers_verdict_sentence(self):
        self.assertEqual([kit_run.probe_word({"verdict": v}) for v in ("PROBE PASS -> using hybrid (batched message GEMMs) for all 2 cell(s)", "PROBE FAIL -> stock-shape GEMMs", "not requested -> stock-shape GEMMs (exact by construction)", "", None)] + [kit_run.probe_word(None)],
                         ["PASS", "FAIL", "not requested", "unobserved", "unobserved", "unobserved"])
