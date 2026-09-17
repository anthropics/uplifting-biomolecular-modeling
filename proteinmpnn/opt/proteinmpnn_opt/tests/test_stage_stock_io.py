"""Staging (the overlay on copies), the input rules, the output listing, stock/PINS.json against the carried sources and
stock/check_pins.py's refusals. numpy for the npz fixtures; no torch, no GPU."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from proteinmpnn_opt import inputs, modes, outputs, stack, stage

TREE = stack.tree_home()
sys.path.insert(0, os.path.join(TREE, "stock"))
import check_pins  # noqa: E402  (stock/check_pins.py, standard library only)


class TestStage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_stage_base_overlays_the_worker_into_the_parser_copy(self):
        p = stage.stage_base(os.path.join(self.tmp, "s"))
        for f in stage.OVERLAY_BASE:
            dst = os.path.join(p[modes.PARSER_DIR], "kit", f)
            self.assertTrue(os.path.isfile(dst))
            self.assertEqual(stack.sha256(dst), stack.sha256(os.path.join(stack.kit_home(), modes.WORKER_DIR, "addon", f)))
        self.assertTrue(os.path.isfile(p["worker"])); self.assertTrue(os.path.isfile(p["fast_parse"]))
        self.assertEqual(os.path.basename(p["worker_lowmem"]), stage.LOWMEM_WORKER); self.assertTrue(os.path.isfile(p["worker_lowmem"]))   # the derived exact executable
        # the tree itself carries no overlay
        self.assertFalse(os.path.exists(os.path.join(stack.parser_dir(), "kit", "mpnn_worker2.py")))


class TestInputs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_unassigned_is_upstreams_none(self):
        """The kit worker's --chain_id_jsonl when the caller gives none: ONE line, JSON null — read as protein_mpnn_run.py reads a --chain_id_jsonl
        (json.loads per line, :69-73) it is chain_id_dict None, tied_featurize's every-chain-designed default; never a chain rule of the package's."""
        p = inputs.write_unassigned(os.path.join(self.tmp, inputs.UNASSIGNED))
        self.assertEqual(p, os.path.join(self.tmp, inputs.UNASSIGNED))
        lines = open(p).read().splitlines()
        self.assertEqual(lines, ["null"]); self.assertIsNone(json.loads(lines[0]))
        parsed = os.path.join(self.tmp, "p.jsonl")
        with open(parsed, "w") as fh:
            fh.write(json.dumps({"name": "t1", "seq_chain_B": "AA", "seq_chain_A": "GG", "seq_chain_C": "WW"}) + "\n\n")
            fh.write(json.dumps({"name": "t2", "seq_chain_A": "M"}) + "\n")
        self.assertEqual(inputs.parse_count(parsed), 2)

    def test_a_single_pdb_is_an_input(self):
        pdb = os.path.join(self.tmp, "one.pdb"); open(pdb, "w").write("ATOM\n")
        self.assertEqual(inputs.classify_base(pdb), "pdb")                              # protein_mpnn_run.py --pdb_path: handed to the stock command line as that flag
        self.assertEqual(inputs.classify_base(self.tmp), "dir")

    def test_a_missing_input_is_refused_by_name(self):
        with self.assertRaises(inputs.InputError):
            inputs.classify_base(os.path.join(self.tmp, "nothing"))


class TestOutputs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _pass(self, name, fa, arr):
        d = os.path.join(self.tmp, name)
        for sub in ("seqs", "scores", "probs"):
            os.makedirs(os.path.join(d, sub))
        open(os.path.join(d, "seqs", "t.fa"), "w").write(fa)
        np.savez(os.path.join(d, "scores", "t.npz"), score=np.array([0.5], dtype=np.float32))
        np.savez(os.path.join(d, "probs", "t.npz"), probs=arr, chain_order=np.array(["A"]), mask=np.ones(3, dtype=np.float32))
        return d

    def test_listing(self):
        a = self._pass("a", ">t\nMKV\n", np.array([[0.1, 0.9]], dtype=np.float32))
        la = outputs.listing(a, "soluble")
        self.assertEqual(la, ["seqs/t.fa", "scores/t.npz", "probs/t.npz"])              # relative paths, per output directory in the writer's order; no digests
        self.assertEqual(outputs.counts(la), {"seqs": 1, "scores": 1, "probs": 1})


class TestStockPins(unittest.TestCase):
    def test_pins_match_the_carried_sources(self):
        pins = check_pins.load_pins()
        readme = open(os.path.join(stack.worker_dir(), modes.README), encoding="utf-8").read()          # the add-on README names the pinned commit it reproduces
        self.assertIn(f"dauparas/ProteinMPNN @ `{pins['upstream']['commit'][:8]}`", readme)
        self.assertIn(f"git -C $MPNN_DIR checkout {pins['upstream']['commit']}", pins["upstream"]["install"])   # the clone line is PINS.json's own
        sor = pins["pinned_stack"]                                         # versions of the pinned stack; `tested_on` names the image (documentation: nothing compares it)
        for k in ("base", "python", "torch", "cuda", "numpy", "lock", "stack_key", "tested_on"): self.assertIsInstance(sor[k], str, k)
        self.assertTrue(os.path.isfile(os.path.join(TREE, sor["lock"])), sor["lock"])              # the stack's one list: environment/requirements-mpnn.lock
        self.assertNotIn("image", sor)                                                                # one documentation line: tested_on
        self.assertNotIn("status", sor)                                                               # facts, no labels
        self.assertEqual(set(pins["variants"]), set(modes.VARIANTS))                                  # the variants are the two repository weight sets
        for v in modes.VARIANTS:
            self.assertIn(pins["variants"][v]["weights"], pins["weights"]); self.assertRegex(pins["weights"][v]["sha256"], r"^[0-9a-f]{64}$", v)
        self.assertEqual(pins["kit_directories"], [f"opt/forward/{modes.WORKER_DIR}", f"opt/forward/{modes.PARSER_DIR}"])
        self.assertEqual(check_pins.check_snapshot(), [])                                              # the carried stock/src entry points are present

    def test_check_pins_refuses_only_what_a_pass_cannot_run_without(self):
        tmp = tempfile.mkdtemp()
        try:
            pins = check_pins.load_pins()
            bad, detail = check_pins.check_base(pins, "soluble", tmp)                                  # an empty directory: no protein_mpnn_run.py (a finding), no .git (a note), no weights (a finding)
            self.assertTrue(any("does not hold protein_mpnn_run.py" in b for b in bad), bad); self.assertIsNone(detail["head"])
            self.assertFalse([b for b in bad if "git" in b and "protein_mpnn_run.py" not in b and "weights" not in b], bad)   # the checkout's commit is never a finding
            self.assertTrue(any("not a git checkout" in n for n in detail["notes"]), detail["notes"]); self.assertEqual(detail["pinned_commit"], pins["upstream"]["commit"])
            ck = os.path.join(tmp, "ck"); os.makedirs(os.path.join(ck, "soluble_model_weights"))       # a plain directory holding the script and the weights, not a git checkout, at no commit: it runs
            open(os.path.join(ck, "protein_mpnn_run.py"), "w").write("# upstream\n"); open(os.path.join(ck, "soluble_model_weights", "v_48_020.pt"), "wb").write(b"w")
            bad, detail = check_pins.check_base(pins, "soluble", ck)
            self.assertEqual(bad, []); self.assertEqual(len(detail["notes"]), 1); self.assertEqual(detail["weights"]["verdict"], "not_pinned")
            with mock.patch.object(check_pins, "git_head", return_value="0" * 40):                     # a checkout at ANOTHER commit: a finding naming both commits and the remedy — another commit changes what stock means
                bad, detail = check_pins.check_base(pins, "soluble", ck)
            self.assertEqual(len(bad), 1, bad); self.assertIn(f"HEAD {'0' * 40} is not the pinned commit {pins['upstream']['commit']}", bad[0]); self.assertIn("git -C", bad[0])
            self.assertEqual(detail["notes"], []); self.assertEqual(detail["head"], "0" * 40)
            with mock.patch.object(check_pins, "git_head", return_value=pins["upstream"]["commit"]):   # at the pin: nothing to note
                self.assertEqual(check_pins.check_base(pins, "soluble", ck), ([], mock.ANY))
                self.assertEqual(check_pins.check_base(pins, "soluble", ck)[1]["notes"], [])
            out = subprocess.run([sys.executable, "-I", os.path.join(TREE, "stock", "check_pins.py"), "--variant", "soluble", "--mpnn-dir", ck, "--json"], capture_output=True, text=True)
            self.assertEqual(out.returncode, 0, out.stderr); self.assertTrue(json.loads(out.stdout)["pinned"]); self.assertIn("check_pins: note: ", out.stderr)
            self.assertNotIn("tracked_files", json.dumps(pins))                                        # no in-repo digests of upstream's files: the pinned commit names them
            bad, _ = check_pins.check_base(pins, "soluble", os.path.join(tmp, "nope"))
            self.assertTrue(any("not a directory" in b for b in bad))
            out = subprocess.run([sys.executable, "-I", os.path.join(TREE, "stock", "check_pins.py"), "--variant", "soluble", "--mpnn-dir", tmp, "--json"],
                                 capture_output=True, text=True)
            self.assertEqual(out.returncode, 3)
            self.assertFalse(json.loads(out.stdout)["pinned"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_the_weights_file_of_a_pass_is_named_never_refused(self):
        """check_weights: an absent file is the one weights finding; present bytes are digested and named — a PINS.json digest as pinned under its
        name, anything else NOT PINNED — and are never a finding. weights_file: the variant's weights directory + upstream's --model_name (the
        launchers' ``<variant>_model_weights`` rule is this one)."""
        pins = check_pins.load_pins()
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        wdir = os.path.join(tmp, "vanilla_model_weights"); os.makedirs(wdir)
        self.assertEqual(check_pins.weights_file(pins, "vanilla", mpnn_dir=tmp), os.path.join(wdir, "v_48_020.pt"))
        self.assertEqual(check_pins.weights_file(pins, "soluble", mpnn_dir="/m", model_name="pmft_v1"), "/m/soluble_model_weights/pmft_v1.pt")
        for v in check_pins.BASE_VARIANTS:
            self.assertEqual(os.path.dirname(check_pins.weights_file(pins, v, mpnn_dir="/m")), f"/m/{v}_model_weights")
        ft = os.path.join(wdir, "pmft_v1.pt"); open(ft, "wb").write(b"fine-tuned bytes, not the repository's")
        sha = stack.sha256(ft)
        not_pinned = f"weights sha256={sha[:12]} NOT PINNED — proceeding (stock/PINS.json names the tested weights)"
        bad, rec = check_pins.check_weights(pins, ft)
        self.assertEqual((bad, rec), ([], {"file": ft, "sha256": sha, "pinned": None, "verdict": "not_pinned", "line": not_pinned}))
        with mock.patch.object(check_pins, "sha256", return_value=pins["weights"]["vanilla"]["sha256"]):       # the repository bytes, by digest
            bad, rec = check_pins.check_weights(pins, ft)
        self.assertEqual((bad, rec["pinned"], rec["verdict"], rec["line"]),
                         ([], "vanilla_model_weights/v_48_020.pt", "pinned", "weights=vanilla_model_weights/v_48_020.pt sha256=c9cb4a671d79 (pinned)"))
        absent = os.path.join(wdir, "absent.pt")
        bad, rec = check_pins.check_weights(pins, absent)
        self.assertEqual((bad, rec["sha256"], rec["line"]), ([f"{absent}: the weights file is absent"], None, None))
        # through the base pin: a second file's digest is data (the findings are the checkout's only), an absent name is a finding
        bad, detail = check_pins.check_base(pins, "vanilla", tmp, model_name="pmft_v1")
        self.assertFalse(any("pmft_v1" in b or "weights" in b for b in bad), bad); self.assertEqual(detail["weights"]["line"], not_pinned)
        bad, detail = check_pins.check_base(pins, "vanilla", tmp, model_name="absent")
        self.assertIn(f"{absent}: the weights file is absent", bad); self.assertIsNone(detail["weights"]["line"])
        # the script: the line on stderr (never among the findings), the record under --json, the exit code the checkout's alone (3: tmp is no checkout)
        out = subprocess.run([sys.executable, "-I", os.path.join(TREE, "stock", "check_pins.py"), "--variant", "vanilla", "--mpnn-dir", tmp,
                              "--model-name", "pmft_v1", "--json"], capture_output=True, text=True)
        self.assertEqual(out.returncode, 3, out.stderr)
        self.assertIn(f"check_pins: {not_pinned}\n", out.stderr)
        self.assertFalse([l for l in out.stderr.splitlines() if "pmft_v1" in l and "NOT PINNED" not in l], out.stderr)
        d = json.loads(out.stdout)
        self.assertEqual((d["detail"]["weights"]["file"], d["detail"]["weights"]["sha256"], d["detail"]["weights"]["verdict"]), (ft, sha, "not_pinned"))
        quiet = subprocess.run([sys.executable, "-I", os.path.join(TREE, "stock", "check_pins.py"), "--variant", "vanilla", "--mpnn-dir", tmp,
                                "--model-name", "pmft_v1", "--quiet"], capture_output=True, text=True)
        self.assertEqual((quiet.returncode, quiet.stdout, quiet.stderr), (3, "", ""))
