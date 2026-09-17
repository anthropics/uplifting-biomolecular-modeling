"""stock/check_pins.py against a stub site: the commit from direct_url.json, the lever-touched files in the installed (non-git) package
hashed against PINS.json `source_sha256` when the RECORD locates them (present-and-patched refused by digest, absent refused by name),
the asserted stack, the weights (warn-and-run: pinned / not pinned by digest, a missing file refused by name), the card."""
import contextlib
import hashlib
import io
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

from . import _stubs

CHECK = os.path.join(_stubs.TREE, "stock", "check_pins.py")


def load_check_pins():
    spec = importlib.util.spec_from_file_location("check_pins", CHECK)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


@unittest.skipUnless(_stubs.tree_present(), "release tree not present around the package")
class TestCheckPins(unittest.TestCase):
    def setUp(self):
        self.state = _stubs.StubState(); self.state.restore()
        self.tmp = tempfile.mkdtemp(prefix="cd_opt_pins_")
        self.pins = _stubs.pins()
        self.cp = load_check_pins()

    def tearDown(self):
        self.state.restore()

    def site(self, **kw):
        site = _stubs.make_stub_stack(self.tmp, **kw)
        sys.path.insert(0, site)
        return site

    def test_commit_and_files(self):
        site = self.site(with_record=True)
        for rel in self.cp.LEVER_FILES:                                                # the pinned files in place: the check passes
            shutil.copy(os.path.join(_stubs.TREE, "stock", "src", rel), os.path.join(site, rel))
        bad, detail = self.cp.check_commit(self.pins)
        self.assertEqual(bad, []); self.assertEqual(detail["files"], {rel: "ok" for rel in self.cp.LEVER_FILES})
        os.remove(os.path.join(site, self.cp.LEVER_FILES[0]))                          # a lever-touched file absent from the install
        bad, detail = self.cp.check_commit(self.pins)
        self.assertEqual(len(bad), 1); self.assertIn("absent from the installed package", bad[0])
        self.assertEqual(detail["files"][self.cp.LEVER_FILES[0]], "missing")

    def test_commit_files_patched_is_refused_by_digest(self):
        """A lever-touched file PRESENT but with different bytes than PINS.json `source_sha256` (a patched, non-stock install of this
        genuinely external package) is refused by digest, not silently accepted as merely present."""
        site = self.site(with_record=True)
        for rel in self.cp.LEVER_FILES:
            shutil.copy(os.path.join(_stubs.TREE, "stock", "src", rel), os.path.join(site, rel))
        bad, detail = self.cp.check_commit(self.pins)
        self.assertEqual(bad, []); self.assertEqual(detail["files"][self.cp.LEVER_FILES[0]], "ok")
        patched = os.path.join(site, self.cp.LEVER_FILES[0])
        open(patched, "ab").write(b"\n# patched\n")                                     # present, but no longer the pinned bytes
        bad, detail = self.cp.check_commit(self.pins)
        self.assertEqual(len(bad), 1); self.assertIn("differs from the pinned commit", bad[0])
        self.assertEqual(detail["files"][self.cp.LEVER_FILES[0]], "differs")

    def test_commit_mismatch_and_no_vcs(self):
        self.site(commit="1" * 40)
        bad, _ = self.cp.check_commit(self.pins)
        self.assertEqual(len(bad), 1); self.assertIn("installed from commit 111111111111", bad[0])
        self.state.restore(); self.tmp = tempfile.mkdtemp()
        site = self.site()
        os.remove(os.path.join(site, "colabdesign-1.1.3.dist-info", "direct_url.json"))
        bad, _ = self.cp.check_commit(self.pins)
        self.assertEqual(bad, [])                                                        # version equal, no VCS record: accepted by version
        self.state.restore(); self.tmp = tempfile.mkdtemp()
        site = self.site(versions={"colabdesign": "1.1.1"})
        os.remove(os.path.join(site, "colabdesign-1.1.1.dist-info", "direct_url.json"))
        bad, _ = self.cp.check_commit(self.pins)
        self.assertIn("without a VCS record", bad[0])

    def test_stack(self):
        self.site()
        bad, detail = self.cp.check_stack(self.pins)
        self.assertEqual([b for b in bad if not b.startswith("stack: python")], [])
        self.assertTrue(all(detail[n]["ok"] for n in ("jax", "jaxlib", "dm-haiku", "numpy")))
        self.state.restore(); self.tmp = tempfile.mkdtemp()
        self.site(versions={"jax": "0.4.35"})
        bad, _ = self.cp.check_stack(self.pins)
        self.assertTrue(any("jax 0.4.35 != pinned 0.6.0" in b for b in bad))

    def weights_dir(self, data=b"weights"):
        d = os.path.join(self.tmp, "params", "params"); os.makedirs(d, exist_ok=True)
        for name in self.pins["weights"]["files"]:
            open(os.path.join(d, name), "wb").write(data)
        return os.path.join(self.tmp, "params"), d

    def pins_matching(self, data=b"weights"):
        """A copy of the pins whose `weights` entries match the 7-byte stand-in files (the real entries are the 373 MB AlphaFold files)."""
        pins = json.loads(json.dumps(self.pins))
        for name in pins["weights"]["files"]:
            pins["weights"]["files"][name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        return pins

    def test_weights_pinned_line(self):
        """Digest == PINS.json `weights`: accepted, worded pinned — `weights=<name> sha256=<12> (pinned)`."""
        root, _ = self.weights_dir()
        bad, detail = self.cp.check_weights(self.pins_matching(), root)
        self.assertEqual(bad, []); self.assertTrue(all(v["pinned"] and v["status"] == "pinned" for v in detail.values()))
        cert, unc = self.cp.weights_lines(detail)
        self.assertEqual(unc, []); self.assertEqual(cert[0], f"weights=params_model_1_multimer_v3.npz sha256={hashlib.sha256(b'weights').hexdigest()[:12]} (pinned)")
        self.assertEqual(len(cert), 5)

    def test_weights_not_pinned_warn_and_run(self):
        """An unknown digest is ACCEPTED: recorded `not_pinned`, ONE stderr line per file, exit 0 — never a finding (warn-and-run)."""
        root, d = self.weights_dir()
        open(os.path.join(d, "params_model_3_multimer_v3.npz"), "wb").write(b"other weights")
        bad, detail = self.cp.check_weights(self.pins_matching(), root)
        self.assertEqual(bad, [])
        self.assertEqual(detail["params_model_3_multimer_v3.npz"]["status"], "not_pinned"); self.assertFalse(detail["params_model_3_multimer_v3.npz"]["pinned"])
        self.assertEqual([v["status"] for k, v in detail.items() if k != "params_model_3_multimer_v3.npz"], ["pinned"] * 4)
        cert, unc = self.cp.weights_lines(detail)
        self.assertEqual(unc, [f"weights=params_model_3_multimer_v3.npz sha256={hashlib.sha256(b'other weights').hexdigest()[:12]} NOT PINNED — this tree was tested with the pinned weights only"])
        bad, detail = self.cp.check_weights(self.pins, root)                            # the real pins against 7-byte files: no file is the pinned one, none refused
        self.assertEqual(bad, []); self.assertEqual([v["status"] for v in detail.values()], ["not_pinned"] * 5)
        # through main(): exit 0, the NOT PINNED lines on stderr (quiet or not), the JSON census carries digest + word
        self.site(); self.cp.nvidia_smi = lambda: dict(_stubs.GPU)
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            self.cp.main(["--quiet", "--no-stack", "--params-dir", root])                  # no SystemExit: the run proceeds
        lines = [l for l in err.getvalue().splitlines() if "NOT PINNED" in l]
        self.assertEqual(len(lines), 5); self.assertTrue(all(l.startswith("check_pins: weights=params_model_") and l.endswith("NOT PINNED — this tree was tested with the pinned weights only") for l in lines), lines)
        outj = io.StringIO()
        with contextlib.redirect_stdout(outj), contextlib.redirect_stderr(io.StringIO()):
            self.cp.main(["--json", "--no-stack", "--params-dir", root])
        rec = json.loads(outj.getvalue())
        self.assertTrue(rec["ok"]); self.assertEqual(rec["findings"], [])
        w3 = rec["detail"]["weights"]["params_model_3_multimer_v3.npz"]
        self.assertEqual((w3["status"], w3["pinned"], w3["sha256"]), ("not_pinned", False, hashlib.sha256(b"other weights").hexdigest()))

    def test_weights_missing_refused(self):
        """A MISSING weights file stays a hard refusal, by name (exit 3)."""
        root, d = self.weights_dir()
        os.remove(os.path.join(d, "params_model_5_multimer_v3.npz"))
        bad, detail = self.cp.check_weights(self.pins_matching(), root)
        self.assertEqual(len(bad), 1); self.assertIn("params_model_5_multimer_v3.npz missing", bad[0]); self.assertEqual(detail["params_model_5_multimer_v3.npz"], "missing")
        self.site(); self.cp.nvidia_smi = lambda: dict(_stubs.GPU)
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as cm:
            self.cp.main(["--quiet", "--no-stack", "--params-dir", root])
        self.assertEqual(cm.exception.code, 3); self.assertIn("params_model_5_multimer_v3.npz missing", err.getvalue())

    def test_gpu(self):
        """Never a finding — the card is named, not refused: another card (name, memory), one below gpu.cc_min (the Pallas kernel's floor) and no GPU
        at all are one note each; the tested card says nothing."""
        self.cp.nvidia_smi = lambda: dict(_stubs.GPU)
        bad, detail = self.cp.check_gpu(self.pins)
        self.assertEqual((bad, detail["notes"]), ([], []))                                                  # the tested card: nothing to say
        self.cp.nvidia_smi = lambda: {"name": "NVIDIA A100-SXM4-80GB", "memory_mib": 81920, "cc": "8.0"}
        bad, detail = self.cp.check_gpu(self.pins)
        self.assertEqual(bad, [])                                                                            # an A100: not a finding
        self.assertEqual(detail["notes"], ["gpu: card 'NVIDIA A100-SXM4-80GB' != 'NVIDIA H100 80GB HBM3' — not the tested card; compute capability 8.0 >= 8.0, the Pallas kernel's floor: proceeding"])
        self.cp.nvidia_smi = lambda: {"name": "NVIDIA H100 80GB HBM3", "memory_mib": 70000, "cc": "9.0"}
        bad, detail = self.cp.check_gpu(self.pins)
        self.assertEqual(bad, []); self.assertIn("memory 70000 MiB outside 1% of 81559 MiB — not the tested card", detail["notes"][0])
        self.cp.nvidia_smi = lambda: {"name": "NVIDIA A100-SXM4-80GB", "memory_mib": 81920, "cc": None}   # an nvidia-smi without the compute_cap field
        bad, detail = self.cp.check_gpu(self.pins)
        self.assertEqual(bad, []); self.assertIn("compute capability not reported by nvidia-smi, the Pallas kernel's floor 8.0 unchecked: proceeding", detail["notes"][0])
        self.cp.nvidia_smi = lambda: {"name": "Tesla V100-SXM2-32GB", "memory_mib": 32768, "cc": "7.0"}
        bad, detail = self.cp.check_gpu(self.pins)
        self.assertEqual((bad, detail["notes"]), ([], ["gpu: card 'Tesla V100-SXM2-32GB' != 'NVIDIA H100 80GB HBM3'; memory 32768 MiB outside 1% of 81559 MiB; compute capability 7.0 < 8.0, the Pallas kernel's floor — not the tested card: proceeding"]))   # below the floor: named, never a finding
        self.cp.nvidia_smi = lambda: {}
        bad, detail = self.cp.check_gpu(self.pins)
        self.assertEqual(bad, []); self.assertIn("no GPU visible", detail["notes"][0])                      # no GPU: a note here (the modes decide: fast refuses by name, off runs)

    def test_main_gpu_lines(self):
        """The tested card prints its stdout line and no note; another card prints `not the pinned card` + ONE `check_pins: NOTE gpu: ...` stderr line and exits 0
        (quiet or not); a capability below the Pallas kernel's floor and no GPU at all are likewise ONE note and exit 0."""
        self.site()
        def run(argv):
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                self.cp.main(argv)
            return out.getvalue(), err.getvalue()
        self.cp.nvidia_smi = lambda: dict(_stubs.GPU)
        out, err = run(["--no-stack"])
        self.assertIn("gpu NVIDIA H100 80GB HBM3 81559 MiB: the pinned card\n", out); self.assertNotIn("NOTE gpu", err)   # (an install-fetched file not yet in place is its own NOTE line: test_install_verb FetchStepGate)
        self.cp.nvidia_smi = lambda: {"name": "NVIDIA A100-SXM4-80GB", "memory_mib": 81920, "cc": "8.0"}
        out, err = run(["--no-stack"])
        self.assertIn("gpu NVIDIA A100-SXM4-80GB 81920 MiB: not the pinned card\n", out)
        self.assertEqual([l for l in err.splitlines() if "gpu" in l], ["check_pins: NOTE gpu: card 'NVIDIA A100-SXM4-80GB' != 'NVIDIA H100 80GB HBM3' — not the tested card; compute capability 8.0 >= 8.0, the Pallas kernel's floor: proceeding"])
        out, err = run(["--quiet", "--no-stack"])
        self.assertEqual(out, ""); self.assertEqual(err.count("check_pins: NOTE gpu:"), 1)                # quiet: the note still prints, once
        self.cp.nvidia_smi = lambda: {"name": "Tesla V100-SXM2-32GB", "memory_mib": 32768, "cc": "7.0"}
        out, err = run(["--quiet", "--no-stack"])                                                            # below the kernel's floor: ONE note and exit 0 — named, never refused
        self.assertEqual(err.count("check_pins: NOTE gpu:"), 1); self.assertIn("compute capability 7.0 < 8.0, the Pallas kernel's floor — not the tested card: proceeding", err)
        self.cp.nvidia_smi = lambda: {}
        out, err = run(["--quiet", "--no-stack"])                                                            # no GPU: one note, exit 0
        self.assertEqual(err.count("check_pins: NOTE gpu: no GPU visible"), 1, err)

    def test_python_pin_rule(self):
        """The interpreter rule (python_pin, the one home run.sh install and the package's ACTIVE line share): the pinned release = `pinned`; another
        patch release of the pinned major.minor series = `not_pinned` — ONE `python=<v> NOT PINNED (pinned <v>) — …` line, never a finding; another
        series = `refused`, a finding of check_stack (exit 3)."""
        self.site()
        want = tuple(int(x) for x in self.pins["python"].split("."))
        cases = ((want + ("final", 0), "pinned"), ((want[0], want[1], want[2] + 3, "final", 0), "not_pinned"),
                 ((want[0], want[1] + 1, 2, "final", 0), "refused"), ((2, 7, 18, "final", 0), "refused"))
        for vi, status in cases:
            with mock.patch.object(sys, "version_info", vi):
                d = self.cp.python_pin(self.pins); bad, detail = self.cp.check_stack(self.pins)
            self.assertEqual((d["status"], d["ok"]), (status, status != "refused"), vi); self.assertEqual(detail["python"], d)
            self.assertEqual([b for b in bad if b.startswith("stack: python")] != [], status == "refused", (vi, bad))
            py = ".".join(map(str, vi[:3]))
            self.assertEqual(self.cp.python_line(d), f"python={py} NOT PINNED (pinned {self.pins['python']}) — this tree was tested with the pinned interpreter only" if status == "not_pinned" else None)
        self.assertEqual(self.cp.python_pin({})["status"], "unpinned"); self.assertIsNone(self.cp.python_line(None))

    def test_main_exit_codes(self):
        self.site()
        self.cp.nvidia_smi = lambda: dict(_stubs.GPU)
        want = tuple(int(x) for x in self.pins["python"].split("."))
        with mock.patch.object(sys, "version_info", (want[0], want[1] + 1, 0, "final", 0)), self.assertRaises(SystemExit) as cm:
            self.cp.main(["--quiet"])                                                    # a python of another release series than the pinned stack's: refused (exit 3)
        self.assertEqual(cm.exception.code, 3)
        err = io.StringIO()
        with mock.patch.object(sys, "version_info", (want[0], want[1], want[2] + 3, "final", 0)), contextlib.redirect_stderr(err):
            self.cp.main(["--quiet"])                                                    # another patch release of the pinned series: ONE warn-and-run line, no exit
        self.assertEqual(err.getvalue().count("NOT PINNED"), 1, err.getvalue())
        self.assertIn(f"check_pins: python={want[0]}.{want[1]}.{want[2] + 3} NOT PINNED (pinned {self.pins['python']}) — this tree was tested with the pinned interpreter only", err.getvalue())
        with mock.patch.object(sys, "version_info", want + ("final", 0)), contextlib.redirect_stderr(err):
            self.cp.main(["--quiet"])                                                    # the pinned interpreter on the pinned stack: silent, no exit
        self.cp.main(["--quiet", "--no-stack"])                                          # commit ok, gpu ok: no exit
        self.cp.main(["--quiet", "--no-stack", "--no-gpu"])


if __name__ == "__main__":
    unittest.main()
