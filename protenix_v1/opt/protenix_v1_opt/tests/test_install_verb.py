"""`run.sh install [--weights DIR] [pip options]` — the verb's argument handling and call sequence (a stub `python` on PATH records every
invocation; nothing is installed), and the weights step's flow (protenix_v1_opt.weights.populate with an injected fetch, gate and pin table;
no network, no upstream). CPU only."""
import io
import json
import os
import stat
import subprocess
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.normpath(os.path.join(HERE, "..", "..", ".."))          # the kit tree: run.sh, stock/, opt/
RUN_SH = os.path.join(TREE, "run.sh")

STUB = """#!/bin/bash
# stub interpreter: one line per invocation in $STUB_LOG; exit codes chosen per call kind by STUB_RC_PIP / STUB_RC_PINS / STUB_RC_WEIGHTS / STUB_RC_PROBE
printf '%s\\n' "$*" >> "$STUB_LOG"
case "$*" in
  *"-I -c import os,sys"*) exit "${STUB_RC_PROBE:-1}" ;;
  *"-m pip install"*) exit "${STUB_RC_PIP:-0}" ;;
  *"check_pins.py"*) exit "${STUB_RC_PINS:-0}" ;;
  *"-m protenix_v1_opt.weights"*) exit "${STUB_RC_WEIGHTS:-0}" ;;
  *) exit 0 ;;
esac
"""
PIP = f"-m pip install -e {TREE}/../common/opt_core -e {TREE}/opt"
PINS = f"-I {TREE}/stock/check_pins.py"


class InstallVerbArguments(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="protenix_v1_install_verb_")
        self.bin = os.path.join(self.tmp, "bin"); os.makedirs(self.bin)
        self.stub = os.path.join(self.bin, "python")
        with open(self.stub, "w") as f: f.write(STUB)
        os.chmod(self.stub, os.stat(self.stub).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        self.log = os.path.join(self.tmp, "calls.log")

    def run_sh(self, args, **rc):
        if os.path.exists(self.log): os.remove(self.log)
        env = {"PATH": self.bin + os.pathsep + "/usr/bin:/bin", "HOME": self.tmp, "STUB_LOG": self.log}
        env.update({f"STUB_RC_{k.upper()}": str(v) for k, v in rc.items()})
        r = subprocess.run(["bash", RUN_SH] + args, capture_output=True, text=True, env=env, cwd=self.tmp)
        lines = open(self.log).read().splitlines() if os.path.exists(self.log) else []
        self.probes = [c for c in lines if c.startswith("-I -c import os,sys")]           # the installed-from-this-tree probe
        return r.returncode, r.stdout + r.stderr, [c for c in lines if not c.startswith("-I -c import os,sys")]

    def test_install_runs_pip_then_the_pin_check(self):
        rc, out, calls = self.run_sh(["install"])
        self.assertEqual(rc, 0, out)
        self.assertEqual(calls, [PIP, PINS], calls)
        self.assertEqual(len(self.probes), 1, self.probes)

    def test_weights_dir_adds_the_weights_step_last(self):
        for args in (["install", "--weights", "/data/protenix_root"], ["install", "--weights=/data/protenix_root"]):
            rc, out, calls = self.run_sh(args)
            self.assertEqual(rc, 0, out)
            self.assertEqual(calls, [PIP, PINS, "-m protenix_v1_opt.weights /data/protenix_root"], calls)

    def test_other_arguments_go_to_pip_as_before(self):
        """The verb's standing behaviour: arguments other than --weights are pip install's; given any, pip runs (no installed-tree probe)."""
        rc, out, calls = self.run_sh(["install", "--no-build-isolation", "-q", "--weights", "/w"])
        self.assertEqual(rc, 0, out)
        self.assertEqual(calls, [PIP + " --no-build-isolation -q", PINS, "-m protenix_v1_opt.weights /w"], calls)
        self.assertEqual(self.probes, [], self.probes)

    def test_usage_errors_call_nothing(self):
        for args in (["install", "--weights"], ["install", "--weights", "--no-deps"], ["install", "--weights="], ["install", "--mode", "exact"],
                     ["install", "--config", "h100"]):
            rc, out, calls = self.run_sh(args)
            self.assertEqual(rc, 2, (args, out))
            self.assertEqual(calls, [], (args, calls))
            self.assertIn("run.sh:", out)

    def test_a_failed_step_stops_the_sequence_with_its_code(self):
        rc, out, calls = self.run_sh(["install", "--weights", "/w"], pip=1)
        self.assertEqual((rc, len(calls)), (1, 1), (out, calls))
        rc, out, calls = self.run_sh(["install", "--weights", "/w"], pins=3)
        self.assertEqual((rc, len(calls)), (3, 2), (out, calls))
        rc, out, calls = self.run_sh(["install", "--weights", "/w"], weights=1)
        self.assertEqual((rc, len(calls)), (1, 3), (out, calls))

    def test_a_tree_already_installed_skips_pip_by_name(self):
        """The container image ships the kit installed (editable, from /kit): `install` there names the skip and goes on to the pin check (and
        --weights) — a read-only image cannot re-run pip. The stub answers the find_spec probe with STUB_RC_PROBE."""
        rc, out, calls = self.run_sh(["install", "--weights", "/w"], probe=0)
        self.assertEqual(rc, 0, out); self.assertIn("installed from this tree already", out)
        self.assertEqual(calls, [PINS, "-m protenix_v1_opt.weights /w"], calls)   # pin check, weights — no pip
        self.assertEqual(len(self.probes), 1, self.probes)

    def test_probe_verb_unchanged(self):
        rc, out, calls = self.run_sh(["probe"])
        self.assertEqual(rc, 0, out)
        self.assertEqual(calls[0], "-c pass", calls)


class WeightsStep(unittest.TestCase):
    """protenix_v1_opt.weights.populate: fetch what is absent through upstream's routine, then the weights gate afresh — a checkpoint off its
    pin or a file still absent fails by name; nothing is deleted."""

    STOCK = {"checkpoint": "checkpoint/model_v1.pt", "checkpoint_sha256": "ab" * 32, "model_name": "model_v1",
             "data_files": {"ccd_components_file": "common/components.cif", "pdb_cluster_file": "common/clusters.txt",
                            "template (use_template=true only)": ["common/obsolete_to_successor.json", "common/release_date_cache.json"]}}

    def setUp(self):
        from protenix_v1_opt import weights
        self.W = weights
        self.dir = tempfile.mkdtemp(prefix="protenix_v1_weights_")
        self.fetched = []; self.gated = []
        self.pinned = True

    def fetch(self, root):                                            # upstream's rule: written only when absent
        self.fetched.append(root)
        for f in self.W.expected_files(self.STOCK):
            p = os.path.join(root, f)
            if not os.path.exists(p):
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "wb") as fh: fh.write(f.encode())

    def gate(self, root, argv=None, announce=True, refresh=False):    # kit.frozen_weights_check's record shape
        self.gated.append((root, tuple(argv), announce, refresh))
        ck = os.path.join(root, self.STOCK["checkpoint"])
        if not os.path.isfile(ck): raise RuntimeError(f"{ck} is absent")
        return {"root": root, "checkpoint": ck, "checkpoint_bytes": os.path.getsize(ck), "weights": self.STOCK["model_name"],
                "sha256": self.STOCK["checkpoint_sha256"] if self.pinned else "cd" * 32, "cached_utc": None, "pinned": self.pinned, "caches": {}}

    def populate(self, **kw):
        out = io.StringIO()
        rc = self.W.populate(self.dir, fetch=kw.pop("fetch", self.fetch), gate=kw.pop("gate", self.gate), stock=self.STOCK, out=out)
        return rc, out.getvalue()

    def test_expected_files_follow_pins(self):
        self.assertEqual(self.W.expected_files(self.STOCK), ["checkpoint/model_v1.pt", "common/components.cif", "common/clusters.txt",
                                                              "common/obsolete_to_successor.json", "common/release_date_cache.json"])
        st = json.load(open(os.path.join(TREE, "stock", "PINS.json")))["stock"]          # the kit's own pin table: the checkpoint + six data files
        files = self.W.expected_files(st)
        self.assertEqual((files[0], len(files)), (st["checkpoint"], 7), files)
        self.assertTrue(all(f.startswith("common/") for f in files[1:]), files)

    def test_absent_files_are_fetched_then_gated_afresh(self):
        rc, out = self.populate()
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.fetched, [self.dir])
        self.assertEqual(self.gated, [(self.dir, (), False, True)])
        self.assertIn("checkpoint/model_v1.pt: fetching", out); self.assertIn("WEIGHTS OK: 5/5", out); self.assertIn(f"PROTENIX_ROOT_DIR={self.dir}", out)
        for f in self.W.expected_files(self.STOCK): self.assertTrue(os.path.isfile(os.path.join(self.dir, f)))

    def test_a_complete_root_fetches_nothing(self):
        self.fetch(self.dir); self.fetched.clear()
        rc, out = self.populate()
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.fetched, [])
        self.assertIn("common/components.cif: present", out); self.assertNotIn("fetching", out)

    def test_a_checkpoint_off_the_pin_is_refused_by_name_and_left_in_place(self):
        self.fetch(self.dir); self.pinned = False
        rc, out = self.populate()
        self.assertEqual(rc, 1, out)
        last = out.splitlines()[-1]
        self.assertIn("REFUSED:", last); self.assertIn("checkpoint/model_v1.pt", last); self.assertIn("ab" * 32, last)
        self.assertTrue(os.path.isfile(os.path.join(self.dir, "checkpoint", "model_v1.pt")))

    def test_a_fetch_error_is_relayed_and_nothing_is_written(self):
        def broken(root): raise ConnectionError("network unreachable")
        rc, out = self.populate(fetch=broken)
        self.assertEqual(rc, 1, out)
        self.assertIn("FAILED: ConnectionError: network unreachable", out)
        self.assertEqual((os.listdir(self.dir), self.gated), ([], []))

    def test_a_file_absent_after_the_fetch_fails_by_name(self):
        def partial(root):                                             # leaves the template pair out
            for f in self.W.expected_files(self.STOCK)[:3]:
                p = os.path.join(root, f); os.makedirs(os.path.dirname(p), exist_ok=True); open(p, "wb").close()
        rc, out = self.populate(fetch=partial)
        self.assertEqual(rc, 1, out)
        self.assertIn("FAILED: absent after upstream's fetch: common/obsolete_to_successor.json, common/release_date_cache.json", out)

    def test_a_gate_refusal_is_relayed(self):
        def refusing(root, **kw): raise RuntimeError("common/components.cif (ccd_components_file) is absent")
        self.fetch(self.dir)
        rc, out = self.populate(gate=refusing)
        self.assertEqual(rc, 1, out); self.assertIn("REFUSED: common/components.cif", out)

    # -- the checkpoint is compared with its pin BEFORE anything can load it: transferred to <name>.part, hashed, then renamed --------------
    def prefetching(self, payload):
        """populate() with the kit's own prefetch_checkpoint bound to a recording retriever that writes ``payload``, and a recording upstream
        fetch that (like upstream) would load a checkpoint it had to download itself."""
        events = []

        def retrieve(url, path):
            events.append(("retrieve", url, os.path.basename(path)))
            with open(path, "wb") as fh: fh.write(payload)

        def upstream(root):                                            # upstream's rule: every absent file written; a checkpoint IT writes is loaded straight away
            ck = os.path.join(root, self.STOCK["checkpoint"])
            events.append(("upstream", "checkpoint present" if os.path.isfile(ck) else "checkpoint downloaded+loaded unverified"))
            self.fetch(root)

        def prefetch(root, stock, out=None):
            return self.W.prefetch_checkpoint(root, stock, retrieve=retrieve, url_for=lambda name: "https://upstream.example/" + name + ".pt", out=out)
        out = io.StringIO()
        rc = self.W.populate(self.dir, fetch=upstream, gate=self.gate, stock=self.STOCK, out=out, prefetch=prefetch)
        return rc, out.getvalue(), events

    def test_the_checkpoint_is_transferred_and_compared_before_upstream_runs(self):
        import hashlib
        payload = b"the pinned checkpoint bytes"
        self.STOCK = dict(self.STOCK, checkpoint_sha256=hashlib.sha256(payload).hexdigest())
        rc, out, events = self.prefetching(payload)
        self.assertEqual(rc, 0, out)
        self.assertEqual(events, [("retrieve", "https://upstream.example/model_v1.pt", "model_v1.pt.part"), ("upstream", "checkpoint present")], events)
        ck = os.path.join(self.dir, "checkpoint", "model_v1.pt")
        self.assertTrue(os.path.isfile(ck)); self.assertFalse(os.path.exists(ck + ".part"))
        self.assertEqual(open(ck, "rb").read(), payload)
        self.assertIn("model_v1.pt: transferring https://upstream.example/model_v1.pt to checkpoint/model_v1.pt.part", out)
        self.assertLess(out.index("= the pin; in place before upstream's routine runs"), out.index("WEIGHTS OK: 5/5"))

    def test_a_transfer_off_the_pin_is_refused_before_anything_loads_it(self):
        rc, out, events = self.prefetching(b"tampered bytes")
        self.assertEqual(rc, 1, out)
        self.assertEqual(events, [("retrieve", "https://upstream.example/model_v1.pt", "model_v1.pt.part")], events)   # upstream's routine never ran
        ck = os.path.join(self.dir, "checkpoint", "model_v1.pt")
        self.assertFalse(os.path.exists(ck)); self.assertTrue(os.path.isfile(ck + ".part"))                            # never under the name anything loads; kept for inspection
        last = out.splitlines()[-1]
        self.assertIn("REFUSED: checkpoint/model_v1.pt.part sha256 ", last); self.assertIn("is not the pin " + "ab" * 32, last); self.assertIn("never loaded", last)
        self.assertEqual(self.gated, [])

    def test_a_present_checkpoint_is_not_transferred_again(self):
        self.fetch(self.dir); os.remove(os.path.join(self.dir, "common", "clusters.txt"))
        rc, out, events = self.prefetching(b"unused")
        self.assertEqual(rc, 0, out)
        self.assertEqual(events, [("upstream", "checkpoint present")], events)

    def test_the_real_route_prefetches_with_upstream_names(self):
        import inspect
        src = inspect.getsource(self.W.prefetch_checkpoint)
        self.assertIn("from protenix.web_service.dependency_url import URL", src); self.assertIn('url_for(stock["model_name"])', src)
        self.assertIn("urllib.request.urlretrieve", src); self.assertIn("os.replace(part, final)", src)
        self.assertLess(src.index("sha256_file(part)"), src.index("os.replace(part, final)"))
        body = inspect.getsource(self.W.populate)
        self.assertLess(body.index("prefetch(root, stock, out=out)"), body.index("(fetch or upstream_fetch)(root)"))

    # -- the data files are hashed against "stock".data_files_sha256 after the fetch: a file off its pin is refused by name -----------------
    def pinned_stock(self, **override):
        import hashlib
        sums = {rel: hashlib.sha256(rel.encode()).hexdigest() for rel in self.W.expected_files(self.STOCK)[1:]}   # self.fetch writes each file's name as its bytes
        sums.update(override)
        return dict(self.STOCK, data_files_sha256=sums)

    def test_data_files_at_their_pins_pass_and_are_named(self):
        self.STOCK = self.pinned_stock()
        rc, out = self.populate()
        self.assertEqual(rc, 0, out)
        self.assertIn("common/components.cif: sha256 ", out); self.assertIn("the 4 data files are at their pins", out.splitlines()[-1])

    def test_a_data_file_off_its_pin_is_refused_by_name_and_left_in_place(self):
        self.STOCK = self.pinned_stock(**{"common/components.cif": "ef" * 32})
        rc, out = self.populate()
        self.assertEqual(rc, 1, out)
        last = out.splitlines()[-1]
        self.assertIn("REFUSED: 1 of 4 data files", last); self.assertIn("common/components.cif", last); self.assertNotIn("export PROTENIX_ROOT_DIR", out)
        self.assertIn("common/components.cif: sha256 ", out); self.assertIn("is not the pin " + "ef" * 32, out)
        self.assertEqual(open(os.path.join(self.dir, "common", "components.cif"), "rb").read(), b"common/components.cif")      # never deleted

    def test_stock_pins_carry_a_digest_for_every_data_file(self):
        import json, re
        stock = json.load(open(os.path.join(TREE, "stock", "PINS.json"), encoding="utf-8"))["stock"]
        rels = self.W.expected_files(stock)[1:]
        self.assertEqual(len(rels), 6); self.assertEqual(set(stock["data_files_sha256"]), set(rels))
        for rel in rels: self.assertRegex(stock["data_files_sha256"][rel], r"^[0-9a-f]{64}$", rel)
        self.assertEqual(self.W.check_data_files(self.dir, dict(stock, data_files_sha256={}), out=io.StringIO()), [])          # a table without digests judges nothing

    def test_usage(self):
        for argv in ([], ["-h"], ["a", "b"]):
            self.assertEqual(self.W.main(argv), 2)


if __name__ == "__main__":
    unittest.main()
